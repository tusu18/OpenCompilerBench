import os
import time
import uuid
import torch
import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from transformers import AutoModel, AutoTokenizer
from onnxruntime.quantization import (
    quantize_dynamic, quantize_static,
    CalibrationDataReader, QuantType, QuantFormat
)
from onnxconverter_common import float16
import onnxoptimizer
from pathlib import Path

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("reports", exist_ok=True)

torch.set_grad_enabled(False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {DEVICE}")


# ─── Workload Definitions ───────────────────────────────────────────────────
class Conv2DBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MatMulBlock(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


# Custom attention implementation that is ONNX-compatible
class OnnxCompatibleAttention(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "dim must be divisible by heads"

        # Linear projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)

        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        # Output projection
        output = self.out_proj(context)
        return output


class AttentionBlock(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.attn = OnnxCompatibleAttention(dim, heads)

    def forward(self, x):
        return self.attn(x)


class LSTMBlock(nn.Module):
    def __init__(self, inp=128, hid=256):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, batch_first=True)

    def forward(self, x):
        return self.lstm(x)[0]


class ResNet50Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2).eval()

    def forward(self, x):
        return self.net(x)


# Fixed BERT implementation for proper ONNX export
class BERTBlock(nn.Module):
    def __init__(self, model_id="bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.net = AutoModel.from_pretrained(model_id).eval().to(DEVICE)

    def forward(self, texts=None, input_ids=None, attention_mask=None, token_type_ids=None):
        # Handle both text input and tensor input for flexibility
        if texts is not None:
            # Process text input
            encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded["attention_mask"].to(DEVICE)
            token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids)).to(DEVICE)

        # Forward pass with explicit arguments
        return self.net(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state


# ONNX-exportable BERT wrapper
class BERTOnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        return self.model.net(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state


WORKLOADS = {
    "conv2d": (Conv2DBlock, {"shape": (3, 224, 224)}),
    "matmul": (MatMulBlock, {"dim": 512}),
    "attention": (AttentionBlock, {"dim": 128, "heads": 4, "seq_len": 32}),
    "lstm": (LSTMBlock, {"inp": 128, "hid": 256, "seq_len": 32}),
    "resnet50": (ResNet50Block, {"shape": (3, 224, 224)}),
    "bert": (BERTBlock, {"model_id": "bert-base-uncased"})
}


# ─── Helper Functions ───────────────────────────────────────────────────────
def make_input(wl_name, batch_size, cfg):
    """Create appropriate input tensors for each workload type."""
    if wl_name in ["conv2d", "resnet50"]:
        c, h, w = cfg["shape"]
        return torch.randn(batch_size, c, h, w, device=DEVICE)
    elif wl_name == "matmul":
        dim = cfg["dim"]
        return torch.randn(batch_size, dim, device=DEVICE)
    elif wl_name in ["attention", "lstm"]:
        seq_len = cfg["seq_len"]
        if wl_name == "attention":
            dim = cfg["dim"]
            return torch.randn(batch_size, seq_len, dim, device=DEVICE)
        else:
            inp_dim = cfg["inp"]
            return torch.randn(batch_size, seq_len, inp_dim, device=DEVICE)
    else:  # bert/gpt
        long_prompt = " ".join(["Artificial intelligence will"] * 100)
        return [long_prompt] * batch_size


def create_bert_inputs(model, texts, batch_size=None):
    """Create proper inputs for BERT model from texts or with dummy values."""
    if texts is not None:
        # Process real text input
        encoded = model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids)).to(DEVICE)
    else:
        # Create dummy inputs with specified batch size
        seq_len = 32
        input_ids = torch.ones((batch_size, seq_len), dtype=torch.long, device=DEVICE)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=DEVICE)
        token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=DEVICE)

    return input_ids, attention_mask, token_type_ids


def export_onnx(model, inp, path, wl_name, batch_size, opset=14):
    """Export model to ONNX format with proper configuration."""
    if os.path.exists(path):
        print(f"ONNX model already exists at {path}, skipping export.")
        return True
    try:
        if wl_name == "bert":
            # Special handling for BERT model
            input_ids, attention_mask, token_type_ids = create_bert_inputs(model, inp)

            # Create a wrapper for BERT
            wrapper = BERTOnnxWrapper(model)

            # Export with proper dynamic axes
            torch.onnx.export(
                wrapper,
                (input_ids, attention_mask, token_type_ids),
                path,
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["output"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "token_type_ids": {0: "batch", 1: "seq"},
                    "output": {0: "batch", 1: "seq"}
                },
                opset_version=opset,
                do_constant_folding=True
            )
        else:
            # For non-BERT models
            dynamic_axes = {"input": {0: "batch"}}
            if wl_name in ["attention", "lstm"]:
                dynamic_axes["input"][1] = "seq"

            torch.onnx.export(
                model, inp, path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=True
            )

        # Verify the model
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX export successful: {path}")
        return True
    except Exception as e:
        print(f"[ERROR] ONNX export failed for {wl_name}: {str(e)}")
        return False


def preprocess_onnx_model(model_path, output_path):
    """
    Pre-process ONNX model following ONNX Runtime recommendations:
    1. Symbolic shape inference
    2. Model optimization
    3. ONNX shape inference
    """
    try:
        # Manual pre-processing steps
        import onnx

        # 1. Load the model
        model = onnx.load(model_path)

        # 2. Apply shape inference
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"[WARN] Shape inference failed: {str(e)}")

        # 3. Use ONNX optimizer
        try:
            import onnxoptimizer
            passes = [
                "eliminate_identity",
                "eliminate_nop_dropout",
                "eliminate_nop_transpose",
                "eliminate_nop_pad",
                "fuse_bn_into_conv",
                "fuse_add_bias_into_conv",
                "fuse_consecutive_squeezes",
                "fuse_consecutive_transposes"
            ]
            optimized_model = onnxoptimizer.optimize(model, passes)
            onnx.save(optimized_model, output_path)
        except Exception as e:
            print(f"[WARN] Optimization failed: {str(e)}")
            # If optimization fails, save the model with shape inference only
            onnx.save(model, output_path)

        return True
    except Exception as e:
        print(f"[ERROR] Model pre-processing failed: {str(e)}")
        # Copy the original model as fallback
        import shutil
        shutil.copy(model_path, output_path)
        return False


def apply_fusion(onnx_path, fusion, wl_name):
    """Apply operator or graph-level fusion to ONNX models."""
    try:
        model = onnx.load(onnx_path)

        if fusion == "op":
            # Operator-level fusion
            passes = ["fuse_pad_into_conv", "fuse_bn_into_conv"]
        elif fusion == "graph":
            # Graph-level fusion - select appropriate passes based on workload
            if wl_name == "bert":
                # Conservative passes for BERT
                passes = ["eliminate_identity", "eliminate_deadend", "fuse_matmul_add_bias_into_gemm"]
            else:
                # More aggressive passes for other models
                passes = [
                    "eliminate_identity", "eliminate_nop_transpose", "eliminate_deadend",
                    "fuse_consecutive_transposes", "fuse_transpose_into_gemm",
                    "fuse_add_bias_into_conv", "fuse_bn_into_conv", "fuse_matmul_add_bias_into_gemm"
                ]
        else:
            return True

        optimized = onnxoptimizer.optimize(model, passes)
        onnx.save(optimized, onnx_path)
        return True
    except Exception as e:
        print(f"[ERROR] Fusion failed for {wl_name}: {str(e)}")
        return False


class CalibrationDataGenerator(CalibrationDataReader):
    """Generate calibration data for static quantization."""

    def __init__(self, wl_name, batch_size, cfg):
        self.wl_name = wl_name
        self.batch_size = batch_size
        self.cfg = cfg
        self.current_index = 0
        self.total_samples = 10  # Number of calibration samples

    def get_next(self):
        if self.current_index >= self.total_samples:
            return None

        self.current_index += 1

        if self.wl_name == "bert":
            # For BERT, create input tensors
            input_ids = np.ones((self.batch_size, 32), dtype=np.int64)
            attention_mask = np.ones((self.batch_size, 32), dtype=np.int64)
            token_type_ids = np.zeros((self.batch_size, 32), dtype=np.int64)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
        else:
            # For other models, create appropriate input shape
            if self.wl_name in ["conv2d", "resnet50"]:
                c, h, w = self.cfg["shape"]
                input_data = np.random.rand(self.batch_size, c, h, w).astype(np.float32)
            elif self.wl_name == "matmul":
                dim = self.cfg["dim"]
                input_data = np.random.rand(self.batch_size, dim).astype(np.float32)
            elif self.wl_name in ["attention", "lstm"]:
                seq_len = self.cfg["seq_len"]
                if self.wl_name == "attention":
                    dim = self.cfg["dim"]
                    input_data = np.random.rand(self.batch_size, seq_len, dim).astype(np.float32)
                else:
                    inp_dim = self.cfg["inp"]
                    input_data = np.random.rand(self.batch_size, seq_len, inp_dim).astype(np.float32)

            return {"input": input_data}


def flush_gpu_cache(force=True):
    """
    Flush GPU L2 cache by allocating and initializing a buffer larger than the cache.

    Args:
        force (bool): Attempt to flush even if cache size can't be determined
    """
    if DEVICE.type != "cuda":
        return

    # Get L2 cache size
    l2_cache_size = 0
    try:
        device_props = torch.cuda.get_device_properties(DEVICE)
        l2_cache_size = device_props.l2_cache_size
    except:
        # If we can't get the L2 cache size, use a default of 4MB
        l2_cache_size = 4 * 1024 * 1024

    # Allocate a buffer twice the size of L2 cache
    buffer_size = l2_cache_size * 2
    if buffer_size > 0:
        try:
            # Allocate and write to the entire buffer to ensure cache is flushed
            buffer = torch.zeros(buffer_size, dtype=torch.uint8, device=DEVICE)
            buffer.fill_(1)  # Force write to entire buffer
            torch.cuda.synchronize()
            del buffer
        except Exception as e:
            print(f"[WARN] Failed to flush L2 cache: {str(e)}")

    # Empty CUDA cache and ensure synchronization
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def profile_runner(runner, runs=10, warmup_runs=5, flush_cache=True,
                   flush_between_runs=False, batch_size=1):
    """
    Profile a runner function with proper warm-up and comprehensive metrics.

    Args:
        runner: Function to benchmark
        runs: Number of benchmark runs
        warmup_runs: Number of warm-up runs before measurement
        flush_cache: Whether to flush cache before measurement runs
        flush_between_runs: Whether to flush cache between individual runs
        batch_size: Batch size for throughput calculation
    """
    latencies = []

    if DEVICE.type == "cuda":
        # Reset GPU state
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print(f"Performing {warmup_runs} warm-up runs...")
        # Warm-up runs to populate caches and stabilize GPU clocks
        for i in range(warmup_runs):
            runner()
            torch.cuda.synchronize()

        # Flush cache before main measurement if requested
        if flush_cache:
            flush_gpu_cache()

        # Measure performance with proper synchronization
        torch.cuda.synchronize()
        for i in range(runs):
            # Optional per-run cache flush for more consistent results
            if flush_between_runs and i > 0:
                flush_gpu_cache()

            start = time.time()
            runner()
            torch.cuda.synchronize()
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms

        # Get peak memory usage
        mem = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        # For CPU, simpler benchmarking without GPU-specific operations
        for _ in range(warmup_runs):
            runner()

        for _ in range(runs):
            start = time.time()
            runner()
            end = time.time()
            latencies.append((end - start) * 1000)
        mem = None

    # Calculate comprehensive statistics
    latencies_np = np.array(latencies)
    avg_latency = np.mean(latencies_np)
    min_latency = np.min(latencies_np)
    max_latency = np.max(latencies_np)
    p50_latency = np.percentile(latencies_np, 50)
    p95_latency = np.percentile(latencies_np, 95)
    p99_latency = np.percentile(latencies_np, 99)
    std_latency = np.std(latencies_np)

    # Calculate throughput metrics (items/second)
    avg_throughput = (batch_size * 1000) / avg_latency if avg_latency > 0 else 0
    max_throughput = (batch_size * 1000) / min_latency if min_latency > 0 else 0

    return {
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "std_latency_ms": std_latency,
        "avg_throughput_items_per_sec": avg_throughput,
        "max_throughput_items_per_sec": max_throughput,
        "peak_memory_mb": mem
    }


def is_compatible(wl_name, backend, quant):
    """Check if the workload, backend and quantization combination is compatible."""
    # Skip int8_dyn for CUDA and TensorRT backends
    if quant == "int8_dyn" and ("cuda" in backend or "trt" in backend):
        return False

    # Skip BERT with TensorRT for simplicity
    if wl_name == "bert" and "trt" in backend:
        return False

    return True


def create_onnx_session(model_path, providers):
    """Create an ONNX Runtime session with proper error handling."""
    try:
        # Set session options for better performance
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1  # Use 1 thread per session for consistent benchmarking

        # Create session
        session = ort.InferenceSession(model_path, options, providers=providers)
        return session
    except Exception as e:
        print(f"[ERROR] Failed to create ONNX Runtime session: {str(e)}")
        return None


def run_inference_solo(backend, model, ts_model, onnx_path, wl_name, inp, batch_size, cfg):
    """
    Create a runner function for a single backend:
      – "torch":       PyTorch eager
      – "inductor":    torch.compile(..., backend="inductor")
      – "ts":          TorchScript
      – "onnx_cuda":   ONNX Runtime on CUDA
      – "onnx_trt":    ONNX Runtime with TensorRT + CUDA fallback
      – "tensorrt":    Pure TensorRT provider
    """
    # 1) PyTorch eager
    if backend == "torch":
        runner = lambda m=model, x=inp: m(x)

    # 2) torch.compile / Inductor
    elif backend == "inductor":
        compiled = torch.compile(model, backend="inductor").eval()
        runner = lambda m=compiled, x=inp: m(x)

    # 3) TorchScript
    elif backend == "ts":
        if ts_model is None:
            return None
        if wl_name == "bert":
            input_ids, attention_mask, token_type_ids = create_bert_inputs(model, inp)
            runner = lambda m=ts_model, ids=input_ids, mask=attention_mask, token=token_type_ids: m(ids, mask, token)
        else:
            runner = lambda m=ts_model, x=inp: m(x)

    # 4) ONNX Runtime variants
    elif backend.startswith("onnx"):
        # choose providers per backend
        if backend == "onnx_trt":
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif backend == "onnx_cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:  # fallback plain ONNX on CPU
            providers = ["CPUExecutionProvider"]

        session = create_onnx_session(onnx_path, providers)
        if session is None:
            return None

        if wl_name == "bert":
            input_ids, attention_mask, token_type_ids = create_bert_inputs(model, inp)
            runner = lambda s=session, ids=input_ids, mask=attention_mask, token=token_type_ids: s.run(
                None,
                {
                    "input_ids": ids.cpu().numpy(),
                    "attention_mask": mask.cpu().numpy(),
                    "token_type_ids": token.cpu().numpy()
                }
            )[0]
        else:
            runner = lambda s=session, x=inp: s.run(None, {"input": x.cpu().numpy()})[0]

    # 5) Pure TensorRT provider
    elif backend == "tensorrt":
        providers = ["TensorrtExecutionProvider"]
        session = create_onnx_session(onnx_path, providers)
        if session is None:
            return None

        if wl_name == "bert":
            input_ids, attention_mask, token_type_ids = create_bert_inputs(model, inp)
            runner = lambda s=session, ids=input_ids, mask=attention_mask, token=token_type_ids: s.run(
                None,
                {
                    "input_ids": ids.cpu().numpy(),
                    "attention_mask": mask.cpu().numpy(),
                    "token_type_ids": token.cpu().numpy()
                }
            )[0]
        else:
            runner = lambda s=session, x=inp: s.run(None, {"input": x.cpu().numpy()})[0]

    else:
        print(f"[ERROR] Unknown backend: {backend}")
        return None

    return runner


def run_hybrid_pipeline(hybrid_config, model, ts_model, onnx_path, wl_name, inp, batch_size, cfg):
    """Create a true hybrid pipeline that combines different backends."""

    # PyTorch for preprocessing + ONNX Runtime for inference
    if hybrid_config == "torch+onnx_cuda" and wl_name == "bert":
        # Load ONNX model with CUDA provider
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = create_onnx_session(onnx_path, providers)
        if session is None:
            return None

        def runner(m=model, s=session, texts=inp):
            # Use PyTorch for tokenization
            encoded = m.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = encoded["input_ids"].cpu().numpy()
            attention_mask = encoded["attention_mask"].cpu().numpy()
            token_type_ids = encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).cpu().numpy()

            # Use ONNX Runtime for inference
            return s.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            })[0]

        return runner

    # TorchScript for preprocessing + ONNX Runtime for inference
    elif hybrid_config == "ts+onnx_cuda" and wl_name == "bert":
        # This is a more complex case since tokenization is hard to script
        # We'll use a pre-tokenized approach for benchmarking

        # Load ONNX model with CUDA provider
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = create_onnx_session(onnx_path, providers)
        if session is None:
            return None

        # Pre-tokenize the input
        encoded = model.tokenizer(inp, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids)).to(DEVICE)

        def runner(s=session, ids=input_ids, mask=attention_mask, token=token_type_ids):
            # Use ONNX Runtime for inference with pre-tokenized input
            return s.run(None, {
                "input_ids": ids.cpu().numpy(),
                "attention_mask": mask.cpu().numpy(),
                "token_type_ids": token.cpu().numpy()
            })[0]

        return runner

    # PyTorch for feature extraction + TorchScript for inference
    elif hybrid_config == "torch+ts" and wl_name in ["conv2d", "resnet50"]:
        if ts_model is None:
            return None

        # For vision models, simulate feature extraction in PyTorch and classification in TorchScript
        def runner(m=model, ts=ts_model, x=inp):
            # Extract features with PyTorch (simulate by using part of the model)
            features = x  # In a real scenario, you'd do some preprocessing here
            # Run inference with TorchScript
            return ts(features)

        return runner

    # PyTorch for preprocessing + TorchScript + ONNX Runtime for different parts
    elif hybrid_config == "torch+ts+onnx_cuda" and wl_name == "attention":
        if ts_model is None:
            return None

        # Load ONNX model with CUDA provider
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = create_onnx_session(onnx_path, providers)
        if session is None:
            return None

        # Simulate a complex pipeline where different components run on different backends
        def runner(m=model, ts=ts_model, s=session, x=inp):
            # Step 1: Preprocess with PyTorch
            preprocessed = x + 0.01  # Simple preprocessing for simulation

            # Step 2: Run part of the model with TorchScript
            intermediate = ts(preprocessed)

            # Step 3: Run final part with ONNX Runtime
            return s.run(None, {"input": intermediate.cpu().numpy()})[0]

        return runner

    # For other workloads, implement appropriate hybrid pipelines
    elif wl_name == "matmul":
        # For MatMul, we can split the operation
        if hybrid_config == "torch+onnx_cuda":
            # Load ONNX model with CUDA provider
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            session = create_onnx_session(onnx_path, providers)
            if session is None:
                return None

            def runner(m=model, s=session, x=inp):
                # Preprocess in PyTorch
                preprocessed = x * 0.5 + 0.5  # Normalize to [0,1] range
                # Run inference in ONNX Runtime
                return s.run(None, {"input": preprocessed.cpu().numpy()})[0]

            return runner

    # If no specific hybrid pipeline is implemented for this combination
    print(f"[WARN] No specific hybrid pipeline implemented for {hybrid_config} with {wl_name}")
    return None


def apply_fp16_quantization(model_path, output_path):
    """Apply FP16 quantization using onnxconverter-common."""
    try:
        model = onnx.load(model_path)
        model_fp16 = float16.convert_float_to_float16(
            model,
            min_positive_val=1e-7,
            max_finite_val=1e4,
            keep_io_types=True  # Keep inputs/outputs as float32 for compatibility
        )
        onnx.save(model_fp16, output_path)
        return output_path
    except Exception as e:
        print(f"[WARN] FP16 quantization failed: {str(e)}, using original model")
        return model_path


def generate_reports(results_df):
    """Generate reports analyzing both latency and throughput."""
    os.makedirs("reports", exist_ok=True)

    # Create latency and throughput comparison charts
    for metric in ["avg_latency_ms", "avg_throughput_items_per_sec"]:
        plt.figure(figsize=(14, 10))

        workloads = results_df["workload"].unique()
        rows = (len(workloads) + 1) // 2
        cols = 2

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        for i, wl_name in enumerate(workloads):
            wl_data = results_df[results_df["workload"] == wl_name]

            # Group by backend, fusion, quant, and batch
            grouped = wl_data.groupby(["backend", "fusion", "quant", "batch"])[metric].mean().reset_index()

            # Plot relationships between batch size and performance metric
            for config in grouped.groupby(["backend", "fusion", "quant"]):
                config_name, config_data = config
                axes[i].plot(config_data["batch"], config_data[metric],
                             marker='o', label=f"{config_name[0]}/{config_name[1]}/{config_name[2]}")

            title = "Latency" if "latency" in metric else "Throughput"
            axes[i].set_title(f"{wl_name} - {title} vs Batch Size")
            axes[i].set_xlabel("Batch Size")
            axes[i].set_ylabel("ms" if "latency" in metric else "items/sec")
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].legend(fontsize=8)

        # Hide any unused subplots
        for j in range(len(workloads), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        metric_name = "latency" if "latency" in metric else "throughput"
        plt.savefig(f"reports/{metric_name}_analysis.png")
        plt.close()


def analyze_batch_size_performance(results_df, output_dir="reports"):
    """Analyze how different backends perform across batch sizes."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter to just the basic backends without fusion/quantization for clarity
    basic_results = results_df[
        (results_df["fusion"] == "none") &
        (results_df["quant"] == "fp32") &
        (~results_df["backend"].str.contains("\\+"))
        ]

    # Group by workload, backend, and batch size
    grouped = basic_results.groupby(["workload", "backend", "batch"])[
        ["avg_latency_ms", "avg_throughput_items_per_sec"]
    ].mean().reset_index()

    # Plot each workload in a separate subplot
    workloads = grouped["workload"].unique()
    cols = 2
    rows = (len(workloads) + 1) // 2

    # Create latency figure
    fig_lat, axes_lat = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows > 1:
        axes_lat = axes_lat.flatten()
    else:
        axes_lat = [axes_lat]

    # Create throughput figure
    fig_thpt, axes_thpt = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows > 1:
        axes_thpt = axes_thpt.flatten()
    else:
        axes_thpt = [axes_thpt]

    for i, workload in enumerate(workloads):
        workload_data = grouped[grouped["workload"] == workload]

        for backend in workload_data["backend"].unique():
            backend_data = workload_data[workload_data["backend"] == backend]

            # Plot latency
            axes_lat[i].plot(backend_data["batch"], backend_data["avg_latency_ms"],
                             marker='o', label=backend)

            # Plot throughput
            axes_thpt[i].plot(backend_data["batch"], backend_data["avg_throughput_items_per_sec"],
                              marker='o', label=backend)

        # Configure latency plot
        axes_lat[i].set_title(f"{workload} - Latency vs Batch Size")
        axes_lat[i].set_xlabel("Batch Size")
        axes_lat[i].set_ylabel("Latency (ms)")
        axes_lat[i].grid(True, linestyle='--', alpha=0.7)
        axes_lat[i].legend()

        # Configure throughput plot
        axes_thpt[i].set_title(f"{workload} - Throughput vs Batch Size")
        axes_thpt[i].set_xlabel("Batch Size")
        axes_thpt[i].set_ylabel("Throughput (items/sec)")
        axes_thpt[i].grid(True, linestyle='--', alpha=0.7)
        axes_thpt[i].legend()

    # Hide any unused subplots
    for j in range(len(workloads), len(axes_lat)):
        axes_lat[j].axis('off')
        axes_thpt[j].axis('off')

    fig_lat.tight_layout()
    fig_thpt.tight_layout()

    # Save figures
    fig_lat.savefig(f"{output_dir}/latency_by_batch.png")
    fig_thpt.savefig(f"{output_dir}/throughput_by_batch.png")
    plt.close(fig_lat)
    plt.close(fig_thpt)

    # Create a summary table
    summary = []
    for workload in workloads:
        workload_data = grouped[grouped["workload"] == workload]

        for batch in workload_data["batch"].unique():
            batch_data = workload_data[workload_data["batch"] == batch]

            # Find best backend for latency
            lat_idx = batch_data["avg_latency_ms"].idxmin()
            best_lat_backend = batch_data.loc[lat_idx]["backend"]
            best_lat = batch_data.loc[lat_idx]["avg_latency_ms"]

            # Find best backend for throughput
            thpt_idx = batch_data["avg_throughput_items_per_sec"].idxmax()
            best_thpt_backend = batch_data.loc[thpt_idx]["backend"]
            best_thpt = batch_data.loc[thpt_idx]["avg_throughput_items_per_sec"]

            summary.append({
                "workload": workload,
                "batch_size": batch,
                "best_latency_backend": best_lat_backend,
                "best_latency_ms": best_lat,
                "best_throughput_backend": best_thpt_backend,
                "best_throughput_items_per_sec": best_thpt
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_dir}/batch_size_summary.csv", index=False)

    return summary_df


# ─── Main Benchmark Function ───────────────────────────────────────────────────
def run_benchmark():
    # Define solo backends
    solo_backends = ["torch", "inductor", "ts", "onnx_cuda", "tensorrt", "onnx_trt"]

    # Define hybrid configurations - true pipelines, not just averages
    hybrid_configs = [
        "torch+onnx_cuda",
        "ts+onnx_cuda",
        "torch+ts",
        "torch+ts+onnx_cuda",
        "inductor+onnx_cuda",
        "inductor+ts",
        "inductor+onnx_trt",
        "ts+onnx_trt",
        "torch+onnx_trt",
        "inductor+ts+onnx_cuda",
    ]

    fusions = ["none", "op", "graph"]
    quants = ["fp32", "fp16", "int8_dyn", "int8_static"]
    batches = [1]

    # Configurable benchmarking parameters
    benchmark_config = {
        "runs": 10,  # Number of measurement runs
        "warmup_runs": 5,  # Number of warm-up runs
        "flush_cache": True,  # Flush cache before measurement
        "flush_between_runs": True  # Flush between individual runs for consistency
    }

    all_results = []

    for wl_name, (wl_cls, cfg) in WORKLOADS.items():
        for batch in batches:
            print(f"\n{'=' * 80}\nBenchmarking {wl_name} with batch size {batch}\n{'=' * 80}")

            # Reset GPU state
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Instantiate model and input
            model = wl_cls().to(DEVICE).eval()
            inp = make_input(wl_name, batch, cfg)

            # Create TorchScript model
            ts_model = None
            try:
                if wl_name == "bert":
                    # For BERT, use a scriptable wrapper
                    wrapper = BERTOnnxWrapper(model)
                    input_ids, attention_mask, token_type_ids = create_bert_inputs(model, inp)
                    ts_model = torch.jit.trace(
                        wrapper,
                        (input_ids, attention_mask, token_type_ids),
                        strict=False
                    ).eval().to(DEVICE)
                else:
                    # For other models, use standard tracing
                    ts_model = torch.jit.trace(model, inp, strict=False).eval().to(DEVICE)
            except Exception as e:
                print(f"[ERROR] Failed to create TorchScript model for {wl_name}: {str(e)}")
                # Try to diagnose the issue
                if "keyword-arg" in str(e) and wl_name == "bert":
                    print("[INFO] BERT model failed to trace due to keyword arguments. Using wrapper instead.")
                    try:
                        # Try with a simpler wrapper
                        class SimpleBERTWrapper(nn.Module):
                            def __init__(self, model):
                                super().__init__()
                                self.model = model

                            def forward(self, input_ids, attention_mask):
                                return self.model.net(input_ids=input_ids,
                                                      attention_mask=attention_mask).last_hidden_state

                        wrapper = SimpleBERTWrapper(model)
                        input_ids, attention_mask, _ = create_bert_inputs(model, inp)
                        ts_model = torch.jit.trace(
                            wrapper,
                            (input_ids, attention_mask),
                            strict=False
                        ).eval().to(DEVICE)
                    except Exception as e2:
                        print(f"[ERROR] Alternative tracing also failed: {str(e2)}")

            # Export ONNX
            os.makedirs("models", exist_ok=True)
            onnx_path = f"models/{wl_name}_bs{batch}.onnx"
            onnx_export_success = export_onnx(model, inp, onnx_path, wl_name, batch)

            # Run solo backends
            for backend in solo_backends:
                # Skip backends that require ONNX if export failed
                if not onnx_export_success and backend.startswith("onnx"):
                    print(f"Skipping {backend} for {wl_name} as ONNX export failed")
                    continue

                # Skip TorchScript if creation failed
                if backend == "ts" and ts_model is None:
                    print(f"Skipping {backend} for {wl_name} as TorchScript creation failed")
                    continue

                for fusion in fusions:
                    # Skip fusion if not using ONNX backend
                    if not backend.startswith("onnx") and fusion != "none":
                        continue

                    # copy & fuse
                    fused_path = f"models/{wl_name}_bs{batch}_{fusion}.onnx"
                    if backend.startswith("onnx"):
                        os.system(f"cp {onnx_path} {fused_path}")
                        if fusion != "none":
                            fusion_success = apply_fusion(fused_path, fusion, wl_name)
                            if not fusion_success:
                                print(f"Skipping {fusion} fusion for {wl_name} as fusion failed")
                                continue

                    for quant in quants:
                        # Skip incompatible configurations
                        if not is_compatible(wl_name, backend, quant):
                            print(f"Skipping incompatible: {wl_name} with {backend}/{fusion}/{quant}")
                            continue

                        # Skip quantization if not using ONNX backend
                        if not backend.startswith("onnx") and quant != "fp32":
                            continue

                        # Reset GPU state before each test
                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()

                        # Apply quantization
                        onnx_qpath = fused_path
                        if backend.startswith("onnx") and quant != "fp32":
                            # Pre-process the model before quantization
                            preprocessed_path = f"models/{wl_name}_bs{batch}_{fusion}_preprocessed.onnx"
                            preprocess_onnx_model(fused_path, preprocessed_path)

                            if quant == "fp16":
                                fp16_path = f"models/{wl_name}_bs{batch}_{fusion}_fp16.onnx"
                                onnx_qpath = apply_fp16_quantization(preprocessed_path, fp16_path)
                            elif quant == "int8_dyn":
                                try:
                                    target = f"models/{wl_name}_bs{batch}_{fusion}_int8_dyn.onnx"

                                    # Exclude conv nodes from quantization
                                    model_onnx = onnx.load(preprocessed_path)
                                    conv_nodes = [node.name for node in model_onnx.graph.node if node.op_type == 'Conv']

                                    quantize_dynamic(
                                        preprocessed_path,
                                        target,
                                        weight_type=QuantType.QInt8,
                                        nodes_to_exclude=conv_nodes
                                    )
                                    onnx_qpath = target
                                except Exception as e:
                                    print(
                                        f"[WARN] Dynamic INT8 quantization failed: {str(e)}, using pre-processed model")
                                    onnx_qpath = preprocessed_path
                            elif quant == "int8_static":
                                try:
                                    target = f"models/{wl_name}_bs{batch}_{fusion}_int8_static.onnx"

                                    # Create calibration data reader
                                    dr = CalibrationDataGenerator(wl_name, batch, cfg)

                                    # Exclude conv nodes from quantization
                                    model_onnx = onnx.load(preprocessed_path)
                                    conv_nodes = [node.name for node in model_onnx.graph.node if node.op_type == 'Conv']

                                    quantize_static(
                                        preprocessed_path,
                                        target,
                                        dr,
                                        weight_type=QuantType.QInt8,
                                        quant_format=QuantFormat.QDQ,
                                        nodes_to_exclude=conv_nodes
                                    )
                                    onnx_qpath = target
                                except Exception as e:
                                    print(
                                        f"[WARN] Static INT8 quantization failed: {str(e)}, using pre-processed model")
                                    onnx_qpath = preprocessed_path

                        label = f"{backend}/{fusion}/{quant}"

                        # Create runner function
                        runner = run_inference_solo(backend, model, ts_model, onnx_qpath, wl_name, inp, batch, cfg)
                        if runner is None:
                            print(f"Failed to create runner for {wl_name} with {label}")
                            continue

                        # Profile with proper warm-up and cache flushing
                        try:
                            metrics = profile_runner(
                                runner,
                                runs=benchmark_config["runs"],
                                warmup_runs=benchmark_config["warmup_runs"],
                                flush_cache=benchmark_config["flush_cache"],
                                flush_between_runs=benchmark_config["flush_between_runs"],
                                batch_size=batch
                            )

                            result = {
                                "id": str(uuid.uuid4()),
                                "workload": wl_name,
                                "batch": batch,
                                "backend": backend,
                                "fusion": fusion,
                                "quant": quant,
                                # Latency metrics
                                "avg_latency_ms": metrics["avg_latency_ms"],
                                "min_latency_ms": metrics["min_latency_ms"],
                                "max_latency_ms": metrics["max_latency_ms"],
                                "p50_latency_ms": metrics["p50_latency_ms"],
                                "p95_latency_ms": metrics["p95_latency_ms"],
                                "p99_latency_ms": metrics["p99_latency_ms"],
                                "std_latency_ms": metrics["std_latency_ms"],
                                # Throughput metrics
                                "avg_throughput_items_per_sec": metrics["avg_throughput_items_per_sec"],
                                "max_throughput_items_per_sec": metrics["max_throughput_items_per_sec"],
                                "peak_memory_mb": metrics["peak_memory_mb"],
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            all_results.append(result)

                            print(f"{wl_name:8s} bs={batch:2d} | {label:30s} → "
                                  f"Latency: {metrics['avg_latency_ms']:6.2f}±{metrics['std_latency_ms']:5.2f} ms, "
                                  f"Throughput: {metrics['avg_throughput_items_per_sec']:8.2f} items/s, "
                                  f"Memory: {metrics['peak_memory_mb'] if metrics['peak_memory_mb'] else '-':6} MB")
                        except Exception as e:
                            print(f"[ERROR] Failed to run {wl_name} with {label}: {str(e)}")

                        # Clean up after each test
                        # Clean up after each test
                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()

            # Run hybrid configurations
            for hybrid in hybrid_configs:
                # Skip incompatible workloads
                if wl_name == "bert" and hybrid in ["torch+ts", "torch+ts+onnx_cuda"]:
                    # These hybrids are not implemented for BERT
                    continue

                # Create runner
                runner = run_hybrid_pipeline(hybrid, model, ts_model, onnx_path, wl_name, inp, batch, cfg)
                if runner is None:
                    print(f"Skipping hybrid {hybrid} for {wl_name} due to missing runner")
                    continue

                label = f"{hybrid}/none/fp32"

                try:
                    metrics = profile_runner(
                        runner,
                        runs=benchmark_config["runs"],
                        warmup_runs=benchmark_config["warmup_runs"],
                        flush_cache=benchmark_config["flush_cache"],
                        flush_between_runs=benchmark_config["flush_between_runs"],
                        batch_size=batch
                    )

                    result = {
                        "id": str(uuid.uuid4()),
                        "workload": wl_name,
                        "batch": batch,
                        "backend": hybrid,
                        "fusion": "none",
                        "quant": "fp32",
                        # Latency metrics
                        "avg_latency_ms": metrics["avg_latency_ms"],
                        "min_latency_ms": metrics["min_latency_ms"],
                        "max_latency_ms": metrics["max_latency_ms"],
                        "p50_latency_ms": metrics["p50_latency_ms"],
                        "p95_latency_ms": metrics["p95_latency_ms"],
                        "p99_latency_ms": metrics["p99_latency_ms"],
                        "std_latency_ms": metrics["std_latency_ms"],
                        # Throughput metrics
                        "avg_throughput_items_per_sec": metrics["avg_throughput_items_per_sec"],
                        "max_throughput_items_per_sec": metrics["max_throughput_items_per_sec"],
                        "peak_memory_mb": metrics["peak_memory_mb"],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    all_results.append(result)

                    print(f"{wl_name:8s} bs={batch:2d} | {label:30s} → "
                          f"Latency: {metrics['avg_latency_ms']:6.2f}±{metrics['std_latency_ms']:5.2f} ms, "
                          f"Throughput: {metrics['avg_throughput_items_per_sec']:8.2f} items/s, "
                          f"Memory: {metrics['peak_memory_mb'] if metrics['peak_memory_mb'] else '-':6} MB")
                except Exception as e:
                    print(f"[CRITICAL] Benchmark crashed: {e}")
                    # dump partial results
                    df_partial = pd.DataFrame(all_results)
                    df_partial.to_csv("results/benchmark_results_partial.csv", index=False)
                    print("✅ Partial results saved to results/benchmark_results_partial.csv")
                    # re-raise so you still see the traceback
                    raise

                # Clean up after each test
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

    # Save results and generate reports
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/benchmark_results.csv", index=False)

    # Generate reports
    generate_reports(results_df)
    analyze_batch_size_performance(results_df)

    return results_df


# Run the benchmark if this script is executed
if __name__ == "__main__":
    results = run_benchmark()
    print("Benchmarking complete. Results saved to results/benchmark_results.csv")