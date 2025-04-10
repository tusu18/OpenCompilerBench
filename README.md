
# 🔧 OpenCompBench: Benchmarking & Hybrid Compiler Pipelines for ML Models

![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Research%20Prototype-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-yellow)

OpenCompBench is a research-driven benchmarking framework designed to evaluate, compare, and intelligently orchestrate machine learning compilers like **TVM**, **TensorRT**, **TFLite**, **ONNX Runtime**, **IREE**, and more.

The project includes:
- 📊 Standardized benchmarks for **vision, NLP, speech, and multimodal models**
- ⚙️ Support for **quantization strategies** (FP32, FP16, INT8)
- 🔁 Construction of **hybrid compiler pipelines** (e.g., TVM for vision, TensorRT for LLMs)
- 🧠 Foundation for a **meta-compiler** to auto-select the best compilation stack

---

## 🚀 Motivation

Deploying foundation models like **LLaMA**, **DeepSeek**, **Whisper**, or **BLIP-2** across diverse hardware platforms — from **A100 GPUs** to **Android phones** to **microcontrollers** — requires **carefully optimized compiler stacks**.

However, today's ecosystem suffers from several pain points:

- ❌ No **universal compiler** works well across models and hardware.
- ❌ Teams rely on **trial-and-error** to find the best compiler+quantization combo.
- ❌ There's a **lack of standardized benchmarks** to guide these decisions.


## ⚠️ Current Problems in Production with ML Compiler Stacks

| Issue                        | Description                                                                                     | Example                                                                 |
|-----------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| One-size-fits-all Compilers | Teams use a single compiler (e.g., ONNX + TensorRT) for all models, even when suboptimal.      | MobileBERT with ONNX on Android is 2× slower than TFLite INT8.         |
| Inefficient HW Match        | Models compiled for GPU often fail or underperform on CPU/mobile without modification.          | Whisper-Large for A100 crashes on Jetson Nano.                         |
| Long Compilation Cycles     | Auto-tuning in compilers like TVM or AITemplate can take hours per target per model.            | TVM takes 8+ hours to optimize ViT on Jetson Orin.                     |
| Manual Trial-and-Error      | Teams manually explore compiler + quantization + batch size settings to find best combination.  | Engineers try TensorRT FP16, TVM INT8, TFLite separately on YOLOv8.    |
| Hardware Fragmentation      | Each platform requires separate compilation targets (CoreML, TFLite, TensorRT, etc.).           | Apps include TFLite (Android), CoreML (iOS), ONNX (desktop) versions.  |
| Unpredictable Latency       | Compilers may pick unsafe quantization or kernel schedules, leading to latency spikes/crashes. | TensorRT INT8 on LLaMA-7B drops BLEU from 84 → 68 without calibration. |

> These issues underscore the need for tools like OpenCompBench and meta-compilers that provide data-driven, optimized compiler selection — rather than relying on guesswork or defaults.

---

### 📊 Why Benchmarking + Data Collection Matters

We aim to solve these challenges by building a large-scale dataset that maps:

| Model | Compiler Stack | Quantization | Target Device | Latency | Memory | Accuracy |
|-------|----------------|--------------|----------------|---------|--------|----------|

This **data is the foundation** for building a **Meta-Compiler** — an intelligent system that can:

- 🔍 Predict the best compiler pipeline for any new model/device
- ⚙️ Automatically tune quantization, layout, and batching
- 📦 Output a deployment-ready binary with optimal performance

---

### 🧠 Use Case Example

Imagine uploading a model to a deployment service and the system responds:

> "You're deploying a ViT-L encoder with a Transformer decoder on Jetson Orin Nano.  
> Based on our benchmark data, we recommend:  
> - TVM for vision, quantized INT8  
> - TensorRT for transformer decoder in FP16  
> - Expected latency: 820ms, Memory: 1.2GB, Accuracy drop: <1%"

That’s what this benchmark-powered meta-compiler aims to deliver.


## 📦 Supported Compilers

| Compiler      | Features |
|---------------|----------|
| [TVM](https://tvm.apache.org/)            | Auto-tuning, heterogeneous HW, IR introspection |
| [TensorRT](https://developer.nvidia.com/tensorrt) | Low-latency GPU inference, INT8/FP16 support     |
| [TFLite](https://www.tensorflow.org/lite) | Mobile & embedded edge optimization              |
| [ONNX Runtime](https://onnxruntime.ai/)   | Broad platform compatibility                     |
| [IREE](https://github.com/openxla/iree)   | MLIR-based cross-platform runtime                |

---

## 📊 Benchmarking Metrics

Each model-compiler-device combination is profiled on:

| Metric             | Description                        |
|--------------------|------------------------------------|
| Inference Latency  | Time per inference (batch = 1, N)  |
| Throughput         | Inferences/sec                     |
| Peak Memory Usage  | RAM/VRAM used at runtime           |
| Model Size         | Size of compiled binary            |
| Accuracy Retention | Drop vs original model             |
| Compiler Time      | Time to generate executable        |
| Cross-HW Portability | Can run on CPU, GPU, mobile, etc |

---

## 📦 Supported Models

| Domain       | Models                              |
|--------------|-------------------------------------|
| Vision       | ResNet50, YOLOv8, ViT, DETR         |
| NLP / LLMs   | GPT2, LLaMA-7B, mBART, T5            |
| Multimodal   | CLIP, BLIP-2, Flamingo              |
| Speech       | Whisper, Conformer                  |
| Edge Models  | MobileBERT, TinyViT, DistilWhisper  |

---

## 📈 Meta-Compiler (Experimental)

This project includes a prototype **judge model** that:
- Takes model structure, device specs, and constraints (e.g., latency < 50ms)
- Predicts the best compiler + quantization stack
- Outputs an optimized pipeline for deployment

> Future goal: plug into CI/CD for auto-compiled models per device.

---

## 🤝 Contributing

We welcome:
- New compiler runners
- Additional benchmark models
- Meta-compiler improvements
- Device-specific tuning strategies

Submit a PR or open an issue to get started 🚀

---

## 📜 License

MIT License — free to use, modify, and build upon.

---

## 💬 Acknowledgements

Inspired by:
- Apache TVM
- NVIDIA TensorRT
- MLPerf Inference
- Hugging Face Transformers

---

