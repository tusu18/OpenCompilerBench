
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


---

## ⚠️ Current Problems in Production with ML Compiler Stacks (Detailed Table)

| 🔹 Issue | 💬 Description | 🧠 Real-World Example | 📰 In the News |
|---------|----------------|----------------------|----------------|
| **1. One-size-fits-all Compilers** | Teams pick one compiler (e.g., ONNX + TensorRT) for all models, ignoring model/device differences. | MobileBERT with ONNX on Android is 2× slower than TFLite INT8. TFLite integrates better with Android NNAPI. | Google’s Gemini 1.5 uses mixed compilers (TPU + mobile); Hugging Face supports ONNX, TFLite, TensorRT in Optimum. |
| **2. Inefficient Model–Hardware Match** | Models compiled for GPU often crash or underperform on mobile/edge devices. | Whisper-large compiled for A100 fails on Jetson Nano due to unsupported ops or VRAM limits. | OpenAI Whisper forks exist for TinyML; Edge Impulse and Apple push platform-specific compilers. |
| **3. Long Compilation Cycles** | TVM, AITemplate require hours for model-device tuning, slowing production. | TVM’s Ansor takes 8+ hours to compile ViT on Jetson Orin. | Meta’s AITemplate (used in Stable Diffusion & LLaMA) is fast but takes time to compile. Transfer tuning is a workaround. |
| **4. Manual Trial-and-Error** | Engineers manually test compilers, quantization, batch sizes. | YOLOv8 tested with TensorRT FP16 (unstable), TVM INT8 (accurate), TFLite (mobile). Time-consuming & non-reproducible. | Hugging Face + OpenVINO provide partial automation. NVIDIA Triton still needs manual tuning. |
| **5. Hardware Fragmentation** | Models must be compiled separately for Android, iOS, web, desktop. | AR app with CLIP needs TFLite (Android), CoreML (iOS), ONNX (web), each with separate compilation/testing. | Google uses TFLite, TF.js, XLA; Apple uses CoreML + ANE with custom toolchain. |
| **6. Unpredictable Latency / Memory** | Compilers may cause latency spikes or memory crashes with unsafe quantization. | LLaMA-7B INT8 quantized in TensorRT without calibration drops BLEU from 84 → 68 and spikes memory usage. | NVIDIA advises calibration caches and per-layer tuning. Mistral and Phi-3 focus on safe quantization. |

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

