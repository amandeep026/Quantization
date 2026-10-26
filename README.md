# Pillar 4: Automated Hardware-Aware Quantization and Heterogeneous Execution Engine

This repository contains the production-grade, mathematically rigorous implementation of **Precision Foundations**, **Post-Training Quantization (PTQ)**, and **Heterogeneous Hardware Acceleration & Runtime Configuration** for **Pillar 4 (Automated Hardware-Aware Quantization)** of the *Optimizing AI Architecture* project.

---

## 1. Architectural Overview & System Structure

```
c:\Users\amand\OneDrive\Desktop\Amandeep minor\
├── quantization/
│   ├── __init__.py               # Exported API and unified package namespace (60 exports)
│   ├── precision.py              # FP32, FP16, BF16, INT8, INT4 specs, bit disassembly & sub-byte packing
│   ├── ptq.py                    # PTQ Core: Symmetric, Asymmetric, Granularities (Tensor/Channel/Group)
│   ├── calibration.py            # Calibration & Clipping: MinMax, Percentile, MSE, KL-Divergence
│   ├── layers.py                 # QuantizedLinear layer with INT8/INT4 storage, GEMM & hardware dispatch
│   ├── metrics.py                # SQNR (dB), MSE, MAE, Cosine Similarity, Compression Ratio
│   ├── simd.py                   # CPU SIMD: AVX-512 (512-bit ZMM, VNNI VPDPBUSD) & ARM NEON (128-bit Q, SDOT)
│   ├── gpu_cuda.py               # GPU Acceleration: CUDA & Tensor Cores (Warp MMA tiles, Volta-Hopper evolution)
│   ├── npu_mps.py                # NPU (2D Systolic Array WS/OS, zero-skipping) & Apple MPS (UMA zero-copy)
│   ├── roofline.py               # Memory-Bound vs. Compute-Bound Operations & Roofline Model Analyzer
│   ├── hardware_profiler.py      # Hardware Profiling: RAM, VRAM, Empirical Bandwidth, CPU Caches, and Ridge Point
│   └── runtime_config.py         # Dynamic Runtime Configuration: Threads, Layer Offloading (-ngl), and Context Sizing
├── tests/
│   ├── __init__.py
│   ├── test_quantization.py      # Precision, PTQ, and calibration unit tests (17 tests - 100% PASS)
│   ├── test_hardware_acceleration.py # SIMD, CUDA/Tensor Cores, NPU/MPS & Profiler tests (22 tests - 100% PASS)
│   └── test_runtime_config.py    # Roofline, Hardware Profiling, and Runtime Config tests (10 tests - 100% PASS)
├── demo_quantization.py          # Part 1-3 interactive PTQ demonstration & calibration benchmark
├── demo_hardware_acceleration.py # Part 4-6 interactive hardware acceleration & profiler benchmark
├── demo_runtime_configuration.py # Part 7-9 interactive Roofline, Bandwidth, and Layer Offloading benchmark
└── README.md                     # Complete architectural formulations, mathematical proofs & technical reference
```

---

## 2. Floating-Point vs. Integer Precision

### 2.1 Theoretical Specifications & Bit Layouts

| Format | Total Bits | Bytes / Element | Sign Bits | Exponent Bits | Mantissa Bits | Dynamic Range ($[\min, \max]$) | Machine $\epsilon$ | Memory Savings vs FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32** | 32 | 4.0 | 1 | 8 (bias 127) | 23 | $\approx \pm 3.40 \times 10^{38}$ | $1.19 \times 10^{-7}$ | Baseline (0%) |
| **FP16** | 16 | 2.0 | 1 | 5 (bias 15) | 10 | $[-65504, 65504]$ | $9.77 \times 10^{-4}$ | **50.0%** |
| **BF16** | 16 | 2.0 | 1 | 8 (bias 127) | 7 | $\approx \pm 3.39 \times 10^{38}$ | $7.81 \times 10^{-3}$ | **50.0%** |
| **INT8** | 8 | 1.0 | 1 | 0 | 7 | $[-128, 127]$ | Fixed Point | **75.0%** |
| **INT4** | 4 | 0.5 | 1 | 0 | 3 | $[-8, 7]$ or $[0, 15]$ | Fixed Point | **87.5%** |

### 2.2 Mathematical Representation

#### IEEE 754 Floating-Point Formula
$$x = (-1)^s \times 2^{e - \text{bias}} \times \left(1 + \sum_{i=1}^m b_{m-i} 2^{-i}\right)$$

* **FP32 vs. FP16 vs. BF16 Trade-off:**
  * **FP16** allocates 5 exponent bits, causing frequent gradient/activation overflow when values exceed $65,504$ (requiring dynamic loss scaling).
  * **BF16** preserves FP32's 8 exponent bits, matching its vast dynamic range ($\approx 10^{38}$) with zero overflow risk, while trading off mantissa precision (7 bits vs 10 bits in FP16).
  * **INT8 / INT4** eliminate floating-point exponents entirely, discretizing real values into uniform linear grid points.

### 2.3 Sub-Byte Bit Packing for INT4
Standard computer memory architectures cannot directly address 4-bit nibbles. We implement hardware-aligned **sub-byte packing**:
$$\text{packed\_byte}_i = (\text{nibble}_{2i+1} \ll 4) \;\Big|\; (\text{nibble}_{2i} \;\&\; 0\text{x}0\text{F})$$
During dequantization, the engine unpacks each byte and executes **two's complement sign-extension**:
$$q = \begin{cases} \text{raw\_nibble} - 16 & \text{if } \text{raw\_nibble} \ge 8 \\ \text{raw\_nibble} & \text{otherwise} \end{cases}$$

---

## 3. Post-Training Quantization (PTQ)

Post-Training Quantization maps a continuous real-valued tensor $x \in \mathbb{R}$ into discrete integer values $q \in [q_{\min}, q_{\max}]$.

### 3.1 Symmetric Quantization
Symmetric quantization restricts the integer grid to be strictly symmetric about zero ($Z = 0$).

* **Range:** $[-x_{\max}, x_{\max}]$ where $x_{\max} = \max(|x_{\min}|, |x_{\max}|)$.
* **Scale Factor ($S$):**
  $$S = \frac{x_{\max}}{q_{\max}}$$
* **Quantization Operator:**
  $$q = \text{clamp}\left( \left\lfloor \frac{x}{S} \right\rceil, -q_{\max}, q_{\max} \right)$$
* **Dequantization Operator:**
  $$\hat{x} = q \times S$$
* **Best Used For:** Model weights (which are typically zero-centered Gaussian distributions) and high-speed GEMM compute engines.

### 3.2 Asymmetric (Affine) Quantization
Asymmetric quantization allows an arbitrary integer zero-point ($Z \in \mathbb{Z}$) that precisely aligns real $0.0$ to an integer bin.

* **Scale Factor ($S$):**
  $$S = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$$
* **Zero-Point ($Z$):**
  $$Z = \text{clamp}\left( \left\lfloor -\frac{x_{\min}}{S} \right\rceil + q_{\min}, \; q_{\min}, \; q_{\max} \right)$$
* **Quantization Operator:**
  $$q = \text{clamp}\left( \left\lfloor \frac{x}{S} \right\rceil + Z, \; q_{\min}, \; q_{\max} \right)$$
* **Dequantization Operator:**
  $$\hat{x} = (q - Z) \times S$$
* **Best Used For:** Asymmetric/skewed distributions such as post-ReLU or post-GeLU activations ($x \ge 0$). Symmetric quantization wastes 50% of integer bins on negative values for non-negative activations, whereas Asymmetric PTQ achieves $+6\text{ to }+12\text{ dB}$ higher SQNR.

### 3.3 Scaling Granularities
1. **Per-Tensor Quantization:** Single scalar scale $S$ and zero-point $Z$ for the entire tensor.
2. **Per-Channel (Per-Axis) Quantization:** Dedicated scale $S_c$ and zero-point $Z_c$ for each output channel $c \in \{1, \dots, C_{\text{out}}\}$.
3. **Group-Wise / Block-Wise Quantization:** Divides weights along the dimension into small discrete blocks of size $G \in \{32, 64, 128\}$ (as utilized in GGUF, GPTQ, and AWQ).

---

## 4. Clipping and Calibration Strategies

1. **MinMax Calibration:** $x_{\min} = \min(x), x_{\max} = \max(x)$.
2. **Percentile / Quantile Clipping:** $x_{\min} = \text{percentile}(x, 100 - p), x_{\max} = \text{percentile}(x, p)$ ($p = 99.9\%$).
3. **MSE-Optimal Calibration (L2 Error Minimization):**
   $$\alpha^* = \arg\min_{\alpha} \frac{1}{N} \sum_{i=1}^N \left( x_i - \text{dequant}(\text{quant}(x_i, \alpha)) \right)^2$$
4. **KL-Divergence / Relative Entropy Calibration (TensorRT Style):**
   $$D_{\text{KL}}(P \parallel Q) = \sum_{i=1}^T P(i) \log\left( \frac{P(i)}{Q(i)} \right)$$

---

## 5. Integer GEMM and Zero-Point Correction

In fully quantized neural network layers, matrix multiplication between quantized activations $Q_X$ and quantized weights $Q_W$ expands as:

$$Y = X W^T = \left[ S_X (Q_X - Z_X) \right] \left[ S_W (Q_W - Z_W) \right]^T$$
$$Y = S_X S_W \left( Q_X Q_W^T \;-\; Z_W Q_X \mathbf{1}^T \;-\; Z_X \mathbf{1} Q_W^T \;+\; Z_X Z_W \mathbf{1}\mathbf{1}^T \right)$$

When both activations and weights are symmetrically quantized ($Z_X = 0, Z_W = 0$), all three offset correction terms vanish:
$$Y = S_X S_W \left( Q_X Q_W^T \right)$$
This runs directly on integer SIMD and Tensor Core instructions with zero correction overhead.

---

## 6. CPU SIMD: AVX-512 and ARM NEON

Modern CPUs achieve multi-gigahertz vector throughput by packaging multiple arithmetic operations into wide vector registers.

```mermaid
graph TD
    subgraph AVX512_Register["Intel AVX-512 (ZMM Register - 512 bits / 64 bytes)"]
        A1["Lane 0..3 (4x INT8)"] --> ACC1["INT32 Acc 0"]
        A2["Lane 4..7 (4x INT8)"] --> ACC2["INT32 Acc 1"]
        A3["..."] --> ACC3["..."]
        A4["Lane 60..63 (4x INT8)"] --> ACC16["INT32 Acc 15"]
    end
    subgraph ARM_NEON_Register["ARM NEON (Q Register - 128 bits / 16 bytes)"]
        N1["Lane 0..3 (4x INT8)"] --> NACC1["INT32 Acc 0"]
        N2["Lane 4..7 (4x INT8)"] --> NACC2["INT32 Acc 1"]
        N3["Lane 8..11 (4x INT8)"] --> NACC3["INT32 Acc 2"]
        N4["Lane 12..15 (4x INT8)"] --> NACC4["INT32 Acc 3"]
    end
```

### 6.1 Architectural Comparison

| Feature | Intel AVX-512 VNNI | Intel AVX2 + FMA | ARM NEON | Scalar Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Vector Register Family** | `ZMM0` - `ZMM31` (32 regs) | `YMM0` - `YMM15` (16 regs) | `Q0` - `Q15` / `V0` - `V31` | `RAX` - `R15` (GPR) |
| **Register Width** | **512 bits** (64 bytes) | **256 bits** (32 bytes) | **128 bits** (16 bytes) | **32 / 64 bits** (4-8 bytes) |
| **FP32 Vector Lanes** | 16 parallel FP32 | 8 parallel FP32 | 4 parallel FP32 | 1 element |
| **INT8 Vector Lanes** | 64 parallel INT8 | 32 parallel INT8 | 16 parallel INT8 | 1 element |
| **INT32 Accumulators** | 16 lanes | 8 lanes | 4 lanes | 1 accumulator |
| **Key Dot-Product Op** | `VPDPBUSD` (DL Boost) | Emulated via 16-bit unpack | `SDOT` / `UDOT` (ARMv8.2+) | Scalar Multiply-Add |
| **Theoretical INT8 Speedup**| **64.0x vs Scalar** | **16.0x vs Scalar** | **16.0x vs Scalar** | 1.0x (Baseline) |

---

## 7. Memory-Bound vs. Compute-Bound Operations (The Roofline Model)

The **Roofline Model** defines the fundamental physical ceiling of compute performance based on memory bandwidth and operational intensity.

```mermaid
graph LR
    subgraph Roofline_Curve["The Roofline Model Curve"]
        direction TB
        Mem_Bound["Memory-Bound Slanted Ceiling: P = Intensity x Bandwidth"] --> Ridge["Ridge Point I* = Peak Compute / Peak Bandwidth"]
        Ridge --> Comp_Bound["Compute-Bound Flat Ceiling: P = Peak Compute GFLOPS"]
    end
```

### 7.1 Mathematical Formulations

1. **Arithmetic (Operational) Intensity ($I$):**
   $$I = \frac{\text{Total Operations (FLOPs or MACs)}}{\text{Total Memory Traffic (Bytes accessed)}} \quad \left[\frac{\text{FLOPs}}{\text{Byte}} \text{ or } \frac{\text{OPs}}{\text{Byte}}\right]$$
2. **Machine Balance (Ridge Point $I^*$):**
   $$I^* = \frac{\text{Peak Compute Throughput (GFLOPS)}}{\text{Peak Memory Bandwidth (GB/s)}} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]$$
3. **Attainable Performance Ceiling ($P$):**
   $$P = \min\left( P_{\text{peak}}, \; I \times B_{\text{peak}} \right)$$

### 7.2 Why Autoregressive Decoding is Strictly Memory-Bound

In token-by-token generation ($M=1$), every single token generated requires loading the entire model parameter weight matrix $W \in \mathbb{R}^{N \times K}$ from memory into cache/registers:

$$I_{\text{decode}} = \frac{2 \times 1 \times N \times K}{N \times K \times \text{bytes\_per\_weight} + 2K \times \text{bytes\_per\_act}} \approx \frac{2}{\text{bytes\_per\_weight}}$$

| Precision | Bytes per Weight | Operational Intensity ($I$) | Machine Ridge Point ($I^*$) | Operational Regime | Speedup vs FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32** | 4.0 | **0.50 OPs/Byte** | $\approx 80\text{--}150$ | **Memory-Bound** | Baseline (1.0x) |
| **FP16** | 2.0 | **1.00 OPs/Byte** | $\approx 80\text{--}150$ | **Memory-Bound** | **2.0x Speedup** |
| **INT8** | 1.0 | **2.00 OPs/Byte** | $\approx 80\text{--}150$ | **Memory-Bound** | **4.0x Speedup** |
| **INT4** | 0.5 | **4.00 OPs/Byte** | $\approx 80\text{--}150$ | **Memory-Bound** | **8.0x Speedup** |

* **Key Architectural Insight:** In memory-bound token generation, faster ALUs or Tensor Cores do **not** accelerate generation; only **reducing memory bus traffic via quantization** accelerates token throughput!
* **Contrast with Prompt Prefill ($M = 512$):** In context prefill, weights are reused across 512 tokens ($I \approx 400\text{--}1600\text{ OPs/Byte} \gg I^*$), pushing execution into the **compute-bound** regime where Tensor Cores and AVX-512 provide massive speedup.

---

## 8. Hardware Profiling: RAM, VRAM, Bandwidth, CPU and GPU

Pillar 4 implements a deep hardware inspection module that measures real empirical bandwidth and collects machine topology metrics:

1. **System RAM Profiling:**
   - Total, Available, and Free physical RAM via OS APIs.
   - **Empirical Memory Bandwidth Micro-Benchmark:** Uses high-resolution hardware counters (`time.perf_counter_ns`) over multi-MB aligned buffers to measure real sustained host DRAM copy bandwidth (e.g., $18\text{--}25\text{ GB/s}$ on dual-channel DDR4/DDR5).
2. **GPU & VRAM Profiling:**
   - Detects total, free, and allocated VRAM via `torch.cuda.mem_get_info()`.
   - VRAM memory bus width, memory type (GDDR6, GDDR6X, HBM2e, HBM3), and theoretical bandwidth ($300\text{--}3350\text{ GB/s}$).
   - Streaming Multiprocessors (SMs), CUDA cores, and Tensor Core generational tier.
3. **CPU Profiling & Topology:**
   - Physical core count vs logical hyperthreads.
   - CPU Cache Hierarchy modeling (L1 data/instruction caches, L2 private per-core caches, L3 shared cache).
   - Machine Ridge Point calculation:
     $$I^*_{\text{host}} = \frac{P_{\text{peak\_cpu}}}{B_{\text{measured\_ram}}}$$

---

## 9. Dynamic Runtime Configuration

### 9.1 Dynamic Thread Allocation Policy
Amdahl's Law and thread synchronization contention dictate that allocating too many threads for memory-bound tasks degrades performance due to L3 cache thrashing and memory channel congestion:

$$N_{\text{threads\_decode}} = \max\left(1, \; \min\left(N_{\text{physical\_cores}}, \; 4\text{ to }6\right)\right)$$
$$N_{\text{threads\_prefill}} = \max\left(1, \; N_{\text{physical\_cores}} - 1\right)$$
* Always leaves at least 1-2 logical cores reserved for OS scheduling, driver events, and I/O tasks.

---

### 9.2 Key-Value (KV) Cache Memory Scaling & Safe Context Window

Key-Value cache memory grows linearly with context length $T$ and batch size $B$:

$$M_{\text{KV}} = 2 \times B \times T \times L \times N_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_dtype}$$

* **Grouped-Query Attention (GQA):** Utilizing $N_{\text{kv\_heads}} = 8$ instead of 32 (as in Llama-3-8B and Mistral-7B) provides an immediate $4\times$ reduction in KV cache memory!
* **KV-Cache Quantization:**
  - FP16: $256\text{ MB}$ per 2048 tokens ($B=1$).
  - INT8: $128\text{ MB}$ per 2048 tokens ($2\times$ memory reduction).
  - INT4: $64\text{ MB}$ per 2048 tokens ($4\times$ memory reduction).
* **Maximum Safe Context Calculation:** Dynamically solves for $T_{\text{max}}$ to ensure available VRAM is never exceeded, preventing Out-Of-Memory (OOM) crashes.

---

### 9.3 Automated Layer Offloading Planner (Split-Device GPU / CPU Inference)

In heterogeneous consumer devices with limited VRAM (e.g., 4 GB or 6 GB), a full model cannot fit into GPU memory. The automated Layer Offloading Planner partitions transformer layers between GPU and CPU (equivalent to `llama.cpp`'s `-ngl` parameter):

$$M_{\text{layer}} = \frac{M_{\text{weights}}}{L} + M_{\text{KV\_layer}}$$
$$L_{\text{gpu}} = \text{clamp}\left( \left\lfloor \frac{VRAM_{\text{available}} - VRAM_{\text{reserve}}}{M_{\text{layer}}} \right\rfloor, \; 0, \; L \right)$$
$$L_{\text{cpu}} = L - L_{\text{gpu}}$$

```mermaid
graph LR
    Input["Input Tokens"] --> GPU_Layers["Layers 0 .. L_gpu-1 on GPU Tensor Cores"]
    GPU_Layers --> Handoff["PCIe / UMA Activation Bus Transfer"]
    Handoff --> CPU_Layers["Layers L_gpu .. L-1 on CPU Vector SIMD"]
    CPU_Layers --> Output["Logits Output"]
```

* **Live Execution Table for Llama-3-8B (32 Layers, INT4 Weights):**

| VRAM Budget | GPU Layers ($L_{\text{gpu}}$) | CPU Layers ($L_{\text{cpu}}$) | GPU Offload % | GPU VRAM Used | CPU RAM Used | Generation Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2.0 GB** | 13 / 32 | 19 / 32 | **40.6%** | 1456 MB | 2128 MB | 17.1 tok/s |
| **4.0 GB** | 31 / 32 | 1 / 32 | **96.9%** | 3472 MB | 112 MB | 43.5 tok/s |
| **6.0 GB+**| 32 / 32 | 0 / 32 | **100.0%** | 3584 MB | 0 MB | 50.0 tok/s |

---

## 10. Verification and Execution Instructions

### 10.1 Run Complete Unit Test Suite (49 Tests - 100% Pass)
```bash
# Run all 49 tests across the repository
python -m unittest discover tests -v

# Or run individual test modules
python -m unittest tests/test_quantization.py -v
python -m unittest tests/test_hardware_acceleration.py -v
python -m unittest tests/test_runtime_config.py -v
```

### 10.2 Run Runtime Configuration & Roofline Benchmark
```bash
python demo_runtime_configuration.py
```

### 10.3 Run Hardware Acceleration & SIMD Benchmark
```bash
python demo_hardware_acceleration.py
```

### 10.4 Run PTQ Precision & Calibration Benchmark
```bash
python demo_quantization.py
```

### 10.5 Launch the Interactive Web Dashboard & Code Explorer
```bash
# Starts the live dashboard server on port 8080
python dashboard_server.py 8080

# Then open in any browser:
# http://localhost:8080/
```
The dashboard provides a visual control center with live host telemetry, PTQ simulation, Roofline curve canvas, NPU 2D systolic array visualizer, layer offloading planner (-ngl slider), and a code explorer to inspect, copy, or download all repository code.
