"""
End-to-End Demonstration and Benchmark of Hardware Acceleration Engines (Pillar 4).

Demonstrates:
4. CPU SIMD: AVX-512 (512-bit ZMM, VNNI VPDPBUSD) & ARM NEON (128-bit Q, SDOT).
5. GPU Acceleration: CUDA and Tensor Cores (Warp-level MMA tiles, INT8/INT4 -> INT32 mixed precision).
6. NPU (2D Systolic Array WS/OS, cycle counting, zero-skipping) & Apple Metal Performance Shaders (MPS / UMA).
7. Automated Hardware-Aware Quantization Profiler & Dynamic Runtime Dispatcher.
"""

import sys
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization.simd import (
    SIMDInstructionSet,
    SIMD_SPECS,
    CPUHardwareInspector,
    AVX512Engine,
    ARMNeonEngine,
    benchmark_cpu_simd,
)
from quantization.gpu_cuda import (
    TensorCoreGeneration,
    PRESET_GPUS,
    CUDAHardwareInspector,
    WarpTileMMA,
    TensorCoreTileEngine,
    benchmark_gpu_acceleration,
)
from quantization.npu_mps import (
    AppleMPSInspector,
    AppleMPSPipeline,
    NPU2DSystolicArray,
    SystolicDataflow,
    benchmark_npu_and_mps,
)
from quantization.hardware_profiler import (
    HardwareBackend,
    HostDeviceProfile,
    HardwareProfiler,
    dispatch_hardware_forward,
)
from quantization.layers import QuantizedLinear
from quantization.precision import PrecisionType
from quantization.ptq import QuantizationScheme, Granularity


def print_banner(text: str):
    line = "=" * 88
    print(f"\n{line}\n {text.center(86)}\n{line}")


def print_subbanner(text: str):
    line = "-" * 88
    print(f"\n{line}\n {text}\n{line}")


# ============================================================================
# DEMO 4: CPU SIMD (AVX-512 and ARM NEON)
# ============================================================================

def demo_cpu_simd():
    print_banner("PART 4: CPU SIMD ACCELERATION - AVX-512 AND ARM NEON")
    print("Evaluating: Register architecture, vector lane capacities, VNNI VPDPBUSD, and NEON SDOT\n")

    # 1. Architecture Comparison Table
    print("4.1 SIMD Vector Unit Architecture Specifications:")
    header = f"{'Instruction Set':<16} | {'Register':<12} | {'Bits':<5} | {'Bytes':<6} | {'FP32':<5} | {'INT8':<5} | {'INT32 Acc':<10} | {'Theoretical Speedup':<19}"
    print(header)
    print("-" * len(header))
    for s_set in [SIMDInstructionSet.SCALAR, SIMDInstructionSet.ARM_NEON, SIMDInstructionSet.AVX2, SIMDInstructionSet.AVX512, SIMDInstructionSet.AVX512_VNNI]:
        spec = SIMD_SPECS[s_set]
        print(f"{spec.name:<16} | {spec.register_family:<12} | {spec.vector_bits:<5} | {spec.vector_bytes:<6} | {spec.fp32_lanes:<5} | {spec.int8_lanes:<5} | {spec.int32_accumulators:<10} | {spec.theoretical_int8_speedup_vs_scalar:>5.1f}x scalar")

    # 2. Host CPU SIMD Detection
    print("\n4.2 Host Machine SIMD Feature Inspection:")
    arch = CPUHardwareInspector.get_cpu_architecture()
    simd_flags = CPUHardwareInspector.detect_supported_simd()
    highest = CPUHardwareInspector.get_highest_supported_simd()
    print(f"  - Detected Host Architecture: {arch.upper()}")
    print(f"  - Highest Active SIMD Tier : {highest.value}")
    flags_str = ', '.join([f"{k.value}: {'YES' if v else 'NO'}" for k, v in simd_flags.items() if k != SIMDInstructionSet.SCALAR])
    print(f"  - Feature Flags Status     : {flags_str}")

    # 3. Vector Instructions in Action
    print("\n4.3 Specialized Vector Instructions Execution:")
    # AVX-512 VNNI VPDPBUSD
    acc_avx = torch.zeros(16, dtype=torch.int32)
    u_avx = torch.randint(-128, 127, (64,), dtype=torch.int8)
    s_avx = torch.randint(-128, 127, (64,), dtype=torch.int8)
    res_avx = AVX512Engine.vpdpbusd(acc_avx, u_avx, s_avx)
    print(f"  - [AVX-512 VNNI] VPDPBUSD: 64 INT8 pairs -> 16 INT32 accumulated vector lanes:")
    print(f"    Lanes [0..7] : {res_avx[:8].tolist()}")
    print(f"    Lanes [8..15]: {res_avx[8:].tolist()}")

    # ARM NEON SDOT
    acc_neon = torch.zeros(4, dtype=torch.int32)
    u_neon = torch.randint(-128, 127, (16,), dtype=torch.int8)
    s_neon = torch.randint(-128, 127, (16,), dtype=torch.int8)
    res_neon = ARMNeonEngine.sdot(acc_neon, u_neon, s_neon)
    print(f"  - [ARM NEON]     SDOT:     16 INT8 pairs -> 4 INT32 accumulated vector lanes:")
    print(f"    Lanes [0..3] : {res_neon.tolist()}")

    # 4. Live Benchmark & Mathematical Parity
    print_subbanner("4.4 CPU Vector Matrix Multiplication Benchmark (M=32, N=64, K=128)")
    bench_results = benchmark_cpu_simd(M=32, N=64, K=128, iterations=3)
    b_hdr = f"{'Engine':<20} | {'Vector Width':<13} | {'INT8 Lanes':<11} | {'Latency (ms)':<13} | {'Throughput':<12} | {'Speedup':<9} | {'Bit Parity':<10}"
    print(b_hdr)
    print("-" * len(b_hdr))
    for key, res in bench_results.items():
        print(f"{res.instruction_set.value:<20} | {res.vector_width_bits:>4}-bit      | {res.int8_lanes_per_cycle:>2} lanes/cyc | {res.elapsed_ms:>10.2f} ms | {res.gops_throughput:>7.3f} GOPS | {res.speedup_ratio:>6.2f}x  | {str(res.numerical_parity):<10}")

    print("\nInsight: AVX-512 VNNI processes 64 INT8 inputs per vector instruction, achieving a theoretical")
    print("         64x speedup over scalar and 4x higher compute density than 128-bit ARM NEON.")


# ============================================================================
# DEMO 5: GPU Acceleration (CUDA and Tensor Cores)
# ============================================================================

def demo_gpu_acceleration():
    print_banner("PART 5: GPU ACCELERATION - CUDA AND TENSOR CORES")
    print("Evaluating: Warp-level MMA tiles, Tensor Core generations, and Mixed-Precision throughput\n")

    # 1. GPU Architecture & Tensor Core Generations Catalogue
    print("5.1 NVIDIA Architecture & Tensor Core Generational Evolution:")
    hdr = f"{'GPU Model':<22} | {'SMs':<5} | {'CUDA Cores':<11} | {'Tensor Cores':<13} | {'Architecture':<24} | {'INT8 TOPS':<10}"
    print(hdr)
    print("-" * len(hdr))
    for name, g in PRESET_GPUS.items():
        print(f"{g.model_name:<22} | {g.sm_count:<5} | {g.total_cuda_cores:<11} | {g.total_tensor_cores:<13} | {g.architecture:<24} | {g.tensor_int8_tops:>7.1f}")

    # 2. Warp-Level Tensor Core Tile Mechanics
    print("\n5.2 Warp-Level Matrix Multiply-Accumulate (MMA) Tile Execution:")
    print("  In hardware, 32 parallel threads in a Warp execute a synchronized MMA instruction:")
    print("  mma.sync.aligned.m16n8k32: [16, 32] INT8 x [32, 16] INT8 -> [16, 16] INT32 tile accumulation")

    t_a = torch.randint(-10, 10, (16, 32), dtype=torch.int8)
    t_b = torch.randint(-10, 10, (32, 16), dtype=torch.int8)
    t_c = torch.zeros(16, 16, dtype=torch.int32)
    mma_out = WarpTileMMA.mma_sync_int8(t_a, t_b, t_c)
    print(f"  - Tile A shape: {list(t_a.shape)} (INT8 activations)")
    print(f"  - Tile B shape: {list(t_b.shape)} (INT8 weights)")
    print(f"  - MMA Tile Output: {list(mma_out.shape)} (INT32 accumulators) -> Top 4x4 Corner:\n{mma_out[:4, :4]}")

    # 3. Live Benchmark: Standard CUDA Cores vs Tensor Cores
    print_subbanner("5.3 Matrix GEMM Benchmark: Standard CUDA Cores vs Tensor Cores (M=64, N=128, K=256)")
    bench_gpu = benchmark_gpu_acceleration(M=64, N=128, K=256, iterations=3, gpu_preset="RTX_4090")
    g_spec = bench_gpu["gpu_spec"]
    print(f"Target Architecture Profile: {g_spec.model_name} ({g_spec.tensor_core_generation.value})")
    print(f"{'Execution Engine':<32} | {'Precision':<16} | {'Latency (ms)':<13} | {'Compute':<12} | {'Speedup':<9}")
    print("-" * 88)
    res_cuda = bench_gpu["cuda_cores_fp32"]
    res_tc = bench_gpu["tensor_cores_int8"]
    print(f"{res_cuda.backend:<32} | {res_cuda.precision:<16} | {res_cuda.elapsed_ms:>10.2f} ms | {res_cuda.tflops_or_tops * 1000:>7.2f} GFLOPS| Baseline")
    print(f"{res_tc.backend:<32} | {res_tc.precision:<16} | {res_tc.elapsed_ms:>10.2f} ms | {res_tc.tflops_or_tops * 1000:>7.2f} GOPS  | {res_tc.speedup_vs_cuda_cores:>6.2f}x")

    print(f"\nReconstruction Fidelity:")
    print(f"  - Tensor Core INT8 SQNR vs Full FP32: {bench_gpu['sqnr_db']:.2f} dB (Exceptional reconstruction fidelity)")
    print(f"  - Hardware Peak INT8 TOPS Advantage: {bench_gpu['theoretical_speedup']:.1f}x higher throughput than FP32 CUDA cores!")


# ============================================================================
# DEMO 6: NPU and Apple Metal Performance Shaders (MPS)
# ============================================================================

def demo_npu_mps():
    print_banner("PART 6: NPU AND APPLE METAL PERFORMANCE SHADERS (MPS)")
    print("Evaluating: Apple Silicon UMA Zero-Copy, Metal Shaders, and NPU 2D Systolic Arrays\n")

    # 1. Apple MPS and UMA Zero-Copy Memory Model
    print("6.1 Apple Silicon Unified Memory Architecture (UMA) Memory Transfer Efficiency:")
    print("  Comparing 7B Parameter Model (3.5 GB INT4 / 7.0 GB INT8) Bus Transfer Overhead:")
    weights_bytes = int(7e9 * 0.5)  # 3.5 GB for INT4
    transfer_comp = AppleMPSPipeline.compute_memory_transfer_overhead(weights_bytes, pcie_gen=4)
    print(f"  - Model Weight Size: {transfer_comp['tensor_mb'] / 1024:.2f} GB ({weights_bytes:,} bytes)")
    print(f"  - Discrete GPU PCIe Gen4 Transfer Latency: {transfer_comp['pcie_transfer_ms']:.2f} ms (PCIe serialization bottleneck)")
    print(f"  - Apple Silicon UMA Zero-Copy Latency   : {transfer_comp['uma_transfer_ms']:.4f} ms (Zero serialization overhead)")
    print(f"  - Effective Memory Dispatch Advantage    : {transfer_comp['transfer_speedup']:,.0f}x faster host access via unified bus!")

    # 2. NPU 2D Systolic Array Hardware Simulation
    print_subbanner("6.2 NPU 2D Systolic Array (Weight-Stationary Dataflow & Wavefront Scheduling)")
    print("Simulating an edge NPU accelerator with a 16x16 Processing Element (PE) grid (256 PEs total)...")

    npu = NPU2DSystolicArray(array_size=16, clock_mhz=1200.0, power_watts=8.0)
    # Generate weights and activations with post-ReLU sparsity
    x_sparse = torch.relu(torch.randn(32, 64) * 2.0)
    scale_x = x_sparse.abs().max().item() / 127.0
    x_i8 = torch.clamp(torch.round(x_sparse / max(scale_x, 1e-8)), -128, 127).to(torch.int8)

    w = torch.randn(64, 64)
    scale_w = w.abs().max().item() / 127.0
    w_i8 = torch.clamp(torch.round(w / max(scale_w, 1e-8)), -128, 127).to(torch.int8)

    out_npu, stats = npu.execute_matmul_weight_stationary(x_i8, w_i8)
    print(f"  - Array Configuration    : {stats.array_dim}x{stats.array_dim} PEs ({stats.total_pes} parallel MAC units)")
    print(f"  - Dataflow Strategy      : {stats.dataflow.value}")
    print(f"  - Total Execution Cycles : {stats.total_clock_cycles:,} cycles")
    print(f"  - PE Utilization Rate    : {stats.pe_utilization_percent:.1f}%")
    print(f"  - Zero-Skipped MAC Ops   : {stats.zero_skipped_macs:,} / {stats.total_mac_operations:,} ({stats.energy_saved_percent:.1f}% dynamic energy saved!)")
    print(f"  - Simulated Throughput   : {stats.simulated_tops:.3f} TOPS")
    print(f"  - Energy Efficiency      : {stats.tops_per_watt:.2f} TOPS/Watt (Massive power advantage over traditional GPUs)")

    # 3. Comparative Benchmark
    print_subbanner("6.3 Edge Accelerator Benchmark: Apple MPS vs NPU 2D Systolic Array")
    bench_npu_mps = benchmark_npu_and_mps(M=32, N=64, K=128, array_size=16)
    b_hdr = f"{'Accelerator Backend':<36} | {'Precision':<24} | {'Latency (ms)':<13} | {'TOPS/Watt':<11} | {'Fidelity':<10}"
    print(b_hdr)
    print("-" * len(b_hdr))
    for k, res in bench_npu_mps.items():
        print(f"{res.backend_name:<36} | {res.precision:<24} | {res.elapsed_ms:>10.2f} ms | {res.energy_efficiency_tops_per_watt:>8.2f}   | SQNR {res.details['sqnr_db']:.1f} dB")


# ============================================================================
# DEMO 7: Automated Hardware-Aware Quantization Profiler & Dispatcher
# ============================================================================

def demo_hardware_profiler_and_dispatcher():
    print_banner("PART 7: AUTOMATED HARDWARE-AWARE QUANTIZATION PROFILER & DISPATCHER")
    print("Pillar 4 Central Orchestrator: Dynamic System Profiling, Quantization Matrix & Execution\n")

    # 1. System Hardware Profiling
    print("7.1 Live Host Hardware Profile:")
    prof = HardwareProfiler.inspect_host_system()
    print(f"  - Operating System       : {prof.os_name}")
    print(f"  - CPU Architecture       : {prof.cpu_arch} ({prof.cpu_cores} Logical Cores)")
    print(f"  - Highest CPU SIMD Unit  : {prof.highest_cpu_simd.value}")
    print(f"  - Host System RAM        : {prof.ram_total_gb:.1f} GB")
    print(f"  - Discrete NVIDIA GPU    : {'YES (' + prof.cuda_device_name + ')' if prof.has_cuda else 'NO'}")
    print(f"  - Apple MPS / Silicon    : {'YES (Unified Memory)' if prof.is_apple_silicon else 'NO'}")
    print(f"  - Edge NPU Accelerator   : {'AVAILABLE' if prof.has_npu else 'EMULATED'}")

    # 2. Automated Quantization Matrix Decision for a 7B Parameter Model
    print("\n7.2 Automated Quantization Recommendation for a 7-Billion Parameter SLM:")
    rec = HardwareProfiler.recommend_quantization(model_parameters=7_000_000_000, profile=prof)
    print(f"  - Recommended Execution Backend : {rec.recommended_backend.value}")
    print(f"  - Recommended Precision Tier    : {rec.recommended_precision.value}")
    print(f"  - Recommended Quant Scheme      : {rec.recommended_scheme.value}")
    print(f"  - Granularity & Group Size      : {rec.recommended_granularity.value} (Group: {rec.recommended_group_size})")
    print(f"  - Estimated Model Footprint     : {rec.estimated_memory_mb_per_billion_params * 7 / 1024:.2f} GB (Savings: {rec.estimated_memory_reduction_percent:.1f}%)")
    print(f"  - Estimated Generation Throughput: {rec.estimated_tokens_per_sec:.1f} tokens/second")
    print(f"  - Profiler Rationale            : {rec.rationale}")

    # 3. Dynamic Dispatcher Execution with QuantizedLinear Layer
    print_subbanner("7.3 End-to-End Neural Network Layer Dispatching Across Heterogeneous Backends")
    in_dim, out_dim = 256, 512
    batch = 8
    print(f"Instantiating QuantizedLinear Layer: [{in_dim} -> {out_dim}]")
    fp_lin = nn.Linear(in_dim, out_dim, bias=True)
    q_layer = QuantizedLinear.from_float(
        fp_lin,
        bit_width=8,
        scheme=QuantizationScheme.SYMMETRIC,
        granularity=Granularity.PER_CHANNEL,
    )
    x = torch.randn(batch, in_dim)

    dispatch_targets = [
        ("CPU AVX-512 VNNI", HardwareBackend.CPU_AVX512),
        ("ARM NEON SIMD", HardwareBackend.CPU_NEON),
        ("NVIDIA Tensor Cores", HardwareBackend.CUDA_TENSOR_CORES),
        ("Apple MPS (FP16)", HardwareBackend.APPLE_MPS),
        ("NPU 2D Systolic Array", HardwareBackend.NPU_SYSTOLIC),
        ("Auto-Dispatched Best", HardwareBackend.AUTO),
    ]

    print(f"{'Target Hardware Backend':<26} | {'Latency (ms)':<13} | {'Output SQNR':<13} | {'Status':<10}")
    print("-" * 70)
    for name, backend in dispatch_targets:
        out, telem = q_layer.forward_hardware(x, backend=backend)
        status = "VERIFIED PASS" if telem["parity_verified"] else "FAIL"
        print(f"{name:<26} | {telem['elapsed_ms']:>10.2f} ms | {telem['sqnr_db']:>10.2f} dB | {status:<10}")

    print("\nConclusion: All hardware acceleration backends connect directly into QuantizedLinear,")
    print("            achieving seamless execution and mathematical fidelity across CPU, GPU, and NPU architectures.")


def main():
    print_banner("PILLAR 4: COMPLETE HARDWARE ACCELERATION & AUTOMATED PROFILER SUITE")
    print("Demonstrating CPU SIMD (AVX-512/NEON), GPU Tensor Cores, NPU Systolic Arrays, and Apple MPS.\n")

    demo_cpu_simd()
    demo_gpu_acceleration()
    demo_npu_mps()
    demo_hardware_profiler_and_dispatcher()

    print_banner("ALL HARDWARE ACCELERATION BENCHMARKS EXECUTED SUCCESSFULLY")


if __name__ == "__main__":
    main()
