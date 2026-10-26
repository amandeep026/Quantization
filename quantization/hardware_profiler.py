"""
Automated Hardware-Aware Quantization Profiler and Runtime Dispatcher.

This is the central orchestration engine for Pillar 4:
1. Profiles host CPU SIMD, GPU Tensor Cores, Apple MPS, and NPU accelerators.
2. Applies the Hardware Quantization Matrix to automatically select the optimal
   precision tier (FP16, BF16, INT8, INT4-Group32) and execution backend.
3. Dispatches QuantizedLinear layers to the optimal hardware engine.
"""

from dataclasses import dataclass
from enum import Enum
import os
import platform
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .gpu_cuda import CUDAHardwareInspector, GPUSpecification, TensorCoreGeneration, TensorCoreTileEngine
from .npu_mps import AppleMPSInspector, AppleMPSPipeline, NPU2DSystolicArray, SystolicExecutionStats
from .precision import PrecisionType
from .ptq import Granularity, QuantizationScheme, QuantizedTensor, dequantize, quantize
from .simd import ARMNeonEngine, AVX512Engine, CPUHardwareInspector, SIMDInstructionSet


class HardwareBackend(str, Enum):
    """Execution backends supported by the Pillar 4 runtime engine."""
    AUTO = "Auto-Detect Optimal Backend"
    CPU_SCALAR = "CPU Scalar Execution"
    CPU_AVX2 = "CPU AVX2 Vector Engine"
    CPU_AVX512 = "CPU AVX-512 VNNI Vector Engine"
    CPU_NEON = "ARM NEON 128-bit SIMD Engine"
    CUDA_CORES = "NVIDIA Standard CUDA Cores"
    CUDA_TENSOR_CORES = "NVIDIA Tensor Cores (MMA Tiles)"
    APPLE_MPS = "Apple Metal Performance Shaders (MPS / UMA)"
    NPU_SYSTOLIC = "NPU 2D Systolic Array (INT8 WS)"


@dataclass
class HostDeviceProfile:
    """Comprehensive hardware profile of the host machine."""
    os_name: str
    cpu_arch: str
    cpu_cores: int
    highest_cpu_simd: SIMDInstructionSet
    supported_simds: Dict[SIMDInstructionSet, bool]
    ram_total_gb: float
    has_cuda: bool
    cuda_device_name: Optional[str]
    cuda_compute_capability: Optional[Tuple[int, int]]
    cuda_vram_gb: Optional[float]
    tensor_core_gen: TensorCoreGeneration
    has_mps: bool
    is_apple_silicon: bool
    has_npu: bool
    # Module 8: Hardware Profiling (RAM, VRAM, Bandwidth, CPU, GPU)
    ram_available_gb: float = 0.0
    ram_bandwidth_gbps: float = 25.0
    gpu_vram_free_gb: Optional[float] = None
    gpu_bandwidth_gbps: Optional[float] = None
    cpu_physical_cores: int = 4
    cpu_logical_threads: int = 4
    cpu_cache_l1_kb: int = 32
    cpu_cache_l2_kb: int = 512
    cpu_cache_l3_mb: int = 16
    ridge_point_flops_per_byte: float = 50.0


@dataclass
class HardwareQuantizationRecommendation:
    """Automated configuration recommended by the Pillar 4 Profiler."""
    recommended_backend: HardwareBackend
    recommended_precision: PrecisionType
    recommended_scheme: QuantizationScheme
    recommended_granularity: Granularity
    recommended_group_size: Optional[int]
    estimated_memory_reduction_percent: float
    estimated_memory_mb_per_billion_params: float
    estimated_tokens_per_sec: float
    rationale: str


class HardwareProfiler:
    """
    Introspects host hardware, formulates optimal quantization decisions,
    and dispatches execution across heterogeneous compute backends.
    """

    @classmethod
    def measure_ram_bandwidth_gbps(cls, buffer_size_mb: int = 16, iterations: int = 3) -> float:
        """
        Measures real empirical RAM copy bandwidth in GB/s using high-resolution hardware counters.
        """
        try:
            n_elements = (buffer_size_mb * 1024 * 1024) // 4
            a = torch.ones(n_elements, dtype=torch.float32)
            b = torch.empty_like(a)
            # Warmup
            b.copy_(a)
            times = []
            for _ in range(iterations):
                t0 = time.perf_counter_ns()
                b.copy_(a)
                t1 = time.perf_counter_ns()
                times.append(t1 - t0)
            best_ns = min(times)
            bytes_transferred = buffer_size_mb * 1024 * 1024 * 2  # 1 read + 1 write
            return float((bytes_transferred / (best_ns / 1e9)) / 1e9)
        except Exception:
            return 25.0  # DDR4/DDR5 baseline fallback

    @classmethod
    def inspect_host_system(cls, measure_bandwidth: bool = True) -> HostDeviceProfile:
        """Collects a complete hardware capability profile of the host system."""
        arch = CPUHardwareInspector.get_cpu_architecture()
        cpu_simds = CPUHardwareInspector.detect_supported_simd()
        highest_simd = CPUHardwareInspector.get_highest_supported_simd()
        cpu_cores = os.cpu_count() or 4
        physical_cores = max(1, cpu_cores // 2) if cpu_cores >= 8 else max(1, cpu_cores - 1)

        # RAM estimation
        ram_gb = 16.0
        ram_avail_gb = 8.0
        try:
            if platform.system() == "Windows":
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                ram_gb = stat.ullTotalPhys / (1024 ** 3)
                ram_avail_gb = stat.ullAvailPhys / (1024 ** 3)
            elif platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "MemTotal" in line:
                            ram_gb = int(line.split()[1]) / (1024 * 1024)
                        elif "MemAvailable" in line:
                            ram_avail_gb = int(line.split()[1]) / (1024 * 1024)
        except Exception:
            pass

        # Empirical RAM bandwidth
        ram_bw = cls.measure_ram_bandwidth_gbps(buffer_size_mb=16) if measure_bandwidth else 25.0

        # CUDA check
        has_cuda = CUDAHardwareInspector.is_cuda_available()
        cuda_name = None
        cuda_cap = None
        cuda_vram = None
        cuda_vram_free = None
        gpu_bw = None
        tc_gen = TensorCoreGeneration.NONE

        if has_cuda:
            spec = CUDAHardwareInspector.get_active_gpu_spec()
            cuda_name = spec.model_name
            cuda_cap = spec.compute_capability
            cuda_vram = spec.memory_capacity_gb
            gpu_bw = spec.memory_bandwidth_gbps
            tc_gen = spec.tensor_core_generation
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                cuda_vram_free = free_b / (1024 ** 3)
            except Exception:
                cuda_vram_free = cuda_vram

        # Apple MPS check
        has_mps = AppleMPSInspector.is_mps_available()
        is_apple = AppleMPSInspector.is_apple_silicon()

        # Machine ridge point estimation: Peak GFLOPS / Memory Bandwidth
        # Host CPU peak compute estimate: physical_cores * 3.0 GHz * FMA lanes * 2
        fma_lanes = 16 if highest_simd in (SIMDInstructionSet.AVX512, SIMDInstructionSet.AVX512_VNNI) else (8 if highest_simd == SIMDInstructionSet.AVX2 else 4)
        cpu_peak_gflops = physical_cores * 3.0 * fma_lanes * 2.0
        ridge_point = cpu_peak_gflops / max(ram_bw, 1.0)

        return HostDeviceProfile(
            os_name=platform.system(),
            cpu_arch=arch,
            cpu_cores=cpu_cores,
            highest_cpu_simd=highest_simd,
            supported_simds=cpu_simds,
            ram_total_gb=ram_gb,
            has_cuda=has_cuda,
            cuda_device_name=cuda_name,
            cuda_compute_capability=cuda_cap,
            cuda_vram_gb=cuda_vram,
            tensor_core_gen=tc_gen,
            has_mps=has_mps,
            is_apple_silicon=is_apple,
            has_npu=(is_apple or "aarch64" in arch),  # Model NPU availability on edge/ARM
            ram_available_gb=ram_avail_gb,
            ram_bandwidth_gbps=ram_bw,
            gpu_vram_free_gb=cuda_vram_free,
            gpu_bandwidth_gbps=gpu_bw,
            cpu_physical_cores=physical_cores,
            cpu_logical_threads=cpu_cores,
            cpu_cache_l1_kb=32 * physical_cores,
            cpu_cache_l2_kb=1024 * physical_cores,
            cpu_cache_l3_mb=16,
            ridge_point_flops_per_byte=ridge_point,
        )

    @classmethod
    def recommend_quantization(
        cls,
        model_parameters: int = 7_000_000_000,  # e.g., 7B Small Language Model
        latency_critical: bool = True,
        profile: Optional[HostDeviceProfile] = None,
    ) -> HardwareQuantizationRecommendation:
        """
        Hardware-Aware Quantization Decision Matrix (Pillar 4 Core Algorithm):
        
        Evaluates parameter memory requirements against available VRAM/RAM
        and available vector/matrix hardware units.
        """
        if profile is None:
            profile = cls.inspect_host_system()

        param_billions = model_parameters / 1e9

        # Memory footprint in GB for different precisions
        mem_fp32 = param_billions * 4.0
        mem_fp16 = param_billions * 2.0
        mem_int8 = param_billions * 1.0
        mem_int4 = param_billions * 0.5

        # Decision 1: High-Performance GPU with Tensor Cores Available
        if profile.has_cuda and profile.cuda_vram_gb is not None:
            vram = profile.cuda_vram_gb
            tc = profile.tensor_core_gen

            if vram >= (mem_int8 + 2.0) and tc != TensorCoreGeneration.NONE:
                return HardwareQuantizationRecommendation(
                    recommended_backend=HardwareBackend.CUDA_TENSOR_CORES,
                    recommended_precision=PrecisionType.INT8,
                    recommended_scheme=QuantizationScheme.SYMMETRIC,
                    recommended_granularity=Granularity.PER_CHANNEL,
                    recommended_group_size=None,
                    estimated_memory_reduction_percent=75.0,
                    estimated_memory_mb_per_billion_params=1000.0,
                    estimated_tokens_per_sec=45.0,
                    rationale=f"NVIDIA Tensor Cores ({tc.value}) with {vram:.1f} GB VRAM detect sufficient capacity for INT8 IMMA matrix tiles.",
                )
            else:
                return HardwareQuantizationRecommendation(
                    recommended_backend=HardwareBackend.CUDA_TENSOR_CORES,
                    recommended_precision=PrecisionType.INT4,
                    recommended_scheme=QuantizationScheme.SYMMETRIC,
                    recommended_granularity=Granularity.GROUP_WISE,
                    recommended_group_size=32,
                    estimated_memory_reduction_percent=87.5,
                    estimated_memory_mb_per_billion_params=500.0,
                    estimated_tokens_per_sec=35.0,
                    rationale=f"VRAM constrained ({vram:.1f} GB for {param_billions:.1f}B params). Selecting INT4 Group-32 PTQ to fit into memory.",
                )

        # Decision 2: Apple Silicon with Metal Performance Shaders (MPS / UMA)
        if profile.is_apple_silicon or profile.has_mps:
            return HardwareQuantizationRecommendation(
                recommended_backend=HardwareBackend.APPLE_MPS,
                recommended_precision=PrecisionType.FP16,
                recommended_scheme=QuantizationScheme.SYMMETRIC,
                recommended_granularity=Granularity.PER_CHANNEL,
                recommended_group_size=None,
                estimated_memory_reduction_percent=50.0,
                estimated_memory_mb_per_billion_params=2000.0,
                estimated_tokens_per_sec=28.0,
                rationale="Apple Silicon Unified Memory Architecture (UMA) detected. Zero-copy FP16 Metal Performance Shaders yield optimal throughput.",
            )

        # Decision 3: CPU with AVX-512 VNNI (Intel DL Boost)
        if profile.highest_cpu_simd in (SIMDInstructionSet.AVX512_VNNI, SIMDInstructionSet.AVX512):
            return HardwareQuantizationRecommendation(
                recommended_backend=HardwareBackend.CPU_AVX512,
                recommended_precision=PrecisionType.INT8,
                recommended_scheme=QuantizationScheme.SYMMETRIC,
                recommended_granularity=Granularity.PER_CHANNEL,
                recommended_group_size=None,
                estimated_memory_reduction_percent=75.0,
                estimated_memory_mb_per_billion_params=1000.0,
                estimated_tokens_per_sec=18.0,
                rationale="AVX-512 VNNI vector instruction set available. Utilizing 64-lane VPDPBUSD dot products for maximum INT8 CPU compute.",
            )

        # Decision 4: ARM CPU with NEON SIMD
        if profile.cpu_arch == "aarch64" or profile.highest_cpu_simd == SIMDInstructionSet.ARM_NEON:
            return HardwareQuantizationRecommendation(
                recommended_backend=HardwareBackend.CPU_NEON,
                recommended_precision=PrecisionType.INT8,
                recommended_scheme=QuantizationScheme.SYMMETRIC,
                recommended_granularity=Granularity.GROUP_WISE,
                recommended_group_size=64,
                estimated_memory_reduction_percent=75.0,
                estimated_memory_mb_per_billion_params=1000.0,
                estimated_tokens_per_sec=12.0,
                rationale="ARM NEON 128-bit vector lanes detected. Dispatching 16-element INT8 SDOT dot products.",
            )

        # Decision 5: General CPU with AVX2 (e.g. Standard PC Host)
        return HardwareQuantizationRecommendation(
            recommended_backend=HardwareBackend.CPU_AVX2,
            recommended_precision=PrecisionType.INT8,
            recommended_scheme=QuantizationScheme.SYMMETRIC,
            recommended_granularity=Granularity.PER_CHANNEL,
            recommended_group_size=None,
            estimated_memory_reduction_percent=75.0,
            estimated_memory_mb_per_billion_params=1000.0,
            estimated_tokens_per_sec=14.0,
            rationale=f"Host CPU has AVX2 with {profile.ram_total_gb:.1f} GB RAM. Standard 8-bit Symmetric PTQ selected for optimal cache efficiency.",
        )


# ============================================================================
# Universal Hardware Dispatcher
# ============================================================================

def dispatch_hardware_forward(
    layer: Any,  # QuantizedLinear instance
    x: torch.Tensor,
    backend: HardwareBackend = HardwareBackend.AUTO,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Executes a QuantizedLinear forward pass routed to the selected hardware backend.
    
    Supports:
    - Auto-dispatch (picks best backend for host)
    - CPU AVX-512 VNNI
    - CPU ARM NEON
    - CUDA Tensor Cores (MMA Tiles)
    - Apple MPS
    - NPU 2D Systolic Array
    """
    if backend == HardwareBackend.AUTO:
        rec = HardwareProfiler.recommend_quantization()
        target_backend = rec.recommended_backend
    else:
        target_backend = backend

    t0 = time.perf_counter()
    w_dequant = layer.get_dequantized_weight(dtype=torch.float32)

    orig_shape = x.shape
    if x.dim() > 2:
        x_in = x.reshape(-1, orig_shape[-1])
    else:
        x_in = x

    # Reference FP32 result for SQNR verification
    y_ref = torch.nn.functional.linear(x_in, w_dequant, layer.bias)

    # 1. CPU AVX-512 Backend
    if target_backend == HardwareBackend.CPU_AVX512:
        # Quantize activations to INT8
        scale_x = x_in.abs().max().item() / 127.0
        x_i8 = torch.clamp(torch.round(x_in / max(scale_x, 1e-8)), -128, 127).to(torch.int8)

        # Get layer INT8 weights
        if (layer.quantized_weight is not None and layer.quantized_weight.bit_width == 8 
                and not getattr(layer.quantized_weight, "is_packed", False)
                and layer.quantized_weight.granularity in (Granularity.PER_CHANNEL, Granularity.PER_TENSOR)):
            w_i8 = layer.quantized_weight.quantized_data.to(torch.int8)
            scale_w = layer.quantized_weight.scale
        else:
            scale_w = w_dequant.abs().max().item() / 127.0
            w_i8 = torch.clamp(torch.round(w_dequant / max(scale_w, 1e-8)), -128, 127).to(torch.int8)

        y_i32 = AVX512Engine.vector_gemm_int8(x_i8, w_i8)
        scale_mult = (scale_x * scale_w.view(1, -1)) if isinstance(scale_w, torch.Tensor) else (scale_x * scale_w)
        y_out = y_i32.to(torch.float32) * scale_mult
        if layer.bias is not None:
            y_out = y_out + layer.bias

    # 2. CPU ARM NEON Backend
    elif target_backend == HardwareBackend.CPU_NEON:
        scale_x = x_in.abs().max().item() / 127.0
        x_i8 = torch.clamp(torch.round(x_in / max(scale_x, 1e-8)), -128, 127).to(torch.int8)
        if (layer.quantized_weight is not None and layer.quantized_weight.bit_width == 8 
                and not getattr(layer.quantized_weight, "is_packed", False)
                and layer.quantized_weight.granularity in (Granularity.PER_CHANNEL, Granularity.PER_TENSOR)):
            w_i8 = layer.quantized_weight.quantized_data.to(torch.int8)
            scale_w = layer.quantized_weight.scale
        else:
            scale_w = w_dequant.abs().max().item() / 127.0
            w_i8 = torch.clamp(torch.round(w_dequant / max(scale_w, 1e-8)), -128, 127).to(torch.int8)

        y_i32 = ARMNeonEngine.vector_gemm_int8(x_i8, w_i8)
        scale_mult = (scale_x * scale_w.view(1, -1)) if isinstance(scale_w, torch.Tensor) else (scale_x * scale_w)
        y_out = y_i32.to(torch.float32) * scale_mult
        if layer.bias is not None:
            y_out = y_out + layer.bias

    # 3. CUDA Tensor Cores Backend
    elif target_backend in (HardwareBackend.CUDA_TENSOR_CORES, HardwareBackend.CUDA_CORES):
        scale_x = x_in.abs().max().item() / 127.0
        x_i8 = torch.clamp(torch.round(x_in / max(scale_x, 1e-8)), -128, 127).to(torch.int8)
        if (layer.quantized_weight is not None and layer.quantized_weight.bit_width == 8 
                and not getattr(layer.quantized_weight, "is_packed", False)
                and layer.quantized_weight.granularity in (Granularity.PER_CHANNEL, Granularity.PER_TENSOR)):
            w_i8 = layer.quantized_weight.quantized_data.to(torch.int8)
            scale_w = layer.quantized_weight.scale
        else:
            scale_w = w_dequant.abs().max().item() / 127.0
            w_i8 = torch.clamp(torch.round(w_dequant / max(scale_w, 1e-8)), -128, 127).to(torch.int8)

        y_out, stats = TensorCoreTileEngine.quantized_matmul_tensor_cores(
            x_i8, w_i8, scale_act=scale_x, scale_weight=scale_w, bias=layer.bias
        )

    # 4. Apple MPS Backend
    elif target_backend == HardwareBackend.APPLE_MPS:
        y_out = AppleMPSPipeline.mps_gemm_forward(x_in, w_dequant, layer.bias, use_fp16=True)

    # 5. NPU 2D Systolic Array Backend
    elif target_backend == HardwareBackend.NPU_SYSTOLIC:
        scale_x = x_in.abs().max().item() / 127.0
        x_i8 = torch.clamp(torch.round(x_in / max(scale_x, 1e-8)), -128, 127).to(torch.int8)
        if (layer.quantized_weight is not None and layer.quantized_weight.bit_width == 8 
                and not getattr(layer.quantized_weight, "is_packed", False)
                and layer.quantized_weight.granularity in (Granularity.PER_CHANNEL, Granularity.PER_TENSOR)):
            w_i8 = layer.quantized_weight.quantized_data.to(torch.int8)
            scale_w = layer.quantized_weight.scale
        else:
            scale_w = w_dequant.abs().max().item() / 127.0
            w_i8 = torch.clamp(torch.round(w_dequant / max(scale_w, 1e-8)), -128, 127).to(torch.int8)

        npu = NPU2DSystolicArray(array_size=16)
        y_i32, stats = npu.execute_matmul_weight_stationary(x_i8, w_i8)
        scale_mult = (scale_x * scale_w.view(1, -1)) if isinstance(scale_w, torch.Tensor) else (scale_x * scale_w)
        y_out = y_i32.to(torch.float32) * scale_mult
        if layer.bias is not None:
            y_out = y_out + layer.bias

    # 6. CPU AVX2 / Scalar Fallback
    else:
        y_out = y_ref

    # Restore multi-dimensional shapes if x was flattened
    if x.dim() > 2:
        y_out = y_out.reshape(*orig_shape[:-1], -1)
        y_ref = y_ref.reshape(*orig_shape[:-1], -1)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Verification: SQNR vs Full-Precision reference
    signal_power = torch.sum(y_ref ** 2)
    noise_power = torch.sum((y_ref - y_out) ** 2)
    sqnr_db = 10.0 * torch.log10(signal_power / torch.clamp(noise_power, min=1e-12)).item()

    telemetry = {
        "backend": target_backend.value,
        "elapsed_ms": elapsed_ms,
        "sqnr_db": sqnr_db,
        "parity_verified": (sqnr_db > 30.0),
        "input_shape": list(x.shape),
        "output_shape": list(y_out.shape),
    }

    return y_out, telemetry
