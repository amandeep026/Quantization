"""
Quantization, Hardware Acceleration & Runtime Engine for Pillar 4.

This package provides a comprehensive, mathematically rigorous implementation of:
1. Floating-Point vs. Integer Precisions: FP32, FP16, BF16, INT8, and packed INT4.
2. Post-Training Quantization (PTQ):
   - Symmetric and Asymmetric (Affine) schemes.
   - Scaling granularities: Per-Tensor, Per-Channel (Per-Axis), and Block-wise / Group-wise.
   - Clipping and Calibration: MinMax, Percentile, MSE-optimal search, and KL-Divergence.
3. Quantized Linear Layers and GEMM forward pass simulation.
4. CPU SIMD Acceleration: AVX-512 (512-bit ZMM, VNNI VPDPBUSD) & ARM NEON (128-bit Q, SDOT).
5. GPU Acceleration: CUDA & Tensor Cores (Warp-level MMA tiles, mixed-precision INT8/INT4 -> INT32).
6. NPU (2D Systolic Array WS/OS with zero-skipping) & Apple Metal Performance Shaders (MPS / UMA).
7. Memory-Bound vs. Compute-Bound Analysis: The Roofline Model (Operational Intensity, Ridge Point, Decoding vs Prefill).
8. Hardware Profiling: Empirical RAM bandwidth, VRAM headroom, CPU cache hierarchy, and hardware metrics.
9. Dynamic Runtime Configuration: Dynamic thread allocation, automated layer offloading (-ngl), and context batch sizing.
"""

from .precision import (
    PrecisionType,
    pack_int4,
    unpack_int4,
    convert_precision,
    inspect_precision_profile,
    get_dtype_memory_size,
)
from .ptq import (
    QuantizationScheme,
    Granularity,
    QuantizedTensor,
    calculate_scale_and_zeropoint,
    quantize,
    dequantize,
)
from .calibration import (
    CalibrationMethod,
    Calibrator,
    calibrate_tensor,
)
from .layers import (
    QuantizedLinear,
)
from .metrics import (
    compute_metrics,
    compute_sqnr,
    print_comparison_table,
)
from .simd import (
    SIMDInstructionSet,
    SIMDVectorSpec,
    SIMD_SPECS,
    CPUHardwareInspector,
    AVX512Engine,
    ARMNeonEngine,
    benchmark_cpu_simd,
)
from .gpu_cuda import (
    TensorCoreGeneration,
    GPUSpecification,
    PRESET_GPUS,
    CUDAHardwareInspector,
    WarpTileMMA,
    TensorCoreTileEngine,
    benchmark_gpu_acceleration,
)
from .npu_mps import (
    AppleSiliconGeneration,
    AppleMPSConfig,
    AppleMPSInspector,
    AppleMPSPipeline,
    SystolicDataflow,
    ProcessingElement,
    SystolicExecutionStats,
    NPU2DSystolicArray,
    benchmark_npu_and_mps,
)
from .hardware_profiler import (
    HardwareBackend,
    HostDeviceProfile,
    HardwareQuantizationRecommendation,
    HardwareProfiler,
    dispatch_hardware_forward,
)
from .roofline import (
    OperationalRegime,
    InferencePhase,
    RooflinePoint,
    MachineRooflineSpec,
    RooflineAnalyzer,
)
from .runtime_config import (
    OptimizationGoal,
    ModelArchitectureSpec,
    PRESET_MODELS,
    ThreadAllocationPolicy,
    KVCacheManager,
    LayerOffloadPlan,
    LayerOffloadPlanner,
    DynamicRuntimeManager,
)

__all__ = [
    # Precision & PTQ
    "PrecisionType",
    "pack_int4",
    "unpack_int4",
    "convert_precision",
    "inspect_precision_profile",
    "get_dtype_memory_size",
    "QuantizationScheme",
    "Granularity",
    "QuantizedTensor",
    "calculate_scale_and_zeropoint",
    "quantize",
    "dequantize",
    "CalibrationMethod",
    "Calibrator",
    "calibrate_tensor",
    "QuantizedLinear",
    "compute_metrics",
    "compute_sqnr",
    "print_comparison_table",
    # CPU SIMD
    "SIMDInstructionSet",
    "SIMDVectorSpec",
    "SIMD_SPECS",
    "CPUHardwareInspector",
    "AVX512Engine",
    "ARMNeonEngine",
    "benchmark_cpu_simd",
    # GPU CUDA & Tensor Cores
    "TensorCoreGeneration",
    "GPUSpecification",
    "PRESET_GPUS",
    "CUDAHardwareInspector",
    "WarpTileMMA",
    "TensorCoreTileEngine",
    "benchmark_gpu_acceleration",
    # NPU & Apple MPS
    "AppleSiliconGeneration",
    "AppleMPSConfig",
    "AppleMPSInspector",
    "AppleMPSPipeline",
    "SystolicDataflow",
    "ProcessingElement",
    "SystolicExecutionStats",
    "NPU2DSystolicArray",
    "benchmark_npu_and_mps",
    # Hardware Profiler & Runtime Dispatcher
    "HardwareBackend",
    "HostDeviceProfile",
    "HardwareQuantizationRecommendation",
    "HardwareProfiler",
    "dispatch_hardware_forward",
    # Roofline Model (Memory-Bound vs Compute-Bound)
    "OperationalRegime",
    "InferencePhase",
    "RooflinePoint",
    "MachineRooflineSpec",
    "RooflineAnalyzer",
    # Dynamic Runtime Configuration
    "OptimizationGoal",
    "ModelArchitectureSpec",
    "PRESET_MODELS",
    "ThreadAllocationPolicy",
    "KVCacheManager",
    "LayerOffloadPlan",
    "LayerOffloadPlanner",
    "DynamicRuntimeManager",
]
