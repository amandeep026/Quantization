"""
CPU SIMD Vectorization Module: AVX-512 and ARM NEON.

Implements:
1. Hardware CPU SIMD instruction set detection (AVX, AVX2, AVX-512F, AVX-512VNNI, ARM NEON).
2. AVX-512 Vector Engine:
   - 512-bit ZMM register modeling (16x FP32 lanes, 64x INT8 lanes).
   - VNNI VPDPBUSD dot-product instruction (4-way INT8 dot product into INT32 accumulators).
   - FP32 Fused Multiply-Add (VFMADD231PS) and vector masking with k-opmask registers.
3. ARM NEON Vector Engine:
   - 128-bit Q-register modeling (4x FP32 lanes, 16x INT8 lanes).
   - SDOT/UDOT instruction (4-way 8-bit dot product with INT32 accumulation).
   - 4-lane FP32 FMLA fused multiply-accumulate.
4. Comparative Vector Benchmark & Cycle Simulator:
   - Pure scalar baseline vs ARM NEON (128-bit) vs AVX-512 (512-bit).
   - Arithmetic intensity, theoretical FLOPs/cycle, and speedup ratios.
"""

from dataclasses import dataclass
from enum import Enum
import math
import os
import platform
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class SIMDInstructionSet(str, Enum):
    """CPU SIMD instruction set extensions."""
    SCALAR = "SCALAR"
    SSE4_2 = "SSE4.2"
    AVX = "AVX"
    AVX2 = "AVX2"
    AVX512 = "AVX-512F"
    AVX512_VNNI = "AVX-512_VNNI"
    ARM_NEON = "ARM_NEON"


@dataclass(frozen=True)
class SIMDVectorSpec:
    """Architectural specifications for CPU vector execution units."""
    name: str
    register_family: str
    num_registers: int
    vector_bits: int
    vector_bytes: int
    fp32_lanes: int
    fp16_lanes: int
    int8_lanes: int
    int32_accumulators: int
    fma_ops_per_cycle: int
    theoretical_int8_speedup_vs_scalar: float


SIMD_SPECS: Dict[SIMDInstructionSet, SIMDVectorSpec] = {
    SIMDInstructionSet.SCALAR: SIMDVectorSpec(
        name="Scalar x86/ARM",
        register_family="GPR (RAX/R0)",
        num_registers=16,
        vector_bits=32,
        vector_bytes=4,
        fp32_lanes=1,
        fp16_lanes=1,
        int8_lanes=1,
        int32_accumulators=1,
        fma_ops_per_cycle=1,
        theoretical_int8_speedup_vs_scalar=1.0,
    ),
    SIMDInstructionSet.SSE4_2: SIMDVectorSpec(
        name="Intel/AMD SSE4.2",
        register_family="XMM0-XMM15",
        num_registers=16,
        vector_bits=128,
        vector_bytes=16,
        fp32_lanes=4,
        fp16_lanes=8,
        int8_lanes=16,
        int32_accumulators=4,
        fma_ops_per_cycle=4,
        theoretical_int8_speedup_vs_scalar=4.0,
    ),
    SIMDInstructionSet.AVX: SIMDVectorSpec(
        name="Intel/AMD AVX",
        register_family="YMM0-YMM15",
        num_registers=16,
        vector_bits=256,
        vector_bytes=32,
        fp32_lanes=8,
        fp16_lanes=16,
        int8_lanes=32,
        int32_accumulators=8,
        fma_ops_per_cycle=8,
        theoretical_int8_speedup_vs_scalar=8.0,
    ),
    SIMDInstructionSet.AVX2: SIMDVectorSpec(
        name="Intel/AMD AVX2 + FMA3",
        register_family="YMM0-YMM15",
        num_registers=16,
        vector_bits=256,
        vector_bytes=32,
        fp32_lanes=8,
        fp16_lanes=16,
        int8_lanes=32,
        int32_accumulators=8,
        fma_ops_per_cycle=16,  # 2 FMA ports * 8 FP32
        theoretical_int8_speedup_vs_scalar=16.0,
    ),
    SIMDInstructionSet.ARM_NEON: SIMDVectorSpec(
        name="ARM Advanced SIMD (NEON)",
        register_family="Q0-Q15 (V0-V31)",
        num_registers=32,
        vector_bits=128,
        vector_bytes=16,
        fp32_lanes=4,
        fp16_lanes=8,
        int8_lanes=16,
        int32_accumulators=4,
        fma_ops_per_cycle=8,  # Dual-issue 128-bit pipes
        theoretical_int8_speedup_vs_scalar=16.0,  # via SDOT 4-way MAC
    ),
    SIMDInstructionSet.AVX512: SIMDVectorSpec(
        name="Intel AVX-512 Foundation",
        register_family="ZMM0-ZMM31",
        num_registers=32,
        vector_bits=512,
        vector_bytes=64,
        fp32_lanes=16,
        fp16_lanes=32,
        int8_lanes=64,
        int32_accumulators=16,
        fma_ops_per_cycle=32,  # 2 x 512-bit FMA units * 16
        theoretical_int8_speedup_vs_scalar=32.0,
    ),
    SIMDInstructionSet.AVX512_VNNI: SIMDVectorSpec(
        name="Intel AVX-512 VNNI (DL Boost)",
        register_family="ZMM0-ZMM31",
        num_registers=32,
        vector_bits=512,
        vector_bytes=64,
        fp32_lanes=16,
        fp16_lanes=32,
        int8_lanes=64,
        int32_accumulators=16,
        fma_ops_per_cycle=64,  # VPDPBUSD: 4 x INT8 multiply-adds per lane * 16 lanes
        theoretical_int8_speedup_vs_scalar=64.0,
    ),
}


class CPUHardwareInspector:
    """Detects host CPU architectural features and SIMD extensions."""

    @staticmethod
    def get_cpu_architecture() -> str:
        """Returns normalized CPU architecture string (x86_64 or aarch64)."""
        mach = platform.machine().lower()
        if mach in ("amd64", "x86_64", "x64"):
            return "x86_64"
        elif mach in ("arm64", "aarch64"):
            return "aarch64"
        return mach

    @classmethod
    def detect_supported_simd(cls) -> Dict[SIMDInstructionSet, bool]:
        """
        Inspects hardware and OS feature flags to determine SIMD instruction set support.
        """
        arch = cls.get_cpu_architecture()
        results = {
            SIMDInstructionSet.SCALAR: True,
            SIMDInstructionSet.SSE4_2: False,
            SIMDInstructionSet.AVX: False,
            SIMDInstructionSet.AVX2: False,
            SIMDInstructionSet.AVX512: False,
            SIMDInstructionSet.AVX512_VNNI: False,
            SIMDInstructionSet.ARM_NEON: False,
        }

        if arch == "aarch64":
            results[SIMDInstructionSet.ARM_NEON] = True
            return results

        # x86_64 detection
        if platform.system() == "Windows":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # PF_AVX_INSTRUCTIONS_AVAILABLE = 17
                # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
                # PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41
                results[SIMDInstructionSet.SSE4_2] = True
                results[SIMDInstructionSet.AVX] = bool(kernel32.IsProcessorFeaturePresent(17))
                results[SIMDInstructionSet.AVX2] = bool(kernel32.IsProcessorFeaturePresent(40))
                results[SIMDInstructionSet.AVX512] = bool(kernel32.IsProcessorFeaturePresent(41))
            except Exception:
                results[SIMDInstructionSet.AVX] = True
                results[SIMDInstructionSet.AVX2] = True
        elif platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                    content = f.read().lower()
                    results[SIMDInstructionSet.SSE4_2] = "sse4_2" in content
                    results[SIMDInstructionSet.AVX] = "avx" in content
                    results[SIMDInstructionSet.AVX2] = "avx2" in content
                    results[SIMDInstructionSet.AVX512] = "avx512f" in content
                    results[SIMDInstructionSet.AVX512_VNNI] = "avx512_vnni" in content or "vnni" in content
            except Exception:
                pass
        elif platform.system() == "Darwin":
            results[SIMDInstructionSet.AVX] = True
            results[SIMDInstructionSet.AVX2] = True

        return results

    @classmethod
    def get_highest_supported_simd(cls) -> SIMDInstructionSet:
        """Returns the highest-performing SIMD instruction set available on this host."""
        supported = cls.detect_supported_simd()
        priority_order = [
            SIMDInstructionSet.AVX512_VNNI,
            SIMDInstructionSet.AVX512,
            SIMDInstructionSet.ARM_NEON,
            SIMDInstructionSet.AVX2,
            SIMDInstructionSet.AVX,
            SIMDInstructionSet.SSE4_2,
            SIMDInstructionSet.SCALAR,
        ]
        for simd in priority_order:
            if supported.get(simd, False):
                return simd
        return SIMDInstructionSet.SCALAR


# ============================================================================
# AVX-512 Vector Engine (512-bit ZMM registers, 16x FP32 / 64x INT8)
# ============================================================================

class AVX512Engine:
    """
    High-fidelity functional modeling of Intel AVX-512 execution units.
    
    Architectural characteristics:
    - 32x 512-bit ZMM registers (ZMM0 to ZMM31).
    - 16 FP32 elements per vector instruction (VFMADD231PS).
    - 64 INT8 elements per vector instruction (VPDPBUSD).
    - Opmask registers (k0-k7) for conditional predication and zero-masking.
    """

    REGISTER_BITS = 512
    FP32_LANES = 16
    INT8_LANES = 64
    INT32_ACCUMULATORS = 16

    @classmethod
    def vfmadd231ps(
        cls,
        zmm_acc: torch.Tensor,
        zmm_a: torch.Tensor,
        zmm_b: torch.Tensor,
        k_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        AVX-512 Fused Multiply-Add: ZMM_acc = (ZMM_a * ZMM_b) + ZMM_acc
        Computes 16 parallel FP32 multiply-accumulate operations in a single cycle.
        Supports conditional predication via k_mask (16-bit mask).
        """
        assert zmm_a.numel() == cls.FP32_LANES, f"AVX-512 expects {cls.FP32_LANES} FP32 lanes"
        result = torch.addcmul(zmm_acc.to(torch.float32), zmm_a.to(torch.float32), zmm_b.to(torch.float32))
        if k_mask is not None:
            # Opmask predication: k_mask is a 16-element boolean tensor
            result = torch.where(k_mask, result, zmm_acc)
        return result

    @classmethod
    def vpdpbusd(
        cls,
        zmm_acc_int32: torch.Tensor,
        zmm_u_int8: torch.Tensor,
        zmm_s_int8: torch.Tensor,
    ) -> torch.Tensor:
        """
        AVX-512 VNNI Vector Neural Network Instruction:
        VPDPBUSD: Dot Product of 4x unsigned/signed 8-bit integers with 32-bit accumulation.
        
        Input:
            zmm_u_int8: 64 elements of 8-bit integers (e.g. unsigned/signed activations)
            zmm_s_int8: 64 elements of 8-bit integers (signed weights)
            zmm_acc_int32: 16 elements of 32-bit integer accumulators
            
        Operation:
            Each of the 16 lanes computes:
            acc[i] += u[4i + 0]*s[4i + 0] + u[4i + 1]*s[4i + 1] + u[4i + 2]*s[4i + 2] + u[4i + 3]*s[4i + 3]
        """
        assert zmm_u_int8.numel() == cls.INT8_LANES, f"Expected 64 INT8 values, got {zmm_u_int8.numel()}"
        assert zmm_s_int8.numel() == cls.INT8_LANES, f"Expected 64 INT8 values, got {zmm_s_int8.numel()}"
        assert zmm_acc_int32.numel() == cls.INT32_ACCUMULATORS, f"Expected 16 INT32 accumulators, got {zmm_acc_int32.numel()}"

        # Reshape to 16 lanes x 4-element dot products
        u_reshaped = zmm_u_int8.to(torch.int32).reshape(16, 4)
        s_reshaped = zmm_s_int8.to(torch.int32).reshape(16, 4)
        dot4 = torch.sum(u_reshaped * s_reshaped, dim=1, dtype=torch.int32)

        return (zmm_acc_int32.to(torch.int32) + dot4).to(torch.int32)

    @classmethod
    def vector_gemm_int8(
        cls,
        q_act: torch.Tensor,     # Shape: [M, K] in INT8
        q_weight: torch.Tensor,  # Shape: [N, K] in INT8 (transposed weight)
    ) -> torch.Tensor:
        """
        Computes Matrix Multiplication Q_act @ Q_weight.T using 512-bit AVX-512 VNNI chunks.
        K is processed in blocks of 64 INT8 elements (1 ZMM register width).
        Accumulates into 32-bit integers to prevent overflow.
        """
        M, K = q_act.shape
        N, K_w = q_weight.shape
        assert K == K_w, "Dimension mismatch in GEMM"

        # Pad K to multiple of 64 if needed
        pad_k = (cls.INT8_LANES - (K % cls.INT8_LANES)) % cls.INT8_LANES
        if pad_k > 0:
            act_padded = torch.nn.functional.pad(q_act, (0, pad_k), value=0)
            weight_padded = torch.nn.functional.pad(q_weight, (0, pad_k), value=0)
        else:
            act_padded = q_act
            weight_padded = q_weight

        K_padded = act_padded.shape[1]
        out_int32 = torch.zeros((M, N), dtype=torch.int32)

        # Vectorized chunk loop
        for m in range(M):
            act_row = act_padded[m].to(torch.int32)
            for n in range(N):
                w_row = weight_padded[n].to(torch.int32)
                acc = 0
                for k_chunk in range(0, K_padded, cls.INT8_LANES):
                    a_chunk = act_row[k_chunk : k_chunk + cls.INT8_LANES]
                    w_chunk = w_row[k_chunk : k_chunk + cls.INT8_LANES]
                    # Simulate VPDPBUSD: 16 lanes of 4-element dot products
                    dot_16lanes = (a_chunk.view(16, 4) * w_chunk.view(16, 4)).sum()
                    acc += dot_16lanes.item()
                out_int32[m, n] = acc

        return out_int32


# ============================================================================
# ARM NEON Vector Engine (128-bit Q registers, 4x FP32 / 16x INT8)
# ============================================================================

class ARMNeonEngine:
    """
    High-fidelity functional modeling of ARM Advanced SIMD (NEON) vector units.
    
    Architectural characteristics:
    - 32x 128-bit vector registers (V0-V31 / Q0-Q15).
    - 4 FP32 elements per vector instruction (FMLA).
    - 16 INT8 elements per vector instruction (SDOT/UDOT).
    - 4 INT32 accumulators per 128-bit vector register.
    """

    REGISTER_BITS = 128
    FP32_LANES = 4
    INT8_LANES = 16
    INT32_ACCUMULATORS = 4

    @classmethod
    def fmla(
        cls,
        q_acc: torch.Tensor,
        q_a: torch.Tensor,
        q_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        ARM NEON Floating-Point Multiply-Accumulate:
        FMLA Q_acc, Q_a, Q_b -> Q_acc += Q_a * Q_b (4 parallel FP32 lanes).
        """
        assert q_a.numel() == cls.FP32_LANES, f"NEON expects {cls.FP32_LANES} FP32 lanes"
        return torch.addcmul(q_acc.to(torch.float32), q_a.to(torch.float32), q_b.to(torch.float32))

    @classmethod
    def sdot(
        cls,
        q_acc_int32: torch.Tensor,
        q_u_int8: torch.Tensor,
        q_s_int8: torch.Tensor,
    ) -> torch.Tensor:
        """
        ARM NEON SDOT (Signed Dot Product):
        Multiplies 4 groups of four 8-bit integers and accumulates into four 32-bit vector lanes.
        
        Input:
            q_u_int8: 16 elements of 8-bit signed integers
            q_s_int8: 16 elements of 8-bit signed integers
            q_acc_int32: 4 elements of 32-bit accumulators
            
        Operation:
            acc[0] += u[0]*s[0] + u[1]*s[1] + u[2]*s[2] + u[3]*s[3]
            acc[1] += u[4]*s[4] + u[5]*s[5] + u[6]*s[6] + u[7]*s[7]
            acc[2] += u[8]*s[8] + u[9]*s[9] + u[10]*s[10] + u[11]*s[11]
            acc[3] += u[12]*s[12] + u[13]*s[13] + u[14]*s[14] + u[15]*s[15]
        """
        assert q_u_int8.numel() == cls.INT8_LANES, f"Expected 16 INT8 values, got {q_u_int8.numel()}"
        assert q_s_int8.numel() == cls.INT8_LANES, f"Expected 16 INT8 values, got {q_s_int8.numel()}"
        assert q_acc_int32.numel() == cls.INT32_ACCUMULATORS, f"Expected 4 INT32 accumulators, got {q_acc_int32.numel()}"

        u_reshaped = q_u_int8.to(torch.int32).reshape(4, 4)
        s_reshaped = q_s_int8.to(torch.int32).reshape(4, 4)
        dot4 = torch.sum(u_reshaped * s_reshaped, dim=1, dtype=torch.int32)

        return (q_acc_int32.to(torch.int32) + dot4).to(torch.int32)

    @classmethod
    def vector_gemm_int8(
        cls,
        q_act: torch.Tensor,     # Shape: [M, K] in INT8
        q_weight: torch.Tensor,  # Shape: [N, K] in INT8
    ) -> torch.Tensor:
        """
        Computes Matrix Multiplication Q_act @ Q_weight.T using 128-bit ARM NEON SDOT chunks.
        K is processed in blocks of 16 INT8 elements.
        """
        M, K = q_act.shape
        N, K_w = q_weight.shape
        assert K == K_w, "Dimension mismatch in GEMM"

        pad_k = (cls.INT8_LANES - (K % cls.INT8_LANES)) % cls.INT8_LANES
        if pad_k > 0:
            act_padded = torch.nn.functional.pad(q_act, (0, pad_k), value=0)
            weight_padded = torch.nn.functional.pad(q_weight, (0, pad_k), value=0)
        else:
            act_padded = q_act
            weight_padded = q_weight

        K_padded = act_padded.shape[1]
        out_int32 = torch.zeros((M, N), dtype=torch.int32)

        for m in range(M):
            act_row = act_padded[m].to(torch.int32)
            for n in range(N):
                w_row = weight_padded[n].to(torch.int32)
                acc = 0
                for k_chunk in range(0, K_padded, cls.INT8_LANES):
                    a_chunk = act_row[k_chunk : k_chunk + cls.INT8_LANES]
                    w_chunk = w_row[k_chunk : k_chunk + cls.INT8_LANES]
                    # Simulate SDOT: 4 lanes of 4-element dot products
                    dot_4lanes = (a_chunk.view(4, 4) * w_chunk.view(4, 4)).sum()
                    acc += dot_4lanes.item()
                out_int32[m, n] = acc

        return out_int32


# ============================================================================
# Scalar Loop Reference (Baseline)
# ============================================================================

def scalar_gemm_int8(q_act: torch.Tensor, q_weight: torch.Tensor) -> torch.Tensor:
    """
    Un-vectorized, scalar element-by-element baseline loop.
    Demonstrates baseline execution without SIMD instruction registers.
    """
    M, K = q_act.shape
    N, _ = q_weight.shape
    out = torch.zeros((M, N), dtype=torch.int32)

    a_np = q_act.cpu().numpy().astype(np.int32)
    w_np = q_weight.cpu().numpy().astype(np.int32)

    for m in range(M):
        for n in range(N):
            s = 0
            for k in range(K):
                s += int(a_np[m, k]) * int(w_np[n, k])
            out[m, n] = s
    return out


# ============================================================================
# SIMD Benchmark and Profiling Runner
# ============================================================================

@dataclass
class SIMDBenchmarkResult:
    """Benchmark metrics for a specific SIMD instruction set."""
    instruction_set: SIMDInstructionSet
    vector_width_bits: int
    int8_lanes_per_cycle: int
    elapsed_ms: float
    gops_throughput: float
    speedup_ratio: float
    numerical_parity: bool


def benchmark_cpu_simd(
    M: int = 32,
    N: int = 64,
    K: int = 128,
    iterations: int = 5,
) -> Dict[str, SIMDBenchmarkResult]:
    """
    Runs a rigorous benchmark comparing:
    1. Scalar execution loop (1 INT8 op/cycle)
    2. ARM NEON SDOT (128-bit, 16 INT8 ops/cycle)
    3. Intel AVX-512 VNNI (512-bit, 64 INT8 ops/cycle)
    
    Validates bit-exact parity across all backends.
    """
    torch.manual_seed(42)
    q_act = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    q_weight = torch.randint(-128, 127, (N, K), dtype=torch.int8)

    total_ops = 2 * M * N * K  # 1 multiply + 1 accumulate per element

    # 1. Scalar Baseline
    t0 = time.perf_counter()
    for _ in range(iterations):
        res_scalar = scalar_gemm_int8(q_act, q_weight)
    t_scalar = (time.perf_counter() - t0) / iterations * 1000.0

    # 2. ARM NEON Engine
    t0 = time.perf_counter()
    for _ in range(iterations):
        res_neon = ARMNeonEngine.vector_gemm_int8(q_act, q_weight)
    t_neon = (time.perf_counter() - t0) / iterations * 1000.0

    # 3. AVX-512 VNNI Engine
    t0 = time.perf_counter()
    for _ in range(iterations):
        res_avx512 = AVX512Engine.vector_gemm_int8(q_act, q_weight)
    t_avx512 = (time.perf_counter() - t0) / iterations * 1000.0

    # Verify bit-exact numerical parity
    parity_neon = torch.equal(res_scalar, res_neon)
    parity_avx512 = torch.equal(res_scalar, res_avx512)

    gops_scalar = (total_ops / (t_scalar / 1000.0)) / 1e9
    gops_neon = (total_ops / (t_neon / 1000.0)) / 1e9
    gops_avx512 = (total_ops / (t_avx512 / 1000.0)) / 1e9

    return {
        "scalar": SIMDBenchmarkResult(
            instruction_set=SIMDInstructionSet.SCALAR,
            vector_width_bits=32,
            int8_lanes_per_cycle=1,
            elapsed_ms=t_scalar,
            gops_throughput=gops_scalar,
            speedup_ratio=1.0,
            numerical_parity=True,
        ),
        "arm_neon": SIMDBenchmarkResult(
            instruction_set=SIMDInstructionSet.ARM_NEON,
            vector_width_bits=128,
            int8_lanes_per_cycle=16,
            elapsed_ms=t_neon,
            gops_throughput=gops_neon,
            speedup_ratio=t_scalar / max(t_neon, 1e-6),
            numerical_parity=parity_neon,
        ),
        "avx512_vnni": SIMDBenchmarkResult(
            instruction_set=SIMDInstructionSet.AVX512_VNNI,
            vector_width_bits=512,
            int8_lanes_per_cycle=64,
            elapsed_ms=t_avx512,
            gops_throughput=gops_avx512,
            speedup_ratio=t_scalar / max(t_avx512, 1e-6),
            numerical_parity=parity_avx512,
        ),
    }
