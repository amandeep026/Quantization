"""
Module 7: Memory-Bound vs. Compute-Bound Operations (Roofline Model Analysis).

Implements:
1. Operational (Arithmetic) Intensity calculation for neural network layers and LLM phases:
   - Token-by-token Autoregressive Decoding (M = 1, Memory-Bound).
   - Prompt Processing / Context Prefill (M >= 512, Compute-Bound).
2. The Roofline Model:
   - Machine balance / Ridge point calculation (Peak Compute / Peak Bandwidth).
   - Attainable throughput prediction across precision tiers (FP32, FP16, INT8, INT4).
3. Bottleneck Analysis & Quantization Speedup Projection:
   - Quantifies memory bandwidth speedup vs compute speedup.
   - Computes operational boundary transition points.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

from .precision import PrecisionType, PRECISION_SPECS


class OperationalRegime(str, Enum):
    """Operational bottleneck classification according to the Roofline model."""
    MEMORY_BOUND = "Memory-Bound (Bandwidth-Constrained)"
    COMPUTE_BOUND = "Compute-Bound (ALU/Tensor Core Constrained)"
    BALANCED = "Balanced (Near Ridge Point)"


class InferencePhase(str, Enum):
    """Language model inference phases with fundamentally distinct operational intensities."""
    AUTOREGRESSIVE_DECODING = "Autoregressive Decoding (Token-by-Token, Batch=1)"
    BATCHED_DECODING = "Batched Generation (Concurrent Users, Batch=B)"
    PROMPT_PREFILL = "Prompt Prefill / Context Ingestion (Parallel GEMM, Batch=M)"


@dataclass(frozen=True)
class RooflinePoint:
    """Represents a specific operational workload on the Roofline curve."""
    name: str
    phase: InferencePhase
    precision: PrecisionType
    batch_size: int
    context_tokens: int
    total_operations: int            # FLOPs or Integer OPs
    total_memory_bytes: int          # Bytes transferred over memory bus
    arithmetic_intensity: float      # FLOPs / Byte (or OPs / Byte)
    attainable_performance_gflops: float
    regime: OperationalRegime
    speedup_vs_fp32: float
    bandwidth_utilization_percent: float
    compute_utilization_percent: float


@dataclass(frozen=True)
class MachineRooflineSpec:
    """Hardware roofline limits: peak compute and peak memory bandwidth."""
    device_name: str
    peak_compute_gflops: float        # Peak compute throughput (GFLOPS or GOPS)
    peak_bandwidth_gbps: float        # Peak memory bandwidth (GB/s)
    ridge_point: float                # I* = Peak Compute / Peak Bandwidth (FLOPs/Byte)


class RooflineAnalyzer:
    """
    Mathematical analyzer for the Roofline Model across heterogeneous AI hardware.
    
    Formula:
        Attainable Performance P = min( P_peak,  I * Bandwidth_peak )
        where Arithmetic Intensity I = Operations / Memory_Bytes
    """

    @classmethod
    def calculate_layer_intensity(
        cls,
        in_features: int,
        out_features: int,
        batch_size: int = 1,
        precision: PrecisionType = PrecisionType.FP16,
        activation_precision: PrecisionType = PrecisionType.FP16,
    ) -> Tuple[int, int, float]:
        """
        Calculates exact operations, memory traffic, and arithmetic intensity for a Linear layer.
        
        Args:
            in_features: Input dimension (K)
            out_features: Output dimension (N)
            batch_size: Number of tokens/vectors processed in parallel (M)
            precision: Weight precision
            activation_precision: Activation precision
            
        Returns:
            (total_ops, total_bytes, arithmetic_intensity)
        """
        bytes_per_weight = PRECISION_SPECS[precision].bytes_per_element
        bytes_per_act = PRECISION_SPECS[activation_precision].bytes_per_element

        # Total Multiply-Accumulate operations: 2 * M * N * K (1 mul + 1 add per element)
        total_ops = 2 * batch_size * out_features * in_features

        # Memory traffic:
        # Weights: loaded from memory: N * K * bytes_per_weight
        # Activations (Input): M * K * bytes_per_act
        # Activations (Output): M * N * bytes_per_act
        weight_bytes = int(out_features * in_features * bytes_per_weight)
        input_bytes = int(batch_size * in_features * bytes_per_act)
        output_bytes = int(batch_size * out_features * bytes_per_act)
        total_bytes = weight_bytes + input_bytes + output_bytes

        intensity = total_ops / max(total_bytes, 1)
        return total_ops, total_bytes, intensity

    @classmethod
    def evaluate_roofline_point(
        cls,
        name: str,
        phase: InferencePhase,
        in_features: int,
        out_features: int,
        batch_size: int,
        precision: PrecisionType,
        machine: MachineRooflineSpec,
        activation_precision: Optional[PrecisionType] = None,
    ) -> RooflinePoint:
        """Evaluates a workload against a specific machine's Roofline curve."""
        if activation_precision is None:
            activation_precision = precision

        total_ops, total_bytes, intensity = cls.calculate_layer_intensity(
            in_features=in_features,
            out_features=out_features,
            batch_size=batch_size,
            precision=precision,
            activation_precision=activation_precision,
        )

        # Roofline performance calculation: P = min(P_peak, I * Bandwidth)
        bandwidth_limited_gflops = intensity * machine.peak_bandwidth_gbps
        attainable_gflops = min(machine.peak_compute_gflops, bandwidth_limited_gflops)

        # Operational regime classification
        if intensity < machine.ridge_point * 0.9:
            regime = OperationalRegime.MEMORY_BOUND
        elif intensity > machine.ridge_point * 1.1:
            regime = OperationalRegime.COMPUTE_BOUND
        else:
            regime = OperationalRegime.BALANCED

        # Utilization ratios
        compute_util = (attainable_gflops / max(machine.peak_compute_gflops, 1e-6)) * 100.0
        bw_util = (min(bandwidth_limited_gflops, machine.peak_compute_gflops) / max(bandwidth_limited_gflops, 1e-6)) * 100.0

        # Baseline FP32 comparison
        _, fp32_bytes, fp32_intensity = cls.calculate_layer_intensity(
            in_features, out_features, batch_size, PrecisionType.FP32, PrecisionType.FP32
        )
        fp32_attainable = min(machine.peak_compute_gflops, fp32_intensity * machine.peak_bandwidth_gbps)
        speedup = attainable_gflops / max(fp32_attainable, 1e-6)

        return RooflinePoint(
            name=name,
            phase=phase,
            precision=precision,
            batch_size=batch_size,
            context_tokens=batch_size,
            total_operations=total_ops,
            total_memory_bytes=total_bytes,
            arithmetic_intensity=intensity,
            attainable_performance_gflops=attainable_gflops,
            regime=regime,
            speedup_vs_fp32=speedup,
            bandwidth_utilization_percent=min(bw_util, 100.0),
            compute_utilization_percent=min(compute_util, 100.0),
        )

    @classmethod
    def compare_inference_phases(
        cls,
        in_features: int = 4096,
        out_features: int = 4096,
        machine: Optional[MachineRooflineSpec] = None,
    ) -> Dict[str, RooflinePoint]:
        """
        Demonstrates the fundamental architectural contrast between:
        1. Token-by-token Decoding (M=1) -> Severely Memory-Bound.
        2. Batched Generation (M=16) -> Transitional.
        3. Prompt Prefill (M=1024) -> Heavily Compute-Bound.
        """
        if machine is None:
            # Representative hardware profile: 50 TFLOPS FP16/INT8 compute, 500 GB/s bandwidth
            # Ridge point I* = 50,000 / 500 = 100 FLOPs/Byte
            machine = MachineRooflineSpec(
                device_name="Generic Edge Accelerator",
                peak_compute_gflops=50_000.0,
                peak_bandwidth_gbps=500.0,
                ridge_point=100.0,
            )

        scenarios = [
            ("Autoregressive (FP16, M=1)", InferencePhase.AUTOREGRESSIVE_DECODING, 1, PrecisionType.FP16),
            ("Autoregressive (INT8, M=1)", InferencePhase.AUTOREGRESSIVE_DECODING, 1, PrecisionType.INT8),
            ("Autoregressive (INT4, M=1)", InferencePhase.AUTOREGRESSIVE_DECODING, 1, PrecisionType.INT4),
            ("Batched Decode (INT8, M=16)", InferencePhase.BATCHED_DECODING, 16, PrecisionType.INT8),
            ("Prompt Prefill (FP16, M=512)", InferencePhase.PROMPT_PREFILL, 512, PrecisionType.FP16),
            ("Prompt Prefill (INT8, M=512)", InferencePhase.PROMPT_PREFILL, 512, PrecisionType.INT8),
            ("Prompt Prefill (INT4, M=512)", InferencePhase.PROMPT_PREFILL, 512, PrecisionType.INT4),
        ]

        results = {}
        for name, phase, b_size, prec in scenarios:
            results[name] = cls.evaluate_roofline_point(
                name=name,
                phase=phase,
                in_features=in_features,
                out_features=out_features,
                batch_size=b_size,
                precision=prec,
                machine=machine,
            )
        return results
