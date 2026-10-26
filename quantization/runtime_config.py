"""
Module 9: Dynamic Runtime Configuration.
Thread Count Allocation, Layer Offloading (Split-Device), and Context Batch Sizing.

Implements:
1. Dynamic Thread Allocation:
   - Balances Amdahl's Law, memory bus saturation, and hyperthreading contention.
   - Separate allocation heuristics for Memory-Bound Decoding vs Compute-Bound Prefill.
2. Automated Layer Offloading Planner:
   - Partitions transformer layers between GPU VRAM and System RAM (like llama.cpp -ngl).
   - Dynamic VRAM budget accounting: weights, KV-cache, activation buffers, and runtime headroom.
3. Context Window & Batch Size Optimization:
   - KV Cache memory scaling across context tokens and batch sizes.
   - Dynamic KV Cache quantization (FP16, INT8, INT4).
   - Maximum safe context calculation preventing Out-Of-Memory (OOM) failures.
4. Split-Device Execution Simulation:
   - Simulates multi-layer pipeline inference across GPU Tensor Cores and CPU SIMD units.
"""

from dataclasses import dataclass
from enum import Enum
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpu_cuda import CUDAHardwareInspector, GPUSpecification, TensorCoreGeneration
from .layers import QuantizedLinear
from .precision import PrecisionType, PRECISION_SPECS
from .ptq import Granularity, QuantizationScheme
from .simd import CPUHardwareInspector, SIMDInstructionSet


class OptimizationGoal(str, Enum):
    """User performance priorities for dynamic runtime configuration."""
    LATENCY_FIRST = "Latency-First (Interactive Single-User Chat)"
    THROUGHPUT_FIRST = "Throughput-First (Batched Multi-User Serving)"
    BALANCED = "Balanced (Optimal Trade-off)"
    MEMORY_CONSTRAINED = "Memory-Constrained (Strict OOM Avoidance)"


@dataclass(frozen=True)
class ModelArchitectureSpec:
    """Architectural parameters of a Transformer Language Model."""
    name: str
    total_layers: int
    hidden_dim: int
    intermediate_dim: int
    num_attention_heads: int
    num_kv_heads: int                  # Supports Grouped-Query Attention (GQA)
    head_dim: int
    vocab_size: int
    total_parameters: int              # Exact parameter count

    @property
    def bytes_per_layer_weights(self) -> Dict[PrecisionType, int]:
        """Calculates weight bytes per transformer layer across precisions."""
        # Standard Decoder Layer:
        # Self-Attention: Q_proj, K_proj, V_proj, O_proj
        # MLP: Gate_proj, Up_proj, Down_proj
        q_params = self.hidden_dim * (self.num_attention_heads * self.head_dim)
        k_params = self.hidden_dim * (self.num_kv_heads * self.head_dim)
        v_params = self.hidden_dim * (self.num_kv_heads * self.head_dim)
        o_params = (self.num_attention_heads * self.head_dim) * self.hidden_dim
        
        # SwiGLU MLP: 3 projections
        mlp_params = 3 * (self.hidden_dim * self.intermediate_dim)
        
        # Norms and biases
        misc_params = 2 * self.hidden_dim
        
        params_per_layer = q_params + k_params + v_params + o_params + mlp_params + misc_params

        result = {}
        for p in [PrecisionType.FP32, PrecisionType.FP16, PrecisionType.BF16, PrecisionType.INT8, PrecisionType.INT4]:
            b_per_elem = PRECISION_SPECS[p].bytes_per_element
            result[p] = int(params_per_layer * b_per_elem)
        return result


# Pre-configured catalogue of popular Small Language Models (SLMs)
PRESET_MODELS: Dict[str, ModelArchitectureSpec] = {
    "Llama-3-8B": ModelArchitectureSpec(
        name="Meta Llama 3 8B",
        total_layers=32,
        hidden_dim=4096,
        intermediate_dim=14336,
        num_attention_heads=32,
        num_kv_heads=8,  # GQA
        head_dim=128,
        vocab_size=128256,
        total_parameters=8_030_000_000,
    ),
    "Mistral-7B": ModelArchitectureSpec(
        name="Mistral 7B v0.3",
        total_layers=32,
        hidden_dim=4096,
        intermediate_dim=14336,
        num_attention_heads=32,
        num_kv_heads=8,  # GQA
        head_dim=128,
        vocab_size=32768,
        total_parameters=7_240_000_000,
    ),
    "Phi-3-mini-3.8B": ModelArchitectureSpec(
        name="Microsoft Phi-3 Mini 3.8B",
        total_layers=32,
        hidden_dim=3072,
        intermediate_dim=8192,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=96,
        vocab_size=32064,
        total_parameters=3_820_000_000,
    ),
    "TinyLlama-1.1B": ModelArchitectureSpec(
        name="TinyLlama 1.1B",
        total_layers=22,
        hidden_dim=2048,
        intermediate_dim=5632,
        num_attention_heads=32,
        num_kv_heads=4,  # GQA
        head_dim=64,
        vocab_size=32000,
        total_parameters=1_100_000_000,
    ),
}


# ============================================================================
# Dynamic Thread Allocation Policy
# ============================================================================

class ThreadAllocationPolicy:
    """
    Computes optimal CPU thread counts based on hardware topology and inference phase.
    
    Principles:
    1. Memory-Bound Decoding: Excess threads cause L3 cache thrashing and memory bus contention.
       Typically scales up to 4-6 threads before memory bandwidth saturates.
    2. Compute-Bound Prefill: Scales linearly with physical core count.
    3. OS Headroom: Always reserves at least 1-2 logical cores for OS, I/O, and driver scheduling.
    """

    @classmethod
    def compute_optimal_threads(
        cls,
        available_cores: Optional[int] = None,
        goal: OptimizationGoal = OptimizationGoal.BALANCED,
    ) -> Dict[str, int]:
        """
        Determines thread allocations for decoding and prefill phases.
        """
        total_cores = available_cores or (os.cpu_count() or 4)
        
        # Estimate physical cores vs hyperthreads (typically physical = total // 2 or total)
        # On x86 hyperthreaded systems, physical is total // 2
        estimated_physical = max(1, total_cores // 2) if total_cores >= 8 else max(1, total_cores - 1)

        # 1. OS Reserve
        reserve = 1 if total_cores <= 8 else 2
        usable_cores = max(1, total_cores - reserve)

        # 2. Decoding Thread Allocation (Memory-Bound)
        # Memory bus saturates around 4-6 threads on standard dual-channel DDR4/DDR5
        if goal == OptimizationGoal.LATENCY_FIRST:
            decode_threads = min(usable_cores, 4)
        elif goal == OptimizationGoal.THROUGHPUT_FIRST:
            decode_threads = min(usable_cores, 8)
        else:
            decode_threads = min(usable_cores, 6)

        # 3. Prefill Thread Allocation (Compute-Bound)
        # Can utilize all physical performance cores
        prefill_threads = max(1, min(usable_cores, estimated_physical))

        return {
            "total_system_cores": total_cores,
            "physical_cores_estimated": estimated_physical,
            "os_reserved_cores": reserve,
            "optimal_decoding_threads": decode_threads,
            "optimal_prefill_threads": prefill_threads,
            "recommended_default_threads": decode_threads,
        }


# ============================================================================
# KV-Cache Sizing & Context Window Optimization
# ============================================================================

class KVCacheManager:
    """
    Calculates exact Key-Value cache memory requirements across context length,
    batch size, and KV-cache quantization precision.
    """

    @classmethod
    def calculate_kv_cache_bytes(
        cls,
        model: ModelArchitectureSpec,
        batch_size: int,
        context_tokens: int,
        precision: PrecisionType = PrecisionType.FP16,
    ) -> int:
        """
        KV Cache formula:
            Memory = 2 (K and V) * batch_size * context_tokens * layers * num_kv_heads * head_dim * bytes_per_elem
        """
        bytes_per_elem = PRECISION_SPECS[precision].bytes_per_element
        # Note: Uses num_kv_heads to account for Grouped-Query Attention (GQA) memory savings!
        kv_elements = 2 * batch_size * context_tokens * model.total_layers * model.num_kv_heads * model.head_dim
        return int(kv_elements * bytes_per_elem)

    @classmethod
    def find_max_safe_context(
        cls,
        model: ModelArchitectureSpec,
        available_vram_bytes: int,
        batch_size: int = 1,
        precision: PrecisionType = PrecisionType.FP16,
    ) -> int:
        """Computes the maximum context length that fits within a specific memory headroom."""
        bytes_per_elem = PRECISION_SPECS[precision].bytes_per_element
        bytes_per_token = 2 * batch_size * model.total_layers * model.num_kv_heads * model.head_dim * bytes_per_elem
        if bytes_per_token <= 0:
            return 2048
        max_tokens = int(available_vram_bytes // bytes_per_token)
        # Round down to power-of-2 or 128 block
        return max(512, (max_tokens // 128) * 128)


# ============================================================================
# Layer Offloading Planner (Split-Device GPU / CPU Pipeline)
# ============================================================================

@dataclass
class LayerOffloadPlan:
    """Offload schedule partitioning layers between GPU and CPU."""
    model_name: str
    weight_precision: PrecisionType
    kv_precision: PrecisionType
    total_layers: int
    gpu_layers: int
    cpu_layers: int
    gpu_offload_percentage: float
    gpu_weight_mb: float
    gpu_kv_mb: float
    gpu_total_mb: float
    cpu_weight_mb: float
    cpu_kv_mb: float
    cpu_total_mb: float
    is_fully_offloaded: bool
    context_tokens: int
    batch_size: int
    estimated_tokens_per_sec: float


class LayerOffloadPlanner:
    """
    Computes optimal GPU layer offloading (equivalent to llama.cpp -ngl parameter).
    
    Balances:
    1. Available GPU VRAM headroom.
    2. Weight memory per layer.
    3. KV Cache memory scaling with context length.
    4. PCIe transfer handoff latency between GPU and CPU.
    """

    @classmethod
    def plan_offloading(
        cls,
        model: ModelArchitectureSpec,
        vram_available_gb: float,
        ram_available_gb: float,
        context_tokens: int = 2048,
        batch_size: int = 1,
        weight_precision: PrecisionType = PrecisionType.INT4,
        kv_precision: PrecisionType = PrecisionType.FP16,
        vram_safety_margin_gb: float = 0.5,  # 500 MB reserve for CUDA runtime context
    ) -> LayerOffloadPlan:
        """Computes the exact number of transformer layers to offload to GPU."""
        vram_budget_bytes = max(0, int((vram_available_gb - vram_safety_margin_gb) * (1024 ** 3)))
        
        # 1. Calculate per-layer weight size
        layer_weights_dict = model.bytes_per_layer_weights
        bytes_per_layer_weight = layer_weights_dict[weight_precision]

        # 2. Calculate per-layer KV cache size
        bytes_per_elem_kv = PRECISION_SPECS[kv_precision].bytes_per_element
        bytes_per_layer_kv = int(2 * batch_size * context_tokens * model.num_kv_heads * model.head_dim * bytes_per_elem_kv)

        bytes_per_layer_total_gpu = bytes_per_layer_weight + bytes_per_layer_kv

        # 3. Compute how many full layers fit into GPU budget
        if bytes_per_layer_total_gpu <= 0:
            gpu_layers = 0
        else:
            gpu_layers = min(model.total_layers, vram_budget_bytes // bytes_per_layer_total_gpu)

        cpu_layers = model.total_layers - gpu_layers
        gpu_pct = (gpu_layers / model.total_layers) * 100.0

        # Memory calculations
        gpu_w_mb = (gpu_layers * bytes_per_layer_weight) / (1024 * 1024)
        gpu_kv_mb = (gpu_layers * bytes_per_layer_kv) / (1024 * 1024)
        gpu_tot_mb = gpu_w_mb + gpu_kv_mb

        cpu_w_mb = (cpu_layers * bytes_per_layer_weight) / (1024 * 1024)
        cpu_kv_mb = (cpu_layers * bytes_per_layer_kv) / (1024 * 1024)
        cpu_tot_mb = cpu_w_mb + cpu_kv_mb

        # Generation speed model:
        # Full GPU: ~40-60 tps
        # Full CPU (INT4): ~8-15 tps
        # Hybrid: Weighted harmonic mean
        gpu_speed_base = 50.0 if weight_precision == PrecisionType.INT4 else 35.0
        cpu_speed_base = 12.0 if weight_precision == PrecisionType.INT4 else 7.0

        if gpu_layers == model.total_layers:
            est_tps = gpu_speed_base
        elif gpu_layers == 0:
            est_tps = cpu_speed_base
        else:
            # Time per token = (gpu_frac / gpu_speed) + (cpu_frac / cpu_speed) + transfer_overhead
            gpu_frac = gpu_layers / model.total_layers
            cpu_frac = cpu_layers / model.total_layers
            # PCIe activation handoff: 4096 hidden * 2 bytes = 8 KB ~ 0.0003 ms
            est_time = (gpu_frac / gpu_speed_base) + (cpu_frac / cpu_speed_base) + 0.001
            est_tps = 1.0 / est_time

        return LayerOffloadPlan(
            model_name=model.name,
            weight_precision=weight_precision,
            kv_precision=kv_precision,
            total_layers=model.total_layers,
            gpu_layers=int(gpu_layers),
            cpu_layers=int(cpu_layers),
            gpu_offload_percentage=gpu_pct,
            gpu_weight_mb=gpu_w_mb,
            gpu_kv_mb=gpu_kv_mb,
            gpu_total_mb=gpu_tot_mb,
            cpu_weight_mb=cpu_w_mb,
            cpu_kv_mb=cpu_kv_mb,
            cpu_total_mb=cpu_tot_mb,
            is_fully_offloaded=(gpu_layers == model.total_layers),
            context_tokens=context_tokens,
            batch_size=batch_size,
            estimated_tokens_per_sec=est_tps,
        )


# ============================================================================
# Dynamic Runtime Manager & Split-Device Inference Simulation
# ============================================================================

class DynamicRuntimeManager:
    """
    Central coordinator configuring execution parameters and simulating
    hybrid GPU/CPU split-device inference.
    """

    @classmethod
    def generate_optimal_config(
        cls,
        model: ModelArchitectureSpec,
        vram_gb: float,
        ram_gb: float,
        goal: OptimizationGoal = OptimizationGoal.BALANCED,
        requested_context: int = 4096,
        requested_batch: int = 1,
    ) -> Dict[str, Any]:
        """
        Creates an end-to-end execution configuration tailored to active hardware.
        """
        # 1. Select optimal precision tier based on goal and memory
        if goal == OptimizationGoal.MEMORY_CONSTRAINED:
            w_prec = PrecisionType.INT4
            kv_prec = PrecisionType.INT8
        elif goal == OptimizationGoal.THROUGHPUT_FIRST and vram_gb >= 16.0:
            w_prec = PrecisionType.INT8
            kv_prec = PrecisionType.FP16
        else:
            w_prec = PrecisionType.INT4
            kv_prec = PrecisionType.FP16

        # 2. Thread allocation
        threads = ThreadAllocationPolicy.compute_optimal_threads(goal=goal)

        # 3. Offload plan
        plan = LayerOffloadPlanner.plan_offloading(
            model=model,
            vram_available_gb=vram_gb,
            ram_available_gb=ram_gb,
            context_tokens=requested_context,
            batch_size=requested_batch,
            weight_precision=w_prec,
            kv_precision=kv_prec,
        )

        # 4. Safe context check
        max_safe_ctx = KVCacheManager.find_max_safe_context(
            model=model,
            available_vram_bytes=int(vram_gb * (1024 ** 3)),
            batch_size=requested_batch,
            precision=kv_prec,
        )

        return {
            "model_name": model.name,
            "goal": goal.value,
            "threads": threads,
            "offload_plan": plan,
            "max_safe_context_length": max_safe_ctx,
            "kv_cache_mb": (plan.gpu_kv_mb + plan.cpu_kv_mb),
            "recommendation_summary": (
                f"Offload {plan.gpu_layers}/{plan.total_layers} layers ({plan.gpu_offload_percentage:.1f}%) to GPU, "
                f"{plan.cpu_layers} layers on CPU with {threads['optimal_decoding_threads']} threads. "
                f"Est Speed: {plan.estimated_tokens_per_sec:.1f} tokens/sec."
            ),
        }

    @classmethod
    def simulate_split_device_forward(
        cls,
        x: torch.Tensor,
        hidden_dim: int = 256,
        total_layers: int = 4,
        gpu_layers: int = 2,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes a functional multi-layer pipeline simulating split-device inference:
        - Layers 0 .. gpu_layers-1: Simulated on GPU Tensor Cores.
        - Handoff across bus.
        - Layers gpu_layers .. total_layers-1: Simulated on CPU SIMD units.
        """
        t0 = time.perf_counter()
        current_act = x

        # Build pipeline of layers
        torch.manual_seed(42)
        layers = [
            QuantizedLinear(in_features=hidden_dim, out_features=hidden_dim, bias=False, bit_width=8)
            for _ in range(total_layers)
        ]
        # Quantize layers
        for layer in layers:
            w = torch.randn(hidden_dim, hidden_dim) * 0.05
            layer.quantized_weight = QuantizedLinear.from_float(
                nn.Linear(hidden_dim, hidden_dim, bias=False), bit_width=8
            ).quantized_weight

        # 1. Execute GPU Stages
        gpu_time_ms = 0.0
        for i in range(gpu_layers):
            t_stage = time.perf_counter()
            # Hardware forward to CUDA Tensor Cores
            current_act, _ = layers[i].forward_hardware(current_act, backend="NVIDIA Tensor Cores (MMA Tiles)")
            gpu_time_ms += (time.perf_counter() - t_stage) * 1000.0

        # 2. Handoff Simulation (GPU -> CPU bus transfer)
        t_handoff = time.perf_counter()
        handoff_bytes = current_act.numel() * 4
        # Simulate transfer time: ~25 GB/s PCIe = 40 ns/KB
        handoff_time_ms = (handoff_bytes / 25e9) * 1000.0 + (time.perf_counter() - t_handoff) * 1000.0

        # 3. Execute CPU Stages
        cpu_time_ms = 0.0
        for i in range(gpu_layers, total_layers):
            t_stage = time.perf_counter()
            # Hardware forward to CPU SIMD
            current_act, _ = layers[i].forward_hardware(current_act, backend="CPU AVX-512 VNNI Vector Engine")
            cpu_time_ms += (time.perf_counter() - t_stage) * 1000.0

        total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        stats = {
            "total_layers": total_layers,
            "gpu_layers": gpu_layers,
            "cpu_layers": total_layers - gpu_layers,
            "gpu_time_ms": gpu_time_ms,
            "handoff_time_ms": handoff_time_ms,
            "cpu_time_ms": cpu_time_ms,
            "total_latency_ms": total_elapsed_ms,
            "output_shape": list(current_act.shape),
        }

        return current_act, stats
