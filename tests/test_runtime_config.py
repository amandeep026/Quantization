"""
Unit Test Suite for Roofline Model, Hardware Profiling, and Dynamic Runtime Configuration.

Tests:
1. Memory-Bound vs. Compute-Bound Operations (Roofline Model, Arithmetic Intensity, Ridge Point).
2. Hardware Profiling: RAM, VRAM, Empirical Bandwidth Benchmark, CPU Caches, and GPU Metrics.
3. Dynamic Runtime Configuration:
   - Dynamic Thread Allocation Policy (Decoding vs Prefill).
   - KV-Cache Memory Sizing & Maximum Safe Context Calculation.
   - Automated Layer Offloading Planner (Split-Device GPU / CPU allocation).
   - Split-Device Pipeline Execution Simulation with QuantizedLinear.
"""

import unittest
import torch
import torch.nn as nn

from quantization.roofline import (
    OperationalRegime,
    InferencePhase,
    RooflinePoint,
    MachineRooflineSpec,
    RooflineAnalyzer,
)
from quantization.hardware_profiler import (
    HardwareProfiler,
    HostDeviceProfile,
    HardwareBackend,
)
from quantization.runtime_config import (
    OptimizationGoal,
    ModelArchitectureSpec,
    PRESET_MODELS,
    ThreadAllocationPolicy,
    KVCacheManager,
    LayerOffloadPlan,
    LayerOffloadPlanner,
    DynamicRuntimeManager,
)
from quantization.precision import PrecisionType


class TestRooflineModel(unittest.TestCase):
    """Verifies Arithmetic Intensity, Ridge Points, and Roofline classifications."""

    def test_arithmetic_intensity_decoding_vs_prefill(self):
        """
        Decoding (M=1) must have low arithmetic intensity (~1-4 OPs/Byte).
        Prefill (M=512) must have high arithmetic intensity (> 400 OPs/Byte).
        """
        # Decoding (M=1, FP16)
        _, _, intensity_decode = RooflineAnalyzer.calculate_layer_intensity(
            in_features=4096, out_features=4096, batch_size=1, precision=PrecisionType.FP16
        )
        self.assertAlmostEqual(intensity_decode, 1.0, delta=0.05)

        # Decoding (M=1, INT4)
        _, _, intensity_decode_int4 = RooflineAnalyzer.calculate_layer_intensity(
            in_features=4096, out_features=4096, batch_size=1, precision=PrecisionType.INT4
        )
        self.assertAlmostEqual(intensity_decode_int4, 4.0, delta=0.05)

        # Prefill (M=512, FP16)
        _, _, intensity_prefill = RooflineAnalyzer.calculate_layer_intensity(
            in_features=4096, out_features=4096, batch_size=512, precision=PrecisionType.FP16
        )
        self.assertGreater(intensity_prefill, 400.0)

    def test_roofline_regime_classification(self):
        machine = MachineRooflineSpec(
            device_name="Test Hardware",
            peak_compute_gflops=10_000.0,
            peak_bandwidth_gbps=100.0,
            ridge_point=100.0,  # 10,000 / 100 = 100 FLOPs/Byte
        )

        # Point 1: Intensity = 2.0 < 100 -> Must be Memory-Bound
        point_mem = RooflineAnalyzer.evaluate_roofline_point(
            name="Decoding INT8",
            phase=InferencePhase.AUTOREGRESSIVE_DECODING,
            in_features=2048,
            out_features=2048,
            batch_size=1,
            precision=PrecisionType.INT8,
            machine=machine,
        )
        self.assertEqual(point_mem.regime, OperationalRegime.MEMORY_BOUND)
        self.assertLess(point_mem.attainable_performance_gflops, machine.peak_compute_gflops)

        # Point 2: Intensity = 500 > 100 -> Must be Compute-Bound
        point_comp = RooflineAnalyzer.evaluate_roofline_point(
            name="Prefill INT8",
            phase=InferencePhase.PROMPT_PREFILL,
            in_features=2048,
            out_features=2048,
            batch_size=512,
            precision=PrecisionType.INT8,
            machine=machine,
        )
        self.assertEqual(point_comp.regime, OperationalRegime.COMPUTE_BOUND)
        self.assertEqual(point_comp.attainable_performance_gflops, machine.peak_compute_gflops)

    def test_quantization_speedup_in_memory_bound_regime(self):
        """In memory-bound decoding, reducing bitwidth directly accelerates performance."""
        res = RooflineAnalyzer.compare_inference_phases(in_features=4096, out_features=4096)
        fp16_perf = res["Autoregressive (FP16, M=1)"].attainable_performance_gflops
        int8_perf = res["Autoregressive (INT8, M=1)"].attainable_performance_gflops
        int4_perf = res["Autoregressive (INT4, M=1)"].attainable_performance_gflops

        # INT8 should be ~2x FP16, and INT4 should be ~4x FP16
        self.assertAlmostEqual(int8_perf / fp16_perf, 2.0, delta=0.05)
        self.assertAlmostEqual(int4_perf / fp16_perf, 4.0, delta=0.05)


class TestHardwareProfiling(unittest.TestCase):
    """Verifies RAM, VRAM, empirical bandwidth benchmarks, CPU caches, and ridge points."""

    def test_empirical_ram_bandwidth_measurement(self):
        """Verifies empirical RAM bandwidth measurement executes and returns reasonable GB/s."""
        bw = HardwareProfiler.measure_ram_bandwidth_gbps(buffer_size_mb=8, iterations=2)
        self.assertIsInstance(bw, float)
        self.assertGreater(bw, 1.0)  # Modern RAM is at least > 1 GB/s

    def test_enriched_host_device_profile(self):
        """Verifies all Module 8 hardware profiling fields are populated."""
        prof = HardwareProfiler.inspect_host_system(measure_bandwidth=True)
        self.assertGreater(prof.ram_total_gb, 0.0)
        self.assertGreater(prof.ram_available_gb, 0.0)
        self.assertGreater(prof.ram_bandwidth_gbps, 0.0)
        self.assertGreater(prof.cpu_physical_cores, 0)
        self.assertGreater(prof.cpu_logical_threads, 0)
        self.assertGreater(prof.cpu_cache_l1_kb, 0)
        self.assertGreater(prof.ridge_point_flops_per_byte, 0.0)


class TestDynamicRuntimeConfiguration(unittest.TestCase):
    """Verifies thread allocation, KV-cache scaling, layer offloading, and split-device execution."""

    def test_thread_allocation_policy(self):
        threads = ThreadAllocationPolicy.compute_optimal_threads(available_cores=16)
        # Check OS reserve
        self.assertGreaterEqual(threads["os_reserved_cores"], 1)
        # Decoding threads should be clamped to avoid memory bus saturation
        self.assertLessEqual(threads["optimal_decoding_threads"], 8)
        # Prefill threads can use physical cores
        self.assertGreaterEqual(threads["optimal_prefill_threads"], 4)

    def test_kv_cache_memory_scaling(self):
        llama = PRESET_MODELS["Llama-3-8B"]
        # Batch 1, 2048 tokens, FP16
        bytes_2048 = KVCacheManager.calculate_kv_cache_bytes(llama, batch_size=1, context_tokens=2048, precision=PrecisionType.FP16)
        # Batch 1, 4096 tokens, FP16 -> exactly double
        bytes_4096 = KVCacheManager.calculate_kv_cache_bytes(llama, batch_size=1, context_tokens=4096, precision=PrecisionType.FP16)
        self.assertEqual(bytes_4096, bytes_2048 * 2)

        # INT8 KV cache -> exactly half of FP16
        bytes_int8 = KVCacheManager.calculate_kv_cache_bytes(llama, batch_size=1, context_tokens=2048, precision=PrecisionType.INT8)
        self.assertEqual(bytes_int8, bytes_2048 // 2)

    def test_max_safe_context_calculation(self):
        llama = PRESET_MODELS["Llama-3-8B"]
        # With 4 GB VRAM headroom for KV cache
        max_ctx = KVCacheManager.find_max_safe_context(
            model=llama,
            available_vram_bytes=4 * (1024 ** 3),
            batch_size=1,
            precision=PrecisionType.FP16,
        )
        self.assertGreaterEqual(max_ctx, 2048)

    def test_layer_offloading_planner(self):
        llama = PRESET_MODELS["Llama-3-8B"]

        # Case 1: Abundant VRAM (24 GB) -> 100% offload (32/32 layers)
        plan_24gb = LayerOffloadPlanner.plan_offloading(
            model=llama,
            vram_available_gb=24.0,
            ram_available_gb=32.0,
            context_tokens=2048,
            weight_precision=PrecisionType.INT4,
        )
        self.assertEqual(plan_24gb.gpu_layers, 32)
        self.assertEqual(plan_24gb.cpu_layers, 0)
        self.assertTrue(plan_24gb.is_fully_offloaded)

        # Case 2: Constrained VRAM (4 GB) -> Partial offload
        plan_4gb = LayerOffloadPlanner.plan_offloading(
            model=llama,
            vram_available_gb=4.0,
            ram_available_gb=16.0,
            context_tokens=2048,
            weight_precision=PrecisionType.INT4,
        )
        self.assertGreater(plan_4gb.gpu_layers, 0)
        self.assertLess(plan_4gb.gpu_layers, 32)
        self.assertEqual(plan_4gb.gpu_layers + plan_4gb.cpu_layers, 32)
        self.assertFalse(plan_4gb.is_fully_offloaded)

        # Case 3: Zero VRAM -> 0 GPU layers, 32 CPU layers
        plan_0gb = LayerOffloadPlanner.plan_offloading(
            model=llama,
            vram_available_gb=0.0,
            ram_available_gb=16.0,
            context_tokens=2048,
            weight_precision=PrecisionType.INT4,
        )
        self.assertEqual(plan_0gb.gpu_layers, 0)
        self.assertEqual(plan_0gb.cpu_layers, 32)

    def test_split_device_forward_simulation(self):
        """Verifies multi-layer pipeline simulating GPU Tensor Core -> CPU SIMD handoff."""
        torch.manual_seed(101)
        x = torch.randn(2, 64)
        out, stats = DynamicRuntimeManager.simulate_split_device_forward(
            x=x, hidden_dim=64, total_layers=4, gpu_layers=2
        )
        self.assertEqual(out.shape, (2, 64))
        self.assertEqual(stats["gpu_layers"], 2)
        self.assertEqual(stats["cpu_layers"], 2)
        self.assertGreater(stats["total_latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
