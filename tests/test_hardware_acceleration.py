"""
Comprehensive Unit Test Suite for Hardware Acceleration Modules.

Tests:
1. CPU SIMD: AVX-512 (512-bit ZMM, VNNI VPDPBUSD) & ARM NEON (128-bit Q, SDOT).
2. GPU Acceleration: CUDA & Tensor Cores (Warp MMA tiles, INT8/INT4 -> INT32, mixed precision).
3. NPU (2D Systolic Array WS/OS, cycle counting, zero-skipping) & Apple Metal Performance Shaders (MPS / UMA).
4. Automated Hardware-Aware Profiler and Runtime Dispatcher.
5. QuantizedLinear end-to-end integration across all hardware backends.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization.simd import (
    SIMDInstructionSet,
    SIMD_SPECS,
    CPUHardwareInspector,
    AVX512Engine,
    ARMNeonEngine,
    scalar_gemm_int8,
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


class TestCPUSIMD(unittest.TestCase):
    """Verifies CPU SIMD vector units, instruction emulation, and speedups."""

    def test_simd_specifications(self):
        self.assertEqual(SIMD_SPECS[SIMDInstructionSet.AVX512].vector_bits, 512)
        self.assertEqual(SIMD_SPECS[SIMDInstructionSet.AVX512].fp32_lanes, 16)
        self.assertEqual(SIMD_SPECS[SIMDInstructionSet.AVX512].int8_lanes, 64)

        self.assertEqual(SIMD_SPECS[SIMDInstructionSet.ARM_NEON].vector_bits, 128)
        self.assertEqual(SIMD_SPECS[SIMDInstructionSet.ARM_NEON].fp32_lanes, 4)
        self.assertEqual(SIMD_SPECS[SIMDInstructionSet.ARM_NEON].int8_lanes, 16)

    def test_cpu_inspector(self):
        arch = CPUHardwareInspector.get_cpu_architecture()
        self.assertIn(arch, ("x86_64", "aarch64"))
        simd_dict = CPUHardwareInspector.detect_supported_simd()
        self.assertTrue(simd_dict[SIMDInstructionSet.SCALAR])
        highest = CPUHardwareInspector.get_highest_supported_simd()
        self.assertIsInstance(highest, SIMDInstructionSet)

    def test_avx512_vfmadd231ps(self):
        """Tests 16-lane FP32 FMA and opmask predication."""
        acc = torch.zeros(16, dtype=torch.float32)
        a = torch.ones(16, dtype=torch.float32) * 2.0
        b = torch.ones(16, dtype=torch.float32) * 3.0

        # Normal unmasked FMA: 0 + 2*3 = 6
        res = AVX512Engine.vfmadd231ps(acc, a, b)
        torch.testing.assert_close(res, torch.ones(16) * 6.0)

        # Opmask predicated FMA: only first 8 lanes updated
        mask = torch.tensor([True] * 8 + [False] * 8)
        res_masked = AVX512Engine.vfmadd231ps(acc, a, b, k_mask=mask)
        expected = torch.tensor([6.0] * 8 + [0.0] * 8)
        torch.testing.assert_close(res_masked, expected)

    def test_avx512_vpdpbusd(self):
        """Tests AVX-512 VNNI 64-element INT8 dot product into 16 INT32 lanes."""
        acc = torch.zeros(16, dtype=torch.int32)
        u = torch.ones(64, dtype=torch.int8) * 2
        s = torch.ones(64, dtype=torch.int8) * 3

        # Each lane has 4 elements: 4 * (2 * 3) = 24
        res = AVX512Engine.vpdpbusd(acc, u, s)
        self.assertEqual(res.numel(), 16)
        torch.testing.assert_close(res, torch.full((16,), 24, dtype=torch.int32))

    def test_arm_neon_fmla_and_sdot(self):
        """Tests ARM NEON 4-lane FP32 FMLA and 16-element INT8 SDOT."""
        # 1. FMLA (4 lanes)
        acc_fp = torch.zeros(4, dtype=torch.float32)
        a_fp = torch.ones(4, dtype=torch.float32) * 1.5
        b_fp = torch.ones(4, dtype=torch.float32) * 4.0
        res_fp = ARMNeonEngine.fmla(acc_fp, a_fp, b_fp)
        torch.testing.assert_close(res_fp, torch.ones(4) * 6.0)

        # 2. SDOT (16 elements -> 4 lanes of 4-element dot products)
        acc_i32 = torch.zeros(4, dtype=torch.int32)
        u_i8 = torch.ones(16, dtype=torch.int8) * 3
        s_i8 = torch.ones(16, dtype=torch.int8) * 5
        # Each lane: 4 * (3 * 5) = 60
        res_i32 = ARMNeonEngine.sdot(acc_i32, u_i8, s_i8)
        torch.testing.assert_close(res_i32, torch.full((4,), 60, dtype=torch.int32))

    def test_simd_vs_scalar_mathematical_parity(self):
        """Ensures AVX-512, ARM NEON, and Scalar loop give identical bit-exact results."""
        torch.manual_seed(99)
        q_act = torch.randint(-64, 64, (8, 128), dtype=torch.int8)
        q_w = torch.randint(-64, 64, (16, 128), dtype=torch.int8)

        res_scalar = scalar_gemm_int8(q_act, q_w)
        res_neon = ARMNeonEngine.vector_gemm_int8(q_act, q_w)
        res_avx512 = AVX512Engine.vector_gemm_int8(q_act, q_w)

        self.assertTrue(torch.equal(res_scalar, res_neon))
        self.assertTrue(torch.equal(res_scalar, res_avx512))

    def test_simd_benchmark_runner(self):
        bench = benchmark_cpu_simd(M=8, N=16, K=64, iterations=1)
        self.assertIn("scalar", bench)
        self.assertIn("arm_neon", bench)
        self.assertIn("avx512_vnni", bench)
        self.assertTrue(bench["arm_neon"].numerical_parity)
        self.assertTrue(bench["avx512_vnni"].numerical_parity)


class TestGPUAcceleration(unittest.TestCase):
    """Verifies CUDA specifications, Tensor Core generations, and Warp MMA tile engines."""

    def test_preset_gpus(self):
        rtx4090 = PRESET_GPUS["RTX_4090"]
        self.assertEqual(rtx4090.tensor_core_generation, TensorCoreGeneration.ADA_SM89)
        self.assertGreater(rtx4090.tensor_int8_tops, 600.0)

        a100 = PRESET_GPUS["A100_SXM4_80GB"]
        self.assertEqual(a100.tensor_core_generation, TensorCoreGeneration.AMPERE_SM80_86)
        self.assertEqual(a100.memory_capacity_gb, 80.0)

        h100 = PRESET_GPUS["H100_SXM5"]
        self.assertEqual(h100.tensor_core_generation, TensorCoreGeneration.HOPPER_SM90)

    def test_warp_tile_mma_int8(self):
        """Verifies [16, 32] x [32, 16] -> [16, 16] INT8 MMA tile math."""
        tile_a = torch.ones(16, 32, dtype=torch.int8) * 2
        tile_b = torch.ones(32, 16, dtype=torch.int8) * 3
        tile_c = torch.zeros(16, 16, dtype=torch.int32)

        # Each element: 32 * (2 * 3) = 192
        out_tile = WarpTileMMA.mma_sync_int8(tile_a, tile_b, tile_c)
        self.assertEqual(out_tile.shape, (16, 16))
        self.assertEqual(out_tile.dtype, torch.int32)
        torch.testing.assert_close(out_tile, torch.ones(16, 16, dtype=torch.int32) * 192)

    def test_warp_tile_mma_fp16(self):
        """Verifies [16, 16] x [16, 16] -> [16, 16] FP16 HMMA tile math."""
        tile_a = torch.ones(16, 16, dtype=torch.float16) * 1.5
        tile_b = torch.ones(16, 16, dtype=torch.float16) * 2.0
        out_tile = WarpTileMMA.mma_sync_fp16(tile_a, tile_b)
        # Each element: 16 * (1.5 * 2.0) = 48.0
        torch.testing.assert_close(out_tile, torch.ones(16, 16, dtype=torch.float32) * 48.0)

    def test_tensor_core_engine_matrix_sizes(self):
        """Tests arbitrary matrix dimensions with automatic warp tile padding and scale factors."""
        torch.manual_seed(42)
        M, N, K = 25, 37, 70  # Non-multiples of 16/32 to test padding
        x_i8 = torch.randint(-64, 64, (M, K), dtype=torch.int8)
        w_i8 = torch.randint(-64, 64, (N, K), dtype=torch.int8)

        out, stats = TensorCoreTileEngine.quantized_matmul_tensor_cores(
            x_i8, w_i8, scale_act=0.05, scale_weight=0.02
        )
        self.assertEqual(out.shape, (M, N))
        self.assertGreater(stats["warp_mma_instructions"], 0)

        # Check math parity against direct integer matrix multiply
        ref_int = torch.matmul(x_i8.to(torch.int32), w_i8.t().to(torch.int32))
        ref_scaled = ref_int.to(torch.float32) * (0.05 * 0.02)
        torch.testing.assert_close(out, ref_scaled, atol=1e-4, rtol=1e-4)

    def test_gpu_acceleration_benchmark(self):
        bench = benchmark_gpu_acceleration(M=16, N=32, K=64, iterations=1)
        self.assertIn("cuda_cores_fp32", bench)
        self.assertIn("tensor_cores_int8", bench)
        self.assertTrue(bench["tensor_cores_int8"].mathematical_parity)
        self.assertGreater(bench["sqnr_db"], 30.0)


class TestNPUMPSAcceleration(unittest.TestCase):
    """Verifies Apple MPS (UMA zero-copy model) and NPU 2D Systolic Array simulation."""

    def test_apple_mps_uma_overhead(self):
        # 40 MB tensor
        overhead = AppleMPSPipeline.compute_memory_transfer_overhead(tensor_bytes=40 * 1024 * 1024)
        self.assertGreater(overhead["pcie_transfer_ms"], 1.0)
        self.assertLess(overhead["uma_transfer_ms"], 0.001)
        self.assertGreater(overhead["transfer_speedup"], 1000.0)

    def test_apple_mps_fp16_fidelity(self):
        torch.manual_seed(10)
        x = torch.randn(16, 64)
        w = torch.randn(32, 64)
        y_ref = F.linear(x, w)
        y_mps = AppleMPSPipeline.mps_gemm_forward(x, w, use_fp16=True)

        signal_power = torch.sum(y_ref ** 2)
        noise_power = torch.sum((y_ref - y_mps) ** 2)
        sqnr = 10.0 * torch.log10(signal_power / noise_power).item()
        self.assertGreater(sqnr, 50.0)  # FP16 gives > 50 dB SQNR

    def test_npu_systolic_array_execution(self):
        """Verifies NPU 2D Systolic Array Weight-Stationary execution."""
        torch.manual_seed(20)
        npu = NPU2DSystolicArray(array_size=16)

        x_i8 = torch.randint(-50, 50, (16, 32), dtype=torch.int8)
        w_i8 = torch.randint(-50, 50, (32, 32), dtype=torch.int8)

        out_npu, stats = npu.execute_matmul_weight_stationary(x_i8, w_i8)
        ref = torch.matmul(x_i8.to(torch.int32), w_i8.t().to(torch.int32))

        # Check exact integer parity
        self.assertTrue(torch.equal(out_npu, ref))
        self.assertGreater(stats.total_clock_cycles, 0)
        self.assertGreater(stats.total_mac_operations, 0)

    def test_npu_zero_skipping_sparsity(self):
        """Verifies that sparse tensors trigger zero-skipping power optimization."""
        npu = NPU2DSystolicArray(array_size=16)
        # Create heavily sparse activation matrix (e.g. 75% zeros)
        x_dense = torch.randint(1, 10, (16, 16), dtype=torch.int8)
        x_sparse = x_dense.clone()
        x_sparse[x_sparse > 3] = 0  # Most elements become 0
        w = torch.randint(1, 10, (16, 16), dtype=torch.int8)

        _, stats_dense = npu.execute_matmul_weight_stationary(x_dense, w)
        _, stats_sparse = npu.execute_matmul_weight_stationary(x_sparse, w)

        self.assertGreater(stats_sparse.zero_skipped_macs, stats_dense.zero_skipped_macs)
        self.assertGreater(stats_sparse.energy_saved_percent, stats_dense.energy_saved_percent)

    def test_npu_mps_benchmark_runner(self):
        bench = benchmark_npu_and_mps(M=16, N=32, K=64, array_size=16)
        self.assertIn("apple_mps", bench)
        self.assertIn("npu_systolic", bench)
        self.assertTrue(bench["apple_mps"].numerical_parity)
        self.assertTrue(bench["npu_systolic"].numerical_parity)


class TestHardwareProfilerAndDispatcher(unittest.TestCase):
    """Verifies the Pillar 4 Hardware Profiler and dynamic QuantizedLinear routing."""

    def test_inspect_host_system(self):
        profile = HardwareProfiler.inspect_host_system()
        self.assertIsInstance(profile, HostDeviceProfile)
        self.assertGreater(profile.cpu_cores, 0)
        self.assertGreater(profile.ram_total_gb, 0.0)

    def test_recommendation_matrix_scenarios(self):
        # Scenario 1: GPU with 24GB VRAM
        profile_gpu_large = HostDeviceProfile(
            os_name="Windows",
            cpu_arch="x86_64",
            cpu_cores=16,
            highest_cpu_simd=SIMDInstructionSet.AVX2,
            supported_simds={SIMDInstructionSet.AVX2: True},
            ram_total_gb=32.0,
            has_cuda=True,
            cuda_device_name="RTX 4090",
            cuda_compute_capability=(8, 9),
            cuda_vram_gb=24.0,
            tensor_core_gen=TensorCoreGeneration.ADA_SM89,
            has_mps=False,
            is_apple_silicon=False,
            has_npu=False,
        )
        rec = HardwareProfiler.recommend_quantization(model_parameters=7_000_000_000, profile=profile_gpu_large)
        self.assertEqual(rec.recommended_backend, HardwareBackend.CUDA_TENSOR_CORES)
        self.assertEqual(rec.recommended_precision, PrecisionType.INT8)

        # Scenario 2: Apple Silicon
        profile_apple = HostDeviceProfile(
            os_name="Darwin",
            cpu_arch="aarch64",
            cpu_cores=12,
            highest_cpu_simd=SIMDInstructionSet.ARM_NEON,
            supported_simds={SIMDInstructionSet.ARM_NEON: True},
            ram_total_gb=64.0,
            has_cuda=False,
            cuda_device_name=None,
            cuda_compute_capability=None,
            cuda_vram_gb=None,
            tensor_core_gen=TensorCoreGeneration.NONE,
            has_mps=True,
            is_apple_silicon=True,
            has_npu=True,
        )
        rec_apple = HardwareProfiler.recommend_quantization(model_parameters=7_000_000_000, profile=profile_apple)
        self.assertEqual(rec_apple.recommended_backend, HardwareBackend.APPLE_MPS)
        self.assertEqual(rec_apple.recommended_precision, PrecisionType.FP16)

        # Scenario 3: CPU with AVX-512 VNNI
        profile_avx512 = HostDeviceProfile(
            os_name="Linux",
            cpu_arch="x86_64",
            cpu_cores=32,
            highest_cpu_simd=SIMDInstructionSet.AVX512_VNNI,
            supported_simds={SIMDInstructionSet.AVX512_VNNI: True},
            ram_total_gb=128.0,
            has_cuda=False,
            cuda_device_name=None,
            cuda_compute_capability=None,
            cuda_vram_gb=None,
            tensor_core_gen=TensorCoreGeneration.NONE,
            has_mps=False,
            is_apple_silicon=False,
            has_npu=False,
        )
        rec_avx512 = HardwareProfiler.recommend_quantization(model_parameters=7_000_000_000, profile=profile_avx512)
        self.assertEqual(rec_avx512.recommended_backend, HardwareBackend.CPU_AVX512)
        self.assertEqual(rec_avx512.recommended_precision, PrecisionType.INT8)

    def test_quantized_linear_forward_hardware_routing(self):
        """Verifies QuantizedLinear.forward_hardware routes across all backends with high SQNR."""
        torch.manual_seed(77)
        in_dim, out_dim = 64, 128
        fp_linear = nn.Linear(in_dim, out_dim, bias=True)
        q_layer = QuantizedLinear.from_float(
            fp_linear,
            bit_width=8,
            scheme=QuantizationScheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
        )
        x = torch.randn(8, in_dim)

        backends_to_test = [
            HardwareBackend.CPU_AVX512,
            HardwareBackend.CPU_NEON,
            HardwareBackend.CUDA_TENSOR_CORES,
            HardwareBackend.APPLE_MPS,
            HardwareBackend.NPU_SYSTOLIC,
            HardwareBackend.AUTO,
        ]

        for b in backends_to_test:
            out, telemetry = q_layer.forward_hardware(x, backend=b)
            self.assertEqual(out.shape, (8, out_dim))
            self.assertGreater(telemetry["sqnr_db"], 30.0, f"Backend {b.value} failed SQNR threshold")
            self.assertTrue(telemetry["parity_verified"])

    def test_multi_dimensional_tensor_hardware_routing(self):
        """Verifies 3D Transformer activations [batch, seq_len, in_dim] maintain shape across backends."""
        in_dim, out_dim = 32, 64
        fp_linear = nn.Linear(in_dim, out_dim)
        q_layer = QuantizedLinear.from_float(fp_linear, bit_width=8)
        x_3d = torch.randn(2, 5, in_dim)

        out_avx, _ = q_layer.forward_hardware(x_3d, backend=HardwareBackend.CPU_AVX512)
        self.assertEqual(out_avx.shape, (2, 5, out_dim))

        out_tc, _ = q_layer.forward_hardware(x_3d, backend=HardwareBackend.CUDA_TENSOR_CORES)
        self.assertEqual(out_tc.shape, (2, 5, out_dim))

        out_mps, _ = q_layer.forward_hardware(x_3d, backend=HardwareBackend.APPLE_MPS)
        self.assertEqual(out_mps.shape, (2, 5, out_dim))

    def test_int4_and_groupwise_layer_hardware_routing(self):
        """Verifies packed INT4 and group-wise QuantizedLinear execute through hardware backends."""
        in_dim, out_dim = 64, 128
        fp_linear = nn.Linear(in_dim, out_dim)
        
        # Packed INT4 layer
        q_int4 = QuantizedLinear.from_float(fp_linear, bit_width=4, granularity=Granularity.GROUP_WISE, group_size=32)
        x = torch.randn(4, in_dim)

        out_int4, telem = q_int4.forward_hardware(x, backend=HardwareBackend.AUTO)
        self.assertEqual(out_int4.shape, (4, out_dim))
        self.assertGreater(telem["sqnr_db"], 20.0)


if __name__ == "__main__":
    unittest.main()

