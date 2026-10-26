"""
Live Dashboard Server for Pillar 4: Automated Hardware-Aware Quantization & Heterogeneous Execution Engine.

Serves the interactive web dashboard and exposes JSON REST APIs for:
- Live Hardware Telemetry & Micro-Benchmarks
- Interactive Quantization & Calibration Studio
- CPU SIMD, GPU Tensor Cores, and NPU 2D Systolic Array Simulators
- Roofline Model Analyzer & Interactive Curve Generator
- Dynamic Layer Offloading (-ngl) & KV-Cache Sizing
- Full Codebase Browser & Live Test Runner
"""

import http.server
import io
import json
import os
import platform
import socketserver
import sys
import threading
import time
import unittest
from urllib.parse import parse_qs, urlparse

# Ensure local workspace is in python path
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import torch
import numpy as np

from quantization.precision import (
    PrecisionType,
    PRECISION_SPECS,
    float_to_bitstring,
    pack_int4,
    unpack_int4,
    get_dtype_memory_size,
    convert_precision,
)
from quantization.ptq import (
    QuantizationScheme,
    Granularity,
    quantize,
    dequantize,
    calculate_scale_and_zeropoint,
)
from quantization.calibration import (
    CalibrationMethod,
    Calibrator,
    calibrate_tensor,
)
from quantization.layers import QuantizedLinear
from quantization.metrics import (
    compute_metrics,
    compute_sqnr,
)
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
    AppleSiliconGeneration,
    AppleMPSInspector,
    AppleMPSPipeline,
    NPU2DSystolicArray,
    benchmark_npu_and_mps,
)
from quantization.roofline import (
    OperationalRegime,
    InferencePhase,
    MachineRooflineSpec,
    RooflineAnalyzer,
)
from quantization.hardware_profiler import (
    HardwareBackend,
    HostDeviceProfile,
    HardwareProfiler,
    dispatch_hardware_forward,
)
from quantization.runtime_config import (
    OptimizationGoal,
    ModelArchitectureSpec,
    PRESET_MODELS,
    ThreadAllocationPolicy,
    KVCacheManager,
    LayerOffloadPlanner,
    DynamicRuntimeManager,
)

PORT = 8080
STATIC_DIR = os.path.join(WORKSPACE_ROOT, "dashboard")


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the dashboard and handling JSON API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.path = "/index.html"
            return super().do_GET()

        if url.path == "/api/system_profile":
            self._handle_system_profile()
        elif url.path == "/api/code_files":
            self._handle_code_files()
        else:
            return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            payload = {}

        if url.path == "/api/quantize_test":
            self._handle_quantize_test(payload)
        elif url.path == "/api/calibrate":
            self._handle_calibrate(payload)
        elif url.path == "/api/roofline":
            self._handle_roofline(payload)
        elif url.path == "/api/offload_plan":
            self._handle_offload_plan(payload)
        elif url.path == "/api/run_simd_benchmark":
            self._handle_run_simd_benchmark(payload)
        elif url.path == "/api/run_npu_simulation":
            self._handle_run_npu_simulation(payload)
        elif url.path == "/api/run_tests":
            self._handle_run_tests()
        elif url.path == "/api/disassemble_float":
            self._handle_disassemble_float(payload)
        elif url.path == "/api/measure_bandwidth":
            self._handle_measure_bandwidth()
        else:
            self._send_json({"error": f"Endpoint not found: {url.path}"}, status=404)

    def _send_json(self, data, status=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ========================================================================
    # API ENDPOINTS
    # ========================================================================

    def _handle_system_profile(self):
        try:
            profile = HardwareProfiler.inspect_host_system(measure_bandwidth=False)
            rec = HardwareProfiler.recommend_quantization(profile=profile)
            data = {
                "os_name": profile.os_name,
                "cpu_arch": profile.cpu_arch,
                "cpu_cores": profile.cpu_cores,
                "cpu_physical_cores": profile.cpu_physical_cores,
                "cpu_logical_threads": profile.cpu_logical_threads,
                "highest_cpu_simd": profile.highest_cpu_simd.value,
                "supported_simds": {k.value: v for k, v in profile.supported_simds.items()},
                "ram_total_gb": round(profile.ram_total_gb, 2),
                "ram_available_gb": round(profile.ram_available_gb, 2),
                "ram_bandwidth_gbps": round(profile.ram_bandwidth_gbps, 2),
                "cpu_cache_l1_kb": profile.cpu_cache_l1_kb,
                "cpu_cache_l2_kb": profile.cpu_cache_l2_kb,
                "cpu_cache_l3_mb": profile.cpu_cache_l3_mb,
                "ridge_point_flops_per_byte": round(profile.ridge_point_flops_per_byte, 2),
                "has_cuda": profile.has_cuda,
                "cuda_device_name": profile.cuda_device_name,
                "cuda_compute_capability": profile.cuda_compute_capability,
                "cuda_vram_gb": round(profile.cuda_vram_gb, 2) if profile.cuda_vram_gb else None,
                "cuda_vram_free_gb": round(profile.gpu_vram_free_gb, 2) if profile.gpu_vram_free_gb else None,
                "tensor_core_gen": profile.tensor_core_gen.value,
                "has_mps": profile.has_mps,
                "is_apple_silicon": profile.is_apple_silicon,
                "has_npu": profile.has_npu,
                "recommendation": {
                    "backend": rec.recommended_backend.value,
                    "precision": rec.recommended_precision.value,
                    "scheme": rec.recommended_scheme.value,
                    "granularity": rec.recommended_granularity.value,
                    "group_size": rec.recommended_group_size,
                    "savings_percent": rec.estimated_memory_reduction_percent,
                    "memory_mb_per_billion": rec.estimated_memory_mb_per_billion_params,
                    "estimated_tps": rec.estimated_tokens_per_sec,
                    "rationale": rec.rationale,
                },
            }
            self._send_json({"success": True, "data": data})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _handle_measure_bandwidth(self):
        try:
            bw = HardwareProfiler.measure_ram_bandwidth_gbps(buffer_size_mb=16, iterations=3)
            self._send_json({"success": True, "bandwidth_gbps": round(bw, 2)})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _handle_disassemble_float(self, payload):
        try:
            val = float(payload.get("value", -12.375))
            bits_map = {}
            for p in PrecisionType:
                bits_map[p.value] = float_to_bitstring(val, p)
            self._send_json({"success": True, "value": val, "formats": bits_map})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _handle_quantize_test(self, payload):
        try:
            dist = payload.get("distribution", "gaussian")
            bit_width = int(payload.get("bit_width", 8))
            scheme_str = payload.get("scheme", "symmetric")
            gran_str = payload.get("granularity", "per_tensor")
            group_size = int(payload.get("group_size", 32))
            num_elements = int(payload.get("num_elements", 1024))

            scheme = QuantizationScheme.SYMMETRIC if scheme_str == "symmetric" else QuantizationScheme.ASYMMETRIC
            if gran_str == "per_channel":
                granularity = Granularity.PER_CHANNEL
            elif gran_str == "group_wise":
                granularity = Granularity.GROUP_WISE
            else:
                granularity = Granularity.PER_TENSOR

            torch.manual_seed(42)
            if dist == "skewed":
                # Post-ReLU skewed distribution [0.0, 10.0]
                t = torch.relu(torch.randn(32, num_elements // 32)) * 4.0
            elif dist == "outliers":
                t = torch.randn(32, num_elements // 32)
                t[0, 0] = 35.0  # Outlier spike
                t[1, 1] = -30.0
            else:
                # Normal Gaussian
                t = torch.randn(32, num_elements // 32)

            orig_bytes = t.numel() * 4
            q_tensor = quantize(
                t,
                bit_width=bit_width,
                scheme=scheme,
                granularity=granularity,
                group_size=group_size,
                pack_int4_data=(bit_width == 4),
            )
            dequant = dequantize(q_tensor)
            quant_bytes = q_tensor.memory_bytes()

            report = compute_metrics(
                original=t,
                reconstructed=dequant,
                original_bytes=orig_bytes,
                quantized_bytes=quant_bytes,
                name=f"{scheme.value.capitalize()} {bit_width}-bit ({granularity.value})",
                bit_width=bit_width,
            )

            # Sample slice for visual comparison
            flat_orig = t.flatten()[:20].tolist()
            flat_rec = dequant.flatten()[:20].tolist()
            if q_tensor.is_packed:
                flat_q = unpack_int4(q_tensor.quantized_data, original_numel=q_tensor.original_numel).flatten()[:20].tolist()
            else:
                flat_q = q_tensor.quantized_data.flatten()[:20].tolist()

            scale_val = q_tensor.scale.flatten()[:5].tolist()
            zp_val = q_tensor.zero_point.flatten()[:5].tolist() if q_tensor.zero_point is not None else [0]

            self._send_json({
                "success": True,
                "metrics": {
                    "sqnr_db": round(report.sqnr_db, 2),
                    "mse": report.mse,
                    "mae": report.mae,
                    "max_error": report.max_error,
                    "cosine_similarity": round(report.cosine_similarity, 6),
                    "original_bytes": report.original_bytes,
                    "quantized_bytes": report.quantized_bytes,
                    "compression_ratio": round(report.compression_ratio, 2),
                    "savings_percent": round((1.0 - (report.quantized_bytes / report.original_bytes)) * 100.0, 1),
                },
                "samples": {
                    "original": [round(x, 4) for x in flat_orig],
                    "quantized_int": flat_q,
                    "reconstructed": [round(x, 4) for x in flat_rec],
                    "scale": [round(s, 6) for s in scale_val],
                    "zero_point": zp_val,
                },
            })
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _handle_calibrate(self, payload):
        try:
            percentile = float(payload.get("percentile", 99.9))
            bit_width = int(payload.get("bit_width", 8))

            torch.manual_seed(123)
            # Create synthetic distribution with outlier spikes
            clean = torch.randn(4000)
            dirty = clean.clone()
            dirty[0:10] = torch.tensor([45.0, -42.0, 38.0, -35.0, 48.0, -40.0, 33.0, -31.0, 50.0, -47.0])

            methods = [
                CalibrationMethod.MINMAX,
                CalibrationMethod.PERCENTILE,
                CalibrationMethod.MSE,
                CalibrationMethod.KL_DIVERGENCE,
            ]
            results = {}

            for m in methods:
                c_min, c_max = calibrate_tensor(
                    dirty, method=m, bit_width=bit_width, percentile=percentile, scheme=QuantizationScheme.SYMMETRIC
                )
                q_t = quantize(
                    dirty,
                    bit_width=bit_width,
                    scheme=QuantizationScheme.SYMMETRIC,
                    granularity=Granularity.PER_TENSOR,
                    custom_min=c_min,
                    custom_max=c_max,
                )
                rec = dequantize(q_t)
                sqnr = compute_sqnr(dirty, rec)
                mse = float(torch.mean((dirty - rec) ** 2).item())

                results[m.value] = {
                    "min_val": round(float(c_min.item()), 3),
                    "max_val": round(float(c_max.item()), 3),
                    "threshold_range": round(float(c_max.item() - c_min.item()), 3),
                    "sqnr_db": round(sqnr, 2),
                    "mse": mse,
                }

            # Histogram of raw data for charting
            hist, bin_edges = np.histogram(dirty.numpy(), bins=50)
            hist_data = {
                "counts": hist.tolist(),
                "bin_edges": [round(float(b), 2) for b in bin_edges.tolist()],
            }

            self._send_json({
                "success": True,
                "calibrations": results,
                "histogram": hist_data,
            })
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _handle_roofline(self, payload):
        try:
            device_choice = payload.get("device", "RTX_4090")
            if device_choice in PRESET_GPUS:
                gpu_spec = PRESET_GPUS[device_choice]
                spec = MachineRooflineSpec(
                    device_name=gpu_spec.model_name,
                    peak_compute_gflops=gpu_spec.fp32_tflops * 1000.0,
                    peak_bandwidth_gbps=gpu_spec.memory_bandwidth_gbps,
                    ridge_point=(gpu_spec.fp32_tflops * 1000.0) / gpu_spec.memory_bandwidth_gbps,
                )
            else:
                peak_compute = float(payload.get("peak_gflops", 50000.0))
                peak_bw = float(payload.get("peak_bandwidth_gbps", 500.0))
                spec = MachineRooflineSpec(
                    device_name="Custom Machine",
                    peak_compute_gflops=peak_compute,
                    peak_bandwidth_gbps=peak_bw,
                    ridge_point=peak_compute / max(peak_bw, 1.0),
                )

            in_feat = int(payload.get("in_features", 4096))
            out_feat = int(payload.get("out_features", 4096))

            points = RooflineAnalyzer.compare_inference_phases(in_features=in_feat, out_features=out_feat, machine=spec)
            points_data = {}
            for name, pt in points.items():
                points_data[name] = {
                    "name": pt.name,
                    "phase": pt.phase.value,
                    "precision": pt.precision.value,
                    "batch_size": pt.batch_size,
                    "intensity": round(pt.arithmetic_intensity, 2),
                    "attainable_gflops": round(pt.attainable_performance_gflops, 1),
                    "attainable_tflops": round(pt.attainable_performance_gflops / 1000.0, 2),
                    "regime": pt.regime.value,
                    "speedup": round(pt.speedup_vs_fp32, 2),
                    "bandwidth_util_pct": round(pt.bandwidth_utilization_percent, 1),
                    "compute_util_pct": round(pt.compute_utilization_percent, 1),
                }

            self._send_json({
                "success": True,
                "machine": {
                    "name": spec.device_name,
                    "peak_compute_gflops": spec.peak_compute_gflops,
                    "peak_compute_tflops": round(spec.peak_compute_gflops / 1000.0, 1),
                    "peak_bandwidth_gbps": spec.peak_bandwidth_gbps,
                    "ridge_point": round(spec.ridge_point, 1),
                },
                "points": points_data,
            })
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _handle_offload_plan(self, payload):
        try:
            model_name = payload.get("model", "Llama-3-8B")
            model = PRESET_MODELS.get(model_name, PRESET_MODELS["Llama-3-8B"])
            vram_gb = float(payload.get("vram_gb", 6.0))
            ram_gb = float(payload.get("ram_gb", 16.0))
            context_tokens = int(payload.get("context_tokens", 2048))
            batch_size = int(payload.get("batch_size", 1))

            w_prec = PrecisionType(payload.get("weight_precision", "INT4"))
            kv_prec = PrecisionType(payload.get("kv_precision", "FP16"))

            plan = LayerOffloadPlanner.plan_offloading(
                model=model,
                vram_available_gb=vram_gb,
                ram_available_gb=ram_gb,
                context_tokens=context_tokens,
                batch_size=batch_size,
                weight_precision=w_prec,
                kv_precision=kv_prec,
            )

            # Threads
            threads = ThreadAllocationPolicy.compute_optimal_threads()

            # Max safe context
            max_safe_ctx = KVCacheManager.find_max_safe_context(
                model=model,
                available_vram_bytes=int(vram_gb * (1024 ** 3)),
                batch_size=batch_size,
                precision=kv_prec,
            )

            self._send_json({
                "success": True,
                "plan": {
                    "model_name": plan.model_name,
                    "total_layers": plan.total_layers,
                    "gpu_layers": plan.gpu_layers,
                    "cpu_layers": plan.cpu_layers,
                    "gpu_offload_percentage": round(plan.gpu_offload_percentage, 1),
                    "gpu_weight_mb": round(plan.gpu_weight_mb, 1),
                    "gpu_kv_mb": round(plan.gpu_kv_mb, 1),
                    "gpu_total_mb": round(plan.gpu_total_mb, 1),
                    "cpu_weight_mb": round(plan.cpu_weight_mb, 1),
                    "cpu_kv_mb": round(plan.cpu_kv_mb, 1),
                    "cpu_total_mb": round(plan.cpu_total_mb, 1),
                    "is_fully_offloaded": plan.is_fully_offloaded,
                    "estimated_tps": round(plan.estimated_tokens_per_sec, 1),
                },
                "threads": threads,
                "max_safe_context": max_safe_ctx,
            })
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _handle_run_simd_benchmark(self, payload):
        try:
            M = int(payload.get("M", 16))
            N = int(payload.get("N", 32))
            K = int(payload.get("K", 64))
            iterations = int(payload.get("iterations", 3))

            bench = benchmark_cpu_simd(M=M, N=N, K=K, iterations=iterations)
            data = {}
            for k, res in bench.items():
                data[k] = {
                    "instruction_set": res.instruction_set.value,
                    "vector_width_bits": res.vector_width_bits,
                    "int8_lanes_per_cycle": res.int8_lanes_per_cycle,
                    "elapsed_ms": round(res.elapsed_ms, 3),
                    "gops_throughput": round(res.gops_throughput, 4),
                    "speedup_ratio": round(res.speedup_ratio, 2),
                    "numerical_parity": res.numerical_parity,
                }
            self._send_json({"success": True, "results": data})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _handle_run_npu_simulation(self, payload):
        try:
            array_size = int(payload.get("array_size", 16))
            M = int(payload.get("M", 32))
            N = int(payload.get("N", 64))
            K = int(payload.get("K", 128))
            sparsity_pct = float(payload.get("sparsity_percent", 50.0))

            torch.manual_seed(99)
            x = torch.randint(-100, 100, (M, K), dtype=torch.int8)
            w = torch.randint(-100, 100, (N, K), dtype=torch.int8)

            # Apply sparsity
            mask = (torch.rand(M, K) < (sparsity_pct / 100.0))
            x[mask] = 0

            npu = NPU2DSystolicArray(array_size=array_size, clock_mhz=1200.0, power_watts=8.0)
            t0 = time.perf_counter()
            out, stats = npu.execute_matmul_weight_stationary(x, w)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            ref = torch.matmul(x.to(torch.int32), w.t().to(torch.int32))
            parity = bool(torch.equal(out, ref))

            self._send_json({
                "success": True,
                "stats": {
                    "array_dim": stats.array_dim,
                    "total_pes": stats.total_pes,
                    "dataflow": stats.dataflow.value,
                    "total_clock_cycles": stats.total_clock_cycles,
                    "theoretical_min_cycles": stats.theoretical_minimum_cycles,
                    "total_mac_operations": stats.total_mac_operations,
                    "pe_utilization_percent": round(stats.pe_utilization_percent, 1),
                    "zero_skipped_macs": stats.zero_skipped_macs,
                    "energy_saved_percent": round(stats.energy_saved_percent, 1),
                    "simulated_tops": round(stats.simulated_tops, 3),
                    "tops_per_watt": round(stats.tops_per_watt, 3),
                    "wall_time_ms": round(elapsed_ms, 2),
                    "parity": parity,
                },
            })
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _handle_run_tests(self):
        try:
            loader = unittest.TestLoader()
            suite = loader.discover(start_dir=os.path.join(WORKSPACE_ROOT, "tests"))

            stream = io.StringIO()
            runner = unittest.TextTestRunner(stream=stream, verbosity=2)
            result = runner.run(suite)

            self._send_json({
                "success": True,
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "was_successful": result.wasSuccessful(),
                "log_output": stream.getvalue(),
            })
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)

    def _handle_code_files(self):
        try:
            files_to_expose = [
                ("quantization/__init__.py", "Core Package", "Package Root & Unified Public API (60 exports)"),
                ("quantization/precision.py", "Precision", "FP32, FP16, BF16, INT8, INT4 Specifications & Sub-Byte Packing"),
                ("quantization/ptq.py", "PTQ Engine", "Symmetric, Asymmetric, Per-Tensor, Per-Channel & Group-Wise PTQ"),
                ("quantization/calibration.py", "Calibration", "MinMax, Percentile, MSE-optimal search & KL-Divergence"),
                ("quantization/layers.py", "Neural Layers", "QuantizedLinear Layer, Integer GEMM & Hardware Dispatch"),
                ("quantization/metrics.py", "Metrics", "SQNR (dB), MSE, MAE, Cosine Similarity & Comparison Tables"),
                ("quantization/simd.py", "CPU SIMD", "AVX-512 VNNI (VPDPBUSD) & ARM NEON (SDOT) Vector Engines"),
                ("quantization/gpu_cuda.py", "GPU Acceleration", "NVIDIA CUDA Tensor Cores & Warp-Level MMA Tiles"),
                ("quantization/npu_mps.py", "NPU & MPS", "2D Systolic Array (WS/OS, zero-skipping) & Apple Silicon UMA"),
                ("quantization/roofline.py", "Roofline Model", "Operational Intensity, Ridge Points & Memory vs Compute Regimes"),
                ("quantization/hardware_profiler.py", "Profiler", "Hardware Telemetry, Empirical Bandwidth & Quantization Matrix"),
                ("quantization/runtime_config.py", "Runtime Config", "Dynamic Thread Policies, Layer Offload Planner (-ngl) & KV-Cache"),
                ("tests/test_quantization.py", "Tests", "Unit Tests: Precision, PTQ & Calibration (17 tests)"),
                ("tests/test_hardware_acceleration.py", "Tests", "Unit Tests: SIMD, Tensor Cores, NPU/MPS & Dispatch (22 tests)"),
                ("tests/test_runtime_config.py", "Tests", "Unit Tests: Roofline, Profiling & Offloading (10 tests)"),
                ("demo_quantization.py", "Demos", "Interactive Benchmark: PTQ & Calibration (Parts 1-3)"),
                ("demo_hardware_acceleration.py", "Demos", "Interactive Benchmark: Hardware Engines (Parts 4-7)"),
                ("demo_runtime_configuration.py", "Demos", "Interactive Benchmark: Roofline & Offloading (Parts 7-9)"),
                ("README.md", "Documentation", "Comprehensive Architectural & Mathematical Documentation"),
            ]

            file_list = []
            for rel_path, category, desc in files_to_expose:
                full_path = os.path.join(WORKSPACE_ROOT, rel_path.replace("/", os.sep))
                if os.path.exists(full_path):
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    file_list.append({
                        "name": os.path.basename(rel_path),
                        "path": rel_path,
                        "category": category,
                        "description": desc,
                        "size_bytes": os.path.getsize(full_path),
                        "lines_count": len(content.splitlines()),
                        "content": content,
                    })

            self._send_json({"success": True, "files": file_list})
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def start_server(port=PORT):
    server = ThreadedTCPServer(("0.0.0.0", port), DashboardHandler)
    print(f"==========================================================================")
    print(f" PILLAR 4 QUANTIZATION & HARDWARE ACCELERATION DASHBOARD SERVER")
    print(f" Running live at: http://localhost:{port}/")
    print(f" Serving dashboard from: {STATIC_DIR}")
    print(f"==========================================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    p = PORT
    if len(sys.argv) > 1:
        try:
            p = int(sys.argv[1])
        except ValueError:
            pass
    start_server(p)
