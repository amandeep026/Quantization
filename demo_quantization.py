"""
End-to-End Demonstration and Benchmark of Quantization Foundations.

Demonstrates:
1. Floating-Point vs. Integer Precision (FP32, FP16, BF16, INT8, INT4).
   - Bit-level memory anatomy, IEEE 754 representations, dynamic ranges, and memory footprints.
2. Post-Training Quantization (PTQ):
   - Symmetric vs. Asymmetric (Affine) Quantization on symmetric and skewed distributions.
   - Scaling Granularities: Per-Tensor vs. Per-Channel vs. Group-Wise (Block-wise).
3. Calibration & Clipping Strategies:
   - MinMax vs. Percentile vs. MSE-optimal vs. KL-Divergence.
4. Quantized Neural Network Inference:
   - Drop-in QuantizedLinear in INT8 and packed INT4 with simulated integer GEMM.
"""

import sys
import time
import torch
import torch.nn as nn

from quantization.precision import (
    PrecisionType,
    PRECISION_SPECS,
    float_to_bitstring,
    pack_int4,
    unpack_int4,
    get_dtype_memory_size,
)
from quantization.ptq import (
    QuantizationScheme,
    Granularity,
    quantize,
    dequantize,
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
    print_comparison_table,
    QuantizationReport,
)


def print_banner(text: str):
    line = "=" * 80
    print(f"\n{line}\n {text.center(78)}\n{line}")


def print_subbanner(text: str):
    line = "-" * 80
    print(f"\n{line}\n {text}\n{line}")


# ============================================================================
# DEMO 1: Precision Bit-Level Anatomy & Memory Scaling
# ============================================================================

def demo_precision_anatomy():
    print_banner("PART 1: FLOATING-POINT VS. INTEGER PRECISION ANATOMY")
    print("Evaluating: FP32, FP16, BF16, INT8, and sub-byte packed INT4\n")

    # 1. Structural Comparison Table
    print("1.1 Architecture & Theoretical Limits:")
    print(f"{'Precision':<10} | {'Bits':<5} | {'Bytes':<6} | {'Sign':<5} | {'Exp':<5} | {'Mantissa':<9} | {'Min Range':<15} | {'Max Range':<15}")
    print("-" * 88)
    for p_type in PrecisionType:
        s = PRECISION_SPECS[p_type]
        print(f"{p_type.value:<10} | {s.total_bits:<5} | {s.bytes_per_element:<6} | {s.sign_bits:<5} | {s.exponent_bits:<5} | {s.mantissa_bits:<9} | {s.dynamic_range_min:<15.2e} | {s.dynamic_range_max:<15.2e}")

    # 2. Bitstring Disassembly of a Floating-Point Value
    sample_val = -12.375
    print(f"\n1.2 Bit-Level Disassembly for Scalar Value: {sample_val}")
    print(f"{'Format':<25} | {'Sign':<5} | {'Exponent / Int':<16} | {'Mantissa':<25} | {'Hex Pattern':<12}")
    print("-" * 92)
    for p_type in PrecisionType:
        b = float_to_bitstring(sample_val, p_type)
        print(f"{b['format']:<25} | {b['sign']:<5} | {str(b['exponent']):<16} | {str(b['mantissa']):<25} | {b['hex']:<12}")

    # 3. Memory Scaling for a 1,000,000-parameter weight tensor
    numel = 1_000_000
    dummy = torch.empty(numel, dtype=torch.float32)
    print(f"\n1.3 Memory Footprint for a 1M Parameter Tensor (Shape: [1000, 1000]):")
    fp32_size = get_dtype_memory_size(dummy, PrecisionType.FP32)
    for p_type in PrecisionType:
        bytes_size = get_dtype_memory_size(dummy, p_type)
        savings = (1.0 - (bytes_size / fp32_size)) * 100.0
        mb_size = bytes_size / (1024 * 1024)
        print(f"  - {p_type.value:<6}: {bytes_size:>10,} bytes ({mb_size:6.2f} MB) -> Memory Reduction: {savings:5.1f}%")

    # 4. INT4 Sub-Byte Packing Verification
    print("\n1.4 Sub-Byte Bit-Packing Verification (INT4 -> uint8):")
    raw_int4 = torch.tensor([-8, 7, -3, 2, 0, -1, 5, -6], dtype=torch.int32)
    packed_u8, n_orig, shape_orig = pack_int4(raw_int4, signed=True)
    unpacked_int4 = unpack_int4(packed_u8, original_numel=n_orig, original_shape=shape_orig, signed=True)
    print(f"  Original 4-bit integers (8 values): {raw_int4.tolist()}")
    print(f"  Packed into uint8 storage (4 bytes)  : {[hex(x) for x in packed_u8.tolist()]}")
    print(f"  Unpacked integers (exact restore)    : {unpacked_int4.tolist()}")
    assert torch.equal(raw_int4, unpacked_int4.to(torch.int32)), "INT4 Bit-packing verification failed!"
    print("  -> Bit-packing round-trip verification: PASS (Lossless bit restoration)")


# ============================================================================
# DEMO 2: Symmetric vs. Asymmetric PTQ
# ============================================================================

def demo_symmetric_vs_asymmetric():
    print_banner("PART 2: POST-TRAINING QUANTIZATION (PTQ) - SYMMETRIC VS. ASYMMETRIC")

    torch.manual_seed(42)

    # Scenario A: Zero-Centered Weight Distribution N(0, 1)
    print_subbanner("Scenario A: Zero-Centered Weights (Normal Distribution N(0, 1))")
    weights = torch.randn(1024, 1024) * 0.5
    w_bytes = weights.numel() * 4

    # Symmetric Quantization
    q_sym = quantize(weights, bit_width=8, scheme=QuantizationScheme.SYMMETRIC, granularity=Granularity.PER_TENSOR)
    rec_sym = dequantize(q_sym)
    rep_sym = compute_metrics(weights, rec_sym, w_bytes, q_sym.memory_bytes(), "Weights (8-bit Symmetric)", 8)

    # Asymmetric Quantization
    q_asym = quantize(weights, bit_width=8, scheme=QuantizationScheme.ASYMMETRIC, granularity=Granularity.PER_TENSOR)
    rec_asym = dequantize(q_asym)
    rep_asym = compute_metrics(weights, rec_asym, w_bytes, q_asym.memory_bytes(), "Weights (8-bit Asymmetric)", 8)

    print_comparison_table([rep_sym, rep_asym], title="Symmetric vs Asymmetric on Zero-Centered Weights")
    print("Insight: On symmetric zero-centered distributions, Symmetric PTQ matches Asymmetric fidelity")
    print("         without the computational overhead of storing and subtracting integer zero-points.")

    # Scenario B: Skewed Activation Distribution (e.g. Post-ReLU / GeLU [0, 12.0])
    print_subbanner("Scenario B: Skewed Activations (Post-ReLU Distribution [0.0, 12.0])")
    activations = torch.relu(torch.randn(1024, 1024) * 2.5)  # strictly >= 0
    act_bytes = activations.numel() * 4

    q_act_sym = quantize(activations, bit_width=8, scheme=QuantizationScheme.SYMMETRIC, granularity=Granularity.PER_TENSOR)
    rec_act_sym = dequantize(q_act_sym)
    rep_act_sym = compute_metrics(activations, rec_act_sym, act_bytes, q_act_sym.memory_bytes(), "ReLU Act (8-bit Symmetric)", 8)

    q_act_asym = quantize(activations, bit_width=8, scheme=QuantizationScheme.ASYMMETRIC, granularity=Granularity.PER_TENSOR)
    rec_act_asym = dequantize(q_act_asym)
    rep_act_asym = compute_metrics(activations, rec_act_asym, act_bytes, q_act_asym.memory_bytes(), "ReLU Act (8-bit Asymmetric)", 8)

    print_comparison_table([rep_act_sym, rep_act_asym], title="Symmetric vs Asymmetric on Skewed Activations")
    print(f"Insight: For asymmetric activations, Asymmetric PTQ achieves a massive gain of +{rep_act_asym.sqnr_db - rep_act_sym.sqnr_db:.2f} dB SQNR!")
    print("         Reason: Symmetric PTQ wastes 50% of the integer range (bins -127..-1) on values that never occur.")
    print(f"         Zero-point integer alignment: Asymmetric Zero-Point = {q_act_asym.zero_point.item()} (maps real 0.0 with 0 error).")


# ============================================================================
# DEMO 3: Scaling Granularities (Per-Tensor vs. Per-Channel vs. Group-Wise)
# ============================================================================

def demo_granularities():
    print_banner("PART 3: SCALING GRANULARITIES (PER-TENSOR VS. PER-CHANNEL VS. GROUP-WISE)")
    print("Simulating a realistic weight matrix with cross-channel magnitude variance (e.g., in LLM Linear layers)...\n")

    torch.manual_seed(101)
    out_features, in_features = 256, 512
    weights = torch.randn(out_features, in_features)

    # Introduce realistic channel variance: channel 0 has scale 0.01, channel 1 has scale 10.0
    channel_scales = torch.exp(torch.linspace(-2.0, 2.0, out_features)).unsqueeze(1)
    weights = weights * channel_scales
    w_bytes = weights.numel() * 4

    # 1. Per-Tensor INT8
    q_tensor = quantize(weights, bit_width=8, scheme=QuantizationScheme.SYMMETRIC, granularity=Granularity.PER_TENSOR)
    rec_tensor = dequantize(q_tensor)
    rep_tensor = compute_metrics(weights, rec_tensor, w_bytes, q_tensor.memory_bytes(), "INT8 Per-Tensor", 8)

    # 2. Per-Channel INT8 (axis 0 = output channels)
    q_channel = quantize(weights, bit_width=8, scheme=QuantizationScheme.SYMMETRIC, granularity=Granularity.PER_CHANNEL, axis=0)
    rec_channel = dequantize(q_channel)
    rep_channel = compute_metrics(weights, rec_channel, w_bytes, q_channel.memory_bytes(), "INT8 Per-Channel", 8)

    # 3. Group-Wise INT8 (group_size = 64)
    q_group64 = quantize(weights, bit_width=8, scheme=QuantizationScheme.SYMMETRIC, granularity=Granularity.GROUP_WISE, group_size=64)
    rec_group64 = dequantize(q_group64)
    rep_group64 = compute_metrics(weights, rec_group64, w_bytes, q_group64.memory_bytes(), "INT8 Group-Wise (G=64)", 8)

    # 4. Group-Wise INT4 (group_size = 32, as in modern GGUF Q4 / AWQ)
    q_group4 = quantize(weights, bit_width=4, scheme=QuantizationScheme.SYMMETRIC, granularity=Granularity.GROUP_WISE, group_size=32)
    rec_group4 = dequantize(q_group4)
    rep_group4 = compute_metrics(weights, rec_group4, w_bytes, q_group4.memory_bytes(), "INT4 Group-Wise (G=32)", 4)

    print_comparison_table([rep_tensor, rep_channel, rep_group64, rep_group4], title="Quantization Granularity Comparison")
    print(f"Analysis:")
    print(f"  - Per-Tensor SQNR: {rep_tensor.sqnr_db:.2f} dB (severely degraded because large channels dominate scale).")
    print(f"  - Per-Channel SQNR: {rep_channel.sqnr_db:.2f} dB (+{rep_channel.sqnr_db - rep_tensor.sqnr_db:.2f} dB improvement!).")
    print(f"  - Group-Wise INT4 achieves {rep_group4.sqnr_db:.2f} dB SQNR while cutting memory by {rep_group4.compression_ratio:.2f}x!")


# ============================================================================
# DEMO 4: Calibration & Clipping Strategies
# ============================================================================

def demo_calibration_strategies():
    print_banner("PART 4: CLIPPING & CALIBRATION STRATEGIES (OUTLIER SUPPRESSION)")
    print("Simulating LLM activation distribution with heavy outlier tails (e.g., emerging outlier features in Transformer blocks)...\n")

    torch.manual_seed(999)
    # 99.8% Gaussian values, 0.2% extreme outliers up to 40x the standard deviation
    n = 20_000
    base_data = torch.randn(n) * 1.5
    outlier_indices = torch.randperm(n)[:40]
    base_data[outlier_indices] = torch.sign(base_data[outlier_indices]) * (20.0 + torch.rand(40) * 30.0)

    data_bytes = base_data.numel() * 4
    bit_width = 8

    # 1. MinMax Calibration (No clipping)
    min_mm, max_mm = calibrate_tensor(base_data, method=CalibrationMethod.MINMAX, bit_width=bit_width)
    q_mm = quantize(base_data, bit_width=bit_width, scheme=QuantizationScheme.SYMMETRIC, custom_min=min_mm, custom_max=max_mm)
    rec_mm = dequantize(q_mm)
    rep_mm = compute_metrics(base_data, rec_mm, data_bytes, q_mm.memory_bytes(), "MinMax (No Clipping)", bit_width)

    # 2. Percentile Calibration (99.9%)
    min_p99, max_p99 = calibrate_tensor(base_data, method=CalibrationMethod.PERCENTILE, bit_width=bit_width, percentile=99.9)
    q_p99 = quantize(base_data, bit_width=bit_width, scheme=QuantizationScheme.SYMMETRIC, custom_min=min_p99, custom_max=max_p99)
    rec_p99 = dequantize(q_p99)
    rep_p99 = compute_metrics(base_data, rec_p99, data_bytes, q_p99.memory_bytes(), "Percentile (99.9% Clipping)", bit_width)

    # 3. MSE Optimal Grid Search
    min_mse, max_mse = calibrate_tensor(base_data, method=CalibrationMethod.MSE, bit_width=bit_width)
    q_mse = quantize(base_data, bit_width=bit_width, scheme=QuantizationScheme.SYMMETRIC, custom_min=min_mse, custom_max=max_mse)
    rec_mse = dequantize(q_mse)
    rep_mse = compute_metrics(base_data, rec_mse, data_bytes, q_mse.memory_bytes(), "MSE-Optimal Search", bit_width)

    # 4. KL-Divergence / Relative Entropy (TensorRT style)
    min_kl, max_kl = calibrate_tensor(base_data, method=CalibrationMethod.KL_DIVERGENCE, bit_width=bit_width)
    q_kl = quantize(base_data, bit_width=bit_width, scheme=QuantizationScheme.SYMMETRIC, custom_min=min_kl, custom_max=max_kl)
    rec_kl = dequantize(q_kl)
    rep_kl = compute_metrics(base_data, rec_kl, data_bytes, q_kl.memory_bytes(), "KL-Divergence (TensorRT)", bit_width)

    print_comparison_table([rep_mm, rep_p99, rep_mse, rep_kl], title="Calibration & Clipping Benchmark on Outlier-Heavy Data")
    print(f"Calibration Threshold Comparison:")
    print(f"  - MinMax dynamic range boundary       : [-{max_mm.abs().item():.2f}, +{max_mm.abs().item():.2f}]")
    print(f"  - Percentile (99.9%) dynamic boundary : [-{max_p99.abs().item():.2f}, +{max_p99.abs().item():.2f}]")
    print(f"  - MSE-Optimal boundary                : [-{max_mse.abs().item():.2f}, +{max_mse.abs().item():.2f}]")
    print(f"  - KL-Divergence saturation boundary   : [-{max_kl.abs().item():.2f}, +{max_kl.abs().item():.2f}]")
    print(f"\nConclusion: Clipping outlier extremes with MSE or KL-Divergence increases SQNR by +{rep_mse.sqnr_db - rep_mm.sqnr_db:.2f} dB!")


# ============================================================================
# DEMO 5: End-to-End Neural Network Quantized Layer & Inference
# ============================================================================

def demo_quantized_neural_network():
    print_banner("PART 5: END-TO-END NEURAL NETWORK INFERENCE & QUANTIZED LINEAR LAYER")
    
    in_features, out_features = 1024, 2048
    batch_size = 16
    print(f"Instantiating Linear Projection Layer: [{in_features} -> {out_features}] (Parameters: {in_features * out_features:,})")

    torch.manual_seed(777)
    fp32_linear = nn.Linear(in_features, out_features, bias=True)
    x = torch.randn(batch_size, in_features)

    # Baseline FP32 Inference
    t0 = time.perf_counter()
    y_fp32 = fp32_linear(x)
    t_fp32 = (time.perf_counter() - t0) * 1000.0

    # 1. Convert to QuantizedLinear INT8 (Per-Channel Symmetric)
    q_linear_int8 = QuantizedLinear.from_float(
        fp32_linear,
        bit_width=8,
        scheme=QuantizationScheme.SYMMETRIC,
        granularity=Granularity.PER_CHANNEL,
    )
    t0 = time.perf_counter()
    y_int8 = q_linear_int8(x)
    t_int8 = (time.perf_counter() - t0) * 1000.0

    # 2. Convert to QuantizedLinear INT4 (Per-Channel Packed 4-bit)
    q_linear_int4 = QuantizedLinear.from_float(
        fp32_linear,
        bit_width=4,
        scheme=QuantizationScheme.SYMMETRIC,
        granularity=Granularity.PER_CHANNEL,
    )
    t0 = time.perf_counter()
    y_int4 = q_linear_int4(x)
    t_int4 = (time.perf_counter() - t0) * 1000.0

    # 3. Simulate Full Integer GEMM (Quantized Activations INT8 x Quantized Weights INT8)
    y_gemm_sim, _ = q_linear_int8.quantized_gemm_forward(x, act_bit_width=8, act_scheme=QuantizationScheme.SYMMETRIC)

    # Compute Metrics
    orig_bytes = in_features * out_features * 4 + out_features * 4
    mem_int8 = q_linear_int8.memory_footprint_bytes()
    mem_int4 = q_linear_int4.memory_footprint_bytes()

    rep_int8 = compute_metrics(y_fp32, y_int8, orig_bytes, mem_int8["quantized_bytes"], "INT8 Per-Channel Layer", 8)
    rep_int4 = compute_metrics(y_fp32, y_int4, orig_bytes, mem_int4["quantized_bytes"], "INT4 Packed Layer", 4)
    rep_gemm = compute_metrics(y_fp32, y_gemm_sim, orig_bytes, mem_int8["quantized_bytes"], "INT8 x INT8 Full GEMM", 8)

    print_comparison_table([rep_int8, rep_int4, rep_gemm], title="Neural Network Linear Layer Quantization Fidelity")

    print("Memory Footprint Breakdown:")
    print(f"  - FP32 Full-Precision : {orig_bytes:>10,} bytes ({orig_bytes / (1024*1024):.2f} MB)")
    print(f"  - INT8 Quantized Layer: {mem_int8['quantized_bytes']:>10,} bytes ({mem_int8['quantized_bytes'] / (1024*1024):.2f} MB) -> Saved: {mem_int8['saved_bytes'] / (1024*1024):.2f} MB ({(1 - mem_int8['quantized_bytes']/orig_bytes)*100:.1f}% reduction)")
    print(f"  - INT4 Packed Layer   : {mem_int4['quantized_bytes']:>10,} bytes ({mem_int4['quantized_bytes'] / (1024*1024):.2f} MB) -> Saved: {mem_int4['saved_bytes'] / (1024*1024):.2f} MB ({(1 - mem_int4['quantized_bytes']/orig_bytes)*100:.1f}% reduction)")

    print(f"\nForward Output Fidelity:")
    print(f"  - INT8 Layer Output SQNR : {rep_int8.sqnr_db:.2f} dB (Cosine Sim: {rep_int8.cosine_similarity:.6f})")
    print(f"  - INT4 Layer Output SQNR : {rep_int4.sqnr_db:.2f} dB (Cosine Sim: {rep_int4.cosine_similarity:.6f})")
    print(f"  - Full INT8 GEMM SQNR    : {rep_gemm.sqnr_db:.2f} dB (Cosine Sim: {rep_gemm.cosine_similarity:.6f})")


def main():
    print_banner("PILLAR 4: AUTOMATED HARDWARE-AWARE QUANTIZATION FOUNDATIONS")
    print("Executing full technical pipeline covering Precision, PTQ, Scaling, and Calibration.\n")

    demo_precision_anatomy()
    demo_symmetric_vs_asymmetric()
    demo_granularities()
    demo_calibration_strategies()
    demo_quantized_neural_network()

    print_banner("ALL QUANTIZATION MODULES AND BENCHMARKS EXECUTED SUCCESSFULLY")


if __name__ == "__main__":
    main()
