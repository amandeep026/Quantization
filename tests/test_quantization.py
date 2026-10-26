"""
Comprehensive Unit Test Suite for Quantization Engine.

Tests:
1. Precision specifications, bit-level disassembly, and conversions (FP32, FP16, BF16, INT8, INT4).
2. INT4 bit packing & unpacking roundtrips (signed, unsigned, even/odd lengths, arbitrary shapes).
3. Symmetric vs. Asymmetric PTQ (zero-point alignment, scale computation).
4. Granularities: Per-Tensor, Per-Channel, Group-Wise (blocks of 32/64).
5. Calibration & Clipping algorithms: MinMax, Percentile, MSE-optimal, KL-Divergence.
6. QuantizedLinear forward pass fidelity, SQNR, and memory reduction.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from quantization.precision import (
    PrecisionType,
    PRECISION_SPECS,
    pack_int4,
    unpack_int4,
    convert_precision,
    float_to_bitstring,
    get_dtype_memory_size,
)
from quantization.ptq import (
    QuantizationScheme,
    Granularity,
    calculate_scale_and_zeropoint,
    quantize,
    dequantize,
)
from quantization.calibration import (
    CalibrationMethod,
    Calibrator,
    calibrate_tensor,
)
from quantization.layers import QuantizedLinear
from quantization.metrics import compute_metrics, compute_sqnr


class TestPrecisionFormats(unittest.TestCase):
    """Verifies precision definitions, bit layout, and conversions."""

    def test_precision_specs(self):
        self.assertEqual(PRECISION_SPECS[PrecisionType.FP32].total_bits, 32)
        self.assertEqual(PRECISION_SPECS[PrecisionType.FP16].total_bits, 16)
        self.assertEqual(PRECISION_SPECS[PrecisionType.BF16].total_bits, 16)
        self.assertEqual(PRECISION_SPECS[PrecisionType.INT8].total_bits, 8)
        self.assertEqual(PRECISION_SPECS[PrecisionType.INT4].total_bits, 4)

    def test_bitstring_disassembly(self):
        val = 1.5
        fp32_bits = float_to_bitstring(val, PrecisionType.FP32)
        self.assertEqual(fp32_bits["sign"], "0")
        self.assertEqual(len(fp32_bits["exponent"]), 8)
        self.assertEqual(len(fp32_bits["mantissa"]), 23)

        fp16_bits = float_to_bitstring(val, PrecisionType.FP16)
        self.assertEqual(len(fp16_bits["exponent"]), 5)
        self.assertEqual(len(fp16_bits["mantissa"]), 10)

        bf16_bits = float_to_bitstring(val, PrecisionType.BF16)
        self.assertEqual(len(bf16_bits["exponent"]), 8)
        self.assertEqual(len(bf16_bits["mantissa"]), 7)

    def test_precision_conversions(self):
        t = torch.randn(10, 10)
        self.assertEqual(convert_precision(t, PrecisionType.FP32).dtype, torch.float32)
        self.assertEqual(convert_precision(t, PrecisionType.FP16).dtype, torch.float16)
        self.assertEqual(convert_precision(t, PrecisionType.BF16).dtype, torch.bfloat16)
        self.assertEqual(convert_precision(t, PrecisionType.INT8).dtype, torch.int8)

    def test_int4_pack_unpack_even_length(self):
        """Tests exact reconstruction of signed INT4 numbers [-8..7]."""
        original = torch.tensor([-8, -5, -1, 0, 1, 3, 7, -2], dtype=torch.int32)
        packed, numel, shape = pack_int4(original, signed=True)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.numel(), 4)  # 8 integers -> 4 bytes

        unpacked = unpack_int4(packed, original_numel=numel, original_shape=shape, signed=True)
        torch.testing.assert_close(unpacked.to(torch.int32), original)

    def test_int4_pack_unpack_odd_length(self):
        """Tests odd-length padding and truncation handling."""
        original = torch.tensor([-7, 2, -3, 5, 0], dtype=torch.int32)
        packed, numel, shape = pack_int4(original, signed=True)
        self.assertEqual(packed.numel(), 3)  # 5 elements padded to 6 -> 3 bytes

        unpacked = unpack_int4(packed, original_numel=numel, original_shape=shape, signed=True)
        torch.testing.assert_close(unpacked.to(torch.int32), original)

    def test_int4_pack_unpack_multidimensional(self):
        """Tests 2D matrix packing and shape restoration."""
        original = torch.randint(-8, 8, (16, 32), dtype=torch.int32)
        packed, numel, shape = pack_int4(original, signed=True)
        unpacked = unpack_int4(packed, original_numel=numel, original_shape=shape, signed=True)
        torch.testing.assert_close(unpacked.to(torch.int32), original)


class TestPostTrainingQuantization(unittest.TestCase):
    """Verifies symmetric vs asymmetric PTQ and granularities."""

    def test_symmetric_quantization(self):
        """Symmetric scale maps 0.0 directly to 0 with zero_point = None."""
        t = torch.randn(64, 64)
        q_tensor = quantize(t, bit_width=8, scheme=QuantizationScheme.SYMMETRIC, granularity=Granularity.PER_TENSOR)
        self.assertIsNone(q_tensor.zero_point)

        dequant = dequantize(q_tensor)
        sqnr = compute_sqnr(t, dequant)
        # Uniform 8-bit quantization of normal distribution typically yields >35 dB SQNR
        self.assertGreater(sqnr, 35.0)

    def test_asymmetric_quantization_zero_point(self):
        """Asymmetric PTQ maps real 0.0 to exact integer zero-point."""
        # Skewed positive distribution like post-ReLU activations [0.0, 10.0]
        t = torch.relu(torch.randn(100, 100)) * 5.0
        q_tensor = quantize(t, bit_width=8, scheme=QuantizationScheme.ASYMMETRIC, granularity=Granularity.PER_TENSOR)
        
        self.assertIsNotNone(q_tensor.zero_point)
        zp = q_tensor.zero_point.item()
        
        # Test that 0.0 maps exactly to zp
        zero_tensor = torch.tensor([0.0])
        zero_q = torch.clamp(torch.round(zero_tensor / q_tensor.scale) + zp, -128, 127)
        self.assertEqual(zero_q.item(), zp)

        # Dequantizing zp should yield 0.0 exactly
        zero_dequant = (zero_q - zp) * q_tensor.scale
        self.assertAlmostEqual(zero_dequant.item(), 0.0, places=6)

    def test_per_channel_quantization(self):
        """Per-channel should produce distinct scales for each output channel."""
        # Row 0 has small scale, row 1 has large scale
        t = torch.randn(4, 64)
        t[0] *= 0.01
        t[1] *= 10.0

        q_tensor = quantize(t, bit_width=8, scheme=QuantizationScheme.SYMMETRIC, granularity=Granularity.PER_CHANNEL, axis=0)
        self.assertEqual(q_tensor.scale.shape[0], 4)
        self.assertGreater(q_tensor.scale[1].item(), q_tensor.scale[0].item() * 50)

        dequant = dequantize(q_tensor)
        sqnr = compute_sqnr(t, dequant)
        self.assertGreater(sqnr, 40.0)

    def test_group_wise_quantization(self):
        """Group-wise quantization divides weights into blocks (e.g. 32 elements)."""
        t = torch.randn(64, 128)
        q_tensor = quantize(t, bit_width=8, scheme=QuantizationScheme.SYMMETRIC, granularity=Granularity.GROUP_WISE, group_size=32)
        dequant = dequantize(q_tensor)
        self.assertEqual(dequant.shape, t.shape)
        sqnr = compute_sqnr(t, dequant)
        self.assertGreater(sqnr, 38.0)

    def test_int4_quantization_and_dequantization(self):
        """Verifies packed INT4 quantization end-to-end."""
        t = torch.randn(32, 64)
        q_tensor = quantize(
            t,
            bit_width=4,
            scheme=QuantizationScheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
            pack_int4_data=True,
        )
        self.assertTrue(q_tensor.is_packed)
        self.assertEqual(q_tensor.quantized_data.dtype, torch.uint8)

        dequant = dequantize(q_tensor)
        self.assertEqual(dequant.shape, t.shape)
        sqnr = compute_sqnr(t, dequant)
        # 4-bit quantization typically achieves 18-25 dB SQNR
        self.assertGreater(sqnr, 18.0)


class TestCalibrationStrategies(unittest.TestCase):
    """Verifies calibration algorithms: MinMax, Percentile, MSE, and KL-Divergence."""

    def setUp(self):
        # Create normal distribution with intentional high-magnitude outliers
        torch.manual_seed(42)
        self.clean_data = torch.randn(2000)
        self.data_with_outliers = self.clean_data.clone()
        self.data_with_outliers[0] = 50.0  # Extreme outlier
        self.data_with_outliers[1] = -45.0

    def test_minmax_calibration(self):
        min_v, max_v = Calibrator.calibrate_minmax(self.data_with_outliers)
        self.assertAlmostEqual(max_v.item(), 50.0, places=2)
        self.assertAlmostEqual(min_v.item(), -45.0, places=2)

    def test_percentile_calibration_suppresses_outliers(self):
        min_v, max_v = Calibrator.calibrate_percentile(self.data_with_outliers, percentile=99.9)
        # Percentile should clip extreme outliers (50 and -45)
        self.assertLess(max_v.item(), 20.0)
        self.assertGreater(min_v.item(), -20.0)

    def test_mse_calibration(self):
        min_v, max_v = Calibrator.calibrate_mse(self.data_with_outliers, bit_width=8, scheme=QuantizationScheme.SYMMETRIC)
        self.assertLess(max_v.item(), 50.0)
        self.assertGreater(max_v.item(), 0.0)

    def test_kl_divergence_calibration(self):
        min_v, max_v = Calibrator.calibrate_kl_divergence(self.data_with_outliers, bit_width=8)
        self.assertLess(max_v.item(), 50.0)
        self.assertGreater(max_v.item(), 0.0)


class TestQuantizedLinearLayer(unittest.TestCase):
    """Verifies QuantizedLinear layer behavior, forward pass, and memory savings."""

    def test_int8_linear_conversion_and_inference(self):
        in_feat, out_feat = 128, 64
        fp_linear = nn.Linear(in_feat, out_feat, bias=True)
        
        q_linear = QuantizedLinear.from_float(
            fp_linear,
            bit_width=8,
            scheme=QuantizationScheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
        )

        x = torch.randn(8, in_feat)
        y_fp = fp_linear(x)
        y_q = q_linear(x)

        sqnr = compute_sqnr(y_fp, y_q)
        self.assertGreater(sqnr, 35.0)

        mem = q_linear.memory_footprint_bytes()
        self.assertLess(mem["quantized_bytes"], mem["fp32_bytes"] * 0.35)  # Nearly 4x smaller

    def test_int4_linear_conversion_and_inference(self):
        in_feat, out_feat = 128, 64
        fp_linear = nn.Linear(in_feat, out_feat, bias=True)
        
        q_linear = QuantizedLinear.from_float(
            fp_linear,
            bit_width=4,
            scheme=QuantizationScheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
        )

        x = torch.randn(8, in_feat)
        y_fp = fp_linear(x)
        y_q = q_linear(x)

        sqnr = compute_sqnr(y_fp, y_q)
        self.assertGreater(sqnr, 18.0)

        mem = q_linear.memory_footprint_bytes()
        # 4-bit should be roughly 7x-8x smaller than FP32
        self.assertLess(mem["quantized_bytes"], mem["fp32_bytes"] * 0.20)


if __name__ == "__main__":
    unittest.main()
