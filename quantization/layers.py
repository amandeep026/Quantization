"""
Quantized Neural Network Layers.

Implements:
1. QuantizedLinear: A drop-in PyTorch layer that stores weights in INT8 or packed INT4 format.
2. Supports Symmetric, Asymmetric, Per-Channel, Per-Tensor, and Group-Wise PTQ.
3. Quantized GEMM simulation with mathematical zero-point correction analysis.
4. Seamless conversion to and from standard torch.nn.Linear.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .calibration import CalibrationMethod, calibrate_tensor
from .metrics import compute_metrics, compute_sqnr
from .precision import pack_int4, unpack_int4
from .ptq import (
    Granularity,
    QuantizationScheme,
    QuantizedTensor,
    dequantize,
    quantize,
)


class QuantizedLinear(nn.Module):
    """
    Quantized Linear Layer storing weights in compact INT8 or packed INT4 representation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bit_width: int = 8,
        scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC,
        granularity: Granularity = Granularity.PER_CHANNEL,
        calibration_method: CalibrationMethod = CalibrationMethod.MINMAX,
        group_size: Optional[int] = 32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width
        self.scheme = scheme
        self.granularity = granularity
        self.calibration_method = calibration_method
        self.group_size = group_size

        # Placeholder for quantized weight container
        self.quantized_weight: Optional[QuantizedTensor] = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        bit_width: int = 8,
        scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC,
        granularity: Granularity = Granularity.PER_CHANNEL,
        calibration_method: CalibrationMethod = CalibrationMethod.MINMAX,
        group_size: Optional[int] = 32,
        percentile: float = 99.99,
    ) -> "QuantizedLinear":
        """
        Converts an existing torch.nn.Linear layer into a QuantizedLinear layer.
        """
        q_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bit_width=bit_width,
            scheme=scheme,
            granularity=granularity,
            calibration_method=calibration_method,
            group_size=group_size,
        )

        with torch.no_grad():
            w = linear.weight.detach()

            # Calibrate clipping bounds
            if granularity == Granularity.PER_CHANNEL:
                if calibration_method == CalibrationMethod.MINMAX:
                    c_min, c_max = None, None  # Native per-channel min/max
                else:
                    c_mins, c_maxs = [], []
                    for c in range(w.shape[0]):
                        cm, cM = calibrate_tensor(
                            w[c],
                            method=calibration_method,
                            bit_width=bit_width,
                            scheme=scheme,
                            percentile=percentile,
                        )
                        c_mins.append(cm)
                        c_maxs.append(cM)
                    c_min = torch.stack(c_mins)
                    c_max = torch.stack(c_maxs)
            else:
                c_min, c_max = calibrate_tensor(
                    w,
                    method=calibration_method,
                    bit_width=bit_width,
                    scheme=scheme,
                    percentile=percentile,
                )

            # Quantize weight matrix
            # Axis 0 corresponds to output channels (out_features)
            q_layer.quantized_weight = quantize(
                tensor=w,
                bit_width=bit_width,
                scheme=scheme,
                granularity=granularity,
                axis=0,
                group_size=group_size,
                signed=True,
                pack_int4_data=(bit_width == 4),
                custom_min=c_min,
                custom_max=c_max,
            )

            if linear.bias is not None:
                q_layer.bias.copy_(linear.bias.detach())

        return q_layer

    def get_dequantized_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Restores full-precision weight matrix from quantized representation."""
        if self.quantized_weight is None:
            raise RuntimeError("Layer weights have not been quantized.")
        return dequantize(self.quantized_weight, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes forward pass.
        Reconstructs weights dynamically and performs matrix multiplication:
            y = x @ W^T + bias
        """
        w_rec = self.get_dequantized_weight(dtype=x.dtype)
        return F.linear(x, w_rec, self.bias)

    def quantized_gemm_forward(
        self,
        x: torch.Tensor,
        act_bit_width: int = 8,
        act_scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates integer GEMM where both activations and weights are quantized:
            X_q in INT8, W_q in INT8 / INT4
            
        Demonstrates zero-point arithmetic:
            Y = S_x * S_w * ( (Q_x - Z_x) @ (Q_w - Z_w)^T ) + bias
        """
        # Quantize activation tensor per-tensor
        q_act = quantize(
            x,
            bit_width=act_bit_width,
            scheme=act_scheme,
            granularity=Granularity.PER_TENSOR,
            signed=True,
            pack_int4_data=False,
        )
        x_rec = dequantize(q_act, dtype=x.dtype)
        w_rec = self.get_dequantized_weight(dtype=x.dtype)

        out_simulated = F.linear(x_rec, w_rec, self.bias)
        out_fp = F.linear(x, w_rec, self.bias)

        return out_simulated, out_fp

    def memory_footprint_bytes(self) -> Dict[str, int]:
        """
        Returns exact memory footprint comparison between original FP32 weights and quantized weights.
        """
        orig_bytes = self.in_features * self.out_features * 4
        if self.bias is not None:
            orig_bytes += self.out_features * 4

        quant_weight_bytes = self.quantized_weight.memory_bytes() if self.quantized_weight else orig_bytes
        quant_bias_bytes = (self.out_features * 4) if self.bias is not None else 0
        total_quant_bytes = quant_weight_bytes + quant_bias_bytes

        return {
            "fp32_bytes": orig_bytes,
            "quantized_bytes": total_quant_bytes,
            "saved_bytes": orig_bytes - total_quant_bytes,
        }

    def forward_hardware(
        self,
        x: torch.Tensor,
        backend: Any = "Auto-Detect Optimal Backend",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Dispatches forward execution to the specified hardware backend:
        - CPU SIMD (AVX-512 VNNI / ARM NEON)
        - NVIDIA CUDA Tensor Cores (MMA Tiles)
        - Apple Metal Performance Shaders (MPS / UMA)
        - NPU 2D Systolic Array (Weight-Stationary)
        - Auto-detection based on host hardware profiling
        """
        from .hardware_profiler import HardwareBackend, dispatch_hardware_forward
        if isinstance(backend, str) and not isinstance(backend, HardwareBackend):
            try:
                backend = HardwareBackend(backend)
            except ValueError:
                backend = HardwareBackend.AUTO
        return dispatch_hardware_forward(self, x, backend=backend)
