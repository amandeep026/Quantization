"""
Post-Training Quantization (PTQ) Core Engine.

Implements:
1. Symmetric vs. Asymmetric (Affine) Quantization
2. Scaling Granularities:
   - Per-Tensor
   - Per-Channel / Per-Axis (e.g., across output channels in Linear layers)
   - Group-wise / Block-wise (e.g., blocks of 32, 64, or 128 elements, matching GGUF/GPTQ)
3. Precision widths: 8-bit (INT8) and 4-bit (INT4, with sub-byte packing)
4. Exact integer zero-point alignment and numerical error protection.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple, Union

import torch

from .precision import pack_int4, unpack_int4


class QuantizationScheme(str, Enum):
    """Mapping scheme between continuous float values and discrete integer grid."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class Granularity(str, Enum):
    """Spatial resolution of scaling factors across the tensor dimensions."""
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    GROUP_WISE = "group_wise"


@dataclass
class QuantizedTensor:
    """
    Encapsulates all components of a quantized tensor for storage and inference.
    """
    quantized_data: torch.Tensor          # Stored integer values (or packed uint8 for INT4)
    scale: torch.Tensor                   # Floating point scaling factors
    zero_point: Optional[torch.Tensor]    # Integer zero points (None or zero for symmetric)
    original_shape: Tuple[int, ...]       # Unquantized shape
    scheme: QuantizationScheme
    granularity: Granularity
    bit_width: int                        # 8 or 4
    axis: int = 0
    group_size: Optional[int] = None
    is_packed: bool = False
    signed: bool = True
    original_numel: int = 0

    def memory_bytes(self) -> int:
        """Calculates exact memory consumption in bytes of the quantized representation."""
        data_bytes = self.quantized_data.element_size() * self.quantized_data.numel()
        scale_bytes = self.scale.element_size() * self.scale.numel()
        zp_bytes = (self.zero_point.element_size() * self.zero_point.numel()) if self.zero_point is not None else 0
        return data_bytes + scale_bytes + zp_bytes


# ============================================================================
# Quantization Parameter Calculations
# ============================================================================

def get_quant_range(bit_width: int, scheme: QuantizationScheme, signed: bool = True) -> Tuple[int, int]:
    """
    Returns (q_min, q_max) for the specified bit width and scheme.
    For symmetric quantization, range is strictly symmetric around zero:
        INT8: [-127, 127]
        INT4: [-7, 7]
    For asymmetric quantization:
        Signed INT8: [-128, 127]
        Unsigned INT8: [0, 255]
        Signed INT4: [-8, 7]
        Unsigned INT4: [0, 15]
    """
    if scheme == QuantizationScheme.SYMMETRIC:
        q_max = (1 << (bit_width - 1)) - 1
        q_min = -q_max
        return q_min, q_max
    else:
        if signed:
            q_min = -(1 << (bit_width - 1))
            q_max = (1 << (bit_width - 1)) - 1
            return q_min, q_max
        else:
            q_min = 0
            q_max = (1 << bit_width) - 1
            return q_min, q_max


def calculate_scale_and_zeropoint(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    bit_width: int,
    scheme: QuantizationScheme,
    signed: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Computes optimal scale factor S and integer zero-point Z.
    
    Mathematical Formulation:
    -------------------------
    Symmetric:
        x_abs_max = max(|min_val|, |max_val|)
        S = x_abs_max / q_max
        Z = 0
        
    Asymmetric:
        S = (max_val - min_val) / (q_max - q_min)
        Z = round(-min_val / S) + q_min, clamped to [q_min, q_max]
    """
    q_min, q_max = get_quant_range(bit_width, scheme, signed)

    if scheme == QuantizationScheme.SYMMETRIC:
        abs_max = torch.max(min_val.abs(), max_val.abs())
        # Avoid division by zero for flat tensors
        abs_max = torch.clamp(abs_max, min=eps)
        scale = abs_max / float(q_max)
        zero_point = None
    else:
        range_val = max_val - min_val
        range_val = torch.clamp(range_val, min=eps)
        scale = range_val / float(q_max - q_min)
        
        # Z = round(-min_val / scale) + q_min
        # Clamped to ensure zero point fits strictly into integer grid
        zp = torch.round(-min_val / scale) + q_min
        zero_point = torch.clamp(zp, q_min, q_max).to(torch.int32)

    return scale, zero_point


# ============================================================================
# Core Quantize and Dequantize Routines
# ============================================================================

def quantize(
    tensor: torch.Tensor,
    bit_width: int = 8,
    scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC,
    granularity: Granularity = Granularity.PER_TENSOR,
    axis: int = 0,
    group_size: Optional[int] = None,
    signed: bool = True,
    pack_int4_data: bool = True,
    custom_min: Optional[torch.Tensor] = None,
    custom_max: Optional[torch.Tensor] = None,
) -> QuantizedTensor:
    """
    Quantizes a floating-point tensor to INT8 or INT4 with chosen scheme and granularity.
    
    Args:
        tensor: Input FP32/FP16/BF16 tensor.
        bit_width: 8 for INT8, 4 for INT4.
        scheme: SYMMETRIC or ASYMMETRIC.
        granularity: PER_TENSOR, PER_CHANNEL, or GROUP_WISE.
        axis: Target axis for per-channel or group-wise scaling.
        group_size: Number of elements per group for GROUP_WISE (e.g., 32, 64, 128).
        signed: Whether integer range is signed.
        pack_int4_data: If True and bit_width == 4, packs 2 elements per uint8 byte.
        custom_min: Pre-calibrated minimum (optional, for clipping).
        custom_max: Pre-calibrated maximum (optional, for clipping).
        
    Returns:
        QuantizedTensor: Object containing quantized data, scales, zero points, and metadata.
    """
    if bit_width not in (4, 8):
        raise ValueError(f"bit_width must be 4 or 8, got {bit_width}")

    orig_shape = tuple(tensor.shape)
    orig_numel = tensor.numel()
    q_min, q_max = get_quant_range(bit_width, scheme, signed)

    # ------------------------------------------------------------------------
    # 1. Granularity Reshaping & Parameter Computation
    # ------------------------------------------------------------------------
    if granularity == Granularity.PER_TENSOR:
        if custom_min is not None and custom_max is not None:
            min_v = custom_min
            max_v = custom_max
        else:
            min_v = tensor.min()
            max_v = tensor.max()

        scale, zp = calculate_scale_and_zeropoint(min_v, max_v, bit_width, scheme, signed)
        
        # Quantize
        if scheme == QuantizationScheme.SYMMETRIC:
            q = torch.round(tensor / scale)
        else:
            q = torch.round(tensor / scale) + zp

        q = torch.clamp(q, q_min, q_max)

    elif granularity == Granularity.PER_CHANNEL:
        # Per-channel along specified axis (e.g., output channels in linear weights)
        num_channels = tensor.shape[axis]
        # Reshape to 2D: [num_channels, -1]
        dims = list(range(tensor.ndim))
        dims[0], dims[axis] = dims[axis], dims[0]
        permuted = tensor.permute(*dims)
        flat_channels = permuted.reshape(num_channels, -1)

        if custom_min is not None and custom_max is not None:
            if custom_min.numel() == 1:
                min_v = custom_min.expand(num_channels, 1)
                max_v = custom_max.expand(num_channels, 1)
            else:
                min_v = custom_min.reshape(num_channels, 1)
                max_v = custom_max.reshape(num_channels, 1)
        else:
            min_v = flat_channels.min(dim=1, keepdim=True).values
            max_v = flat_channels.max(dim=1, keepdim=True).values

        scale, zp = calculate_scale_and_zeropoint(min_v, max_v, bit_width, scheme, signed)

        if scheme == QuantizationScheme.SYMMETRIC:
            q_flat = torch.round(flat_channels / scale)
        else:
            q_flat = torch.round(flat_channels / scale) + zp

        q_flat = torch.clamp(q_flat, q_min, q_max)
        
        # Un-permute back to original tensor layout
        unflattened = q_flat.reshape(permuted.shape)
        q = unflattened.permute(*dims)

        # Broadcast-ready shapes for scale and zp
        scale_shape = [1] * tensor.ndim
        scale_shape[axis] = num_channels
        scale = scale.reshape(scale_shape)
        if zp is not None:
            zp = zp.reshape(scale_shape)

    elif granularity == Granularity.GROUP_WISE:
        if group_size is None or group_size <= 0:
            group_size = 32
        
        # Flatten tensor along target axis or total elements
        flat_tensor = tensor.reshape(-1)
        total_len = flat_tensor.numel()
        
        # Pad flat tensor if not divisible by group_size
        remainder = total_len % group_size
        padding_needed = (group_size - remainder) % group_size
        if padding_needed > 0:
            flat_tensor = torch.cat([flat_tensor, torch.zeros(padding_needed, device=tensor.device, dtype=tensor.dtype)])
            
        num_groups = flat_tensor.numel() // group_size
        grouped = flat_tensor.reshape(num_groups, group_size)

        if custom_min is not None and custom_max is not None:
            if custom_min.numel() == 1:
                min_v = custom_min.expand(num_groups, 1)
                max_v = custom_max.expand(num_groups, 1)
            else:
                min_v = custom_min.reshape(num_groups, 1)
                max_v = custom_max.reshape(num_groups, 1)
        else:
            min_v = grouped.min(dim=1, keepdim=True).values
            max_v = grouped.max(dim=1, keepdim=True).values

        scale, zp = calculate_scale_and_zeropoint(min_v, max_v, bit_width, scheme, signed)

        if scheme == QuantizationScheme.SYMMETRIC:
            q_grouped = torch.round(grouped / scale)
        else:
            q_grouped = torch.round(grouped / scale) + zp

        q_grouped = torch.clamp(q_grouped, q_min, q_max)
        
        # Un-pad and restore original shape
        q = q_grouped.reshape(-1)[:total_len].reshape(orig_shape)

    else:
        raise ValueError(f"Unsupported granularity: {granularity}")

    # ------------------------------------------------------------------------
    # 2. Storage Encoding (INT8 tensor vs. Packed INT4 uint8)
    # ------------------------------------------------------------------------
    is_packed = False
    if bit_width == 4 and pack_int4_data:
        packed_data, _, _ = pack_int4(q, signed=signed)
        final_data = packed_data
        is_packed = True
    else:
        final_data = q.to(torch.int8 if signed else torch.uint8)

    return QuantizedTensor(
        quantized_data=final_data,
        scale=scale,
        zero_point=zp,
        original_shape=orig_shape,
        scheme=scheme,
        granularity=granularity,
        bit_width=bit_width,
        axis=axis,
        group_size=group_size,
        is_packed=is_packed,
        signed=signed,
        original_numel=orig_numel,
    )


def dequantize(q_tensor: QuantizedTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Dequantizes a QuantizedTensor back to continuous floating-point values.
    
    Mathematical Formulation:
    -------------------------
    Symmetric:
        x_hat = q * S
        
    Asymmetric:
        x_hat = (q - Z) * S
    """
    # 1. Unpack if stored as packed 4-bit
    if q_tensor.is_packed:
        q = unpack_int4(
            q_tensor.quantized_data,
            original_numel=q_tensor.original_numel,
            original_shape=q_tensor.original_shape,
            signed=q_tensor.signed,
        ).to(dtype)
    else:
        q = q_tensor.quantized_data.to(dtype)

    scale = q_tensor.scale.to(dtype)
    zp = q_tensor.zero_point.to(dtype) if q_tensor.zero_point is not None else None

    # 2. Perform dequantization according to granularity and scheme
    if q_tensor.granularity == Granularity.PER_TENSOR or q_tensor.granularity == Granularity.PER_CHANNEL:
        if q_tensor.scheme == QuantizationScheme.SYMMETRIC:
            dequantized = q * scale
        else:
            dequantized = (q - zp) * scale
        return dequantized

    elif q_tensor.granularity == Granularity.GROUP_WISE:
        group_size = q_tensor.group_size or 32
        flat_q = q.reshape(-1)
        total_len = flat_q.numel()

        remainder = total_len % group_size
        padding_needed = (group_size - remainder) % group_size
        if padding_needed > 0:
            flat_q = torch.cat([flat_q, torch.zeros(padding_needed, device=flat_q.device, dtype=dtype)])

        num_groups = flat_q.numel() // group_size
        grouped_q = flat_q.reshape(num_groups, group_size)

        if q_tensor.scheme == QuantizationScheme.SYMMETRIC:
            grouped_dequant = grouped_q * scale
        else:
            grouped_dequant = (grouped_q - zp) * scale

        dequantized = grouped_dequant.reshape(-1)[:total_len].reshape(q_tensor.original_shape)
        return dequantized

    else:
        raise ValueError(f"Unknown granularity: {q_tensor.granularity}")
