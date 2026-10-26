"""
Precision Types, Bit-Level Representation, and Memory Profiling.

Covers:
- FP32: IEEE 754 Single-Precision (1 sign, 8 exponent, 23 mantissa)
- FP16: IEEE 754 Half-Precision (1 sign, 5 exponent, 10 mantissa)
- BF16: Brain Floating Point (1 sign, 8 exponent, 7 mantissa)
- INT8: 8-bit Signed Integer (-128 to 127)
- INT4: 4-bit Signed/Unsigned Integer with bit-packing (2 elements per byte)
"""

from dataclasses import dataclass
from enum import Enum
import struct
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


class PrecisionType(str, Enum):
    """Enumeration of supported precision formats."""
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"
    INT4 = "INT4"


@dataclass(frozen=True)
class PrecisionSpecs:
    """Theoretical and architectural specifications of a numerical format."""
    name: str
    total_bits: int
    bytes_per_element: float
    sign_bits: int
    exponent_bits: int
    mantissa_bits: int
    dynamic_range_min: float
    dynamic_range_max: float
    machine_epsilon: Optional[float]
    description: str


PRECISION_SPECS: Dict[PrecisionType, PrecisionSpecs] = {
    PrecisionType.FP32: PrecisionSpecs(
        name="FP32 (IEEE 754 Single Precision)",
        total_bits=32,
        bytes_per_element=4.0,
        sign_bits=1,
        exponent_bits=8,
        mantissa_bits=23,
        dynamic_range_min=-3.4028235e38,
        dynamic_range_max=3.4028235e38,
        machine_epsilon=1.1920929e-7,
        description="Standard single-precision floating point with 8-bit exponent and 24-bit effective significand.",
    ),
    PrecisionType.FP16: PrecisionSpecs(
        name="FP16 (IEEE 754 Half Precision)",
        total_bits=16,
        bytes_per_element=2.0,
        sign_bits=1,
        exponent_bits=5,
        mantissa_bits=10,
        dynamic_range_min=-65504.0,
        dynamic_range_max=65504.0,
        machine_epsilon=9.765625e-4,
        description="Half precision with narrow 5-bit exponent. Highly susceptible to underflow/overflow.",
    ),
    PrecisionType.BF16: PrecisionSpecs(
        name="BF16 (Bfloat16 / Brain Floating Point)",
        total_bits=16,
        bytes_per_element=2.0,
        sign_bits=1,
        exponent_bits=8,
        mantissa_bits=7,
        dynamic_range_min=-3.3895314e38,
        dynamic_range_max=3.3895314e38,
        machine_epsilon=7.8125e-3,
        description="Truncated FP32 with identical dynamic range (8-bit exponent) but lower 7-bit precision.",
    ),
    PrecisionType.INT8: PrecisionSpecs(
        name="INT8 (8-bit Signed Integer)",
        total_bits=8,
        bytes_per_element=1.0,
        sign_bits=1,
        exponent_bits=0,
        mantissa_bits=7,
        dynamic_range_min=-128.0,
        dynamic_range_max=127.0,
        machine_epsilon=None,
        description="Uniform fixed-point 8-bit signed integer. 4x memory compression relative to FP32.",
    ),
    PrecisionType.INT4: PrecisionSpecs(
        name="INT4 (4-bit Signed/Unsigned Integer)",
        total_bits=4,
        bytes_per_element=0.5,
        sign_bits=1,
        exponent_bits=0,
        mantissa_bits=3,
        dynamic_range_min=-8.0,
        dynamic_range_max=7.0,
        machine_epsilon=None,
        description="Sub-byte 4-bit integer packed 2 values per byte. 8x memory compression relative to FP32.",
    ),
}


# ============================================================================
# INT4 Bit Packing & Unpacking (Sub-Byte Packing: 2 integers per uint8 byte)
# ============================================================================

def pack_int4(tensor: torch.Tensor, signed: bool = True) -> Tuple[torch.Tensor, int, Tuple[int, ...]]:
    """
    Packs a tensor of 4-bit integers into a contiguous uint8 storage tensor.
    
    Each byte holds two 4-bit nibbles:
        - Low nibble (bits 0..3): even index value
        - High nibble (bits 4..7): odd index value
        
    Args:
        tensor: Tensor with integer values in range [-8, 7] (signed) or [0, 15] (unsigned).
        signed: Whether the integers are signed [-8, 7] or unsigned [0, 15].
        
    Returns:
        packed: torch.uint8 tensor of length ceil(N / 2).
        original_numel: Original number of elements before padding.
        original_shape: Original tensor shape.
    """
    original_shape = tuple(tensor.shape)
    flat = tensor.detach().reshape(-1)
    original_numel = flat.numel()

    # Enforce integer range
    if signed:
        clamped = torch.clamp(flat.to(torch.int32), -8, 7)
        # Convert signed two's complement to 4-bit positive mask [0..15]
        nibbles = clamped & 0x0F
    else:
        clamped = torch.clamp(flat.to(torch.int32), 0, 15)
        nibbles = clamped & 0x0F

    # Pad with zero if length is odd
    if original_numel % 2 != 0:
        nibbles = torch.cat([nibbles, torch.tensor([0], dtype=torch.int32, device=nibbles.device)])

    low_nibbles = nibbles[0::2]
    high_nibbles = nibbles[1::2]

    # Combine: (high << 4) | low
    packed_values = (high_nibbles << 4) | low_nibbles
    packed_tensor = packed_values.to(torch.uint8)

    return packed_tensor, original_numel, original_shape


def unpack_int4(
    packed_tensor: torch.Tensor,
    original_numel: int,
    original_shape: Optional[Tuple[int, ...]] = None,
    signed: bool = True,
) -> torch.Tensor:
    """
    Unpacks a uint8 storage tensor into reconstructed 4-bit integer values.
    
    Performs sign-extension when signed=True to restore negative numbers from 2's complement.
    
    Args:
        packed_tensor: uint8 tensor containing packed pairs of 4-bit nibbles.
        original_numel: Exact count of original values.
        original_shape: Optional target shape to restore.
        signed: If True, sign extends values in [8..15] to [-8..-1].
        
    Returns:
        unpacked: torch.int8 tensor matching original_shape.
    """
    packed_flat = packed_tensor.reshape(-1).to(torch.int32)

    # Extract nibbles
    low_nibbles = packed_flat & 0x0F
    high_nibbles = (packed_flat >> 4) & 0x0F

    # Interleave low and high nibbles
    interleaved = torch.empty(packed_flat.numel() * 2, dtype=torch.int32, device=packed_tensor.device)
    interleaved[0::2] = low_nibbles
    interleaved[1::2] = high_nibbles

    # Slice off padding
    unpacked = interleaved[:original_numel]

    # Sign extension for 4-bit signed integers:
    # If the 4th bit is set (value >= 8), subtract 16 (e.g. 15 -> -1, 8 -> -8)
    if signed:
        unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)

    unpacked = unpacked.to(torch.int8)
    if original_shape is not None:
        unpacked = unpacked.reshape(original_shape)

    return unpacked


# ============================================================================
# Bit-Level Inspection and Disassembly
# ============================================================================

def float_to_bitstring(val: float, precision: PrecisionType) -> Dict[str, str]:
    """
    Deconstructs a floating-point scalar into sign, exponent, and mantissa bit patterns.
    """
    if precision == PrecisionType.FP32:
        packed = struct.pack(">f", float(val))
        bits = "".join(f"{b:08b}" for b in packed)
        return {
            "format": "FP32 (1-8-23)",
            "sign": bits[0],
            "exponent": bits[1:9],
            "mantissa": bits[9:32],
            "full_binary": f"{bits[0]}_{bits[1:9]}_{bits[9:32]}",
            "hex": packed.hex().upper(),
        }
    elif precision == PrecisionType.FP16:
        # Using numpy float16
        np_val = np.float16(val)
        raw_int = np_val.view(np.uint16)
        bits = f"{raw_int:016b}"
        return {
            "format": "FP16 (1-5-10)",
            "sign": bits[0],
            "exponent": bits[1:6],
            "mantissa": bits[6:16],
            "full_binary": f"{bits[0]}_{bits[1:6]}_{bits[6:16]}",
            "hex": f"{raw_int:04X}",
        }
    elif precision == PrecisionType.BF16:
        # BF16 is the high 16 bits of FP32
        packed = struct.pack(">f", float(val))
        bits = "".join(f"{b:08b}" for b in packed)[:16]
        return {
            "format": "BF16 (1-8-7)",
            "sign": bits[0],
            "exponent": bits[1:9],
            "mantissa": bits[9:16],
            "full_binary": f"{bits[0]}_{bits[1:9]}_{bits[9:16]}",
            "hex": packed[:2].hex().upper(),
        }
    elif precision == PrecisionType.INT8:
        int_val = int(round(val))
        int_val = max(-128, min(127, int_val))
        u8 = int_val & 0xFF
        bits = f"{u8:08b}"
        return {
            "format": "INT8 (Two's Complement)",
            "sign": bits[0],
            "exponent": "N/A",
            "mantissa": bits[1:8],
            "full_binary": bits,
            "hex": f"{u8:02X}",
        }
    elif precision == PrecisionType.INT4:
        int_val = int(round(val))
        int_val = max(-8, min(7, int_val))
        u4 = int_val & 0x0F
        bits = f"{u4:04b}"
        return {
            "format": "INT4 (4-bit Two's Complement)",
            "sign": bits[0],
            "exponent": "N/A",
            "mantissa": bits[1:4],
            "full_binary": bits,
            "hex": f"{u4:01X}",
        }
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def convert_precision(tensor: torch.Tensor, target_precision: PrecisionType) -> torch.Tensor:
    """
    Converts a tensor to the specified precision format.
    For INT4, values are rounded, clamped to [-8, 7], and returned as torch.int8.
    """
    if target_precision == PrecisionType.FP32:
        return tensor.to(torch.float32)
    elif target_precision == PrecisionType.FP16:
        return tensor.to(torch.float16)
    elif target_precision == PrecisionType.BF16:
        return tensor.to(torch.bfloat16)
    elif target_precision == PrecisionType.INT8:
        return torch.clamp(torch.round(tensor), -128, 127).to(torch.int8)
    elif target_precision == PrecisionType.INT4:
        return torch.clamp(torch.round(tensor), -8, 7).to(torch.int8)
    else:
        raise ValueError(f"Unknown precision: {target_precision}")


def get_dtype_memory_size(tensor: torch.Tensor, precision: PrecisionType) -> int:
    """
    Calculates actual memory footprint in bytes for a given tensor under the target precision.
    For INT4, each element uses exactly 0.5 bytes (ceil(numel / 2)).
    """
    numel = tensor.numel()
    if precision == PrecisionType.FP32:
        return numel * 4
    elif precision in (PrecisionType.FP16, PrecisionType.BF16):
        return numel * 2
    elif precision == PrecisionType.INT8:
        return numel * 1
    elif precision == PrecisionType.INT4:
        return (numel + 1) // 2  # 4 bits = 0.5 bytes, packed into bytes
    else:
        return tensor.element_size() * numel


def inspect_precision_profile(val: float) -> Dict[str, Dict[str, str]]:
    """
    Generates a cross-precision profile for a scalar value, comparing its bit patterns
    across FP32, FP16, BF16, INT8, and INT4.
    """
    return {p.value: float_to_bitstring(val, p) for p in PrecisionType}
