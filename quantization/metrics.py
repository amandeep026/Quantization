"""
Fidelity, Distortion, and Memory Efficiency Metrics.

Implements:
1. Signal-to-Quantization-Noise Ratio (SQNR in dB)
2. Mean Squared Error (MSE) & Mean Absolute Error (MAE)
3. Maximum Absolute Distortion
4. Cosine Similarity (directional alignment of weight/activation vectors)
5. Memory Reduction & Compression Ratio
6. ASCII / Markdown Comparison Table Generator
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Union

import torch


@dataclass
class QuantizationReport:
    """Consolidated metrics summarizing quantization fidelity and efficiency."""
    name: str
    bit_width: int
    original_bytes: int
    quantized_bytes: int
    compression_ratio: float
    sqnr_db: float
    mse: float
    mae: float
    max_error: float
    cosine_similarity: float


def compute_sqnr(original: torch.Tensor, reconstructed: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes Signal-to-Quantization-Noise Ratio (SQNR) in decibels (dB).
    Formula: 10 * log10( sum(x^2) / sum((x - x_hat)^2) )
    """
    orig = original.detach().float()
    rec = reconstructed.detach().float()
    signal_power = torch.sum(orig ** 2).item()
    noise_power = torch.sum((orig - rec) ** 2).item()
    if noise_power < eps:
        return 999.0  # Perfect reconstruction
    if signal_power < eps:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


def compute_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    original_bytes: int,
    quantized_bytes: int,
    name: str = "Quantization Test",
    bit_width: int = 8,
) -> QuantizationReport:
    """
    Computes all standard mathematical distortion and efficiency metrics.
    """
    orig_f = original.detach().reshape(-1).float()
    rec_f = reconstructed.detach().reshape(-1).float()

    diff = orig_f - rec_f
    mse = torch.mean(diff ** 2).item()
    mae = torch.mean(diff.abs()).item()
    max_err = torch.max(diff.abs()).item()

    # Cosine Similarity
    norm_orig = torch.norm(orig_f, p=2)
    norm_rec = torch.norm(rec_f, p=2)
    if norm_orig > 1e-8 and norm_rec > 1e-8:
        cos_sim = (torch.dot(orig_f, rec_f) / (norm_orig * norm_rec)).item()
    else:
        cos_sim = 1.0

    sqnr = compute_sqnr(orig_f, rec_f)
    compression_ratio = original_bytes / max(1, quantized_bytes)

    return QuantizationReport(
        name=name,
        bit_width=bit_width,
        original_bytes=original_bytes,
        quantized_bytes=quantized_bytes,
        compression_ratio=compression_ratio,
        sqnr_db=sqnr,
        mse=mse,
        mae=mae,
        max_error=max_err,
        cosine_similarity=cos_sim,
    )


def print_comparison_table(reports: List[QuantizationReport], title: str = "Quantization Benchmark"):
    """
    Renders a formatted ASCII table comparing multiple quantization configurations.
    """
    headers = ["Configuration", "Bits", "Orig Bytes", "Quant Bytes", "Comp Ratio", "SQNR (dB)", "MSE", "Cos Sim"]
    col_widths = [26, 6, 12, 13, 12, 11, 12, 10]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)

    print(f"\n{'=' * len(header_line)}")
    print(f" {title} ")
    print(f"{'=' * len(header_line)}")
    print(header_line)
    print(separator)

    for r in reports:
        row = [
            r.name[:col_widths[0]].ljust(col_widths[0]),
            str(r.bit_width).ljust(col_widths[1]),
            f"{r.original_bytes:,}".rjust(col_widths[2]),
            f"{r.quantized_bytes:,}".rjust(col_widths[3]),
            f"{r.compression_ratio:.2f}x".rjust(col_widths[4]),
            f"{r.sqnr_db:.2f} dB".rjust(col_widths[5]),
            f"{r.mse:.2e}".rjust(col_widths[6]),
            f"{r.cosine_similarity:.6f}".rjust(col_widths[7]),
        ]
        print(" | ".join(row))

    print(f"{'=' * len(header_line)}\n")
