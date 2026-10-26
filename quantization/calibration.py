"""
Clipping and Calibration Strategies for Post-Training Quantization (PTQ).

Implements:
1. MinMax Calibration (baseline, full dynamic range, sensitive to outliers)
2. Percentile / Quantile Clipping (e.g. 99.9%, 99.99% for suppressing outlier tails)
3. MSE-Optimal Grid Search (L2 reconstruction error minimization over thresholds)
4. KL-Divergence / Relative Entropy Calibration (TensorRT-style information loss minimization)
"""

from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .ptq import QuantizationScheme, calculate_scale_and_zeropoint, get_quant_range


class CalibrationMethod(str, Enum):
    """Supported calibration and clipping algorithms."""
    MINMAX = "minmax"
    PERCENTILE = "percentile"
    MSE = "mse"
    KL_DIVERGENCE = "kl_divergence"


class Calibrator:
    """
    Executes calibration algorithms to compute optimal dynamic range boundaries [min_val, max_val]
    for post-training quantization.
    """

    @staticmethod
    def calibrate_minmax(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MinMax calibration: Uses the absolute extrema of the tensor.
        Fastest, but vulnerable to extreme outliers stretching the quantization grid.
        """
        return tensor.min(), tensor.max()

    @staticmethod
    def calibrate_percentile(
        tensor: torch.Tensor,
        percentile: float = 99.99,
        scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Percentile calibration: Truncates the top (100 - percentile)% and bottom (100 - percentile)% outliers.
        
        Args:
            tensor: Input tensor.
            percentile: Upper percentile boundary (e.g. 99.9 or 99.99).
            scheme: SYMMETRIC enforces equal bounds around 0.
        """
        flat = tensor.detach().reshape(-1).float()
        
        if scheme == QuantizationScheme.SYMMETRIC:
            abs_flat = flat.abs()
            k = int(round(percentile / 100.0 * abs_flat.numel()))
            k = max(1, min(abs_flat.numel(), k))
            # Use torch.kthvalue for fast deterministic percentile
            threshold = torch.kthvalue(abs_flat, k).values
            return -threshold, threshold
        else:
            lower_k = max(1, int(round((100.0 - percentile) / 100.0 * flat.numel())))
            upper_k = min(flat.numel(), int(round(percentile / 100.0 * flat.numel())))
            min_val = torch.kthvalue(flat, lower_k).values
            max_val = torch.kthvalue(flat, upper_k).values
            return min_val, max_val

    @staticmethod
    def calibrate_mse(
        tensor: torch.Tensor,
        bit_width: int = 8,
        scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC,
        num_candidates: int = 40,
        min_percentile: float = 95.0,
        max_percentile: float = 100.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MSE (L2-norm) optimal clipping:
        Performs a grid search over candidate clipping ranges to directly minimize
        reconstruction error: argmin_alpha || x - dequant(quant(x, alpha)) ||^2
        """
        flat = tensor.detach().reshape(-1).float()
        q_min, q_max = get_quant_range(bit_width, scheme, signed=True)

        # Sample if tensor is huge to ensure fast real-time calibration
        if flat.numel() > 100_000:
            indices = torch.randperm(flat.numel())[:100_000]
            flat_sample = flat[indices]
        else:
            flat_sample = flat

        best_mse = float("inf")
        best_min = flat_sample.min()
        best_max = flat_sample.max()

        if scheme == QuantizationScheme.SYMMETRIC:
            abs_max_orig = flat_sample.abs().max().item()
            if abs_max_orig < 1e-8:
                return flat_sample.min(), flat_sample.max()

            # Sweep scale multipliers from 0.7 to 1.0 of max absolute value
            multipliers = torch.linspace(min_percentile / 100.0, max_percentile / 100.0, num_candidates)
            for m in multipliers:
                cand_max = abs_max_orig * m.item()
                cand_min = -cand_max

                scale = cand_max / float(q_max)
                q = torch.clamp(torch.round(flat_sample / scale), q_min, q_max)
                x_rec = q * scale
                mse = torch.mean((flat_sample - x_rec) ** 2).item()

                if mse < best_mse:
                    best_mse = mse
                    best_min = torch.tensor(cand_min, device=tensor.device, dtype=tensor.dtype)
                    best_max = torch.tensor(cand_max, device=tensor.device, dtype=tensor.dtype)

        else:
            orig_min = flat_sample.min().item()
            orig_max = flat_sample.max().item()
            if abs(orig_max - orig_min) < 1e-8:
                return flat_sample.min(), flat_sample.max()

            # Sweep shrink factors symmetrically inwards
            shrink_factors = torch.linspace(0.0, 0.1, num_candidates)
            for s in shrink_factors:
                delta = (orig_max - orig_min) * s.item()
                cand_min = orig_min + delta
                cand_max = orig_max - delta

                scale = (cand_max - cand_min) / float(q_max - q_min)
                scale = max(scale, 1e-8)
                zp = torch.clamp(torch.round(torch.tensor(-cand_min / scale)) + q_min, q_min, q_max)
                q = torch.clamp(torch.round(flat_sample / scale) + zp, q_min, q_max)
                x_rec = (q - zp) * scale
                mse = torch.mean((flat_sample - x_rec) ** 2).item()

                if mse < best_mse:
                    best_mse = mse
                    best_min = torch.tensor(cand_min, device=tensor.device, dtype=tensor.dtype)
                    best_max = torch.tensor(cand_max, device=tensor.device, dtype=tensor.dtype)

        return best_min, best_max

    @staticmethod
    def calibrate_kl_divergence(
        tensor: torch.Tensor,
        bit_width: int = 8,
        num_histogram_bins: int = 2048,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Kullback-Leibler (KL) Divergence Calibration (TensorRT style):
        Finds the saturation threshold that minimizes relative entropy (information loss)
        between the original unquantized histogram distribution and the quantized distribution.
        """
        flat = tensor.detach().reshape(-1).float().abs()
        max_val = flat.max().item()

        if max_val < 1e-8:
            return tensor.min(), tensor.max()

        target_quant_bins = 1 << (bit_width - 1)  # 128 for INT8, 8 for INT4

        # 1. Compute fine-grained reference histogram
        hist = torch.histc(flat, bins=num_histogram_bins, min=0.0, max=max_val)
        hist = hist.cpu().numpy()

        bin_width = max_val / num_histogram_bins
        best_kl = float("inf")
        best_threshold_bin = num_histogram_bins

        # 2. Iterate over candidate saturation thresholds T from target_quant_bins to num_histogram_bins
        step = max(1, (num_histogram_bins - target_quant_bins) // 60)
        for t in range(target_quant_bins, num_histogram_bins, step):
            # Reference distribution P truncated at t
            p = hist[:t].copy()
            # Collapse outlier tail into the last bin (saturation)
            p[-1] += np.sum(hist[t:])

            # Skip trivial distributions
            if np.sum(p) == 0:
                continue

            # Quantize P into target_quant_bins
            q_bins = np.zeros(target_quant_bins, dtype=np.float64)
            bin_size = t / float(target_quant_bins)
            for j in range(target_quant_bins):
                start = j * bin_size
                end = (j + 1) * bin_size
                left_idx = int(np.floor(start))
                right_idx = int(np.ceil(end))
                q_bins[j] = np.sum(p[left_idx:min(right_idx, t)])

            # Expand Q back to length t to compare with P
            q_expanded = np.zeros(t, dtype=np.float64)
            for j in range(target_quant_bins):
                start = j * bin_size
                end = (j + 1) * bin_size
                left_idx = int(np.floor(start))
                right_idx = int(np.ceil(end))
                count = max(1, min(right_idx, t) - left_idx)
                q_expanded[left_idx:min(right_idx, t)] = q_bins[j] / count

            # Normalize to valid probability distributions
            p_prob = p / np.sum(p)
            q_prob = q_expanded / np.sum(q_expanded)

            # Avoid division by zero in log: smooth zero bins
            eps = 1e-12
            p_prob = np.clip(p_prob, eps, 1.0)
            q_prob = np.clip(q_prob, eps, 1.0)

            # Compute KL Divergence: sum(P * log(P / Q))
            kl_div = np.sum(p_prob * np.log(p_prob / q_prob))

            if kl_div < best_kl:
                best_kl = kl_div
                best_threshold_bin = t

        optimal_threshold = (best_threshold_bin + 0.5) * bin_width
        thresh_t = torch.tensor(optimal_threshold, device=tensor.device, dtype=tensor.dtype)
        return -thresh_t, thresh_t


def calibrate_tensor(
    tensor: torch.Tensor,
    method: CalibrationMethod = CalibrationMethod.MINMAX,
    bit_width: int = 8,
    scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC,
    percentile: float = 99.99,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified entry point for tensor calibration.
    """
    if method == CalibrationMethod.MINMAX:
        return Calibrator.calibrate_minmax(tensor)
    elif method == CalibrationMethod.PERCENTILE:
        return Calibrator.calibrate_percentile(tensor, percentile=percentile, scheme=scheme)
    elif method == CalibrationMethod.MSE:
        return Calibrator.calibrate_mse(tensor, bit_width=bit_width, scheme=scheme)
    elif method == CalibrationMethod.KL_DIVERGENCE:
        return Calibrator.calibrate_kl_divergence(tensor, bit_width=bit_width)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
