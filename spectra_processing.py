from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import torch
import numpy as np
from scipy.ndimage import uniform_filter1d

import typing as tp

def normalize_spectrum(B: torch.Tensor,
                       y: torch.Tensor,
                       mode: str = "integral") -> torch.Tensor:
    """Normalize a spectrum tensor y defined on field values B.

    Modes supported:
      - 'integral': integrate absolute values (Riemann) and divide
      - 'max': divide by max absolute value
      - None or 'none': return copy
    """
    if mode is None or mode == "none":
        return y.clone()
    step = float(B[1] - B[0]) if B.numel() > 1 else 1.0
    if mode == "max":
        denom = float(y.abs().max())
        if denom == 0:
            return y.clone()
        return y / denom
    if mode == "integral":
        denom = float((y.abs().sum() * step).item())
        if denom == 0:
            return y.clone()
        return y / denom
    raise ValueError(f"Unknown norm mode: {mode}")


def signal_to_amplitude(y_vals):
    """ Rotate the signal so that the mean imaginary part is zero, then compute amplitude. """

    img = np.imag(y_vals)
    real = np.real(y_vals)
    theta = np.arctan2(np.mean(img), np.mean(real))

    real_rot = real * np.cos(theta) + img * np.sin(theta)
    img_rot = -real * np.sin(theta) + img * np.cos(theta)

    return real_rot, img_rot


def create_baseline_mask(x_vals: np.ndarray,
                         baseline_areas: list[tuple[float, float]]):
    """
    Create a mask that includes only baseline regions

    Parameters:
    -----------
    x_vals : array
        X coordinates
    baseline_areas : list of tuples
        List of (x_start, x_end) tuples defining baseline regions to INCLUDE

    Returns:
    --------
    mask : array of bool
        True for baseline regions, False for peak regions
    """
    if not baseline_areas:
        return np.ones(len(x_vals), dtype=bool)
    mask = np.zeros(len(x_vals), dtype=bool)
    for x_start, x_end in baseline_areas:
        baseline_region = (x_vals >= x_start) & (x_vals <= x_end)
        mask = mask | baseline_region
    return mask


def correct_baseline_polynomial(x_vals: np.ndarray, y_vals: np.ndarray, mask: np.ndarray, poly_order: int):
    """
    Remove baseline by fitting polynomial to regions excluding the peak
    """

    coeffs = np.polyfit(x_vals[mask], y_vals[mask], poly_order)
    baseline = np.polyval(coeffs, x_vals)
    y_corrected = y_vals - baseline
    return y_corrected, baseline


def correct_baseline_saturation(y_vals: np.ndarray, sat_last_indexes: int):
    """
    Remove baseline by fitting polynomial to regions excluding the peak
    """
    satur_value = y_vals[-sat_last_indexes:].mean()
    baseline = np.ones_like(y_vals) * satur_value
    y_corrected = y_vals - satur_value
    return y_corrected, baseline


def correct_baseline_als(y_vals: np.ndarray, mask: np.ndarray, lam=1e6, p=0.01, niter=10):
    """
    ALS baseline with masking capability

    Parameters:
    -----------
    y : array
        Input signal
    mask : array of bool, optional
        True for points to INCLUDE in baseline fitting
        False for points to EXCLUDE (e.g., peak regions)
    lam : float
        Smoothness parameter
    p : float
        Asymmetry parameter
    niter : int
        Number of iterations
    """
    L = len(y_vals)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))

    w = np.ones(L)

    if mask is not None:
        w[~mask] = 1e-10

    for i in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y_vals)
        w_new = p * (y_vals > z) + (1 - p) * (y_vals < z)

        if mask is not None:
            w_new[~mask] = 1e-10

        w = w_new

    return y_vals - z, z


def _percentile_baseline(x_vals, y_vals, window_size=None, percentile=10,
                         proximity_threshold=0.15):
    """
    Detect baseline using local percentile analysis.
    Points close to local low percentile are likely baseline.
    """
    if window_size is None:
        window_size = len(y_vals) // 20

    baseline_mask = np.zeros(len(y_vals), dtype=bool)

    # Calculate local percentiles using sliding window
    half_window = window_size // 2
    for i in range(len(y_vals)):
        start = max(0, i - half_window)
        end = min(len(y_vals), i + half_window + 1)

        local_percentile = np.percentile(y_vals[start:end], percentile)

        # Check if current point is close to local percentile
        if y_vals[i] <= local_percentile * (1 + proximity_threshold):
            baseline_mask[i] = True

    return baseline_mask


def correct_baseline(x_vals: np.ndarray, y_vals: np.ndarray,
                     baseline_areas: tp.Optional[list[tuple[float, float]]] = None,
                     method="poly", poly_order=0, lam=2e7, p=0.05, niter=10, sat_last_indexes=10):
    if method is "saturation":
        y_corrected, baseline = correct_baseline_saturation(y_vals, sat_last_indexes=sat_last_indexes)
    else:
        if baseline_areas is None:
            mask = _percentile_baseline(x_vals, y_vals)
        else:
            mask = create_baseline_mask(x_vals, baseline_areas)
        if method == "poly":
            y_corrected, baseline = correct_baseline_polynomial(x_vals, y_vals, mask, poly_order=poly_order)
        elif method == "als":
            y_corrected, baseline = correct_baseline_als(y_vals, mask, lam=lam, p=p, niter=niter)
        else:
            raise ValueError("Wrong method. It must be 'poly' or 'als'")

    return y_corrected, baseline


