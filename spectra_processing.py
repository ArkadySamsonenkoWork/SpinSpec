from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

import numpy as np


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


def correct_baseline(x_vals: np.ndarray, y_vals: np.ndarray,
                     baseline_areas: list[tuple[float, float]],
                     method="poly", poly_order=0, lam=2e7, p=0.05, niter=10):
    mask = create_baseline_mask(x_vals, baseline_areas)
    if method == "poly":
        y_corrected, baseline = correct_baseline_polynomial(x_vals, y_vals, mask, poly_order=poly_order)
    elif method == "als":
        y_corrected, baseline = correct_baseline_als(y_vals, mask, lam=lam, p=p, niter=niter)
    else:
        raise ValueError("Wrong method. It must be 'poly' or 'als'")

    return y_corrected, baseline


