"""Pure-NumPy evaluation metrics for the RF-LEGO AE benchmark.

These functions take only NumPy arrays (model outputs, baseline outputs and the
ground truth stored in the benchmark ``.npz``) so that ``evaluate.py`` never has
to import the model package internals or the synthesized training-data builders.

Metric families:
- Frequency transform: PSLR and PAPR.
- Beamformer: angle MAE.
- Detector: exact-bin Detection Rate at a fixed, method-native score cutoff
  placed in the ``10^-3``-order FAR regime.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Frequency-transform metrics
# --------------------------------------------------------------------------- #
def _circular_distance(idx: np.ndarray, center: int, length: int) -> np.ndarray:
    d = np.abs(idx - center)
    return np.minimum(d, length - d)


def pslr_db(mag: np.ndarray, guard: int = 3) -> float:
    """Peak-to-side-lobe ratio in dB for a 1-D magnitude spectrum.

    The main lobe is excluded by a circular guard band of ``guard`` bins on each
    side of the global peak; the metric is peak over the largest remaining bin.
    """
    mag = np.asarray(mag, dtype=np.float64)
    n = mag.shape[-1]
    peak = int(np.argmax(mag))
    idx = np.arange(n)
    side = mag[_circular_distance(idx, peak, n) > guard]
    if side.size == 0 or mag[peak] <= 0:
        return float("nan")
    side_max = float(np.max(side))
    if side_max <= 0:
        return float("inf")
    return 20.0 * np.log10(mag[peak] / side_max)


def papr_db(mag: np.ndarray) -> float:
    """Peak-to-average power ratio in dB (peak power over mean power)."""
    p = np.asarray(mag, dtype=np.float64) ** 2
    mean_p = float(np.mean(p))
    if mean_p <= 0:
        return float("nan")
    return 10.0 * np.log10(float(np.max(p)) / mean_p)


def ft_metrics(mag_batch: np.ndarray, gt_peak_bin: np.ndarray, guard: int = 3) -> dict:
    """Mean PSLR / PAPR over a batch of spectra."""
    pslr = np.array([pslr_db(m, guard) for m in mag_batch], dtype=np.float64)
    papr = np.array([papr_db(m) for m in mag_batch], dtype=np.float64)
    return {
        "pslr_db": float(np.nanmean(pslr)),
        "papr_db": float(np.nanmean(papr)),
        "_pslr_per_sample": pslr,
        "_papr_per_sample": papr,
    }


# --------------------------------------------------------------------------- #
# Beamformer metrics
# --------------------------------------------------------------------------- #
def angle_mae(
    spectrum_batch: np.ndarray,
    gt_angles_deg: np.ndarray,
    n_targets: np.ndarray,
    angle_grid_deg: np.ndarray,
) -> float:
    """Mean absolute angle error (deg) over single-target samples only."""
    errs = []
    for spec, gt, k in zip(spectrum_batch, gt_angles_deg, n_targets):
        if int(k) != 1:
            continue
        est = angle_grid_deg[int(np.argmax(spec))]
        errs.append(abs(est - float(gt[0])))
    return float(np.mean(errs)) if errs else float("nan")


def beamformer_metrics(
    spectrum_batch: np.ndarray,
    gt: dict,
) -> dict:
    """Angle MAE for a batch of spectra."""
    grid = gt["angle_grid_deg"]
    return {
        "angle_mae_deg": angle_mae(spectrum_batch, gt["gt_angles_deg"], gt["n_targets"], grid),
    }


# --------------------------------------------------------------------------- #
# Detector operating-point metrics
# --------------------------------------------------------------------------- #
def fixed_threshold_op(stat: np.ndarray, label: np.ndarray, cutoff: float) -> dict:
    """Exact-bin Detection Rate at a fixed, label-independent score cutoff.

    The cutoff is supplied by the caller: an analytical CA-CFAR multiplier or a
    method-native RF-LEGO score cutoff.
    """
    stat = np.asarray(stat, dtype=np.float64).ravel()
    label = np.asarray(label).ravel()
    pos = stat[label > 0.5]
    if pos.size == 0:
        return {
            "dr": float("nan"),
            "cutoff": float(cutoff),
            "true_positives": 0,
            "positive_count": 0,
        }
    true_positives = int(np.sum(pos >= cutoff))
    return {
        "dr": float(true_positives / pos.size),
        "cutoff": float(cutoff),
        "true_positives": true_positives,
        "positive_count": int(pos.size),
    }
