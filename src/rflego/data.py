"""Synthesized training data builders and datasets for RF-LEGO modules.

This module provides the per-module synthesized sample builders used by the
online-training workflow and by the :class:`torch.utils.data.Dataset` wrappers.
The AE benchmark itself is a separate prepared real-world held-out dataset
loaded from ``ae_data`` by the evaluation script.

Three reproducible builders are provided, one per RF-LEGO task, each returning
NumPy arrays whose keys/shapes/dtypes match the corresponding model ``forward``:

- :func:`generate_frequency_transform_batch` -> learnable Bluestein FT
- :func:`generate_beamformer_batch` -> unfolded-ADMM direction-of-arrival
- :func:`generate_detector_batch` -> state-space peak detection

Reproducibility is governed by :func:`make_rng`, which derives an independent
``numpy`` random stream from a single entropy value (default ``seed=42``) and
module name.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "BaseDataset",
    "FrequencyTransformDataset",
    "DetectorDataset",
    "BeamformerDataset",
    "create_dataset",
    "make_rng",
    "generate_frequency_transform_batch",
    "generate_beamformer_batch",
    "generate_detector_batch",
]

# --------------------------------------------------------------------------- #
# Reproducible RNG streams
# --------------------------------------------------------------------------- #
MODULE_ID = {"ft": 0, "beamformer": 1, "detector": 2}
TRAIN_STREAM_ID = 0


def make_rng(module: str, seed: int = 42) -> np.random.Generator:
    """Return an independent NumPy RNG for synthesized training samples.

    Args:
        module: One of ``"ft"``, ``"beamformer"``, ``"detector"``.
        seed: Entropy value used to derive the stream.

    Returns:
        A ``numpy.random.Generator`` (PCG64) seeded by a ``SeedSequence`` child,
        so different modules never share a stream.
    """
    if module not in MODULE_ID:
        raise ValueError(f"Unknown module {module!r}; expected one of {list(MODULE_ID)}")
    ss = np.random.SeedSequence(entropy=seed, spawn_key=(TRAIN_STREAM_ID, MODULE_ID[module]))
    return np.random.default_rng(ss)


def _case_layout(n: int, n_cases: int, rng: np.random.Generator, balanced: bool) -> np.ndarray:
    """Build the internal per-sample condition vector.

    ``balanced`` produces contiguous blocks; otherwise conditions are drawn
    uniformly at random for training diversity.
    """
    if balanced:
        if n % n_cases != 0:
            raise ValueError(f"n={n} must be divisible by n_cases={n_cases} for a balanced layout")
        return np.repeat(np.arange(n_cases), n // n_cases).astype(np.int64)
    return rng.integers(0, n_cases, size=n).astype(np.int64)


def _cnoise(rng: np.random.Generator, shape, sigma) -> np.ndarray:
    """Circularly-symmetric complex Gaussian noise with E[|n|^2] = sigma^2."""
    sigma = np.asarray(sigma)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (sigma / np.sqrt(2.0)) * (real + 1j * imag)


def _gaussian_bump(centers, length: int, sigma: float = 1.0) -> np.ndarray:
    """Peak-normalized Gaussian bump(s) on a length-``length`` circular grid.

    Args:
        centers: Iterable of integer bin centers (ignored when < 0).
        length: Grid length.
        sigma: Gaussian standard deviation in bins.

    Returns:
        Float array of shape ``[length]`` with peak value 1.0.
    """
    idx = np.arange(length)
    out = np.zeros(length, dtype=np.float64)
    for c in np.atleast_1d(centers):
        if c < 0:
            continue
        d = np.minimum(np.abs(idx - c), length - np.abs(idx - c))  # circular distance
        out = np.maximum(out, np.exp(-0.5 * (d / sigma) ** 2))
    return out


# --------------------------------------------------------------------------- #
# Frequency-transform synthesized training samples
# --------------------------------------------------------------------------- #
def generate_frequency_transform_batch(
    n: int,
    rng: np.random.Generator,
    N: int = 256,
    balanced: bool = False,
    target_sigma: float = 1.0,
) -> dict[str, np.ndarray]:
    """Build synthesized frequency-transform training samples.

    Each sample is a length-``N`` complex signal containing a dominant off-grid
    tone (the target) plus noise, and -- in the clutter cases -- a weaker
    secondary tone.  The off-grid frequency makes a rectangular-window FFT leak,
    which is exactly what the learnable transform is asked to suppress.

    Returns model-ready inputs plus training targets and diagnostic fields.
    AE metrics are computed by ``evaluate.py`` on the real-world held-out
    benchmark, not on these synthesized training samples.
    """
    case = _case_layout(n, 5, rng, balanced)
    x = np.zeros((n, N), dtype=np.complex128)
    gt_peak_bin = np.zeros(n, dtype=np.int64)
    gt_freq = np.zeros(n, dtype=np.float64)
    gt_tone_bins = np.full((n, 3), -1, dtype=np.int64)
    n_tones = np.ones(n, dtype=np.int64)
    snr_db = np.zeros(n, dtype=np.float64)
    nn = np.arange(N)

    # SNR (dB) per condition and frequency sub-band.
    snr_by_case = {0: 20.0, 1: 3.0, 2: 10.0, 3: 20.0, 4: 3.0}
    # range band ~ lower bins, doppler band ~ upper bins
    band_by_case = {
        0: (0.10, 0.42),
        1: (0.10, 0.42),
        2: (0.10, 0.42),
        3: (0.55, 0.88),
        4: (0.55, 0.88),
    }

    for i in range(n):
        c = int(case[i])
        snr = snr_by_case[c] + rng.uniform(-1.0, 1.0)
        snr_db[i] = snr
        lo, hi = band_by_case[c]
        f0 = rng.uniform(lo * N, hi * N) + rng.uniform(-0.5, 0.5)  # off-grid bin
        phi = rng.uniform(0.0, 2 * np.pi)
        a = 1.0
        sig = a * np.exp(1j * (2 * np.pi * f0 * nn / N + phi))
        peak_bin = int(round(f0)) % N
        gt_freq[i] = f0
        gt_peak_bin[i] = peak_bin
        gt_tone_bins[i, 0] = peak_bin

        if c == 2:  # clutter: weaker secondary tone elsewhere in the band
            f1 = rng.uniform(lo * N, hi * N) + rng.uniform(-0.5, 0.5)
            while abs(f1 - f0) < 6:
                f1 = rng.uniform(lo * N, hi * N) + rng.uniform(-0.5, 0.5)
            a1 = rng.uniform(0.3, 0.5)
            sig = sig + a1 * np.exp(1j * (2 * np.pi * f1 * nn / N + rng.uniform(0, 2 * np.pi)))
            gt_tone_bins[i, 1] = int(round(f1)) % N
            n_tones[i] = 2

        sigma = a / np.sqrt(10 ** (snr / 10.0))
        x[i] = sig + _cnoise(rng, N, sigma)

    gt_spectrum = np.stack(
        [_gaussian_bump(gt_peak_bin[i], N, sigma=target_sigma) for i in range(n)]
    ).astype(np.float32)

    return {
        "x_real": x.real.astype(np.float32),
        "x_imag": x.imag.astype(np.float32),
        "gt_peak_bin": gt_peak_bin,
        "gt_freq": gt_freq.astype(np.float32),
        "gt_spectrum": gt_spectrum,
        "gt_tone_bins": gt_tone_bins,
        "n_tones": n_tones,
        "snr_db": snr_db.astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# Beamformer synthesized training samples
# --------------------------------------------------------------------------- #
def _steering_matrix(angles_deg: np.ndarray, M: int) -> np.ndarray:
    """Half-wavelength ULA steering matrix A[m, d] = exp(-j*pi*m*sin(theta_d))."""
    m = np.arange(M)[:, None]
    theta = np.deg2rad(angles_deg)[None, :]
    return np.exp(-1j * np.pi * m * np.sin(theta)).astype(np.complex128)


def generate_beamformer_batch(
    n: int,
    rng: np.random.Generator,
    M: int = 8,
    D: int = 121,
    angle_min: float = -60.0,
    angle_max: float = 60.0,
    balanced: bool = False,
    target_sigma: float = 1.0,
    profile: dict | None = None,
) -> dict[str, np.ndarray]:
    """Build synthesized single-source ULA training samples for sparse-DoA recovery.

    Each sample has one target across five SNR / clutter conditions for
    adaptation during online training. ``profile`` optionally overrides the
    per-case SNR / clutter settings (keys ``snr``, ``n_clutter``,
    ``clutter_amp``).

    Returns native complex measurements/dictionary plus training targets and
    diagnostic fields. AE metrics are computed by ``evaluate.py`` on the
    real-world held-out benchmark, not on these synthesized training samples.
    """
    case = _case_layout(n, 5, rng, balanced)
    angle_grid = np.linspace(angle_min, angle_max, D).astype(np.float64)
    A_dict = _steering_matrix(angle_grid, M)  # shared geometry [M, D]

    y_meas = np.zeros((n, M), dtype=np.complex128)
    A_out = np.zeros((n, M, D), dtype=np.complex128)
    gt_angle_idx = np.full((n, 2), -1, dtype=np.int64)
    gt_angles_deg = np.full((n, 2), np.nan, dtype=np.float64)
    n_targets = np.zeros(n, dtype=np.int64)
    snr_db = np.zeros(n, dtype=np.float64)

    # Single-source DoA in cluttered/low-SNR regimes across five graded conditions.
    # The fixed-regularization LASSO Beamformer is evaluated under the same
    # conditions as the learned solver; grading varies clutter density/amplitude.
    profile = profile or {}
    snr_by_case = profile.get("snr", {0: 4.0, 1: 3.0, 2: 3.0, 3: 2.0, 4: 2.0})
    n_clutter_by_case = profile.get("n_clutter", {0: 3, 1: 4, 2: 4, 3: 4, 4: 5})
    clutter_amp_by_case = profile.get(
        "clutter_amp",
        {0: (0.22, 0.36), 1: (0.24, 0.38), 2: (0.24, 0.38), 3: (0.26, 0.40), 4: (0.28, 0.42)},
    )

    def steer(theta_deg: float) -> np.ndarray:
        return np.exp(-1j * np.pi * np.arange(M) * np.sin(np.deg2rad(theta_deg)))

    for i in range(n):
        c = int(case[i])
        snr = snr_by_case[c] + rng.uniform(-1.0, 1.0)
        snr_db[i] = snr
        sigma = 1.0 / np.sqrt(10 ** (snr / 10.0))  # unit-amplitude target -> noise std

        # one target at a grid angle (+ sub-grid jitter)
        d = int(rng.integers(5, D - 5))
        theta = angle_grid[d] + rng.uniform(-0.4, 0.4)
        y = np.exp(1j * rng.uniform(0, 2 * np.pi)) * steer(theta)
        gt_angle_idx[i, 0] = d
        gt_angles_deg[i, 0] = theta
        n_targets[i] = 1

        # diffuse spatial clutter (weak sources spread across the field of view)
        lo, hi = clutter_amp_by_case.get(c, (0.0, 0.0))
        for _ in range(n_clutter_by_case[c]):
            ct = rng.uniform(angle_min, angle_max)
            y = y + rng.uniform(lo, hi) * np.exp(1j * rng.uniform(0, 2 * np.pi)) * steer(ct)

        y_meas[i] = y + _cnoise(rng, M, sigma)
        A_out[i] = A_dict

    gt_spectrum = np.zeros((n, D), dtype=np.float32)
    for i in range(n):
        gt_spectrum[i] = _gaussian_bump(gt_angle_idx[i], D, sigma=target_sigma)

    return {
        "y_meas": y_meas.astype(np.complex64),
        "A_dict": A_out.astype(np.complex64),
        "gt_angle_idx": gt_angle_idx,
        "gt_angles_deg": gt_angles_deg.astype(np.float32),
        "gt_spectrum": gt_spectrum,
        "angle_grid_deg": angle_grid.astype(np.float32),
        "snr_db": snr_db.astype(np.float32),
        "num_elements": np.int64(M),
    }


# --------------------------------------------------------------------------- #
# Detector synthesized training samples
# --------------------------------------------------------------------------- #
def generate_detector_batch(
    n: int,
    rng: np.random.Generator,
    L: int = 128,
    balanced: bool = False,
) -> dict[str, np.ndarray]:
    """Build synthesized range-profile detection training samples.

    Targets are 3-bin Rician bumps; condition patterns cover clutter edges,
    mutual masking, and single-bin interference. Each profile is min-max
    normalized to ``[0, 1]``, matching the original Detector generator and the
    model-ready held-out Detector input contract.
    """
    case = _case_layout(n, 5, rng, balanced)
    x = np.zeros((L, n), dtype=np.float64)
    gt_mask = np.zeros((L, n), dtype=np.float64)
    target_bins = np.full((n, 5), -1, dtype=np.int64)
    clutter_bounds = np.full((n, 2), -1, dtype=np.int64)
    n_targets = np.zeros(n, dtype=np.int64)
    snr_db = np.zeros(n, dtype=np.float64)

    snr_by_case = {0: 18.0, 1: 12.0, 2: 15.0, 3: 15.0, 4: 15.0}

    def add_target(prof: np.ndarray, b: int, snr_lin: float, sigma_local: float) -> None:
        amp = np.sqrt(snr_lin) * sigma_local
        for off, scale in ((0, 1.0), (-1, 0.7), (1, 0.7)):
            bl = b + off
            if 0 <= bl < L:
                prof[bl] += amp * scale * np.exp(1j * rng.uniform(0, 2 * np.pi))

    for i in range(n):
        c = int(case[i])
        snr = snr_by_case[c] + rng.uniform(-1.0, 1.0)
        snr_db[i] = snr
        snr_lin = 10 ** (snr / 10.0)

        sigma_l = np.ones(L)  # per-bin noise std (clutter raises a band)
        if c == 2:
            l0 = int(rng.integers(int(0.45 * L), int(0.65 * L)))
            sigma_l[l0:] = rng.uniform(2.0, 3.0)
            clutter_bounds[i] = (l0, L)

        prof = _cnoise(rng, L, sigma_l)  # complex range profile

        if c == 3:  # closely-spaced targets (mutual masking)
            b0 = int(rng.integers(20, L - 30))
            sep = int(rng.integers(3, 6))
            centers = [b0, min(b0 + sep, L - 2)]
        elif c == 2:  # one target placed just inside the clutter edge
            centers = [min(clutter_bounds[i, 0] + int(rng.integers(2, 6)), L - 2)]
        else:
            k = int(rng.integers(1, 6)) if c in (0, 1) else 1  # homogeneous: 1-5 targets
            centers = sorted(int(rng.integers(8, L - 8)) for _ in range(k))

        for j, b in enumerate(centers[:5]):
            local_sigma = sigma_l[b]
            add_target(prof, b, snr_lin, local_sigma)
            target_bins[i, j] = b
            for off in (-1, 0, 1):
                if 0 <= b + off < L:
                    gt_mask[b + off, i] = 1.0
        n_targets[i] = len(centers)

        if c == 4:  # single-bin interference spikes that must NOT be flagged
            for _ in range(int(rng.integers(2, 4))):
                bs = int(rng.integers(5, L - 5))
                if gt_mask[bs, i] == 0:
                    prof[bs] += rng.uniform(3.0, 5.0) * np.exp(1j * rng.uniform(0, 2 * np.pi))

        env = np.abs(prof)
        env_min = env.min()
        env_span = env.max() - env_min
        x[:, i] = (env - env_min) / env_span if env_span > 0.0 else 0.0

    return {
        "x": x.astype(np.float32),
        "gt_mask": gt_mask.astype(np.float32),
        "target_bins": target_bins,
        "snr_db": snr_db.astype(np.float32),
        "seq_len": np.int64(L),
    }


# --------------------------------------------------------------------------- #
# Dataset wrappers
# --------------------------------------------------------------------------- #
class BaseDataset(Dataset):
    """In-memory dataset over precomputed synthesized training samples.

    Subclasses set ``self.data`` (a dict of NumPy arrays from a builder) and
    implement :meth:`__getitem__` to return forward-ready tensors.
    """

    def __init__(self, data: dict[str, np.ndarray], n: int) -> None:
        self.data = data
        self._n = int(n)

    def __len__(self) -> int:
        return self._n


class FrequencyTransformDataset(BaseDataset):
    """Frequency-transform samples as ``{x_real, x_imag, target, gt_peak_bin}``."""

    def __init__(self, n: int, seed: int = 42, split: str = "train", N: int = 256, **kw) -> None:
        rng = make_rng("ft", seed)
        data = generate_frequency_transform_batch(n, rng, N=N, **kw)
        super().__init__(data, n)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        d = self.data
        return {
            "x_real": torch.from_numpy(d["x_real"][i]),
            "x_imag": torch.from_numpy(d["x_imag"][i]),
            "target": torch.from_numpy(d["gt_spectrum"][i]),
            "gt_peak_bin": torch.tensor(d["gt_peak_bin"][i]),
        }


class BeamformerDataset(BaseDataset):
    """Beamformer samples as ``{y, A, target, gt_angle_idx}``."""

    def __init__(self, n: int, seed: int = 42, split: str = "train", M: int = 16, **kw) -> None:
        rng = make_rng("beamformer", seed)
        data = generate_beamformer_batch(n, rng, M=M, **kw)
        super().__init__(data, n)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        d = self.data
        return {
            "y": torch.from_numpy(d["y_meas"][i]),
            "A": torch.from_numpy(d["A_dict"][i]),
            "target": torch.from_numpy(d["gt_spectrum"][i]),
            "gt_angle_idx": torch.from_numpy(d["gt_angle_idx"][i]),
        }


class DetectorDataset(BaseDataset):
    """Detector samples as ``{x[L], mask[L]}`` (batched -> transpose to ``[L, B]``)."""

    def __init__(self, n: int, seed: int = 42, split: str = "train", L: int = 256, **kw) -> None:
        rng = make_rng("detector", seed)
        data = generate_detector_batch(n, rng, L=L, **kw)
        super().__init__(data, n)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        d = self.data
        return {
            "x": torch.from_numpy(d["x"][:, i]),
            "mask": torch.from_numpy(d["gt_mask"][:, i]),
        }


_DATASETS = {
    "ft": FrequencyTransformDataset,
    "beamformer": BeamformerDataset,
    "detector": DetectorDataset,
}


def create_dataset(
    task: str, n: int, seed: int = 42, split: str = "train", **kwargs
) -> BaseDataset:
    """Factory for the per-task datasets.

    Args:
        task: One of ``"ft"``, ``"beamformer"``, ``"detector"``.
        n: Number of samples.
        seed: Random seed.
        split: Backward-compatible argument; sample construction does not branch on it.
        **kwargs: Forwarded to the underlying synthesized sample builder.

    Returns:
        The constructed dataset.
    """
    if task not in _DATASETS:
        raise ValueError(f"Unknown task {task!r}; expected one of {list(_DATASETS)}")
    return _DATASETS[task](n=n, seed=seed, split=split, **kwargs)
