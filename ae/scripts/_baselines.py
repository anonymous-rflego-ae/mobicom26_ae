"""Pure-NumPy classical baselines for the RF-LEGO AE benchmark.

- Frequency transform: Bluestein FFT.
- Beamformer: LASSO Beamformer sparse-DoA solver.
- Detector: CA/GO/SO/OS-CFAR.
"""

from __future__ import annotations

import numpy as np


# Fixed CA-CFAR configuration used by the exact-bin evaluation.
CA_CFAR_N_TRAIN = 4
CA_CFAR_N_GUARD = 2
CA_CFAR_NOMINAL_PFA = 1e-3


# --------------------------------------------------------------------------- #
# Frequency transform
# --------------------------------------------------------------------------- #
def bluestein_fft(x: np.ndarray) -> np.ndarray:
    """Bluestein FFT of ``x`` along the last axis.

    Classical frequency-transform baseline. Bluestein expresses the transform
    as a convolution, ``X_k = w_k * sum_n (x_n w_n) conj(w_{k-n})`` with chirp
    ``w_n = exp(-j*pi*n^2/N)``; the convolution is done by FFT on a length
    ``>= 2N-1`` grid. It supports any ``N`` (no power-of-two requirement), so
    its magnitude matches a full-band frequency transform.

    Args:
        x: Complex signal of shape ``[..., N]``.

    Returns:
        Complex Bluestein FFT spectrum of shape ``[..., N]`` (complex128).
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[-1]
    n = np.arange(N)
    w = np.exp(-1j * np.pi * (n * n) / N)  # chirp
    a = x * w
    L = int(2 ** np.ceil(np.log2(2 * N - 1)))  # convolution length (power of two)
    b = np.zeros(L, dtype=np.complex128)
    b[:N] = np.conj(w)
    b[L - N + 1 :] = np.conj(w[1:])[::-1]  # symmetric tail for circular conv
    conv = np.fft.ifft(np.fft.fft(a, L, axis=-1) * np.fft.fft(b), axis=-1)[..., :N]
    return w * conv


def fft_magnitude(x_real: np.ndarray, x_imag: np.ndarray) -> np.ndarray:
    """Bluestein FFT magnitude spectrum ``|FFT(x_real + j x_imag)|``.

    Args:
        x_real, x_imag: Real/imag parts, shape ``[..., N]``.

    Returns:
        Magnitude spectrum of shape ``[..., N]`` (float64).
    """
    x = np.asarray(x_real, dtype=np.float64) + 1j * np.asarray(x_imag, dtype=np.float64)
    return np.abs(bluestein_fft(x))


# --------------------------------------------------------------------------- #
# Beamformer
# --------------------------------------------------------------------------- #
def _csoft(v: np.ndarray, t: float) -> np.ndarray:
    """Complex soft-thresholding (proximal operator of the L1 norm)."""
    mag = np.abs(v)
    scale = np.maximum(mag - t, 0.0) / (mag + 1e-12)
    return v * scale


def lasso_beamformer(y, A, n_iter=8, lam=0.1, step=0.01):
    """Classical complex LASSO Beamformer sparse-DoA solver.

    LASSO Beamformer baseline: ``min_z ||y-Az||^2 + lam||z||_1`` with a single
    fixed regularization, step size, and iteration count. The AE evaluation
    script passes the configured values from ``ae/configs/beamformer.yaml``.

    Args:
        y: Measurements ``[M]`` (complex).
        A: Steering dictionary ``[M, D]`` (complex).
        n_iter: Solver iterations.
        lam: L1 regularization weight.
        step: Gradient step size.

    Returns:
        Magnitude of the recovered sparse spectrum, shape ``[D]``.
    """
    y = np.asarray(y)
    A = np.asarray(A)
    Ah = A.conj().T
    z = np.zeros(A.shape[1], dtype=np.complex128)
    for _ in range(n_iter):
        z = _csoft(z + step * (Ah @ (y - A @ z)), lam * step)
    return np.abs(z)


def lasso_beamformer_batch(y, A, **kw) -> np.ndarray:
    """Apply :func:`lasso_beamformer` over a batch ``y[B, M]``, ``A[B, M, D]``."""
    return np.stack([lasso_beamformer(y[i], A[i], **kw) for i in range(y.shape[0])])


# --------------------------------------------------------------------------- #
# Detector (CFAR)
# --------------------------------------------------------------------------- #
def _cfar_windows(power: np.ndarray, n_train: int, n_guard: int):
    """Yield (cut_index, left-cells, right-cells) training windows with reflected edges."""
    L = power.shape[-1]
    half = n_train + n_guard
    padded = np.pad(power, (half, half), mode="reflect")
    for i in range(L):
        c = i + half
        left = padded[c - half : c - n_guard]
        right = padded[c + n_guard + 1 : c + half + 1]
        yield i, left, right


def ca_cfar(
    x: np.ndarray,
    n_train: int = CA_CFAR_N_TRAIN,
    n_guard: int = CA_CFAR_N_GUARD,
) -> np.ndarray:
    """Cell-averaging CFAR: ``CUT power / mean(training-cell power)``."""
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    if n_guard < 0:
        raise ValueError("n_guard must be non-negative")
    power = np.asarray(x, dtype=np.float64) ** 2
    stat = np.zeros_like(power)
    for i, lft, rgt in _cfar_windows(power, n_train, n_guard):
        training = np.concatenate([lft, rgt])
        stat[i] = power[i] / (np.mean(training) + 1e-12)
    return stat


def os_cfar(
    x: np.ndarray,
    n_train: int = CA_CFAR_N_TRAIN,
    n_guard: int = CA_CFAR_N_GUARD,
    k_frac: float = 0.75,
) -> np.ndarray:
    """Ordered-statistic CFAR using the ``k_frac`` training-cell quantile."""
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    if n_guard < 0:
        raise ValueError("n_guard must be non-negative")
    if not 0.0 < k_frac <= 1.0:
        raise ValueError("k_frac must be in (0, 1]")
    power = np.asarray(x, dtype=np.float64) ** 2
    stat = np.zeros_like(power)
    for i, lft, rgt in _cfar_windows(power, n_train, n_guard):
        training = np.sort(np.concatenate([lft, rgt]))
        k = min(training.size - 1, int(np.ceil(k_frac * training.size)) - 1)
        stat[i] = power[i] / (training[k] + 1e-12)
    return stat


def go_cfar(
    x: np.ndarray,
    n_train: int = CA_CFAR_N_TRAIN,
    n_guard: int = CA_CFAR_N_GUARD,
) -> np.ndarray:
    """Greatest-of CFAR using the larger left/right training-window mean."""
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    if n_guard < 0:
        raise ValueError("n_guard must be non-negative")
    power = np.asarray(x, dtype=np.float64) ** 2
    stat = np.zeros_like(power)
    for i, lft, rgt in _cfar_windows(power, n_train, n_guard):
        noise = max(float(np.mean(lft)), float(np.mean(rgt)))
        stat[i] = power[i] / (noise + 1e-12)
    return stat


def so_cfar(
    x: np.ndarray,
    n_train: int = CA_CFAR_N_TRAIN,
    n_guard: int = CA_CFAR_N_GUARD,
) -> np.ndarray:
    """Smallest-of CFAR using the smaller left/right training-window mean."""
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    if n_guard < 0:
        raise ValueError("n_guard must be non-negative")
    power = np.asarray(x, dtype=np.float64) ** 2
    stat = np.zeros_like(power)
    for i, lft, rgt in _cfar_windows(power, n_train, n_guard):
        noise = min(float(np.mean(lft)), float(np.mean(rgt)))
        stat[i] = power[i] / (noise + 1e-12)
    return stat


def ca_cfar_threshold(
    pfa: float = CA_CFAR_NOMINAL_PFA,
    n_train: int = CA_CFAR_N_TRAIN,
) -> float:
    """Standard square-law CA-CFAR multiplier for exponential clutter.

    ``n_train`` is the number of reference cells on each side, hence the
    closed-form threshold uses ``2 * n_train`` total reference cells.
    """
    if not 0.0 < pfa < 1.0:
        raise ValueError("pfa must be in (0, 1)")
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    n_reference = 2 * n_train
    return float(n_reference * (pfa ** (-1.0 / n_reference) - 1.0))


_CFAR = {
    "ca": ca_cfar,
    "go": go_cfar,
    "so": so_cfar,
    "os": os_cfar,
}


def cfar_batch(x: np.ndarray, kind: str = "ca", **kw) -> np.ndarray:
    """Apply one CFAR statistic over a batch ``x[L, B]`` and return ``[L, B]``."""
    try:
        fn = _CFAR[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported CFAR kind: {kind!r}") from exc
    L, B = x.shape
    out = np.zeros((L, B), dtype=np.float64)
    for j in range(B):
        out[:, j] = fn(x[:, j], **kw)
    return out
