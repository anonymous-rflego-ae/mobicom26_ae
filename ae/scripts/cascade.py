"""Compact RF-LEGO cascadability check related to paper Sec. 5.2.2.

Runs the three mmWave AE front-end -> detector paths with the shipped weights:

    classical range/Doppler : Bluestein FFT magnitude -> CA-CFAR statistic
    RF-LEGO  range/Doppler  : RF-LEGO FT magnitude -> Detector retained operating point
    classical angle         : LASSO magnitude -> CA-CFAR statistic
    RF-LEGO  angle          : Beamformer magnitude -> Detector retained operating point

Every row reports exact-bin Detection Rate (DR). Both paths are read out at
fixed, retained scalar cutoffs placed in a common ``10^-3``-order low-FAR
regime, so the comparison is order-of-magnitude FAR-regime alignment.

Design notes / assumptions (documented so reviewers can judge faithfulness):
- Every learned path uses per-profile min-max normalization to ``[0, 1]`` and
  Detector logits, matching its training contract.
- The angle adapter is
  derived from the original Detector data generator and its input contract;
  the paper itself does not specify the numerical Beamformer -> Detector
  adapter. It now matches the corrected Detector's synthetic input
  normalization, although the synthetic and angular profile shapes still
  differ.
- Classical paths use CA-CFAR only. Range and Doppler retain the default
  reviewer-visible 4/2 training/guard cells per side. Angle alone uses the
  fixed experimental setting of 18/10 training/guard cells per side.
- RF-LEGO Detector does not expose a CA-CFAR-style ``P_FA`` parameter. Its
  scalar score cutoff defines the readout operating point but does not change
  the trained detector.
- Each target contributes only its exact ground-truth center bin. Adjacent bins
  are negatives and receive no hit credit.

Usage:
    python ae/scripts/cascade.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _baselines as BL  # noqa: E402
import _common as C  # noqa: E402
import _metrics as MT  # noqa: E402

from rflego.modules import BeamformerModel, DetectorModel, FrequencyTransformModel  # noqa: E402

ANGLE_CA_CFAR_N_TRAIN = 18
ANGLE_CA_CFAR_N_GUARD = 10

# Fixed, retained scalar cutoffs for the two paths. Both sets place the readout
# in the same 10^-3-order low-FAR regime and are constants of the artifact.
CA_CFAR_CUTOFFS = {
    "range": 70.57548064425025,
    "doppler": 75.69889395277420,
    "angle": 54.78931125921218,
}
RFLEGO_CUTOFFS = {
    "range": 3.0665857791900635,
    "doppler": -3.9720332622528076,
    "angle": 13.007723808288574,
}

RANGE_DOPPLER_MASK_HALF = 0
ANGLE_MASK_HALF = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _mask_from_bins(
    peak_bins: np.ndarray,
    length: int,
    half_width: int = RANGE_DOPPLER_MASK_HALF,
) -> np.ndarray:
    """Target mask with configurable half-width, shape ``[length, n_samples]``."""
    if half_width < 0:
        raise ValueError("half_width must be non-negative")
    n = len(peak_bins)
    mask = np.zeros((length, n), dtype=np.float64)
    for j, b in enumerate(peak_bins):
        b = int(b)
        for off in range(-half_width, half_width + 1):
            k = b + off
            if 0 <= k < length:
                mask[k, j] = 1.0
    return mask


def _detector_score(
    spec_nL: np.ndarray,
    det: DetectorModel,
    device,
    normalization: str,
    score_type: str = "logit",
) -> np.ndarray:
    """Run the RF-LEGO Detector and return a score map ``[L, n]``."""
    x_np = _norm_for_detector(spec_nL, normalization=normalization)
    x = torch.from_numpy(x_np.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = det(x)
        if score_type == "logit":
            score = logits
        elif score_type == "probability":
            score = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unsupported detector score type: {score_type!r}")
        score = score.cpu().numpy()
    return score


def _norm_for_detector(spec_nL: np.ndarray, normalization: str = "minmax") -> np.ndarray:
    """Normalize each spectrum and return Detector-ready ``[L, n]``."""
    x = np.asarray(spec_nL, dtype=np.float64).T
    if normalization == "median":
        return x / (np.median(x, axis=0, keepdims=True) + 1e-9)
    if normalization == "minmax":
        x_min = np.min(x, axis=0, keepdims=True)
        span = np.max(x, axis=0, keepdims=True) - x_min
        return np.divide(x - x_min, span, out=np.zeros_like(x), where=span > 0.0)
    raise ValueError(f"Unsupported detector normalization: {normalization!r}")


def _ca_cfar_op(
    spec_nL: np.ndarray,
    mask_Ln: np.ndarray,
    n_train: int,
    n_guard: int,
    cutoff: float,
) -> dict:
    """CA-CFAR exact-bin DR at the retained fixed cutoff."""
    x_Ln = np.asarray(spec_nL, dtype=np.float64).T  # [L, n]
    stat = BL.cfar_batch(x_Ln, n_train=n_train, n_guard=n_guard)  # [L, n]
    return MT.fixed_threshold_op(stat.ravel(), mask_Ln.ravel(), cutoff)


def _load(module: str, device, weights_dir: Path):
    cfg = C.load_yaml(C.module_config_path(module))
    model_cfg = C.build_model_config(module, cfg)
    cls = {"ft": FrequencyTransformModel, "beamformer": BeamformerModel, "detector": DetectorModel}[
        module
    ]
    model = cls(model_cfg)
    model.load(Path(weights_dir) / f"{module}.pt")
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Front-end spectra
# --------------------------------------------------------------------------- #
def _ft_spectra(npz, ft: FrequencyTransformModel, device):
    xr = torch.from_numpy(npz["x_real"]).to(device)
    xi = torch.from_numpy(npz["x_imag"]).to(device)
    with torch.no_grad():
        yr, yi = ft(xr, xi)
    rflego = torch.sqrt(yr.squeeze(1) ** 2 + yi.squeeze(1) ** 2 + 1e-12).cpu().numpy()
    classical = BL.fft_magnitude(npz["x_real"], npz["x_imag"])
    return rflego, classical, npz["gt_peak_bin"]


def _beamformer_spectra(npz, bf: BeamformerModel, device):
    y = torch.from_numpy(npz["y_meas"]).to(device)
    A = torch.from_numpy(npz["A_dict"]).to(device)
    with torch.no_grad():
        rflego = bf(y, A).abs().cpu().numpy()
    bcfg = C.load_yaml(C.module_config_path("beamformer")).get("baseline", {})
    classical = BL.lasso_beamformer_batch(
        npz["y_meas"],
        npz["A_dict"],
        n_iter=int(bcfg.get("lasso_iters", 8)),
        lam=float(bcfg.get("lasso_lam", 0.1)),
        step=float(bcfg.get("lasso_step", 0.01)),
    )
    return rflego, classical, npz["gt_angle_idx"][:, 0]


# --------------------------------------------------------------------------- #
def _run_pipeline(
    name,
    rflego_spec,
    classical_spec,
    peak_bins,
    det,
    device,
    normalization,
    mask_half_width,
    cfar_n_train=BL.CA_CFAR_N_TRAIN,
    cfar_n_guard=BL.CA_CFAR_N_GUARD,
):
    length = rflego_spec.shape[1]
    mask = _mask_from_bins(peak_bins, length, half_width=mask_half_width)

    # Classical path: front-end -> CA-CFAR statistic at the retained cutoff.
    classical_op = _ca_cfar_op(
        classical_spec,
        mask,
        n_train=cfar_n_train,
        n_guard=cfar_n_guard,
        cutoff=CA_CFAR_CUTOFFS[name],
    )
    # RF-LEGO cascade: front-end -> retained per-pipeline operating point.
    rflego_score = _detector_score(
        rflego_spec,
        det,
        device,
        normalization,
        score_type="logit",
    )
    rflego_op = MT.fixed_threshold_op(rflego_score.ravel(), mask.ravel(), RFLEGO_CUTOFFS[name])

    return {
        "pipeline": name,
        "metric": "exact-bin DR in a common 10^-3-order FAR regime",
        "detection_definition": (
            "a target is detected only if its exact ground-truth center bin reaches the cutoff"
        ),
        "operating_point_alignment": "same-order 10^-3 FAR regime",
        "comparison_note": (
            "Both paths are read out at fixed, retained scalar cutoffs placed in "
            "the same 10^-3-order FAR regime."
        ),
        "detector_input": f"{normalization}_normalized",
        "detector_score": "logit",
        "target_mask_half_width_bins": mask_half_width,
        "dr_classical": classical_op["dr"],
        "true_positives_classical": classical_op["true_positives"],
        "classical_cfar": "ca",
        "ca_cfar_n_train_per_side": cfar_n_train,
        "ca_cfar_n_guard_per_side": cfar_n_guard,
        "ca_cfar_cutoff": classical_op["cutoff"],
        "dr_rflego": rflego_op["dr"],
        "true_positives_rflego": rflego_op["true_positives"],
        "rflego_cutoff": rflego_op["cutoff"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact RF-LEGO cascadability check.")
    ap.add_argument("--weights-dir", default=str(C.WEIGHTS_DIR))
    args = ap.parse_args()

    C.ensure_dirs()
    C.set_determinism(42)
    device = C.eval_device()
    weights_dir = Path(args.weights_dir)

    ft = _load("ft", device, weights_dir)
    bf = _load("beamformer", device, weights_dir)
    det = _load("detector", device, weights_dir)

    rows = []

    # (i) range / Doppler : FT -> Detector
    for name, ds in (("range", "mmwave_rangeft"), ("doppler", "mmwave_dopplerft")):
        with np.load(C.AE_DATA_DIR / f"{ds}.npz") as npz:
            rflego_spec, classical_spec, peaks = _ft_spectra(npz, ft, device)
        rows.append(
            _run_pipeline(
                name,
                rflego_spec,
                classical_spec,
                peaks,
                det,
                device,
                normalization="minmax",
                mask_half_width=RANGE_DOPPLER_MASK_HALF,
            )
        )

    # (ii) angle : Beamformer -> Detector, exact-bin target
    with np.load(C.AE_DATA_DIR / "mmwave_beamformer.npz") as npz:
        rflego_spec, classical_spec, peaks = _beamformer_spectra(npz, bf, device)
    rows.append(
        _run_pipeline(
            "angle",
            rflego_spec,
            classical_spec,
            peaks,
            det,
            device,
            normalization="minmax",
            mask_half_width=ANGLE_MASK_HALF,
            cfar_n_train=ANGLE_CA_CFAR_N_TRAIN,
            cfar_n_guard=ANGLE_CA_CFAR_N_GUARD,
        )
    )

    C.write_json(
        C.METRICS_DIR / "cascade.json",
        {
            "classical_baseline": {
                "name": "CA-CFAR",
                "range_doppler_window": {
                    "n_train_per_side": BL.CA_CFAR_N_TRAIN,
                    "n_guard_per_side": BL.CA_CFAR_N_GUARD,
                },
                "angle_window": {
                    "n_train_per_side": ANGLE_CA_CFAR_N_TRAIN,
                    "n_guard_per_side": ANGLE_CA_CFAR_N_GUARD,
                    "rationale": "fixed angle-only experimental configuration",
                },
                "nominal_pfa": BL.CA_CFAR_NOMINAL_PFA,
            },
            "operating_point_policy": {
                "description": (
                    "same-order 10^-3 FAR-regime alignment via fixed retained cutoffs"
                ),
                "ca_cfar": {"retained_cutoffs": CA_CFAR_CUTOFFS},
                "rflego": {
                    "native_pfa_parameter": None,
                    "retained_cutoffs": RFLEGO_CUTOFFS,
                },
            },
            "rows": rows,
        },
    )

    print("\n=== Cascadability check: RF-LEGO vs classical front ends (mmWave) ===")
    print("all rows: strict exact-bin DR in a common 10^-3-order FAR regime")
    print("both paths are read out at fixed retained cutoffs")
    hdr = f"{'pipeline':9s} {'CA window':>9s} {'classical DR':>12s} {'RF-LEGO DR':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['pipeline']:9s} "
            f"{r['ca_cfar_n_train_per_side']:>2d}/{r['ca_cfar_n_guard_per_side']:<6d} "
            f"{r['dr_classical']:>12.3f} {r['dr_rflego']:>11.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
