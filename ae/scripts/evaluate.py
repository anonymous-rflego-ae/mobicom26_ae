"""Evaluate an RF-LEGO module on the AE benchmark.

Discovers the prepared real-world held-out benchmark ``.npz`` files under
``ae_data`` (model-ready inputs + ground truth), runs the trained RF-LEGO model
and the classical baseline(s),
computes metrics from *both* sets of outputs, and writes:

- ``ae/results/metrics/<dataset>.json``  — scalars and improvements;
- ``ae/results/cache/<dataset>_plotcache.npz`` — arrays for the result plots.

Usage:
    python ae/scripts/evaluate.py --module all
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

CA_CFAR_NOMINAL_PFA = BL.CA_CFAR_NOMINAL_PFA
RFLEGO_DEFAULT_PROBABILITY_THRESHOLD = 0.5
DETECTOR_SOURCE_MASK_HALF_WIDTH = 1
DETECTOR_MATCH_HALF_WIDTH = 0
FIXED_VIS_SAMPLES = {
    # Fixed representatives for the Beamformer reproduction panel.
    "mmwave_beamformer": [5, 34],
    "mmwave_dopplerft": [23, 18],
    "mmwave_rangeft": [37, 15],
    "uwb_detector": [43, 46],
    "uwb_dopplerft": [0, 36],
    "wifi_dopplerft": [7, 26],
}


def _pick_diverse_top(score: np.ndarray, gt_key: np.ndarray, k: int, fill: bool = False) -> np.ndarray:
    """Pick high-scoring examples while preferring distinct ground-truth keys."""
    score = np.asarray(score, dtype=np.float64)
    gt_key = np.asarray(gt_key)
    order = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]
    picked: list[int] = []
    seen = set()
    for idx in order:
        key = np.asarray(gt_key[idx]).ravel()
        key_tuple = tuple(np.round(key.astype(np.float64), 3))
        if key_tuple in seen:
            continue
        picked.append(int(idx))
        seen.add(key_tuple)
        if len(picked) == k:
            return np.asarray(picked, dtype=np.int64)
    if not fill:
        return np.asarray(picked, dtype=np.int64)
    for idx in order:
        if len(picked) == k:
            break
        if int(idx) not in picked:
            picked.append(int(idx))
    return np.asarray(picked, dtype=np.int64)


def _fixed_first(dataset_id: str | None, n: int, candidates: np.ndarray) -> np.ndarray:
    """Put a fixed visualization sample first; keep candidate entries after it."""
    fixed = [idx for idx in FIXED_VIS_SAMPLES.get(dataset_id or "", []) if 0 <= idx < n]
    if not fixed:
        return candidates
    rest = [int(idx) for idx in candidates if int(idx) not in fixed]
    return np.asarray([*fixed, *rest], dtype=np.int64)


def _load_model(module: str, config_path: Path, weights_path: Path, device):
    cfg = C.load_yaml(config_path)
    model_cfg = C.build_model_config(module, cfg)
    cls = {"ft": FrequencyTransformModel, "beamformer": BeamformerModel, "detector": DetectorModel}[
        module
    ]
    model = cls(model_cfg)
    model.load(weights_path)
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
def evaluate_ft(npz, model, device, dataset_id: str | None = None) -> tuple[dict, dict]:
    xr = torch.from_numpy(npz["x_real"]).to(device)
    xi = torch.from_numpy(npz["x_imag"]).to(device)
    with torch.no_grad():
        yr, yi = model(xr, xi)
    mag = torch.sqrt(yr.squeeze(1) ** 2 + yi.squeeze(1) ** 2 + 1e-12).cpu().numpy()
    fft_mag = BL.fft_magnitude(npz["x_real"], npz["x_imag"])
    gt = npz["gt_peak_bin"]

    r = MT.ft_metrics(mag, gt)
    b = MT.ft_metrics(fft_mag, gt)
    d_pslr = r["pslr_db"] - b["pslr_db"]
    d_papr = r["papr_db"] - b["papr_db"]
    pslr_delta = r["_pslr_per_sample"] - b["_pslr_per_sample"]
    papr_delta = r["_papr_per_sample"] - b["_papr_per_sample"]

    metrics = {
        "module": "ft",
        "sample_count": int(mag.shape[0]),
        "baseline": {
            "name": "Bluestein FFT",
            "pslr_db": b["pslr_db"],
            "papr_db": b["papr_db"],
        },
        "rflego": {
            "pslr_db": r["pslr_db"],
            "papr_db": r["papr_db"],
        },
        "improvement": {"pslr_db": d_pslr, "papr_db": d_papr},
    }
    score = np.nan_to_num(pslr_delta + 0.5 * papr_delta, nan=-np.inf)
    rep = _fixed_first(dataset_id, mag.shape[0], _pick_diverse_top(score, gt, min(6, mag.shape[0])))
    cache = {
        "rflego_mag": mag[rep].astype(np.float32),
        "fft_mag": fft_mag[rep].astype(np.float32),
        "gt_peak_bin": gt[rep],
        "pslr_rflego": r["_pslr_per_sample"][rep].astype(np.float32),
        "pslr_fft": b["_pslr_per_sample"][rep].astype(np.float32),
        "papr_rflego": r["_papr_per_sample"][rep].astype(np.float32),
        "papr_fft": b["_papr_per_sample"][rep].astype(np.float32),
        "sample_idx": rep.astype(np.int64),
    }
    return metrics, cache


def evaluate_beamformer(npz, model, device, dataset_id: str | None = None) -> tuple[dict, dict]:
    y = torch.from_numpy(npz["y_meas"]).to(device)
    A = torch.from_numpy(npz["A_dict"]).to(device)
    with torch.no_grad():
        z = model(y, A)
    mag = z.abs().cpu().numpy()
    grid = npz["angle_grid_deg"]
    gt_ang = npz["gt_angles_deg"][:, 0]
    n_targets = np.ones(npz["gt_angle_idx"].shape[0], dtype=np.int64)
    gt = {
        "gt_angle_idx": npz["gt_angle_idx"],
        "gt_angles_deg": npz["gt_angles_deg"],
        "n_targets": n_targets,
        "angle_grid_deg": npz["angle_grid_deg"],
    }

    # LASSO Beamformer sparse DoA baseline. The same lambda, step, and iteration
    # count are used for every benchmark sample.
    bcfg = C.load_yaml(C.module_config_path("beamformer")).get("baseline", {})
    lam = float(bcfg.get("lasso_lam", 0.1))
    step = float(bcfg.get("lasso_step", 0.01))
    iters = int(bcfg.get("lasso_iters", 8))
    lasso = BL.lasso_beamformer_batch(
        npz["y_meas"], npz["A_dict"], n_iter=iters, lam=lam, step=step
    )

    def per_sample_err(spec):
        return np.abs(grid[np.argmax(spec, axis=1)] - gt_ang)

    err_r = per_sample_err(mag)
    err_l = per_sample_err(lasso)
    est_r = grid[np.argmax(mag, axis=1)]
    est_l = grid[np.argmax(lasso, axis=1)]

    r = MT.beamformer_metrics(mag, gt)
    b_lasso = MT.beamformer_metrics(lasso, gt)
    red = (b_lasso["angle_mae_deg"] - r["angle_mae_deg"]) / b_lasso["angle_mae_deg"] * 100.0

    improvement_per_sample = err_l - err_r
    rep = _fixed_first(
        dataset_id,
        mag.shape[0],
        _pick_diverse_top(improvement_per_sample, gt_ang, min(4, mag.shape[0])),
    )

    metrics = {
        "module": "beamformer",
        "sample_count": int(mag.shape[0]),
        "baseline": {
            "name": "LASSO Beamformer",
            "angle_mae_deg": b_lasso["angle_mae_deg"],
        },
        "rflego": {
            "angle_mae_deg": r["angle_mae_deg"],
        },
        "improvement": {"mae_reduction_percent": red},
    }
    cache = {
        "rflego": mag[rep].astype(np.float32),
        "lasso": lasso[rep].astype(np.float32),
        "gt_angles_deg": npz["gt_angles_deg"][rep],
        "angle_grid_deg": grid,
        "rep_lasso_mae": err_l[rep].astype(np.float32),
        "rep_rflego_mae": err_r[rep].astype(np.float32),
        "rep_lasso_est_deg": est_l[rep].astype(np.float32),
        "rep_rflego_est_deg": est_r[rep].astype(np.float32),
        "sample_idx": rep.astype(np.int64),
    }
    return metrics, cache


def _target_bins_from_mask(label_Ln: np.ndarray, half_width: int = 1) -> np.ndarray:
    """Recover target centers from non-overlapping fixed-width mask windows.

    Older prepared Detector NPZ files retain only ``gt_mask``. Their target
    windows are exact centered three-bin runs, so adjacent windows are split
    into consecutive fixed-width chunks. Any non-conforming mask is rejected
    instead of silently inventing target identities.
    """
    label = np.asarray(label_Ln)
    if label.ndim != 2:
        raise ValueError("label_Ln must have shape [length, n_samples]")
    if half_width < 0:
        raise ValueError("half_width must be non-negative")
    window_width = 2 * half_width + 1
    per_sample: list[list[int]] = []
    for sample_index in range(label.shape[1]):
        positive = np.flatnonzero(label[:, sample_index] > 0.5)
        runs = np.split(positive, np.flatnonzero(np.diff(positive) > 1) + 1) if positive.size else []
        centers: list[int] = []
        for run in runs:
            if run.size % window_width != 0:
                raise ValueError(
                    f"sample {sample_index}: positive-mask run length {run.size} "
                    f"is not divisible by target window width {window_width}"
                )
            for start in range(0, run.size, window_width):
                chunk = run[start : start + window_width]
                expected = np.arange(chunk[0], chunk[0] + window_width)
                if not np.array_equal(chunk, expected):
                    raise ValueError(f"sample {sample_index}: target mask is not contiguous")
                centers.append(int(chunk[half_width]))
        per_sample.append(centers)

    max_targets = max((len(centers) for centers in per_sample), default=0)
    target_bins = np.full((label.shape[1], max_targets), -1, dtype=np.int64)
    for sample_index, centers in enumerate(per_sample):
        target_bins[sample_index, : len(centers)] = centers
    return target_bins


def _mask_from_target_bins(
    target_bins_nK: np.ndarray,
    length: int,
    half_width: int = 0,
) -> np.ndarray:
    """Build a ``[length, n_samples]`` mask around valid target centers."""
    targets = np.asarray(target_bins_nK)
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.ndim != 2:
        raise ValueError("target_bins_nK must have shape [n_samples, n_targets]")
    if half_width < 0:
        raise ValueError("half_width must be non-negative")
    mask = np.zeros((length, targets.shape[0]), dtype=np.float64)
    for sample_index, centers in enumerate(targets):
        for value in centers:
            center = int(value)
            if center < 0:
                continue
            if center >= length:
                raise ValueError(f"target bin {center} is outside [0, {length})")
            lo = max(0, center - half_width)
            hi = min(length, center + half_width + 1)
            mask[lo:hi, sample_index] = 1.0
    return mask


def evaluate_detector(npz, model, device, dataset_id: str | None = None) -> tuple[dict, dict]:
    x = torch.from_numpy(npz["x"]).to(device)  # [L, B]
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).cpu().numpy()  # [L, B]
    source_label = npz["gt_mask"]
    ca_stat = BL.cfar_batch(npz["x"])
    go_stat = BL.cfar_batch(npz["x"], kind="go")
    so_stat = BL.cfar_batch(npz["x"], kind="so")
    os_stat = BL.cfar_batch(npz["x"], kind="os")

    target_bins = (
        np.asarray(npz["target_bins"])
        if "target_bins" in npz.files
        else _target_bins_from_mask(
            source_label,
            half_width=DETECTOR_SOURCE_MASK_HALF_WIDTH,
        )
    )
    evaluation_label = _mask_from_target_bins(
        target_bins,
        length=prob.shape[0],
        half_width=DETECTOR_MATCH_HALF_WIDTH,
    )
    r = MT.fixed_threshold_op(
        prob.ravel(),
        evaluation_label.ravel(),
        RFLEGO_DEFAULT_PROBABILITY_THRESHOLD,
    )
    ca_op = MT.fixed_threshold_op(
        ca_stat.ravel(),
        evaluation_label.ravel(),
        BL.ca_cfar_threshold(CA_CFAR_NOMINAL_PFA),
    )

    ex_scores = []
    target_centers = []
    for j in range(prob.shape[1]):
        sample_targets = target_bins[j][target_bins[j] >= 0]
        pos = prob[:, j][sample_targets]
        neg = prob[:, j][evaluation_label[:, j] <= 0.5]
        target_centers.append(float(np.mean(sample_targets)) if sample_targets.size else -1.0)
        if pos.size == 0 or neg.size == 0:
            ex_scores.append(-np.inf)
        else:
            ex_scores.append(float(np.mean(pos) - np.quantile(neg, 0.99)))
    ex_idx = [idx for idx in FIXED_VIS_SAMPLES.get(dataset_id or "", []) if 0 <= idx < prob.shape[1]]
    if not ex_idx:
        ex_idx = [int(_pick_diverse_top(np.asarray(ex_scores), np.asarray(target_centers), 1)[0])]

    def _norm(v):
        selected = v[:, ex_idx]
        return (selected / (selected.max(axis=0, keepdims=True) + 1e-9)).astype(np.float32)

    metrics = {
        "module": "detector",
        "sample_count": int(prob.shape[1]),
        "operating_point": "exact-bin DR at method-native fixed thresholds",
        "operating_point_alignment": "method-native fixed decision rules",
        "comparison_note": (
            "CA-CFAR uses its nominal P_FA of 10^-3 as the design point, while RF-LEGO "
            "Detector has no native P_FA parameter and uses its fixed probability "
            "cutoff."
        ),
        "detection_definition": (
            "each target is detected only if its exact center bin reaches cutoff"
        ),
        "baseline": {
            "name": "CA-CFAR",
            "parameter_policy": "single fixed configuration",
            "n_train_per_side": BL.CA_CFAR_N_TRAIN,
            "n_guard_per_side": BL.CA_CFAR_N_GUARD,
            "nominal_pfa": CA_CFAR_NOMINAL_PFA,
            "threshold_multiplier": ca_op["cutoff"],
            "dr": ca_op["dr"],
            "true_positives": ca_op["true_positives"],
        },
        "rflego": {
            "decision_rule": "sigmoid(logit) >= 0.5",
            "probability_threshold": RFLEGO_DEFAULT_PROBABILITY_THRESHOLD,
            "dr": r["dr"],
            "true_positives": r["true_positives"],
        },
    }
    cache = {
        "example_x": npz["x"][:, ex_idx].astype(np.float32),
        "example_prob": (
            prob[:, ex_idx] / (prob[:, ex_idx].max(axis=0, keepdims=True) + 1e-9)
        ).astype(np.float32),
        "example_ca": _norm(ca_stat),
        "example_go": _norm(go_stat),
        "example_so": _norm(so_stat),
        "example_os": _norm(os_stat),
        "example_mask": evaluation_label[:, ex_idx].astype(np.float32),
        "example_idx": np.asarray(ex_idx, dtype=np.int64),
        "example_target_center": np.asarray([target_centers[i] for i in ex_idx], dtype=np.float32),
        "ca_nominal_pfa": np.float32(CA_CFAR_NOMINAL_PFA),
        "rflego_probability_threshold": np.float32(RFLEGO_DEFAULT_PROBABILITY_THRESHOLD),
    }
    return metrics, cache


_EVALUATORS = {"ft": evaluate_ft, "beamformer": evaluate_beamformer, "detector": evaluate_detector}


def _result_text(metrics: dict) -> str:
    module = metrics["module"]
    imp = metrics.get("improvement", {})
    if module == "ft":
        return f"PSLR {imp['pslr_db']:+.2f} dB; PAPR {imp['papr_db']:+.2f} dB"
    if module == "beamformer":
        return f"angle-MAE {imp['mae_reduction_percent']:+.1f}%"
    if module == "detector":
        rflego = metrics["rflego"]
        return f"default DR {rflego['dr']:.3f}"
    return str(imp)


def _analysis_text(metrics: dict) -> str:
    module = metrics["module"]
    imp = metrics.get("improvement", {})
    if module == "ft":
        if imp["pslr_db"] > 0 and imp["papr_db"] > 0:
            return "RF-LEGO improves sidelobe suppression and peak concentration."
        if imp["pslr_db"] > 0:
            return "RF-LEGO improves sidelobe suppression, while PAPR is weaker."
        return "This split is not favorable to the learned FT weights."
    if module == "beamformer":
        if imp["mae_reduction_percent"] > 30:
            return "Strong DoA error reduction over the sparse LASSO baseline."
        if imp["mae_reduction_percent"] > 0:
            return "Positive DoA gain, but the margin is modest."
        return "LASSO is stronger on this split."
    if module == "detector":
        return "Method-native fixed decision rules in the 10^-3-order FAR regime."
    return ""


def _attach_dataset(metrics: dict, spec: C.BenchmarkSpec) -> dict:
    metrics = dict(metrics)
    metrics["dataset"] = {
        "id": spec.id,
        "modality": spec.modality,
        "modality_label": spec.modality_label,
        "task": spec.task,
        "task_label": spec.task_label,
        "path": str(spec.path.relative_to(C.REPO_ROOT)),
    }
    return metrics


def _summary_row(metrics: dict) -> dict:
    ds = metrics["dataset"]
    return {
        "id": ds["id"],
        "modality": ds["modality_label"],
        "module": ds["task_label"],
        "rflego_module": metrics["module"],
        "samples": metrics["sample_count"],
        "results": _result_text(metrics),
        "analysis": _analysis_text(metrics),
    }


def _print_summary(rows: list[dict]) -> None:
    headers = ["modality", "module", "samples", "results", "analysis"]
    widths = {
        h: max(len(h), *(len(str(row[h])) for row in rows))
        for h in headers
    }
    print("\n=== AE modality results ===")
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("-" * (sum(widths.values()) + 2 * (len(headers) - 1)))
    for row in rows:
        print("  ".join(str(row[h]).ljust(widths[h]) for h in headers))


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate RF-LEGO on the AE benchmark.")
    ap.add_argument("--module", default="all", choices=["all", *C.MODULES])
    ap.add_argument(
        "--dataset",
        default="all",
        help="Dataset id under ae_data, e.g. mmwave_dopplerft, or 'all'.",
    )
    ap.add_argument("--weights-dir", default=str(C.WEIGHTS_DIR))
    args = ap.parse_args()

    C.ensure_dirs()
    C.set_determinism(42)
    device = C.eval_device()
    specs = C.discover_benchmarks()
    if args.module != "all":
        specs = [spec for spec in specs if spec.module == args.module]
    if args.dataset != "all":
        specs = [spec for spec in specs if spec.id == args.dataset]
    if not specs:
        raise FileNotFoundError("No matching AE benchmark .npz files found under ae_data.")

    models = {}
    rows = []
    for spec in specs:
        with np.load(spec.path) as npz:
            if spec.module not in models:
                weights = Path(args.weights_dir) / f"{spec.module}.pt"
                models[spec.module] = _load_model(
                    spec.module, C.module_config_path(spec.module), weights, device
                )
            metrics, cache = _EVALUATORS[spec.module](npz, models[spec.module], device, spec.id)
        metrics = _attach_dataset(metrics, spec)
        C.write_json(C.METRICS_DIR / f"{spec.id}.json", metrics)
        np.savez(C.CACHE_DIR / f"{spec.id}_plotcache.npz", **cache)
        row = _summary_row(metrics)
        rows.append(row)
        print(f"[evaluate] {spec.id}: {row['results']}")

    C.write_json(C.METRICS_DIR / "summary.json", {"rows": rows})
    _print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
