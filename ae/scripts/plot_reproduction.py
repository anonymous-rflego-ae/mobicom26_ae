"""Render the AE module-result plots from cached evaluation arrays.

Consumes only the per-dataset plot caches written by ``evaluate.py``
(``ae/results/cache/<dataset>_plotcache.npz``) plus the metrics JSON; it performs
no model inference or metric recomputation.

Outputs:
    ae/results/figures/result_<dataset>.png

Usage:
    python ae/scripts/plot_reproduction.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

DB_FLOOR = -60.0


def _db(mag: np.ndarray) -> np.ndarray:
    mag = np.asarray(mag, dtype=np.float64)
    peak = np.max(mag) + 1e-12
    return np.clip(20.0 * np.log10(mag / peak + 1e-12), DB_FLOOR, 0.0)


def _lin(mag: np.ndarray) -> np.ndarray:
    """Peak-normalized linear magnitude in [0, 1]."""
    mag = np.asarray(mag, dtype=np.float64)
    return mag / (np.max(mag) + 1e-12)


# --------------------------------------------------------------------------- #
def _dataset_title(metrics: dict) -> str:
    ds = metrics.get("dataset", {})
    modality = ds.get("modality_label", "AE")
    task = ds.get("task_label", metrics.get("module", "").upper())
    return f"{modality} {task}"


def plot_ft(dataset_id: str, metrics: dict) -> Path:
    cache = np.load(C.CACHE_DIR / f"{dataset_id}_plotcache.npz")
    bins = np.arange(cache["rflego_mag"].shape[1])

    n = min(2, cache["rflego_mag"].shape[0])
    fig, axes = plt.subplots(1, n, figsize=(7.2 * n, 4.8), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(
            bins,
            _lin(cache["fft_mag"][i]),
            color="#d62728",
            lw=1.6,
            ls="--",
            label="Bluestein FFT",
        )
        ax.plot(bins, _lin(cache["rflego_mag"][i]), color="#1f77b4", lw=2.2, label="RF-LEGO FT")
        gtb = int(cache["gt_peak_bin"][i])
        ax.plot([gtb, gtb], [0, 1], color="gold", lw=10, alpha=0.35, solid_capstyle="round")
        ax.plot([gtb], [1.0], marker="*", ms=18, color="gold", mec="0.4", zorder=6, label="true bin")
        sample = int(cache["sample_idx"][i]) if "sample_idx" in cache else i
        ax.set_title(f"sample {sample} | true bin {gtb}", fontsize=11)
        ax.set_xlabel("frequency bin")
        ax.set_ylabel("normalized spectrum")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(_dataset_title(metrics), fontsize=13)
    fig.tight_layout()
    out = C.FIGURES_DIR / f"result_{dataset_id}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_beamformer(dataset_id: str, metrics: dict) -> Path:
    cache = np.load(C.CACHE_DIR / f"{dataset_id}_plotcache.npz")
    grid = np.deg2rad(cache["angle_grid_deg"])
    c_lasso, c_rflego = "#1f77b4", "#d62728"

    n = min(2, cache["rflego"].shape[0])
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5.8 * n, 5.4),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )
    for i, ax in enumerate(axes.ravel()):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-60)
        ax.set_thetamax(60)
        ax.set_rlim(0, 1.05)
        tgt = np.deg2rad(float(cache["gt_angles_deg"][i, 0]))
        ax.plot([tgt, tgt], [0, 1], color="gold", lw=10, alpha=0.35, solid_capstyle="round")
        ax.plot([tgt], [1.0], marker="*", ms=20, color="gold", mec="0.4", zorder=6)
        ax.plot(grid, _lin(cache["lasso"][i]), color=c_lasso, lw=2.0, label="LASSO Beamformer")
        ax.plot(grid, _lin(cache["rflego"][i]), color=c_rflego, lw=2.6, label="RF-LEGO Beamformer")
        sample = int(cache["sample_idx"][i]) if "sample_idx" in cache else i
        gt_deg = float(cache["gt_angles_deg"][i, 0])
        ax.set_title(
            f"sample {sample} | true {gt_deg:g} deg\n"
            f"LASSO err {cache['rep_lasso_mae'][i]:.1f}, RF-LEGO err {cache['rep_rflego_mae'][i]:.1f}",
            fontsize=10,
            pad=16,
        )
        ax.set_thetagrids([-60, -30, 0, 30, 60])
        if i == 0:
            ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(-0.08, 0.0))

    fig.suptitle(_dataset_title(metrics), fontsize=13)
    fig.tight_layout()
    out = C.FIGURES_DIR / f"result_{dataset_id}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_detector(dataset_id: str, metrics: dict) -> Path:
    cache = np.load(C.CACHE_DIR / f"{dataset_id}_plotcache.npz")

    example_x = cache["example_x"]
    example_prob = cache["example_prob"]
    example_mask = cache["example_mask"]
    if example_x.ndim == 1:
        example_x = example_x[:, None]
        example_prob = example_prob[:, None]
        example_mask = example_mask[:, None]
    n = min(2, example_x.shape[1])
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 4.4), squeeze=False)
    L = min(188, example_x.shape[0])
    bins = np.arange(L)
    ex_idx = np.asarray(cache["example_idx"]).ravel() if "example_idx" in cache else np.arange(n)
    true_bins = (
        np.rint(np.asarray(cache["example_target_center"]).ravel()).astype(int)
        if "example_target_center" in cache
        else np.full(n, -1, dtype=int)
    )
    for i, ax in enumerate(axes.ravel()):
        x = example_x[:L, i]
        ax.plot(bins, x / (x.max() + 1e-9), color="0.75", lw=1.0, label="input")
        ax.plot(
            bins,
            cache["example_ca"][:L, i],
            color="#4d4d4d",
            lw=1.3,
            ls="--",
            label="CA-CFAR",
        )
        ax.plot(
            bins,
            cache["example_go"][:L, i],
            color="#cc79a7",
            lw=1.3,
            ls="-.",
            label="GO-CFAR",
        )
        ax.plot(
            bins,
            cache["example_so"][:L, i],
            color="#56b4e9",
            lw=1.4,
            ls=":",
            label="SO-CFAR",
        )
        ax.plot(
            bins,
            cache["example_os"][:L, i],
            color="#009e73",
            lw=1.3,
            ls=(0, (5, 2)),
            label="OS-CFAR",
        )
        ax.plot(
            bins,
            example_prob[:L, i],
            color="#d55e00",
            lw=2.3,
            label="RF-LEGO Detector",
        )
        true_bin = int(true_bins[i]) if i < true_bins.size else -1
        if 0 <= true_bin < L:
            ax.plot([true_bin, true_bin], [0, 1], color="gold", lw=10, alpha=0.35, solid_capstyle="round")
            ax.plot([true_bin], [1.0], marker="*", ms=18, color="gold", mec="0.4", zorder=6, label="true bin")
        ax.set_xlabel("range bin")
        ax.set_ylabel("normalized amplitude / probability")
        ax.set_xlim(0, L - 1)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"sample {int(ex_idx[i])} | true bin {true_bin}", fontsize=11)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    fig.suptitle(_dataset_title(metrics), fontsize=13)
    fig.tight_layout()
    out = C.FIGURES_DIR / f"result_{dataset_id}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out



_PLOTTERS = {"ft": plot_ft, "beamformer": plot_beamformer, "detector": plot_detector}


def _score(metrics: dict) -> float:
    imp = metrics.get("improvement", {})
    module = metrics.get("module")
    if module == "ft":
        return float(imp.get("pslr_db", 0.0) + 0.5 * imp.get("papr_db", 0.0))
    if module == "beamformer":
        return float(imp.get("mae_reduction_percent", 0.0))
    if module == "detector":
        return float(metrics.get("rflego", {}).get("dr", 0.0))
    return 0.0


def _candidate_metrics(module: str) -> list[dict]:
    summary_path = C.METRICS_DIR / "summary.json"
    if summary_path.exists():
        ids = [row["id"] for row in C.read_json(summary_path)["rows"]]
    else:
        ids = [p.name[: -len("_plotcache.npz")] for p in C.CACHE_DIR.glob("*_plotcache.npz")]
    metrics = []
    for dataset_id in ids:
        path = C.METRICS_DIR / f"{dataset_id}.json"
        if path.exists():
            item = C.read_json(path)
            if module == "all" or item.get("module") == module:
                metrics.append(item)
    return metrics


def _select_metrics(module: str, dataset: str, select: str, top_k: int) -> list[dict]:
    if dataset != "all":
        return [C.read_json(C.METRICS_DIR / f"{dataset}.json")]
    metrics = _candidate_metrics(module)
    if select == "best":
        metrics = sorted(metrics, key=_score, reverse=True)[:top_k]
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Render the AE module-result plots.")
    ap.add_argument("--module", default="all", choices=["all", *C.MODULES])
    ap.add_argument("--dataset", default="all", help="Dataset id, e.g. uwb_dopplerft.")
    ap.add_argument("--select", default="all", choices=["all", "best"])
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()
    C.ensure_dirs()
    metrics_list = _select_metrics(args.module, args.dataset, args.select, args.top_k)
    if not metrics_list:
        raise FileNotFoundError("No metrics/cache files found. Run evaluate.py first.")
    for metrics in metrics_list:
        dataset_id = metrics["dataset"]["id"]
        module = metrics["module"]
        out = _PLOTTERS[module](dataset_id, metrics)
        print(f"[plot] {dataset_id}: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
