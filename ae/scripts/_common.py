"""Shared infrastructure for the RF-LEGO AE scripts.

Centralizes: project paths, YAML -> dataclass config loading, determinism setup,
environment capture and JSON I/O. Importing this
module does not trigger any data generation or training.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Make ``rflego`` importable even when the editable-install ``.pth`` is not active
# (e.g. a fresh Colab runtime before ``pip install -e .`` takes effect, or a venv
# whose site ``.pth`` was not processed).
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if importlib.util.find_spec("rflego") is None and (_REPO_SRC / "rflego").is_dir():
    sys.path.insert(0, str(_REPO_SRC))

from rflego.config import BeamformerConfig, DetectorConfig, FrequencyTransformConfig  # noqa: E402
from rflego.utils import set_seed  # noqa: E402

# --------------------------------------------------------------------------- #
# Project paths (resolved relative to this file -> Colab/clone friendly)
# --------------------------------------------------------------------------- #
SCRIPTS_DIR = Path(__file__).resolve().parent
AE_DIR = SCRIPTS_DIR.parent
REPO_ROOT = AE_DIR.parent
AE_DATA_DIR = REPO_ROOT / "ae_data"
CONFIGS_DIR = AE_DIR / "configs"
RESULTS_DIR = AE_DIR / "results"
WEIGHTS_DIR = RESULTS_DIR / "weights"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "cache"
ENV_DIR = RESULTS_DIR / "env"

MODULES = ("ft", "beamformer", "detector")
MODALITY_LABELS = {
    "mmwave": "mmWave",
    "uwb": "UWB",
    "wifi": "WiFi",
}
TASK_LABELS = {
    "beamformer": "Beamformer",
    "detector": "Detector",
    "dopplerft": "Doppler FT",
    "rangeft": "Range FT",
    "ft": "FT",
}
_CONFIG_CLASSES = {
    "ft": FrequencyTransformConfig,
    "beamformer": BeamformerConfig,
    "detector": DetectorConfig,
}
AE_DATASET_IDS = (
    "mmwave_beamformer",
    "mmwave_dopplerft",
    "mmwave_rangeft",
    "uwb_detector",
    "uwb_dopplerft",
    "wifi_dopplerft",
)


@dataclasses.dataclass(frozen=True)
class BenchmarkSpec:
    """A single AE benchmark file discovered under ``ae_data``."""

    id: str
    modality: str
    modality_label: str
    task: str
    task_label: str
    module: str
    path: Path


def infer_module_from_npz(path: Path | str) -> str:
    """Infer the RF-LEGO module family from a benchmark ``.npz`` schema."""
    with np.load(path) as npz:
        keys = set(npz.files)
    if {"x_real", "x_imag", "gt_peak_bin"} <= keys:
        return "ft"
    if {"y_meas", "A_dict", "gt_angles_deg", "gt_angle_idx", "angle_grid_deg"} <= keys:
        return "beamformer"
    if {"x", "gt_mask"} <= keys:
        return "detector"
    raise ValueError(f"Could not infer AE module from {path}: keys={sorted(keys)}")


def _flat_benchmark_spec(path: Path) -> BenchmarkSpec:
    stem = path.stem
    if "_" in stem:
        modality, task = stem.split("_", 1)
    else:
        modality, task = "unknown", stem
    module = infer_module_from_npz(path)
    return BenchmarkSpec(
        id=stem,
        modality=modality,
        modality_label=MODALITY_LABELS.get(modality, modality),
        task=task,
        task_label=TASK_LABELS.get(task, task.replace("_", " ").title()),
        module=module,
        path=path,
    )


def discover_benchmarks() -> list[BenchmarkSpec]:
    """Return the current AE benchmark files in deterministic presentation order."""
    found = {p.stem: p for p in AE_DATA_DIR.glob("*.npz") if not p.name.startswith(".")}
    expected = set(AE_DATASET_IDS)
    missing = [f"{dataset_id}.npz" for dataset_id in AE_DATASET_IDS if dataset_id not in found]
    unexpected = sorted(dataset_id for dataset_id in found if dataset_id not in expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing required files: " + ", ".join(missing))
        if unexpected:
            details.append(
                "unexpected ae_data/*.npz files: "
                + ", ".join(f"{dataset_id}.npz" for dataset_id in unexpected)
            )
        raise FileNotFoundError("Invalid AE benchmark set under ae_data/; " + "; ".join(details))
    return [_flat_benchmark_spec(found[dataset_id]) for dataset_id in AE_DATASET_IDS]


def benchmark_by_id() -> dict[str, BenchmarkSpec]:
    """Return discovered benchmark specs keyed by their output-safe id."""
    return {spec.id: spec for spec in discover_benchmarks()}


def ensure_dirs() -> None:
    """Create the runtime-output directories under ``ae/results``."""
    for d in (RESULTS_DIR, WEIGHTS_DIR, METRICS_DIR, FIGURES_DIR, CACHE_DIR, ENV_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def set_determinism(seed: int = 42) -> None:
    """Seed Python/NumPy/PyTorch and request deterministic algorithms."""
    set_seed(seed)
    if os.environ.get("RFLEGO_STRICT_DETERMINISM") == "1":
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_device() -> torch.device:
    """AE evaluation always runs on CPU for stable, portable results."""
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# YAML config loading
# --------------------------------------------------------------------------- #
def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_config(module: str, cfg: dict[str, Any]):
    """Build the model dataclass for ``module`` from a loaded YAML dict.

    Only keys matching the dataclass fields are passed through, so the YAML may
    carry extra ``train``/``data`` sections without breaking construction.
    """
    cls = _CONFIG_CLASSES[module]
    model_cfg = cfg.get("model", {})
    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in model_cfg.items() if k in fields})


def module_config_path(module: str) -> Path:
    """Default AE config path for a module (``ae/configs/<module>.yaml``)."""
    return CONFIGS_DIR / f"{module}.yaml"


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def read_json(path: Path | str) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path | str, obj: Any) -> None:
    """Write JSON with sorted keys and NumPy-aware encoding."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def environment_info() -> dict[str, Any]:
    """Capture a reproducibility snapshot of the runtime environment."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
        "seed": 42,
    }
