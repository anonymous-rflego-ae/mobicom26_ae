"""Training entry point for the RF-LEGO AE modules.

Self-contained, seeded (seed=42 by default) training loops for users who want
to download the artifact and train their own RF-LEGO weights locally. The
one-click AE notebook loads the shipped pretrained weights instead. The scalar
optimizer/split/dropout settings match paper Sec. 4; the compact AE generators,
model sizes, and explicit step counts are documented artifact choices.

Training uses deterministic synthesized training samples from ``rflego.data``.
Evaluation remains on the prepared real-world held-out AE benchmark loaded by
``evaluate.py``.

Usage:
    python ae/scripts/train_models.py --module all
    python ae/scripts/train_models.py --module ft --steps 300   # step override
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

from rflego.config import BeamformerConfig, DetectorConfig, FrequencyTransformConfig  # noqa: E402
from rflego.data import (  # noqa: E402
    MODULE_ID,
    generate_beamformer_batch,
    generate_detector_batch,
    generate_frequency_transform_batch,
    make_rng,
)
from rflego.modules import BeamformerModel, DetectorModel, FrequencyTransformModel  # noqa: E402
from rflego.utils import count_parameters  # noqa: E402


def _split_indices(
    n_total: int,
    val_fraction: float,
    module: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the paper's deterministic, disjoint 80/20 train-validation split."""
    if n_total < 2:
        raise ValueError("n_total must be at least 2")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed + 2000 + MODULE_ID[module])
    order = rng.permutation(n_total)
    n_val = int(round(n_total * val_fraction))
    n_val = min(max(n_val, 1), n_total - 1)
    val_idx = np.sort(order[:n_val]).astype(np.int64)
    train_idx = np.sort(order[n_val:]).astype(np.int64)
    return train_idx, val_idx


def _batch_indices(indices: np.ndarray, batch: int, module: str, seed: int):
    """Deterministic infinite stream of shuffled mini-batches over train indices."""
    if batch <= 0:
        raise ValueError("batch size must be positive")
    rng = np.random.default_rng(seed + 1000 + MODULE_ID[module])
    while True:
        order = indices[rng.permutation(len(indices))]
        for i in range(0, len(order), batch):
            yield order[i : i + batch]


def _cosine_loss(mag: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-invariant shape loss: 1 - mean cosine similarity to the target."""
    return (1.0 - F.cosine_similarity(mag, target, dim=1)).mean()


def _overrides(cfg: dict, args) -> dict:
    train = dict(cfg.get("train", {}))
    if args.steps is not None:
        train["steps"] = args.steps
    if args.n_total is not None:
        train["n_total"] = args.n_total
    return train


def _seed(cfg: dict, args) -> int:
    if args.seed is not None:
        return int(args.seed)
    return int(cfg.get("train", {}).get("seed", 42))


def _resolve_device(name: str) -> torch.device:
    """Resolve a training device without silently selecting unsupported MPS complex ops."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is unavailable")
        raise RuntimeError("MPS does not support the complex-valued FT/Beamformer training path")
    return torch.device(name)


def _index_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _paper_split(train_cfg: dict, module: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return _split_indices(
        int(train_cfg["n_total"]),
        float(train_cfg["val_fraction"]),
        module,
        seed,
    )


def _make_optimizer(model: torch.nn.Module, train_cfg: dict) -> torch.optim.AdamW:
    """Build the AdamW optimizer specified by paper Sec. 4."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )


LossForIndices = Callable[[torch.Tensor], torch.Tensor]


@torch.no_grad()
def _validation_loss(
    model: torch.nn.Module,
    loss_for_indices: LossForIndices,
    val_idx: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    """Evaluate mean validation loss over every held-out synthetic frame."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_count = 0
    for start in range(0, len(val_idx), batch_size):
        idx_np = val_idx[start : start + batch_size]
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        loss = loss_for_indices(idx)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite validation loss")
        total_loss += float(loss.item()) * len(idx_np)
        total_count += len(idx_np)
    model.train(was_training)
    return total_loss / total_count


def _train_with_validation(
    *,
    module: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_for_indices: LossForIndices,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_cfg: dict,
    seed: int,
    device: torch.device,
    verbose: bool,
) -> dict:
    """Step-based optimization with epoch-boundary validation and best-state selection."""
    steps = int(train_cfg["steps"])
    batch_size = int(train_cfg["batch_size"])
    if steps <= 0:
        raise ValueError("steps must be positive")

    batches = _batch_indices(train_idx, batch_size, module, seed)
    steps_per_epoch = math.ceil(len(train_idx) / batch_size)
    val_every = int(train_cfg.get("val_every_steps", steps_per_epoch))
    if val_every <= 0:
        raise ValueError("val_every_steps must be positive")

    history: list[float] = []
    val_history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_step = -1
    best_state: dict[str, torch.Tensor] | None = None
    log_every = max(1, min(200, val_every))

    for step in range(steps):
        model.train()
        idx = torch.as_tensor(next(batches), dtype=torch.long, device=device)
        loss = loss_for_indices(idx)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{module}: non-finite training loss at step {step + 1}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.item()))

        should_validate = (step + 1) % val_every == 0 or step + 1 == steps
        if should_validate:
            val_loss = _validation_loss(model, loss_for_indices, val_idx, batch_size, device)
            val_history.append({"step": step + 1, "loss": val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step + 1
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

        if verbose and ((step + 1) % log_every == 0 or step == 0 or step + 1 == steps):
            suffix = ""
            if should_validate:
                suffix = f"  val {val_history[-1]['loss']:.4f}"
            print(f"  [{module}] step {step + 1:4d}/{steps}  loss {loss.item():.4f}{suffix}")

    if best_state is None:
        raise RuntimeError(f"{module}: validation did not produce a checkpoint")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "val_history": val_history,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "params": count_parameters(model),
        "split": {
            "n_total": len(train_idx) + len(val_idx),
            "n_train": len(train_idx),
            "n_validation": len(val_idx),
            "train_indices_sha256": _index_sha256(train_idx),
            "validation_indices_sha256": _index_sha256(val_idx),
        },
    }


# --------------------------------------------------------------------------- #
# Per-module training
# --------------------------------------------------------------------------- #
def train_ft(cfg: dict, args, device, verbose: bool) -> dict:
    seed = _seed(cfg, args)
    C.set_determinism(seed)
    m = cfg["model"]
    t = _overrides(cfg, args)
    N = int(m["sequence_length"])
    model = FrequencyTransformModel(
        FrequencyTransformConfig(
            sequence_length=N,
            num_conv_layers=int(m["num_conv_layers"]),
            dropout=float(m["dropout"]),
            device=str(device),
        )
    ).to(device)

    data = generate_frequency_transform_batch(
        int(t["n_total"]), make_rng("ft", seed), N=N, target_sigma=float(t["target_sigma"])
    )
    xr = torch.from_numpy(data["x_real"]).to(device)
    xi = torch.from_numpy(data["x_imag"]).to(device)
    target = torch.from_numpy(data["gt_spectrum"]).to(device)
    offpeak = (target < 0.05).float()

    train_idx, val_idx = _paper_split(t, "ft", seed)
    opt = _make_optimizer(model, t)
    sidelobe = float(t["sidelobe_l1"])

    def loss_for_indices(idx: torch.Tensor) -> torch.Tensor:
        yr, yi = model(xr[idx], xi[idx])
        mag = torch.sqrt(yr.squeeze(1) ** 2 + yi.squeeze(1) ** 2 + 1e-12)
        magn = mag / (mag.amax(dim=1, keepdim=True) + 1e-8)
        return _cosine_loss(mag, target[idx]) + sidelobe * (magn * offpeak[idx]).mean()

    return _train_with_validation(
        module="ft",
        model=model,
        optimizer=opt,
        loss_for_indices=loss_for_indices,
        train_idx=train_idx,
        val_idx=val_idx,
        train_cfg=t,
        seed=seed,
        device=device,
        verbose=verbose,
    )


def train_beamformer(cfg: dict, args, device, verbose: bool) -> dict:
    seed = _seed(cfg, args)
    C.set_determinism(seed)
    m = cfg["model"]
    t = _overrides(cfg, args)
    Mel = int(cfg.get("data", {}).get("M", 16))
    model = BeamformerModel(
        BeamformerConfig(
            dict_length=int(m["dict_length"]),
            num_layers=int(m["num_layers"]),
            dropout=float(m["dropout"]),
            device=str(device),
        )
    ).to(device)

    data = generate_beamformer_batch(
        int(t["n_total"]),
        make_rng("beamformer", seed),
        M=Mel,
        target_sigma=float(t["target_sigma"]),
    )
    y = torch.from_numpy(data["y_meas"]).to(device)
    # The steering geometry is identical for every synthesized frame. Keep one
    # dictionary and expand it as a view per mini-batch instead of retaining a
    # second 30,000 x M x D copy on the training device.
    A = torch.from_numpy(data["A_dict"][0].copy()).to(device)
    del data["A_dict"]
    target = torch.from_numpy(data["gt_spectrum"]).to(device)

    train_idx, val_idx = _paper_split(t, "beamformer", seed)
    opt = _make_optimizer(model, t)
    l1 = float(t["l1"])

    def loss_for_indices(idx: torch.Tensor) -> torch.Tensor:
        A_batch = A.unsqueeze(0).expand(idx.shape[0], -1, -1)
        z = model(y[idx], A_batch)
        mag = z.abs()
        magn = mag / (mag.amax(dim=1, keepdim=True) + 1e-8)
        return _cosine_loss(mag, target[idx]) + l1 * magn.mean()

    return _train_with_validation(
        module="beamformer",
        model=model,
        optimizer=opt,
        loss_for_indices=loss_for_indices,
        train_idx=train_idx,
        val_idx=val_idx,
        train_cfg=t,
        seed=seed,
        device=device,
        verbose=verbose,
    )


def train_detector(cfg: dict, args, device, verbose: bool) -> dict:
    seed = _seed(cfg, args)
    C.set_determinism(seed)
    m = cfg["model"]
    t = _overrides(cfg, args)
    L = int(cfg.get("data", {}).get("L", 256))
    model = DetectorModel(
        DetectorConfig(
            hidden_dim=int(m["hidden_dim"]),
            order=int(m["order"]),
            num_layers=int(m["num_layers"]),
            channels=int(m["channels"]),
            dt_min=float(m["dt_min"]),
            dt_max=float(m["dt_max"]),
            dropout=float(m["dropout"]),
            device=str(device),
        )
    ).to(device)

    data = generate_detector_batch(int(t["n_total"]), make_rng("detector", seed), L=L)
    x = torch.from_numpy(data["x"]).to(device)  # [L, n]
    mask = torch.from_numpy(data["gt_mask"]).to(device)  # [L, n]

    pos_weight = torch.tensor(float(t["pos_weight"]), device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_idx, val_idx = _paper_split(t, "detector", seed)
    opt = _make_optimizer(model, t)

    def loss_for_indices(idx: torch.Tensor) -> torch.Tensor:
        logits = model(x[:, idx])  # [L, B]
        return loss_fn(logits, mask[:, idx])

    return _train_with_validation(
        module="detector",
        model=model,
        optimizer=opt,
        loss_for_indices=loss_for_indices,
        train_idx=train_idx,
        val_idx=val_idx,
        train_cfg=t,
        seed=seed,
        device=device,
        verbose=verbose,
    )


_TRAINERS = {"ft": train_ft, "beamformer": train_beamformer, "detector": train_detector}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train RF-LEGO AE module weights.")
    ap.add_argument("--module", default="all", choices=["all", *C.MODULES])
    ap.add_argument("--steps", type=int, default=None, help="Override training steps (quick demo)")
    ap.add_argument("--n-total", type=int, default=None, help="Override synthesized frames per module")
    ap.add_argument("--seed", type=int, default=None, help="Override training seed")
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--out", default=str(C.WEIGHTS_DIR), help="Weights output directory")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    modules = C.MODULES if args.module == "all" else (args.module,)
    verbose = not args.quiet

    for module in modules:
        cfg = C.load_yaml(C.module_config_path(module))
        print(f"\n=== training {module} ===")
        t0 = time.time()
        res = _TRAINERS[module](cfg, args, device, verbose)
        dt = time.time() - t0
        model = res["model"].eval()
        wpath = out_dir / f"{module}.pt"
        temporary_wpath = out_dir / f".{module}.pt.tmp"
        torch.save(model.state_dict(), temporary_wpath)
        temporary_wpath.replace(wpath)
        train_cfg = _overrides(cfg, args)
        C.write_json(
            C.METRICS_DIR / f"{module}_train.json",
            {
                "module": module,
                "seed": _seed(cfg, args),
                "device": str(device),
                "optimizer": "AdamW",
                "train_config": train_cfg,
                "model_config": cfg["model"],
                "split": res["split"],
                "loss_history": res["history"],
                "validation_history": res["val_history"],
                "last_training_loss": res["history"][-1],
                "best_validation_loss": res["best_val_loss"],
                "best_step": res["best_step"],
                "saved_checkpoint": "best_validation",
                "params": res["params"],
                "train_seconds": dt,
                "weights_file": str(wpath),
            },
        )
        print(
            f"  -> {wpath}  | params={res['params']:,} | "
            f"best_val={res['best_val_loss']:.4f}@{res['best_step']} | {dt:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
