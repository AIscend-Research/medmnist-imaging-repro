"""Config-driven entry point for a single training run.

Usage (CLI):
    python -m src.run --dataset dermamnist --model resnet18 --size 28 --seed 0 \
        --epochs 100 --results-dir results

Or import :func:`run_config` from a notebook.
"""

from __future__ import annotations

import os
import io
import sys
import json
import time
import argparse
import platform
from dataclasses import dataclass, asdict, field

import numpy as np
import torch

from . import data as datamod
from . import metrics as metricsmod
from .models import build_model, count_parameters, model_size_mb
from .train import set_seed, run_training
from .evaluate import predict, save_predictions, prediction_filename
from .reference import REFERENCE


@dataclass
class RunConfig:
    dataset: str = "dermamnist"
    model: str = "resnet18"
    size: int = 28
    seed: int = 0
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    amp: bool = False
    deterministic: bool = True
    width_mult: float = 1.0
    # bias-mitigation flags (extensions)
    weighted_sampler: bool = False
    weighted_loss: bool = False
    num_workers: int = 2
    ckpt_every: int = 5
    resume: bool = True
    download: bool = True
    root: str = None
    results_dir: str = "results"
    tag: str = ""  # extra label folded into the run directory name

    def flag(self):
        return self.dataset

    def run_name(self):
        parts = [self.dataset, self.model, f"s{self.size}", f"seed{self.seed}"]
        if self.width_mult != 1.0:
            parts.append(f"w{self.width_mult}")
        if self.weighted_sampler:
            parts.append("wsampler")
        if self.weighted_loss:
            parts.append("wloss")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def _versions():
    import sklearn
    import torchvision
    import medmnist
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        torchvision=torchvision.__version__,
        medmnist=medmnist.__version__,
        numpy=np.__version__,
        sklearn=sklearn.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )


def run_config(cfg: RunConfig, log_fn=print, verify_metrics=True):
    """Execute one run end-to-end and write artifacts. Returns a result dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = datamod.get_info(cfg.dataset)
    task = info["task"]
    n_classes = len(info["label"])

    used_det = set_seed(cfg.seed, deterministic=cfg.deterministic)
    run_dir = os.path.join(cfg.results_dir, cfg.run_name())
    os.makedirs(run_dir, exist_ok=True)
    log_fn(f"=== {cfg.run_name()} | task={task} classes={n_classes} device={device} ===")

    sampler = datamod.make_weighted_sampler(cfg.dataset, cfg.root, cfg.download) \
        if cfg.weighted_sampler else None
    loaders = datamod.get_loaders(
        cfg.dataset, cfg.size, batch_size=cfg.batch_size, root=cfg.root,
        download=cfg.download, num_workers=cfg.num_workers, sampler=sampler,
        pin_memory=(device == "cuda"))

    class_weight = datamod.class_weights(cfg.dataset, cfg.root, cfg.download) \
        if cfg.weighted_loss else None

    model = build_model(cfg.model, cfg.size, n_classes, in_channels=3,
                        width_mult=cfg.width_mult)
    n_params = count_parameters(model)
    size_mb = model_size_mb(model)
    log_fn(f"model params={n_params:,} size={size_mb:.2f} MB")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = run_training(
        model, loaders, task, epochs=cfg.epochs, lr=cfg.lr, device=device,
        class_weight=class_weight, use_amp=cfg.amp, deterministic=cfg.deterministic,
        seed=cfg.seed, ckpt_dir=run_dir, ckpt_every=cfg.ckpt_every,
        resume=cfg.resume, log_fn=log_fn)
    wall = time.time() - t0
    peak_gpu_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                       if device == "cuda" else None)

    # Save test predictions in the medmnist-convention filename.
    pred_name = prediction_filename(cfg.flag(), "test", result["test_auc"],
                                    result["test_acc"], cfg.seed, size=cfg.size)
    save_predictions(result["test_y_score"], os.path.join(run_dir, pred_name))

    # Optional oracle agreement check on the real test predictions.
    agreement = None
    if verify_metrics:
        try:
            agreement = metricsmod.check_agreement(
                result["test_y_true"], result["test_y_score"], task,
                flag=cfg.flag(), split="test", size=28, root=cfg.root,
                tol=1e-3, verbose=False)
            log_fn(f"[verify] metrics agree with medmnist.Evaluator (auc={agreement['auc']:.4f})")
        except Exception as e:  # never let verification kill a completed run
            log_fn(f"[verify] WARNING agreement check skipped: {e}")

    ref = REFERENCE.get((cfg.dataset, cfg.model, cfg.size))
    run_json = dict(
        config=asdict(cfg),
        task=task, n_classes=n_classes,
        n_params=n_params, model_size_mb=size_mb,
        best_epoch=result["best_epoch"], best_val_auc=result["best_val_auc"],
        train_auc=result["train_auc"], train_acc=result["train_acc"],
        val_auc=result["val_auc"], val_acc=result["val_acc"],
        test_auc=result["test_auc"], test_acc=result["test_acc"],
        reference_auc=(ref[0] if ref else None),
        reference_acc=(ref[1] if ref else None),
        delta_auc=(result["test_auc"] - ref[0] if ref else None),
        delta_acc=(result["test_acc"] - ref[1] if ref else None),
        wall_clock_s=wall,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        deterministic_cudnn=used_det,
        amp=cfg.amp,
        seed=cfg.seed,
        prediction_file=pred_name,
        history=result["history"],
        agreement=agreement,
        versions=_versions(),
    )
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(run_json, f, indent=2)

    log_fn(f"DONE {cfg.run_name()} test_auc={result['test_auc']:.4f} "
           f"test_acc={result['test_acc']:.4f} "
           + (f"(paper {ref[0]:.3f}/{ref[1]:.3f})" if ref else "")
           + f" [{wall/60:.1f} min]")
    return run_json


# --------------------------------------------------------------------------- #
# Run-matrix enumeration (tiers)
# --------------------------------------------------------------------------- #

# The eleven non-Derma MedMNIST2D datasets are split by data scale. The small /
# medium ones get three seeds; the four large ones get a single seed (seed 0),
# because a single run is stable at that data scale and three would triple the
# cost. This split is a compute decision, disclosed in the writeup.
SMALL_MEDIUM_DATASETS = (
    "retinamnist", "breastmnist", "pneumoniamnist", "bloodmnist",
    "organamnist", "organcmnist", "organsmnist",
)
LARGE_DATASETS = ("tissuemnist", "octmnist", "pathmnist", "chestmnist")


def run_matrix(seeds=(0, 1, 2)):
    """Return the tiered list of ``(tier, dataset, model, size, seed)`` tuples.

    This is the compute-minimal slice of MedMNIST v2 Table 3 the replication
    targets:

    * **Tier 1** — DermaMNIST (primary), ResNet-18/50 x sizes 28/224, all
      ``seeds``. Twelve runs at three seeds.
    * **Tier 2** — the small/medium datasets at ResNet-18 @ 28, all ``seeds``.
    * **Tier 3** — the four large datasets at ResNet-18 @ 28, **seed 0 only**
      (single-seed for compute reasons; stable at that data scale).
    """
    matrix = []
    # Tier 1: DermaMNIST, all 4 model x size, all seeds.
    for model in ("resnet18", "resnet50"):
        for size in (28, 224):
            for s in seeds:
                matrix.append((1, "dermamnist", model, size, s))
    # Tier 2: small/medium datasets, ResNet-18 @ 28, all seeds.
    for dataset in SMALL_MEDIUM_DATASETS:
        for s in seeds:
            matrix.append((2, dataset, "resnet18", 28, s))
    # Tier 3: large datasets, ResNet-18 @ 28, single seed (seed 0).
    for dataset in LARGE_DATASETS:
        matrix.append((3, dataset, "resnet18", 28, 0))
    return matrix


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="MedMNIST v2 replication: single run")
    p.add_argument("--dataset", default="dermamnist")
    p.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--size", type=int, default=28, choices=[28, 224])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--weighted-sampler", action="store_true")
    p.add_argument("--weighted-loss", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--no-download", dest="download", action="store_false")
    p.add_argument("--root", default=None)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--tag", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    cfg = RunConfig(**vars(args))
    run_config(cfg)


if __name__ == "__main__":
    main()
