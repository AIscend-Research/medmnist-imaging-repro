"""Zero-GPU-cost follow-up analyses against an already-trained checkpoint.

Runs the threshold-tuned baseline control and the (bug-fixed) corruption
robustness-by-rarity sweep as pure inference against a saved DermaMNIST
``best_model.pth`` -- no training, so this runs fine on CPU (a Kaggle
CPU-only session or a local machine both work, and neither touches GPU
quota).

Uses the real pipeline (``src.models``/``src.data``/``src.extensions``) so
it matches the checkpoint's actual architecture -- the CIFAR-style stem at
28x28 -- rather than a mismatched stock torchvision ResNet at 224 the way
an earlier version of this script did.

Usage:
    python scripts/cpu_followup_analysis.py \
        --checkpoint results/replication/dermamnist_resnet18_s28_seed0/best_model.pth

Output lands in results/extension/ (or --out), matching the filenames the
notebook's extensions section would produce, so it's a drop-in supplement.
"""
from __future__ import annotations

import argparse
import os

import torch

from src import data as dm
from src import evaluate as ev
from src import extensions as ext
from src import metrics as mx
from src import models as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to a best_model.pth from results/replication/dermamnist_resnet18_s28_seed*/")
    ap.add_argument("--dataset", default="dermamnist")
    ap.add_argument("--size", type=int, default=28)
    ap.add_argument("--width-mult", type=float, default=1.0,
                    help="Must match the width the checkpoint was trained with.")
    ap.add_argument("--out", default="results/extension")
    args = ap.parse_args()

    device = "cpu"
    os.makedirs(args.out, exist_ok=True)

    info = dm.get_info(args.dataset)
    n_classes = len(info["label"])
    model = M.build_model("resnet18", args.size, n_classes, width_mult=args.width_mult).to(device)
    ev.load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    train_loader, val_loader, test_loader = dm.get_loaders(
        args.dataset, args.size, batch_size=128, pin_memory=False)

    print("Running baseline inference on test set (CPU)...")
    yt, ys = ev.predict(model, test_loader, info["task"], device)
    auc, acc = mx.evaluate(yt, ys, info["task"])
    print(f"Baseline: AUC={auc:.4f} Acc={acc:.4f}")

    # --- Threshold-tuned baseline control (no retraining) ---
    print("\nTuning per-class thresholds on validation set...")
    yt_val, ys_val = ev.predict(model, val_loader, info["task"], device)
    thresholds = ext.tune_thresholds(yt_val, ys_val, n_classes)
    names = ext._labels(args.dataset)
    for n, t in zip(names, thresholds):
        print(f"  {n:<45}: {t:.2f}")

    yp_tuned = ext.apply_thresholds(ys, thresholds)
    tuned_table = ext.per_class_table_from_preds(args.dataset, yt, yp_tuned, ys)
    tuned_table.to_csv(os.path.join(args.out, "perclass_threshold_tuned.csv"), index=False)
    tuned_summary = ext.aggregate_from_table(tuned_table)
    print("\n=== Threshold-tuned baseline (same model, no retraining) ===")
    print(tuned_summary)
    print(tuned_table.to_string(index=False))

    # --- Robustness-by-rarity sweep (uses the fixed common/rare AUC logic) ---
    print("\nRunning corruption sweep on test set (CPU, inference-only)...")
    test_imgs = dm.get_dataset(args.dataset, "test", args.size).imgs
    rob = ext.robustness_eval(model, args.dataset, test_imgs, yt, device=device)
    rob.to_csv(os.path.join(args.out, "robustness.csv"), index=False)
    ext.plot_robustness(rob, args.out)
    print(rob.to_string(index=False))

    print(f"\nSaved: {args.out}/perclass_threshold_tuned.csv")
    print(f"Saved: {args.out}/robustness.csv")
    print(f"Saved: {args.out}/figures/ext_robustness.png")


if __name__ == "__main__":
    main()
