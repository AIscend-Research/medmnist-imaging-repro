"""Recompute the extension analyses from a saved prediction matrix — no GPU.

Every extension that consumes only ``(y_true, y_score)`` — per-class ROC/PR,
confusion matrix, frequency-vs-performance, calibration/ECE, confidence — can be
regenerated from a score CSV that a *previous* run already dumped. That turns an
hour of GPU into a few seconds of CPU, and it is how the archived
``results/reproduction_authors_code/`` runs get the same treatment as new ones
without retraining them.

Score CSVs are in the MedMNIST convention written by both the authors' script
and our :func:`src.evaluate.save_predictions`: no header, column 0 is the row
index, columns 1..C are class scores.

    python -m src.from_predictions \\
        --scores "results/reproduction_authors_code/personB_resnet50_derma224/output/dermamnist/260720_082717/dermamnist_test_[AUC]0.912_[ACC]0.738@run1.csv" \\
        --dataset dermamnist --tag personB_resnet50_224 \\
        --out results/reproduction_authors_code/personB_resnet50_derma224/recomputed

Needs the analysis stack (numpy/pandas/sklearn/scipy/matplotlib/medmnist) but
never touches a GPU. A CPU-only torch wheel is sufficient.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

# Tolerance for the self-check against the AUC/ACC baked into the filename.
# Loose enough for the 3-decimal rounding in the name, tight enough to catch a
# row-order mismatch (which would move AUC by far more than this).
FILENAME_TOL = 1e-3

_FNAME_RE = re.compile(r"\[AUC\](?P<auc>[0-9.]+)_\[ACC\](?P<acc>[0-9.]+)")


def load_scores(path):
    """Read a MedMNIST-convention score CSV into an ``(N, C)`` float array."""
    raw = np.loadtxt(path, delimiter=",", dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"{path}: expected an (N, 1+C) matrix, got {raw.shape}")
    index, scores = raw[:, 0], raw[:, 1:]
    expected = np.arange(len(index))
    if not np.array_equal(index.astype(int), expected):
        raise ValueError(
            f"{path}: column 0 is not a 0..N-1 row index — the file may have a "
            "header row or be in a different format")
    return scores


def load_labels(dataset, split="test", root=None, download=True):
    """Ground-truth labels for ``split``, in npz order (the dump order)."""
    from .data import get_dataset
    ds = get_dataset(dataset, split, 28, root=root, download=download)
    return np.asarray(ds.labels)


def metrics_from_filename(path):
    """The ``[AUC]x_[ACC]y`` pair baked into the filename, or ``None``."""
    m = _FNAME_RE.search(os.path.basename(path))
    return (float(m.group("auc")), float(m.group("acc"))) if m else None


def verify_alignment(path, auc, acc, tol=FILENAME_TOL):
    """Check recomputed metrics against the filename's — catches row misalignment.

    Returns a dict describing the check (``status`` is ``ok``/``skipped``), and
    raises ``AssertionError`` on a real mismatch.
    """
    claimed = metrics_from_filename(path)
    if claimed is None:
        return dict(status="skipped", reason="no [AUC]/[ACC] in filename")
    c_auc, c_acc = claimed
    d_auc, d_acc = abs(auc - c_auc), abs(acc - c_acc)
    assert d_auc <= tol and d_acc <= tol, (
        f"recomputed metrics disagree with {os.path.basename(path)}: "
        f"AUC {auc:.4f} vs {c_auc:.3f} (Δ{d_auc:.4f}), "
        f"ACC {acc:.4f} vs {c_acc:.3f} (Δ{d_acc:.4f}). "
        "Most likely the scores and labels are not in the same order.")
    return dict(status="ok", filename_auc=c_auc, filename_acc=c_acc,
                delta_auc=d_auc, delta_acc=d_acc)


def analyze(dataset, scores_path, out_dir, split="test", tag="recomputed",
            root=None, download=True, figures=True):
    """Run every prediction-only extension and write CSVs + figures.

    Returns a summary dict (also written to ``<out_dir>/summary.json``).
    """
    from . import extensions as ext
    from . import metrics as mx
    from .data import get_info, get_dataset

    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")

    y_score = load_scores(scores_path)
    y_true = load_labels(dataset, split, root=root, download=download)
    if len(y_true) != len(y_score):
        raise ValueError(
            f"{len(y_score)} score rows but {len(y_true)} {split} labels for "
            f"{dataset} — wrong split or wrong dataset?")

    task = get_info(dataset)["task"]
    auc, acc = mx.evaluate(y_true, y_score, task)
    alignment = verify_alignment(scores_path, auc, acc)
    print(f"[{tag}] AUC={auc:.4f} ACC={acc:.4f}  (alignment check: {alignment['status']})")

    summary = dict(dataset=dataset, split=split, tag=tag, task=task,
                   n=int(len(y_true)), auc=float(auc), acc=float(acc),
                   scores_path=os.path.relpath(scores_path),
                   alignment=alignment)

    # Per-class table + confusion matrix + per-class ROC.
    table = ext.per_class_analysis(dataset, y_true, y_score, out_dir, tag=tag)
    print(table.to_string(index=False))

    # Calibration / ECE (writes calibration_ece.csv).
    cal = ext.calibration_analysis(dataset, y_true, y_score, out_dir=out_dir,
                                   root=root, download=download)
    summary["ece"] = {k: float(cal[k]["ece"]) for k in ("overall", "common", "rare")}
    summary["rare_classes"] = cal["rare_classes"]

    if figures:
        os.makedirs(fig_dir, exist_ok=True)
        freq = ext.frequency_performance(dataset, table, fig_dir,
                                         stem=f"ext_freq_perf_{tag}",
                                         root=root, download=download)
        summary["frequency_performance"] = {
            k: float(v) for k, v in freq.items() if k != "png"}
        ext.plot_pr_curves(dataset, y_true, y_score, fig_dir,
                           stem=f"ext_pr_curves_{tag}")
        ext.plot_per_class_performance(dataset, table, fig_dir,
                                       stem=f"ext_per_class_perf_{tag}",
                                       root=root, download=download)
        ext.plot_class_distribution(dataset, fig_dir,
                                    stem=f"ext_class_distribution_{tag}",
                                    root=root, download=download)
        ext.plot_calibration(dataset, y_true, y_score, fig_dir,
                             stem=f"ext_calibration_{tag}",
                             root=root, download=download)
        # The misclassification gallery needs the images themselves, not just
        # the scores — but they come straight from the npz, so this is still
        # GPU-free.
        images = get_dataset(dataset, split, 28, root=root, download=download).imgs
        ext.plot_misclassified_gallery(dataset, y_true, y_score, images, fig_dir,
                                       stem=f"ext_misclassified_{tag}",
                                       root=root, download=download)
        print(f"[{tag}] figures -> {fig_dir}")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{tag}] artifacts -> {out_dir}")
    return summary


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scores", required=True, help="path to the score CSV")
    p.add_argument("--dataset", default="dermamnist")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--tag", default="recomputed", help="suffix for output filenames")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--root", default=None, help="medmnist data root")
    p.add_argument("--no-download", dest="download", action="store_false")
    p.add_argument("--no-figures", dest="figures", action="store_false")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    analyze(a.dataset, a.scores, a.out, split=a.split, tag=a.tag,
            root=a.root, download=a.download, figures=a.figures)


if __name__ == "__main__":
    main()
