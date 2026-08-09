"""Aggregate per-run ``run.json`` files into mean±std tables vs the paper.

Emits ``report/comparison.csv`` and ``report/comparison.md``: one row per
(dataset, model, size) with our mean±std AUC/ACC, the paper's reference value,
the signed delta, and an out-of-tolerance flag.
"""

from __future__ import annotations

import os
import glob
import json
import numpy as np

from .reference import REFERENCE

# Tolerances from the task's definition of done.
AUC_TOL = 0.02
ACC_TOL = 0.03


def load_runs(results_dir="results"):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "run.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def _is_baseline(r):
    c = r["config"]
    return (c.get("width_mult", 1.0) == 1.0 and not c.get("weighted_sampler")
            and not c.get("weighted_loss") and not c.get("tag"))


def aggregate(results_dir="results", report_dir="report"):
    runs = [r for r in load_runs(results_dir) if _is_baseline(r)]
    groups = {}
    for r in runs:
        c = r["config"]
        key = (c["dataset"], c["model"], c["size"])
        groups.setdefault(key, []).append(r)

    rows = []
    for key in sorted(groups):
        dataset, model, size = key
        g = groups[key]
        aucs = np.array([r["test_auc"] for r in g])
        accs = np.array([r["test_acc"] for r in g])
        ref = REFERENCE.get(key)
        ref_auc, ref_acc = (ref if ref else (None, None))
        d_auc = float(aucs.mean() - ref_auc) if ref else None
        d_acc = float(accs.mean() - ref_acc) if ref else None
        flag = ""
        if ref:
            if abs(d_auc) > AUC_TOL or abs(d_acc) > ACC_TOL:
                flag = "OUT_OF_TOL"
        rows.append(dict(
            dataset=dataset, model=model, size=size, n_seeds=len(g),
            auc_mean=float(aucs.mean()), auc_std=float(aucs.std(ddof=0)),
            acc_mean=float(accs.mean()), acc_std=float(accs.std(ddof=0)),
            ref_auc=ref_auc, ref_acc=ref_acc,
            delta_auc=d_auc, delta_acc=d_acc, flag=flag,
        ))

    os.makedirs(report_dir, exist_ok=True)
    _write_csv(rows, os.path.join(report_dir, "comparison.csv"))
    _write_md(rows, os.path.join(report_dir, "comparison.md"))
    return rows


def _write_csv(rows, path):
    import csv
    fields = ["dataset", "model", "size", "n_seeds", "auc_mean", "auc_std",
              "acc_mean", "acc_std", "ref_auc", "ref_acc", "delta_auc",
              "delta_acc", "flag"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"


def _write_md(rows, path):
    lines = [
        "# MedMNIST v2 replication — comparison vs paper (Table 3)",
        "",
        "AUC / ACC reported as our mean ± std across seeds; delta = ours − paper.",
        f"Tolerance: |ΔAUC| ≤ {AUC_TOL}, |ΔACC| ≤ {ACC_TOL}. Flag marks configs outside it.",
        "",
        "| Dataset | Model | Size | Seeds | Our AUC | Paper AUC | ΔAUC | Our ACC | Paper ACC | ΔACC | Flag |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        our_auc = f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}"
        our_acc = f"{r['acc_mean']:.3f} ± {r['acc_std']:.3f}"
        lines.append(
            f"| {r['dataset']} | {r['model']} | {r['size']} | {r['n_seeds']} | "
            f"{our_auc} | {_fmt(r['ref_auc'])} | {_fmt(r['delta_auc'])} | "
            f"{our_acc} | {_fmt(r['ref_acc'])} | {_fmt(r['delta_acc'])} | "
            f"{r['flag']} |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import sys
    rd = sys.argv[1] if len(sys.argv) > 1 else "results"
    rows = aggregate(rd)
    for r in rows:
        print(r)
