"""Figures for the ReScience replication paper.

Every figure is generated from the saved run artifacts under ``results/`` (each
run's ``run.json``) — never from hardcoded numbers — and written as both a PDF
and a 300-dpi PNG into ``report/figures/`` via :mod:`src.plotting`. Regenerating
the figures therefore never requires re-running training.

Figures:

1. ``fig1_reproduction`` — (a) our test AUC/ACC vs the paper's, scatter with a
   y=x line and per-point error bars across the twelve R18@28 datasets; plus
   (b) the DermaMNIST four-config grouped bars (ours vs paper, AUC and ACC).
2. ``fig2_training_curves`` — validation AUC and training loss vs epoch for a
   representative run, with vertical markers at the LR-decay epochs.
3. ``fig3_seed_variance`` — strip/box of test AUC across seeds for the
   multi-seed datasets.
4. ``fig4_delta_heatmap`` (optional) — datasets x {AUC, ACC} signed deltas.
5. ``fig5_compute_footprint`` — per-config training time / throughput and peak
   GPU memory, read straight from the ``run.json`` logs (documents cost).
"""

from __future__ import annotations

import os
import glob
import json

import numpy as np

from . import plotting as P
from .reference import REFERENCE


# --------------------------------------------------------------------------- #
# Loading / grouping saved runs (side-effect free)
# --------------------------------------------------------------------------- #

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


def group_baselines(results_dir="results"):
    """Group baseline runs by ``(dataset, model, size)`` with mean/std/seeds."""
    groups = {}
    for r in load_runs(results_dir):
        if not _is_baseline(r):
            continue
        c = r["config"]
        key = (c["dataset"], c["model"], c["size"])
        groups.setdefault(key, []).append(r)

    stats = {}
    for key, g in groups.items():
        aucs = np.array([r["test_auc"] for r in g], dtype=float)
        accs = np.array([r["test_acc"] for r in g], dtype=float)
        ref = REFERENCE.get(key)
        stats[key] = dict(
            n=len(g), aucs=aucs, accs=accs,
            auc_mean=float(aucs.mean()), auc_std=float(aucs.std(ddof=0)),
            acc_mean=float(accs.mean()), acc_std=float(accs.std(ddof=0)),
            ref_auc=(ref[0] if ref else None), ref_acc=(ref[1] if ref else None),
        )
    return stats


def _short(dataset):
    return dataset.replace("mnist", "")


# --------------------------------------------------------------------------- #
# Figure 1 — reproduction comparison
# --------------------------------------------------------------------------- #

def _scatter_panel(ax, stats, metric):
    """One ours-vs-paper scatter (metric in {'auc','acc'}) over R18@28 datasets."""
    mean_key, std_key, ref_key = f"{metric}_mean", f"{metric}_std", f"ref_{metric}"
    xs, ys, es, names = [], [], [], []
    for (dataset, model, size), s in stats.items():
        if model != "resnet18" or size != 28 or s[ref_key] is None:
            continue
        xs.append(s[ref_key]); ys.append(s[mean_key])
        es.append(s[std_key]); names.append(_short(dataset))
    if not xs:
        ax.text(0.5, 0.5, "no R18@28 runs yet", ha="center", va="center",
                transform=ax.transAxes)
        return
    xs, ys, es = np.array(xs), np.array(ys), np.array(es)

    lo = min(xs.min(), ys.min()) - 0.03
    hi = 1.005
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", zorder=0)
    ax.errorbar(xs, ys, yerr=es, fmt="o", ms=4, color=P.PALETTE[0],
                ecolor=P.PALETTE[0], elinewidth=0.8, capsize=2, zorder=2)
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(3, 3),
                    fontsize=5.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"paper {metric.upper()}")
    ax.set_ylabel(f"our {metric.upper()}")
    ax.set_title(f"{metric.upper()} — R18 @ 28 (12 datasets)")


def _derma_bars(ax, stats, metric):
    """DermaMNIST four-config grouped bars: ours vs paper for one metric."""
    configs = [("resnet18", 28), ("resnet18", 224), ("resnet50", 28), ("resnet50", 224)]
    labels, ours, errs, paper = [], [], [], []
    for model, size in configs:
        s = stats.get(("dermamnist", model, size))
        labels.append(f"{model[-2:]}\n@{size}")
        if s is None:
            ours.append(np.nan); errs.append(0.0)
            ref = REFERENCE.get(("dermamnist", model, size))
            paper.append(ref[0 if metric == "auc" else 1] if ref else np.nan)
        else:
            ours.append(s[f"{metric}_mean"]); errs.append(s[f"{metric}_std"])
            paper.append(s[f"ref_{metric}"])
    x = np.arange(len(configs))
    ax.bar(x - 0.2, ours, 0.38, yerr=errs, capsize=2, color=P.PALETTE[0],
           label="ours")
    ax.bar(x + 0.2, paper, 0.38, color=P.PALETTE[1], label="paper")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(metric.upper())
    finite = [v for v in ours + paper if np.isfinite(v)]
    if finite:
        ax.set_ylim(min(finite) - 0.03, 1.0)
    ax.set_title(f"DermaMNIST — {metric.upper()}")


def fig_reproduction(results_dir="results", report_dir="report"):
    stats = group_baselines(results_dir)
    P.set_style()
    fig, axes = P.new_fig(width=P.COL_WIDTH, ncols=2, nrows=2,
                          height=P.COL_WIDTH)
    _scatter_panel(axes[0, 0], stats, "auc")
    _scatter_panel(axes[0, 1], stats, "acc")
    _derma_bars(axes[1, 0], stats, "auc")
    _derma_bars(axes[1, 1], stats, "acc")
    axes[1, 1].legend(loc="lower right")
    fig.suptitle("Replication vs MedMNIST v2 (Table 3)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return P.savefig(fig, "fig1_reproduction", report_dir)


# --------------------------------------------------------------------------- #
# Figure 2 — training curves for a representative run
# --------------------------------------------------------------------------- #

def _find_run(results_dir, dataset, model, size, seed):
    for r in load_runs(results_dir):
        c = r["config"]
        if (c["dataset"] == dataset and c["model"] == model
                and c["size"] == size and c.get("seed") == seed
                and _is_baseline(r)):
            return r
    return None


def fig_training_curves(results_dir="results", report_dir="report",
                        dataset="dermamnist", model="resnet18", size=28, seed=0):
    r = _find_run(results_dir, dataset, model, size, seed)
    P.set_style()
    fig, (ax1, ax2) = P.new_fig(width=P.COL_WIDTH * 1.05, ncols=2,
                                height=P.COL_WIDTH * 0.9)
    if r is None or not r.get("history"):
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "no run history yet", ha="center", va="center",
                    transform=ax.transAxes)
        fig.tight_layout()
        return P.savefig(fig, "fig2_training_curves", report_dir)

    hist = r["history"]
    ep = np.array([h["epoch"] for h in hist])
    val_auc = np.array([h["val_auc"] for h in hist])
    loss = np.array([h["train_loss"] for h in hist])
    epochs = r["config"].get("epochs", int(ep.max()) + 1)
    milestones = [int(0.5 * epochs), int(0.75 * epochs)]

    ax1.plot(ep, val_auc, color=P.PALETTE[0])
    ax1.set_xlabel("epoch"); ax1.set_ylabel("validation macro-AUC")
    ax1.set_title("validation AUC")

    ax2.plot(ep, loss, color=P.PALETTE[3])
    ax2.set_xlabel("epoch"); ax2.set_ylabel("training loss")
    ax2.set_title("training loss")

    for ax in (ax1, ax2):
        for m in milestones:
            ax.axvline(m, color="0.6", ls=":", lw=0.8)
    fig.suptitle(f"{_short(dataset)} {model} @ {size} (seed {seed}); "
                 f"LR decays at {milestones}", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return P.savefig(fig, "fig2_training_curves", report_dir)


# --------------------------------------------------------------------------- #
# Figure 3 — seed variance
# --------------------------------------------------------------------------- #

def fig_seed_variance(results_dir="results", report_dir="report"):
    stats = group_baselines(results_dir)
    multi = {k: v for k, v in stats.items() if v["n"] > 1}
    P.set_style()
    fig, ax = P.new_fig(width=P.COL_WIDTH * 1.6, height=P.COL_WIDTH * 0.9)
    if not multi:
        ax.text(0.5, 0.5, "no multi-seed runs yet", ha="center", va="center",
                transform=ax.transAxes)
        fig.tight_layout()
        return P.savefig(fig, "fig3_seed_variance", report_dir)

    keys = sorted(multi, key=lambda k: multi[k]["auc_mean"])
    labels, data = [], []
    for k in keys:
        dataset, model, size = k
        labels.append(f"{_short(dataset)}\n{model[-2:]}@{size}")
        data.append(multi[k]["aucs"])
    x = np.arange(len(keys))
    ax.boxplot(data, positions=x, widths=0.5, showfliers=False,
               medianprops=dict(color=P.PALETTE[1]))
    rng = np.random.RandomState(0)
    for i, d in enumerate(data):
        jitter = (rng.rand(len(d)) - 0.5) * 0.18
        ax.scatter(np.full(len(d), i) + jitter, d, s=12, color=P.PALETTE[0],
                   zorder=3, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("test macro-AUC")
    ax.set_title("Test AUC across seeds (multi-seed datasets)")
    fig.tight_layout()
    return P.savefig(fig, "fig3_seed_variance", report_dir)


# --------------------------------------------------------------------------- #
# Figure 4 (optional) — replication delta heatmap
# --------------------------------------------------------------------------- #

def fig_delta_heatmap(results_dir="results", report_dir="report"):
    stats = group_baselines(results_dir)
    rows = []
    for (dataset, model, size), s in stats.items():
        if model != "resnet18" or size != 28 or s["ref_auc"] is None:
            continue
        rows.append((_short(dataset),
                     s["auc_mean"] - s["ref_auc"],
                     s["acc_mean"] - s["ref_acc"]))
    P.set_style()
    fig, ax = P.new_fig(width=P.COL_WIDTH, height=P.COL_WIDTH * 1.6)
    if not rows:
        ax.text(0.5, 0.5, "no R18@28 runs yet", ha="center", va="center",
                transform=ax.transAxes)
        fig.tight_layout()
        return P.savefig(fig, "fig4_delta_heatmap", report_dir)

    rows.sort(key=lambda t: t[1])
    names = [r[0] for r in rows]
    mat = np.array([[r[1], r[2]] for r in rows])
    vmax = float(np.abs(mat).max()) or 0.02
    im = ax.imshow(mat, cmap=P.DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ΔAUC", "ΔACC"])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6)
    for i in range(len(names)):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center",
                    fontsize=5.5,
                    color="white" if abs(mat[i, j]) > vmax * 0.6 else "black")
    ax.set_title("ours − paper (R18 @ 28)")
    fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
    fig.tight_layout()
    return P.savefig(fig, "fig4_delta_heatmap", report_dir)


# --------------------------------------------------------------------------- #
# Figure 5 — compute footprint (from run.json logs; nearly free)
# --------------------------------------------------------------------------- #

def _config_cost(results_dir):
    """Per-config median epoch time, throughput, and peak GPU memory (seed 0)."""
    rows = {}
    for r in load_runs(results_dir):
        if not _is_baseline(r):
            continue
        c = r["config"]
        if c.get("seed", 0) != 0:
            continue
        hist = r.get("history", [])
        if not hist:
            continue
        ep_t = np.median([h.get("epoch_time_s", np.nan) for h in hist])
        ips = np.median([h.get("imgs_per_sec", np.nan) for h in hist])
        key = (c["dataset"], c["model"], c["size"])
        rows[key] = dict(
            epoch_time_s=float(ep_t), imgs_per_sec=float(ips),
            peak_gpu_mem_mb=r.get("peak_gpu_mem_mb"),
            wall_min=r.get("wall_clock_s", 0) / 60.0,
        )
    return rows


def fig_compute_footprint(results_dir="results", report_dir="report"):
    rows = _config_cost(results_dir)
    P.set_style()
    fig, (ax1, ax2) = P.new_fig(width=P.COL_WIDTH * 1.7, ncols=2,
                                height=P.COL_WIDTH * 1.0)
    if not rows:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "no run logs yet", ha="center", va="center",
                    transform=ax.transAxes)
        fig.tight_layout()
        return P.savefig(fig, "fig5_compute_footprint", report_dir)

    keys = sorted(rows, key=lambda k: rows[k]["epoch_time_s"])
    labels = [f"{_short(d)}\n{m[-2:]}@{s}" for (d, m, s) in keys]
    x = np.arange(len(keys))

    times = [rows[k]["epoch_time_s"] for k in keys]
    ax1.bar(x, times, color=P.PALETTE[0])
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=5.5, rotation=0)
    ax1.set_ylabel("median epoch time (s)")
    ax1.set_title("Training cost per config (seed 0)")

    mem = [rows[k]["peak_gpu_mem_mb"] for k in keys]
    if any(m is not None for m in mem):
        mem = [0.0 if m is None else m for m in mem]
        ax2.bar(x, mem, color=P.PALETTE[2])
        ax2.set_ylabel("peak GPU memory (MB)")
        ax2.set_title("Peak GPU memory per config")
    else:  # CPU runs never logged memory; fall back to throughput.
        ips = [rows[k]["imgs_per_sec"] for k in keys]
        ax2.bar(x, ips, color=P.PALETTE[2])
        ax2.set_ylabel("throughput (images/s)")
        ax2.set_title("Throughput per config")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=5.5)
    fig.tight_layout()
    return P.savefig(fig, "fig5_compute_footprint", report_dir)


# --------------------------------------------------------------------------- #

def generate_all(results_dir="results", report_dir="report"):
    """Generate every replication figure; return the list of PNG paths."""
    paths = [
        fig_reproduction(results_dir, report_dir),
        fig_training_curves(results_dir, report_dir),
        fig_seed_variance(results_dir, report_dir),
        fig_delta_heatmap(results_dir, report_dir),
        fig_compute_footprint(results_dir, report_dir),
    ]
    return paths


if __name__ == "__main__":
    import sys
    rd = sys.argv[1] if len(sys.argv) > 1 else "results"
    rep = sys.argv[2] if len(sys.argv) > 2 else "report"
    for p in generate_all(rd, rep):
        print("wrote", p)
