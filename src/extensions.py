"""Extensions built on the corrected pipeline (kept separable from the baseline).

* Per-class analysis (per-class ROC-AUC, PR-AUC/average precision, P/R/F1,
  confusion matrix, per-class ROC and PR curves), with ±std across seeds.
* Bias mitigation comparison (baseline vs weighted sampler vs weighted loss).
* Lightweight variant profiling (params, MB, latency) + efficiency tradeoff.
* Confidence & calibration (reliability, ECE, per-class confidence).
* Corruption robustness (inference-only: Gaussian noise, JPEG, brightness).
* Class distribution / frequency-vs-performance / misclassification gallery.

All reuse ``src.models`` / ``src.data`` and only depend on saved predictions or
a trained model. New figures route through ``src.plotting`` (colorblind style,
PDF + 300-dpi PNG into ``report/figures/``).
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd

from medmnist import INFO
from . import metrics as metricsmod


def _labels(dataset):
    return [INFO[dataset]["label"][str(i)] for i in range(len(INFO[dataset]["label"]))]


# --------------------------------------------------------------------------- #
# Per-class analysis
# --------------------------------------------------------------------------- #

def per_class_table(dataset, y_true, y_score):
    n = len(INFO[dataset]["label"])
    rows = metricsmod.per_class_metrics(y_true, y_score, n)
    names = _labels(dataset)
    for r in rows:
        r["label"] = names[r["cls"]]
    return pd.DataFrame(rows)[
        ["cls", "label", "support", "auc", "ap", "precision", "recall", "f1"]]


def per_class_multiseed(dataset, preds_by_seed):
    """Aggregate per-class metrics across seeds into mean±std.

    ``preds_by_seed``: list of ``(y_true, y_score)`` tuples (one per seed).
    Returns a DataFrame with ``<metric>_mean`` / ``<metric>_std`` columns for
    auc / ap / precision / recall / f1, so the per-class figures can carry
    ±std error bars from the three DermaMNIST seeds.
    """
    metric_cols = ["auc", "ap", "precision", "recall", "f1"]
    per_seed = [per_class_table(dataset, yt, ys) for yt, ys in preds_by_seed]
    base = per_seed[0][["cls", "label", "support"]].copy()
    for m in metric_cols:
        stack = np.vstack([df[m].values for df in per_seed])
        base[f"{m}_mean"] = stack.mean(axis=0)
        base[f"{m}_std"] = stack.std(axis=0, ddof=0)
    return base


def plot_confusion_matrix(dataset, y_true, y_score, path, normalize=True, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    yt = np.asarray(y_true).squeeze()
    yp = np.argmax(np.asarray(y_score), axis=-1)
    n = len(INFO[dataset]["label"])
    cm = confusion_matrix(yt, yp, labels=list(range(n))).astype(float)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    names = _labels(dataset)

    fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.2 * n + 1))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=(1 if normalize else cm.max()))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title or f"{dataset} confusion matrix" + (" (row-normalized)" if normalize else ""))
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > (0.5 if normalize else cm.max() / 2) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_roc(dataset, y_true, y_score, path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    yt = np.asarray(y_true).squeeze()
    ys = np.asarray(y_score)
    n = len(INFO[dataset]["label"])
    names = _labels(dataset)

    fig, ax = plt.subplots(figsize=(6, 6))
    for c in range(n):
        yb = (yt == c).astype(int)
        if yb.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yb, ys[:, c])
        auc_c = roc_auc_score(yb, ys[:, c])
        ax.plot(fpr, tpr, lw=1.5, label=f"{names[c]} (AUC={auc_c:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title or f"{dataset} per-class ROC (one-vs-rest)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_class_analysis(dataset, y_true, y_score, out_dir, tag="baseline"):
    os.makedirs(out_dir, exist_ok=True)
    df = per_class_table(dataset, y_true, y_score)
    df.to_csv(os.path.join(out_dir, f"perclass_{tag}.csv"), index=False)
    plot_confusion_matrix(dataset, y_true, y_score,
                          os.path.join(out_dir, f"confusion_{tag}.png"))
    plot_per_class_roc(dataset, y_true, y_score,
                       os.path.join(out_dir, f"roc_{tag}.png"))
    return df


# --------------------------------------------------------------------------- #
# Bias mitigation comparison
# --------------------------------------------------------------------------- #

def bias_comparison(dataset, variants, out_dir):
    """Compare mitigation variants.

    ``variants``: dict ``name -> (y_true, y_score)``. Writes a per-class F1
    comparison figure and a summary CSV with aggregate + worst-class metrics,
    stating the equity/accuracy tradeoff.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    names = _labels(dataset)
    n = len(names)
    task = INFO[dataset]["task"]

    tables, summary = {}, []
    for name, (yt, ys) in variants.items():
        df = per_class_table(dataset, yt, ys)
        tables[name] = df
        auc, acc = metricsmod.evaluate(yt, ys, task)
        summary.append(dict(
            variant=name, auc=auc, acc=acc,
            macro_f1=float(df["f1"].mean()),
            min_class_f1=float(df["f1"].min()),
            min_class_recall=float(df["recall"].min()),
            worst_class=names[int(df["f1"].idxmin())],
        ))
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, "bias_summary.csv"), index=False)

    # Grouped per-class F1 bar chart.
    fig, ax = plt.subplots(figsize=(1.1 * n + 3, 5))
    width = 0.8 / max(len(variants), 1)
    x = np.arange(n)
    for i, (name, df) in enumerate(tables.items()):
        ax.bar(x + i * width, df["f1"].values, width=width, label=name)
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1"); ax.set_title(f"{dataset} per-class F1 by mitigation strategy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bias_perclass_f1.png"), dpi=150)
    plt.close(fig)
    return summary_df, tables


# --------------------------------------------------------------------------- #
# Lightweight variant profiling
# --------------------------------------------------------------------------- #

def profile_model(model, device="cuda", size=28, n_warmup=10, n_iter=50, batch=1):
    """Return params / MB / inference latency (ms/image)."""
    import torch
    from .models import count_parameters, model_size_mb

    model = model.to(device).eval()
    x = torch.randn(batch, 3, size, size, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
    ms_per_image = (dt / n_iter / batch) * 1000
    return dict(n_params=count_parameters(model), size_mb=model_size_mb(model),
                latency_ms_per_image=ms_per_image)


# --------------------------------------------------------------------------- #
# Report-level figures (aggregate across runs)
# --------------------------------------------------------------------------- #

def plot_training_curves(results_dir, out_path, dataset=None):
    """Validation macro-AUC over epochs for each baseline run (seed 0)."""
    import glob, json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for p in sorted(glob.glob(os.path.join(results_dir, "*", "run.json"))):
        r = json.load(open(p))
        c = r["config"]
        if c.get("seed", 0) != 0 or c.get("tag") or c.get("weighted_sampler") \
                or c.get("weighted_loss") or c.get("width_mult", 1.0) != 1.0:
            continue
        if dataset and c["dataset"] != dataset:
            continue
        hist = r.get("history", [])
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        va = [h["val_auc"] for h in hist]
        ax.plot(ep, va, lw=1.5, label=f"{c['dataset']} {c['model']} s{c['size']}")
        plotted += 1
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation macro-AUC")
    ax.set_title("Validation AUC over training (seed 0)")
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return plotted


def plot_comparison_bars(rows, out_path):
    """Grouped bars: our mean AUC/ACC vs paper reference, per config."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("ref_auc") is not None]
    if not rows:
        return 0
    labels = [f"{r['dataset'][:5]}\n{r['model'][-2:]} s{r['size']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.3 * len(rows)), 5))
    for ax, key, refkey, stdkey, title in [
        (axes[0], "auc_mean", "ref_auc", "auc_std", "AUC"),
        (axes[1], "acc_mean", "ref_acc", "acc_std", "ACC"),
    ]:
        ours = [r[key] for r in rows]
        ref = [r[refkey] for r in rows]
        err = [r[stdkey] for r in rows]
        ax.bar(x - 0.2, ours, 0.4, yerr=err, capsize=3, label="ours (mean±std)")
        ax.bar(x + 0.2, ref, 0.4, label="paper")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title); ax.legend(fontsize=8)
        ax.set_ylim(min(min(ours), min(ref)) - 0.03, 1.0)
    fig.suptitle("Replication vs MedMNIST v2 (Table 3)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return len(rows)


def rare_vs_common_degradation(dataset, full_table, light_table, out_dir):
    """Compare per-class F1 of full vs lightweight, ranked by class frequency."""
    os.makedirs(out_dir, exist_ok=True)
    merged = full_table[["cls", "label", "support", "f1"]].rename(columns={"f1": "f1_full"})
    merged = merged.merge(light_table[["cls", "f1"]].rename(columns={"f1": "f1_light"}), on="cls")
    merged["f1_drop"] = merged["f1_full"] - merged["f1_light"]
    merged = merged.sort_values("support")
    merged.to_csv(os.path.join(out_dir, "lightweight_degradation.csv"), index=False)
    # Correlation of frequency vs degradation: negative -> rare classes hurt more.
    corr = float(np.corrcoef(merged["support"], merged["f1_drop"])[0, 1]) \
        if len(merged) > 2 else float("nan")
    return merged, corr


# --------------------------------------------------------------------------- #
# Rare/common split (by training frequency) — used by calibration/robustness
# --------------------------------------------------------------------------- #

def rare_common_split(dataset, root=None, download=True):
    """Return ``(rare_classes, common_classes, counts)`` by training frequency.

    Classes with a below-median training count are 'rare'. For DermaMNIST this
    isolates the minority lesion classes (everything but the ~67% nevi bulk).
    """
    from .data import class_counts
    counts = class_counts(dataset, root=root, download=download)
    med = np.median(counts)
    rare = [int(c) for c in range(len(counts)) if counts[c] < med]
    common = [int(c) for c in range(len(counts)) if counts[c] >= med]
    return rare, common, counts


# --------------------------------------------------------------------------- #
# Per-class precision-recall curves (honest under imbalance)
# --------------------------------------------------------------------------- #

def plot_pr_curves(dataset, y_true, y_score, report_dir, stem="ext_pr_curves"):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from . import plotting as P

    yt = np.asarray(y_true).squeeze()
    ys = np.asarray(y_score)
    n = len(INFO[dataset]["label"])
    names = _labels(dataset)

    P.set_style()
    fig, ax = P.new_fig(width=P.COL_WIDTH * 1.3, height=P.COL_WIDTH * 1.2)
    for c in range(n):
        yb = (yt == c).astype(int)
        if yb.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(yb, ys[:, c])
        ap = average_precision_score(yb, ys[:, c])
        ax.plot(rec, prec, lw=1.3,
                color=P.PALETTE[c % len(P.PALETTE)],
                label=f"{names[c]} (AP={ap:.3f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{dataset} per-class PR (one-vs-rest)")
    ax.legend(fontsize=5.5, loc="lower left")
    fig.tight_layout()
    return P.savefig(fig, stem, report_dir)


# --------------------------------------------------------------------------- #
# Confidence & calibration (inference-only)
# --------------------------------------------------------------------------- #

def _reliability(confidences, correct, n_bins=15):
    """Binned reliability curve + expected calibration error."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    N = max(len(confidences), 1)
    ece = 0.0
    centers, bin_conf, bin_acc, bin_n = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask |= confidences <= lo  # include the very smallest confidences
        centers.append((lo + hi) / 2)
        if mask.sum() == 0:
            bin_conf.append(np.nan); bin_acc.append(np.nan); bin_n.append(0)
            continue
        conf = confidences[mask].mean()
        acc = correct[mask].mean()
        bin_conf.append(conf); bin_acc.append(acc); bin_n.append(int(mask.sum()))
        ece += (mask.sum() / N) * abs(acc - conf)
    return dict(centers=np.array(centers), conf=np.array(bin_conf),
                acc=np.array(bin_acc), n=np.array(bin_n), ece=float(ece))


def calibration_analysis(dataset, y_true, y_score, out_dir=None, n_bins=15,
                         root=None, download=True):
    """Overall + common/rare reliability and ECE. Returns a dict of curves."""
    yt = np.asarray(y_true).squeeze()
    ys = np.asarray(y_score)
    conf = ys.max(axis=1)
    pred = ys.argmax(axis=1)
    correct = (pred == yt).astype(float)

    rare, common, _ = rare_common_split(dataset, root=root, download=download)
    rare_mask = np.isin(yt, rare)
    common_mask = np.isin(yt, common)

    out = dict(
        overall=_reliability(conf, correct, n_bins),
        common=_reliability(conf[common_mask], correct[common_mask], n_bins),
        rare=_reliability(conf[rare_mask], correct[rare_mask], n_bins),
        rare_classes=rare, common_classes=common,
    )
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(dict(
            split=["overall", "common", "rare"],
            ece=[out["overall"]["ece"], out["common"]["ece"], out["rare"]["ece"]],
        )).to_csv(os.path.join(out_dir, "calibration_ece.csv"), index=False)
    return out


def plot_calibration(dataset, y_true, y_score, report_dir,
                     stem="ext_calibration", root=None, download=True):
    from . import plotting as P

    cal = calibration_analysis(dataset, y_true, y_score, out_dir=None,
                               root=root, download=download)
    yt = np.asarray(y_true).squeeze()
    ys = np.asarray(y_score)
    conf = ys.max(axis=1)
    n = len(INFO[dataset]["label"])
    names = _labels(dataset)
    _, _, counts = rare_common_split(dataset, root=root, download=download)
    order = list(np.argsort(counts))  # ascending frequency

    P.set_style()
    fig, (ax1, ax2) = P.new_fig(width=P.COL_WIDTH * 1.8, ncols=2,
                                height=P.COL_WIDTH * 1.0)
    # (a) reliability diagram: overall + common + rare.
    ax1.plot([0, 1], [0, 1], color="0.6", ls="--", lw=0.8)
    for key, color, lab in [("overall", P.PALETTE[0], "overall"),
                            ("common", P.PALETTE[2], "common"),
                            ("rare", P.PALETTE[3], "rare")]:
        r = cal[key]
        m = ~np.isnan(r["acc"])
        ax1.plot(r["conf"][m], r["acc"][m], "o-", ms=3, color=color,
                 label=f"{lab} (ECE={r['ece']:.3f})")
    ax1.set_xlabel("confidence"); ax1.set_ylabel("accuracy")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_title("Reliability")
    ax1.legend(loc="upper left")

    # (b) per-class softmax-confidence boxplots, ordered by frequency.
    data = [conf[yt == c] for c in order]
    data = [d if len(d) else np.array([np.nan]) for d in data]
    ax2.boxplot(data, positions=np.arange(n), widths=0.6, showfliers=False,
                medianprops=dict(color=P.PALETTE[1]))
    ax2.set_xticks(np.arange(n))
    ax2.set_xticklabels([names[c] for c in order], rotation=45, ha="right",
                        fontsize=6)
    ax2.set_ylabel("softmax confidence")
    ax2.set_title("Confidence by class (rare → common)")
    fig.tight_layout()
    return P.savefig(fig, stem, report_dir)


# --------------------------------------------------------------------------- #
# Misclassification gallery (rare-class errors; clinically dangerous cases)
# --------------------------------------------------------------------------- #

def plot_misclassified_gallery(dataset, y_true, y_score, images, report_dir,
                               stem="ext_misclassified", n=12,
                               root=None, download=True):
    """Grid of misclassified rare-class test images with true/pred labels.

    ``images``: uint8 array ``(N, H, W, 3)`` aligned with ``y_true`` (e.g. the
    medmnist dataset's ``.imgs``). Prioritizes rare-class errors, and within
    them the highest-confidence mistakes (the most dangerously wrong ones).
    """
    from . import plotting as P

    yt = np.asarray(y_true).squeeze()
    ys = np.asarray(y_score)
    pred = ys.argmax(axis=1)
    conf = ys.max(axis=1)
    names = _labels(dataset)
    rare, _, _ = rare_common_split(dataset, root=root, download=download)

    wrong = np.where(pred != yt)[0]
    rare_wrong = [i for i in wrong if yt[i] in rare]
    pool = rare_wrong if rare_wrong else list(wrong)
    pool = sorted(pool, key=lambda i: -conf[i])[:n]  # most confident errors

    P.set_style()
    cols = 4
    rows = int(np.ceil(max(len(pool), 1) / cols))
    fig, axes = P.new_fig(width=P.COL_WIDTH * 2.0, ncols=cols, nrows=rows,
                          height=P.COL_WIDTH * 0.55)
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    if not pool:
        axes[0].text(0.5, 0.5, "no misclassifications", ha="center",
                     va="center", transform=axes[0].transAxes)
    for ax, i in zip(axes, pool):
        img = np.asarray(images[i])
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"T:{names[int(yt[i])]}\nP:{names[int(pred[i])]} "
                     f"({conf[i]:.2f})", fontsize=5.5, color=P.PALETTE[3])
    fig.suptitle(f"{dataset} — misclassified rare-class cases", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return P.savefig(fig, stem, report_dir)


# --------------------------------------------------------------------------- #
# Class distribution & frequency-vs-performance
# --------------------------------------------------------------------------- #

def plot_class_distribution(dataset, report_dir, stem="ext_class_distribution",
                            root=None, download=True):
    from . import plotting as P
    from .data import class_counts

    counts = class_counts(dataset, root=root, download=download)
    names = _labels(dataset)
    order = list(np.argsort(counts)[::-1])
    P.set_style()
    fig, ax = P.new_fig(width=P.COL_WIDTH * 1.4, height=P.COL_WIDTH * 0.9)
    ax.bar(range(len(order)), [counts[c] for c in order], color=P.PALETTE[0])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[c] for c in order], rotation=45, ha="right",
                       fontsize=6)
    ax.set_ylabel("training samples")
    ax.set_title(f"{dataset} training class distribution")
    for i, c in enumerate(order):
        ax.text(i, counts[c], f"{int(counts[c])}", ha="center", va="bottom",
                fontsize=5.5)
    fig.tight_layout()
    return P.savefig(fig, stem, report_dir)


def frequency_performance(dataset, table, report_dir, stem="ext_freq_perf",
                          root=None, download=True):
    """Scatter of per-class accuracy (recall) vs training count (log x) + fit.

    ``table`` is a per-class DataFrame (single-seed ``per_class_table`` or the
    ``*_mean`` columns of ``per_class_multiseed``). Returns Pearson/Spearman r.
    """
    from scipy.stats import pearsonr, spearmanr
    from . import plotting as P
    from .data import class_counts

    counts = class_counts(dataset, root=root, download=download).astype(float)
    recall_col = "recall_mean" if "recall_mean" in table else "recall"
    auc_col = "auc_mean" if "auc_mean" in table else "auc"
    rec = table.sort_values("cls")[recall_col].values
    auc = table.sort_values("cls")[auc_col].values
    names = _labels(dataset)

    x = np.log10(np.clip(counts, 1, None))
    pear = float(pearsonr(counts, rec)[0])
    spear = float(spearmanr(counts, rec)[0])
    pear_auc = float(pearsonr(counts, auc)[0])

    P.set_style()
    fig, ax = P.new_fig(width=P.COL_WIDTH * 1.3, height=P.COL_WIDTH * 0.95)
    ax.scatter(counts, rec, s=22, color=P.PALETTE[0], zorder=3)
    for i in range(len(counts)):
        ax.annotate(names[i], (counts[i], rec[i]), textcoords="offset points",
                    xytext=(3, 3), fontsize=5.5)
    coef = np.polyfit(x, rec, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(10 ** xs, np.polyval(coef, xs), color=P.PALETTE[3], lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("training count (log)"); ax.set_ylabel("per-class recall")
    ax.set_title(f"Frequency vs recall (Pearson r={pear:.2f}, "
                 f"Spearman ρ={spear:.2f})")
    fig.tight_layout()
    png = P.savefig(fig, stem, report_dir)
    return dict(pearson_recall=pear, spearman_recall=spear,
                pearson_auc=pear_auc, png=png)


def plot_per_class_performance(dataset, table, report_dir,
                               stem="ext_per_class_perf",
                               root=None, download=True):
    """Per-class AUC and F1 grouped bars, ordered by frequency, ±std if present."""
    from . import plotting as P
    from .data import class_counts

    counts = class_counts(dataset, root=root, download=download)
    names = _labels(dataset)
    order = list(np.argsort(counts))  # ascending (rare first)
    t = table.set_index("cls")
    auc_m = "auc_mean" if "auc_mean" in table else "auc"
    f1_m = "f1_mean" if "f1_mean" in table else "f1"
    auc_e = "auc_std" if "auc_std" in table else None
    f1_e = "f1_std" if "f1_std" in table else None

    x = np.arange(len(order))
    P.set_style()
    fig, ax = P.new_fig(width=P.COL_WIDTH * 1.7, height=P.COL_WIDTH * 1.0)
    ax.bar(x - 0.2, [t.loc[c, auc_m] for c in order], 0.38,
           yerr=([t.loc[c, auc_e] for c in order] if auc_e else None),
           capsize=2, color=P.PALETTE[0], label="AUC")
    ax.bar(x + 0.2, [t.loc[c, f1_m] for c in order], 0.38,
           yerr=([t.loc[c, f1_e] for c in order] if f1_e else None),
           capsize=2, color=P.PALETTE[1], label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{names[c]}\n(n={int(counts[c])})" for c in order],
                       rotation=45, ha="right", fontsize=5.5)
    ax.set_ylabel("score"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{dataset} per-class performance (rare → common)")
    ax.legend()
    fig.tight_layout()
    return P.savefig(fig, stem, report_dir)


# --------------------------------------------------------------------------- #
# Mitigation before/after (per-class AUC + rare-class recall)
# --------------------------------------------------------------------------- #

def plot_mitigation(dataset, tables, report_dir, stem="ext_mitigation",
                    root=None, download=True):
    """Per-class AUC grouped bars for each variant, with macro-AUC/ACC labels.

    ``tables``: dict ``variant -> per_class_table`` plus a matching
    ``aggregate`` dict ``variant -> (macro_auc, acc)`` embedded via attributes
    is avoided; instead we recompute macro from the tables.
    """
    from . import plotting as P
    from .data import class_counts

    counts = class_counts(dataset, root=root, download=download)
    names = _labels(dataset)
    order = list(np.argsort(counts))
    x = np.arange(len(order))
    variants = list(tables.keys())
    width = 0.8 / max(len(variants), 1)

    P.set_style()
    fig, ax = P.new_fig(width=P.COL_WIDTH * 1.9, height=P.COL_WIDTH * 1.0)
    for i, v in enumerate(variants):
        t = tables[v].set_index("cls")
        col = "auc_mean" if "auc_mean" in tables[v] else "auc"
        ax.bar(x + i * width, [t.loc[c, col] for c in order], width,
               color=P.PALETTE[i % len(P.PALETTE)],
               label=f"{v} (macro-AUC {np.nanmean([t.loc[c, col] for c in range(len(counts))]):.3f})")
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels([names[c] for c in order], rotation=45, ha="right",
                       fontsize=5.5)
    ax.set_ylabel("per-class AUC"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{dataset} mitigation: per-class AUC (rare → common)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    return P.savefig(fig, stem, report_dir)


# --------------------------------------------------------------------------- #
# Efficiency tradeoff (full vs lightweight)
# --------------------------------------------------------------------------- #

def plot_efficiency(profiles, metrics, report_dir, stem="ext_efficiency"):
    """Two small panels: AUC/ACC vs params, and vs latency.

    ``profiles``: dict ``name -> profile_model(...) dict``.
    ``metrics``:  dict ``name -> (auc, acc)``.
    """
    from . import plotting as P

    names = list(profiles.keys())
    P.set_style()
    fig, (ax1, ax2) = P.new_fig(width=P.COL_WIDTH * 1.8, ncols=2,
                                height=P.COL_WIDTH * 0.9)
    for i, name in enumerate(names):
        p = profiles[name]; auc, acc = metrics[name]
        c = P.PALETTE[i % len(P.PALETTE)]
        ax1.scatter(p["n_params"] / 1e6, auc, s=40, color=c, label=name)
        ax1.scatter(p["n_params"] / 1e6, acc, s=40, marker="^", color=c)
        ax2.scatter(p["latency_ms_per_image"], auc, s=40, color=c, label=name)
        ax2.scatter(p["latency_ms_per_image"], acc, s=40, marker="^", color=c)
    ax1.set_xlabel("parameters (M)"); ax1.set_ylabel("score (● AUC, ▲ ACC)")
    ax1.set_title("Accuracy vs size"); ax1.legend(fontsize=6)
    ax2.set_xlabel("latency (ms/image)"); ax2.set_ylabel("score")
    ax2.set_title("Accuracy vs latency")
    fig.tight_layout()
    return P.savefig(fig, stem, report_dir)


# --------------------------------------------------------------------------- #
# Corruption robustness (inference-only)
# --------------------------------------------------------------------------- #

def apply_corruption(imgs, kind, severity):
    """Corrupt a uint8 batch ``(N,H,W,3)``. Returns a uint8 array.

    ``kind`` in {'gaussian_noise', 'jpeg', 'brightness'}; ``severity`` 1..5.
    """
    from PIL import Image
    import io

    imgs = np.asarray(imgs).astype(np.uint8)
    if kind == "gaussian_noise":
        sigma = [0, 8, 16, 24, 32, 48][severity]
        noisy = imgs.astype(np.float32) + np.random.RandomState(0).normal(
            0, sigma, imgs.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if kind == "brightness":
        factor = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0][severity]
        return np.clip(imgs.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        quality = [100, 30, 20, 12, 8, 5][severity]
        out = np.empty_like(imgs)
        for i in range(len(imgs)):
            buf = io.BytesIO()
            Image.fromarray(imgs[i]).save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            out[i] = np.asarray(Image.open(buf).convert("RGB"))
        return out
    raise ValueError(f"unknown corruption {kind!r}")


def robustness_eval(model, dataset, images, y_true, device="cuda",
                    kinds=("gaussian_noise", "jpeg", "brightness"),
                    severities=(0, 1, 2, 3), root=None, download=True):
    """Evaluate a trained model on corrupted DermaMNIST test images.

    Returns a DataFrame with overall / common / rare macro-AUC per (kind,
    severity). Inference-only; no retraining.
    """
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as TT
    from PIL import Image

    task = INFO[dataset]["task"]
    yt = np.asarray(y_true).squeeze()
    rare, common, _ = rare_common_split(dataset, root=root, download=download)
    tfm = TT.Compose([TT.ToTensor(), TT.Normalize([.5], [.5])])
    model = model.to(device).eval()

    def _auc_for(ys, mask_classes):
        sub = np.isin(yt, mask_classes)
        return metricsmod.getAUC(yt[sub], ys[sub], task) if sub.sum() else np.nan

    rows = []
    for kind in kinds:
        for sev in severities:
            corr = apply_corruption(images, kind, sev)
            batch = torch.stack([tfm(Image.fromarray(im)) for im in corr]).to(device)
            scores = []
            with torch.no_grad():
                for j in range(0, len(batch), 256):
                    logits = model(batch[j:j + 256]).float()
                    scores.append(F.softmax(logits, dim=1).cpu().numpy())
            ys = np.concatenate(scores, axis=0)
            rows.append(dict(
                corruption=kind, severity=sev,
                auc_overall=metricsmod.getAUC(yt, ys, task),
                auc_common=_auc_for(ys, common),
                auc_rare=_auc_for(ys, rare),
            ))
    return pd.DataFrame(rows)


def plot_robustness(df, report_dir, stem="ext_robustness"):
    """AUC vs severity, one line per corruption, common vs rare panels."""
    from . import plotting as P

    P.set_style()
    fig, (ax1, ax2) = P.new_fig(width=P.COL_WIDTH * 1.8, ncols=2,
                                height=P.COL_WIDTH * 0.9)
    kinds = list(dict.fromkeys(df["corruption"]))
    for i, kind in enumerate(kinds):
        d = df[df["corruption"] == kind].sort_values("severity")
        c = P.PALETTE[i % len(P.PALETTE)]
        ax1.plot(d["severity"], d["auc_common"], "o-", ms=3, color=c, label=kind)
        ax2.plot(d["severity"], d["auc_rare"], "o-", ms=3, color=c, label=kind)
    ax1.set_title("common classes"); ax2.set_title("rare classes")
    for ax in (ax1, ax2):
        ax.set_xlabel("severity"); ax.set_ylabel("macro-AUC")
    ax1.legend(fontsize=6)
    fig.suptitle("Corruption robustness", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return P.savefig(fig, stem, report_dir)
