"""Shared plotting style for every figure in the paper.

One place for the colorblind-safe palette, clean matplotlib defaults (no top/
right spines, readable fonts), and single-column sizing (~3.3 in wide). Every
figure script imports from here and saves through :func:`savefig`, which writes
both a vector PDF and a 300-dpi PNG into ``report/figures/``. Regenerating
figures never re-runs training — the figure scripts read only from ``results/``.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Okabe-Ito colorblind-safe qualitative palette (8 hues, deuteranopia-safe).
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Diverging colormap for signed (ours - paper) deltas, centered at zero.
DIVERGING_CMAP = "RdBu_r"

# Single-column figure width (inches) for a two-column paper.
COL_WIDTH = 3.3
GOLDEN = 0.62  # height:width ratio used for default single-panel figures


def set_style():
    """Install the shared rcParams. Idempotent; call once before plotting."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "figure.autolayout": False,
    })
    # Cycle the colorblind-safe palette by default.
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALETTE)


def new_fig(width=COL_WIDTH, height=None, ncols=1, nrows=1, **kw):
    """Return ``(fig, ax)`` sized for the paper. ``height`` defaults to golden."""
    if height is None:
        height = width * GOLDEN
    return plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), **kw)


def despine(ax):
    """Remove the top/right spines on a single Axes (rcParams handles most)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, name, report_dir="report"):
    """Save ``fig`` as both PDF and 300-dpi PNG under ``report_dir/figures/``.

    Returns the PNG path (handy for notebook display). ``name`` is a stem with
    no extension, e.g. ``"fig1_reproduction"``.
    """
    fig_dir = os.path.join(report_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    pdf = os.path.join(fig_dir, f"{name}.pdf")
    png = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return png
