"""The *reproduction* arm: runs executed with the original authors' own code.

This study has two arms, and they must never be pooled into one table:

* **Replication** (``src/``, :mod:`src.aggregate`) — our independent
  reimplementation. Nothing from ``MedMNIST/experiments`` is used.
* **Reproduction** (this module) — the authors' ``train_and_eval_pytorch.py``
  from ``MedMNIST/experiments``, run unmodified on Kaggle. It answers a
  different question: does the authors' released code, on the authors' released
  data, still produce the published numbers?

Both are legitimate and ReScience C cares about the distinction, so the
provenance of every number below is recorded explicitly rather than inferred.

The runs here are **single-seed** (they predate the seeded matrix), so no ±std
is reported — stated rather than papered over.

    python -m src.reproduction_arm            # -> report/reproduction_arm.{csv,md}

Stdlib-only, so it runs without torch.
"""

from __future__ import annotations

import csv
import os

from .reference import REFERENCE

# Same tolerances as the replication arm, so the two tables are read the same way.
AUC_TOL = 0.02
ACC_TOL = 0.03

RESULTS_SUBDIR = os.path.join("results", "reproduction_authors_code")

# --------------------------------------------------------------------------- #
# Manifest of everything that was actually run, with full provenance.
#
# ``size`` is 224 for the runs that passed ``--resize``: the authors' script
# upsamples the 28-pixel npz to 224, which is the setting paper Table 3 reports
# in its 224 column. Getting this wrong silently compares against the wrong
# reference row (see NOTE on personA below).
# --------------------------------------------------------------------------- #
RUNS = [
    dict(
        label="personA",
        dataset="dermamnist", model="resnet18", size=224, seeds=1,
        auc=0.9212, acc=0.7401,
        notebook="notebooks/reproduction_authors_code/personA_resnet18_dermamnist.ipynb",
        code="MedMNIST/experiments @ train_and_eval_pytorch.py (unmodified)",
        command=("python train_and_eval_pytorch.py --data_flag dermamnist "
                 "--num_epochs 100 --batch_size 128 --resize --model_flag resnet18 "
                 "--run run1"),
        results=os.path.join(RESULTS_SUBDIR, "personA_resnet18_derma224"),
        predictions=None,
        # NOTE: the archived reproduction_summary_personA.csv compares this run
        # against AUC 0.917, which is the paper's *28-pixel* ResNet-18 row. The
        # run used --resize, so 224 (AUC 0.920 / ACC 0.754) is the correct
        # reference. Corrected here; the archived CSV is left untouched as a
        # record of what was originally computed.
        note="reference corrected from the 28px row to the 224px row (run used --resize)",
    ),
    dict(
        label="personB",
        dataset="dermamnist", model="resnet50", size=224, seeds=1,
        auc=0.9119, acc=0.7377,
        notebook="notebooks/reproduction_authors_code/personB_resnet50_dermamnist.ipynb",
        code="MedMNIST/experiments @ train_and_eval_pytorch.py (unmodified)",
        command=("python train_and_eval_pytorch.py --data_flag dermamnist "
                 "--num_epochs 100 --batch_size 128 --resize --model_flag resnet50 "
                 "--run run1"),
        results=os.path.join(RESULTS_SUBDIR, "personB_resnet50_derma224"),
        predictions=os.path.join(
            RESULTS_SUBDIR, "personB_resnet50_derma224", "output", "dermamnist",
            "260720_082717",
            "dermamnist_test_[AUC]0.912_[ACC]0.738@run1.csv"),
        note="full test softmax matrix retained -> prediction-only extensions "
             "recomputable via src.from_predictions (no GPU)",
    ),
]


def rows():
    """One dict per reproduction run, with reference deltas and tolerance flag."""
    out = []
    for r in RUNS:
        key = (r["dataset"], r["model"], r["size"])
        ref = REFERENCE.get(key)
        ref_auc, ref_acc = ref if ref else (None, None)
        d_auc = r["auc"] - ref_auc if ref else None
        d_acc = r["acc"] - ref_acc if ref else None
        flag = ""
        if ref and (abs(d_auc) > AUC_TOL or abs(d_acc) > ACC_TOL):
            flag = "OUT_OF_TOL"
        out.append(dict(
            label=r["label"], dataset=r["dataset"], model=r["model"],
            size=r["size"], n_seeds=r["seeds"],
            auc=r["auc"], acc=r["acc"],
            ref_auc=ref_auc, ref_acc=ref_acc,
            delta_auc=d_auc, delta_acc=d_acc, flag=flag,
            code=r["code"], notebook=r["notebook"], results=r["results"],
            predictions=r["predictions"] or "", note=r.get("note", ""),
        ))
    return out


def missing_artifacts(repo_root="."):
    """Paths in the manifest that are not on disk — keeps the table honest."""
    gone = []
    for r in RUNS:
        for key in ("notebook", "results", "predictions"):
            path = r.get(key)
            if path and not os.path.exists(os.path.join(repo_root, path)):
                gone.append((r["label"], key, path))
    return gone


def _fmt(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"


def _write_csv(data, path):
    fields = ["label", "dataset", "model", "size", "n_seeds", "auc", "acc",
              "ref_auc", "ref_acc", "delta_auc", "delta_acc", "flag", "code",
              "notebook", "results", "predictions", "note"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


def _write_md(data, path, gone):
    lines = [
        "# Reproduction arm — the authors' own code",
        "",
        "Runs executed with `train_and_eval_pytorch.py` from "
        "[MedMNIST/experiments](https://github.com/MedMNIST/experiments), "
        "unmodified. This is **not** the replication arm; for our independent "
        "reimplementation see `comparison.md`, generated by `src.aggregate`.",
        "",
        "`--resize` upsamples the 28-pixel data to 224, so these compare "
        "against the paper's **224** column of Table 3.",
        "",
        f"Tolerance: |ΔAUC| ≤ {AUC_TOL}, |ΔACC| ≤ {ACC_TOL}. "
        "Delta = ours − paper. **Single seed per config — no variance is "
        "claimed.**",
        "",
        "| Run | Dataset | Model | Size | Seeds | Our AUC | Paper AUC | ΔAUC | "
        "Our ACC | Paper ACC | ΔACC | Flag |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in data:
        lines.append(
            f"| {r['label']} | {r['dataset']} | {r['model']} | {r['size']} | "
            f"{r['n_seeds']} | {_fmt(r['auc'], 4)} | {_fmt(r['ref_auc'])} | "
            f"{_fmt(r['delta_auc'], 4)} | {_fmt(r['acc'], 4)} | "
            f"{_fmt(r['ref_acc'])} | {_fmt(r['delta_acc'], 4)} | {r['flag']} |")

    lines += ["", "## Provenance", ""]
    for r in data:
        lines.append(f"**{r['label']}** — `{r['code']}`")
        lines.append(f"- notebook: `{r['notebook']}`")
        lines.append(f"- artifacts: `{r['results']}`")
        if r["predictions"]:
            lines.append(f"- test score matrix: `{r['predictions']}`")
        if r["note"]:
            lines.append(f"- note: {r['note']}")
        lines.append("")

    if gone:
        lines += ["## Missing artifacts", ""]
        for label, key, path in gone:
            lines.append(f"- **{label}** {key}: `{path}` not found")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def build(report_dir="report", repo_root="."):
    os.makedirs(report_dir, exist_ok=True)
    data = rows()
    gone = missing_artifacts(repo_root)
    _write_csv(data, os.path.join(report_dir, "reproduction_arm.csv"))
    _write_md(data, os.path.join(report_dir, "reproduction_arm.md"), gone)
    return data, gone


def main(argv=None):
    import sys
    report_dir = argv[0] if argv else "report"
    data, gone = build(report_dir)
    for r in data:
        print(f"{r['label']:>8}  {r['dataset']} {r['model']} @{r['size']}  "
              f"AUC {r['auc']:.4f} (Δ{r['delta_auc']:+.4f})  "
              f"ACC {r['acc']:.4f} (Δ{r['delta_acc']:+.4f})  {r['flag']}")
    for label, key, path in gone:
        print(f"WARNING  {label}: missing {key} -> {path}", file=sys.stderr)
    print(f"\nwrote {report_dir}/reproduction_arm.{{csv,md}}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
