# Notebooks — what ran, with what code, producing what

Three directories, three different provenances. Keeping them apart is not
housekeeping: the numbers they produce answer *different questions* and must not
be pooled into a single table.

| Directory | Code used | Status |
|---|---|---|
| [`replication/`](replication/) | our own `src/` package | **live — this is the deliverable** |
| [`reproduction_authors_code/`](reproduction_authors_code/) | the authors' `MedMNIST/experiments` scripts | frozen; already run, results committed |
| [`extensions_prior/`](extensions_prior/) | our own code, written before `src/` | frozen; superseded by `MODE="extensions"` |

---

## `replication/` — the deliverable

**[`medmnist_replication.ipynb`](replication/medmnist_replication.ipynb)** — one
self-contained notebook, re-run with a different `MODE` each session. It embeds
the whole `src/` package base64-encoded, so it needs no clone on Kaggle.

`MODE="baselines"` → `MODE="extensions"` → `MODE="report"`.

Nothing in `src/` imports, copies, vendors, or adapts anything from
`MedMNIST/experiments`. Every model, transform, training loop, and metric is
ours. The `medmnist` PyPI package supplies only (a) the standardized data and
(b) `Evaluator`, used purely as an oracle to check our metric code. This is what
makes the arm a **replication** rather than a reproduction.

> The embedded copy of `src/` is a duplicate, and duplicates drift. After
> editing anything in `src/`, run `python tools/sync_notebook_src.py`.
> `--check` verifies without writing and exits non-zero on drift.

## `reproduction_authors_code/` — frozen, already run

These clone `github.com/MedMNIST/experiments` and run
`train_and_eval_pytorch.py` unmodified. **Do not re-run them** — their results
are committed and they cost ~9 GPU-hours to regenerate.

| Notebook | Config | Result | Paper (224) | Artifacts |
|---|---|---|---|---|
| `personA_resnet18_dermamnist.ipynb` | R18, DermaMNIST, `--resize` | AUC 0.9212 / ACC 0.7401 | 0.920 / 0.754 | [`results/reproduction_authors_code/personA_resnet18_derma224/`](../results/reproduction_authors_code/personA_resnet18_derma224/) |
| `personB_resnet50_dermamnist.ipynb` | R50, DermaMNIST, `--resize` | AUC 0.9119 / ACC 0.7377 | 0.912 / 0.731 | [`results/reproduction_authors_code/personB_resnet50_derma224/`](../results/reproduction_authors_code/personB_resnet50_derma224/) |

Both land inside the study's tolerance (|ΔAUC| ≤ 0.02, |ΔACC| ≤ 0.03).
Regenerate the table with `python -m src.reproduction_arm`, which reads the
manifest in [`src/reproduction_arm.py`](../src/reproduction_arm.py) and writes
`report/reproduction_arm.{csv,md}`.

Three things to know about these runs:

- **`--resize` means 224.** The authors' script upsamples the 28-pixel `.npz` to
  224, so these belong to the paper's 224 column. The archived
  `reproduction_summary_personA.csv` compares personA against AUC 0.917 — the
  paper's *28-pixel* row. That is the wrong reference; `src.reproduction_arm`
  uses 0.920 and records the correction. The archived CSV is left untouched as a
  record of what was originally computed.
- **Single seed each.** They predate the seeded matrix, so no ±std is available
  and none is claimed.
- **personB kept its full test softmax matrix** (2005 × 7), at
  `.../output/dermamnist/260720_082717/dermamnist_test_[AUC]0.912_[ACC]0.738@run1.csv`.
  Every prediction-only extension can therefore be recomputed for it with no GPU:

  ```bash
  python -m src.from_predictions \
      --scores "results/reproduction_authors_code/personB_resnet50_derma224/output/dermamnist/260720_082717/dermamnist_test_[AUC]0.912_[ACC]0.738@run1.csv" \
      --dataset dermamnist --tag personB_r50_224 \
      --out results/reproduction_authors_code/personB_resnet50_derma224/recomputed
  ```

  It re-derives AUC/ACC and asserts they match the values in the filename, which
  catches any score/label misalignment before the analysis is trusted.

## `extensions_prior/` — frozen, superseded

**`phase3_extensions_seed0.ipynb`** is *our own* code (torchvision ResNet-18 plus
a hand-written training loop) — it does **not** use `MedMNIST/experiments`. It
trained four DermaMNIST variants at seed 0 and produced the first pass of the
equity study, in [`results/extensions_prior/`](../results/extensions_prior/):

| Variant | Macro AUC | Accuracy | Macro F1 | Melanoma F1 | Dermatofibroma F1 |
|---|---|---|---|---|---|
| Baseline | 0.9148 | 0.7392 | 0.4191 | 0.1753 | 0.0800 |
| Weighted sampler | 0.9043 | 0.7471 | 0.4937 | 0.4499 | 0.1935 |
| Weighted loss | 0.9068 | 0.6349 | 0.4637 | 0.4179 | 0.1774 |
| Lightweight 0.5× | 0.9135 | 0.7347 | 0.4913 | 0.3203 | 0.2632 |

That is the equity/accuracy tradeoff in one table, already measured: a weighted
sampler more than doubles melanoma F1 at a cost of ~1 AUC point.

It is superseded by `MODE="extensions"` in the replication notebook, which does
the same study across three seeds, through the audited `src/` code path, with
PR-AUC, calibration, and robustness added. Keep these results as the
single-seed precedent; report the `src/` numbers as the study's.

## Removed

`personC_resnet18_pathmnist.ipynb` and `all_remaining_B_C_pathmnist.ipynb` were
deleted. Neither ever produced a committed result, and leaving unrun notebooks
beside run ones makes it impossible to tell at a glance what the evidence
actually is. PathMNIST is covered by Tier 3 of the replication matrix.
