# Reproducing and Extending MedMNIST v2

### Bias auditing and equity analysis for biomedical image classification benchmarks

A ReScience C submission built on **MedMNIST v2** (Yang et al., *Nature
Scientific Data* 2023, [doi:10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)).

We ask two questions the original benchmark does not answer on its own: do the
ResNet baselines in Table 3 hold up, and *who does the benchmark serve* — which
lesion classes does a well-performing DermaMNIST model quietly fail, and what
happens to them under mitigation, compression, and corruption.

> This repository is a fork of [MedMNIST/MedMNIST](https://github.com/MedMNIST/MedMNIST).
> The upstream package lives in [`medmnist/`](medmnist/) and is used unmodified;
> the original project README is preserved at [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md).
> Everything under `src/`, `notebooks/`, `configs/`, `tools/`, and `results/` is ours.

---

## Two arms, kept separate

The study reports two independently-sourced sets of numbers. Pooling them would
be a category error, so nothing in the pipeline ever merges them.

| Arm | Code | Question | Table |
|---|---|---|---|
| **Replication** | our own [`src/`](src/) | does an *independent* reimplementation of the described method reach the published numbers? | `report/comparison.md` |
| **Reproduction** | the authors' [`MedMNIST/experiments`](https://github.com/MedMNIST/experiments) | does the *authors' released code* still produce its published numbers? | `report/reproduction_arm.md` |

`src.aggregate` only ever reads `run.json` files written by our own pipeline, so
the reproduction runs cannot leak into the replication table even by accident.

Current state of the reproduction arm — both inside tolerance
(|ΔAUC| ≤ 0.02, |ΔACC| ≤ 0.03), single seed each:

| Run | Config | Ours | Paper | ΔAUC | ΔACC |
|---|---|---|---|---|---|
| personA | R18, DermaMNIST @224 | 0.9212 / 0.7401 | 0.920 / 0.754 | +0.0012 | −0.0139 |
| personB | R50, DermaMNIST @224 | 0.9119 / 0.7377 | 0.912 / 0.731 | −0.0001 | +0.0067 |

## Layout

```
src/                   our replication package (see REPLICATION.md for the spec)
  reference.py           paper Table-3 numbers, dependency-free
  metrics.py             our AUC/ACC + agreement check vs medmnist.Evaluator
  models.py              CIFAR-style ResNet18/50 @28, torchvision @224, 0.5x variant
  data.py                loaders, transforms, bias-mitigation samplers/weights
  train.py               training loop, seeding, best-val-AUC selection, resume, AMP
  evaluate.py            inference + prediction dumping
  run.py                 RunConfig, single-run entry point, tiered run matrix
  aggregate.py           run.json -> report/comparison.{csv,md} with deltas
  extensions.py          per-class, mitigation, lightweight, robustness, calibration
  figures_replication.py the ReScience figures
  plotting.py            shared figure style (colorblind, PDF + 300-dpi PNG)
  from_predictions.py    recompute extensions from a saved score matrix (no GPU)
  reproduction_arm.py    the authors'-code arm, with per-run provenance

notebooks/             what ran, with what code -> notebooks/README.md
  replication/           medmnist_replication.ipynb  <- the deliverable
  reproduction_authors_code/   frozen; authors' scripts; already run
  extensions_prior/      frozen; our pre-src/ equity pass at seed 0

results/
  replication/           new run dirs land here (one per run, with run.json)
  reproduction_authors_code/   personA + personB artifacts, incl. personB's scores
  extensions_prior/      seed-0 equity results (per-class, mitigation, robustness)
  dataset_overview/      DermaMNIST class distribution + montage

configs/matrix.json    the enumerated 37-run tiered matrix
tools/                 sync_notebook_src.py — keeps the notebook's embedded src/ honest
report/                generated: comparison.md, reproduction_arm.md, figures/
docs/UPSTREAM_README.md  the original MedMNIST project README
```

## Running it

Full method spec, tolerances, and the per-run reproduction commands are in
**[REPLICATION.md](REPLICATION.md)**. The short version:

```bash
pip install -r requirements-replication.txt

# smoke-test the pipeline end to end (~2 min, CPU-tolerable)
python -m src.run --dataset dermamnist --model resnet18 --size 28 --seed 0 \
    --epochs 3 --results-dir results/replication_smoke

# verify our metrics against the medmnist.Evaluator oracle
python -m src.metrics_test

# aggregate whatever has finished
python -m src.aggregate results/replication      # -> report/comparison.{csv,md}
python -m src.reproduction_arm                   # -> report/reproduction_arm.{csv,md}
```

On Kaggle, use [`notebooks/replication/medmnist_replication.ipynb`](notebooks/replication/medmnist_replication.ipynb).
It ships with `SMOKE = True` — leave it on for the first run, then set it to
`False`. Because a 12-hour session cannot train the whole matrix, run
`MODE="baselines"` over several sessions (chaining each session's *output* into
the next as `PREV_RESULTS`), then `MODE="extensions"` once, then `MODE="report"`
once. Finished runs and mid-run checkpoints both resume, so a session hitting
the wall costs at most a few epochs.

## Extensions

All on DermaMNIST (7-class skin lesion, 10,015 images from HAM10000), separable
from the baseline — the replication reproduces without running any of them.

- **Per-class audit** — ROC-AUC *and* PR-AUC, precision/recall/F1 with ±std
  across seeds; confusion matrix, per-class ROC and PR curves.
- **Frequency vs performance** — Pearson/Spearman of training count against
  per-class recall and AUC. The core equity claim.
- **Bias mitigation** — `WeightedRandomSampler` and inverse-frequency weighted
  loss, 3 seeds each, with the equity/accuracy tradeoff stated rather than
  buried.
- **Lightweight variant** — ResNet-18 at 0.5× width: params, MB, ms/image, and
  whether compression costs rare classes disproportionately.
- **Corruption robustness** — Gaussian noise, JPEG, brightness at several
  severities, split common vs rare. Inference-only.
- **Confidence & calibration** — reliability diagram, ECE overall/common/rare,
  per-class softmax confidence, rare-class misclassification gallery.

## Licensing and intended use

Code follows the upstream Apache 2.0 licence. DermaMNIST is **CC BY-NC 4.0**
(non-commercial); PathMNIST is CC BY 4.0. As upstream states, MedMNIST is
**not intended for clinical use** — and the equity findings here are an argument
about benchmark composition, not a claim about any deployable system.
