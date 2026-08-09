# Independent replication of the MedMNIST v2 ResNet baselines

This directory (`src/`, `configs/`, `report/`, `results/`, plus the Kaggle
notebook in `notebooks/replication/`) is an **independent reimplementation** of the
ResNet baselines from MedMNIST v2 (Yang et al., *Scientific Data* 2023,
<https://doi.org/10.1038/s41597-022-01721-8>), written for a ReScience C
*replication* (not a *reproduction* of the authors' own scripts).

**Hard rule:** nothing here imports, copies, vendors, or adapts any file from
`github.com/MedMNIST/experiments`. Every model, transform, training loop, and
metric is our own. The only thing taken from the `medmnist` PyPI package is
(a) the standardized dataset via its loaders and (b) `medmnist.Evaluator`, used
*only* as an oracle to check our own metric code agrees with theirs.

## What matches the paper (the spec that makes the numbers line up)

| Aspect | Setting |
|---|---|
| Backbone @ size 28 | Custom CIFAR-style ResNet: 3×3 stride-1 stem, **no** max-pool, stages stride `[1,2,2,2]`, widths `[64,128,256,512]`. R18=BasicBlock `[2,2,2,2]`, R50=Bottleneck `[3,4,6,3]` |
| Backbone @ size 224 | `torchvision.models.resnet18/50(weights=None)` — standard 7×7 stride-2 stem + maxpool |
| Data | Always the **28-pixel** `.npz`, `as_rgb=True`. 224 runs **upsample the 28px data** in the transform (nearest-neighbour) — native 224 images are *not* used |
| Transform 28 | `ToTensor`, `Normalize([.5],[.5])`, no augmentation |
| Transform 224 | `Resize((224,224), NEAREST)`, `ToTensor`, `Normalize([.5],[.5])` |
| Optimizer | Adam, lr `1e-3` |
| Scheduler | `MultiStepLR`, γ=0.1, milestones `[0.5E, 0.75E]` |
| Batch / epochs | 128 / 100 |
| Loss | `CrossEntropyLoss` (`BCEWithLogitsLoss` for multi-label tasks) |
| Model selection | best validation macro-AUC checkpoint; report its **test** AUC/ACC |
| Metrics | ACC = argmax accuracy; AUC = macro one-vs-rest ROC-AUC on softmax probs |
| Seeds | 3 seeds {0,1,2} per config; report mean ± std |

## Package layout

```
src/
  reference.py   # paper Table-3 numbers (dependency-free)
  metrics.py     # our AUC/ACC + agreement check vs medmnist.Evaluator
  models.py      # CIFAR-style ResNet18/50 (size 28) + torchvision wrapper (224) + 0.5x lightweight
  data.py        # medmnist loaders + the two transform pipelines + bias-mitigation samplers/weights
  train.py       # training loop, seeding, best-val-AUC selection, checkpoint/resume, AMP
  evaluate.py    # inference, prediction dumping (medmnist filename convention), checkpoint eval
  run.py         # RunConfig + run_config() + CLI + tiered run_matrix()
  aggregate.py   # read all run.json -> report/comparison.{csv,md} with deltas vs paper
  plotting.py    # shared figure style: colorblind palette, clean defaults, PDF+PNG saver
  figures_replication.py  # the ReScience figures, generated from results/ alone
  extensions.py  # per-class analysis, bias mitigation comparison, lightweight profiling
  from_predictions.py     # recompute prediction-only extensions from a score CSV (no GPU)
  reproduction_arm.py     # the separate authors'-code arm + its provenance manifest
configs/matrix.json   # enumerated tiered run matrix (~37 runs)
tools/sync_notebook_src.py  # keeps the notebook's embedded src/ copy in sync
results/replication/<run_name>/  # per run: best_model.pth, last.pth, predictions csv, run.json
results/reproduction_authors_code/  # frozen artifacts from the authors'-code arm
results/extensions_prior/           # frozen seed-0 equity pass (pre-src/)
report/                # comparison.csv + comparison.md + figures/ (PDF + 300-dpi PNG)
```

**Two arms.** Everything above is the *replication* arm. Runs executed with the
authors' own `train_and_eval_pytorch.py` are a separate *reproduction* arm,
tabulated by `python -m src.reproduction_arm` into `report/reproduction_arm.md`.
`src.aggregate` reads only `run.json` files written by our pipeline, so the two
can never be pooled. See `notebooks/README.md` for the provenance of each.

## Reproduce a single run

```bash
pip install -r requirements-replication.txt   # or: conda env create -f environment.yml

# DermaMNIST, ResNet-18, size 28, seed 0 (should land near AUC 0.917 / ACC 0.735)
python -m src.run --dataset dermamnist --model resnet18 --size 28 --seed 0 \
    --epochs 100 --results-dir results/replication

# ResNet-50 at 224 with mixed precision + checkpoint every 5 epochs (Kaggle-friendly)
python -m src.run --dataset dermamnist --model resnet50 --size 224 --seed 0 \
    --epochs 100 --amp --ckpt-every 5 --results-dir results/replication

# resume automatically continues from results/replication/<run_name>/last.pth
python -m src.run --dataset pathmnist --model resnet50 --size 224 --seed 0 --amp
```

Each run writes `results/replication/<run_name>/run.json` with the full config, final
train/val/test AUC+ACC, best epoch, wall-clock time, seed, whether cudnn was
deterministic / AMP was used, and pinned versions of
python/torch/torchvision/medmnist/numpy/scikit-learn.

## Verify the metric implementation

```bash
python -m src.metrics_test   # (or run the agreement cell in the notebook)
```

`metrics.check_agreement()` asserts our AUC/ACC match both
`medmnist.evaluator.getAUC/getACC` and a real `medmnist.Evaluator` object to
within `1e-3`, and is called automatically at the end of every run.

## Aggregate results vs the paper

```bash
python -m src.aggregate results/replication   # writes report/comparison.{csv,md}
python -m src.reproduction_arm               # writes report/reproduction_arm.{csv,md}
```

Tolerances (definition of done): |ΔAUC| ≤ 0.02, |ΔACC| ≤ 0.03. Any config
outside is flagged (`OUT_OF_TOL`), not silently accepted.

## Run matrix (tiers)

A compute-minimal slice of Table 3: breadth at 28×28 (all twelve datasets at
R18) plus the full four-config matrix on DermaMNIST. ~37 runs, ≈12–15 GPU-hours
on a P100.

* **Tier 1** — DermaMNIST × {R18,R50} × {28,224} × 3 seeds = **12 runs** (paper focus + extension target).
* **Tier 2** — small/medium datasets {Retina, Breast, Pneumonia, Blood, OrganA, OrganC, OrganS} × R18 × 28 × 3 seeds = **21 runs**.
* **Tier 3** — large datasets {Tissue, OCT, Path, Chest} × R18 × 28 × **1 seed** (seed 0) = **4 runs**. Single-seed for compute reasons (stable at that data scale); disclosed in the writeup.

## Extensions (separable; the baseline reproduces without running any of these)

* **Per-class analysis (DermaMNIST):** per-class ROC-AUC **and PR-AUC (average precision)**, precision/recall/F1 with ±std across the three seeds; confusion matrix, per-class ROC and PR curves.
* **Frequency vs performance:** Pearson/Spearman of training count vs per-class recall/AUC (the core equity claim).
* **Bias mitigation:** `WeightedRandomSampler` and inverse-frequency class-weighted loss, 3 seeds each; before/after per-class + aggregate metrics with the equity/accuracy tradeoff stated.
* **Lightweight variant:** ResNet-18 at 0.5× width `[32,64,128,256]`; params, MB, latency (ms/image), efficiency tradeoff, and rare-vs-common class degradation.
* **Corruption robustness (inference-only):** Gaussian noise, JPEG, brightness at several severities; AUC vs severity split common vs rare.
* **Confidence & calibration (inference-only):** reliability diagram, expected calibration error (overall / common / rare), per-class softmax confidence, and a rare-class misclassification gallery.

Extension CSVs land in `results/extension/` and figures in
`results/extension/figures/` (PDF + 300-dpi PNG); the report cell copies them
into `report/figures/` and writes `report/extension.md`.

## Kaggle

**One self-contained notebook** produces every result and figure for the paper:
`notebooks/replication/medmnist_replication.ipynb`. It writes the `src/` package from
embedded cells (base64), so it needs no GitHub clone, and is driven by a single
`CONFIG` cell:

* `SMOKE` (shipped as `False`) — set it to `True` for 3 epochs, seed 0, DermaMNIST @28 only.
  ~2 minutes, and it exercises the whole path (download → model → training loop →
  metric oracle → `run.json`) before any GPU quota goes into 100-epoch runs. It
  writes to a separate `*_smoke` results directory, so a smoke run can never be
  mistaken for a real one. Set it to `False` for everything below.
* `MODE = "baselines"` — train the tiered run matrix. Resumable across Kaggle's
  ~9-hour cap: attach the previous session's *output* as an input dataset and set
  `PREV_RESULTS` so finished runs and mid-run `last.pth` checkpoints continue.
  Pick `TIERS`, `SEEDS`, `EPOCHS`, `USE_AMP`, and a `MAX_MINUTES` budget guard.
* `MODE = "extensions"` — per-class analysis, bias mitigation (weighted
  sampler + weighted loss), and the lightweight 0.5× variant on DermaMNIST.
* `MODE = "report"` — aggregate all `run.json` into `report/comparison.{csv,md}`
  and render every figure into `report/figures/` as PDF + 300-dpi PNG:
  `fig1_reproduction` (ours-vs-paper AUC/ACC scatter across the twelve R18@28
  datasets + DermaMNIST four-config bars), `fig2_training_curves` (val AUC &
  train loss with LR-decay markers), `fig3_seed_variance` (test-AUC spread across
  seeds), `fig4_delta_heatmap` (signed ours−paper deltas), plus the extension
  figures (per-class ROC/confusion, bias F1 comparison, lightweight profile).

Because a single 9-hour session cannot train all ~37 runs, the intended workflow
is: run `MODE="baselines"` over several sessions (chaining outputs → inputs),
then `MODE="extensions"` once, then `MODE="report"` once. It is still **one
notebook file**, re-run with a different `MODE`/`TIERS`.
