# Text/framing fixes for reviewer comments (no retraining required)

These are wording and structure changes to make in the manuscript directly.
Nothing here depends on GPU access — they're fixes to how existing results are
presented, not new experiments.

## 1. Seed-protocol framing (Reviewer 2, "characterized on only one side")

Current framing computes `delta = ours - paper` and checks it against a fixed
tolerance, treating the paper's numbers as exact. State explicitly what is and
isn't known about seed variance on both sides:

- State whether the original MedMNIST v2 paper reports single-seed or
  multi-seed numbers for each dataset (check their repo/paper methodology
  section — if undocumented, say so explicitly rather than assuming).
- For the 9 configurations with only one of our own seeds, say plainly that a
  pass/fail against tolerance is a single draw, not a distributional claim.
- For BreastMNIST and PneumoniaMNIST (3 seeds, Figure 3), reframe the
  agreement claim as "our spread overlaps the paper's reported value" rather
  than "our point estimate crossed a line."

Suggested sentence for the results section:
> "Because neither our nor the original paper's protocol reports seed variance
> for every dataset, a pass/fail against the pre-registered tolerance should be
> read as evidence at the resolution of a single draw, not as a statement about
> the full sampling distribution. Where we have multiple seeds (BreastMNIST,
> PneumoniaMNIST, DermaMNIST, RetinaMNIST), we report the spread directly
> (Figure 3) rather than collapsing it to a point delta."

## 2. OCTMNIST single-seed caveat (Reviewer 1)

OCTMNIST is both single-seed and a tolerance miss (ΔACC = -0.048). Add a
sentence flagging that this specific miss cannot be distinguished from
single-seed noise without a second run:

> "OCTMNIST, Tissue, Path, and Chest were each run once; we describe them as
> stable at this data scale by extrapolation from the smaller datasets' seed
> variance (Figure 3), not from direct measurement. This matters most for
> OCTMNIST, our only tolerance miss among these four (ΔACC = -0.048) — we
> cannot currently rule out that this is a single-seed draw rather than a
> systematic gap."

(A second OCTMNIST seed would resolve this properly — see the compute-blocked
items list. Until then, this caveat should stay in the text.)

## 3. Lead with the per-class table, not the correlation coefficients (both reviewers)

Section 5.1's headline currently leans on `r = 0.81` (recall) vs `r = -0.28`
(AUC) computed over n=7 classes. Both reviewers flag that these are fragile
point estimates from a tiny sample dominated by one leverage point
(Melanocytic nevi, 67% of training data).

Restructure so the per-class metrics table (dermatofibroma: AUC 0.917, recall
~2/23) comes first and carries the argument — it doesn't depend on any
correlation coefficient. Then present the correlation as a secondary,
heavily-caveated summary statistic, using the real numbers from
`results/phase3/freq_correlation_robustness.csv`:

> "Per-class recall correlates with training frequency (Pearson r = 0.68,
> 95% CI [-0.14, 0.95]; Spearman rho = 0.54), while per-class AUC does not
> (r = -0.25, 95% CI [-0.84, 0.62]; Spearman rho = -0.43). With only seven
> classes, these intervals are wide, and the correlation is sensitive to a
> single leverage point: excluding Melanocytic nevi (67% of training data),
> the recall correlation falls to r = 0.21. We report the correlation for
> completeness, but the qualitative dissociation — visible directly in
> Table 2 without any correlation coefficient — is the more robust claim."

Note: recompute the exact r-values quoted above once the paper's own
production run (not this repo's placeholder baseline CSV) is finalized —
plug in whatever `freq_correlation_robustness.csv` shows for the model you
actually report as your headline baseline.

## 4. Report Spearman alongside Pearson everywhere r is quoted

Both reviews independently flagged that the paper reports only Pearson r
while Figure 5's own title uses Spearman rho — and that the two disagree
enough to be informative on their own. Anywhere a Pearson r is stated in
prose, add the Spearman rho next to it (values now available in
`results/phase3/freq_correlation_robustness.csv` for all four models, not
just the baseline).

## 5. Mitigation section: name the threshold confound up front

Regardless of whether the threshold-tuned baseline control has been rerun
yet (see compute-blocked items), the text should state the confound
explicitly rather than presenting Table 3's tradeoff as clean:

> "Because recall, F1, and min-class F1 are all argmax quantities, and
> weighted loss/sampling shift the decision boundary by construction, part of
> the reported gain may be attributable to a threshold shift rather than a
> change in what the model learned. We control for this with a
> threshold-tuned baseline (tuned on validation, per-class, F1-maximizing)
> [cite Table 3 row / Figure once available]."

If the threshold-tuned row isn't in yet when this goes out, say so plainly
rather than omitting the caveat — reviewers already flagged its absence once.
