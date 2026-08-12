# DermaMNIST equity & extension study

ResNet-18 @ 28, 3 baseline seed(s). Kept separate from the replication's comparison.md.

## Frequency vs performance
- Pearson r (count vs recall) = 0.805; Spearman rho = 0.536; Pearson (count vs AUC) = -0.282.

## Bias mitigation (equity vs accuracy tradeoff)

| variant          |      auc |      acc |   macro_f1 |   min_class_f1 |   min_class_recall | worst_class    |
|:-----------------|---------:|---------:|-----------:|---------------:|-------------------:|:---------------|
| baseline         | 0.913728 | 0.730175 |   0.464696 |       0.16     |          0.0869565 | dermatofibroma |
| weighted_sampler | 0.896921 | 0.734663 |   0.485058 |       0.214286 |          0.130435  | dermatofibroma |
| weighted_loss    | 0.893548 | 0.633416 |   0.449221 |       0.138889 |          0.434783  | dermatofibroma |

## Lightweight variant
- params 11.17M -> 2.80M; latency 2.019 -> 2.039 ms/img; freq~F1-drop corr = -0.401.
