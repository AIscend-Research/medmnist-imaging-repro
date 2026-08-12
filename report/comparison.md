# MedMNIST v2 replication — comparison vs paper (Table 3)

AUC / ACC reported as our mean ± std across seeds; delta = ours − paper.
Tolerance: |ΔAUC| ≤ 0.02, |ΔACC| ≤ 0.03. Flag marks configs outside it.

| Dataset | Model | Size | Seeds | Our AUC | Paper AUC | ΔAUC | Our ACC | Paper ACC | ΔACC | Flag |
|---|---|---|---|---|---|---|---|---|---|---|
| bloodmnist | resnet18 | 28 | 3 | 0.997 ± 0.000 | 0.998 | -0.001 | 0.954 ± 0.001 | 0.958 | -0.004 |  |
| breastmnist | resnet18 | 28 | 3 | 0.874 ± 0.012 | 0.901 | -0.027 | 0.806 ± 0.008 | 0.863 | -0.057 | OUT_OF_TOL |
| chestmnist | resnet18 | 28 | 1 | 0.767 ± 0.000 | 0.768 | -0.001 | 0.948 ± 0.000 | 0.947 | 0.001 |  |
| dermamnist | resnet18 | 28 | 3 | 0.914 ± 0.002 | 0.917 | -0.003 | 0.729 ± 0.006 | 0.735 | -0.006 |  |
| dermamnist | resnet50 | 28 | 3 | 0.906 ± 0.002 | 0.913 | -0.007 | 0.724 ± 0.002 | 0.735 | -0.011 |  |
| octmnist | resnet18 | 28 | 1 | 0.936 ± 0.000 | 0.943 | -0.007 | 0.695 ± 0.000 | 0.743 | -0.048 | OUT_OF_TOL |
| organamnist | resnet18 | 28 | 3 | 0.997 ± 0.000 | 0.997 | -0.000 | 0.932 ± 0.005 | 0.935 | -0.003 |  |
| organcmnist | resnet18 | 28 | 3 | 0.993 ± 0.000 | 0.992 | 0.001 | 0.913 ± 0.005 | 0.900 | 0.013 |  |
| organsmnist | resnet18 | 28 | 3 | 0.970 ± 0.002 | 0.972 | -0.002 | 0.754 ± 0.010 | 0.782 | -0.028 |  |
| pathmnist | resnet18 | 28 | 1 | 0.987 ± 0.000 | 0.983 | 0.004 | 0.915 ± 0.000 | 0.907 | 0.008 |  |
| pneumoniamnist | resnet18 | 28 | 3 | 0.942 ± 0.014 | 0.944 | -0.002 | 0.812 ± 0.054 | 0.854 | -0.042 | OUT_OF_TOL |
| retinamnist | resnet18 | 28 | 3 | 0.738 ± 0.001 | 0.717 | 0.021 | 0.511 ± 0.025 | 0.524 | -0.013 | OUT_OF_TOL |
| tissuemnist | resnet18 | 28 | 1 | 0.928 ± 0.000 | 0.930 | -0.002 | 0.673 ± 0.000 | 0.676 | -0.003 |  |
