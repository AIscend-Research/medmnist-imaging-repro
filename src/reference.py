"""Paper reference numbers (MedMNIST v2, Yang et al. 2023, Table 3).

Kept dependency-free so aggregation can import it without torch.
Format: (dataset, model, size) -> (AUC, ACC).
"""

REFERENCE = {
    # DermaMNIST — the primary, in-depth four-config matrix (R18/R50 x 28/224).
    ("dermamnist", "resnet18", 28): (0.917, 0.735),
    ("dermamnist", "resnet18", 224): (0.920, 0.754),
    ("dermamnist", "resnet50", 28): (0.913, 0.735),
    ("dermamnist", "resnet50", 224): (0.912, 0.731),
    # The other eleven MedMNIST2D datasets — ResNet-18 @ 28 only (our sweep).
    ("pathmnist", "resnet18", 28): (0.983, 0.907),
    ("chestmnist", "resnet18", 28): (0.768, 0.947),
    ("octmnist", "resnet18", 28): (0.943, 0.743),
    ("pneumoniamnist", "resnet18", 28): (0.944, 0.854),
    ("retinamnist", "resnet18", 28): (0.717, 0.524),
    ("breastmnist", "resnet18", 28): (0.901, 0.863),
    ("bloodmnist", "resnet18", 28): (0.998, 0.958),
    ("tissuemnist", "resnet18", 28): (0.930, 0.676),
    ("organamnist", "resnet18", 28): (0.997, 0.935),
    ("organcmnist", "resnet18", 28): (0.992, 0.900),
    ("organsmnist", "resnet18", 28): (0.972, 0.782),
    # Extra R18/R50 @ 224/28 PathMNIST references (kept for optional depth runs).
    ("pathmnist", "resnet18", 224): (0.989, 0.909),
    ("pathmnist", "resnet50", 28): (0.990, 0.911),
    ("pathmnist", "resnet50", 224): (0.989, 0.892),
}

# Convenience: the twelve MedMNIST2D datasets in the spec's canonical order,
# each with the ResNet-18 @ 28 reference the replication sweep targets.
DATASETS_2D = [
    "pathmnist", "chestmnist", "dermamnist", "octmnist",
    "pneumoniamnist", "retinamnist", "breastmnist", "bloodmnist",
    "tissuemnist", "organamnist", "organcmnist", "organsmnist",
]
