"""Independent reimplementation of the MedMNIST v2 ResNet baselines.

Nothing in this package is imported or adapted from ``MedMNIST/experiments``.
The only MedMNIST code used is the ``medmnist`` PyPI package: its dataset
loaders (for the standardized data) and ``Evaluator`` (as a metric oracle).
"""

__all__ = ["metrics", "models", "data", "train", "evaluate", "run", "aggregate",
           "extensions"]
