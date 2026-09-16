"""Independent reimplementation of the MedMNIST v2 ResNet baselines.

Nothing in this package is imported or adapted from ``MedMNIST/experiments``.
The only MedMNIST code used is the ``medmnist`` PyPI package: its dataset
loaders (for the standardized data) and ``Evaluator`` (as a metric oracle).

The one exception is :mod:`src.reproduction_arm`, which does not implement
anything — it only tabulates, with explicit provenance, the runs that were
executed with the authors' own code as a separate *reproduction* arm.
"""

__all__ = ["metrics", "models", "data", "train", "evaluate", "run", "aggregate",
           "extensions", "figures_replication", "plotting", "reference",
           "from_predictions", "reproduction_arm"]
