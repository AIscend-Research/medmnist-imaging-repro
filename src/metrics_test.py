"""Standalone agreement test: our metrics vs the medmnist oracle.

Run with:  python -m src.metrics_test
Uses synthetic predictions only (no torch / no dataset download needed).
Exits non-zero if any metric drifts beyond 1e-6 from medmnist's getAUC/getACC.
"""

from __future__ import annotations

import numpy as np

from . import metrics


def main():
    rng = np.random.RandomState(0)

    # multi-class (DermaMNIST=7, PathMNIST=9)
    for C in (7, 9):
        N = 3000
        y_true = rng.randint(0, C, size=(N, 1))
        logits = rng.randn(N, C) + np.eye(C)[y_true.squeeze()] * 1.5
        e = np.exp(logits - logits.max(1, keepdims=True))
        y_score = e / e.sum(1, keepdims=True)
        metrics.check_agreement(y_true, y_score, "multi-class", tol=1e-6)

    # multi-label, binary-class (ChestMNIST=14)
    L, N = 14, 2000
    y_true = (rng.rand(N, L) > 0.7).astype(int)
    y_score = rng.rand(N, L)
    metrics.check_agreement(y_true, y_score, "multi-label, binary-class", tol=1e-6)

    # binary-class
    y_true = rng.randint(0, 2, size=(N, 1))
    y_score = rng.rand(N, 2)
    y_score = y_score / y_score.sum(1, keepdims=True)
    metrics.check_agreement(y_true, y_score, "binary-class", tol=1e-6)

    print("ALL METRIC AGREEMENT TESTS PASSED (tol=1e-6)")


if __name__ == "__main__":
    main()
