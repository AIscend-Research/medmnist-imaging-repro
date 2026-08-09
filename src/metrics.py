"""Our own implementation of the MedMNIST evaluation metrics.

We reimplement macro one-vs-rest ROC-AUC and argmax accuracy from scratch and
provide :func:`check_agreement`, which asserts our numbers match
``medmnist.Evaluator`` (used *only* as an oracle) to within a tolerance.

Nothing here is copied from ``MedMNIST/experiments``; the only MedMNIST code we
touch is the ``medmnist`` PyPI package's ``Evaluator`` in the verification path.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def _squeeze(y_true, y_score):
    return np.asarray(y_true).squeeze(), np.asarray(y_score).squeeze()


def getAUC(y_true, y_score, task):
    """Macro-averaged AUC.

    For multi-class tasks this is the mean over classes of the one-vs-rest
    ROC-AUC (``(y_true == c)`` vs ``y_score[:, c]``), which is exactly what
    scikit-learn's ``roc_auc_score(average='macro', multi_class='ovr')`` and
    ``medmnist.Evaluator`` compute. ``multi-label, binary-class`` averages the
    per-label binary AUC; ``binary-class`` uses the positive-class score.
    """
    y_true, y_score = _squeeze(y_true, y_score)

    if task == "multi-label, binary-class":
        aucs = [roc_auc_score(y_true[:, i], y_score[:, i]) for i in range(y_score.shape[1])]
        return float(np.mean(aucs))

    if task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        return float(roc_auc_score(y_true, y_score))

    # multi-class / ordinal-regression: macro one-vs-rest.
    aucs = []
    for c in range(y_score.shape[1]):
        aucs.append(roc_auc_score((y_true == c).astype(float), y_score[:, c]))
    return float(np.mean(aucs))


def getACC(y_true, y_score, task, threshold=0.5):
    """Accuracy. Argmax for multi-class; per-label thresholding otherwise."""
    y_true, y_score = _squeeze(y_true, y_score)

    if task == "multi-label, binary-class":
        y_pred = y_score > threshold
        accs = [accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
        return float(np.mean(accs))

    if task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        return float(accuracy_score(y_true, y_score > threshold))

    return float(accuracy_score(y_true, np.argmax(y_score, axis=-1)))


def evaluate(y_true, y_score, task):
    """Return ``(auc, acc)`` for a set of predictions."""
    return getAUC(y_true, y_score, task), getACC(y_true, y_score, task)


def per_class_metrics(y_true, y_score, num_classes):
    """Per-class AUC / precision / recall / F1 (multi-class only).

    Returns a list of dicts, one per class. Used by the per-class extension.
    """
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score

    y_true, y_score = _squeeze(y_true, y_score)
    y_pred = np.argmax(y_score, axis=-1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    rows = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(float)
        try:
            auc_c = roc_auc_score(y_bin, y_score[:, c])
        except ValueError:  # a class absent from this split
            auc_c = float("nan")
        try:
            # PR-AUC (average precision): the honest read under imbalance, where
            # ROC-AUC is optimistic for the rare classes.
            ap_c = average_precision_score(y_bin, y_score[:, c])
        except ValueError:
            ap_c = float("nan")
        rows.append(
            dict(
                cls=int(c),
                auc=float(auc_c),
                ap=float(ap_c),
                precision=float(precision[c]),
                recall=float(recall[c]),
                f1=float(f1[c]),
                support=int(support[c]),
            )
        )
    return rows


def check_agreement(y_true, y_score, task, flag=None, split=None, size=28,
                    root=None, tol=1e-3, verbose=True):
    """Assert our metrics agree with ``medmnist.Evaluator`` within ``tol``.

    Two independent oracle paths are checked:

    1. ``medmnist.evaluator.getAUC/getACC`` on the same arrays.
    2. If ``flag``/``split`` are given, a real ``medmnist.Evaluator`` object
       (which reloads labels from the ``.npz`` on disk) evaluating ``y_score``.

    Raises ``AssertionError`` if either drifts beyond ``tol``.
    """
    from medmnist.evaluator import getAUC as ref_getAUC, getACC as ref_getACC

    our_auc = getAUC(y_true, y_score, task)
    our_acc = getACC(y_true, y_score, task)

    ref_auc = ref_getAUC(np.asarray(y_true), np.asarray(y_score), task)
    ref_acc = ref_getACC(np.asarray(y_true), np.asarray(y_score), task)

    assert abs(our_auc - ref_auc) < tol, f"AUC drift: ours={our_auc:.6f} ref={ref_auc:.6f}"
    assert abs(our_acc - ref_acc) < tol, f"ACC drift: ours={our_acc:.6f} ref={ref_acc:.6f}"

    eval_auc = eval_acc = None
    if flag is not None and split is not None:
        from medmnist import Evaluator

        kwargs = {} if root is None else {"root": root}
        evaluator = Evaluator(flag, split, size=size, **kwargs)
        m = evaluator.evaluate(np.asarray(y_score))
        eval_auc, eval_acc = float(m.AUC), float(m.ACC)
        assert abs(our_auc - eval_auc) < tol, f"AUC drift vs Evaluator: ours={our_auc:.6f} ref={eval_auc:.6f}"
        assert abs(our_acc - eval_acc) < tol, f"ACC drift vs Evaluator: ours={our_acc:.6f} ref={eval_acc:.6f}"

    if verbose:
        print(f"[metrics] agreement OK  ours=({our_auc:.4f},{our_acc:.4f}) "
              f"getX=({ref_auc:.4f},{ref_acc:.4f})"
              + (f" Evaluator=({eval_auc:.4f},{eval_acc:.4f})" if eval_auc is not None else ""))
    return dict(auc=our_auc, acc=our_acc)
