"""Inference, prediction dumping, and checkpoint evaluation."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import metrics


@torch.no_grad()
def predict(model, loader, task, device, use_amp=False):
    """Run the model over ``loader``; return ``(y_true, y_score)`` numpy arrays.

    ``y_score`` is softmax probabilities (multi-class) or sigmoids
    (multi-label). ``y_true`` is ``(N, 1)`` int for multi-class, ``(N, C)`` for
    multi-label -- matching what ``medmnist.Evaluator`` expects.
    """
    model.eval()
    scores, trues = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        if task == "multi-label, binary-class":
            s = torch.sigmoid(logits)
        else:
            s = F.softmax(logits, dim=1)
        scores.append(s.cpu().numpy())
        trues.append(y.numpy())
    y_score = np.concatenate(scores, axis=0)
    y_true = np.concatenate(trues, axis=0)
    return y_true, y_score


def evaluate_split(model, loader, task, device, use_amp=False):
    """Return ``(auc, acc, y_true, y_score)`` for a split."""
    y_true, y_score = predict(model, loader, task, device, use_amp)
    auc, acc = metrics.evaluate(y_true, y_score, task)
    return auc, acc, y_true, y_score


def prediction_filename(flag, split, auc, acc, seed, size=28):
    """MedMNIST-convention self-describing filename."""
    size_flag = "" if size == 28 else f"_{size}"
    # size_flag is folded into `flag` position per medmnist parser expectations.
    return f"{flag}{size_flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@seed{seed}.csv"


def save_predictions(y_score, path):
    """Save scores in the medmnist result format (index, score_0, ...)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(y_score).to_csv(path, header=None)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
