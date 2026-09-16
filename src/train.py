"""Training loop: seeding, best-val-AUC checkpointing, resume, optional AMP.

Protocol (MedMNIST v2 baseline):
    Adam(lr=1e-3), MultiStepLR(gamma=0.1, milestones=[0.5E, 0.75E]),
    batch 128, 100 epochs, CrossEntropyLoss (BCEWithLogits for multi-label),
    no augmentation. Model selection = checkpoint with the highest validation
    macro-AUC seen so far; that checkpoint's test AUC/ACC is reported.
"""

from __future__ import annotations

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

from .evaluate import evaluate_split, load_checkpoint


def set_seed(seed, deterministic=True):
    """Seed python / numpy / torch / CUDA. Returns whether cudnn is deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    return deterministic


def make_criterion(task, class_weight=None, device=None):
    if task == "multi-label, binary-class":
        return nn.BCEWithLogitsLoss()
    w = class_weight.to(device) if (class_weight is not None and device is not None) else class_weight
    return nn.CrossEntropyLoss(weight=w)


def _targets_for_loss(y, task, device):
    if task == "multi-label, binary-class":
        return y.to(torch.float32).to(device)
    return y.squeeze(dim=1).long().to(device) if y.ndim == 2 else y.long().to(device)


def train_one_epoch(model, loader, criterion, optimizer, task, device,
                    scaler=None, use_amp=False):
    model.train()
    running, n_imgs, t0 = 0.0, 0, time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        target = _targets_for_loss(y, task, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, target)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * x.size(0)
        n_imgs += x.size(0)
    dt = time.time() - t0
    return running / max(n_imgs, 1), n_imgs / max(dt, 1e-9), dt


def run_training(model, loaders, task, *, epochs=100, lr=1e-3, device="cuda",
                 class_weight=None, use_amp=False, deterministic=True, seed=0,
                 ckpt_dir=None, ckpt_every=5, resume=True, log_fn=print):
    """Train, selecting the best-val-AUC checkpoint. Resumable.

    Returns a history/result dict. Writes ``best_model.pth`` and periodic
    ``last.pth`` into ``ckpt_dir`` so a run can continue in a fresh Kaggle
    session via ``resume=True``.
    """
    train_loader, val_loader, test_loader = loaders
    model = model.to(device)

    criterion = make_criterion(task, class_weight=class_weight, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    milestones = [int(0.5 * epochs), int(0.75 * epochs)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_val_auc = -1.0
    best_epoch = -1
    history = []

    last_path = os.path.join(ckpt_dir, "last.pth") if ckpt_dir else None
    best_path = os.path.join(ckpt_dir, "best_model.pth") if ckpt_dir else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    if resume and last_path and os.path.exists(last_path):
        ckpt = load_checkpoint(last_path, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt.get("best_val_auc", -1.0)
        best_epoch = ckpt.get("best_epoch", -1)
        history = ckpt.get("history", [])
        log_fn(f"[resume] continuing from epoch {start_epoch} (best_val_auc={best_val_auc:.4f})")

    for epoch in range(start_epoch, epochs):
        loss, ips, dt = train_one_epoch(
            model, train_loader, criterion, optimizer, task, device, scaler, use_amp)
        scheduler.step()

        val_auc, val_acc, _, _ = evaluate_split(model, val_loader, task, device, use_amp)
        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc, best_epoch = val_auc, epoch
            if best_path:
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_auc": val_auc, "val_acc": val_acc}, best_path)

        history.append(dict(epoch=epoch, train_loss=loss, val_auc=val_auc,
                            val_acc=val_acc, imgs_per_sec=ips, epoch_time_s=dt,
                            lr=optimizer.param_groups[0]["lr"]))
        log_fn(f"[e{epoch:03d}] loss={loss:.4f} val_auc={val_auc:.4f} "
               f"val_acc={val_acc:.4f} {ips:.0f} img/s {dt:.1f}s"
               + ("  *best*" if is_best else ""))

        if last_path and ((epoch + 1) % ckpt_every == 0 or epoch == epochs - 1):
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "best_val_auc": best_val_auc,
                        "best_epoch": best_epoch, "history": history}, last_path)

    # Reload best checkpoint for final reporting.
    if best_path and os.path.exists(best_path):
        load_checkpoint(best_path, model, map_location=device)

    train_auc, train_acc, _, _ = evaluate_split(model, train_loader, task, device, use_amp)
    val_auc, val_acc, _, _ = evaluate_split(model, val_loader, task, device, use_amp)
    test_auc, test_acc, y_true, y_score = evaluate_split(model, test_loader, task, device, use_amp)

    return dict(
        best_epoch=best_epoch, best_val_auc=best_val_auc,
        train_auc=train_auc, train_acc=train_acc,
        val_auc=val_auc, val_acc=val_acc,
        test_auc=test_auc, test_acc=test_acc,
        history=history, test_y_true=y_true, test_y_score=y_score,
    )
