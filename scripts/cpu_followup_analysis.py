"""
Runs the threshold-tuned baseline control and the robustness-by-rarity sweep
from kaggle_notebooks/phase3_full.ipynb (sections E and H) as pure inference
against an already-trained baseline checkpoint. No training happens here, so
this runs fine on CPU — it only needs `best_baseline.pth` from the Kaggle run
that already produced results/phase3/perclass_baseline.csv etc.

Usage:
    python scripts/cpu_followup_analysis.py --checkpoint /path/to/best_baseline.pth

Output goes to results/phase3/, matching the filenames the notebook itself
would produce, so it's a drop-in replacement for the two GPU-only steps.
"""
import argparse
import os

import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from medmnist import DermaMNIST
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import label_binarize

N_CLASSES = 7
BATCH = 128
CLASS_NAMES = [
    'Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
    'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions',
]
RARE_CLASSES = ['Dermatofibroma', 'Vascular lesions', 'Actinic keratoses']
COMMON_CLASSES = [c for c in CLASS_NAMES if c not in RARE_CLASSES]
RARE_IDX = [CLASS_NAMES.index(c) for c in RARE_CLASSES]
COMMON_IDX = [CLASS_NAMES.index(c) for c in COMMON_CLASSES]


def get_predictions(model, loader, device):
    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            s = torch.softmax(model(x.to(device)), dim=1)
            scores.append(s.cpu().numpy())
            targets.append(y.numpy().flatten())
    return np.concatenate(targets), np.vstack(scores).argmax(1), np.vstack(scores)


def per_class_metrics(y_true, y_pred, y_score):
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    return pd.DataFrame({
        'Class':     CLASS_NAMES,
        'N':         np.bincount(y_true, minlength=N_CLASSES),
        'AUC':       [round(roc_auc_score(y_bin[:, i], y_score[:, i]), 4) for i in range(N_CLASSES)],
        'Accuracy':  np.round(confusion_matrix(y_true, y_pred).diagonal() /
                               confusion_matrix(y_true, y_pred).sum(axis=1), 4),
        'Precision': np.round(precision_score(y_true, y_pred, average=None, zero_division=0), 4),
        'Recall':    np.round(recall_score(y_true, y_pred, average=None, zero_division=0), 4),
        'F1':        np.round(f1_score(y_true, y_pred, average=None, zero_division=0), 4),
    })


def aggregate(y_true, y_pred, y_score):
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    return {
        'macro_auc': round(roc_auc_score(y_bin, y_score, average='macro'), 4),
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'macro_f1':  round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
    }


def tune_thresholds(y_true_val, y_score_val, n_classes):
    y_bin = label_binarize(y_true_val, classes=list(range(n_classes)))
    thresholds = np.full(n_classes, 0.5)
    for i in range(n_classes):
        best_t, best_f1 = 0.5, -1
        for t in np.linspace(0.01, 0.99, 99):
            pred = (y_score_val[:, i] >= t).astype(int)
            f1 = f1_score(y_bin[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds


def apply_thresholds(y_score, thresholds):
    return (y_score / thresholds[None, :]).argmax(1)


def corrupt_batch(x_batch, corruption, severity):
    x = x_batch.clone()
    if severity == 0:
        return x
    if corruption == 'gaussian':
        x = x + torch.randn_like(x) * (0.05 * severity)
    elif corruption == 'brightness_dark':
        x = x * max(1.0 - 0.15 * severity, 0.05)
    elif corruption == 'brightness_bright':
        x = x * (1.0 + 0.15 * severity)
    return x


def evaluate_corrupted(model, loader, corruption, severity, device):
    model.eval()
    all_scores, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            xc = corrupt_batch(x.to(device), corruption, severity)
            s = torch.softmax(model(xc), dim=1)
            all_scores.append(s.cpu().numpy())
            all_targets.append(y.numpy().flatten())
    yt = np.concatenate(all_targets)
    ys = np.vstack(all_scores)
    yp = ys.argmax(1)
    return yt, yp, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='Path to best_baseline.pth from the Kaggle run')
    ap.add_argument('--out', default='results/phase3', help='Output directory')
    args = ap.parse_args()

    device = torch.device('cpu')
    os.makedirs(args.out, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])
    ds_val = DermaMNIST(split='val', transform=transform, download=True)
    ds_test = DermaMNIST(split='test', transform=transform, download=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False, num_workers=0)
    loader_test = DataLoader(ds_test, batch_size=BATCH, shuffle=False, num_workers=0)

    model = resnet18(weights=None, num_classes=N_CLASSES).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['net'] if 'net' in ckpt else ckpt)
    model.eval()

    print('Running baseline inference on test set (CPU)...')
    yt_base, yp_base, ys_base = get_predictions(model, loader_test, device)
    agg_base = aggregate(yt_base, yp_base, ys_base)
    print(f'Baseline: AUC={agg_base["macro_auc"]:.4f} Acc={agg_base["accuracy"]:.4f}')

    # --- Threshold-tuned baseline control ---
    print('\nTuning per-class thresholds on validation set...')
    yt_val, _, ys_val = get_predictions(model, loader_val, device)
    thresholds = tune_thresholds(yt_val, ys_val, N_CLASSES)
    for n, t in zip(CLASS_NAMES, thresholds):
        print(f'  {n:<36}: {t:.2f}')

    yp_tuned = apply_thresholds(ys_base, thresholds)
    df_tuned = per_class_metrics(yt_base, yp_tuned, ys_base)
    agg_tuned = aggregate(yt_base, yp_tuned, ys_base)
    df_tuned.to_csv(f'{args.out}/perclass_baseline_threshold_tuned.csv', index=False)

    equity_path = f'{args.out}/equity_tradeoff.csv'
    equity_df = pd.read_csv(equity_path)
    equity_df = equity_df[equity_df.Model != 'Baseline (threshold-tuned)']
    tuned_row = {
        'Model': 'Baseline (threshold-tuned)',
        'Macro AUC': agg_tuned['macro_auc'],
        'Accuracy': agg_tuned['accuracy'],
        'Macro F1': agg_tuned['macro_f1'],
        'Dermato F1': float(df_tuned[df_tuned.Class == 'Dermatofibroma']['F1'].values[0]),
        'Melanoma F1': float(df_tuned[df_tuned.Class == 'Melanoma']['F1'].values[0]),
        'Vascular F1': float(df_tuned[df_tuned.Class == 'Vascular lesions']['F1'].values[0]),
        'Nevi F1': float(df_tuned[df_tuned.Class == 'Melanocytic nevi']['F1'].values[0]),
        'Actinic F1': float(df_tuned[df_tuned.Class == 'Actinic keratoses']['F1'].values[0]),
    }
    equity_df = pd.concat([equity_df, pd.DataFrame([tuned_row])], ignore_index=True)
    equity_df.to_csv(equity_path, index=False)
    print('\n=== Equity-Accuracy Tradeoff Table (updated) ===')
    print(equity_df.to_string(index=False))

    # --- Robustness-by-rarity sweep ---
    print('\nRunning corruption severity sweep on test set (CPU, inference-only)...')
    corruptions = ['gaussian', 'brightness_dark', 'brightness_bright']
    severities = [0, 1, 2, 3, 4, 5]
    rob_rows = []
    for corr in corruptions:
        for sev in severities:
            if sev == 0 and corr != corruptions[0]:
                continue
            yt_c, yp_c, ys_c = evaluate_corrupted(model, loader_test, corr, sev, device)
            agg_c = aggregate(yt_c, yp_c, ys_c)
            f1s_c = f1_score(yt_c, yp_c, average=None, zero_division=0)
            rob_rows.append({
                'Model': 'Baseline',
                'Corruption': 'Clean' if sev == 0 else corr,
                'Severity': sev,
                **agg_c,
                'Common F1 (avg)': float(np.mean(f1s_c[COMMON_IDX])),
                'Rare F1 (avg)': float(np.mean(f1s_c[RARE_IDX])),
                'Dermato F1': float(f1s_c[CLASS_NAMES.index('Dermatofibroma')]),
                'Melanoma F1': float(f1s_c[CLASS_NAMES.index('Melanoma')]),
                'Nevi F1': float(f1s_c[CLASS_NAMES.index('Melanocytic nevi')]),
            })
            print(f'{("Clean" if sev == 0 else corr):<18} sev={sev}: AUC={agg_c["macro_auc"]:.4f} '
                  f'Common_F1={np.mean(f1s_c[COMMON_IDX]):.3f} Rare_F1={np.mean(f1s_c[RARE_IDX]):.3f}')

    rob_df = pd.DataFrame(rob_rows).round(4)
    rob_df.to_csv(f'{args.out}/robustness_results.csv', index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(corruptions), figsize=(5 * len(corruptions), 4.5))
    clean_row = rob_df[rob_df.Corruption == 'Clean'].iloc[0]
    for ax, corr in zip(axes, corruptions):
        sub = pd.concat([pd.DataFrame([clean_row]), rob_df[rob_df.Corruption == corr]])
        sub = sub.sort_values('Severity')
        ax2 = ax.twinx()
        ax.plot(sub['Severity'], sub['Common F1 (avg)'], 'o-', color='#4878d0', label='Common F1 (avg)')
        ax.plot(sub['Severity'], sub['Rare F1 (avg)'], 'o-', color='#d65f5f', label='Rare F1 (avg)')
        ax2.plot(sub['Severity'], sub['macro_auc'], 's--', color='gray', alpha=0.6, label='Macro AUC')
        ax.set_xlabel('Corruption severity')
        ax.set_ylabel('F1 (avg)')
        ax2.set_ylabel('Macro AUC')
        ax.set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        ax.set_title(corr)
        if corr == corruptions[0]:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, fontsize=7, loc='lower left')

    fig.suptitle('Robustness by Class Rarity: Common vs. Rare F1 Under Corruption', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{args.out}/robustness_by_rarity.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved: {args.out}/robustness_by_rarity.png')
    print(f'Saved: {args.out}/robustness_results.csv')
    print(f'Saved: {equity_path} (with threshold-tuned row)')
    print(f'Saved: {args.out}/perclass_baseline_threshold_tuned.csv')


if __name__ == '__main__':
    main()
