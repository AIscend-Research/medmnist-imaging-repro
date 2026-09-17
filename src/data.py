"""Dataset loading and transform pipelines.

We take only the standardized dataset from the ``medmnist`` package (its loader
classes) plus the label array. Everything else here is ours.

Key protocol points, faithful to MedMNIST v2 (2023):

* Always load the **28-pixel** ``.npz`` (``size=28``), ``as_rgb=True``.
* The 224 configs *resize the 28-pixel data inside the transform* using
  nearest-neighbour interpolation. We deliberately do **not** load MedMNIST+'s
  native 224 images, which did not exist for the 2023 paper and would give
  different (better) numbers.
* Normalisation is ``mean=[.5], std=[.5]`` broadcast across the 3 channels
  (roughly maps to [-1, 1]). No train-time augmentation in the baseline.
"""

from __future__ import annotations

import glob
import hashlib
import os
import shutil

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import medmnist
from medmnist import INFO
from medmnist.dataset import DEFAULT_ROOT


def _md5(path, chunk_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _stage_from_mirror(dataset, root, size_flag=""):
    """Copy a valid cached/mirrored ``.npz`` into place so ``medmnist`` never
    has to touch Zenodo at all.

    Zenodo (the package's only download host) has real outages -- retrying a
    dead host just burns GPU time. If a same-named ``.npz`` already sits
    anywhere under a Kaggle input mount (``/kaggle/input/**``) or in
    ``MEDMNIST_MIRROR_DIRS`` (colon-separated), and its MD5 matches the
    checksum ``medmnist`` itself would verify, stage it into ``root`` so the
    integrity check in ``torchvision.datasets.utils.download_url`` short-
    circuits before any network call. A non-matching or missing mirror is a
    silent no-op -- the normal (possibly retried) download still runs.
    """
    info = INFO[dataset]
    filename = f"{dataset}{size_flag}.npz"
    dest = os.path.join(root, filename)
    if os.path.exists(dest) and _md5(dest) == info[f"MD5{size_flag}"]:
        return True  # already staged/cached correctly

    search_dirs = ["/kaggle/input"] + [
        d for d in os.environ.get("MEDMNIST_MIRROR_DIRS", "").split(":") if d
    ]
    candidates = []
    for d in search_dirs:
        candidates.extend(glob.glob(os.path.join(d, "**", filename), recursive=True))

    for cand in candidates:
        try:
            if _md5(cand) == info[f"MD5{size_flag}"]:
                os.makedirs(root, exist_ok=True)
                shutil.copyfile(cand, dest)
                print(f"[data] staged {dataset}{size_flag}.npz from mirror {cand} "
                      "(MD5 verified) -- skipping Zenodo download")
                return True
        except OSError:
            continue
    return False


def build_transform(size):
    """Transform for the given resolution (no augmentation; baseline)."""
    if size == 28:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[.5], std=[.5]),
        ])
    if size == 224:
        # Nearest-neighbour specifically; the 28-pixel source is upsampled here.
        return T.Compose([
            T.Resize((224, 224), interpolation=Image.NEAREST),
            T.ToTensor(),
            T.Normalize(mean=[.5], std=[.5]),
        ])
    raise ValueError(f"unsupported size {size}")


def get_info(dataset):
    return INFO[dataset]


def get_dataset(dataset, split, size, root=None, download=True,
               download_retries=4, download_backoff_s=20):
    """Return a ``medmnist`` dataset object for ``split`` with our transform.

    Data is always loaded from the 28-pixel ``.npz``; ``size`` only controls the
    transform (the 224 pipeline resizes internally).

    Zenodo (the ``medmnist`` package's download host) intermittently returns
    504s on an otherwise-fine connection; the package itself doesn't retry, so
    a single blip kills an otherwise-healthy training run. Retry the whole
    construction (which re-attempts the download if the .npz still isn't
    cached) with backoff before giving up.
    """
    info = INFO[dataset]
    DataClass = getattr(medmnist, info["python_class"])
    transform = build_transform(size)
    kwargs = dict(split=split, transform=transform, download=download,
                  as_rgb=True, size=28)
    if root is not None:
        kwargs["root"] = root

    if download:
        _stage_from_mirror(dataset, root or DEFAULT_ROOT)

    last_err = None
    for attempt in range(download_retries):
        try:
            return DataClass(**kwargs)
        except RuntimeError as e:
            last_err = e
            if attempt < download_retries - 1:
                import time
                wait = download_backoff_s * (attempt + 1)
                print(f"[data] {dataset} download failed (attempt {attempt + 1}/"
                      f"{download_retries}), retrying in {wait}s: {e}")
                time.sleep(wait)
    raise last_err


def get_loaders(dataset, size, batch_size=128, root=None, download=True,
                num_workers=2, sampler=None, eval_batch_size=None,
                pin_memory=True):
    """Build train/val/test dataloaders.

    ``sampler`` (e.g. a ``WeightedRandomSampler``) replaces shuffling on the
    train loader when provided; used by the bias-mitigation extension.
    """
    train_set = get_dataset(dataset, "train", size, root, download)
    val_set = get_dataset(dataset, "val", size, root, download)
    test_set = get_dataset(dataset, "test", size, root, download)

    eval_bs = eval_batch_size or batch_size
    shuffle = sampler is None
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=eval_bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=eval_bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def class_counts(dataset, root=None, download=True):
    """Per-class training-sample counts (multi-class datasets)."""
    train_set = get_dataset(dataset, "train", 28, root, download)
    labels = np.asarray(train_set.labels).squeeze().astype(int)
    n_classes = len(INFO[dataset]["label"])
    counts = np.bincount(labels, minlength=n_classes)
    return counts


def class_weights(dataset, root=None, download=True, normalize=True):
    """Inverse-frequency class weights for weighted CrossEntropyLoss.

    ``w_c ∝ 1 / count_c``; optionally normalised to mean 1.
    """
    counts = class_counts(dataset, root, download).astype(np.float64)
    counts = np.clip(counts, 1, None)
    w = 1.0 / counts
    if normalize:
        w = w * len(w) / w.sum()
    return torch.tensor(w, dtype=torch.float32)


def make_weighted_sampler(dataset, root=None, download=True):
    """A ``WeightedRandomSampler`` giving each class equal expected mass."""
    train_set = get_dataset(dataset, "train", 28, root, download)
    labels = np.asarray(train_set.labels).squeeze().astype(int)
    counts = np.bincount(labels, minlength=len(INFO[dataset]["label"])).astype(np.float64)
    counts = np.clip(counts, 1, None)
    sample_w = (1.0 / counts)[labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w), replacement=True)
    return sampler
