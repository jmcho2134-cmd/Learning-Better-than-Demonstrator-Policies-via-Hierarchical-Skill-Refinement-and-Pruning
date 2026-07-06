#!/usr/bin/env python
"""
phase_segmenter/dataset.py
==========================

Turn the per-demo feature bank into training samples for the posterior.

Key properties (per the M1 spec):
  * Split train/val BY DEMO (never leak timesteps of one demo across the split).
  * Normalization stats (mean/std) are computed from the TRAIN split ONLY and
    saved to ``checkpoint_dir/norm_stats.npz`` for reuse at inference.
  * Sequences are chunked into fixed-length windows (``input_window`` = k) with a
    sliding ``stride``; short tails are padded and masked out of the loss.

Weak labels (phase_segmenter/weak_labels.py) are computed on the fly from the
stored features — they are cheap and require no simulator.

*** Imports torch; only run by train.py / infer.py at runtime. ***
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from m1_config import resolve_path, get_phase_names
from phase_segmenter.weak_labels import weak_labels

IGNORE_INDEX = -100  # torch CE ignore index for padded timesteps


# ---------------------------------------------------------------------------
# Loading processed feature files
# ---------------------------------------------------------------------------
def load_processed(cfg):
    """Load every per-demo feature .npz into a list of demo dicts.

    Each dict: features (T,D) float32, labels (T,) int64 (smoothed weak labels),
    feature_names list, source str, g (3,).
    """
    proc_dir = resolve_path(cfg["paths"]["processed_out"])
    paths = sorted(glob.glob(os.path.join(proc_dir, "*.npz")))
    # Exclude the pipeline's own *_m1.npz outputs so we don't re-ingest them.
    paths = [p for p in paths if not p.endswith("_m1.npz")]
    if not paths:
        raise SystemExit(
            f"no feature files in {proc_dir}. Run feature_bank/build_feature_bank.py first."
        )

    demos = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        feats = d["features"].astype(np.float32)
        names = [str(x) for x in d["feature_names"]]
        wl = weak_labels(feats, names, cfg)
        demos.append({
            "features": feats,
            "labels": wl["smoothed"].astype(np.int64),
            "labels_raw": wl["raw"].astype(np.int64),
            "feature_names": names,
            "source": os.path.basename(p),
            "g": d["g"].astype(np.float32) if "g" in d else None,
        })
    return demos


# ---------------------------------------------------------------------------
# Split + normalization
# ---------------------------------------------------------------------------
def split_by_demo(demos, val_frac, seed):
    """Deterministically split the demo LIST into (train, val)."""
    n = len(demos)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * n))) if n > 1 else 0
    val_idx = set(idx[:n_val].tolist())
    train = [demos[i] for i in range(n) if i not in val_idx]
    val = [demos[i] for i in range(n) if i in val_idx]
    if not train:  # tiny datasets: keep at least one train demo
        train, val = demos, []
    return train, val


def compute_norm_stats(train_demos):
    """Mean/std over all TRAIN timesteps (std floored to avoid divide-by-zero)."""
    allfeat = np.concatenate([d["features"] for d in train_demos], axis=0)
    mean = allfeat.mean(axis=0).astype(np.float32)
    std = allfeat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def save_norm_stats(path, mean, std, feature_names, phase_names, input_window):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        mean=mean, std=std,
        feature_names=np.array(feature_names),
        phase_names=np.array(phase_names),
        input_window=np.int64(input_window),
    )


def load_norm_stats(path):
    d = np.load(path, allow_pickle=True)
    return {
        "mean": d["mean"].astype(np.float32),
        "std": d["std"].astype(np.float32),
        "feature_names": [str(x) for x in d["feature_names"]],
        "phase_names": [str(x) for x in d["phase_names"]],
        "input_window": int(d["input_window"]),
    }


# ---------------------------------------------------------------------------
# Windowed dataset
# ---------------------------------------------------------------------------
def _window_starts(T, k, stride):
    """Sliding-window start indices covering [0, T), including a final flush."""
    if T <= k:
        return [0]
    starts = list(range(0, T - k + 1, max(1, stride)))
    if starts[-1] != T - k:
        starts.append(T - k)
    return starts


class PhaseWindowDataset(Dataset):
    """Fixed-length (k) windows of normalized features + per-timestep labels."""

    def __init__(self, demos, mean, std, input_window, stride):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.k = int(input_window)
        self.stride = int(stride)
        self.samples = []  # (features (k,D), labels (k,), mask (k,))

        for d in demos:
            feats = (d["features"] - self.mean) / self.std
            labels = d["labels"]
            T = feats.shape[0]
            D = feats.shape[1]
            for s in _window_starts(T, self.k, self.stride):
                end = min(s + self.k, T)
                win = np.zeros((self.k, D), dtype=np.float32)
                lab = np.full((self.k,), IGNORE_INDEX, dtype=np.int64)
                mask = np.zeros((self.k,), dtype=np.float32)
                n = end - s
                win[:n] = feats[s:end]
                lab[:n] = labels[s:end]
                mask[:n] = 1.0
                self.samples.append((win, lab, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        win, lab, mask = self.samples[i]
        return (
            torch.from_numpy(win),
            torch.from_numpy(lab),
            torch.from_numpy(mask),
        )
