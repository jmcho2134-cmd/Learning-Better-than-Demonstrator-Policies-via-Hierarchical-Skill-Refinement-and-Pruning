#!/usr/bin/env python
"""
phase_segmenter/infer.py
========================

Load a trained posterior and segment any feature sequence into z_t. Reusable both
for M1 (segment stored demos) and M5 (segment fresh rollouts).

    from phase_segmenter.infer import PhaseSegmenter
    seg = PhaseSegmenter(checkpoint_path, norm_stats_path)
    z_t = seg.segment(features)      # features (T, D) -> z_t (T,)

Causality note: a unidirectional (``bidirectional=False``) model's output at t
depends only on features <= t, so a single full-sequence pass IS causal and valid
for online M5 use. A bidirectional model cannot be run truly causally; ``segment``
still works for OFFLINE demo segmentation, and ``is_causal`` reports False so M5
callers can require a causal checkpoint.

*** Imports torch; run by the USER at inference time. ***
"""

import os

import numpy as np
import torch

from phase_segmenter.dataset import load_norm_stats
from phase_segmenter.posterior import QOmega


class PhaseSegmenter:
    def __init__(self, checkpoint_path, norm_stats_path=None, device="cpu"):
        if not os.path.isfile(checkpoint_path):
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
        self.device = device
        # weights_only=False: our checkpoint is a trusted local file that also
        # stores a plain-dict "metadata" payload. Newer torch defaults
        # weights_only=True, which can reject that payload. TODO(verify): if your
        # torch is <2.0 and lacks this kwarg, drop it.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        meta = ckpt["metadata"]
        self.phase_names = list(meta["phase_names"])
        self.feature_names = list(meta["feature_names"])

        # norm stats: explicit path > path recorded in the checkpoint
        norm_path = norm_stats_path or ckpt.get("norm_stats")
        if not norm_path or not os.path.isfile(norm_path):
            raise SystemExit(
                f"norm_stats not found (looked at {norm_path}). "
                "Pass norm_stats_path explicitly."
            )
        stats = load_norm_stats(norm_path)
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)

        self.model = QOmega(
            input_dim=meta["input_dim"],
            num_phases=meta["num_phases"],
            hidden_dim=meta["hidden_dim"],
            num_layers=meta["num_layers"],
            bidirectional=meta["bidirectional"],
            aux_target_dim=meta.get("aux_target_dim", 0),
        ).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @property
    def is_causal(self):
        """True iff the model can be used for genuine online/causal segmentation."""
        return not self.model.bidirectional

    def _normalize(self, features):
        features = np.asarray(features, dtype=np.float32)
        if features.shape[1] != self.mean.shape[0]:
            raise SystemExit(
                f"feature dim {features.shape[1]} != model input dim {self.mean.shape[0]}"
            )
        return (features - self.mean) / self.std

    @torch.no_grad()
    def segment_probs(self, features):
        """features (T, D) -> per-phase probabilities (T, K)."""
        x = torch.from_numpy(self._normalize(features)).unsqueeze(0).to(self.device)  # (1,T,D)
        logits = self.model(x)                       # (1,T,K)
        probs = torch.softmax(logits, dim=-1)[0]     # (T,K)
        return probs.cpu().numpy()

    @torch.no_grad()
    def segment(self, features):
        """features (T, D) -> z_t (T,) int phase ids (argmax)."""
        probs = self.segment_probs(features)
        return probs.argmax(axis=-1).astype(np.int64)
