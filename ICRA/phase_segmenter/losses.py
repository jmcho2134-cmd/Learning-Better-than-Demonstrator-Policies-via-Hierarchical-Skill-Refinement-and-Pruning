#!/usr/bin/env python
"""
phase_segmenter/losses.py
=========================

Losses for the posterior:

  * ``ce_loss`` (PRIMARY): cross-entropy of per-timestep logits vs the weak
    pseudo-labels (padded timesteps carry IGNORE_INDEX and are skipped).
  * ``temporal_smoothness_loss``: discourages flip-flopping by penalizing the
    squared change of the softmax distribution between adjacent valid timesteps.
  * ``masked_mse`` (OPTIONAL DI aux): MSE for the directed-information-flavored
    head predicting the next action / next consequence-delta conditioned on z_t.
    This is a reconstruction term, NOT an AIRL discriminator or a reward.

train.py combines them with weights (w_ce, w_smooth, alpha_ds) from the config.
"""

import torch
import torch.nn.functional as F

from phase_segmenter.dataset import IGNORE_INDEX


def ce_loss(logits, labels):
    """Cross-entropy over valid timesteps. logits (B,L,K), labels (B,L)."""
    B, L, K = logits.shape
    return F.cross_entropy(
        logits.reshape(B * L, K),
        labels.reshape(B * L),
        ignore_index=IGNORE_INDEX,
    )


def temporal_smoothness_loss(logits, mask):
    """Mean squared change of adjacent softmax distributions over valid pairs.

    logits (B,L,K), mask (B,L) in {0,1}. Returns 0 if there are no valid pairs.
    """
    probs = F.softmax(logits, dim=-1)
    diff = probs[:, 1:, :] - probs[:, :-1, :]         # (B, L-1, K)
    pair_mask = mask[:, 1:] * mask[:, :-1]            # (B, L-1)
    sq = (diff ** 2).sum(dim=-1)                      # (B, L-1)
    denom = pair_mask.sum().clamp(min=1.0)
    return (sq * pair_mask).sum() / denom


def masked_mse(pred, target, valid):
    """Masked MSE. pred/target (B,L,A), valid (B,L) in {0,1}."""
    err = ((pred - target) ** 2).sum(dim=-1)          # (B, L)
    denom = valid.sum().clamp(min=1.0)
    return (err * valid).sum() / denom
