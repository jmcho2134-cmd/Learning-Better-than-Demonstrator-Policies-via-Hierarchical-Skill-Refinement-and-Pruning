#!/usr/bin/env python
"""
phase_segmenter/posterior.py
============================

The learned posterior ``q_omega(z_t | feature-history)`` — the deployed M1
artifact. This is the piece BORROWED from H-AIRL (a per-timestep latent-option
posterior). We DROP H-AIRL's discriminator / imitation reward / hierarchical RL
policy / adversarial training / EM loop.

Architecture: features (B, L, D) -> GRU -> MLP head -> per-timestep phase logits
(B, L, K).

Directionality:
  * ``bidirectional=True`` (training default): a BiGRU sees the whole window, giving
    cleaner phase boundaries when segmenting a COMPLETE demo offline.
  * ``bidirectional=False``: a causal (past+current only) GRU. Use this if you need
    true ONLINE segmentation of fresh M5 rollouts. A bidirectional model cannot be
    run truly causally; infer.py warns and falls back to full-sequence in that case.

Optional DI-flavored aux head (built only when ``aux_target_dim > 0``, i.e. the
config's ``alpha_ds > 0``): a small MLP that predicts the next action (or next
consequence-delta) from ``[feature_t, softmax(logits_t)]``, trained by MSE. This
makes z_t predictive of the trajectory (directed-information flavor). It is NOT an
AIRL discriminator and NOT a reward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QOmega(nn.Module):
    def __init__(
        self,
        input_dim,
        num_phases,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        aux_target_dim=0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_phases = int(num_phases)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.aux_target_dim = int(aux_target_dim)

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        feat_out = self.hidden_dim * (2 if self.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(feat_out, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_phases),
        )

        # Optional directed-information-flavored aux head.
        if self.aux_target_dim > 0:
            self.aux_head = nn.Sequential(
                nn.Linear(self.input_dim + self.num_phases, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.aux_target_dim),
            )
        else:
            self.aux_head = None

    # -- config-driven constructor ------------------------------------------
    @classmethod
    def from_config(cls, cfg, input_dim, num_phases):
        m = cfg.get("model", {})
        aux_dim = 4 if float(cfg.get("loss", {}).get("alpha_ds", 0.0)) > 0 else 0
        return cls(
            input_dim=input_dim,
            num_phases=num_phases,
            hidden_dim=m.get("hidden_dim", 128),
            num_layers=m.get("num_layers", 2),
            dropout=m.get("dropout", 0.2),
            bidirectional=m.get("bidirectional", True),
            aux_target_dim=aux_dim,
        )

    # -- forward ------------------------------------------------------------
    def forward(self, x):
        """x: (B, L, D) -> logits: (B, L, K)."""
        gru_out, _ = self.gru(x)          # (B, L, feat_out)
        logits = self.head(gru_out)       # (B, L, K)
        return logits

    def aux_predict(self, x, logits):
        """DI-flavored prediction from [features, softmax(logits)].

        Returns (B, L, aux_target_dim) or None if the aux head is disabled.
        Typically compared (MSE) against the NEXT-step action/consequence-delta by
        the caller (train.py shifts the target by one).
        """
        if self.aux_head is None:
            return None
        probs = F.softmax(logits, dim=-1)
        aux_in = torch.cat([x, probs], dim=-1)
        return self.aux_head(aux_in)


def build_metadata(model, cfg, feature_names, phase_names):
    """Serializable dict describing the model, saved alongside the checkpoint."""
    return {
        "input_dim": model.input_dim,
        "num_phases": model.num_phases,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "bidirectional": model.bidirectional,
        "aux_target_dim": model.aux_target_dim,
        "feature_names": list(feature_names),
        "phase_names": list(phase_names),
    }
