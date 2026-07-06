#!/usr/bin/env python
"""
phase_segmenter/train.py
========================

Train the posterior q_omega on the weak-label bootstrap signal.

    python phase_segmenter/train.py --config configs/m1_goal_phase_pickplace_can.yaml

  * Split train/val BY DEMO; normalization stats from TRAIN only (saved).
  * Primary loss: CE vs weak labels (w_ce). Optional temporal smoothness (w_smooth)
    and DI-flavored next-action reconstruction (alpha_ds, default 0).
  * Reports accuracy + macro-F1 vs weak labels (macro-F1 needs scikit-learn; falls
    back to accuracy if it is not installed).
  * Saves ``checkpoint_dir/norm_stats.npz`` and ``checkpoint_dir/best.pt``.

*** Imports torch; run by the USER after building the feature bank. ***
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from m1_config import load_config, resolve_path, get_phase_names
from feature_bank.features import feature_index, FEATURE_DIM
from phase_segmenter.dataset import (
    load_processed, split_by_demo, compute_norm_stats, save_norm_stats,
    PhaseWindowDataset, IGNORE_INDEX,
)
from phase_segmenter.posterior import QOmega, build_metadata
from phase_segmenter import losses as L


def _device(cfg):
    want = str(cfg.get("training", {}).get("device", "auto")).lower()
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return want


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _macro_f1(y_true, y_pred, num_classes):
    """Macro-F1 via sklearn if available, else return None."""
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        return None
    return float(f1_score(y_true, y_pred, labels=list(range(num_classes)),
                          average="macro", zero_division=0))


def run_epoch(model, loader, opt, cfg, device, action_cols, train=True):
    model.train(train)
    w_ce = float(cfg["loss"]["w_ce"])
    w_smooth = float(cfg["loss"].get("w_smooth", 0.0))
    alpha_ds = float(cfg["loss"].get("alpha_ds", 0.0))

    total, n_batches = 0.0, 0
    all_true, all_pred = [], []
    for win, lab, mask in loader:
        win, lab, mask = win.to(device), lab.to(device), mask.to(device)
        logits = model(win)                         # (B,L,K)
        loss = w_ce * L.ce_loss(logits, lab)
        if w_smooth > 0:
            loss = loss + w_smooth * L.temporal_smoothness_loss(logits, mask)
        if alpha_ds > 0 and model.aux_head is not None:
            aux_pred = model.aux_predict(win, logits)          # (B,L,4)
            target = win[:, 1:, action_cols]                   # (B,L-1,4) normalized next action
            pred = aux_pred[:, :-1, :]                         # (B,L-1,4)
            valid = mask[:, 1:] * mask[:, :-1]
            loss = loss + alpha_ds * L.masked_mse(pred, target, valid)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total += float(loss.item())
        n_batches += 1

        # collect predictions on VALID timesteps for metrics
        with torch.no_grad():
            pred_cls = logits.argmax(dim=-1)                   # (B,L)
            valid = (lab != IGNORE_INDEX)
            all_true.append(lab[valid].cpu().numpy())
            all_pred.append(pred_cls[valid].cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    acc = float((y_true == y_pred).mean()) if y_true.size else float("nan")
    return total / max(1, n_batches), acc, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Train the phase-segmenter posterior.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["training"].get("seed", 0))
    _set_seed(seed)
    device = _device(cfg)
    print(f"[train] device={device}  seed={seed}")

    phase_names = get_phase_names(cfg)
    K = len(phase_names)

    # --- data ---
    demos = load_processed(cfg)
    print(f"[train] loaded {len(demos)} demo(s)")
    train_demos, val_demos = split_by_demo(
        demos, float(cfg["training"].get("val_split_by_demo", 0.2)), seed
    )
    print(f"[train] split -> {len(train_demos)} train / {len(val_demos)} val (by demo)")

    mean, std = compute_norm_stats(train_demos)
    ckpt_dir = resolve_path(cfg["paths"]["checkpoint_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)
    norm_path = os.path.join(ckpt_dir, "norm_stats.npz")
    save_norm_stats(norm_path, mean, std, demos[0]["feature_names"], phase_names,
                    int(cfg["model"]["input_window"]))
    print(f"[train] saved norm stats -> {norm_path}")

    k = int(cfg["model"]["input_window"])
    stride = int(cfg["model"].get("stride", k))
    bs = int(cfg["training"].get("batch_size", 32))
    nw = int(cfg["training"].get("num_workers", 0))
    train_ds = PhaseWindowDataset(train_demos, mean, std, k, stride)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    if val_demos:
        val_ds = PhaseWindowDataset(val_demos, mean, std, k, stride)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    else:
        val_loader = None

    input_dim = mean.shape[0]
    if input_dim != FEATURE_DIM:
        print(f"[train][warn] feature dim {input_dim} != expected {FEATURE_DIM}")

    # --- model / opt ---
    model = QOmega.from_config(cfg, input_dim=input_dim, num_phases=K).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    action_cols = [feature_index(n) for n in ("action_0", "action_1", "action_2", "action_3")]

    # --- loop ---
    epochs = int(cfg["training"].get("epochs", 40))
    best_score, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, opt, cfg, device, action_cols, train=True)
        if val_loader is not None:
            va_loss, va_acc, yt, yp = run_epoch(model, val_loader, None, cfg, device, action_cols, train=False)
            f1 = _macro_f1(yt, yp, K)
            score = f1 if f1 is not None else va_acc
            f1_str = f"{f1:.3f}" if f1 is not None else "n/a(no sklearn)"
            print(f"[ep {ep:03d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.3f} macroF1 {f1_str}")
        else:
            score = tr_acc
            print(f"[ep {ep:03d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | (no val split)")

        if score >= best_score:
            best_score = score
            best_state = {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}

    # --- save best ---
    if best_state is None:
        best_state = {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    torch.save(
        {
            "state_dict": best_state,
            "metadata": build_metadata(model, cfg, demos[0]["feature_names"], phase_names),
            "norm_stats": norm_path,
            "best_score": best_score,
        },
        ckpt_path,
    )
    print(f"[train] saved best checkpoint (score={best_score:.3f}) -> {ckpt_path}")


if __name__ == "__main__":
    main()
