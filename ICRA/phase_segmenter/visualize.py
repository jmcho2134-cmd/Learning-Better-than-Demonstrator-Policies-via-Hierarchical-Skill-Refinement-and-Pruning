#!/usr/bin/env python
"""
phase_segmenter/visualize.py
============================

Per-demo phase timeline: weak (bootstrap) labels vs the trained posterior's
prediction. Saves a PNG (matplotlib, optional) and a CSV to ``data/reports/``.

    python phase_segmenter/visualize.py --config configs/m1_goal_phase_pickplace_can.yaml

If matplotlib is not installed the CSV is still written and a one-line note is
printed (plots are a convenience, not required by the pipeline). If no trained
checkpoint exists yet, only the weak labels are shown.

*** Imports torch only when a checkpoint exists (via infer). ***
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m1_config import load_config, resolve_path, get_phase_names
from phase_segmenter.dataset import load_processed


def _load_segmenter(cfg):
    """Return a PhaseSegmenter if a checkpoint exists, else None."""
    ckpt_dir = resolve_path(cfg["paths"]["checkpoint_dir"])
    ckpt = os.path.join(ckpt_dir, "best.pt")
    norm = os.path.join(ckpt_dir, "norm_stats.npz")
    if not os.path.isfile(ckpt):
        print(f"[viz] no checkpoint at {ckpt}; showing weak labels only.")
        return None
    from phase_segmenter.infer import PhaseSegmenter  # torch import deferred
    return PhaseSegmenter(ckpt, norm_stats_path=norm, device="cpu")


def _write_csv(path, weak, pred, phase_names):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "weak_id", "weak_name", "pred_id", "pred_name"])
        for t in range(len(weak)):
            wn = phase_names[weak[t]] if 0 <= weak[t] < len(phase_names) else str(weak[t])
            if pred is not None:
                pn = phase_names[pred[t]] if 0 <= pred[t] < len(phase_names) else str(pred[t])
                w.writerow([t, int(weak[t]), wn, int(pred[t]), pn])
            else:
                w.writerow([t, int(weak[t]), wn, "", ""])


def _plot(path, weak, pred, phase_names, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; wrote CSV only. (pip install matplotlib)")
        return False
    t = np.arange(len(weak))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(t, weak, where="post", label="weak (bootstrap)", linewidth=2)
    if pred is not None:
        ax.step(t, pred, where="post", label="q_omega (learned)", linewidth=1.5, alpha=0.8)
    ax.set_yticks(range(len(phase_names)))
    ax.set_yticklabels(phase_names)
    ax.set_xlabel("timestep")
    ax.set_ylabel("phase")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize weak vs learned phase timelines.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    phase_names = get_phase_names(cfg)
    reports_dir = resolve_path(cfg["paths"]["reports_out"])
    os.makedirs(reports_dir, exist_ok=True)

    demos = load_processed(cfg)
    seg = _load_segmenter(cfg)

    for d in demos:
        weak = d["labels"]
        pred = seg.segment(d["features"]) if seg is not None else None
        stem = os.path.splitext(d["source"])[0]
        csv_path = os.path.join(reports_dir, stem + "_timeline.csv")
        png_path = os.path.join(reports_dir, stem + "_timeline.png")
        _write_csv(csv_path, weak, pred, phase_names)
        _plot(png_path, weak, pred, phase_names, title=stem)
        print(f"[viz] {stem}: csv -> {csv_path}")
    print(f"[viz] wrote {len(demos)} report(s) to {reports_dir}")


if __name__ == "__main__":
    main()
