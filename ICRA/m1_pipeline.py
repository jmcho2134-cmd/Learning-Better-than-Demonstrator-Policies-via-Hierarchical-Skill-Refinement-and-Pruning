#!/usr/bin/env python
"""
m1_pipeline.py
==============

End-to-end Module 1: turn collected demos into the ``(g, z_t)`` per-demo artifacts
that Module 2's forward-consequence model ``F_eta(s, a, z_t, g)`` consumes.

    python m1_pipeline.py --config configs/m1_goal_phase_pickplace_can.yaml

Steps per demo:
    replay -> RELATIVE features (feature bank) -> weak labels -> z_t (trained
    posterior if a checkpoint exists, else the smoothed weak labels as a labeled
    fallback) ; and the EXTRACTED goal g.

Writes ``data/processed/<run>__<demo>_m1.npz``:
    features (T,15), z_t (T,), g (3,), feature_names, phase_names,
    z_source ("q_omega" | "weak_fallback"), source_file, source_demo.

If the feature bank has not been built yet, this script builds it first.

*** Building features runs robosuite; running the posterior runs torch. Executed
    by the USER after collecting demos. ***
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m1_config import load_config, resolve_path, get_phase_names
from phase_segmenter.weak_labels import weak_labels


def _feature_files(cfg):
    proc = resolve_path(cfg["paths"]["processed_out"])
    files = sorted(glob.glob(os.path.join(proc, "*.npz")))
    return [f for f in files if not f.endswith("_m1.npz")]


def _maybe_build(cfg):
    """Build the feature bank if it is empty."""
    if _feature_files(cfg):
        return
    print("[m1] no feature bank found; building it now...")
    from feature_bank.build_feature_bank import build_all
    build_all(cfg)


def _load_segmenter(cfg):
    ckpt_dir = resolve_path(cfg["paths"]["checkpoint_dir"])
    ckpt = os.path.join(ckpt_dir, "best.pt")
    norm = os.path.join(ckpt_dir, "norm_stats.npz")
    if not os.path.isfile(ckpt):
        print(f"[m1] no trained posterior at {ckpt}.")
        print("     -> z_t will FALL BACK to smoothed weak labels. Train the posterior")
        print("        (phase_segmenter/train.py) for the learned, deployable z_t.")
        return None
    from phase_segmenter.infer import PhaseSegmenter
    seg = PhaseSegmenter(ckpt, norm_stats_path=norm, device="cpu")
    print(f"[m1] using trained posterior {ckpt} (causal={seg.is_causal})")
    return seg


def main():
    parser = argparse.ArgumentParser(description="M1 end-to-end: demos -> (g, z_t) for Module 2.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the feature bank even if it already exists.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.rebuild:
        from feature_bank.build_feature_bank import build_all
        build_all(cfg)
    else:
        _maybe_build(cfg)

    phase_names = get_phase_names(cfg)
    seg = _load_segmenter(cfg)
    proc = resolve_path(cfg["paths"]["processed_out"])

    n = 0
    for path in _feature_files(cfg):
        d = np.load(path, allow_pickle=True)
        features = d["features"].astype(np.float32)
        names = [str(x) for x in d["feature_names"]]

        if seg is not None:
            z_t = seg.segment(features).astype(np.int64)
            z_source = "q_omega"
        else:
            z_t = weak_labels(features, names, cfg)["smoothed"].astype(np.int64)
            z_source = "weak_fallback"

        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(proc, stem + "_m1.npz")
        np.savez(
            out_path,
            features=features,
            z_t=z_t,
            g=d["g"].astype(np.float32) if "g" in d else np.zeros(3, np.float32),
            feature_names=np.array(names),
            phase_names=np.array(phase_names),
            z_source=str(z_source),
            source_file=str(d["source_file"]) if "source_file" in d else "",
            source_demo=str(d["source_demo"]) if "source_demo" in d else "",
        )
        n += 1
        print(f"  [m1] {stem}: z_t({z_t.shape}, {z_source})  g={np.round(d['g'], 3) if 'g' in d else None}  -> {out_path}")

    print(f"[m1] wrote {n} (g, z_t) artifact(s) to {proc}  (files ending in _m1.npz)")
    if seg is None:
        print("[m1] NOTE: z_t used the weak-label fallback. Train the posterior for the "
              "deployable learned segmentation before feeding Module 2.")


if __name__ == "__main__":
    main()
