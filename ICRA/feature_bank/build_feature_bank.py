#!/usr/bin/env python
"""
feature_bank/build_feature_bank.py
==================================

CLI: replay every demo, extract RELATIVE features + the extracted goal g, and
save one ``.npz`` per demo to ``data/processed/``.

    python feature_bank/build_feature_bank.py --config configs/m1_goal_phase_pickplace_can.yaml

Per-demo output ``data/processed/<run>__<demo>.npz`` contains:
    features      (T, 15) float32   RELATIVE features (model input)
    actions       (T, 4)  float32   raw OSC_POSITION actions
    feature_names (15,)    str
    g             (3,)     float32   extracted goal (target placement)
    g_source      ()       str       "target_bin_placements" or "final_obj_pos"
    table_z       ()       float32
    object_name   ()       str
    source_file   ()       str, source_demo () str
    abs_obj_pos / abs_eef_pos / abs_target  (debug only; NEVER a model input)

*** Runs robosuite (rebuilds env + replays). The USER runs this after collecting
    demos; it is authored here, not executed. ***
"""

import argparse
import os

import numpy as np

# --- make the project root importable no matter the CWD ---
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m1_config import load_config, resolve_path
from replay.hdf5_utils import find_demo_files, load_env_info, sorted_demo_keys, read_demo
from replay.replay_states import DemoReplayer
from replay import name_lookup
from feature_bank.features import compute_features, FEATURE_NAMES
from goal.extract_goal import extract_goal


def _processed_name(hdf5_path, demo_key):
    """Stable, collision-free output stem: '<env-folder>__<run-folder>__<demo_key>'.

    collect_demo.py writes demos as ``demos/<Env>_<Robot>/demo_<N>/demo.hdf5``, so
    the run folder (``demo_<N>``) restarts at demo_1 for EACH environment. Include
    the parent (``<Env>_<Robot>``) folder so features from different environments
    don't overwrite each other in data/processed/.
    """
    run = os.path.basename(os.path.dirname(hdf5_path)) or "root"          # demo_<N>
    parent = os.path.basename(os.path.dirname(os.path.dirname(hdf5_path)))  # <Env>_<Robot>
    stem = f"{parent}__{run}" if parent else run
    return f"{stem}__{demo_key}"


def object_type_from_env_info(env_info, fallback="can"):
    """Resolve the PickPlace object type for a demo from its stored env_info.

    "Everything follows collect_demo": prefer an explicit ``object_type`` written
    by the collector; else derive it from the env name (``PickPlaceCan`` -> "can",
    ``PickPlaceMilk`` -> "milk", ...); else fall back to the config value.
    """
    ot = env_info.get("object_type")
    if ot:
        return str(ot).lower()
    name = str(env_info.get("env_name", ""))
    if name.startswith("PickPlace") and len(name) > len("PickPlace"):
        return name[len("PickPlace"):].lower()   # PickPlaceMilk -> "milk"
    return fallback


def build_one_demo(replayer, env_info, demo, cfg):
    """Replay one demo -> (features (T,D), actions (T,A), extras dict).

    Actions may be 4-dim (OSC_POSITION) or 7-dim (OSC_POSE); features only use the
    position deltas + gripper, so the feature dim is 15 regardless.
    """
    object_type = object_type_from_env_info(env_info, cfg.get("task", {}).get("object_type", "can"))
    states = demo["states"]
    actions = np.asarray(demo["actions"], dtype=np.float32)
    T = states.shape[0]
    if actions.shape[0] != T:
        # gather_demonstrations_as_hdf5 deletes the trailing state so len matches;
        # if not, align to the shorter length and warn.
        n = min(T, actions.shape[0])
        print(f"    [warn] {demo['demo_key']}: states({T}) != actions({actions.shape[0]}); "
              f"truncating to {n}.")
        T = n

    feats = np.zeros((T, len(FEATURE_NAMES)), dtype=np.float32)
    abs_obj = np.zeros((T, 3), dtype=np.float32)
    abs_eef = np.zeros((T, 3), dtype=np.float32)

    statics = {}  # filled on first step
    for t, obs in replayer.replay(env_info, demo["model_xml"], states):
        if t >= T:
            break
        env = replayer.env
        if not statics:
            object_id = name_lookup.get_object_id(env, object_type)
            object_name = name_lookup.get_object_name(env, object_type)
            statics = {
                "object_id": object_id,
                "object_name": object_name,
                "target": name_lookup.target_placement(env, object_id),
                "table_z": name_lookup.table_z(env),
            }

        obj_pos = name_lookup.object_pos(env, obs, statics["object_name"])
        eef = name_lookup.eef_pos(env, obs)
        grip = name_lookup.gripper_scalar(name_lookup.gripper_qpos(env, obs))
        feat, dbg = compute_features(
            obj_pos, eef, grip, actions[t], statics["target"], statics["table_z"]
        )
        feats[t] = feat
        abs_obj[t] = dbg["obj_pos"]
        abs_eef[t] = dbg["eef_pos"]

    # Extract the goal g (target placement; fallback to final resting obj pos).
    g, g_source = extract_goal(
        replayer.env, object_type=object_type, final_obj_pos=abs_obj[-1] if T else None
    )

    extras = {
        "feature_names": np.array(FEATURE_NAMES),
        "g": np.asarray(g, dtype=np.float32),
        "g_source": str(g_source),
        "table_z": np.float32(statics.get("table_z", np.nan)),
        "object_name": str(statics.get("object_name", "")),
        "abs_obj_pos": abs_obj,
        "abs_eef_pos": abs_eef,
        "abs_target": np.asarray(statics.get("target", np.zeros(3)), dtype=np.float32),
    }
    return feats, actions[:T], extras


def build_all(cfg):
    """Build the feature bank for every demo; return list of saved paths."""
    demo_root = resolve_path(cfg["paths"]["demo_root"])
    out_dir = resolve_path(cfg["paths"]["processed_out"])
    os.makedirs(out_dir, exist_ok=True)

    import h5py

    files = find_demo_files(demo_root)
    if not files:
        raise SystemExit(f"no demo.hdf5 found under {demo_root}. Collect demos first.")
    print(f"[build] {len(files)} demo file(s) under {demo_root}")

    saved = []
    with DemoReplayer(render=False) as replayer:
        for path in files:
            with h5py.File(path, "r") as f:
                env_info = load_env_info(f)
                for demo_key in sorted_demo_keys(f):
                    demo = read_demo(f, demo_key)
                    feats, acts, extras = build_one_demo(replayer, env_info, demo, cfg)
                    stem = _processed_name(path, demo_key)
                    out_path = os.path.join(out_dir, stem + ".npz")
                    np.savez(
                        out_path,
                        features=feats,
                        actions=acts,
                        source_file=str(path),
                        source_demo=str(demo_key),
                        **extras,
                    )
                    saved.append(out_path)
                    print(f"  [saved] {out_path}   features={feats.shape}  g_source={extras['g_source']}")
    print(f"[build] wrote {len(saved)} feature file(s) to {out_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Build the RELATIVE feature bank from demos.")
    parser.add_argument("--config", required=True, help="Path to the M1 YAML config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_all(cfg)


if __name__ == "__main__":
    main()
