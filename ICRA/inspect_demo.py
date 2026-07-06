#!/usr/bin/env python
"""
inspect_demo.py
===============

Print the structure of a robosuite ``demo.hdf5`` collected by
``collect_pickplace_can.py`` (or robosuite's own collector).

Shows:
  * top-level keys
  * ``data`` group attributes (date, time, repository_version, env, env_info, ...)
  * the list of demo groups (demo_1, demo_2, ...)
  * shape of ``states`` and ``actions`` per demo (confirm action_dim == 4)
  * per-demo ``model_file`` presence, and any env_info / env_kwargs metadata

Usage::

    python inspect_demo.py ./demos/pickplace_can/<timestamp>/demo.hdf5
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np


def _fmt_attr(value):
    """Render an hdf5 attribute compactly (truncate long strings like XML)."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    s = str(value)
    if len(s) > 120:
        return f"{s[:117]}...  (len={len(s)})"
    return s


def _print_env_info(env_info_raw):
    """env_info is a JSON string; pretty-print the fields we care about."""
    if isinstance(env_info_raw, bytes):
        env_info_raw = env_info_raw.decode("utf-8", errors="replace")
    try:
        info = json.loads(env_info_raw)
    except (ValueError, TypeError):
        print("    (env_info is not valid JSON; raw below)")
        print("   ", _fmt_attr(env_info_raw))
        return

    print("    env_info (parsed JSON):")
    for key in ("env_name", "robots", "action_dim", "arm_controller", "control_freq"):
        if key in info:
            print(f"      {key}: {info[key]}")
    # The arm controller type lives inside controller_configs.body_parts.right.type
    cc = info.get("controller_configs")
    if isinstance(cc, dict):
        print(f"      controller_configs.type: {cc.get('type')}")
        body_parts = cc.get("body_parts", {})
        for arm in ("right", "left"):
            part = body_parts.get(arm)
            if isinstance(part, dict):
                print(f"      controller_configs.body_parts.{arm}.type: {part.get('type')}"
                      f"  (output_max len={len(part.get('output_max', []))})")


def inspect(path):
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    print(f"\nInspecting: {os.path.abspath(path)}")
    print("=" * 70)

    with h5py.File(path, "r") as f:
        # --- top-level keys ---
        top_keys = list(f.keys())
        print(f"top-level keys: {top_keys}")

        if "data" not in f:
            print("WARNING: no 'data' group found; this may not be a robosuite demo file.")
            return 0

        data = f["data"]

        # --- data group attributes ---
        print("\n'data' group attributes:")
        for attr_name in data.attrs:
            if attr_name == "env_info":
                # handled specially below
                continue
            print(f"  {attr_name}: {_fmt_attr(data.attrs[attr_name])}")
        if "env_info" in data.attrs:
            print("  env_info:")
            _print_env_info(data.attrs["env_info"])
        # Some pipelines also stash an 'env_kwargs' attr; show it if present.
        if "env_kwargs" in data.attrs:
            print("  env_kwargs:", _fmt_attr(data.attrs["env_kwargs"]))

        # --- demo groups ---
        demo_keys = [k for k in data.keys()]
        # sort demo_1, demo_2, ... numerically when possible
        def _demo_sort_key(name):
            try:
                return int(name.split("_")[-1])
            except (ValueError, IndexError):
                return name
        demo_keys = sorted(demo_keys, key=_demo_sort_key)
        print(f"\ndemo groups ({len(demo_keys)}): {demo_keys}")

        # --- per-demo shapes ---
        action_dims = set()
        for dk in demo_keys:
            grp = data[dk]
            states_shape = grp["states"].shape if "states" in grp else None
            actions = grp["actions"] if "actions" in grp else None
            actions_shape = actions.shape if actions is not None else None
            has_model = "model_file" in grp.attrs
            if actions_shape is not None and len(actions_shape) == 2:
                action_dims.add(actions_shape[1])
            print(f"\n  {dk}:")
            print(f"    states  shape: {states_shape}")
            print(f"    actions shape: {actions_shape}")
            print(f"    model_file attr present: {has_model}")
            # extra per-demo attrs (excluding the big model_file xml)
            other_attrs = [a for a in grp.attrs if a != "model_file"]
            if other_attrs:
                for a in other_attrs:
                    print(f"    attr {a}: {_fmt_attr(grp.attrs[a])}")

        # --- action-dim summary (pipeline accepts 4-dim OSC_POSITION or 7-dim OSC_POSE) ---
        print("\n" + "-" * 70)
        _labels = {4: "OSC_POSITION (3 pos + 1 gripper)", 7: "OSC_POSE (3 pos + 3 rot + 1 gripper)"}
        if action_dims == {4}:
            print("action dim across demos: 4  ->  OSC_POSITION (3 pos + 1 gripper)  [OK]")
        elif action_dims == {7}:
            print("action dim across demos: 7  ->  OSC_POSE (3 pos + 3 rot + 1 gripper)  [OK]")
        elif action_dims and action_dims <= {4, 7}:
            print(f"action dim(s) across demos: {sorted(action_dims)}  [OK: 4 and/or 7 both supported]")
        elif action_dims:
            print(f"action dim(s) across demos: {sorted(action_dims)}  "
                  f"[WARNING: pipeline expects 4 or 7 (>=3 pos + gripper)]")
        else:
            print("no 2-D action datasets found to determine action dim.")
        print("=" * 70 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Inspect a robosuite demo.hdf5 file.")
    parser.add_argument("path", type=str, help="Path to demo.hdf5")
    args = parser.parse_args()
    sys.exit(inspect(args.path))


if __name__ == "__main__":
    main()
