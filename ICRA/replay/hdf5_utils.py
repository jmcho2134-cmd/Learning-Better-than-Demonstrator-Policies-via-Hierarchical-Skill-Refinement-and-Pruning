#!/usr/bin/env python
"""
replay/hdf5_utils.py
====================

Read helpers for the ``demo.hdf5`` files produced by ``collect_pickplace_can.py``.

The demos store ONLY:
  * ``data/demo_*/states``   (T, state_dim)  flattened MuJoCo sim states
  * ``data/demo_*/actions``  (T, 4)          OSC_POSITION actions (3 pos + gripper)
  * ``data/demo_*.attrs["model_file"]``      per-demo MJCF XML string
  * ``data.attrs["env_info"]``               JSON: env_name, robots,
                                             controller_configs, action_dim=4, ...

There is NO observation dataset — anything needing eef/can/target positions must
rebuild the env and REPLAY the states (see replay/replay_states.py).

*** This module only READS files. It never launches a simulator. ***
"""

import glob
import json
import os


# ---------------------------------------------------------------------------
# Finding demo files
# ---------------------------------------------------------------------------
def find_demo_files(demo_root):
    """Return a sorted list of every ``demo.hdf5`` under ``demo_root``.

    Matches the collector's layout ``demo_root/<run>/demo.hdf5`` (either the new
    ``demo_<N>`` folders or the older ``<t1>_<t2>`` timestamp folders). Also
    accepts a ``demo.hdf5`` sitting directly in ``demo_root``.
    """
    if not os.path.isdir(demo_root):
        raise SystemExit(
            f"demo_root does not exist: {demo_root}\n"
            "Collect demos first, e.g.:  python collect_pickplace_can.py --random-can"
        )
    hits = sorted(glob.glob(os.path.join(demo_root, "*", "demo.hdf5")))
    direct = os.path.join(demo_root, "demo.hdf5")
    if os.path.isfile(direct):
        hits = [direct] + hits
    return hits


# ---------------------------------------------------------------------------
# env_info parsing
# ---------------------------------------------------------------------------
def load_env_info(h5file):
    """Parse ``data.attrs["env_info"]`` (JSON) into a dict.

    Args:
        h5file (h5py.File): an open demo file.
    Returns:
        dict with at least env_name, robots, controller_configs (and our extra
        action_dim / arm_controller / control_freq fields).
    """
    data = h5file["data"]
    if "env_info" not in data.attrs:
        raise SystemExit("demo file has no data.attrs['env_info']; cannot rebuild env.")
    raw = data.attrs["env_info"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return json.loads(raw)


def make_kwargs_from_env_info(env_info):
    """Split env_info into (make_kwargs, control_freq).

    IMPORTANT: our collector writes extra keys (action_dim, arm_controller,
    control_freq) into env_info that are NOT valid ``robosuite.make`` kwargs.
    Splatting env_info directly into make() would raise. We therefore keep only
    the make-valid keys and return control_freq separately.

    Returns:
        (make_kwargs: dict, control_freq: int|None)
    """
    # Keys robosuite.make actually accepts from our metadata.
    allowed = ("env_name", "robots", "controller_configs", "env_configuration")
    make_kwargs = {k: env_info[k] for k in allowed if k in env_info}
    control_freq = env_info.get("control_freq", None)
    return make_kwargs, control_freq


def assert_action_dim_4(env_info):
    """Fail loudly if the stored metadata is not the 4-dim OSC_POSITION space."""
    ad = env_info.get("action_dim")
    if ad is not None and int(ad) != 4:
        raise SystemExit(
            f"env_info reports action_dim={ad}, expected 4 (OSC_POSITION). "
            "These demos are incompatible with the 4-dim downstream pipeline."
        )


# ---------------------------------------------------------------------------
# Iterating demos within one file
# ---------------------------------------------------------------------------
def sorted_demo_keys(h5file):
    """Return demo group keys sorted numerically (demo_1, demo_2, ..., demo_10)."""
    keys = list(h5file["data"].keys())

    def _key(name):
        try:
            return int(name.split("_")[-1])
        except (ValueError, IndexError):
            return name

    return sorted(keys, key=_key)


def read_demo(h5file, demo_key):
    """Read one demo's raw arrays + model xml.

    Returns:
        dict(states=(T,S) float64, actions=(T,4) float32, model_xml=str,
             demo_key=str)
    """
    grp = h5file["data"][demo_key]
    states = grp["states"][()]
    actions = grp["actions"][()]
    model_xml = grp.attrs.get("model_file", None)
    if isinstance(model_xml, bytes):
        model_xml = model_xml.decode("utf-8", errors="replace")
    if model_xml is None:
        raise SystemExit(f"{demo_key} has no model_file attr; cannot replay it deterministically.")
    return {
        "states": states,
        "actions": actions,
        "model_xml": model_xml,
        "demo_key": demo_key,
    }
