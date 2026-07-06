#!/usr/bin/env python
"""
replay/name_lookup.py
=====================

Robust accessors for the physical quantities Module 1 needs out of a replayed
PickPlaceCan env: can position, end-effector position, gripper opening, the
can's target placement, and the table/bin surface height.

Every accessor prefers the low-dim observation dict (returned by
``env._get_observations(force_update=True)``) and falls back to direct MuJoCo
``sim`` reads, raising an INFORMATIVE error if neither works. Accessors whose
exact key/attr we could not confirm by executing robosuite (authoring only) are
marked ``# TODO(verify)`` so they are easy to check on the first real run.

Verified against robosuite 1.5 source:
  * env.object_to_id == {"milk":0,"bread":1,"cereal":2,"can":3}   (pick_place.py:218)
  * env.obj_names == ["Milk","Bread","Cereal","Can"]              (pick_place.py:220)
  * env.obj_body_id[obj.name] -> body id                          (pick_place.py:_setup_references)
  * env.target_bin_placements[object_id] -> (x,y,z)              (pick_place.py:576-587)
  * env.bin1_pos[2] -> table/bin surface z                        (pick_place.py:233)
  * eef site via env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]] (pick_place.py:_check_success)
"""

import numpy as np

# Single-robot prefix used by robosuite observation keys ("robot0_eef_pos", ...).
# TODO(verify): assumes one robot indexed 0; multi-robot would need robotN_.
ROBOT_PREFIX = "robot0_"


# ---------------------------------------------------------------------------
# Object identity
# ---------------------------------------------------------------------------
def get_object_id(env, object_type="can"):
    """Integer id of the active object (default the can)."""
    mapping = getattr(env, "object_to_id", None)
    if mapping and object_type in mapping:
        return int(mapping[object_type])
    # Fallback: some builds expose env.object_id directly for single_object_mode.
    if hasattr(env, "object_id") and env.object_id is not None:
        return int(env.object_id)
    raise SystemExit(
        f"could not resolve object id for '{object_type}'. "
        f"env.object_to_id={mapping}, env.object_id={getattr(env, 'object_id', None)}"
    )


def get_object_name(env, object_type="can"):
    """robosuite object name (e.g. 'Can') used as the obj_body_id / obs-key stem."""
    names = getattr(env, "obj_names", None)
    if names:
        oid = get_object_id(env, object_type)
        if 0 <= oid < len(names):
            return names[oid]
    # Fallback: capitalize the type ("can" -> "Can").
    return object_type.capitalize()


# ---------------------------------------------------------------------------
# Per-timestep quantities (obs-first, sim-fallback)
# ---------------------------------------------------------------------------
def object_pos(env, obs, object_name="Can"):
    """World position (3,) of the object's body."""
    key = f"{object_name}_pos"
    if obs is not None and key in obs:
        return np.asarray(obs[key], dtype=np.float64)
    # sim fallback
    body_id_map = getattr(env, "obj_body_id", None)
    if body_id_map and object_name in body_id_map:
        return np.asarray(env.sim.data.body_xpos[body_id_map[object_name]], dtype=np.float64)
    raise SystemExit(
        f"could not read object position for '{object_name}': "
        f"neither obs['{key}'] nor env.obj_body_id['{object_name}'] available."
    )


def eef_pos(env, obs, arm=None):
    """World position (3,) of the (right) end-effector site."""
    key = f"{ROBOT_PREFIX}eef_pos"
    if obs is not None and key in obs:
        return np.asarray(obs[key], dtype=np.float64)
    # Fallback: scan obs for any *eef_pos key.
    if obs is not None:
        for k in obs:
            if k.endswith("eef_pos"):
                return np.asarray(obs[k], dtype=np.float64)
    # sim fallback via the eef site used by _check_success.
    robot = env.robots[0]
    the_arm = arm or robot.arms[0]
    try:
        site_id = robot.eef_site_id[the_arm]
        return np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64)
    except Exception as exc:  # pragma: no cover - runtime guard
        raise SystemExit(f"could not read eef position (obs key '{key}' missing; sim fallback failed: {exc})")


def gripper_qpos(env, obs):
    """Raw gripper finger joint positions (Panda: 2 values)."""
    key = f"{ROBOT_PREFIX}gripper_qpos"
    if obs is not None and key in obs:
        return np.asarray(obs[key], dtype=np.float64)
    if obs is not None:
        for k in obs:
            if k.endswith("gripper_qpos"):
                return np.asarray(obs[k], dtype=np.float64)
    # No robust sim fallback without hard-coding joint names; surface a clear error.
    # TODO(verify): if this triggers, expose gripper_qpos by ensuring the obs is
    # produced with force_update=True, or read env.robots[0]._joint_positions.
    raise SystemExit(
        f"could not read '{key}' from obs; ensure _get_observations(force_update=True) "
        "was used and the gripper obs is enabled."
    )


def gripper_scalar(qpos):
    """Collapse the finger qpos to ONE physical-opening scalar.

    Panda's two prismatic fingers move symmetrically; the total opening is the
    sum of their absolute displacements. Larger => more OPEN. This is a *feature*
    for the network (physical state), distinct from the commanded gripper action.

    NOTE: weak-label logic uses the COMMANDED gripper (action[3]) instead of this
    scalar, to avoid depending on exact finger-qpos ranges we could not measure
    without running robosuite. TODO(verify): confirm sign/scale on first run.
    """
    q = np.asarray(qpos, dtype=np.float64).ravel()
    return float(np.sum(np.abs(q)))


# ---------------------------------------------------------------------------
# Static (per-episode) quantities
# ---------------------------------------------------------------------------
def target_placement(env, object_id):
    """Target (x,y,z) placement of the object in its goal bin.

    This is the extracted 'desired goal' g for the can. Populated in
    _setup_references, so valid after the first reset.
    """
    tbp = getattr(env, "target_bin_placements", None)
    if tbp is None:
        raise SystemExit("env has no target_bin_placements; cannot extract the goal.")
    tbp = np.asarray(tbp, dtype=np.float64)
    if not (0 <= object_id < tbp.shape[0]):
        raise SystemExit(f"object_id {object_id} out of range for target_bin_placements {tbp.shape}.")
    return tbp[object_id].copy()


def table_z(env):
    """Z of the table/bin-1 surface (objects rest on this)."""
    bin1 = getattr(env, "bin1_pos", None)
    if bin1 is not None:
        return float(np.asarray(bin1, dtype=np.float64)[2])
    # TODO(verify): fallback if bin1_pos is renamed in a future robosuite.
    raise SystemExit("env has no bin1_pos; cannot determine table surface height.")
