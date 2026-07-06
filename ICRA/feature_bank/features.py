#!/usr/bin/env python
"""
feature_bank/features.py
========================

RELATIVE, position-invariant features for Module 1/2.

WHY RELATIVE: the can spawns at different positions across demos (``--random-can``).
Features are expressed as eef->obj and obj->target vectors (plus scalar distances,
height, gripper, and the action), so a policy/segmenter learns "close the gap"
behaviour that generalizes across can positions instead of memorizing world
coordinates. Absolute world coordinates MAY be stored for debugging but MUST NOT
be model inputs.

Feature layout (D = 15):
    [0:3]   eef_to_obj        obj_pos - eef_pos
    [3:6]   obj_to_target     target  - obj_pos
    [6]     eef_obj_dist      ||eef_to_obj||
    [7]     obj_target_dist   ||obj_to_target||
    [8]     obj_height        obj_pos.z - table_z
    [9]     gripper_scalar    physical finger opening (larger => more open)
    [10:14] action            raw 4-dim OSC_POSITION action (3 pos deltas + grip)
    [14]    action_norm       ||action[:3]||  (EE motion magnitude)
"""

import numpy as np

FEATURE_NAMES = [
    "eef_to_obj_x", "eef_to_obj_y", "eef_to_obj_z",
    "obj_to_target_x", "obj_to_target_y", "obj_to_target_z",
    "eef_obj_dist",
    "obj_target_dist",
    "obj_height",
    "gripper_scalar",
    "action_0", "action_1", "action_2", "action_3",
    "action_norm",
]
FEATURE_DIM = len(FEATURE_NAMES)  # 15


def feature_index(name):
    """Index of a named feature within a feature vector (raises if unknown)."""
    return FEATURE_NAMES.index(name)


def compute_features(obj_pos, eef_pos, grip_scalar, action, target, table_z):
    """Assemble the (15,) RELATIVE feature vector for one timestep.

    Args:
        obj_pos (3,), eef_pos (3,): world positions (from replay).
        grip_scalar (float): physical gripper opening (name_lookup.gripper_scalar).
        action (4,): the recorded OSC_POSITION action at this step.
        target (3,): the can's goal-bin target placement (per-episode constant).
        table_z (float): table/bin surface height (per-episode constant).

    Returns:
        feat (float32, 15), abs_debug (dict of absolute coords — NOT model input)
    """
    obj_pos = np.asarray(obj_pos, dtype=np.float64)
    eef_pos = np.asarray(eef_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64).ravel()
    if action.shape[0] != 4:
        raise SystemExit(f"expected a 4-dim action, got shape {action.shape}. "
                         "Feature extraction assumes OSC_POSITION (3 pos + gripper).")

    eef_to_obj = obj_pos - eef_pos
    obj_to_target = target - obj_pos
    eef_obj_dist = float(np.linalg.norm(eef_to_obj))
    obj_target_dist = float(np.linalg.norm(obj_to_target))
    obj_height = float(obj_pos[2] - table_z)
    action_norm = float(np.linalg.norm(action[:3]))

    feat = np.array(
        [
            eef_to_obj[0], eef_to_obj[1], eef_to_obj[2],
            obj_to_target[0], obj_to_target[1], obj_to_target[2],
            eef_obj_dist,
            obj_target_dist,
            obj_height,
            float(grip_scalar),
            action[0], action[1], action[2], action[3],
            action_norm,
        ],
        dtype=np.float32,
    )
    assert feat.shape[0] == FEATURE_DIM, (feat.shape, FEATURE_DIM)

    abs_debug = {
        "obj_pos": obj_pos.astype(np.float32),
        "eef_pos": eef_pos.astype(np.float32),
        "target": target.astype(np.float32),
    }
    return feat, abs_debug
