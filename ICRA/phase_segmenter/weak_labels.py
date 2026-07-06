#!/usr/bin/env python
"""
phase_segmenter/weak_labels.py
==============================

Weak, physical-event pseudo-labels used to BOOTSTRAP the posterior. These are NOT
the deployed artifact — the trained network is. They give the posterior a signal
to learn named phases (Approach/Grasp/Lift/Transport/Place) that IGNORE sidetracks.

Event rules operate purely on the RELATIVE features (feature_bank/features.py), so
they are position-invariant. Key design choices (per the M1 spec):
  * "object not held" -> ALWAYS "approach". Sidetracks while not holding are
    absorbed into approach and never split into their own phase.
  * NO monotonic phase-order enforcement. The event definitions already keep
    sidetracks inside approach; we do not force a fixed sequence.
  * "gripper closed" uses the COMMANDED gripper (action[3]) rather than a
    finger-qpos threshold, because the commanded value is unambiguous
    (+1 close / -1 open) and does not depend on exact qpos ranges we could not
    measure without running robosuite. TODO(verify): confirm action[3] sign.

Output: raw per-frame labels + a lightly smoothed copy (single-frame spikes removed
by a small majority/mode filter). Both are kept.
"""

import numpy as np

from m1_config import get_phase_names
from feature_bank.features import feature_index


def _mode_filter(labels, width):
    """Majority (mode) filter over a categorical label sequence.

    Removes single-frame spikes without needing scipy. ``width`` should be odd;
    even values are bumped up by one. width<=1 is a no-op.
    """
    labels = np.asarray(labels)
    if width is None or width <= 1:
        return labels.copy()
    if width % 2 == 0:
        width += 1
    half = width // 2
    T = labels.shape[0]
    out = labels.copy()
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        window = labels[lo:hi]
        vals, counts = np.unique(window, return_counts=True)
        out[t] = vals[np.argmax(counts)]
    return out


def weak_labels(features, feature_names, cfg):
    """Assign a phase id per timestep from physical events.

    Args:
        features (T, D) float: RELATIVE features (feature_bank layout).
        feature_names (list[str]): names matching the feature columns.
        cfg (dict): the loaded config (reads phases + weak_labels thresholds).

    Returns:
        dict(raw=(T,) int64, smoothed=(T,) int64, phase_names=list[str])
    """
    features = np.asarray(features, dtype=np.float64)
    T = features.shape[0]

    names = get_phase_names(cfg)
    pid = {n: i for i, n in enumerate(names)}
    # Phases we rely on; any missing one falls back to "approach" (0).
    P_APPROACH = pid.get("approach", 0)
    P_GRASP = pid.get("grasp", P_APPROACH)
    P_LIFT = pid.get("lift", P_APPROACH)
    P_TRANSPORT = pid.get("transport", P_APPROACH)
    P_PLACE = pid.get("place", P_APPROACH)

    wl = cfg.get("weak_labels", {})
    held_dist = float(wl.get("held_dist", 0.06))
    gripper_closed_thr = float(wl.get("gripper_closed", 0.0))
    lift_height = float(wl.get("lift_height", 0.04))
    target_xy_thr = float(wl.get("target_xy", 0.05))
    smooth_width = int(wl.get("smooth_width", 3))

    # --- pull the columns we need by NAME (robust to layout changes) ---
    def col(name):
        return features[:, feature_index(name)]

    eef_obj_dist = col("eef_obj_dist")
    obj_height = col("obj_height")
    grip_cmd = col("action_3")  # commanded gripper: >thr => closing
    o2t_x = col("obj_to_target_x")
    o2t_y = col("obj_to_target_y")
    obj_target_xy = np.sqrt(o2t_x ** 2 + o2t_y ** 2)

    # --- per-frame boolean events ---
    gripper_closed = grip_cmd > gripper_closed_thr
    held = (eef_obj_dist < held_dist) & gripper_closed
    lifted = obj_height > lift_height
    at_target_xy = obj_target_xy < target_xy_thr

    # "moving toward target": xy distance decreasing vs previous frame.
    dxy = np.zeros(T)
    if T > 1:
        dxy[1:] = np.diff(obj_target_xy)
    moving_toward = dxy < -1e-4

    # "settled/low": object lowered back near the bin (used for place).
    settled_low = obj_height < lift_height

    raw = np.full(T, P_APPROACH, dtype=np.int64)
    for t in range(T):
        if at_target_xy[t] and ((not gripper_closed[t]) or settled_low[t]):
            raw[t] = P_PLACE
        elif held[t] and lifted[t]:
            raw[t] = P_TRANSPORT if moving_toward[t] else P_LIFT
        elif held[t]:
            raw[t] = P_GRASP
        else:
            raw[t] = P_APPROACH  # not held -> default; sidetracks absorbed here

    smoothed = _mode_filter(raw, smooth_width)
    return {"raw": raw, "smoothed": smoothed, "phase_names": names}
