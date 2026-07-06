#!/usr/bin/env python
"""
goal/extract_goal.py
====================

Extract the main goal ``g`` for a demo. This is EXTRACTION, not inference — there
is deliberately NO network here (Proposal 6.1: the main goal is given by the
environment / final placement, only the phase/subgoal z_t is learned).

Primary source: the can's target placement in the goal bin, exposed by the env as
``env.target_bin_placements[object_id]`` (robosuite's built-in desired target;
see pick_place.py:576-587). Fallback: the object's final resting position in the
demo (last replayed can position), used only if the env does not expose the
target for some reason.
"""

import numpy as np

from replay import name_lookup


def extract_goal(env, object_type="can", final_obj_pos=None):
    """Return (g (3,), source_str).

    Args:
        env: a built PickPlaceCan env (after reset, so target_bin_placements is set).
        object_type: key into env.object_to_id (default "can").
        final_obj_pos: optional (3,) final resting object position for fallback.
    """
    # Primary: the environment's target placement (the "desired goal").
    try:
        object_id = name_lookup.get_object_id(env, object_type)
        g = name_lookup.target_placement(env, object_id)
        return np.asarray(g, dtype=np.float32), "target_bin_placements"
    except SystemExit as exc:
        # Fall back only if the target genuinely isn't available.
        if final_obj_pos is not None:
            print(f"[extract_goal] target unavailable ({exc}); "
                  "falling back to final object position.")
            return np.asarray(final_obj_pos, dtype=np.float32), "final_obj_pos"
        raise
