#!/usr/bin/env python
"""
replay/replay_states.py
=======================

Rebuild the PickPlaceCan env from a demo's stored ``env_info`` and replay its
saved MuJoCo states to recover low-dim observations (the demos contain no obs).

Replay recipe (verified against robosuite 1.5
``scripts/playback_demonstrations_from_hdf5.py:67-105``)::

    env.reset()
    xml = env.edit_model_xml(model_xml)      # bake per-demo asset paths
    env.reset_from_xml_string(xml)
    env.sim.reset()
    for state in states:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        obs = env._get_observations(force_update=True)   # fresh low-dim obs

We run HEADLESS (no renderer, no offscreen renderer) — feature extraction needs
only sim state, so this is fast and display-free.

*** Building/stepping an env DOES execute robosuite. This module is imported and
    run by the user LATER (build_feature_bank.py / m1_pipeline.py); it is not run
    at authoring time. ***
"""

import numpy as np

from replay.hdf5_utils import make_kwargs_from_env_info, assert_action_dim_4


def build_env(env_info, render=False):
    """Create the (unwrapped) PickPlaceCan env from stored metadata.

    Mirrors the collector / playback script's construction, but headless by
    default. Fails loudly if the resulting action space is not 4-dim, matching
    collect_pickplace_can.py's guarantee.
    """
    import robosuite  # local import: authoring must not import robosuite

    assert_action_dim_4(env_info)
    make_kwargs, control_freq = make_kwargs_from_env_info(env_info)

    env = robosuite.make(
        **make_kwargs,
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        reward_shaping=True,
        control_freq=control_freq if control_freq is not None else 20,
    )

    # HARD action-dim check (env.action_spec is valid right after make()).
    action_dim = int(env.action_spec[0].shape[0])
    if action_dim != 4:
        env.close()
        raise SystemExit(
            f"rebuilt env has action_dim={action_dim}, expected 4 (OSC_POSITION). "
            "The stored controller_configs did not reproduce the 4-dim action space."
        )
    return env


class DemoReplayer:
    """Holds one env and replays many demos through it.

    The env is rebuilt only when the ``env_name`` changes (all PickPlaceCan demos
    share it), so a whole dataset replays through a single simulator instance.
    Use as a context manager to guarantee ``env.close()``.
    """

    def __init__(self, render=False):
        self.render = render
        self.env = None
        self._env_name = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def ensure_env(self, env_info):
        """(Re)build the env if needed; return it."""
        name = env_info.get("env_name")
        if self.env is None or name != self._env_name:
            self.close()
            self.env = build_env(env_info, render=self.render)
            self._env_name = name
        return self.env

    def replay(self, env_info, model_xml, states):
        """Replay one demo; yield (t, obs_dict) for every stored state.

        obs_dict is the low-dim observation after forcing that state. Callers read
        can/eef/gripper from it via replay/name_lookup.py.
        """
        env = self.ensure_env(env_info)

        env.reset()
        # edit_model_xml rewrites asset paths etc. for THIS machine before load.
        # TODO(verify): playback uses env.edit_model_xml; older robosuite exposed
        # robosuite.utils.mjcf_utils.postprocess_model_xml instead.
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()

        states = np.asarray(states)
        for t in range(states.shape[0]):
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()
            obs = env._get_observations(force_update=True)
            yield t, obs

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
            self._env_name = None
