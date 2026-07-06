#!/usr/bin/env python
"""
collect_pickplace_can.py
========================

Self-contained keyboard-teleop demo collector for **PickPlaceCan + Panda** in
robosuite 1.5, wired so the collected demos use an **OSC_POSITION** arm
controller -> a **4-dim** action (3 end-effector position deltas + 1 gripper).

Why 4-dim matters here
----------------------
Downstream modules in this project (forward-consequence model + Bradley-Terry
reward net) assume a 4-dim OSC_POSITION action space. robosuite's BASIC
composite controller defaults the Panda arm to **OSC_POSE** (6-DOF pose ->
7-dim action), so we explicitly override the arm part to OSC_POSITION and then
FAIL LOUDLY if the resulting action dim is not 4.

The hard part (verified against robosuite 1.5.2 source)
-------------------------------------------------------
robosuite's stock keyboard teleop (`robosuite.devices.Keyboard` +
`Device.input2action`) is hard-wired for OSC_POSE: `Device.input2action`
(devices/device.py) contains

    assert controller.name in ["OSC_POSE", "JOINT_POSITION"], ...

and always builds a 6-DOF arm delta `concat(dpos, drotation)`. With an
OSC_POSITION arm this both trips the assert and feeds a 6-vector to a 3-DOF
controller. So a plain Keyboard device cannot drive an OSC_POSITION env.

We therefore provide `OSCPositionKeyboard`, a small subclass that overrides
`input2action` to emit a **3-DOF position-only** arm delta (dropping rotation)
plus the gripper command, which `Robot.create_action_vector` then assembles
into the 4-dim env action. Everything else (the collection loop, the hdf5
writer, the DataCollectionWrapper) is reused straight from robosuite.

We do NOT modify anything under the robosuite install; all customization lives
in this file.

Usage (activate the conda `robosuite` env first)::

    python collect_pickplace_can.py --hide-other-targets

Quit: press ``q`` in the keyboard listener to end/save the current episode;
press ``Ctrl+C`` in the terminal to stop the program.
"""

import argparse
import copy
import json
import os
import shutil
import time
from glob import glob

import numpy as np

import robosuite as suite

# 1.5 API: load_composite_controller_config replaces the deprecated
# load_controller_config. Import path verified in robosuite/controllers/__init__.py.
from robosuite.controllers import load_composite_controller_config

# The base Wrapper (for our best-effort target-hiding wrapper) and the built-in
# collection wrappers we reuse verbatim.
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper, Wrapper

# Stock keyboard device (used as-is for OSC_POSE; subclassed for OSC_POSITION).
from robosuite.devices import Keyboard

# Reuse robosuite's own collection loop + hdf5 gatherer rather than reimplement.
# These are module-level functions; importing the module does NOT run its
# __main__ block. Signatures (robosuite 1.5.2, verified in source):
#   collect_human_trajectory(env, device, arm, max_fr, goal_update_mode)
#   gather_demonstrations_as_hdf5(directory, out_dir, env_info)
from robosuite.scripts.collect_human_demonstrations import (
    collect_human_trajectory,
    gather_demonstrations_as_hdf5,
)


# ---------------------------------------------------------------------------
# Custom keyboard device: position-only (OSC_POSITION) teleoperation
# ---------------------------------------------------------------------------
class OSCPositionKeyboard(Keyboard):
    """Keyboard device that emits a 3-DOF position-only arm delta.

    This bypasses the stock ``Device.input2action`` (which asserts the arm
    controller is OSC_POSE/JOINT_POSITION and builds a 6-DOF delta) so we can
    drive an OSC_POSITION arm. Rotation keys (e/r/y/h/p/o) are intentionally
    ignored in this mode. Everything else (pynput listener, key bindings,
    gripper toggle, reset key) is inherited from Keyboard.
    """

    # Signature must match the stock method: the collection loop calls
    # device.input2action(goal_update_mode=goal_update_mode).
    def input2action(self, mirror_actions=False, goal_update_mode="target"):
        robot = self.env.robots[self.active_robot]
        active_arm = self.active_arm

        state = self.get_controller_state()
        dpos = state["dpos"]
        raw_drotation = state["raw_drotation"]
        grasp = state["grasp"]
        reset = state["reset"]

        # A reset (the 'q' key) is signalled by returning None -> ends episode.
        if reset:
            return None

        # Reproduce the stock device's per-teleop scaling so "feel" matches the
        # OSC_POSE keyboard: dpos is scaled (*75) and clipped to [-1, 1] inside
        # _postprocess_device_outputs. drotation is computed then discarded.
        drotation = raw_drotation[[1, 0, 2]]
        drotation[2] = -drotation[2]
        dpos, drotation = self._postprocess_device_outputs(dpos, drotation)

        # Map gripper toggle state -> +1 (closed) / -1 (open), as the stock device does.
        grasp = 1 if grasp else -1

        ac_dict = {}
        # Fill zero deltas for every arm so create_action_vector has an entry for
        # each controlled arm part; the active arm is overwritten below.
        for arm in robot.arms:
            ctrl = robot.part_controllers[arm]
            # Guard: this device only makes sense for a 3-DOF OSC_POSITION arm.
            assert ctrl.name == "OSC_POSITION", (
                "OSCPositionKeyboard only supports OSC_POSITION arms; got "
                f"'{ctrl.name}' for arm '{arm}'. Use the stock Keyboard for OSC_POSE."
            )
            ac_dict[f"{arm}_delta"] = np.zeros(ctrl.control_dim)  # control_dim == 3
            ac_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)

        # Active arm: 3-DOF position delta only (rotation dropped), clipped to [-1, 1].
        active_ctrl = robot.part_controllers[active_arm]
        ac_dict[f"{active_arm}_delta"] = np.clip(dpos[: active_ctrl.control_dim], -1.0, 1.0)

        gripper = robot.gripper[active_arm]
        if hasattr(gripper, "grasp_qpos"):
            # Some grippers map the discrete grasp signal to a qpos target.
            ac_dict[f"{active_arm}_gripper"] = getattr(gripper, "grasp_qpos")[grasp]
        else:
            ac_dict[f"{active_arm}_gripper"] = np.array([grasp] * gripper.dof)

        return ac_dict


# ---------------------------------------------------------------------------
# Best-effort visual cleanup wrapper for --hide-other-targets
# ---------------------------------------------------------------------------
class HideVisualTargetsWrapper(Wrapper):
    """Sets the alpha of non-kept visual target markers to 0 after each reset.

    In single_object_mode (PickPlaceCan) only the non-can *collision* objects are
    teleported away; the semi-transparent *visual* target markers for milk/bread/
    cereal stay in the goal bin. This wrapper hides those markers. It is purely
    cosmetic and must never crash the collector, so every step is wrapped in
    try/except and simply warns on failure.

    Placed as the OUTERMOST wrapper so its reset() runs after the full reset
    chain (including DataCollectionWrapper's reset_from_xml_string, which
    otherwise restores the XML default alpha).
    """

    def __init__(self, env, keep_substr="can"):
        super().__init__(env)
        self.keep_substr = keep_substr.lower()
        self._warned = False

    def _hide_targets(self):
        try:
            sim = self.env.sim  # proxied down to the base MujocoEnv
            geom_names = []
            # Preferred path: enumerate the env's visual objects (robosuite 1.5).
            try:
                for obj in self.env.visual_objects:
                    if self.keep_substr in obj.name.lower():
                        continue
                    geom_names.extend(list(obj.visual_geoms))
            except Exception:
                # Fallback to known PickPlace visual-marker geom names.
                geom_names = ["VisualMilk_g0", "VisualBread_g0", "VisualCereal_g0"]

            hidden = 0
            for gname in geom_names:
                try:
                    gid = sim.model.geom_name2id(gname)
                    sim.model.geom_rgba[gid, 3] = 0.0
                    hidden += 1
                except Exception as exc:  # unknown geom name -> skip, keep going
                    print(f"[hide-other-targets] could not hide geom '{gname}': {exc}")
            sim.forward()
            if hidden and not self._warned:
                print(f"[hide-other-targets] hid {hidden} non-'{self.keep_substr}' target marker(s).")
                self._warned = True
        except Exception as exc:
            print(f"[hide-other-targets] warning: hiding failed ({exc}); continuing without it.")

    def reset(self):
        ret = self.env.reset()
        self._hide_targets()
        return ret


# ---------------------------------------------------------------------------
# Controller config: force OSC_POSITION on the arm
# ---------------------------------------------------------------------------
def make_arm_controller_config(arm_controller, robot):
    """Return a BASIC composite-controller config with the arm part(s) set to
    ``arm_controller`` (OSC_POSITION by default -> 4-dim action).

    robosuite 1.5's ``load_composite_controller_config`` FLATTENS
    ``body_parts.arms`` into ``body_parts['right'] / ['left']`` (there is no
    'arms' key in the returned dict). We iterate those arm parts and swap the
    type. IMPORTANT: we must also resize output_max/output_min, because the
    BASIC (OSC_POSE) arm keeps length-6 output arrays and robosuite's
    ``nums2array`` does NOT truncate iterables to control_dim -> a shape
    mismatch during action scaling if left at length 6.
    """
    # BASIC composite controller (generic). load_composite_controller_config
    # accepts a composite name ("BASIC") or a .json path -- NOT an arm-part name.
    config = load_composite_controller_config(controller="BASIC", robot=robot)

    if arm_controller == "OSC_POSITION":
        out_max, out_min = [0.05, 0.05, 0.05], [-0.05, -0.05, -0.05]
    elif arm_controller == "OSC_POSE":
        out_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
        out_min = [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]
    else:
        # Unknown arm controller: set the type but leave output arrays alone and
        # warn -- the user is on their own re: dims (this utility targets OSC_*).
        out_max = out_min = None
        print(f"[warn] unrecognized arm controller '{arm_controller}'; not resizing output_max/min.")

    n_overridden = 0
    for part_name, part in config["body_parts"].items():
        # Only touch arm parts (they start out as OSC_POSE in BASIC).
        if isinstance(part, dict) and part.get("type") in ("OSC_POSE", "OSC_POSITION"):
            part["type"] = arm_controller
            if out_max is not None:
                part["output_max"] = list(out_max)
                part["output_min"] = list(out_min)
            n_overridden += 1

    if n_overridden == 0:
        print("[warn] no OSC arm parts found in BASIC config to override.")
    return config


# ---------------------------------------------------------------------------
# Environment construction + hard action-dim check
# ---------------------------------------------------------------------------
def build_env(args, controller_config):
    """Create the (unwrapped) env and verify the action dim before wrapping."""
    # Mirror robosuite's stock collector call. NOTE: the controller kwarg into
    # suite.make is 'controller_configs' (PLURAL). robosuite auto-enables the
    # offscreen renderer for renderer != 'mjviewer' (base.py), so passing
    # has_offscreen_renderer=False is fine for the 'mujoco' (OpenCV) viewer.
    env = suite.make(
        env_name=args.environment,
        robots=args.robot,
        controller_configs=controller_config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=args.control_freq,
    )

    # env.action_dim is None until the first reset(); env.action_spec is derived
    # from robot.action_limits and is valid immediately after make(). Use it.
    action_dim = int(env.action_spec[0].shape[0])
    arm_part = env.robots[0].part_controllers.get(args.arm)
    arm_controller_name = arm_part.name if arm_part is not None else "<unknown>"

    return env, action_dim, arm_controller_name


def assert_action_dim(args, env, action_dim, arm_controller_name):
    """FAIL LOUDLY if OSC_POSITION did not yield a 4-dim action."""
    if args.controller == "OSC_POSITION" and action_dim != 4:
        env.close()
        raise SystemExit(
            "\n=================== ACTION-SPACE CHECK FAILED ===================\n"
            f"Expected a 4-dim action (OSC_POSITION: 3 pos + 1 gripper) but got "
            f"action_dim = {action_dim} (arm controller: {arm_controller_name}).\n"
            "Downstream code assumes 4-dim OSC_POSITION demos, so refusing to "
            "collect. Check that make_arm_controller_config() overrode the arm to "
            "OSC_POSITION and that output_max/output_min were resized to length 3.\n"
            "================================================================="
        )


# ---------------------------------------------------------------------------
# Pretty launch banner
# ---------------------------------------------------------------------------
def print_launch_banner(args, out_dir, action_dim, arm_controller_name):
    controls = [
        "  arrow keys  : move end-effector in x (up/down) and y (left/right)",
        "  . / ;       : move end-effector down / up (z)",
        "  spacebar    : toggle gripper open/close",
        "  q           : end the current episode (성공 시 저장 여부를 물어봅니다)",
        "  (rotation keys e/r/y/h/p/o are IGNORED in OSC_POSITION mode)",
    ]
    print("\n" + "=" * 70)
    print(" robosuite keyboard demo collector : PickPlaceCan + Panda")
    print("=" * 70)
    print(f" output directory : {out_dir}")
    print(f" env / robot      : {args.environment} / {args.robot}")
    if args.random_can:
        print(f" can position     : RANDOMIZED each episode (--random-can)")
    else:
        print(f" can position     : FIXED via seed={args.seed} (same spot every episode)")
    print(f" renderer / camera: {args.renderer} / {args.camera}")
    print(f" controller       : {arm_controller_name}  ->  action_dim = {action_dim}")
    if action_dim == 4:
        print("                    (3 EE position deltas + 1 gripper)  [OK for pipeline]")
    print(f" control_freq     : {args.control_freq} Hz   (loop cap max_fr = {args.max_fr})")
    print(f" pos/rot sens.    : {args.pos_sensitivity} / {args.rot_sensitivity}")
    print("-" * 70)
    print(" keyboard controls:")
    for line in controls:
        print(line)
    print("-" * 70)
    print(" the keyboard listener uses a global pynput hook, so keystrokes are")
    print(" captured even if the render window is not focused. To stop the whole")
    print(" program, press Ctrl+C in this terminal.")
    print(" SINGLE-DEMO MODE: 성공한 에피소드마다 저장 여부를 물어봅니다.")
    print("   y -> 저장 후 종료   /   n -> 리셋 후 재수집   (성공 1개 저장 시 종료)")
    print(f" 저장 위치(승인 시): {os.path.join(out_dir, 'demo.hdf5')}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Single-demo helpers: detect the just-collected episode + its success flag
# ---------------------------------------------------------------------------
def list_episode_dirs(tmp_directory):
    """Return the set of 'ep_*' episode sub-directory names in tmp_directory."""
    if not os.path.isdir(tmp_directory):
        return set()
    return {d for d in os.listdir(tmp_directory) if d.startswith("ep_")}


def newest_episode_dir(tmp_directory, before):
    """Return the full path of the ep_* dir created since the `before` snapshot.

    collect_human_trajectory() creates exactly one new episode directory per
    call (via DataCollectionWrapper._on_first_interaction), so the set
    difference against a pre-call snapshot pins down the just-collected episode.
    Returns None if nothing new was written (e.g. the user quit before taking a
    single step, so no interaction was ever logged).
    """
    new = sorted(list_episode_dirs(tmp_directory) - before)
    if not new:
        return None
    return os.path.join(tmp_directory, new[-1])


def episode_successful(ep_dir):
    """True if any state_*.npz in ep_dir recorded a successful=True flag.

    Mirrors gather_demonstrations_as_hdf5's own success criterion (OR of the
    per-flush `successful` flags) so what we ask about matches what would be
    written to the hdf5.
    """
    ok = False
    for state_file in glob(os.path.join(ep_dir, "state_*.npz")):
        dic = np.load(state_file, allow_pickle=True)
        ok = ok or bool(dic["successful"])
    return ok


def next_demo_dir(root):
    """Return the path to the next unused 'demo_<N>' folder under root.

    Scans root for existing 'demo_<int>' folders and returns 'demo_<max+1>'
    (or 'demo_1' if none exist). The folder is NOT created here so that quitting
    without saving leaves no empty stub behind.
    """
    os.makedirs(root, exist_ok=True)
    used = []
    for name in os.listdir(root):
        if name.startswith("demo_") and name[len("demo_"):].isdigit():
            used.append(int(name[len("demo_"):]))
    n = (max(used) + 1) if used else 1
    return os.path.join(root, f"demo_{n}")


def ask_keep_demo():
    """Prompt (Korean) whether to save the just-collected successful demo.

    Loops until a clear yes/no is given. Returns True to keep+save, False to
    discard and re-collect. Ctrl+D (EOF) is treated as 'no' (re-collect).
    """
    while True:
        try:
            ans = input(">> 데모 성공! 현재 데이터를 저장하겠습니까? [y/n]: ").strip().lower()
        except EOFError:
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("   y(저장) 또는 n(리셋 후 재수집) 으로 답해주세요.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Keyboard-teleop demo collection for PickPlaceCan + Panda (OSC_POSITION, 4-dim)."
    )
    parser.add_argument("--directory", type=str, default=os.path.join(".", "demos", "pickplace_can"),
                        help="Root directory to store collected demos.")
    parser.add_argument("--environment", type=str, default="PickPlaceCan",
                        help="robosuite environment name.")
    parser.add_argument("--robot", type=str, default="Panda",
                        help="Robot model to use.")
    parser.add_argument("--device", type=str, default="keyboard",
                        help="Teleop device (only 'keyboard' is wired for OSC_POSITION).")
    parser.add_argument("--renderer", type=str, default="mujoco",
                        help="'mujoco' (OpenCV viewer, supports named cameras) or 'mjviewer'.")
    parser.add_argument("--camera", type=str, default="agentview",
                        help="Camera view to render.")
    parser.add_argument("--controller", type=str, default="OSC_POSITION",
                        help="Arm controller type. Default OSC_POSITION enforces a 4-dim action.")
    parser.add_argument("--control-freq", type=int, default=20,
                        help="Environment control frequency (Hz).")
    parser.add_argument("--max-fr", type=int, default=20,
                        help="Cap the collection loop to this many frames/sec.")
    parser.add_argument("--pos-sensitivity", type=float, default=0.7,
                        help="Position input sensitivity.")
    parser.add_argument("--rot-sensitivity", type=float, default=0.7,
                        help="Rotation input sensitivity (unused in OSC_POSITION mode).")
    parser.add_argument("--arm", type=str, default="right",
                        help="Which arm to control (Panda has a single 'right' arm).")
    parser.add_argument("--goal-update-mode", type=str, default="target", choices=["target", "achieved"],
                        help="Passed through to device.input2action.")
    parser.add_argument("--hide-other-targets", action="store_true",
                        help="Best-effort: hide the milk/bread/cereal visual target markers, keep the can.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Fixed RNG seed re-applied before every episode so the can spawns at the "
                             "SAME position every time. Change the value for a different (still fixed) spot.")
    parser.add_argument("--random-can", action="store_true",
                        help="Disable the fixed-seed behaviour and let the can position be randomized "
                             "each episode (the stock robosuite behaviour).")
    args = parser.parse_args()

    if args.device != "keyboard":
        raise SystemExit(
            f"--device '{args.device}' is not supported by this utility. Only 'keyboard' is wired "
            "for OSC_POSITION teleop. Extend OSCPositionKeyboard for other devices if needed."
        )

    # --- controller config: force OSC_POSITION on the arm ---
    controller_config = make_arm_controller_config(args.controller, args.robot)

    # --- build env + HARD action-dim check (before any wrapping / collection) ---
    env, action_dim, arm_controller_name = build_env(args, controller_config)
    assert_action_dim(args, env, action_dim, arm_controller_name)

    # --- output directory: ./demos/pickplace_can/demo_<N>/ ---
    # Named demo_1, demo_2, ... (next unused index). NOT created yet: we only
    # make it when the user accepts a demo, so aborting leaves no empty stub.
    out_dir = next_demo_dir(args.directory)

    # --- env metadata for the hdf5 (so downstream code can verify the action space) ---
    # Mirrors the stock collector's env_info: env_name, robots, controller_configs.
    # We add explicit action-space fields for convenience.
    env_info_dict = {
        "env_name": args.environment,
        "robots": [args.robot],
        "controller_configs": controller_config,
        "action_dim": action_dim,
        "arm_controller": arm_controller_name,
        "control_freq": args.control_freq,
    }
    env_info = json.dumps(env_info_dict)

    # --- wrap: Visualization -> DataCollection -> (optional) HideVisualTargets ---
    # DataCollectionWrapper records the per-episode npz files that the hdf5
    # gatherer consumes. The hide wrapper is outermost so it re-applies after
    # every reset (see class docstring). tmp_directory holds raw episode dumps.
    env = VisualizationWrapper(env)
    tmp_directory = os.path.join("/tmp", "rs_collect_{}".format(str(time.time()).replace(".", "_")))
    env = DataCollectionWrapper(env, tmp_directory)
    if args.hide_other_targets:
        env = HideVisualTargetsWrapper(env, keep_substr="can")

    # --- keyboard device: OSCPositionKeyboard for OSC_POSITION, else stock Keyboard ---
    if args.controller == "OSC_POSITION":
        device = OSCPositionKeyboard(
            env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
    else:
        device = Keyboard(
            env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )

    print_launch_banner(args, out_dir, action_dim, arm_controller_name)

    # --- single-demo collection loop ---
    # We only want ONE demo. Each iteration runs a single episode (reset ->
    # teleop -> success-hold -> env.close()). After it ends we look at just that
    # episode's npz success flag:
    #   * not successful  -> discard the episode dir and silently re-collect.
    #   * successful      -> ask the user whether to save. On 'yes' we gather the
    #                        (only remaining) episode into demo.hdf5 and stop; on
    #                        'no' we delete the episode dir and re-collect.
    # Because we delete every episode we don't keep, tmp_directory holds exactly
    # the accepted episode when gather runs, so demo.hdf5 ends up with 1 demo.
    # Stop at any time with Ctrl+C.
    saved = False
    try:
        while not saved:
            # Fixed can position: re-seed the base env's RNG to the SAME value
            # before every episode. The placement sampler draws the can pose from
            # env.rng on each reset (rebuilt via _load_model -> UniformRandomSampler
            # with rng=self.rng), so an identical seed => identical layout every
            # time. Skip this to restore stock randomization (--random-can).
            if not args.random_can:
                env.unwrapped.rng = np.random.default_rng(args.seed)

            before = list_episode_dirs(tmp_directory)
            collect_human_trajectory(env, device, args.arm, args.max_fr, args.goal_update_mode)

            ep_dir = newest_episode_dir(tmp_directory, before)
            if ep_dir is None:
                print("[info] 이번 에피소드에서 기록된 데이터가 없습니다. 다시 수집합니다.\n")
                continue

            if not episode_successful(ep_dir):
                print("[info] 데모가 성공으로 감지되지 않았습니다. 리셋 후 다시 수집합니다.\n")
                shutil.rmtree(ep_dir, ignore_errors=True)
                continue

            if ask_keep_demo():
                os.makedirs(out_dir, exist_ok=True)  # create demo_<N>/ only now
                gather_demonstrations_as_hdf5(tmp_directory, out_dir, env_info)
                print(f"\n[saved] {os.path.join(out_dir, 'demo.hdf5')}")
                saved = True
            else:
                print("[info] 저장하지 않고 리셋 후 다시 수집합니다.\n")
                shutil.rmtree(ep_dir, ignore_errors=True)
    except KeyboardInterrupt:
        print("\n[done] 사용자에 의해 중단되었습니다 (저장된 데모 없음).")
        return

    print("[done] 데모 1개 수집 완료.")


if __name__ == "__main__":
    main()
