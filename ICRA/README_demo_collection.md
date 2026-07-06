# PickPlaceCan demo collection (robosuite 1.5, Panda, OSC_POSITION)

A small, self-contained keyboard-teleop demo collector for **PickPlaceCan +
Panda**, plus a tool to inspect the resulting HDF5. Built so the collected
demos use an **OSC_POSITION** arm controller — a **4-dim** action (3 EE position
deltas + 1 gripper) — which is what this project's downstream forward-consequence
model and Bradley-Terry reward net expect.

## Files

| File | What it does |
|------|--------------|
| `collect_pickplace_can.py` | Launches PickPlaceCan + Panda with keyboard teleop, OSC_POSITION enforced, saves demos to `./demos/pickplace_can/<timestamp>/demo.hdf5`. |
| `inspect_demo.py` | Prints the structure of a `demo.hdf5` (keys, attrs, per-demo `states`/`actions` shapes → confirm action dim == 4). |
| `run_collect_can.sh` | One-liner wrapper: `python collect_pickplace_can.py --hide-other-targets`. |

## Prerequisites

- The environment with **robosuite 1.5** installed. In this project that is the
  conda env named `robosuite`:

  ```bash
  conda activate robosuite
  ```

  (A plain `python` in a fresh shell may resolve to a broken partial install —
  always activate the env first.)

- **`pynput`** — the keyboard backend robosuite's `Keyboard` device uses. It
  installs a *global* OS-level key listener (already a robosuite dependency).

## Run the collector

```bash
conda activate robosuite
bash run_collect_can.sh
# equivalently:
python collect_pickplace_can.py --hide-other-targets
```

On launch it prints the output directory, the **action dim + controller in
use**, the keyboard controls, how to quit, and the `demo.hdf5` path.

### Keyboard controls

| Key | Action |
|-----|--------|
| **arrow up / down** | move EE −x / +x |
| **arrow left / right** | move EE −y / +y |
| **`.` / `;`** | move EE down / up (−z / +z) |
| **spacebar** | toggle gripper open / close |
| **`q`** | end the current episode (saved if the task succeeded) |
| e / r / y / h / p / o | rotation — **ignored in OSC_POSITION mode** |

- **Focus / backend:** the `Keyboard` device uses a **global `pynput` hook**, so
  keystrokes are captured even when the render window is not focused. On some
  Linux/X11 setups you still want the viewer window up and may need input
  permissions for `pynput`. If keys do nothing, click the viewer window once and
  make sure you're on X11 (pynput support under Wayland can be flaky).
- **How an episode ends:** it auto-saves when the can is successfully placed and
  the success state is held for ~10 steps, **or** when you press `q`. Only
  *successful* episodes are written into `demo.hdf5` (robosuite drops
  unsuccessful ones).
- **How to quit the program:** press **`Ctrl+C`** in the terminal. `demo.hdf5`
  is (re)written after every episode, so stopping is safe.

### CLI options (defaults)

| Option | Default | Notes |
|--------|---------|-------|
| `--directory` | `./demos/pickplace_can` | root output dir |
| `--environment` | `PickPlaceCan` | |
| `--robot` | `Panda` | |
| `--device` | `keyboard` | only keyboard is wired for OSC_POSITION |
| `--renderer` | `mujoco` | OpenCV viewer (supports named cameras); `mjviewer` also available |
| `--camera` | `agentview` | |
| `--controller` | `OSC_POSITION` | **default enforces the 4-dim action**; `OSC_POSE` → 7-dim |
| `--control-freq` | `20` | env control frequency (Hz) |
| `--max-fr` | `20` | caps the teleop loop frame rate |
| `--pos-sensitivity` | `0.7` | |
| `--rot-sensitivity` | `0.7` | unused in OSC_POSITION mode |
| `--hide-other-targets` | off | best-effort: hide milk/bread/cereal markers, keep the can |

## Inspect a collected file

```bash
python inspect_demo.py ./demos/pickplace_can/<timestamp>/demo.hdf5
```

Prints top-level keys, `data` group attributes (including a parsed `env_info`),
the demo group list, and each demo's `states` / `actions` shapes. The
`actions` second dimension should read **4**.

## Expected folder structure

```
demos/
└── pickplace_can/
    └── <t1>_<t2>/            # timestamped run
        └── demo.hdf5
```

Inside `demo.hdf5`:

```
data                                  (group)
├── attrs: date, time, repository_version, env, env_info(JSON), ...
├── demo_1                            (group)
│   ├── attrs: model_file (task MJCF XML string)
│   ├── states   dataset  (T, state_dim)
│   └── actions  dataset  (T, 4)      # 3 EE pos deltas + 1 gripper
├── demo_2
│   └── ...
└── ...
```

## Why OSC_POSITION / 4-dim matters (and how it's enforced)

The BASIC composite controller defaults the Panda arm to **OSC_POSE** → a
6-DOF pose command → a **7-dim** action (6 + 1 gripper). This project's
downstream models assume a **4-dim OSC_POSITION** action (3 EE position deltas +
1 gripper). Mismatched dims would silently corrupt training data, so:

1. `collect_pickplace_can.py` loads the BASIC config via
   `load_composite_controller_config("BASIC")` and overrides the arm part(s) to
   `OSC_POSITION` (also resizing `output_max`/`output_min` to length 3 — required
   because robosuite doesn't truncate those arrays automatically).
2. Right after `suite.make(...)` it checks `env.action_spec` and **fails loudly**
   if the action dim isn't 4.
3. The controller config + action-space info is written into the HDF5
   `data.attrs["env_info"]` (JSON) so downstream code can verify it.
4. robosuite's stock keyboard teleop is hard-wired for OSC_POSE (it asserts the
   arm is `OSC_POSE`/`JOINT_POSITION` and emits a 6-DOF delta). We ship a tiny
   `OSCPositionKeyboard` subclass that emits a **3-DOF position-only** delta so
   keyboard teleop works with the 4-dim OSC_POSITION action space. Rotation keys
   are inert in this mode.

## A note on the scene

`PickPlaceCan` is a **single-object** task — you only pick and place the can.
The environment is built on the **bin arena**, which still renders **multiple bin
compartments** (the goal bin has four quadrants). By default, semi-transparent
**visual target markers** for milk / bread / cereal also remain in the goal bin
even though those objects aren't part of the task. `--hide-other-targets` sets
those markers' alpha to 0 (keeping the can marker). It's best-effort and cosmetic:
if the marker names differ in a future robosuite version it just prints a warning
and continues — it never crashes the collector. The bin compartments themselves
are part of the arena and are not hidden.
