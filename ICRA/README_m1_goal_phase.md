# Module 1 — Goal Inference + Learned Phase Segmenter (H-AIRL-inspired posterior)

Module 1 of the proposal. It turns collected **PickPlaceCan + Panda** demos
(OSC_POSITION, 4-dim) into the two things Module 2's forward-consequence model
`F_eta(s, a, z_t, g)` needs:

- **`g` (main goal)** — *extracted, not inferred*. Taken from the environment's
  target placement for the can (`env.target_bin_placements[object_id]`), with the
  demo's final can position as a fallback. **No network infers `g`.**
- **`z_t` (phase / subgoal)** — a **learned** neural posterior
  `q_omega(z_t | feature-history)` (a BiGRU) that labels each timestep with a phase
  (Approach / Grasp / Lift / Transport / Place, optional Stabilize).

> **This is an *H-AIRL-inspired posterior segmenter*, NOT H-AIRL.** See the
> "What we borrow vs. drop" section below.

---

## What we borrow from H-AIRL vs. what we drop (and why)

**Borrowed**
- The **per-timestep latent-option posterior** `q_omega(z_t | x_{0:t})` — a network
  that infers a phase/option per timestep from a feature history.
- The **directed-information flavor** — an *optional* aux loss (`alpha_ds`) that makes
  `z_t` predictive of the trajectory (next-action / next-consequence reconstruction).

**Deliberately dropped** (they conflict with this project's *better-than-demonstrator*,
**non-imitation** design — the demos are intentionally suboptimal):
- the **AIRL discriminator** / imitation reward,
- the **hierarchical RL policies** (`pi_theta`, `pi_phi`),
- **adversarial training**, and
- the **full EM loop**.

So we keep only the posterior (+ optional DI aux). We do **not** learn a reward or a
policy here.

## Weak labels are a BOOTSTRAP, not the deployed rule

To get *named* phases that ignore sidetracks, the posterior is bootstrapped with
**weak physical-event pseudo-labels** computed from **RELATIVE** features:

| phase | rough event rule (on relative features) |
|-------|------------------------------------------|
| approach  | object **not held** → DEFAULT (absorbs all sidetracks) |
| grasp     | gripper closed **and** object held, not yet lifted |
| lift      | held **and** rising / above table, still far from target (xy) |
| transport | held **and** moving toward target (xy shrinking), not yet at target |
| place     | object near target (xy) **and** gripper opening / settled |

- Sidetracks while **not holding** stay `approach` (never split into their own phase).
- **No monotonic phase-order** is enforced; the event rules already contain sidetracks.
- "gripper closed" uses the **commanded** gripper `action[3]` (robust) rather than a
  finger-qpos threshold.

**Framing:** these labels are only a *bootstrap signal*. The **deployed artifact is
the trained network**, which generalizes across can positions because it consumes
**RELATIVE, position-invariant** features and is trained on **varied-position** demos.
It is learning-based, **not** a fixed rule table at inference. Raising `alpha_ds` and
lowering `w_ce` (config) moves it toward a more H-AIRL-like, less-supervised posterior
— exposed as a research ablation.

---

## Dependencies

The `robosuite` conda env already has `robosuite, h5py, numpy, scipy, tqdm`. Install
the rest **before running**:

```bash
conda activate robosuite
pip install torch pyyaml matplotlib scikit-learn
```

- **torch** (required) — the posterior.
- **pyyaml** (required) — reads the YAML config.
- **matplotlib** (optional) — `visualize.py` plots; CSV is written even without it.
- **scikit-learn** (optional) — macro-F1 in `train.py`; falls back to accuracy.

## Run order

```bash
conda activate robosuite

# 1) Collect ~20-30 SUCCESSFUL demos with VARIED can positions.
#    (Module 1 relies on position variety for generalization.)
python collect_pickplace_can.py --random-can           # repeat until enough demos
#    -> demos accumulate under ./demos/pickplace_can/<run>/demo.hdf5

# 2) Build the RELATIVE feature bank (replays states, extracts features + goal g).
python feature_bank/build_feature_bank.py --config configs/m1_goal_phase_pickplace_can.yaml
#    -> ./data/processed/<run>__<demo>.npz

# 3) Train the posterior (weak-label CE primary; alpha_ds=0 by default).
python phase_segmenter/train.py --config configs/m1_goal_phase_pickplace_can.yaml
#    -> ./checkpoints/phase_segmenter/{norm_stats.npz, best.pt}

# 4) End-to-end: produce (g, z_t) per demo for Module 2.
python m1_pipeline.py --config configs/m1_goal_phase_pickplace_can.yaml
#    -> ./data/processed/<run>__<demo>_m1.npz   (features, z_t, g, ...)

# 5) (optional) Sanity plots: weak labels vs learned prediction.
python phase_segmenter/visualize.py --config configs/m1_goal_phase_pickplace_can.yaml
#    -> ./data/reports/<...>_timeline.{png,csv}
```

> `m1_pipeline.py` will build the feature bank automatically if step 2 was skipped,
> and — if no checkpoint exists yet — will fall back to the smoothed **weak labels**
> for `z_t` (clearly flagged). Train the posterior (step 3) for the deployable,
> learned `z_t`.

## File map

```
configs/m1_goal_phase_pickplace_can.yaml   all knobs (paths, phases, thresholds, model, loss)
m1_config.py                               project-root path + YAML load + phase-name helpers
replay/
  hdf5_utils.py     read demo.hdf5; filter env_info -> make() kwargs
  replay_states.py  rebuild env headless; replay saved MuJoCo states -> obs
  name_lookup.py    robust can/eef/gripper/target/table accessors (obs-first, sim-fallback)
feature_bank/
  features.py           RELATIVE feature vector (D=15) + names
  build_feature_bank.py CLI: demos -> per-demo feature .npz (+ extracted g)
goal/
  extract_goal.py   g = target_bin_placements[can] (fallback: final can pos); NOT a net
phase_segmenter/
  weak_labels.py    physical-event pseudo-labels (bootstrap); sidetracks -> approach
  dataset.py        k-windows, split BY DEMO, train-only norm stats
  posterior.py      q_omega: BiGRU + MLP -> (B,L,K); causal mode; optional DI aux head
  losses.py         CE(weak) + temporal smoothness + optional DI-aux MSE
  train.py          CLI: train/val by demo; save norm_stats + best.pt; acc + macro-F1
  infer.py          load checkpoint -> segment(features) -> z_t   (reusable for M5)
  visualize.py      per-demo timeline PNG + CSV (weak vs learned)
m1_pipeline.py                              CLI: demos -> per-demo (g, z_t) for Module 2
data/{processed,reports}/                   outputs
checkpoints/phase_segmenter/                norm_stats.npz + best.pt
```

## Data contract (what a demo actually stores)

Each `demo.hdf5` (from `collect_pickplace_can.py`) stores **only**
`data/demo_*/states` (MuJoCo states), `data/demo_*/actions` (4-dim), each demo's
`model_file` (MJCF XML), and `data.attrs["env_info"]` (JSON). **There is no
observation dataset**, so Module 1 rebuilds the env from `env_info` and **replays the
states** to recover eef/can/target positions (see `replay/`). `action_dim == 4`
(OSC_POSITION) is asserted end-to-end.

## Assumptions to double-check on the first real run

These are annotated `# TODO(verify)` in the code (we authored against robosuite 1.5
**source**, without executing it):

- Obs keys after `env._get_observations(force_update=True)` at a forced state:
  `robot0_eef_pos`, `robot0_gripper_qpos`, `Can_pos` (and object body key casing
  `obj_body_id["Can"]` vs `object_to_id["can"]`).
- `env.edit_model_xml(model_xml)` is the correct XML post-processor in this build
  (the playback script uses it) vs. older `postprocess_model_xml`.
- `env.target_bin_placements[object_id]` is populated after reset (set in
  `_setup_references`).
- Gripper scalar sign/scale from `robot0_gripper_qpos` (2 finger joints → 1 opening
  scalar). Weak labels avoid this by using the commanded gripper `action[3]`.

## Limitations

- Weak labels are heuristic (a bootstrap); phase boundaries are approximate. The
  trained posterior smooths/generalizes them but is only as good as the label events.
- A **bidirectional** posterior (training default) is for **offline** demo
  segmentation. For genuine **online** M5 use, train with `model.bidirectional: false`
  (a causal GRU); `infer.PhaseSegmenter.is_causal` reports which you have.
- `g` assumes the single-object PickPlaceCan target; multi-object tasks need a
  different goal extractor.
- Nothing here trains a policy or a reward — that is Modules 2+.
```
