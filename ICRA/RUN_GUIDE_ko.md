# 실행 가이드 (데모 수집 → Module 1 goal/phase)

이 문서는 지금까지 만든 코드로 **데모를 수집하고 Module 1(`g`, `z_t`)까지 뽑아내는 전체 실행 방법**을 한국어로 정리한 것입니다. 모든 명령은 **프로젝트 루트**(`/home/jaemin/Desktop/ICRA_뇨롱이`)에서, **conda `robosuite` 환경**을 켠 상태로 실행합니다.

> ⚠️ 지금 디스크 상태는 "처음부터" 입니다 — `demos/pickplace_can`, `data/processed`, `checkpoints/phase_segmenter` 모두 비어 있어(`.gitkeep`만) **1단계 데모 수집이 반드시 먼저** 필요합니다.

---

## 0. 전체 흐름 한눈에

```
[1] 데모 수집        collect_pickplace_can.py --random-can   (GUI+키보드, 1회=성공데모 1개 → 20~30회 반복)
        │            → demos/pickplace_can/demo_1/demo.hdf5, demo_2/…
        ▼
[2] 피처뱅크 빌드    feature_bank/build_feature_bank.py       (robosuite 재생, headless)
        │            → data/processed/<run>__<demo>.npz  (features, actions, g …)
        ▼
[3] posterior 학습   phase_segmenter/train.py                 (torch)
        │            → checkpoints/phase_segmenter/{norm_stats.npz, best.pt}
        ▼
[4] (g, z_t) 생성    m1_pipeline.py                           (torch, Module 2 입력)
        │            → data/processed/<run>__<demo>_m1.npz  (features, z_t, g …)
        ▼
[5] (선택) 시각화    phase_segmenter/visualize.py             (matplotlib 선택)
                     → data/reports/<…>_timeline.{png,csv}
```

| 단계 | GUI 필요? | 핵심 의존성 | 실행 위치 |
|------|:---:|------|------|
| 1 수집 | **필요 (X11 디스플레이 + 키보드)** | robosuite, pynput | 로컬 데스크톱 |
| 2 빌드 | 불필요 (headless) | robosuite, **pyyaml** | 어디서든 |
| 3 학습 | 불필요 | **torch**, (sklearn 선택) | 어디서든 |
| 4 파이프라인 | 불필요 | torch, pyyaml | 어디서든 |
| 5 시각화 | 불필요 | (matplotlib 선택) | 어디서든 |

---

## 1. 사전 준비 (한 번만)

```bash
conda activate robosuite      # /usr/bin/python 은 깨진 스텁이라 반드시 이 env 사용
cd /home/jaemin/Desktop/ICRA_뇨롱이
```

**의존성 설치 — 이거 안 하면 2단계부터 전부 실패합니다.** 현재 `robosuite` env에는 `robosuite, h5py, numpy, scipy, tqdm, pynput`만 있고 아래는 **없습니다**:

```bash
pip install torch pyyaml matplotlib scikit-learn
```

- **torch** — (필수) posterior 학습/추론.
- **pyyaml** — (필수) YAML config 로딩. 없으면 `SystemExit: PyYAML is required...`.
- **matplotlib** — (선택) 5단계 PNG. 없으면 CSV만 저장.
- **scikit-learn** — (선택) 학습 시 macro-F1. 없으면 accuracy만.

---

## 2. 단계별 실행

### [1] 데모 수집 — `collect_pickplace_can.py` 또는 `collect_demo.py --pipeline`

**Module 1은 캔 위치가 다양해야** 일반화가 됩니다. 그래서 반드시 `--random-can`으로 수집하세요.

```bash
python collect_pickplace_can.py --random-can
# 마커까지 숨기고 싶으면:
python collect_pickplace_can.py --random-can --hide-other-targets
```

> 🔗 **연결된 대안 (권장): `collect_demo.py --pipeline`**
> 일반화 수집기 `collect_demo.py` 를 `--pipeline` 로 실행하면 이 파이프라인과
> **바로 연결**됩니다. PickPlaceCan + Panda + OSC_POSITION(4-dim)으로 고정되고,
> 출력이 정확히 `./demos/pickplace_can/demo_<N>/demo.hdf5` (= config `demo_root`)로
> 저장되어 아래 [2] 빌드가 그대로 읽습니다. 수집 개수도 물어봐서 그만큼만 모읍니다.
> ```bash
> python collect_demo.py --pipeline --num-demos 20
> ```
> (자세한 사용법: `README_collect_demo_ko.md`. 캔 위치는 기본으로 매번 랜덤입니다.)

> ❌ `./run_collect_can.sh` 나 옵션 없는 기본 실행은 **`--seed 0` 고정**이라 캔이 매번 같은 자리에 나옵니다(위치 다양성 0). Module 1 데모 수집에는 쓰지 마세요. (고정 위치가 필요한 다른 실험용입니다.)

**이 수집기는 "1회 실행 = 성공 데모 1개" 방식입니다.** 한 번 성공시키고 저장하면 프로그램이 종료됩니다. 따라서 **20~30개를 모으려면 20~30번 다시 실행**해야 하고, 실행할 때마다 `demo_1`, `demo_2`, … 로 자동 번호가 매겨집니다.

**조작키**

| 키 | 동작 |
|----|------|
| 방향키 ↑/↓ | EE x축 이동 |
| 방향키 ←/→ | EE y축 이동 |
| `.` / `;` | EE 아래/위 (z축) |
| 스페이스바 | 그리퍼 열기/닫기 토글 |
| `q` | 현재 에피소드 종료 (성공이면 저장할지 물어봄) |
| `Ctrl+C` | 프로그램 전체 종료 |
| e/r/y/h/p/o (회전) | OSC_POSITION 모드에서 **무시됨** |

**저장 흐름**: 캔을 올바른 통에 넣어 성공(자동 10스텝 유지 또는 `q`)하면 터미널에
`>> 데모 성공! 현재 데이터를 저장하겠습니까? [y/n]:` 이 뜹니다.
- `y` → `demos/pickplace_can/demo_<N>/demo.hdf5` 저장 후 **종료**
- `n` → 저장 안 하고 리셋 후 다시 수집 (성공 못 하면 자동으로 재수집)

**저장 위치**: `demos/pickplace_can/demo_<N>/demo.hdf5` (승인할 때만 폴더 생성 → 취소해도 빈 폴더 안 남음).

**주요 옵션(기본값)**: `--directory ./demos/pickplace_can`, `--environment PickPlaceCan`, `--robot Panda`, `--controller OSC_POSITION`(4-dim 강제), `--control-freq 20`, `--max-fr 20`, `--pos-sensitivity 0.7`, `--seed 0`, `--random-can`(off=고정). 플래그는 **하이픈**입니다(`--control-freq`, `--max-fr`, `--pos-sensitivity`).

> 이 단계만 **그래픽 화면(X11 DISPLAY)** 과 키보드가 필요합니다. SSH/headless에서는 안 됩니다. 키가 안 먹으면 뷰어 창을 한 번 클릭하세요(pynput은 Wayland에서 불안정 → X11 권장).

### [2] (선택) 수집한 데모 확인 — `inspect_demo.py`

경로는 **위치 인자**입니다(`--config` 아님).

```bash
python inspect_demo.py ./demos/pickplace_can/demo_1/demo.hdf5
```

`states`/`actions` 모양과 `actions`의 2번째 차원이 **4**인지 확인합니다.

### [3] 피처뱅크 빌드 — `feature_bank/build_feature_bank.py`

데모를 재생(replay)해 상대(위치 불변) 피처와 목표 `g`를 뽑아 `data/processed/`에 저장합니다.

```bash
python feature_bank/build_feature_bank.py --config configs/m1_goal_phase_pickplace_can.yaml
```

- 입력: `demos/pickplace_can/*/demo.hdf5`
- 출력: `data/processed/<run>__<demo>.npz` (features (T,15), actions, g, feature_names …)
- robosuite로 env를 재구성해 상태를 재생하지만 **화면은 안 띄웁니다(headless)**.

### [4] posterior(q_omega) 학습 — `phase_segmenter/train.py`

약라벨(bootstrap)로 BiGRU posterior를 학습합니다.

```bash
python phase_segmenter/train.py --config configs/m1_goal_phase_pickplace_can.yaml
```

- 입력: `data/processed/*.npz` (`_m1.npz` 제외)
- 출력: `checkpoints/phase_segmenter/norm_stats.npz`, `checkpoints/phase_segmenter/best.pt`
- 매 epoch에 train/val loss·accuracy·macro-F1(sklearn 있을 때) 출력.
- **데이터를 데모 단위로 train/val 분할**합니다. 데모가 너무 적으면 val이 비어 accuracy 기준으로 best를 고릅니다 → **20~30개 권장**.

### [5] 엔드투엔드 (g, z_t) 생성 — `m1_pipeline.py`

Module 2가 먹을 데모별 `(features, z_t, g)`를 만듭니다.

```bash
python m1_pipeline.py --config configs/m1_goal_phase_pickplace_can.yaml
```

- 출력: `data/processed/<run>__<demo>_m1.npz` (features, **z_t**, **g**, feature_names, z_source …)
- 피처뱅크가 없으면 **자동으로 먼저 빌드**합니다. `--rebuild` 로 강제 재빌드 가능.
- ✅ 출력 로그에 **`z_source=q_omega`** 가 찍혀야 학습된 posterior를 쓴 것입니다.
  `z_source=weak_fallback` 이면 **3단계(train)를 안 돌린 것** → 약라벨로 대체된 상태이니 train 먼저 하세요.

### [6] (선택) 시각화 — `phase_segmenter/visualize.py`

데모별 타임라인(약라벨 vs 학습 예측)을 그립니다.

```bash
python phase_segmenter/visualize.py --config configs/m1_goal_phase_pickplace_can.yaml
```

- 출력: `data/reports/<…>_timeline.png` + `.csv`
- ⚠️ 이 스크립트는 피처뱅크를 **자동 빌드하지 않습니다**. 2단계(또는 4단계)를 먼저 돌려 `data/processed/`를 채워두세요. matplotlib 없으면 CSV만 나옵니다.

---

## 3. 출력물 위치 정리

| 무엇 | 경로 |
|------|------|
| 수집된 데모 | `demos/pickplace_can/demo_<N>/demo.hdf5` |
| 피처뱅크 | `data/processed/<run>__<demo>.npz` |
| 학습 산출물 | `checkpoints/phase_segmenter/norm_stats.npz`, `best.pt` |
| Module 2 입력 (g, z_t) | `data/processed/<run>__<demo>_m1.npz` |
| 시각화 리포트 | `data/reports/<…>_timeline.{png,csv}` |

경로는 모두 `configs/m1_goal_phase_pickplace_can.yaml`의 `paths:` 에서 바꿀 수 있고, 프로젝트 루트 기준으로 해석됩니다.

---

## 4. 자주 나는 오류 & 해결

| 증상 | 원인 / 해결 |
|------|------|
| `SystemExit: PyYAML is required...` | `pip install pyyaml` (1단계 의존성 설치) |
| `ModuleNotFoundError: No module named 'torch'` | `pip install torch` |
| `python: command not found` / `No module named robosuite` | `conda activate robosuite` 먼저 (`/usr/bin/python` 쓰지 말 것) |
| 수집 시 `ACTION-SPACE CHECK FAILED ... action_dim = N` | `--controller`를 기본 `OSC_POSITION` 그대로 둘 것 (4-dim 강제, 의도된 fail-loud) |
| 수집 시 `--device '...' is not supported` | `--device`는 `keyboard`(기본)만 지원 |
| 뷰어 창이 안 뜨거나 GLFW/DISPLAY 오류 | 수집은 로컬 X11 데스크톱에서. headless(SSH) 불가 |
| 키 입력이 안 먹음 | 뷰어 창 한 번 클릭, X11 세션 확인(Wayland 불안정) |
| 빌드 시 `no demo.hdf5 found under ...` | 1단계 데모 수집을 먼저 (`--random-can`) |
| 학습/시각화 시 `no feature files in data/processed` | 2단계 빌드를 먼저 실행 |
| `m1_pipeline`이 `z_source=weak_fallback` 출력 | 3단계 train을 건너뜀 → `train.py` 먼저 실행 |
| 첫 실행에서 obs 키/`edit_model_xml` 관련 오류 | 코드의 `# TODO(verify)` 지점 확인(robosuite 버전차) — `README_m1_goal_phase.md`의 "assumptions" 참고 |

---

## 5. 설정 바꾸고 싶을 때

`configs/m1_goal_phase_pickplace_can.yaml` 에서 주로 만지는 값:

- `phases.names` / `phases.use_stabilize` — 단계 개수(K). 기본 5단계(approach,grasp,lift,transport,place).
- `weak_labels.*` — 약라벨 임계값(grasp_dist, held_dist, gripper_closed, lift_height, target_xy).
- `model.*` — `input_window(k)`, `hidden_dim`, `num_layers`, `dropout`, `bidirectional`.
  - 온라인/실시간(M5)에 쓰려면 `bidirectional: false`(causal)로 학습.
- `training.*` — `epochs`, `batch_size`, `lr`, `val_split_by_demo`, `device`(auto/cpu/cuda).
- `loss.alpha_ds` — 기본 0(약라벨 CE 위주). 올리고 `w_ce`를 낮추면 H-AIRL에 가까운 덜-지도 posterior(연구용 ablation).

---

## 6. 복사해서 쓰는 전체 순서

```bash
# ── 준비 (한 번만) ─────────────────────────────
conda activate robosuite
cd /home/jaemin/Desktop/ICRA_뇨롱이
pip install torch pyyaml matplotlib scikit-learn

# ── 1) 데모 수집: 성공할 때까지 조작 → y 저장. 20~30개 될 때까지 반복 실행 ──
python collect_pickplace_can.py --random-can
#   (필요하면 --hide-other-targets 추가; 매 실행이 demo_1, demo_2, … 로 쌓임)

# ── (선택) 확인 ────────────────────────────────
python inspect_demo.py ./demos/pickplace_can/demo_1/demo.hdf5

# ── 2~5) Module 1 파이프라인 ───────────────────
python feature_bank/build_feature_bank.py --config configs/m1_goal_phase_pickplace_can.yaml
python phase_segmenter/train.py           --config configs/m1_goal_phase_pickplace_can.yaml
python m1_pipeline.py                     --config configs/m1_goal_phase_pickplace_can.yaml   # z_source=q_omega 확인
python phase_segmenter/visualize.py       --config configs/m1_goal_phase_pickplace_can.yaml   # 선택
```

이후 `data/processed/*_m1.npz` (features, z_t, g)가 Module 2의 `F_eta(s, a, z_t, g)` 입력이 됩니다.

> 더 자세한 설계 설명(무엇을 H-AIRL에서 차용/제거했는지, 약라벨=부트스트랩, 첫 실행 확인사항)은 `README_m1_goal_phase.md` 를 참고하세요.
