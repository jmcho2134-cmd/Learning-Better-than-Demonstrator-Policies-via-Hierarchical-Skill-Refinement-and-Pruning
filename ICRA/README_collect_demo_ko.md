# `collect_demo.py` — 환경/로봇 선택형 데모 수집기 (카메라 이동 + 팔 롤링)

`collect_pickplace_can.py` 를 robosuite 기본 스크립트
(`robosuite/scripts/collect_human_demonstrations.py`,
`robosuite/demos/demo_random_action.py`) 를 참고해 **일반화**한 버전입니다.

## 무엇이 바뀌었나

| 항목 | `collect_pickplace_can.py` (기존) | `collect_demo.py` (신규) |
|------|-----------------------------------|--------------------------|
| 환경 | `PickPlaceCan` 고정 | **터미널 메뉴로 선택** (`choose_environment`) |
| 로봇 | `Panda` 고정 | **터미널 메뉴로 선택** (`choose_robots`, 2팔/휴머노이드 분기 포함) |
| 컨트롤러 | `OSC_POSITION` 4-dim 강제 | 기본 **`OSC_POSE`** (7-dim, **팔 롤링 O**), `OSC_POSITION` 도 선택 가능 |
| 카메라 | 고정 named 카메라 | 기본 **`mjviewer`** — 마우스로 **카메라 자유 이동**(orbit/pan/zoom) |
| 수집 개수 | 무한/고정 | 시작 시 **수집할 데모 개수를 입력** → 그 개수만큼만 수집하고 종료 |
| 저장 | 성공 1개만 저장 후 종료 | 에피소드마다 **저장 여부(y/n)** 를 묻고, 승인한 데모를 **번호별 폴더**에 저장 |

- **팔 롤링**: `OSC_POSE` 모드에서는 기본 `Keyboard` 디바이스의 회전키
  `e/r`(roll), `y/h`(pitch), `p/o`(yaw) 가 살아 있어 엔드이펙터를 회전시킬 수 있습니다.
- **카메라 이동**: `mjviewer` 렌더러(네이티브 MuJoCo 뷰어) 창에서 **마우스 드래그**로
  카메라를 자유롭게 돌리고/이동하고/줌 할 수 있습니다.
- **저장 여부 확인 + 번호별 저장**: 한 에피소드가 끝날 때마다(`q` 로 종료하거나 태스크
  성공) 터미널에서 `>> 이번 데모를 저장하겠습니까? [y/n]:` 라고 **항상** 물어봅니다.
  - `y` → `demos/<환경이름>_<로봇이름>/demo_<N>/demo.hdf5` 로 저장하고 이어서 다음 데모 수집
  - `n` → 저장하지 않고 리셋 후 재수집
  각 데모는 자기 번호 폴더에 **1개씩** 들어갑니다 (`demo_1`, `demo_2`, …).

  > 저장은 환경의 자동 성공 감지(`_check_success`)에 **의존하지 않습니다**. robosuite 기본
  > `gather_demonstrations_as_hdf5` 는 성공으로 latch 된 에피소드만 기록해서, 성공이
  > 감지되지 않으면 y 를 눌러도 아무것도 저장되지 않습니다. 그래서 이 코드는 자체
  > `save_episode_as_hdf5` 로 **사용자가 y 를 누른 데모는 성공 플래그와 무관하게 저장**합니다.
  > (참고용으로 자동 성공 감지 여부는 터미널에 함께 출력됩니다.)

## 실행

```bash
conda activate robosuite

# (1) 완전 대화형: 환경 → 로봇 선택 후, "몇 개 수집할지" 를 터미널에서 물어봄
python collect_demo.py

# (2) 인자로 지정 (개수까지 지정하면 질문 없이 바로 시작)
python collect_demo.py --environment Lift --robots Panda --num-demos 5

# (3) 다운스트림 4-dim 파이프라인용 (팔 롤링 없음, OSC_POSITION)
python collect_demo.py --controller OSC_POSITION

# (4) OpenCV 뷰어 + 고정 카메라를 원하면
python collect_demo.py --renderer mujoco --camera agentview
```

실행 흐름:

```
환경 선택 → 로봇 선택 → "몇 개의 데모를 수집하시겠습니까?" 입력(예: 5)
   → 데모 1/5 수집(창 조작, q 로 종료) → 저장할까요? [y/n]
        y → demo_1 저장, 다음으로     n → 저장 안 하고 리셋 후 재수집
   → ... → 5개 저장되면 자동 종료
```

- 입력한 **개수만큼 저장되면 자동으로 끝납니다** (더 이상 무한 루프 아님).
- `n` 으로 버린 실패 데모는 개수에 **포함되지 않습니다** (좋은 데모 N개가 모일 때까지 진행).
- 중간에 완전히 멈추려면 터미널에서 `Ctrl+C`.

## 키보드 조작

| 키 | 동작 |
|----|------|
| 방향키 ↑/↓ | EE x축 이동 |
| 방향키 ←/→ | EE y축 이동 |
| `.` / `;` | EE 아래로 / 위로 (z) |
| 스페이스바 | 그리퍼 열기/닫기 토글 |
| `e`/`r`, `y`/`h`, `p`/`o` | **roll / pitch / yaw** (OSC_POSE 모드에서만) |
| `q` | 현재 에피소드 종료(성공 시 저장 여부 y/n 질문) |
| 터미널에서 `Ctrl+C` | 프로그램 전체 종료 |

> 키 입력은 전역 `pynput` 훅으로 잡히므로 렌더 창에 포커스가 없어도 동작합니다.
> `mjviewer` 창에서는 마우스로 카메라를 움직이세요.

## 🔗 Module-1 파이프라인은 collect_demo 를 자동으로 따라옵니다

**collect_demo.py 하나로만 수집하면 됩니다.** 별도 옵션 없이 그냥 수집하면, 아래 파이프라인
코드가 **각 데모의 `env_info`를 읽어 환경·객체·액션 차원을 스스로 맞춥니다.** 더 이상 `can`
이나 4-dim에 하드코딩되어 있지 않습니다.

```bash
# 프로젝트 루트(ICRA/)에서 — 원하는 PickPlace 객체를 골라 수집
python collect_demo.py --environment PickPlaceCan   --robots Panda --num-demos 20
python collect_demo.py --environment PickPlaceMilk  --robots Panda --num-demos 20
# 그대로 다음 단계 (config demo_root: ./demos 아래를 재귀 탐색)
python feature_bank/build_feature_bank.py --config configs/m1_goal_phase_pickplace_can.yaml
```

파이프라인이 collect_demo 에 맞춰 자동 처리하는 것:

| 자동으로 따라오는 것 | 어떻게 |
|----------------------|--------|
| **객체 종류** (can/milk/bread/cereal) | `env_info.object_type`(collect_demo 가 기록) → 없으면 env 이름에서 유도. `build_feature_bank` 가 데모별로 결정 |
| **환경/로봇** | `env_info` 의 `env_name`·`robots` 로 env 재구성 (`replay_states`) |
| **액션 차원 4/7** | features 는 위치델타 3 + 그리퍼(action[-1])만 사용 → 4-dim/7-dim 모두 15-dim 피처. `assert_action_dim_4` 하드체크 제거 |
| **저장 경로/여러 환경 혼합** | `find_demo_files` 가 `./demos` 아래 `**/demo.hdf5` 를 **재귀** 탐색. `demos/PickPlaceCan_Panda/…`, `demos/PickPlaceMilk_Panda/…` 를 모두 자동 포함 |

연결 체인:

```
collect_demo.py  (PickPlace 객체 자유 선택, OSC_POSE/OSC_POSITION 무관)
   └─> demos/<Env>_<Robot>/demo_<N>/demo.hdf5   (states, actions(T,4 또는 T,7), env_info{object_type})
          └─> find_demo_files(./demos, 재귀) → load_env_info → read_demo   (replay/hdf5_utils.py)
                 └─> object_type_from_env_info → DemoReplayer.replay → compute_features(15-dim)
                        └─> phase_segmenter/train.py → m1_pipeline.py → (g, z_t)
```

> 참고: 이 파이프라인의 "목표 g / 단계(approach…place)"는 **PickPlace 계열(집기-놓기)**
> 과제를 가정합니다. can/milk/bread/cereal 은 모두 지원되지만, Lift·Door 같은 완전히 다른
> 과제는 목표·단계 의미가 맞지 않아 `build_feature_bank` 에서 객체/타깃을 못 찾고 명확한
> 에러를 냅니다.

## 주요 CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--environment` | (메뉴) | 환경 이름. 생략 시 터미널 메뉴 |
| `--robots` | (메뉴) | 로봇 이름(복수 가능). 생략 시 터미널 메뉴 |
| `--num-demos` | (질문) | 수집할 데모 개수. 생략 시 터미널에서 물어봄. 이 개수만큼 저장되면 종료 |
| `--env-configuration` | (메뉴) | TwoArm 환경의 팔 구성(예: `bimanual`) |
| `--controller` | `OSC_POSE` | `OSC_POSE`(롤링 O, 7-dim) / `OSC_POSITION`(롤링 X, 4-dim) |
| `--renderer` | `mjviewer` | `mjviewer`(카메라 이동 O) / `mujoco`(OpenCV, 고정 카메라) |
| `--camera` | `agentview` | 초기 카메라 |
| `--control-freq` | `20` | 제어 주파수(Hz) |
| `--pos-sensitivity` / `--rot-sensitivity` | `1.0` / `1.0` | 입력 감도 |
| `--directory` | `./demos` | 저장 루트. 실제 경로는 `./demos/<Env>_<Robot>/demo_<N>/demo.hdf5` |

## 저장 형식 / 폴더 구조

```
demos/
└── <환경이름>_<로봇이름>/        # 예: Lift_Panda
    ├── demo_1/
    │   └── demo.hdf5             # 승인한 데모 1개
    ├── demo_2/
    │   └── demo.hdf5
    └── ...
```

각 `demo_<N>/demo.hdf5` 는 로봇수트 기본과 동일한 HDF5 구조
(`data/demo_i/{states, actions}`, `data.attrs.env_info`)로 저장되며, 폴더당
**데모 1개**만 들어갑니다. `env_info` JSON 에 `action_dim`, `arm_controller`,
그리고 PickPlace 환경이면 `object_type`(can/milk/…)까지 기록되어, 다운스트림이
데모마다 환경·객체·액션 차원을 그대로 따라옵니다.

> ✅ `OSC_POSE`(7-dim, 팔 롤링)든 `OSC_POSITION`(4-dim)이든 **둘 다 파이프라인에서
> 학습 가능**합니다. features 가 위치델타 3개 + 그리퍼만 쓰므로 액션 차원과 무관하게
> 15-dim 피처가 됩니다. (다른 과제가 아니라 PickPlace 계열 객체를 쓰는 한.)
