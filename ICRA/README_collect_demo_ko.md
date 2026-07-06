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

각 `demo_<N>/demo.hdf5` 는 `gather_demonstrations_as_hdf5` 를 그대로 재사용해
기존과 동일한 HDF5 구조(`data/demo_i/{states, actions}`, `data.attrs.env_info`)로
저장되며, 폴더당 **데모 1개**만 들어갑니다. `env_info` JSON 에 `action_dim`,
`arm_controller` 가 기록되어 다운스트림에서 액션 차원을 검증할 수 있습니다.

> ⚠️ `OSC_POSE` 로 수집하면 액션이 **7-dim** 이라 이 프로젝트의 4-dim 파이프라인
> (`feature_bank/build_feature_bank.py` 등)과는 호환되지 않습니다. 파이프라인용
> 데이터는 `--controller OSC_POSITION` 으로 수집하세요. 팔 롤링/자유 카메라는
> 일반 데모 확인·시연용입니다.
