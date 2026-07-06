#!/usr/bin/env python
"""
diagnose_segments.py
====================

"이 데모가 approach -> grasp -> lift -> transport -> place 로 실제로 분절됐나?" 를
**그림 없이 터미널 텍스트로** 바로 보여주는 진단 스크립트.

각 데모마다 최종 단계열(z_t)을 읽어서:
  * 단계 순서 (연속 구간을 합쳐 approach(41) -> grasp(12) -> ... 처럼)
  * 각 단계가 나왔는지 (커버리지)  / place 까지 도달했는지
  * 순서가 대체로 정상(approach->...->place 로 단조 증가)인지
  * 한 줄 ASCII 타임라인 (숫자 = 단계 id)
  * 자동 판정 (OK / 의심 + 이유)
을 출력하고, 마지막에 전체 요약을 냅니다.

읽는 파일 (우선순위):
  1) data/processed/*_m1.npz  의 z_t  (m1_pipeline.py 산출물 = 학습된/최종 분절) [권장]
  2) 없으면 data/processed/*.npz + weak_labels 로 부트스트랩 분절을 대신 보여줌
     (학습 전이라도 대충 확인 가능; "weak(bootstrap)" 라고 표시)

    python diagnose_segments.py --config configs/m1_goal_phase_pickplace_can.yaml

torch 불필요 (저장된 z_t 를 읽기만 함).
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m1_config import load_config, resolve_path, get_phase_names


# 표준 pick-and-place 단계 순서 (id 오름차순이 기대되는 진행 방향).
EXPECTED_ORDER = ["approach", "grasp", "lift", "transport", "place"]


def _runs(z):
    """연속 동일값 구간을 (value, length) 리스트로 압축."""
    out = []
    for v in z:
        v = int(v)
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return [(v, n) for v, n in out]


def _ascii_timeline(z, width=60):
    """z_t 를 width 칸 ASCII 로 다운샘플 (각 칸 = 그 구간의 최빈 단계 id)."""
    z = np.asarray(z)
    T = len(z)
    if T == 0:
        return ""
    if T <= width:
        return "".join(str(int(v)) for v in z)
    idx = np.linspace(0, T, width + 1).astype(int)
    chars = []
    for a, b in zip(idx[:-1], idx[1:]):
        seg = z[a:max(b, a + 1)]
        vals, counts = np.unique(seg, return_counts=True)
        chars.append(str(int(vals[np.argmax(counts)])))
    return "".join(chars)


def diagnose_one(z, phase_names, z_source, stem):
    """한 데모 진단 -> (텍스트 블록, ok(bool))."""
    z = np.asarray(z, dtype=int)
    T = len(z)
    pid = {n: i for i, n in enumerate(phase_names)}

    runs = _runs(z)
    present_ids = sorted(set(int(v) for v in z))
    present_names = [phase_names[i] if 0 <= i < len(phase_names) else str(i) for i in present_ids]

    # 단계별 총 스텝 수 / 비율
    counts = {i: int(np.sum(z == i)) for i in present_ids}

    # 순서 문자열: approach(41) -> grasp(12) -> ...
    seq_str = " -> ".join(
        f"{phase_names[v] if 0 <= v < len(phase_names) else v}({n})" for v, n in runs
    )

    # 뒤로 가는(더 낮은 id 로) 전이 수 = 되돌아감/뒤섞임 지표
    backward = sum(1 for a, b in zip(runs[:-1], runs[1:]) if b[0] < a[0])

    # place 도달 & 끝부분에 있는지 (마지막 20% 안에 place 가 있으면 도달로 간주)
    place_id = pid.get("place", None)
    reached_place = False
    if place_id is not None and place_id in present_ids and T > 0:
        tail = z[int(0.8 * T):]
        reached_place = bool(np.any(tail == place_id))

    # --- 판정 ---
    reasons = []
    # 1) 단계가 사실상 하나뿐이면 분절 실패
    if len(present_ids) <= 1:
        reasons.append("단계가 1개뿐 -> 분절 안 됨(평평)")
    # 2) place 미도달
    if place_id is not None and not reached_place:
        reasons.append("place(마지막 단계)까지 도달 못함")
    # 3) 되돌아감 과다
    if backward >= 2:
        reasons.append(f"단계가 뒤로 되돌아감 {backward}회(뒤섞임)")
    # 4) 너무 잘게 쪼개짐
    if len(runs) > max(2 * len(phase_names), 8):
        reasons.append(f"구간이 너무 잘게 나뉨({len(runs)}개)")

    ok = len(reasons) == 0
    verdict = "✅ 잘 분절됨" if ok else "⚠️ 의심: " + "; ".join(reasons)

    cover = "".join(
        f"{n}{'✓' if pid.get(n) in present_ids else '✗'} " for n in phase_names
    )

    lines = [
        f"demo: {stem}   (T={T}, z_source={z_source})",
        f"  순서   : {seq_str}",
        f"  커버   : {cover.strip()}   ({len(present_ids)}/{len(phase_names)} 단계 등장)",
        f"  타임라인: {_ascii_timeline(z)}",
        f"  되돌아감: {backward}회   place 도달: {'예' if reached_place else '아니오'}",
        f"  판정   : {verdict}",
    ]
    return "\n".join(lines), ok


def _load_from_m1(proc_dir):
    """(_m1.npz 우선) -> list of (stem, z_t, phase_names, z_source)."""
    out = []
    for p in sorted(glob.glob(os.path.join(proc_dir, "*_m1.npz"))):
        d = np.load(p, allow_pickle=True)
        if "z_t" not in d:
            continue
        names = [str(x) for x in d["phase_names"]] if "phase_names" in d else None
        src = str(d["z_source"]) if "z_source" in d else "unknown"
        stem = os.path.splitext(os.path.basename(p))[0]
        out.append((stem, d["z_t"], names, src))
    return out


def _load_from_features(proc_dir, cfg):
    """폴백: _m1.npz 가 없으면 features.npz + weak_labels 로 부트스트랩 분절."""
    from phase_segmenter.weak_labels import weak_labels  # numpy만 필요
    names = get_phase_names(cfg)
    out = []
    for p in sorted(glob.glob(os.path.join(proc_dir, "*.npz"))):
        if p.endswith("_m1.npz"):
            continue
        d = np.load(p, allow_pickle=True)
        if "features" not in d:
            continue
        fn = [str(x) for x in d["feature_names"]]
        z = weak_labels(d["features"], fn, cfg)["smoothed"]
        stem = os.path.splitext(os.path.basename(p))[0]
        out.append((stem, z, names, "weak(bootstrap, 학습 전)"))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="데모가 approach->grasp->lift->transport->place 로 분절됐는지 텍스트로 진단."
    )
    parser.add_argument("--config", required=True, help="M1 YAML config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    phase_names = get_phase_names(cfg)

    proc_dir = resolve_path(cfg["paths"]["processed_out"])
    if not os.path.isdir(proc_dir):
        raise SystemExit(f"processed 폴더가 없습니다: {proc_dir}\n"
                         "먼저 build_feature_bank.py (그리고 train.py -> m1_pipeline.py) 를 도세요.")

    demos = _load_from_m1(proc_dir)
    used = "_m1.npz (z_t)"
    if not demos:
        demos = _load_from_features(proc_dir, cfg)
        used = "features.npz + weak_labels (학습 전 부트스트랩)"

    if not demos:
        raise SystemExit(
            f"{proc_dir} 안에 진단할 파일이 없습니다.\n"
            "  1) python feature_bank/build_feature_bank.py --config <cfg>\n"
            "  2) python phase_segmenter/train.py            --config <cfg>\n"
            "  3) python m1_pipeline.py                      --config <cfg>\n"
            "를 먼저 실행하세요."
        )

    print("=" * 72)
    print(f" 분절 진단  |  기준 파일: {used}")
    print(f" 단계 id    |  " + "  ".join(f"{i}={n}" for i, n in enumerate(phase_names)))
    print("=" * 72)

    n_ok = 0
    for stem, z, names, src in demos:
        pn = names if names else phase_names
        block, ok = diagnose_one(z, pn, src, stem)
        n_ok += int(ok)
        print(block)
        print("-" * 72)

    n = len(demos)
    print(f"\n요약: {n}개 데모 중 {n_ok}개 '잘 분절됨', {n - n_ok}개 '의심'.")
    if demos and demos[0][3].startswith("weak"):
        print("주의: 아직 학습 전(부트스트랩) 분절을 본 것입니다. 진짜 판단은 train.py 후 "
              "m1_pipeline.py 를 돌려 z_source=q_omega 인 _m1.npz 로 다시 진단하세요.")
    if n_ok < n:
        print("팁: '의심'이 많으면 (1) 데모 개수를 20~30개로 늘리고 (2) 각 데모에서 "
              "실제로 집어서 통에 넣어 성공시켰는지, (3) train 로그의 val acc/macroF1 을 확인하세요.")


if __name__ == "__main__":
    main()
