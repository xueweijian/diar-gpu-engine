#!/usr/bin/env python3
"""Generate the M3 stage-1 profile kernel code_file from the K5 runner.

Same one-pin-line discipline as build_k6_entry.py: the M3 kernel is the
stage3 runner with

    DEFAULT_GATE = "k5a"   ->   DEFAULT_GATE = "m3prof"

flipped — which additionally compiles k5_runner with -DDIAR_PROFILE_STAGE
(compile_runner(profile=...)). Everything else is byte-for-byte, machine-
enforced by tests/test_m3_stage1.py (regenerate + compare).

Order matters on every push:
    python3 kaggle/m2_stage3/embed_stage3.py      # refresh anchors (K5 file)
    python3 scripts/build_k6_entry.py             # K6 entry
    python3 scripts/build_m3_entry.py             # M3 entry
    # then push the kernels

Usage:
    python3 scripts/build_m3_entry.py           # generate
    python3 scripts/build_m3_entry.py --check   # verify committed copy
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "kaggle" / "m2_stage3" / "m2_stage3_run.py"
DST = REPO / "kaggle" / "m3_stage1_profile" / "m3_stage1_run.py"

K5A_PIN = 'DEFAULT_GATE = "k5a"'
M3_PIN = 'DEFAULT_GATE = "m3prof"'


def render(src_text: str) -> str:
    if src_text.count(K5A_PIN) != 1:
        raise RuntimeError(f"expected exactly one {K5A_PIN!r} in {SRC}")
    return src_text.replace(K5A_PIN, M3_PIN)


def main() -> int:
    check = "--check" in sys.argv
    src_text = SRC.read_text(encoding="utf-8")
    rendered = render(src_text)
    if check:
        if not DST.exists():
            print(f"missing {DST} — run build_m3_entry.py")
            return 1
        got = DST.read_text(encoding="utf-8")
        if got != rendered:
            print("DRIFT: M3 entry != runner with DEFAULT_GATE flipped "
                  "— run build_m3_entry.py")
            return 1
        print("m3 entry check: ok")
        return 0
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(rendered, encoding="utf-8")
    print(f"m3 entry written: {DST} ({len(rendered)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
