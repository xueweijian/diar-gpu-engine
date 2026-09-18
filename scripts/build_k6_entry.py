#!/usr/bin/env python3
"""Generate the K6 kernel code_file from the K5 runner source.

Kaggle script kernels run exactly one file and take NO command-line
arguments, so the K6 kernel cannot say `--gate k6` at runtime. Instead it
ships a *generated* copy of kaggle/m2_stage3/m2_stage3_run.py with the
single pin line

    DEFAULT_GATE = "k5a"   ->   DEFAULT_GATE = "k6"

flipped. Everything else (embedded gate scripts, C++ b64 blobs) is copied
byte-for-byte, so the K6 entry is NOT a fork — the one-line rule is
machine-enforced by tests/test_m2_stage3_k6.py (regenerate + compare).

Order matters on every push of either kernel:
    python3 kaggle/m2_stage3/embed_stage3.py      # refresh anchors (K5 file)
    python3 scripts/build_k6_entry.py             # regenerate K6 entry
    # then push the kernels

Usage:
    python3 scripts/build_k6_entry.py           # generate
    python3 scripts/build_k6_entry.py --check   # verify committed copy
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "kaggle" / "m2_stage3" / "m2_stage3_run.py"
DST = REPO / "kaggle" / "m2_stage3_k6" / "m2_stage3_run_k6.py"

K5A_PIN = 'DEFAULT_GATE = "k5a"'
K6_PIN = 'DEFAULT_GATE = "k6"'


def render(src_text: str) -> str:
    if src_text.count(K5A_PIN) != 1:
        raise RuntimeError(f"expected exactly one {K5A_PIN!r} in {SRC}")
    return src_text.replace(K5A_PIN, K6_PIN)


def main() -> int:
    check = "--check" in sys.argv
    src_text = SRC.read_text(encoding="utf-8")
    rendered = render(src_text)
    if check:
        if not DST.exists():
            print(f"missing {DST} — run build_k6_entry.py")
            return 1
        got = DST.read_text(encoding="utf-8")
        if got != rendered:
            print("DRIFT: K6 entry != runner with DEFAULT_GATE flipped "
                  "— run build_k6_entry.py")
            return 1
        print("k6 entry check: ok")
        return 0
    DST.write_text(rendered, encoding="utf-8")
    print(f"k6 entry written: {DST} ({len(rendered)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
