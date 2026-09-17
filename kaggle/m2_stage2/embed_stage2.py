"""Concatenate K1/K2/K3 gate scripts into the single code_file the kernel uploads.

Kaggle script kernels upload ONLY code_file, so sibling gate sources never
reach /kaggle/src. Same mechanism as Stage 0 embed_dump.py
(kaggle/m2_refdump/embed_dump.py): the runner (m2_stage2_run.py) carries
an EMBED placeholder per gate; at runtime it prefers a sibling file if
present and otherwise materializes the embedded copy. Locally the runner
keeps its runpy fallback so mechanics stay runnable pre/post embed.

Usage (locally, BEFORE push):
    python3 kaggle/m2_stage2/embed_stage2.py        # embed all three
    python3 kaggle/m2_stage2/embed_stage2.py --check  # verify embedded == sources

Round-trip is pinned by tests/test_m2_stage2_embed.py.
"""
from __future__ import annotations

import argparse
import ast as _ast
import sys
from pathlib import Path

DIR = Path(__file__).parent
RUNNER = DIR / "m2_stage2_run.py"
GATES = ("m2_stage2_k1.py", "m2_stage2_k2.py", "m2_stage2_k3.py")

ANCHOR_PREFIX = "EMBEDDED_"
ANCHOR_OF = {
    "m2_stage2_k1.py": "EMBEDDED_M2_STAGE2_K1",
    "m2_stage2_k2.py": "EMBEDDED_M2_STAGE2_K2",
    "m2_stage2_k3.py": "EMBEDDED_M2_STAGE2_K3",
}

# The runner's dispatch table must call the materialized files by these
# names; _materialize() below writes them next to the runner at runtime.
GATE_MODULE_OF = {
    "m2_stage2_k1.py": "m2_stage2_k1.py",
    "m2_stage2_k2.py": "m2_stage2_k2.py",
    "m2_stage2_k3.py": "m2_stage2_k3.py",
}

SMOKE = "teacher-forced"  # every gate source must contain this marker


def _read_anchor(text: str, anchor: str) -> str:
    tree = _ast.parse(text)
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Assign) and node.targets
                and getattr(node.targets[0], "id", "") == anchor):
            seg = _ast.get_source_segment(text, node.value)
            assert seg is not None, f"cannot reslice {anchor}"
            return seg
    raise AssertionError(f"embed anchor {anchor} missing in runner")


def _gate_guard(name: str) -> str:
    # Cheap provenance: gate sources must carry their job marker so a stale
    # embed fails loudly instead of shipping the wrong gate.
    markers = {
        "m2_stage2_k1.py": "m2_stage2_k1_head_transformer",
        "m2_stage2_k2.py": "m2_stage2_k2_conformer",
        "m2_stage2_k3.py": "m2_stage2_k3_preencode",
    }
    return markers[name]


def embed(check_only: bool = False) -> int:
    text = RUNNER.read_text(encoding="utf-8")
    changed = False
    for gate in GATES:
        anchor = ANCHOR_OF[gate]
        payload = (DIR / gate).read_text(encoding="utf-8")
        assert SMOKE in payload, f"{gate} missing smoke marker {SMOKE!r}"
        assert _gate_guard(gate) in payload, f"{gate} missing job marker"
        old_blob = _read_anchor(text, anchor)
        if check_only:
            old_val = _ast.literal_eval(old_blob)
            if old_val != payload:
                print(f"STALE: {anchor} != {gate}", file=sys.stderr)
                return 1
            continue
        blob = repr(payload)
        text = text.replace(anchor + " = " + old_blob, anchor + " = " + blob, 1)
        changed = True
        print(f"embedded {len(payload)} bytes {gate} -> {anchor}")
    if not check_only and changed:
        RUNNER.write_text(text, encoding="utf-8")
    if check_only:
        print("embed fresh")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify embedded copies match sources")
    args = ap.parse_args()
    return embed(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
