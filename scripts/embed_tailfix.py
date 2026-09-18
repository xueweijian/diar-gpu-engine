"""Inline the vendored NeMo dump script into the tail-fixture kernel.

Kaggle script kernels upload ONLY code_file, so kaggle/m2_tailfix/ never
ships as a directory. Run this locally BEFORE pushing the kernel:

    python3 scripts/embed_tailfix.py

The dump script itself is shared with the Stage 0 refdump kernel
(kaggle/m2_refdump/dump_m2_reference.py) — one copy, no fork drift.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "kaggle" / "m2_tailfix" / "m2_tail_fixture.py"
DUMP = REPO / "kaggle" / "m2_refdump" / "dump_m2_reference.py"
ANCHOR = "EMBEDDED_DUMP_M2_REFERENCE = "


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    tree = ast.parse(text)
    old_blob = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and node.targets
                and getattr(node.targets[0], "id", "") == "EMBEDDED_DUMP_M2_REFERENCE"):
            old_blob = ast.get_source_segment(text, node.value)
            break
    assert old_blob is not None, "embedded assignment missing"
    payload = DUMP.read_text(encoding="utf-8")
    assert "_capture_block_outputs" in payload, "dump fork missing hooks"
    blob = repr(payload)
    TARGET.write_text(text.replace(ANCHOR + old_blob, ANCHOR + blob, 1),
                      encoding="utf-8")
    print(f"embedded {len(payload)} bytes into {TARGET.name}")


if __name__ == "__main__":
    main()
