"""Inline kaggle/m2_refdump/dump_m2_reference.py into m2_stage0_refdump.py.

Kaggle script kernels upload ONLY the code_file, so sibling sources never
reach /kaggle/src. Run this locally BEFORE `kaggle kernels push`:

    python3 kaggle/m2_refdump/embed_dump.py

It rewrites the EMBEDDED_DUMP_M2_REFERENCE placeholder in place. The
shipped kernel then materializes the dump script at runtime (sibling
preferred, embedded fallback). Mechanics test pins the round-trip.
"""
from __future__ import annotations

from pathlib import Path

DIR = Path(__file__).parent
SPIKE = DIR / "m2_stage0_refdump.py"
DUMP = DIR / "dump_m2_reference.py"
ANCHOR = "EMBEDDED_DUMP_M2_REFERENCE = "


def main() -> None:
    text = SPIKE.read_text(encoding="utf-8")
    idx = text.find(ANCHOR)
    assert idx >= 0, "embed anchor missing in spike script"
    # Replace from the anchor to the end of its line (the old literal may
    # span many lines after the first embed; find its closing via AST).
    import ast as _ast
    tree = _ast.parse(text)
    old_blob = None
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Assign) and node.targets
                and getattr(node.targets[0], "id", "") == "EMBEDDED_DUMP_M2_REFERENCE"):
            old_blob = _ast.get_source_segment(text, node.value)
            break
    assert old_blob is not None, "embedded assignment missing"
    payload = DUMP.read_text(encoding="utf-8")
    assert "_capture_block_outputs" in payload, "dump fork missing hooks"
    blob = repr(payload)  # valid Python string literal, no hand-escaping
    SPIKE.write_text(text.replace(ANCHOR + old_blob, ANCHOR + blob, 1),
                     encoding="utf-8")
    print(f"embedded {len(payload)} bytes into {SPIKE.name}")


if __name__ == "__main__":
    main()
