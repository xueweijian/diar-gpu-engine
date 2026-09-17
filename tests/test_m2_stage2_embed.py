"""M2 Stage 2 embed round-trip test (local, no Kaggle/torch needed).

Pins the code_file-only discipline: the runner ships ONLY m2_stage2_run.py,
so K1/K2/K3 must travel embedded. Fails when:
- embed_stage2.py was not re-run after editing a gate or the runner
  (embedded bytes != gate sources);
- an embedded gate lost its job marker (wrong file embedded);
- the embedded payload does not compile;
- the runner lost its materialize fallback (sibling-missing boot fails).
"""
from __future__ import annotations

import ast as _ast
import py_compile
import subprocess
import sys
from pathlib import Path

STAGE2 = Path(__file__).parent.parent / "kaggle" / "m2_stage2"
RUNNER = STAGE2 / "m2_stage2_run.py"
EMBED = STAGE2 / "embed_stage2.py"
GATES = {
    "m2_stage2_k1.py": ("EMBEDDED_M2_STAGE2_K1", "m2_stage2_k1_head_transformer"),
    "m2_stage2_k2.py": ("EMBEDDED_M2_STAGE2_K2", "m2_stage2_k2_conformer"),
    "m2_stage2_k3.py": ("EMBEDDED_M2_STAGE2_K3", "m2_stage2_k3_preencode"),
}


def _embedded(runner_text: str, anchor: str) -> str:
    tree = _ast.parse(runner_text)
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Assign) and node.targets
                and getattr(node.targets[0], "id", "") == anchor):
            seg = _ast.get_source_segment(runner_text, node.value)
            assert seg is not None, f"cannot reslice {anchor}"
            return _ast.literal_eval(seg)
    raise AssertionError(f"anchor {anchor} missing in runner")


def test_embed_fresh() -> None:
    r = subprocess.run([sys.executable, str(EMBED), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"embed stale — re-run embed_stage2.py: {r.stderr[-1500:]}"


def test_embedded_matches_sources_and_compiles() -> None:
    import tempfile
    import os
    runner_text = RUNNER.read_text(encoding="utf-8")
    for gate, (anchor, marker) in GATES.items():
        payload = _embedded(runner_text, anchor)
        src = (STAGE2 / gate).read_text(encoding="utf-8")
        assert payload == src, f"{anchor} != {gate} (re-run embed_stage2.py)"
        assert marker in payload, f"{anchor} missing job marker {marker}"
        assert "teacher-forced" in payload
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, gate)
            Path(p).write_text(payload, encoding="utf-8")
            py_compile.compile(p, doraise=True)


def test_runner_materialize_fallback() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "_materialize" in text, "runner lost embedded-materialize fallback"
    assert "runpy.run_path" in text, "runner lost runpy dispatch"


def test_runner_compiles() -> None:
    py_compile.compile(str(RUNNER), doraise=True)
