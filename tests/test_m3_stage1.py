"""Mechanics tests for the M3 stage-1 profile kernel payload.

Same discipline as test_m2_stage3_k6.py: the M3 entry is NOT a fork — it
is the stage3 runner with DEFAULT_GATE flipped to "m3prof" (which also
turns on the -DDIAR_PROFILE_STAGE build). Byte equality is machine-checked
here; the aggregation helpers of the embedded gate script are exercised on
hand-built profile dicts (no torch, no weights, no runner).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGE3 = REPO / "kaggle" / "m2_stage3"
M3_DIR = REPO / "kaggle" / "m3_stage1_profile"


def test_m3_entry_embedded_payload_is_fresh():
    """Committed M3 entry == runner with the DEFAULT_GATE pin flipped."""
    sys.path.insert(0, str(REPO / "scripts"))
    import build_m3_entry
    src = (STAGE3 / "m2_stage3_run.py").read_text(encoding="utf-8")
    rendered = build_m3_entry.render(src)
    got = (M3_DIR / "m3_stage1_run.py").read_text(encoding="utf-8")
    assert got == rendered, (
        "DRIFT: kaggle/m3_stage1_profile/m3_stage1_run.py is stale — run "
        "scripts/build_m3_entry.py")


def test_m3_entry_pin_and_profile_compile():
    text = (M3_DIR / "m3_stage1_run.py").read_text(encoding="utf-8")
    assert 'DEFAULT_GATE = "m3prof"' in text
    assert 'DEFAULT_GATE = "k5a"' not in text
    # the m3prof route must compile with the profile flag ON
    assert 'compile_runner(sources, profile=(gate_sel == "m3prof"))' in text


def _gate_module():
    spec = importlib.util.spec_from_file_location(
        "m3_stage1_prof", STAGE3 / "m3_stage1_prof.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_aggregate_sorts_and_computes_pct():
    mod = _gate_module()
    rows = mod.aggregate({
        "stem": {"ns": 900, "calls": 9, "ms_mean": 0.1},
        "head": {"ns": 100, "calls": 9, "ms_mean": 0.011},
        "total": {"ns": 1000, "calls": 0, "ms_mean": 0.0},
    })
    assert [r["stage"] for r in rows] == ["stem", "head"]
    assert rows[0]["pct_of_total"] == 90.0
    assert rows[1]["pct_of_total"] == 10.0
    assert all(r["calls"] == 9 for r in rows)


def test_coverage_top_n():
    mod = _gate_module()
    rows = [
        {"stage": "a", "ns": 60, "calls": 1, "ms_mean": 0.0, "pct_of_total": 60.0},
        {"stage": "b", "ns": 30, "calls": 1, "ms_mean": 0.0, "pct_of_total": 30.0},
        {"stage": "c", "ns": 10, "calls": 1, "ms_mean": 0.0, "pct_of_total": 10.0},
    ]
    assert mod.coverage(rows, 2) == 90.0
    assert mod.coverage(rows, 3) == 100.0
    assert mod.coverage([], 3) == 0.0
