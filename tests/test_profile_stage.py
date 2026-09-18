"""M3 Stage 1 — DIAR_PROFILE_STAGE compile-flag profiler mechanics.

Contract (docs/M3-P100-CUDA-PLAN.md §2):
  * zero behavior change: the profile build's selftest output is textually
    identical to the plain build's;
  * the profile report covers every decision-relevant stage with calls>0
    and a total equal to the sum of stage rows;
  * the plain build accepts (and ignores) --profile-out without failing —
    the Kaggle runner passes it unconditionally.

Binaries are built on demand into .local-build (same recipe as
test_m2_stage3_k5a.py; the profile build additionally gets
-DDIAR_PROFILE_STAGE).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / ".local-build"
RUNNER_SRC = REPO / "tools" / "k5_runner.cpp"

# Every engine/sortformer tap site (profile.hpp taxonomy). The selftest
# drives one full chunk through DiarEngine, so all of them must fire.
EXPECTED_STAGES = {
    "fe", "stem", "concat", "xscale", "pe", "conformer", "proj",
    "transformer", "head", "trim", "aosc", "gate",
}


def _build(profile: bool) -> Path:
    suffix = "_prof" if profile else ""
    exe = BUILD / f"k5_runner{suffix}"
    srcs = [str(RUNNER_SRC)]
    srcs += [str(p) for p in sorted((REPO / "src").glob("*.cpp"))]
    headers = sorted((REPO / "include" / "diar").glob("*.hpp"))
    newest = max(p.stat().st_mtime for p in [*map(Path, srcs), *headers])
    if exe.exists() and exe.stat().st_mtime >= newest:
        return exe
    BUILD.mkdir(parents=True, exist_ok=True)
    cmd = ["g++", "-std=c++17", "-O2", "-o", str(exe), *srcs,
           "-I", str(REPO / "include")]
    if profile:
        cmd.insert(1, "-DDIAR_PROFILE_STAGE")
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    return exe


def _selftest(exe: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(exe), "--selftest", *extra],
                          capture_output=True, text=True, timeout=300)


def test_profile_build_selftest_identical_to_plain():
    plain = _selftest(_build(profile=False))
    prof = _selftest(_build(profile=True))
    assert plain.returncode == 0 and prof.returncode == 0
    # Zero behavior change: identical selftest transcript, including the
    # printed row/frame counts.
    assert plain.stdout == prof.stdout


def test_profile_report_covers_all_stages(tmp_path: Path):
    out = tmp_path / "profile.json"
    p = _selftest(_build(profile=True), "--profile-out", str(out))
    assert p.returncode == 0
    rep = json.loads(out.read_text())

    missing = EXPECTED_STAGES - set(rep)
    assert not missing, f"stages never fired: {sorted(missing)}"
    for stage in EXPECTED_STAGES:
        row = rep[stage]
        assert row["calls"] > 0, stage
        assert row["ns"] > 0, stage
        assert row["ms_mean"] > 0.0, stage
    # total == sum of stage rows (no double counting, no unprofiled holes
    # beyond nanosecond rounding between clock reads).
    total = sum(row["ns"] for k, row in rep.items() if k != "total")
    assert rep["total"]["ns"] >= total
    # Tiny-config single chunk: wall-clock sanity (ms, not seconds).
    assert rep["total"]["ns"] < 60_000_000_000


def test_plain_build_ignores_profile_out(tmp_path: Path):
    out = tmp_path / "profile.json"
    p = _selftest(_build(profile=False), "--profile-out", str(out))
    assert p.returncode == 0
    assert not out.exists(), "plain build must not write a profile report"
