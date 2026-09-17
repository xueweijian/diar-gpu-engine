"""Local tests for run_prob_sweep (no GPU/CUDA/Kaggle/git).

Run:  python3 -m pytest tests/test_prob_sweep.py -q

run_prob_sweep is the v9 verdict engine: per-rep probdump + RTTM bodies,
rep0-vs-repN pairs at both levels, and the five-way verdict
(identical / neural_drift / host_race / sub_hysteresis_drift / no_dump).
These tests load the function from the kernel snippet (same loader pattern
as test_sortformer_matrix.py) and stub the harness, so a kernel/test skew
fails here in 1 second instead of on Kaggle in 55 minutes.
"""
from __future__ import annotations

import struct
import sys
import types
from pathlib import Path

import pytest

KERNEL = Path(__file__).resolve().parents[1] / "kaggle" / "sortformer_matrix" / "sortformer_matrix.py"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import diar_harness as real_harness  # noqa: E402


def load_prob_snippet():
    source = KERNEL.read_text(encoding="utf-8")
    start = source.index("# v8 pipeline-knob sweep:")
    end = source.index("def run_cpu(")
    snippet = ("from __future__ import annotations\nimport os\n"
               "import json\nimport time\nfrom pathlib import Path\n"
               "COMMIT = 'testcommit'\n"
               + source[start:end])
    module = types.ModuleType("matrix_prob_snippet")
    module.__dict__["__name__"] = "matrix_prob_snippet"
    exec(compile(snippet, str(KERNEL), "exec"), module.__dict__)  # noqa: S102 - local test fixture
    # These tests cover the PATCHED sweep path (probs dumps on); the kernel
    # file may carry PROBDUMP_ENABLED=False while an unpatched v14-style
    # session runs. The False path has its own test via monkeypatch.
    module.PROBDUMP_ENABLED = True
    return module


_snippet = load_prob_snippet()
run_prob_sweep = _snippet.run_prob_sweep
DET_SETS_V9 = _snippet.DET_SETS_V9
OFFLINE_FLAG_SENTINEL = _snippet.OFFLINE_FLAG_SENTINEL


def _stub_h(monkeypatch, tmp_path, bodies, prob_values=None, prob_error=None):
    """Stub h with per-call bodies; probs written unless prob_error set."""
    import itertools

    calls: list[dict[str, object]] = []
    body_iter = itertools.cycle(bodies)
    stub = types.ModuleType("matrix_prob_snippet_h")
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    stub.WORK_ROOT = work

    def fake_diarize_once(binary, audio, output, device="cuda:0", preset=None,
                          extra_args=None, timeout=3600, env=None,
                          dump_probs_path=None):
        calls.append({"output": str(output), "preset": preset,
                      "extra_args": list(extra_args or []),
                      "env": dict(env or {}),
                      "dump": str(dump_probs_path) if dump_probs_path else None})
        body = next(body_iter)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for part in body.split("|"):
            s, d, spk = part.split()
            lines.append(f"SPEAKER rec 1 {s} {d} <NA> <NA> {spk} <NA> <NA>")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result: dict[str, object] = {
            "returncode": 0, "wall_seconds": 0.5, "tail": "",
            "probs_sha256": None,
        }
        if dump_probs_path and prob_error is None:
            values = prob_values if prob_values is not None else [0.5] * 8
            dump = Path(str(dump_probs_path))
            dump.parent.mkdir(parents=True, exist_ok=True)
            with dump.open("wb") as handle:
                handle.write(struct.pack("<q", 2))
                handle.write(struct.pack("<i", 4))
                handle.write(struct.pack(f"<{len(values)}f", *values))
            import hashlib
            result["probs_sha256"] = hashlib.sha256(dump.read_bytes()).hexdigest()
        return result

    def fake_read_segments(path):
        segs = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            f = line.split()
            segs.append((float(f[3]), float(f[4]), f[7]))
        return segs

    def fake_read_probdump(path):
        if prob_error is not None:
            raise prob_error
        return real_harness.read_probdump(Path(path))

    stub.diarize_once = fake_diarize_once
    stub.read_segments = fake_read_segments
    stub.read_probdump = fake_read_probdump
    stub.probdiff = real_harness.probdiff
    monkeypatch.setitem(sys.modules, "matrix_prob_snippet", _snippet)
    _snippet.h = stub
    _snippet.N_DET_REPEAT = 3
    return stub, calls


def test_v9_sets_shape_and_offline_sentinel():
    names = [name for name, _, _, _ in DET_SETS_V9]
    assert names == ["short_streaming", "mid_streaming", "mid_offline_full",
                     "mid_offline_preset"]
    reps = {name: n for name, _, n, _ in DET_SETS_V9}
    assert reps["mid_streaming"] == 5
    assert reps["short_streaming"] == 3
    full = [s for s in DET_SETS_V9 if s[0] == "mid_offline_full"][0]
    assert full[1] == OFFLINE_FLAG_SENTINEL  # flag, not a preset value


def test_prob_sweep_identical_when_all_stable(tmp_path, monkeypatch):
    _, calls = _stub_h(monkeypatch, tmp_path, ["0.0 1.0 spk"])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    sweep = run_prob_sweep(Path("/nope/bin"), [("case1", audio, None)])
    case = sweep["cases"][0]
    assert case["verdict"] == "identical"
    assert case["probs_bit_identical"] is True
    assert case["body_identical"] is True
    assert case["probs_available"] is True
    assert len(case["reps"]) == 3
    assert len({c["dump"] for c in calls}) == 3  # isolated dump per rep


def test_prob_sweep_neural_drift_when_both_levels_move(tmp_path, monkeypatch):
    import itertools
    bodies = itertools.cycle(["0.0 1.0 spk", "0.0 1.5 spk"])
    probseq = itertools.cycle([[0.5] * 8, [0.9] * 8])

    stub_calls: list[dict[str, object]] = []
    stub = types.ModuleType("matrix_prob_snippet_h2")
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    stub.WORK_ROOT = work

    def fake_diarize_once(binary, audio, output, device="cuda:0", preset=None,
                          extra_args=None, timeout=3600, env=None,
                          dump_probs_path=None):
        stub_calls.append({})
        body = next(bodies)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for part in body.split("|"):
            s, d, spk = part.split()
            lines.append(f"SPEAKER rec 1 {s} {d} <NA> <NA> {spk} <NA> <NA>")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        values = next(probseq)
        dump = Path(str(dump_probs_path))
        with dump.open("wb") as handle:
            handle.write(struct.pack("<q", 2))
            handle.write(struct.pack("<i", 4))
            handle.write(struct.pack("<8f", *values))
        return {"returncode": 0, "wall_seconds": 0.5, "tail": "",
                "probs_sha256": "x"}

    def fake_read_segments(path):
        return [(float(l.split()[3]), float(l.split()[4]), l.split()[7])
                for l in Path(path).read_text(encoding="utf-8").splitlines()]

    def fake_read_probdump(path):
        return real_harness.read_probdump(Path(path))

    stub.diarize_once = fake_diarize_once
    stub.read_segments = fake_read_segments
    stub.read_probdump = fake_read_probdump
    stub.probdiff = real_harness.probdiff
    monkeypatch.setitem(sys.modules, "matrix_prob_snippet", _snippet)
    _snippet.h = stub
    _snippet.N_DET_REPEAT = 3
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    sweep = run_prob_sweep(Path("/nope/bin"), [("case1", audio, None)])
    case = sweep["cases"][0]
    assert case["verdict"] == "neural_drift"
    assert case["probs_bit_identical"] is False
    assert case["body_identical"] is False


def test_prob_sweep_host_race_when_probs_stable_bodies_move(tmp_path, monkeypatch):
    import itertools
    bodies = itertools.cycle(["0.0 1.0 spk", "0.0 1.5 spk"])
    stub = types.ModuleType("matrix_prob_snippet_h3")
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    stub.WORK_ROOT = work

    def fake_diarize_once(binary, audio, output, device="cuda:0", preset=None,
                          extra_args=None, timeout=3600, env=None,
                          dump_probs_path=None):
        body = next(bodies)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for part in body.split("|"):
            s, d, spk = part.split()
            lines.append(f"SPEAKER rec 1 {s} {d} <NA> <NA> {spk} <NA> <NA>")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        dump = Path(str(dump_probs_path))
        with dump.open("wb") as handle:
            handle.write(struct.pack("<q", 2))
            handle.write(struct.pack("<i", 4))
            handle.write(struct.pack("<8f", *[0.5] * 8))
        return {"returncode": 0, "wall_seconds": 0.5, "tail": "",
                "probs_sha256": "x"}

    def fake_read_segments(path):
        return [(float(l.split()[3]), float(l.split()[4]), l.split()[7])
                for l in Path(path).read_text(encoding="utf-8").splitlines()]

    def fake_read_probdump(path):
        return real_harness.read_probdump(Path(path))

    stub.diarize_once = fake_diarize_once
    stub.read_segments = fake_read_segments
    stub.read_probdump = fake_read_probdump
    stub.probdiff = real_harness.probdiff
    monkeypatch.setitem(sys.modules, "matrix_prob_snippet", _snippet)
    _snippet.h = stub
    _snippet.N_DET_REPEAT = 3
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    sweep = run_prob_sweep(Path("/nope/bin"), [("case1", audio, None)])
    assert sweep["cases"][0]["verdict"] == "host_race"


def test_prob_sweep_no_dump_fails_loud(tmp_path, monkeypatch):
    _stub_h(monkeypatch, tmp_path, ["0.0 1.0 spk"],
            prob_error=FileNotFoundError("no such file"))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    sweep = run_prob_sweep(Path("/nope/bin"), [("case1", audio, None)])
    case = sweep["cases"][0]
    assert case["verdict"] == "run_failed"
    assert case["probs_available"] is False
    assert case["reps"][0]["probs_available"] is False


def test_prob_sweep_offline_sentinel_becomes_flag(tmp_path, monkeypatch):
    _, calls = _stub_h(monkeypatch, tmp_path, ["0.0 1.0 spk"])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    sweep = run_prob_sweep(
        Path("/nope/bin"), [("mid_offline_full", audio, OFFLINE_FLAG_SENTINEL)])
    case = sweep["cases"][0]
    assert case["verdict"] == "identical"
    assert calls[0]["preset"] is None
    assert calls[0]["extra_args"] == ["--offline"]
