"""Local tests for the measurement-matrix kernel (no GPU/CUDA/Kaggle/git).

Run:  python3 -m pytest tests/test_sortformer_matrix.py -q
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

KERNEL = Path(__file__).resolve().parents[1] / "kaggle" / "sortformer_matrix" / "sortformer_matrix.py"


def load_kernel_functions():
    """Load pure helpers without executing the Kaggle-only module body.

    The kernel imports diar_harness at module top level; extract only the
    dependency-free functions (DET_SETS, select_entries, run_det_sweep's
    summarizer) so tests need no GPU, CUDA, dataset mounts or network.
    run_det_sweep itself calls h.diarize_once, so tests exercise it with a
    stubbed harness module injected into sys.modules.
    """
    source = KERNEL.read_text(encoding="utf-8")
    start = source.index("DET_SETS")
    end = source.index("def fit_scaling(")
    snippet = "from __future__ import annotations\nimport os\n" + source[start:end]
    module = types.ModuleType("matrix_det_snippet")
    module.__dict__["__name__"] = "matrix_det_snippet"
    exec(compile(snippet, str(KERNEL), "exec"), module.__dict__)  # noqa: S102 - local test fixture
    return module


_snippet = load_kernel_functions()
select_entries = _snippet.select_entries
run_det_sweep = _snippet.run_det_sweep
DET_SETS = _snippet.DET_SETS


def test_select_entries_filters_entry_dicts_by_label_prefix():
    entries = [
        {"label": "real_short_x", "audio_seconds": 57.0},
        {"label": "real_mid_y", "audio_seconds": 357.0},
        {"label": "synth_300s", "audio_seconds": 300.0},
    ]
    matched = select_entries(entries, "real_")
    assert [e["label"] for e in matched] == ["real_short_x", "real_mid_y"]


def test_select_entries_does_not_raise_keyerror_on_dicts():
    """Regression: the old tuple-based select() raised KeyError: 0 here."""
    entries = [{"label": "real_long", "audio_seconds": 6432.0}]
    assert select_entries(entries, "real_") == entries


def test_select_entries_empty_and_missing_labels():
    assert select_entries([], "real_") == []
    assert select_entries([{"audio_seconds": 1.0}], "real_") == []


def test_fixture_labels_match_entry_prefix_contract(tmp_path):
    """Kernel builds real labels 'real_short_*'/'real_mid_*'/'real_long'.

    The downstream filter needs the 'real_' prefix; guard the contract.
    """
    labels = ["real_short_abc", "real_mid_xyz", "real_long", "synth_300s"]
    assert all(label.startswith("real_") for label in labels[:3])
    assert not labels[3].startswith("real_")


def test_det_sets_first_is_empty_baseline():
    """The sweep's set[0] is the no-knob control; cross-set comparison keys on it."""
    assert DET_SETS[0][0] == "baseline"
    assert DET_SETS[0][1] == {}
    names = [name for name, _ in DET_SETS]
    assert len(set(names)) == len(names)
    envs = [tuple(sorted(env.items())) for _, env in DET_SETS]
    assert len(set(envs)) == len(envs)


def _stub_harness(monkeypatch, tmp_path, bodies):
    """Stub the `h` harness module the snippet resolved at snippet-exec time.

    run_det_sweep calls h.WORK_ROOT / h.diarize_once / h.read_segments.
    bodies: list of RTTM-body strings returned in call order.
    """
    import itertools

    calls: list[dict[str, object]] = []
    body_iter = itertools.cycle(bodies)

    stub = types.ModuleType("matrix_det_snippet_h")
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    stub.WORK_ROOT = work

    def fake_diarize_once(binary, audio, output, device="cuda:0", preset=None,
                          extra_args=None, timeout=3600, env=None):
        calls.append({"output": str(output), "env": dict(env or {}), "preset": preset})
        body = next(body_iter)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for i, part in enumerate(body.split("|")):
            s, d, spk = part.split()
            lines.append(
                f"SPEAKER rec 1 {s} {d} <NA> <NA> {spk} <NA> <NA>")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"returncode": 0, "wall_seconds": 0.5}

    def fake_read_segments(path):
        segs = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            f = line.split()
            segs.append((float(f[3]), float(f[4]), f[7]))
        return segs

    stub.diarize_once = fake_diarize_once
    stub.read_segments = fake_read_segments
    monkeypatch.setitem(sys.modules, "matrix_det_snippet", _snippet)
    _snippet.h = stub
    return stub, calls


def test_det_sweep_baseline_identical_reports_identical(tmp_path, monkeypatch):
    stub, calls = _stub_harness(monkeypatch, tmp_path, ["0.0 1.0 spk"])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    sweep = run_det_sweep(Path("/nope/bin"), [("case1", audio, None)])
    assert len(sweep["cases"]) == 1
    sets = sweep["cases"][0]["sets"]
    assert len(sets) == len(DET_SETS)
    assert all(s["body_identical"] for s in sets)
    assert all(s["unique_body_hashes"] == 1 for s in sets)
    assert all(s["cross_set_matches_baseline_rep0"] for s in sets[1:])
    assert sets[0]["cross_set_matches_baseline_rep0"] is None
    # every set ran the same repeat count with isolated output paths
    assert len(calls) == len(DET_SETS) * 3
    assert len({c["output"] for c in calls}) == len(calls)


def test_det_sweep_distinguishes_drift_and_passes_env(tmp_path, monkeypatch):
    """Alternating bodies => drift detected; env reaches the child per set."""
    stub, calls = _stub_harness(
        monkeypatch, tmp_path, ["0.0 1.0 spk", "0.0 1.5 spk"])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    sweep = run_det_sweep(Path("/nope/bin"), [("case1", audio, "offline")])
    sets = sweep["cases"][0]["sets"]
    assert all(not s["body_identical"] for s in sets)
    assert all(s["unique_body_hashes"] == 2 for s in sets)
    assert all(s["pairwise_with_rep0"] == round(2 / 3, 4) for s in sets)
    # env of each set forwarded verbatim (baseline empty, others carry knobs)
    env_lists = [sorted(dict(s["env"]).items()) for s in sets]
    assert env_lists[0] == []
    assert any("CUDA_LAUNCH_BLOCKING" in dict(e) for e in env_lists[1:])
    assert any("CUBLAS_WORKSPACE_CONFIG" in dict(e) for e in env_lists[1:])
    assert any("GGML_SKINNY_Q8_CUBLAS_F16" in dict(e) for e in env_lists[1:])
    child_envs = [tuple(sorted(c["env"].items())) for c in calls]
    assert tuple() in child_envs  # baseline set ran knob-free
