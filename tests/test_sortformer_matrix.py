"""Local tests for the measurement-matrix kernel (no GPU/CUDA/Kaggle/git).

Run:  python3 -m pytest tests/test_sortformer_matrix.py -q

v7 (env-knob sweep) is superseded by v8 (pipeline-knob sweep): the kernel's
DET_SETS_V8 is the single source of truth for knob sets; the tests below
import it from the kernel snippet so a kernel/test skew fails loudly here
instead of wasting a 50-minute Kaggle run.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

KERNEL = Path(__file__).resolve().parents[1] / "kaggle" / "sortformer_matrix" / "sortformer_matrix.py"


def load_kernel_functions():
    """Load pure helpers without executing the Kaggle-only module body.

    The kernel imports diar_harness at module top level; extract only the
    dependency-free span (DET_SETS_V8, select_entries, run_det_sweep) so
    tests need no GPU, CUDA, dataset mounts or network. run_det_sweep calls
    h.diarize_once, so tests exercise it with a stubbed harness module.
    """
    source = KERNEL.read_text(encoding="utf-8")
    start = source.index("# v8 pipeline-knob sweep:")
    end = source.index("def run_cpu(")
    snippet = "from __future__ import annotations\nimport os\n" + source[start:end]
    module = types.ModuleType("matrix_det_snippet")
    module.__dict__["__name__"] = "matrix_det_snippet"
    exec(compile(snippet, str(KERNEL), "exec"), module.__dict__)  # noqa: S102 - local test fixture
    return module


_snippet = load_kernel_functions()
select_entries = _snippet.select_entries
run_det_sweep = _snippet.run_det_sweep
DET_SETS_V8 = _snippet.DET_SETS_V8
N_DET_REPEAT = _snippet.N_DET_REPEAT


def test_v8_sets_first_is_empty_baseline():
    """v8 set[0] is the no-knob control; verdicts key on it like v7."""
    assert DET_SETS_V8[0][0] == "baseline"
    assert DET_SETS_V8[0][1] == []
    assert DET_SETS_V8[0][2] == {}
    names = [name for name, _, _ in DET_SETS_V8]
    assert len(set(names)) == len(names)
    combos = [(tuple(args), tuple(sorted(env.items()))) for _, args, env in DET_SETS_V8]
    assert len(set(combos)) == len(combos)


def test_v8_sets_use_only_real_cli_flags():
    """Every v8 extra_arg must be a flag the pinned upstream CLI accepts.

    Pinned against NeMo-Speech.cpp a5b6953 app/diarize.cpp: --no-batching,
    --diar-chunk/--diar-fifo/--diar-spkcache (via ParameterParser fallback),
    --offline. A typo'd flag fails the whole 50-minute kernel at runtime;
    this test fails it in 1 second instead.
    """
    known = {"--no-batching", "--diar-chunk", "--diar-fifo",
             "--diar-spkcache", "--offline"}
    takes_value = {"--diar-chunk", "--diar-fifo", "--diar-spkcache"}
    for name, args, _ in DET_SETS_V8:
        i = 0
        while i < len(args):
            flag = args[i]
            assert flag in known, f"v8 set {name}: unknown flag {flag}"
            if flag in takes_value:
                assert i + 1 < len(args), f"v8 set {name}: {flag} missing value"
                int(args[i + 1])  # geometry values must be integers
                i += 2
            else:
                i += 1


def test_v8_fixed_chunk_matches_streaming_preset_geometry():
    """fixed_chunk must equal the streaming preset, or the comparison is void.

    Upstream aosc_state.h DiarGeometry defaults (= riva_streaming preset):
    spkcache 160 / fifo 80 / chunk 20. If upstream changes these defaults,
    this test forces us to update the set instead of silently comparing
    two different geometries.
    """
    for name, args, _ in DET_SETS_V8:
        if name != "fixed_chunk":
            continue
        pairs = dict(zip(args[::2], args[1::2]))
        assert pairs == {"--diar-chunk": "20", "--diar-fifo": "80",
                         "--diar-spkcache": "160"}


def test_v8_harness_forwards_extra_args_to_child():
    """h.diarize_once(extra_args=...) must reach the child argv verbatim.

    v8's batching/geometry/mode knobs all travel through extra_args
    (v7's knobs used env). If the harness ever drops this parameter,
    every v8 set would silently run the baseline — a whole kernel wasted.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
    import diar_harness as harness

    seen: list[list[str]] = []
    real_run = harness.run

    def fake_run(argv, cwd=None, timeout=1800, env=None):
        seen.append(list(argv))
        return 0, "", 0.01

    harness.run = fake_run
    try:
        out = Path("/tmp/v8_probe_test.rttm")
        harness.diarize_once(
            Path("/nope/binary"), Path("/nope/a.wav"), out,
            extra_args=["--no-batching"])
        assert "--no-batching" in seen[-1]
        harness.diarize_once(
            Path("/nope/binary"), Path("/nope/a.wav"), out,
            extra_args=["--diar-chunk", "20"])
        assert "--diar-chunk" in seen[-1] and "20" in seen[-1]
        # no-knob call must not leak flags from a previous call
        harness.diarize_once(Path("/nope/binary"), Path("/nope/a.wav"), out)
        assert "--no-batching" not in seen[-1]
        assert "--diar-chunk" not in seen[-1]
    finally:
        harness.run = real_run
        if out.exists():
            out.unlink()


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
        calls.append({"output": str(output), "extra_args": list(extra_args or []),
                      "env": dict(env or {}), "preset": preset})
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
    assert len(sets) == len(DET_SETS_V8)
    assert all(s["body_identical"] for s in sets)
    assert all(s["unique_body_hashes"] == 1 for s in sets)
    assert all(s["cross_set_matches_baseline_rep0"] for s in sets[1:])
    assert sets[0]["cross_set_matches_baseline_rep0"] is None
    # every set ran the same repeat count with isolated output paths
    assert len(calls) == len(DET_SETS_V8) * N_DET_REPEAT
    assert len({c["output"] for c in calls}) == len(calls)
    # v8 sets carry (extra_args, env); baseline of both is empty
    assert calls[0]["extra_args"] == [] and calls[0]["env"] == {}


def test_det_sweep_distinguishes_drift_and_passes_knobs(tmp_path, monkeypatch):
    """Alternating bodies => drift detected; knobs reach the child per set."""
    stub, calls = _stub_harness(
        monkeypatch, tmp_path, ["0.0 1.0 spk", "0.0 1.5 spk"])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    sweep = run_det_sweep(Path("/nope/bin"), [("case1", audio, "offline")])
    sets = sweep["cases"][0]["sets"]
    assert all(not s["body_identical"] for s in sets)
    assert all(s["unique_body_hashes"] == 2 for s in sets)
    assert all(s["pairwise_with_rep0"] == round(2 / 3, 4) for s in sets)
    # knobs of each set forwarded verbatim (baseline empty, others carry flags)
    assert sets[0]["extra_args"] == [] and sets[0]["env"] == {}
    flat_args = [a for s in sets[1:] for a in s["extra_args"]]
    assert "--no-batching" in flat_args
    assert "--diar-chunk" in flat_args
    assert "--offline" in flat_args
    child_args = [tuple(c["extra_args"]) for c in calls]
    assert tuple() in child_args  # baseline set ran knob-free
    assert any("--no-batching" in a for a in child_args)
