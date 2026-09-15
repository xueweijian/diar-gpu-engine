"""Local tests for the v9 probdump path (no GPU/CUDA/Kaggle/git).

Run:  python3 -m pytest tests/test_probdump.py -q

The probdump patch gives the v9 verdict its input: per-run frame
probabilities dumped by the patched nemo-speech binary, compared here with
no tolerance games (bit-identical short-circuit, else float64 max/mean +
argmax agreement). These tests pin:

1. patch application: helper/call-site land at unique anchors, idempotent,
   loud on anchor drift (RuntimeError, never silent no-dump);
2. patch/report contract: the injected C++ block is char-identical to
   tests/probdump_oracle.cpp (which compiles standalone and prints
   PROBDUMP_ORACLE_PASS — enforced by run_local_tests.sh);
3. probdump binary format roundtrip via struct (little-endian header +
   row-major f32), incl. truncated/corrupt rejection;
4. probdiff semantics: identical / sub-hysteresis drift / argmax flip /
   shape mismatch;
5. diarize_once dump plumbing: dump_probs_path -> child env DIAR_DUMP_PROBS,
   stale dump removed pre-run, probs_sha256 recorded, env record unpolluted.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import diar_harness as harness  # noqa: E402


@pytest.fixture()
def tmp_work(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(harness, "LOG_PATH", tmp_path / "work" / "run.log")
    monkeypatch.setattr(harness, "OUT_DIR", tmp_path / "out")
    harness.WORK_ROOT.mkdir(parents=True, exist_ok=True)
    harness.OUT_DIR.mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_probdump(path: Path, n_frames: int, n_spk: int, values: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<q", n_frames))
        handle.write(struct.pack("<i", n_spk))
        handle.write(struct.pack(f"<{len(values)}f", *values))
    return path


def _fake_repo(tmp_path: Path, body: str) -> Path:
    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "diarize.cpp").write_text(body, encoding="utf-8")
    return repo


_MINIMAL_CPP = (
    "#include <cstdio>\n"
    "namespace {\n"
    "struct Worker { void join(); };\n"
    "}  // namespace\n"
    "int command_diarize() {\n"
    "    for (auto& worker : worker_threads) worker.join();\n"
    "    written_in_this_run[i] = true;\n"
    "    return 0;\n"
    "}\n"
)


# --- patch application ----------------------------------------------------

def test_apply_probdump_patch_inserts_helper_and_call_site(tmp_path):
    repo = _fake_repo(tmp_path, _MINIMAL_CPP)
    record = harness.apply_probdump_patch(repo)
    assert record == {
        "patched": True, "already_patched": False,
        "file": str(repo / "app" / "diarize.cpp"),
    }
    text = (repo / "app" / "diarize.cpp").read_text(encoding="utf-8")
    assert text.count("diar_probdump_write") >= 2  # def + call
    assert "#include <cstdint>" in text
    assert "#include <cstdlib>" in text
    assert "#include <fstream>" in text
    assert 'std::getenv("DIAR_DUMP_PROBS")' in text
    assert text.count(harness.PROBDUMP_ANCHOR) == 1


def test_apply_probdump_patch_is_idempotent(tmp_path):
    repo = _fake_repo(tmp_path, _MINIMAL_CPP)
    assert harness.apply_probdump_patch(repo)["patched"] is True
    second = harness.apply_probdump_patch(repo)
    assert second == {
        "patched": False, "already_patched": True,
        "file": str(repo / "app" / "diarize.cpp"),
    }
    text = (repo / "app" / "diarize.cpp").read_text(encoding="utf-8")
    assert text.count("static bool diar_probdump_write(") == 1


def test_apply_probdump_patch_fails_loud_on_moved_helper_anchor(tmp_path):
    repo = _fake_repo(tmp_path, "#include <cstdio>\nint f() { return 0; }\n")
    with pytest.raises(RuntimeError, match="helper anchor not unique"):
        harness.apply_probdump_patch(repo)


def test_apply_probdump_patch_fails_loud_on_ambiguous_helper_anchor(tmp_path):
    body = ("#include <cstdio>\n"
            "namespace {\n"
            "}  // namespace\n"
            "}  // namespace\n")
    repo = _fake_repo(tmp_path, body)
    with pytest.raises(RuntimeError, match="helper anchor not unique"):
        harness.apply_probdump_patch(repo)


def test_apply_probdump_patch_fails_loud_on_helper_anchor_inside_function(tmp_path):
    """v10 regression: anchor line unique but at brace depth 2 (inside a
    function) — the patch must refuse instead of injecting a nested
    function definition that cannot compile."""
    body = ("#include <cstdio>\n"
            "int f() {\n"
            "    if (x) { }  // namespace\n"
            "}\n")
    repo = _fake_repo(tmp_path, body)
    with pytest.raises(RuntimeError, match="not at namespace depth"):
        harness.apply_probdump_patch(repo)


def test_apply_probdump_patch_fails_loud_on_moved_join_anchor(tmp_path):
    body = ("#include <cstdio>\n"
            "namespace {\n"
            "}  // namespace\n"
            "void a() { for (auto& w : worker_threads) w.join(); }\n"
            "int f() {\n"
            "    written_in_this_run[i] = true;\n"
            "    for (auto& worker : worker_threads) worker.join();\n"
            "    for (auto& worker : worker_threads) worker.join();\n"
            "    return 0;\n"
            "}\n")
    repo = _fake_repo(tmp_path, body)
    with pytest.raises(RuntimeError, match="join anchor"):
        harness.apply_probdump_patch(repo)


def test_apply_probdump_patch_fails_loud_on_moved_loop_anchor(tmp_path):
    body = ("#include <cstdio>\n"
            "namespace {\n"
            "}  // namespace\n"
            "void a() { for (auto& worker : worker_threads) worker.join(); }\n")
    repo = _fake_repo(tmp_path, body)
    with pytest.raises(RuntimeError, match="loop anchor"):
        harness.apply_probdump_patch(repo)


def test_injected_helper_matches_oracle_char_identical():
    """The C++ in the patch must equal the standalone oracle's block.

    tests/probdump_oracle.cpp compiles with -Werror and prints
    PROBDUMP_ORACLE_PASS; char-identity means the verified bytes are the
    shipped bytes. Update both files together, never one side.
    """
    oracle = (Path(__file__).resolve().parent / "probdump_oracle.cpp"
              ).read_text(encoding="utf-8")
    begin = "// --- BEGIN INJECTED LOGIC (must stay char-identical to the patch) ---"
    end = "// --- END INJECTED LOGIC ---"
    block = oracle.split(begin)[1].split(end)[0].strip("\n")
    assert harness.PROBDUMP_WRITE_HELPER.strip("\n") == block


# --- binary format --------------------------------------------------------

def test_read_probdump_roundtrip(tmp_path):
    values = [0.0, 1.0, 0.5, 0.25, 0.1, 0.2, 0.3, 0.4]
    path = _write_probdump(tmp_path / "a.f32", 2, 4, values)
    n_frames, n_spk, back = harness.read_probdump(path)
    assert (n_frames, n_spk) == (2, 4)
    assert back == pytest.approx(values)
    assert (tmp_path / "a.f32").stat().st_size == 12 + 2 * 4 * 4


def test_read_probdump_rejects_truncated_and_bad_shapes(tmp_path):
    short = tmp_path / "short.f32"
    short.write_bytes(b"\x01\x02\x03")
    with pytest.raises(ValueError, match="short header"):
        harness.read_probdump(short)
    bad = tmp_path / "bad.f32"
    bad.write_bytes(struct.pack("<qi", 0, 4))
    with pytest.raises(ValueError, match="bad shape"):
        harness.read_probdump(bad)
    trunc = tmp_path / "trunc.f32"
    trunc.write_bytes(struct.pack("<qi", 4, 4) + b"\x00" * 8)
    with pytest.raises(ValueError, match="truncated"):
        harness.read_probdump(trunc)
    with pytest.raises(FileNotFoundError):
        harness.read_probdump(tmp_path / "missing.f32")


# --- probdiff ---------------------------------------------------------------

def test_probdiff_bit_identical_short_circuits():
    dump = (10, 4, [0.1 * i for i in range(40)])
    verdict = harness.probdiff(dump, (10, 4, list(dump[2])))
    assert verdict == {
        "shape_equal": True, "frames": 10, "n_spk": 4,
        "bit_identical": True, "max_abs_diff": 0.0, "mean_abs_diff": 0.0,
        "frame_agreement": 1.0,
    }


def test_probdiff_catches_sub_hysteresis_drift():
    """1e-6 drift keeps every argmax but must still report bit_different."""
    base = [0.9, 0.05, 0.03, 0.02] * 5
    drifted = [v + 1e-6 if i % 4 == 0 else v for i, v in enumerate(base)]
    verdict = harness.probdiff((5, 4, base), (5, 4, drifted))
    assert verdict["shape_equal"] is True
    assert verdict["bit_identical"] is False
    assert verdict["max_abs_diff"] == pytest.approx(1e-6, rel=0.1)
    assert verdict["frame_agreement"] == 1.0  # RTTM bodies would hide this


def test_probdiff_reports_argmax_flips():
    base = [0.51, 0.49, 0.0, 0.0] * 4
    flipped = [0.49, 0.51, 0.0, 0.0] * 4
    verdict = harness.probdiff((4, 4, base), (4, 4, flipped))
    assert verdict["bit_identical"] is False
    assert verdict["frame_agreement"] == 0.0
    assert verdict["max_abs_diff"] == pytest.approx(0.02)


def test_probdiff_shape_mismatch_is_not_compared():
    verdict = harness.probdiff((10, 4, [0.0] * 40), (9, 4, [0.0] * 36))
    assert verdict == {"shape_equal": False, "shapes": [(10, 4), (9, 4)]}


# --- diarize_once plumbing ----------------------------------------------------

def test_diarize_once_forwards_dump_path_to_child_env(tmp_work, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(argv, cwd=None, timeout=1800, env=None):
        captured["env"] = dict(env or {})
        return 0, "", 0.01

    monkeypatch.setattr(harness, "run", fake_run)
    monkeypatch.setattr(harness, "REPO_DIR", tmp_work)
    out = tmp_work / "x.rttm"
    dump = tmp_work / "probs" / "x.rep0.f32"
    result = harness.diarize_once(
        Path("/nope/binary"), Path("/nope/a.wav"), out, dump_probs_path=dump)
    assert captured["env"][harness.DIAR_DUMP_PROBS] == str(dump)
    assert result["dump_probs_path"] == str(dump)
    assert result["probs_sha256"] is None  # no binary produced the file
    assert result["env"] == {}  # caller env record stays unpolluted


def test_diarize_once_removes_stale_dump_before_run(tmp_work, monkeypatch):
    seen: list[str] = []

    def fake_run(argv, cwd=None, timeout=1800, env=None):
        # the child would write here; absence at call time proves cleanup
        seen.append("exists" if Path(env[harness.DIAR_DUMP_PROBS]).exists() else "clean")
        Path(env[harness.DIAR_DUMP_PROBS]).write_bytes(b"fresh")
        return 0, "", 0.01

    monkeypatch.setattr(harness, "run", fake_run)
    monkeypatch.setattr(harness, "REPO_DIR", tmp_work)
    out = tmp_work / "y.rttm"
    dump = tmp_work / "y.rep0.f32"
    dump.write_bytes(b"stale-dump-from-previous-run")
    result = harness.diarize_once(
        Path("/nope/binary"), Path("/nope/a.wav"), out, dump_probs_path=dump)
    assert seen == ["clean"]
    assert result["probs_sha256"] == harness.sha256(dump)


def test_diarize_once_without_dump_is_unchanged(tmp_work, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(argv, cwd=None, timeout=1800, env=None):
        captured["env"] = dict(env or {})
        return 0, "", 0.01

    monkeypatch.setattr(harness, "run", fake_run)
    monkeypatch.setattr(harness, "REPO_DIR", tmp_work)
    out = tmp_work / "z.rttm"
    result = harness.diarize_once(
        Path("/nope/binary"), Path("/nope/a.wav"), out,
        env={"CUDA_LAUNCH_BLOCKING": "1"})
    assert harness.DIAR_DUMP_PROBS not in captured["env"]
    assert result["dump_probs_path"] is None
    assert result["probs_sha256"] is None
    assert result["env"] == {"CUDA_LAUNCH_BLOCKING": "1"}
