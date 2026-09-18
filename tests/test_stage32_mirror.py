# Stage 3.2b: C++ assembly vs numpy mirror (reference/mirror_sortformer.py).
# Builds/locates tools/dump_forward, generates deterministic tiny-weight dumps
# for two scenarios, and runs the fp64 mirror comparison at the 1e-4 gate.
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIRROR = REPO / "reference" / "mirror_sortformer.py"
SOURCES = ["tools/dump_forward.cpp", "src/gguf.cpp", "src/sortformer.cpp",
    "src/subsampling.cpp", "src/nn.cpp", "src/layers.cpp", "src/mha.cpp",
    "src/conv.cpp", "src/conformer.cpp", "src/posenc.cpp"]

def _find_or_build() -> Path:
    for cand in [REPO / "build" / "diar_dump_forward", REPO / "build" / "tools" / "diar_dump_forward",
                 REPO / ".local-build" / "diar_dump_forward"]:
        if cand.exists():
            return cand
    out = Path(tempfile.mkdtemp()) / "diar_dump_forward"
    cmd = ["g++", "-std=c++17", "-O2", "-I", str(REPO / "include"), "-I",
           str(REPO / "tests")] + [str(REPO / s) for s in SOURCES] + ["-o", str(out)]
    subprocess.run(cmd, check=True)
    return out

def _run_scenario(exe, args):
    out = Path(tempfile.mkdtemp())
    subprocess.run([str(exe), "--out", str(out)] + args, check=True)
    r = subprocess.run([sys.executable, str(MIRROR), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"mirror mismatch:\n{r.stdout}\n{r.stderr}"
    assert "PASS" in r.stdout, r.stdout

def test_masked_tail_scenario():
    # t_mel=86 (non-multiple of 8), feat_len=80 -> NeMo masked fill rows
    _run_scenario(_find_or_build(), ["--t-mel", "86", "--feat-len", "80",
                                     "--spkcache", "5", "--fifo", "3"])

def test_full_round_scenario():
    # t_mel=160 with larger state, production tail mode (feat_len=-1)
    _run_scenario(_find_or_build(), ["--t-mel", "160", "--feat-len", "-1",
                                     "--spkcache", "12", "--fifo", "6"])
