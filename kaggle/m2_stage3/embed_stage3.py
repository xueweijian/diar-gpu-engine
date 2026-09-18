"""Embed gate scripts + C++ sources into the single code_file kernel payload.

Kaggle script kernels upload ONLY code_file, so m2_stage3_run.py carries:
  EMBEDDED_M2_STAGE3_K5A        m2_stage3_k5a.py source (string literal)
  EMBEDDED_M2_STAGE3_F32WRITER  f32bin_writer.py source (string literal)
  EMBEDDED_CPP_BLOBS_B64        {repo-relative path: base64(source)} for
                                CPP_SOURCES (+ all include/diar headers)

Usage (BEFORE every push):
    python3 kaggle/m2_stage3/embed_stage3.py           # embed
    python3 kaggle/m2_stage3/embed_stage3.py --check   # verify == sources

Round-trip is pinned by tests/test_m2_stage3_embed.py.
"""
from __future__ import annotations

import argparse
import ast as _ast
import base64
import sys
from pathlib import Path

DIR = Path(__file__).parent
RUNNER = DIR / "m2_stage3_run.py"

TEXT_ANCHORS = {
    "m2_stage3_k5a.py": "EMBEDDED_M2_STAGE3_K5A",
    "f32bin_writer.py": "EMBEDDED_M2_STAGE3_F32WRITER",
    "m2_stage3_k6.py": "EMBEDDED_M2_STAGE3_K6",
    "m3_stage1_prof.py": "EMBEDDED_M3_PROF",
}

# Must match CPP_SOURCES in m2_stage3_run.py (asserted below at embed time).
CPP_FILES = [
    "tools/k5_runner.cpp",
    "src/sortformer.cpp", "src/engine.cpp", "src/fe.cpp", "src/diar.cpp",
    "src/aosc.cpp", "src/birth_gate.cpp", "src/gguf.cpp", "src/tailfix.cpp",
    "src/nn.cpp", "src/layers.cpp", "src/mha.cpp", "src/conv.cpp",
    "src/conformer.cpp", "src/subsampling.cpp", "src/posenc.cpp",
    "include/diar/conformer.hpp", "include/diar/conv.hpp", "include/diar/diar.hpp",
    "include/diar/engine.hpp", "include/diar/gguf.hpp", "include/diar/layers.hpp",
    "include/diar/mha.hpp", "include/diar/nn.hpp", "include/diar/posenc.hpp",
    "include/diar/sortformer.hpp", "include/diar/subsampling.hpp",
    "include/diar/tailfix.hpp",
    "src/profile.cpp", "include/diar/profile.hpp",
]
REPO = DIR.parent.parent


def _read_anchor(text: str, anchor: str) -> str:
    tree = _ast.parse(text)
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Assign) and node.targets
                and isinstance(node.targets[0], _ast.Name)
                and node.targets[0].id == anchor):
            return _ast.literal_eval(node.value)
    raise RuntimeError(f"anchor {anchor} not found in {RUNNER}")


def cpp_list_from_runner(text: str) -> list[str]:
    tree = _ast.parse(text)
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Assign) and node.targets
                and isinstance(node.targets[0], _ast.Name)
                and node.targets[0].id == "CPP_SOURCES"):
            return list(_ast.literal_eval(node.value))
    raise RuntimeError("CPP_SOURCES not found")


def build_embedded(runner_text: str) -> tuple[dict[str, str], dict[str, str]]:
    texts: dict[str, str] = {}
    for fname, anchor in TEXT_ANCHORS.items():
        texts[anchor] = (DIR / fname).read_text(encoding="utf-8")
    blobs: dict[str, str] = {}
    cpp_files = cpp_list_from_runner(runner_text)
    if cpp_files != CPP_FILES:
        raise RuntimeError(
            f"CPP_SOURCES drift between runner and embed script: "
            f"{set(cpp_files) ^ set(CPP_FILES)}")
    for rel in CPP_FILES:
        blobs[rel] = base64.b64encode((REPO / rel).read_bytes()).decode("ascii")
    return texts, blobs


def replace_anchor(src: str, anchor: str, py_literal: str) -> str:
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(f"{anchor} = ") or line.startswith(f"{anchor}:"):
            indent = line[: len(line) - len(line.lstrip())]
            body = py_literal if "\n" not in py_literal else py_literal
            lines[i] = f"{anchor} = {body}"
            return "\n".join(lines)
    raise RuntimeError(f"anchor line {anchor} not found")


def embed() -> None:
    runner_text = RUNNER.read_text(encoding="utf-8")
    texts, blobs = build_embedded(runner_text)
    for anchor, value in texts.items():
        runner_text = replace_anchor(runner_text, anchor, repr(value))
    runner_text = replace_anchor(runner_text, "EMBEDDED_CPP_BLOBS_B64", repr(blobs))
    RUNNER.write_text(runner_text, encoding="utf-8")
    print(f"embedded: {list(texts)} + {len(blobs)} cpp blobs "
          f"({sum(len(v) for v in blobs.values())} b64 chars)")


def check() -> int:
    runner_text = RUNNER.read_text(encoding="utf-8")
    texts, blobs = build_embedded(runner_text)
    ok = True
    for anchor, value in texts.items():
        got = _read_anchor(runner_text, anchor)
        if got != value:
            ok = False
            print(f"DRIFT: {anchor} embedded != source "
                  f"({len(got)} vs {len(value)} chars)")
    got_blobs = _read_anchor(runner_text, "EMBEDDED_CPP_BLOBS_B64")
    if got_blobs != blobs:
        ok = False
        diff = set(got_blobs) ^ set(blobs)
        print(f"DRIFT: EMBEDDED_CPP_BLOBS_B64 embedded != sources "
              f"({'paths ' + str(diff) if diff else 'content'})")
    print("embed check:", "ok" if ok else "STALE — run embed_stage3.py")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        return check()
    embed()
    return 0 if check() == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
