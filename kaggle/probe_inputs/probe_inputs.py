"""Cheap probe: what exactly does Kaggle mount for this kernel's dataset_sources?

Run without a GPU so it returns in about a minute. Its only job is to make the
/kaggle/input layout observable, so the measurement kernel does not have to be
re-run (16 min build + 45 min matrix) to discover a mounting problem.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

print("python:", sys.version.split()[0], flush=True)

ROOT = Path("/kaggle/input")
if not ROOT.exists():
    print("MISSING /kaggle/input", flush=True)
else:
    print(f"{ROOT} exists", flush=True)
    for entry in sorted(ROOT.iterdir()):
        kind = "dir" if entry.is_dir() else "file"
        try:
            children = sorted(p.name for p in entry.iterdir()) if entry.is_dir() else []
        except OSError as exc:
            children = [f"<unreadable: {exc}>"]
        print(f"  [{kind}] {entry.name} :: {children[:8]}", flush=True)

print("\n-- all wav/mp3 under /kaggle/input --", flush=True)
found = 0
for pattern in ("*.wav", "*.mp3", "*.flac"):
    for path in sorted(ROOT.rglob(pattern)):
        found += 1
        size = path.stat().st_size
        print(f"  {path} ({size} bytes)", flush=True)
print(f"total audio-ish files: {found}", flush=True)

print("\n-- env hints --", flush=True)
for key in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "KAGGLE_DATA_PROXY_TOKEN"):
    print(f"  {key}: {'set' if os.environ.get(key) else 'unset'}", flush=True)
