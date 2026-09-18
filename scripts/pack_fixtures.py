#!/usr/bin/env python3
"""Pack the four M1 parity fixtures into ONE fixtures.zip (K6 dataset input).

Layout contract (pinned by tests/test_m2_stage3_k6.py): each case dir sits at
the zip TOP LEVEL as <case>/manifest.json + <case>/probs.f32. Kaggle may
present the mounted dataset extracted OR as this zip; the K6 runner's
find_fixtures_dir() accepts both — but either way the case dirs must be
reachable, so no wrapper folder here.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

CASES = [
    "v12-short-streaming-r0",
    "v12-mid-offline-full-r0",
    "v13-mid-streaming-r0",
    "v13-mid-offline-preset-r0",
]
FILES = ("manifest.json", "probs.f32")


def pack(fix_root: Path, out: Path) -> Path:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in CASES:
            for fname in FILES:
                f = fix_root / name / fname
                if not f.exists():
                    raise FileNotFoundError(f)
                z.write(f, f"{name}/{fname}")
    return out


def main() -> int:
    fix_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("parity/fixtures")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("fixtures.zip")
    pack(fix_root, out)
    print(f"{out}: {out.stat().st_size} bytes, {len(CASES)} cases x {len(FILES)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
