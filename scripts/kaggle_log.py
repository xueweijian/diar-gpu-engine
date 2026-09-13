#!/usr/bin/env python3
"""Fetch a Kaggle kernel's log through the API and print it decoded.

Usage: kaggle_log.py <owner/kernel> [--tail N]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request


def main() -> int:
    ref = sys.argv[1]
    tail = 0
    if "--tail" in sys.argv:
        tail = int(sys.argv[sys.argv.index("--tail") + 1])
    token = os.environ["KAGGLE_API_TOKEN"]
    url = f"https://www.kaggle.com/api/v1/kernels/output/list/{ref}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    entries = json.loads(payload["logNullable"]) if payload.get("logNullable") else []
    text = "".join(e.get("data", "") for e in entries)
    if tail:
        text = text[-tail:]
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
