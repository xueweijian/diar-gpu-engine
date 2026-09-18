#!/usr/bin/env python3
"""Dump full output-file URLs for a kernel session (no truncation).

Usage: python3 scripts/kaggle_dump_urls.py <owner>/<slug> [out.json]

Prints name + url-length per file and writes {"files":[{name,url}...],
"log_tail": ...} to out.json (default: stdout summary only, json to
/tmp/<slug>_urls.json). Companion fetch pattern is documented in the
module docstring of k5a harvesting (retry x9, browser UA, ssl ctx).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_via_ip import _sdk_call  # noqa: E402
from kagglesdk.kernels.types.kernels_api_service import (  # noqa: E402
    ApiListKernelSessionOutputRequest as Req,
)


def dump_urls(ref: str) -> dict:
    owner, slug = ref.split("/", 1)
    token = None
    out = {"files": [], "log_tail": ""}
    while True:
        r = Req()
        r.user_name = owner
        r.kernel_slug = slug
        r.page_size = 50
        if token:
            r.page_token = token
        code, text = _sdk_call(
            "kernels.KernelsApiService", "ListKernelSessionOutput", Req.to_dict(r))
        assert code == 200, (code, text[:300])
        resp = json.loads(text)
        out["log_tail"] = resp.get("log") or ""
        for item in resp.get("files") or []:
            out["files"].append({
                "name": item.get("fileName") or item.get("file_name"),
                "url": item.get("url"),
            })
        token = resp.get("nextPageToken") or resp.get("next_page_token")
        if not token:
            return out


def main() -> int:
    ref = sys.argv[1]
    out_path = (sys.argv[2] if len(sys.argv) > 2
                else f"/tmp/{ref.split('/', 1)[1]}_urls.json")
    out = dump_urls(ref)
    Path(out_path).write_text(json.dumps(out), encoding="utf-8")
    print(f"files={len(out['files'])} log_chars={len(out['log_tail'])} -> {out_path}")
    for f in out["files"]:
        print(f["name"], len(f["url"] or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
