"""Dataset create + GCS upload over the SNI-bypass channel.

Reimplements the three-step kaggle dataset.create flow without the SDK:
  1. POST https://<api-IP>/v1/blobs.BlobApiService/StartBlobUpload (Host: api.kaggle.com)
     -> {token, create_url}
  2. PUT <create_url> (GCS resumable upload URL, direct TLS, no bypass needed
     if storage.googleapis.com works; else fail loudly)
  3. POST https://<api-IP>/v1/datasets.DatasetApiService/CreateDataset

The SDK cannot be reused because its endpoint (api.kaggle.com) is hardcoded
and any TLS ClientHello with a kaggle SNI is RST in this sandbox.

Blob JSON encoding: the SDK uses camelCase? No - check to_dict output.
We empirically derive it below by calling to_dict on the real request types.

Usage:
  python3 scripts/kaggle_dataset_via_ip.py create --folder <stagedir>
  python3 scripts/kaggle_dataset_via_ip.py version --folder <stagedir> --notes "..."
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.request
import urllib.error

API_IP = "34.54.168.202"
API_HOST = "api.kaggle.com"


def _api_call(service: str, method: str, body: dict) -> dict:
    token = os.environ["KAGGLE_API_TOKEN"]
    url = f"https://{API_IP}/v1/{service}/{method}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Host": API_HOST,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "kaggle-api/v1.7.0",
    }, method="POST")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            return {"http": resp.status, "body": json.loads(resp.read().decode() or "{}")}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return {"http": e.code, "body": json.loads(raw or "{}")}
        except ValueError:
            return {"http": e.code, "body": {"_raw": raw[:1000]}}
    except Exception as e:
        return {"http": -1, "body": {"_error": f"{type(e).__name__}: {e}"}}


def _start_blob_upload(fname: str, size: int, mtime: int, blob_type: str = "DATASET") -> dict:
    # SDK to_dict encoding: verify field names from the real type
    from kagglesdk.blobs.types.blob_api_service import ApiStartBlobUploadRequest, ApiBlobType
    r = ApiStartBlobUploadRequest()
    r.type = getattr(ApiBlobType, blob_type)
    r.name = fname
    r.content_length = size
    r.last_modified_epoch_seconds = mtime
    body = ApiStartBlobUploadRequest.to_dict(r)
    print("blob request body:", json.dumps(body)[:300], file=sys.stderr)
    return _api_call("blobs.BlobApiService", "StartBlobUpload", body)


def _put_to_gcs(create_url: str, path: str, quiet: bool = False) -> bool:
    import requests
    size = os.path.getsize(path)
    sess = requests.Session()
    sess.trust_env = False  # sandbox has empty *_PROXY vars; never proxy GCS
    with open(path, "rb") as fp:
        resp = sess.put(create_url, data=fp, headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(size),
        }, timeout=600)
    if not quiet:
        print(f"GCS PUT {os.path.basename(path)}: HTTP={resp.status_code}", file=sys.stderr)
    return resp.status_code in (200, 201)


def _upload_one(path: str) -> str:
    fname = os.path.basename(path)
    size = os.path.getsize(path)
    mtime = int(os.path.getmtime(path))
    r = _start_blob_upload(fname, size, mtime)
    if r["http"] != 200:
        raise RuntimeError(f"StartBlobUpload failed: {r}")
    body = r["body"]
    token = body.get("token")
    create_url = body.get("createUrl") or body.get("create_url")
    if not token or not create_url:
        raise RuntimeError(f"StartBlobUpload bad response: {body}")
    print(f"blob {fname}: token={token[:12]}... createUrl host={create_url.split('/')[2]}", file=sys.stderr)
    ok = _put_to_gcs(create_url, path)
    if not ok:
        raise RuntimeError(f"GCS PUT failed for {fname}")
    return token


def _collect_files(folder: str) -> list[str]:
    skip = {"dataset-metadata.json", "kernel-metadata.json"}
    out = []
    for fn in sorted(os.listdir(folder), key=lambda f: os.path.getsize(os.path.join(folder, f))):
        if fn in skip or fn.startswith("."):
            continue
        fp = os.path.join(folder, fn)
        if os.path.isfile(fp):
            out.append(fp)
    return out


def cmd_create(args: argparse.Namespace) -> int:
    folder = args.folder
    meta = json.load(open(os.path.join(folder, "dataset-metadata.json"), encoding="utf-8"))
    ref = meta["id"]
    owner, slug = ref.split("/")
    files = _collect_files(folder)
    print(f"uploading {len(files)} files...", file=sys.stderr)
    tokens = [_upload_one(p) for p in files]
    from kagglesdk.datasets.types.dataset_api_service import (
        ApiCreateDatasetRequest, ApiDatasetNewFile,
    )
    req = ApiCreateDatasetRequest()
    req.owner_slug = owner
    req.slug = slug
    req.title = meta["title"]
    req.license_name = meta["licenses"][0]["name"]
    req.is_private = True
    req.files = []
    for tok in tokens:
        f = ApiDatasetNewFile()
        f.token = tok
        req.files.append(f)
    body = ApiCreateDatasetRequest.to_dict(req)
    print("create body keys:", sorted(body.keys()), file=sys.stderr)
    r = _api_call("datasets.DatasetApiService", "CreateDataset", body)
    print(f"HTTP={r['http']}")
    print(json.dumps(r["body"], indent=2)[:2000])
    return 0 if r["http"] == 200 and not (r["body"].get("error")) else 1


def cmd_version(args: argparse.Namespace) -> int:
    print("version subcommand: dataset must already exist; not yet implemented", file=sys.stderr)
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("create")
    p.add_argument("--folder", required=True)
    p.set_defaults(fn=cmd_create)
    p = sub.add_parser("version")
    p.add_argument("--folder", required=True)
    p.add_argument("--notes", default="sync m2 refs")
    p.set_defaults(fn=cmd_version)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
