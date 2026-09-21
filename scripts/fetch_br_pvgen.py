#!/usr/bin/env python3
"""Download official BR-PVGen files into data/raw/br_pvgen and verify MD5."""

from __future__ import annotations

import hashlib
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.protocol import DOI, RAW_DIR, RAW_FILES, ZENODO_MD5

RECORD = "21511487"
BASE = f"https://zenodo.org/records/{RECORD}/files"


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        return ctx


def download(name: str, dest: Path) -> None:
    url = f"{BASE}/{name}?download=1"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"GET {name}")
    curl = subprocess.run(
        ["curl", "-L", "--fail", "--retry", "5", "-A", "otg-new-repo/1.0", "-o", str(tmp), url],
        check=False,
    )
    if curl.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
        tmp.replace(dest)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "otg-new-repo/1.0"})
    try:
        resp_cm = urllib.request.urlopen(req, context=_ssl_context())
    except ssl.SSLError:
        ctx = ssl._create_unverified_context()
        resp_cm = urllib.request.urlopen(req, context=ctx)
    with resp_cm as resp, tmp.open("wb") as handle:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(dest)


def main() -> int:
    dest_dir = ROOT / RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"DOI {DOI}")
    print(f"dest {dest_dir}")
    failures: list[str] = []
    for name in RAW_FILES:
        path = dest_dir / name
        expected = ZENODO_MD5[name]
        if path.is_file() and md5_file(path) == expected:
            print(f"OK  {name} (already present)")
            continue
        try:
            download(name, path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: download failed: {exc}")
            continue
        got = md5_file(path)
        if got != expected:
            failures.append(f"{name}: md5 {got} != {expected}")
        else:
            print(f"OK  {name} {path.stat().st_size} bytes")
    if failures:
        print("FAIL", file=sys.stderr)
        for line in failures:
            print(line, file=sys.stderr)
        return 1
    print("BR-PVGen ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
