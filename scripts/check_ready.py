#!/usr/bin/env python3
"""Verify new_repo is ready for `python orchestrator.py run --from-stage 01 --to-stage 10`."""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.protocol import RAW_DIR, RAW_FILES, ZENODO_MD5
from orchestrator import discover_all_stage_ids


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []
    raw = ROOT / RAW_DIR
    for name in RAW_FILES:
        path = raw / name
        if not path.is_file():
            errors.append(f"missing {path.relative_to(ROOT)}")
            continue
        got = md5_file(path)
        if got != ZENODO_MD5[name]:
            errors.append(f"md5 mismatch {name}: {got} != {ZENODO_MD5[name]}")
        else:
            print(f"OK  data {name} ({path.stat().st_size} bytes)")

    ids = discover_all_stage_ids(ROOT)
    expected = [f"{i:02d}" for i in range(1, 11)]
    if ids != expected:
        errors.append(f"stages {ids} != {expected}")
    else:
        print("OK  stages 01-10")

    for mod in ("numpy", "sklearn", "duckdb", "pyarrow", "joblib", "arch"):
        try:
            importlib.import_module(mod)
            print(f"OK  import {mod}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"import {mod}: {exc}")

    sot = ROOT / "artifacts" / "sot.json"
    if not sot.is_file():
        errors.append("missing artifacts/sot.json")
    else:
        print("OK  artifacts/sot.json")

    if errors:
        print("NOT READY")
        for line in errors:
            print(" -", line)
        print("If BR-PVGen is missing: python scripts/fetch_br_pvgen.py")
        return 1
    print("READY  python orchestrator.py run --from-stage 01 --to-stage 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
