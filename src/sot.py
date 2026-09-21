"""SoT helpers. Stages must not call write functions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.paths import SOT

EMPTY: dict[str, Any] = {
    "status": "empty",
    "protocol_id": None,
    "protocol_sha256": None,
    "results": {},
    "artifacts": {},
    "stages": {},
    "tables": {},
    "figures": {},
}


def _path(path: Path | None = None) -> Path:
    return Path(path) if path else SOT


def load(path: Path | None = None) -> dict[str, Any]:
    target = _path(path)
    if not target.exists():
        return json.loads(json.dumps(EMPTY))
    return json.loads(target.read_text(encoding="utf-8"))


def snapshot(data: dict[str, Any] | None = None, path: Path | None = None) -> dict[str, Any]:
    if data is None:
        data = load(path)
    return deepcopy(data)


def apply_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    _merge(out, patch)
    return out


def _merge(dst: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        existing = dst.get(key)
        if (
            isinstance(value, dict)
            and isinstance(existing, dict)
            and {"kind", "status"}.issubset(value.keys())
        ):
            dst[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(existing, dict):
            _merge(existing, value)
        else:
            dst[key] = value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sha256_sot(path: Path | None = None) -> str:
    target = _path(path)
    if not target.exists():
        return sha256_bytes(b"")
    return sha256_file(target)


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2) + "\n"


def atomic_write(data: dict[str, Any], path: Path | None = None) -> None:
    """Orchestrator-only."""
    target = _path(path)
    text = canonical_json(data)
    json.loads(text)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    json.loads(target.read_text(encoding="utf-8"))
