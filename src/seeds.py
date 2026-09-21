"""SHA-256 uint32 seeds used by the moving-block bootstrap."""

from __future__ import annotations

import hashlib

from config.protocol import SEED_PREFIX


def seed_uint32(estimand: str, model_family: str, unit_id: str, arm_id: str) -> int:
    key = f"{SEED_PREFIX}|{estimand}|{model_family}|{unit_id}|{arm_id}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[0:4], byteorder="big", signed=False)
