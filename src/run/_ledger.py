"""Remap lib StageResult patches into the freeze-free SoT ledger."""

from __future__ import annotations

from typing import Any

from config.protocol import PROTOCOL_ID
from src.contracts import ArtifactRecord, StageResult
from src.io import tab
from src.sot import sha256_file
from src.paths import PROTOCOL


def ledger_patch(
    *,
    stage: str,
    now: str,
    payload: dict[str, Any],
    recs: list[ArtifactRecord],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "status": "partial",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(PROTOCOL),
        "results": {stage: payload},
        "stages": {
            stage: {
                "kind": "scientific",
                "status": "GO",
                "finished_at_utc": now,
            }
        },
        "artifacts": {stage: {rec.path: tab(rec) for rec in recs}},
    }
    if extra:
        patch.update(extra)
    return patch


def adopt(result: StageResult, stage: str, now: str, payload: dict[str, Any]) -> StageResult:
    """Keep artifacts/validations/status from a lib run; replace the SoT patch."""
    return StageResult(
        status=result.status,
        message=result.message,
        sot_patch=ledger_patch(stage=stage, now=now, payload=payload, recs=result.artifacts)
        if result.status == "GO"
        else {"results": {stage: {"status": result.status, "message": result.message, "finished_at_utc": now}}},
        artifacts=result.artifacts,
        validations=result.validations,
    )
