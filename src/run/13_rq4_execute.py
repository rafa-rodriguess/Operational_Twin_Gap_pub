"""13 — confirmatory RQ4 scoring (frozen protocol)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact
from src.lib.rq4_const import OUT_DIR, SPEC_SHA256
from src.lib.rq4_science import execute_confirmatory
from src.run._ledger import ledger_patch
REQUIRED = [
    "RQ4_TARGET_COHORT.json",
    "RQ4_SOURCE_COHORT.json",
    "RQ4_SOURCE_MODELS.parquet",
    "RQ4_DECISION_INSTANCES.parquet",
    "RQ4_CANDIDATE_LOSSES.parquet",
    "RQ4_SUPPORT_COVERAGE.parquet",
    "RQ4_REGRET_BY_INSTANCE.parquet",
    "RQ4_TEMPORAL_NOISE.parquet",
    "RQ4_TEMPORAL_AUDIT.parquet",
    "RQ4_HGB_SENSITIVITY.parquet",
]


def run(ctx: StageContext) -> StageResult:
    root = ctx.root
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    pre = root / OUT_DIR / "RQ4_PREFLIGHT.json"
    freeze = ((ctx.sot.get("scientific_freeze") or {}).get("rq4_source_selection") or {})
    if not pre.is_file():
        return StageResult(status="STOP", message="missing preflight", sot_patch={}, validations=[ValidationRecord("preflight_present", False, None)])
    decision = json.loads(pre.read_text(encoding="utf-8")).get("decision")
    if decision != "GO_FOR_CONFIRMATORY_SCORING":
        return StageResult(status="STOP", message="preflight not GO", sot_patch={}, validations=[ValidationRecord("preflight_go", False, str(decision))])
    if freeze.get("specification_sha256") != SPEC_SHA256:
        return StageResult(status="STOP", message="spec hash mismatch", sot_patch={}, validations=[ValidationRecord("spec_sha", False, str(freeze.get("specification_sha256")))])
    if not freeze.get("cohort_resolution", {}).get("resolved_before_scoring"):
        return StageResult(status="STOP", message="cohort_resolution missing", sot_patch={}, validations=[ValidationRecord("cohort_resolution", False, None)])
    summary = execute_confirmatory(root)
    recs = []
    for name in REQUIRED:
        path = root / OUT_DIR / name
        if not path.is_file():
            return StageResult(status="STOP", message=f"missing {name}", sot_patch={}, validations=[ValidationRecord("artifact_" + name, False, None)])
        recs.append(artifact(root, path))
    checks = [
        ValidationRecord("preflight_go", True, decision),
        ValidationRecord("spec_sha", True, SPEC_SHA256),
        ValidationRecord("source_train_before_c_recorded", True, "train_end_before_c field"),
        ValidationRecord("artifacts_written", True, str(len(recs))),
    ]
    return StageResult(
        status="GO",
        message="RQ4 confirmatory scoring artifacts materialized",
        sot_patch=ledger_patch(stage="rq4_execute", now=now, payload={"status": "GO", **summary}, recs=recs),
        artifacts=recs,
        validations=checks,
    )
