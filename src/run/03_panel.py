"""03 — plant-level panel, eligibility, and primary-scope list."""

from __future__ import annotations

import csv
from datetime import datetime, timezone

from config.protocol import PRIMARY_GROUP_IDS
from src.contracts import StageContext, StageResult
from src.io import artifact, dump_json
from src.lib import panel
from src.run._ledger import ledger_patch


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    result = panel.run(ctx)

    members_path = ctx.root / "artifacts/twins/members.csv"
    plants = []
    if members_path.is_file():
        with members_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("group_id") in PRIMARY_GROUP_IDS:
                    plants.append(row["plant_id"])
    plants = sorted(set(plants))
    scope = {
        "primary_exact_twin_groups": list(PRIMARY_GROUP_IDS),
        "primary_plants": plants,
        "primary_target_count": len(plants),
        "gself_structural_executability": [{"plant_id": p, "pathological_three_block": False} for p in plants],
        "noncomputable_cs4_transfers": [],
    }
    out = ctx.root / "artifacts/scope/PRIMARY_SCOPE.json"
    dump_json(out, scope)

    recs = list(result.artifacts)
    if out.is_file():
        recs.append(artifact(ctx.root, out))
    payload = {
        "status": result.status,
        "panel": "artifacts/p02c/P02C_PLANT_PANEL.parquet",
        "primary_plants": plants,
        "primary_groups": list(PRIMARY_GROUP_IDS),
        "scope": "artifacts/scope/PRIMARY_SCOPE.json",
    }
    if result.status != "GO":
        return StageResult(
            status=result.status,
            message=result.message,
            sot_patch={"results": {"panel": {"status": result.status, "message": result.message, "finished_at_utc": now}}},
            artifacts=[],
            validations=result.validations,
        )
    return StageResult(
        status=result.status,
        message=result.message,
        sot_patch=ledger_patch(stage="panel", now=now, payload=payload, recs=recs),
        artifacts=recs,
        validations=result.validations,
    )
