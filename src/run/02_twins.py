"""02 — exact nominal twin groups from the eight-field technical key."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from config.protocol import PRIMARY_GROUP_IDS, SENSITIVITY_A_GROUP_ID, NON_EVALUABLE_GROUP_ID
from src.contracts import StageContext, StageResult
from src.lib import twins
from src.run._ledger import adopt
from src.sot import sha256_file


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    result = twins.run(ctx)
    src_members = ctx.root / "artifacts/p02b/P02B_TWIN_MEMBERS.csv"
    src_groups = ctx.root / "artifacts/p02b/P02B_TWIN_GROUPS.csv"
    twins_dir = ctx.root / "artifacts/twins"
    twins_dir.mkdir(parents=True, exist_ok=True)
    if src_members.is_file():
        shutil.copy2(src_members, twins_dir / "members.csv")
    if src_groups.is_file():
        shutil.copy2(src_groups, twins_dir / "groups.csv")

    groups = []
    if src_groups.is_file():
        with src_groups.open(encoding="utf-8", newline="") as handle:
            groups = list(csv.DictReader(handle))
    ids = [g.get("group_id") for g in groups]
    payload = {
        "status": result.status,
        "n_groups": len(groups),
        "group_ids": ids,
        "primary_groups": list(PRIMARY_GROUP_IDS),
        "sensitivity_a_group": SENSITIVITY_A_GROUP_ID,
        "non_evaluable_group": NON_EVALUABLE_GROUP_ID,
        "twin_members": {
            "path": "artifacts/twins/members.csv",
            "sha256": sha256_file(twins_dir / "members.csv") if (twins_dir / "members.csv").is_file() else None,
        },
        "twin_groups": {
            "path": "artifacts/twins/groups.csv",
            "sha256": sha256_file(twins_dir / "groups.csv") if (twins_dir / "groups.csv").is_file() else None,
        },
    }
    return adopt(result, "twins", now, payload)
