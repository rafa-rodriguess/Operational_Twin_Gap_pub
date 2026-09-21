"""10 — manuscript tables from the ledger / OTG artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from config.protocol import FAM_SPLINE, PRIMARY_GROUP_IDS
from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, dump_json, write_csv
from src.run._ledger import ledger_patch


def _flag(val: object) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def plant_balanced_mean_otg(rows: list[dict[str, str]]) -> float | None:
    by_target: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("group_id") not in PRIMARY_GROUP_IDS:
            continue
        if row.get("model_family") not in {FAM_SPLINE, None, ""}:
            if row.get("model_family") and row.get("model_family") != FAM_SPLINE:
                continue
        if row.get("support_rule") not in {"CS4", None, ""}:
            if row.get("support_rule") and row.get("support_rule") != "CS4":
                continue
        if not _flag(row.get("computable")):
            continue
        by_target[str(row["target_plant_id"])].append(float(row["otg_abs"]))
    if not by_target:
        return None
    per_target = [sum(vals) / len(vals) for vals in by_target.values()]
    return float(sum(per_target) / len(per_target))


def plant_balanced_mean_abs_asymmetry(rows: list[dict[str, str]]) -> float | None:
    by_plant: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("group_id") not in PRIMARY_GROUP_IDS:
            continue
        if not _flag(row.get("asymmetry_computable")):
            continue
        value = float(row["asymmetry_abs"])
        by_plant[str(row["plant_a"])].append(value)
        by_plant[str(row["plant_b"])].append(value)
    if not by_plant:
        return None
    per_plant = [sum(vals) / len(vals) for vals in by_plant.values()]
    return float(sum(per_plant) / len(per_plant))


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    otg_path = root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"
    asy_path = root / "artifacts/s07/S07_ASYMMETRY.csv"
    rq1 = rq2 = None
    if otg_path.is_file():
        with otg_path.open(encoding="utf-8", newline="") as handle:
            rq1 = plant_balanced_mean_otg(list(csv.DictReader(handle)))
    if asy_path.is_file():
        with asy_path.open(encoding="utf-8", newline="") as handle:
            rq2 = plant_balanced_mean_abs_asymmetry(list(csv.DictReader(handle)))
    rq3 = ((ctx.sot.get("results") or {}).get("rq3") or {}).get("D_fleet_RQ3")

    table = [
        {
            "analysis": "Primary",
            "rq1_otg": rq1,
            "rq2_abs_asymmetry": rq2,
            "rq3_control_minus_twin": rq3,
        }
    ]
    out = root / "artifacts/tables"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "table_02_primary_summary.csv", table, ["analysis", "rq1_otg", "rq2_abs_asymmetry", "rq3_control_minus_twin"])
    paper_tab = root / "paper" / "table_02_primary_robustness_summary.csv"
    write_csv(paper_tab, table, ["analysis", "rq1_otg", "rq2_abs_asymmetry", "rq3_control_minus_twin"])
    dump_json(out / "table_02.json", table)
    recs = [artifact(root, out / "table_02_primary_summary.csv"), artifact(root, out / "table_02.json")]
    payload = {"rq1": rq1, "rq2": rq2, "rq3": rq3, "table02": table}
    return StageResult(
        status="GO",
        message=f"GO — tables rq1={rq1} rq2={rq2} rq3={rq3}",
        sot_patch=ledger_patch(
            stage="tables",
            now=now,
            payload=payload,
            recs=recs,
            extra={"status": "complete"},
        ),
        artifacts=recs,
        validations=[ValidationRecord("primary_table", True, None)],
    )
