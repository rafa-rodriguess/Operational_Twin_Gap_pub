"""08 — RQ3: same-state matching, pairwise CS4 intersection, control−twin contrast."""

from __future__ import annotations

from datetime import datetime, timezone

from config.protocol import (
    CONTRAST_ORIENTATION,
    FLEET_SYNTHESIS,
    PRIMARY_GROUP_IDS,
    SUPPORT_ALIGNMENT,
    TARGET_AGGREGATION,
)
from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact
from src.lib.rq3_science import (
    compute_pairwise_contrast,
    load_meta,
    select_same_state_controls,
    write_rq3_artifacts,
)
from src.run._ledger import ledger_patch


def _read_members(ctx: StageContext) -> list[dict[str, str]]:
    import csv

    members_path = ctx.root / "artifacts/twins/members.csv"
    if not members_path.is_file():
        members_path = ctx.root / "artifacts/p02b/P02B_TWIN_MEMBERS.csv"
    with members_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle)), members_path


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    meta_path = root / "data/raw/br_pvgen/BR-PVGen_metadata.csv"
    try:
        members, members_path = _read_members(ctx)
    except FileNotFoundError:
        members, members_path = [], None
    if not meta_path.is_file() or not members:
        return StageResult(
            status="STOP",
            message="RQ3 needs metadata and twin members",
            sot_patch={"results": {"rq3": {"status": "STOP", "finished_at_utc": now}}},
            artifacts=[],
            validations=[ValidationRecord("inputs", False, "metadata or members missing")],
        )
    del members_path

    meta = load_meta(meta_path)
    selected, excluded, _group_of, members_by_group = select_same_state_controls(meta, members)
    if not selected:
        return StageResult(
            status="STOP",
            message="RQ3: no unique same-state controls",
            sot_patch={"results": {"rq3": {"status": "STOP", "finished_at_utc": now, "excluded": excluded}}},
            artifacts=[],
            validations=[ValidationRecord("same_state_mapping", False, f"excluded={len(excluded)}")],
        )
    contrast = compute_pairwise_contrast(root, selected, members_by_group)
    if contrast.get("status") == "STOP":
        return StageResult(
            status="STOP",
            message=str(contrast.get("reason") or "RQ3 contrast incomplete"),
            sot_patch={"results": {"rq3": {"status": "STOP", "finished_at_utc": now, **contrast}}},
            artifacts=[],
            validations=[ValidationRecord("pairwise", False, contrast.get("reason"))],
        )

    out = root / "artifacts/rq3"
    write_rq3_artifacts(out, selected, excluded, contrast)
    recs = [artifact(root, p) for p in sorted(out.glob("*")) if p.is_file()]
    d_fleet = contrast.get("D_fleet_RQ3")
    if selected and d_fleet is None:
        return StageResult(
            status="STOP",
            message="RQ3 mapping exists but no computable pairwise contrasts",
            sot_patch={"results": {"rq3": {"status": "STOP", "finished_at_utc": now, "n_rq3_targets": len(selected)}}},
            artifacts=recs,
            validations=[ValidationRecord("pairwise_intersection", False, "D_fleet is None")],
        )
    payload = {
        "n_primary_plants": sum(1 for r in members if r.get("group_id") in PRIMARY_GROUP_IDS),
        "n_rq3_targets": len(selected),
        "n_excluded": len(excluded),
        "excluded": excluded,
        "selected_controls": contrast.get("selected_controls"),
        "support_alignment": SUPPORT_ALIGNMENT,
        "contrast_orientation": CONTRAST_ORIENTATION,
        "target_aggregation": TARGET_AGGREGATION,
        "fleet_synthesis": FLEET_SYNTHESIS,
        "D_fleet_RQ3": d_fleet,
        "n_computable_targets": contrast.get("n_computable_targets"),
    }
    return StageResult(
        status="GO",
        message=f"GO — RQ3 pairwise n_targets={len(selected)} D_fleet={d_fleet}",
        sot_patch=ledger_patch(stage="rq3", now=now, payload=payload, recs=recs),
        artifacts=recs,
        validations=[
            ValidationRecord("same_state_mapping", bool(selected), f"n={len(selected)}"),
            ValidationRecord("pairwise_intersection", d_fleet is not None, f"D_fleet={d_fleet}"),
        ],
    )
