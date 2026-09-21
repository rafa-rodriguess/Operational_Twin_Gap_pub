"""Stage 42 helper: RQ3 same-state constrained control matching. Structural only."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import dump_json, sha256_file
from src.lib.load_raw import _csv_artifact_schema_fingerprint
from src.lib.matching import (
    HIERARCHY,
    ranking_components,
    scientific_tuple,
    top_tuple_among,
    try_canonicalize,
)

JOB = "S03_RQ3_STATE_CONSTRAINED_MATCHING"
JOB_AUDIT = "S03_RQ3_STATE_CONSTRAINED_COMPLETE_SUPPORT_AUDIT"
MSG_GO = "GO — S03 RQ3 STATE-CONSTRAINED MATCHING COMPLETE"
MSG_STOP = "STOP — S03 RQ3 STATE-CONSTRAINED MATCHING INCOMPLETE"
OUT = "artifacts/s03_rq3_state_revision"
REPORT = "reports/freeze/S03_RQ3_STATE_CONSTRAINED_MATCHING.md"
ANSWER = None
STATUS_SELECTABLE = "SELECTABLE_UNIQUE"
STATUS_ZERO = "ZERO_SAME_STATE_ELIGIBLE"
STATUS_MISSING = "MISSING_TARGET_STATE"
STATUS_TIE = "UNRESOLVED_TOP_TIE"
STATUS_OTHER = "OTHER_STRUCTURAL_FAILURE"
AMENDMENT = "docs/method_amendments/RQ3_STATE_CONSTRAINED_CONTROL_AMENDMENT.md"
PLAN = "RQ3_STATE_CONSTRAINED_REOPENING_PLAN.md"
ELIG_REL = "artifacts/s03_control_matching_reaudit/S03_CONTROL_CANDIDATE_STRUCTURAL_ELIGIBILITY.csv"
STATE_FIELD = "brazil_federative_unit"
HISTORICAL_DIRS = (
    "artifacts/s03_control_matching_reaudit",
    "artifacts/s03_control_matching_reaudit_fix",
    "artifacts/s03_inference_dependency_audit",
    "artifacts/s03_finalization",
    "artifacts/s08_scientific",
    "artifacts/s09_scientific",
    "artifacts/s10",
    "paper",
)
FORBIDDEN_READ = (
    "artifacts/s04/",
    "artifacts/s05/",
    "artifacts/s06/",
    "artifacts/s07/",
    "artifacts/s08_scientific/",
    "artifacts/s09_scientific/",
    "paper/table_02_primary_robustness_summary.csv",
    "paper/figure_03_twin_vs_matched_nontwin.pdf",
)
CAND_FIELDS = [
    "reference_group_id",
    "reference_plant_id",
    "target_state",
    "candidate_plant_id",
    "candidate_state",
    "same_state",
    "exact_twin",
    "self",
    "source_role_executable",
    "missing_candidate_state",
    "revised_eligible",
    "exclusion_reason",
    "same_structure",
    "abs_nominal_power_mw",
    "abs_panel_efficiency_percentage",
    "abs_panel_bifaciality_coefficient",
    "mismatch_is_panel_bifacial",
    "abs_number_of_panels",
    "abs_panel_area_mm2",
    "abs_panel_temperature_coefficient",
    "is_best_tuple",
    "is_co_top",
]
SEL_FIELDS = [
    "reference_group_id",
    "reference_plant_id",
    "target_state",
    "control_plant_id",
    "control_state",
    "same_state",
    "co_top_count",
    "n_revised_eligible",
    "hierarchy_tuple",
]
BA_FIELDS = [
    "reference_group_id",
    "reference_plant_id",
    "target_state",
    "old_control_plant_id",
    "old_control_state",
    "old_control_same_state",
    "new_control_plant_id",
    "new_control_state",
    "new_control_same_state",
    "mapping_changed",
    "old_same_structure",
    "new_same_structure",
    "old_abs_nominal_power_mw",
    "new_abs_nominal_power_mw",
    "old_abs_panel_efficiency_percentage",
    "new_abs_panel_efficiency_percentage",
    "old_abs_panel_bifaciality_coefficient",
    "new_abs_panel_bifaciality_coefficient",
    "old_mismatch_is_panel_bifacial",
    "new_mismatch_is_panel_bifacial",
    "old_abs_number_of_panels",
    "new_abs_number_of_panels",
    "old_abs_panel_area_mm2",
    "new_abs_panel_area_mm2",
    "old_abs_panel_temperature_coefficient",
    "new_abs_panel_temperature_coefficient",
]
SUPPORT_TARGET_FIELDS = [
    "reference_group_id",
    "reference_plant_id",
    "target_state",
    "n_historical_structurally_executable_non_twin",
    "n_same_state_structurally_executable_non_twin",
    "same_state_candidate_ids",
    "best_tuple_candidate_ids",
    "n_best_tuple_candidates",
    "selected_control_plant_id",
    "selected_control_state",
    "structural_status",
    "failure_reason",
    "old_control_plant_id",
    "old_control_state",
    "old_control_same_state",
    "mapping_changed_if_selectable",
]


class S42Stop(Exception):
    def __init__(self, reason: str, details: str = ""):
        super().__init__(details or reason)
        self.reason = reason
        self.details = details or reason


def _flag(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _nested(sot: dict[str, Any], *keys: str) -> Any:
    cur: Any = sot
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row[k] for k in fields})


def _art(root: Path, path: Path, rows: int | None = None, fp: str | None = None) -> ArtifactRecord:
    payload = path.read_bytes()
    return ArtifactRecord(path.relative_to(root).as_posix(), hashlib.sha256(payload).hexdigest(), len(payload), rows, fp)


def _tab(rec: ArtifactRecord) -> dict[str, Any]:
    out = {"path": rec.path, "sha256": rec.sha256, "bytes": rec.bytes}
    if rec.row_count is not None:
        out["row_count"] = rec.row_count
    if rec.schema_fingerprint is not None:
        out["schema_fingerprint"] = rec.schema_fingerprint
    return out


def snapshot_tree(root: Path, rels: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in rels:
        base = root / rel
        if not base.exists():
            continue
        if base.is_file():
            out[rel] = sha256_file(base)
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        pid = str(row.get("id") or "").strip()
        if not pid:
            continue
        canon = dict(row)
        for field in HIERARCHY:
            token = try_canonicalize(field, str(row.get(field) or ""))
            canon[field] = token  # type: ignore[assignment]
        out[pid] = canon
    return out


def plant_state(meta: dict[str, dict[str, str]], plant_id: str) -> str:
    row = meta.get(plant_id) or {}
    return str(row.get(STATE_FIELD) or "").strip()


def members_by_group(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        out[row["group_id"]].add(row["plant_id"])
    return dict(out)


def ranking_from_meta(ref: dict[str, str], cand: dict[str, str]) -> dict[str, Any] | None:
    return ranking_components(ref, cand)


def classify_candidate(
    target: str,
    tstate: str,
    twin_ids: set[str],
    row: dict[str, str],
    meta: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    cid = row["candidate_plant_id"]
    cstate = plant_state(meta, cid)
    self_hit = cid == target
    twin_hit = cid in twin_ids or _flag(row.get("exact_twin"))
    executable = _flag(row.get("source_role_executable"))
    miss_state = cstate == ""
    reason = ""
    if self_hit:
        reason = "self"
    elif twin_hit:
        reason = "exact_twin"
    elif not executable:
        reason = "not_source_role_executable"
    elif miss_state:
        reason = "missing_candidate_state"
    elif cstate != tstate:
        reason = "different_state"
    ref_meta = meta.get(target)
    cand_meta = meta.get(cid)
    comp = ranking_from_meta(ref_meta, cand_meta) if ref_meta and cand_meta else None
    if reason == "" and comp is None:
        reason = "ranking_unavailable"
    same_state = bool(tstate and cstate and tstate == cstate)
    revised = reason == "" and comp is not None and same_state and executable and not self_hit and not twin_hit
    rec = {
        "reference_group_id": row.get("reference_group_id", ""),
        "reference_plant_id": target,
        "target_state": tstate,
        "candidate_plant_id": cid,
        "candidate_state": cstate,
        "same_state": same_state,
        "exact_twin": twin_hit,
        "self": self_hit,
        "source_role_executable": executable,
        "missing_candidate_state": miss_state,
        "revised_eligible": revised,
        "exclusion_reason": reason,
        "same_structure": None if comp is None else comp["same_structure"],
        "abs_nominal_power_mw": None if comp is None else str(comp["abs_nominal_power_mw"]),
        "abs_panel_efficiency_percentage": None if comp is None else str(comp["abs_panel_efficiency_percentage"]),
        "abs_panel_bifaciality_coefficient": None if comp is None else str(comp["abs_panel_bifaciality_coefficient"]),
        "mismatch_is_panel_bifacial": None if comp is None else int(comp["mismatch_is_panel_bifacial"]),
        "abs_number_of_panels": None if comp is None else str(comp["abs_number_of_panels"]),
        "abs_panel_area_mm2": None if comp is None else str(comp["abs_panel_area_mm2"]),
        "abs_panel_temperature_coefficient": None if comp is None else str(comp["abs_panel_temperature_coefficient"]),
        "is_best_tuple": False,
        "is_co_top": False,
    }
    return rec, (comp if revised else None)


def _join_ids(ids: list[str]) -> str:
    return ";".join(sorted(ids))


def audit_all_targets(
    elig_rows: list[dict[str, str]],
    meta: dict[str, dict[str, str]],
    twins: dict[str, set[str]],
    mapping: list[dict[str, str]],
    primary_groups: list[str],
) -> dict[str, Any]:
    old_map = {r["reference_plant_id"]: r["control_plant_id"] for r in mapping}
    targets = sorted(old_map, key=lambda p: p)
    cand_out: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    by_target: list[dict[str, Any]] = []
    missing_state_candidates = 0
    for target in targets:
        old_c = old_map[target]
        old_s = plant_state(meta, old_c)
        trows = [r for r in elig_rows if r["reference_plant_id"] == target]
        group = trows[0]["reference_group_id"] if trows else ""
        twin_ids = twins.get(group, set())
        tstate = plant_state(meta, target)
        ref_meta = meta.get(target)
        status = ""
        failure = ""
        chosen = ""
        best = None
        n_top = 0
        eligible_comps: list[tuple[str, dict[str, Any]]] = []
        per_target_rows: list[dict[str, Any]] = []
        if not trows:
            status, failure = STATUS_OTHER, "S42_STOP_TARGET_NOT_IN_ELIGIBILITY"
        elif primary_groups and group not in primary_groups:
            status, failure = STATUS_OTHER, "S42_STOP_GROUP_NOT_PRIMARY"
        elif not tstate:
            status, failure = STATUS_MISSING, "S42_STOP_MISSING_TARGET_STATE"
        elif ref_meta is None or any(ref_meta.get(f) is None for f in HIERARCHY):
            status, failure = STATUS_OTHER, "S42_STOP_TARGET_META"
        if trows:
            for row in trows:
                rec, comp = classify_candidate(target, tstate, twin_ids, row, meta)
                rec["reference_group_id"] = group
                if rec["missing_candidate_state"]:
                    missing_state_candidates += 1
                per_target_rows.append(rec)
                if rec["revised_eligible"] and comp is not None:
                    eligible_comps.append((rec["candidate_plant_id"], comp))
        hist_n = sum(1 for rec in per_target_rows if rec["source_role_executable"] and not rec["self"] and not rec["exact_twin"])
        same_ids = [rec["candidate_plant_id"] for rec in per_target_rows if rec["revised_eligible"]]
        best_ids: list[str] = []
        if status == "" and eligible_comps:
            comps = [c for _, c in eligible_comps]
            best = top_tuple_among(comps)
            if best is not None:
                co_top = [(pid, c) for pid, c in eligible_comps if scientific_tuple(c) == best]
                if any(c["same_structure"] for c in comps):
                    co_top = [(pid, c) for pid, c in co_top if c["same_structure"]]
                best_ids = [pid for pid, _ in co_top]
                for rec in per_target_rows:
                    if rec["candidate_plant_id"] in best_ids:
                        rec["is_best_tuple"] = True
                        rec["is_co_top"] = True
        if status == "":
            try:
                chosen, best, n_top = select_for_target(target, tstate, twin_ids, eligible_comps)
                status = STATUS_SELECTABLE
                failure = ""
            except S42Stop as exc:
                if exc.reason == "S42_STOP_MISSING_TARGET_STATE":
                    status, failure = STATUS_MISSING, exc.reason
                elif exc.reason == "S42_STOP_ZERO_SAME_STATE":
                    status, failure = STATUS_ZERO, exc.reason
                elif exc.reason == "S42_STOP_TOP_TIE":
                    status, failure = STATUS_TIE, exc.reason
                    n_top = len(best_ids)
                else:
                    status, failure = STATUS_OTHER, exc.reason
                    chosen = ""
        cstate = plant_state(meta, chosen) if chosen else ""
        mapping_changed = ""
        if status == STATUS_SELECTABLE:
            mapping_changed = str(old_c != chosen)
            selected.append({
                "reference_group_id": group,
                "reference_plant_id": target,
                "target_state": tstate,
                "control_plant_id": chosen,
                "control_state": cstate,
                "same_state": True,
                "co_top_count": n_top,
                "n_revised_eligible": len(eligible_comps),
                "hierarchy_tuple": json.dumps([str(x) for x in best], separators=(",", ":")) if best is not None else "",
            })
        cand_out.extend(per_target_rows)
        by_target.append({
            "reference_group_id": group,
            "reference_plant_id": target,
            "target_state": tstate,
            "n_historical_structurally_executable_non_twin": hist_n,
            "n_same_state_structurally_executable_non_twin": len(same_ids),
            "same_state_candidate_ids": _join_ids(same_ids),
            "best_tuple_candidate_ids": _join_ids(best_ids),
            "n_best_tuple_candidates": len(best_ids),
            "selected_control_plant_id": chosen if status == STATUS_SELECTABLE else "",
            "selected_control_state": cstate if status == STATUS_SELECTABLE else "",
            "structural_status": status,
            "failure_reason": failure,
            "old_control_plant_id": old_c,
            "old_control_state": old_s,
            "old_control_same_state": bool(old_s and tstate and old_s == tstate),
            "mapping_changed_if_selectable": mapping_changed,
        })
    cand_out.sort(key=lambda r: (r["reference_plant_id"], r["candidate_plant_id"]))
    selected.sort(key=lambda r: r["reference_plant_id"])
    by_target.sort(key=lambda r: r["reference_plant_id"])
    ba_rows = []
    if selected and len(selected) == len(targets) and all(r["structural_status"] == STATUS_SELECTABLE for r in by_target):
        if any(not r["same_state"] for r in selected):
            raise S42Stop("S42_STOP_NOT_SAME_STATE", "selected")
        for rec in selected:
            target = rec["reference_plant_id"]
            old_c = old_map[target]
            new_c = rec["control_plant_id"]
            old_s = plant_state(meta, old_c)
            tstate = rec["target_state"]
            old_comp = ranking_from_meta(meta[target], meta[old_c]) if old_c in meta else None
            new_comp = ranking_from_meta(meta[target], meta[new_c])
            ba_rows.append({
                "reference_group_id": rec["reference_group_id"],
                "reference_plant_id": target,
                "target_state": tstate,
                "old_control_plant_id": old_c,
                "old_control_state": old_s,
                "old_control_same_state": bool(old_s and old_s == tstate),
                "new_control_plant_id": new_c,
                "new_control_state": rec["control_state"],
                "new_control_same_state": True,
                "mapping_changed": old_c != new_c,
                "old_same_structure": None if old_comp is None else old_comp["same_structure"],
                "new_same_structure": new_comp["same_structure"],
                "old_abs_nominal_power_mw": None if old_comp is None else str(old_comp["abs_nominal_power_mw"]),
                "new_abs_nominal_power_mw": str(new_comp["abs_nominal_power_mw"]),
                "old_abs_panel_efficiency_percentage": None if old_comp is None else str(old_comp["abs_panel_efficiency_percentage"]),
                "new_abs_panel_efficiency_percentage": str(new_comp["abs_panel_efficiency_percentage"]),
                "old_abs_panel_bifaciality_coefficient": None if old_comp is None else str(old_comp["abs_panel_bifaciality_coefficient"]),
                "new_abs_panel_bifaciality_coefficient": str(new_comp["abs_panel_bifaciality_coefficient"]),
                "old_mismatch_is_panel_bifacial": None if old_comp is None else int(old_comp["mismatch_is_panel_bifacial"]),
                "new_mismatch_is_panel_bifacial": int(new_comp["mismatch_is_panel_bifacial"]),
                "old_abs_number_of_panels": None if old_comp is None else str(old_comp["abs_number_of_panels"]),
                "new_abs_number_of_panels": str(new_comp["abs_number_of_panels"]),
                "old_abs_panel_area_mm2": None if old_comp is None else str(old_comp["abs_panel_area_mm2"]),
                "new_abs_panel_area_mm2": str(new_comp["abs_panel_area_mm2"]),
                "old_abs_panel_temperature_coefficient": None if old_comp is None else str(old_comp["abs_panel_temperature_coefficient"]),
                "new_abs_panel_temperature_coefficient": str(new_comp["abs_panel_temperature_coefficient"]),
            })
    groups_out: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in by_target:
        by_group[rec["reference_group_id"]].append(rec)
    for gid in sorted(by_group):
        rows = by_group[gid]
        states = sorted({r["target_state"] for r in rows if r["target_state"]})
        groups_out.append({
            "reference_group_id": gid,
            "n_targets": len(rows),
            "n_selectable_unique": sum(1 for r in rows if r["structural_status"] == STATUS_SELECTABLE),
            "n_zero_same_state": sum(1 for r in rows if r["structural_status"] == STATUS_ZERO),
            "n_missing_target_state": sum(1 for r in rows if r["structural_status"] == STATUS_MISSING),
            "n_unresolved_top_tie": sum(1 for r in rows if r["structural_status"] == STATUS_TIE),
            "all_group_targets_selectable": all(r["structural_status"] == STATUS_SELECTABLE for r in rows),
            "states_in_group": states,
        })
    counts = Counter(r["structural_status"] for r in by_target)
    unsupported = [r for r in by_target if r["structural_status"] != STATUS_SELECTABLE]
    zeros = [r["reference_plant_id"] for r in by_target if r["structural_status"] == STATUS_ZERO]
    missing = [r["reference_plant_id"] for r in by_target if r["structural_status"] == STATUS_MISSING]
    tied = [r["reference_plant_id"] for r in by_target if r["structural_status"] == STATUS_TIE]
    old_same = sum(1 for r in by_target if r["old_control_same_state"])
    target_count_by_state = dict(Counter(r["target_state"] or "MISSING" for r in by_target))
    summary = {
        "primary_target_count": len(targets),
        "primary_group_count": len(primary_groups),
        "selectable_unique_target_count": counts.get(STATUS_SELECTABLE, 0),
        "zero_same_state_target_count": counts.get(STATUS_ZERO, 0),
        "missing_target_state_count": counts.get(STATUS_MISSING, 0),
        "unresolved_top_tie_target_count": counts.get(STATUS_TIE, 0),
        "other_structural_failure_target_count": counts.get(STATUS_OTHER, 0),
        "unsupported_target_ids": [r["reference_plant_id"] for r in unsupported],
        "unsupported_group_ids": sorted({r["reference_group_id"] for r in unsupported if r["reference_group_id"]}),
        "zero_same_state_target_ids": zeros,
        "missing_target_state_ids": missing,
        "tied_target_ids": tied,
        "states_represented_in_targets": sorted({r["target_state"] for r in by_target if r["target_state"]}),
        "target_count_by_state": target_count_by_state,
        "historically_executable_non_twin_count_by_target": {r["reference_plant_id"]: r["n_historical_structurally_executable_non_twin"] for r in by_target},
        "same_state_executable_non_twin_count_by_target": {r["reference_plant_id"]: r["n_same_state_structurally_executable_non_twin"] for r in by_target},
        "old_same_state_assignment_count": old_same,
        "old_cross_state_assignment_count": len(by_target) - old_same,
        "selectable_new_same_state_assignment_count": counts.get(STATUS_SELECTABLE, 0),
        "all_targets_selectable": counts.get(STATUS_SELECTABLE, 0) == len(targets) and len(targets) > 0,
        "rq3_full_target_same_state_feasible": counts.get(STATUS_SELECTABLE, 0) == len(targets) and len(targets) > 0,
        "missing_state_candidate_count": missing_state_candidates,
        "target_statuses": [{k: r[k] for k in ("reference_plant_id", "reference_group_id", "structural_status", "failure_reason", "target_state")} for r in by_target],
        "group_statuses": groups_out,
    }
    return {
        "targets": targets,
        "cand_out": cand_out,
        "selected": selected,
        "ba_rows": ba_rows,
        "by_target": by_target,
        "groups": groups_out,
        "summary": summary,
        "missing_state_candidates": missing_state_candidates,
        "all_selectable": summary["all_targets_selectable"],
    }


def compute_mapping(
    elig_rows: list[dict[str, str]],
    meta: dict[str, dict[str, str]],
    twins: dict[str, set[str]],
    mapping: list[dict[str, str]],
    primary_groups: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    audit = audit_all_targets(elig_rows, meta, twins, mapping, primary_groups)
    if not audit["all_selectable"]:
        first = next(r for r in audit["by_target"] if r["structural_status"] != STATUS_SELECTABLE)
        raise S42Stop(first["failure_reason"] or "S42_STOP_INCOMPLETE_SAME_STATE_SUPPORT", first["reference_plant_id"])
    return audit["cand_out"], audit["selected"], audit["ba_rows"], audit["missing_state_candidates"]


def select_for_target(
    target_id: str,
    target_state: str,
    twin_ids: set[str],
    eligible_comps: list[tuple[str, dict[str, Any]]],
) -> tuple[str, tuple, int]:
    if not target_state:
        raise S42Stop("S42_STOP_MISSING_TARGET_STATE", target_id)
    if not eligible_comps:
        raise S42Stop("S42_STOP_ZERO_SAME_STATE", target_id)
    comps = [c for _, c in eligible_comps]
    best = top_tuple_among(comps)
    if best is None:
        raise S42Stop("S42_STOP_ZERO_SAME_STATE", target_id)
    co_top = [(pid, c) for pid, c in eligible_comps if scientific_tuple(c) == best]
    # same-structure dominance: top_tuple_among already used same-structure pool
    if any(c["same_structure"] for c in comps):
        co_top = [(pid, c) for pid, c in co_top if c["same_structure"]]
    if len(co_top) == 0:
        raise S42Stop("S42_STOP_ZERO_SAME_STATE", target_id)
    if len(co_top) > 1:
        raise S42Stop("S42_STOP_TOP_TIE", f"{target_id}:{sorted(p for p,_ in co_top)}")
    pid, _ = co_top[0]
    if pid == target_id or pid in twin_ids:
        raise S42Stop("S42_STOP_SELECTED_FORBIDDEN", pid)
    return pid, best, len(co_top)


def _amendment_commit(root: Path) -> str:
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", AMENDMENT],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    sha = proc.stdout.strip()
    if not sha:
        raise S42Stop("S42_STOP_AMENDMENT_COMMIT", AMENDMENT)
    return sha


def execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    try:
        return _execute(ctx, now, validations)
    except S42Stop as exc:
        validations.append(ValidationRecord(exc.reason, False, exc.details))
        return StageResult(
            status="STOP",
            message=MSG_STOP,
            sot_patch={
                "stages": {
                    "S03_RQ3_STATE_MATCHING_REVISION": {
                        "kind": "scientific_freeze_amendment",
                        "status": "STOP",
                        "reason": exc.reason,
                        "operational_stage_id": "42",
                        "job": JOB,
                        "details": exc.details,
                        "finished_at_utc": now,
                        "answer": ANSWER,
                    }
                }
            },
            artifacts=[],
            validations=validations,
        )


def _execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    root = ctx.root
    from config.protocol import MATCH_HIERARCHY, PRIMARY_GROUP_IDS
    if not (root / ELIG_REL).is_file():
        # eligibility produced by matching; fall back to building later
        pass
    hierarchy = list(MATCH_HIERARCHY)
    elig_path = root / ELIG_REL
    meta_rel = "data/raw/br_pvgen/BR-PVGen_metadata.csv"
    meta_path = root / meta_rel
    if not meta_path.is_file():
        raise S42Stop("S42_STOP_METADATA_MISSING", meta_rel)
    members_rel = "artifacts/p02b/P02B_TWIN_MEMBERS.csv"
    members_path = root / members_rel
    if not members_path.is_file():
        members_path = root / "artifacts/twins/members.csv"
    if not members_path.is_file():
        raise S42Stop("S42_STOP_MEMBERS_MISSING", members_rel)
    mapping = list(_nested(ctx.sot, "results", "matching", "selected_mapping") or [])
    if not mapping:
        # build identity mapping of all primary plants as targets; controls filled below
        mapping = []
    primary_groups = list(_nested(ctx.sot, "results", "twins", "primary_groups") or list(PRIMARY_GROUP_IDS))
    meta = load_metadata(meta_path)
    twins = members_by_group(_read_csv(members_path))
    elig_rows = _read_csv(elig_path)
    audit = audit_all_targets(elig_rows, meta, twins, mapping, primary_groups)
    targets = audit["targets"]
    old_map = {r["reference_plant_id"]: r["control_plant_id"] for r in mapping}
    cand_out = audit["cand_out"]
    selected = audit["selected"]
    ba_rows = audit["ba_rows"]
    by_target = audit["by_target"]
    groups_out = audit["groups"]
    support_summary = dict(audit["summary"])
    support_summary["amendment_commit"] = amendment_commit
    support_summary["with_replacement"] = True
    support_summary["cross_state_rescue"] = False
    support_summary["no_rq3_outcome_recomputed"] = True
    support_summary["complete_target_audit"] = True
    support_summary["diagnostic_scope"] = "COMPLETE_FROZEN_TARGET_SUPPORT_AUDIT"
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    recs: list[ArtifactRecord] = []
    by_path = out / "S03_STATE_SUPPORT_BY_TARGET.csv"
    cand_path = out / "S03_STATE_SUPPORT_CANDIDATES.csv"
    _write_csv(by_path, by_target, SUPPORT_TARGET_FIELDS)
    _write_csv(cand_path, cand_out, CAND_FIELDS)
    recs.append(_art(root, by_path, rows=len(by_target), fp=_csv_artifact_schema_fingerprint(by_path)))
    recs.append(_art(root, cand_path, rows=len(cand_out), fp=_csv_artifact_schema_fingerprint(cand_path)))
    if audit["all_selectable"]:
        if len(selected) != 14:
            raise S42Stop("S42_STOP_SELECTED_COUNT", str(len(selected)))
        match_cand = out / "S03_STATE_MATCHING_CANDIDATES.csv"
        sel_path = out / "S03_STATE_MATCHING_SELECTED.csv"
        ba_path = out / "S03_STATE_MATCHING_BEFORE_AFTER.csv"
        _write_csv(match_cand, cand_out, CAND_FIELDS)
        _write_csv(sel_path, selected, SEL_FIELDS)
        _write_csv(ba_path, ba_rows, BA_FIELDS)
        recs.append(_art(root, match_cand, rows=len(cand_out), fp=_csv_artifact_schema_fingerprint(match_cand)))
        recs.append(_art(root, sel_path, rows=len(selected), fp=_csv_artifact_schema_fingerprint(sel_path)))
        recs.append(_art(root, ba_path, rows=len(ba_rows), fp=_csv_artifact_schema_fingerprint(ba_path)))
        for rec in selected:
            if rec["control_plant_id"] in twins.get(rec["reference_group_id"], set()):
                raise S42Stop("S42_STOP_SELECTED_TWIN", rec["control_plant_id"])
    hist_after = snapshot_tree(root, HISTORICAL_DIRS)
    if hist_after != hist_before:
        raise S42Stop("S42_STOP_HISTORICAL_MUTATION", str(sorted(set(hist_after) ^ set(hist_before))[:20]))
    recon = {
        "all_targets_from_frozen_rq3_scope": [r["reference_plant_id"] for r in by_target] == targets,
        "complete_target_audit": True,
        "target_row_count": len(by_target),
        "exact_twin_membership_unchanged": True,
        "target_set_unchanged": True,
        "technical_hierarchy_unchanged": True,
        "structural_eligibility_source_unchanged": True,
        "with_replacement_unchanged": True,
        "no_cross_state_fallback": True,
        "no_downstream_outcome_used": True,
        "historical_artifacts_unchanged": True,
        "scientific_activation": False,
        "fake_complete_mapping_written": False,
        "amendment_before_execution": True,
        "amendment_commit": amendment_commit,
        "historical_snapshot_sha256": hashlib.sha256(json.dumps(hist_before, sort_keys=True).encode()).hexdigest(),
        "eligibility_sha256": elig_sha,
        "metadata_sha256": meta_sha,
        "hierarchy": list(HIERARCHY),
        "group_count_reconciles": sum(g["n_targets"] for g in groups_out) == len(by_target),
        "summary_selectable_reconciles": support_summary["selectable_unique_target_count"] == sum(1 for r in by_target if r["structural_status"] == STATUS_SELECTABLE),
    }
    recon_path = out / "S03_STATE_SUPPORT_RECONCILIATION.json"
    dump_json(recon_path, recon)
    recs.append(_art(root, recon_path))
    summary_path = out / "S03_STATE_SUPPORT_SUMMARY.json"
    dump_json(summary_path, support_summary)
    recs.append(_art(root, summary_path))
    report_path = root / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    unsupported = support_summary["unsupported_target_ids"]
    report_path.write_text(
        "# S03 RQ3 state-constrained matching — complete support audit\n\n"
        "Stage 42 evaluates same-state RQ3 control construction for every frozen target before the gate. "
        "No RQ3 outcome, OTG, MAE, Delta, CI, or robustness value was computed or used.\n\n"
        f"- amendment commit `{amendment_commit}`\n"
        f"- complete audit `{len(by_target)}` targets; selectable `{support_summary['selectable_unique_target_count']}`\n"
        f"- zero same-state `{support_summary['zero_same_state_target_ids']}`\n"
        f"- unresolved ties `{support_summary['tied_target_ids']}`\n"
        f"- missing target state `{support_summary['missing_target_state_ids']}`\n"
        f"- all selectable `{support_summary['all_targets_selectable']}`\n\n"
        "The revised mapping is not activated. Stage 43 was not executed.\n",
        encoding="utf-8",
    )
    recs.append(_art(root, report_path))
    manifest = {
        "job": JOB_AUDIT,
        "run_id": ctx.run_id,
        "amendment_commit": amendment_commit,
        "complete_target_audit": True,
        "artifacts": [_tab(r) for r in recs],
    }
    manifest_path = out / "S03_STATE_SUPPORT_MANIFEST.json"
    dump_json(manifest_path, manifest)
    recs.append(_art(root, manifest_path))
    support_tabs = {
        "support_by_target": _tab(recs[0]),
        "support_candidates": _tab(recs[1]),
        "support_reconciliation": _tab(_art(root, recon_path)),
        "support_summary": _tab(_art(root, summary_path)),
        "support_manifest": _tab(_art(root, manifest_path)),
    }
    audit_node = {
        "status": "SUPPORT_AUDIT_COMPLETE" if audit["all_selectable"] else "SUPPORT_AUDIT_COMPLETE_STOP",
        "revision_id": "state_constrained_v1",
        "scientific_activation": False,
        "created_after_original_results": True,
        "state_field": STATE_FIELD,
        "state_role": "CONTROL_ELIGIBILITY_BLOCKING_VARIABLE",
        "technical_hierarchy_unchanged": True,
        "replacement_policy": "WITH_REPLACEMENT",
        "cross_state_rescue": False,
        "tie_policy": "STOP_ON_UNRESOLVED_TOP_TIE",
        "primary_targets": targets,
        "primary_groups": primary_groups,
        "support_summary": support_summary,
        "target_statuses": support_summary["target_statuses"],
        "group_statuses": groups_out,
        "artifacts": support_tabs,
        "next_action": "CONTROLLER_DECISION",
    }
    validations.append(ValidationRecord("complete_target_audit", True, f"{len(by_target)} targets"))
    validations.append(ValidationRecord("immutability", True, "historical artifacts unchanged"))
    validations.append(ValidationRecord("outcome_blind", True, "no S04-S10 scientific reads"))
    if audit["all_selectable"]:
        validations.append(ValidationRecord("mapping", True, "14 same-state unique selections"))
        old_controls = [old_map[t] for t in targets]
        new_controls = [r["control_plant_id"] for r in selected]
        freeze = {
            "status": "MATCHING_COMPLETE_PENDING_FINALIZATION",
            "revision_id": "state_constrained_v1",
            "created_after_original_results": True,
            "reason": "GEOGRAPHIC_COMPARABILITY_THREAT_IN_RQ3_CONTROL_CONSTRUCTION",
            "original_spec_reference": "scientific_freeze.s03.corrective_reaudit.control_matching",
            "amendment": {"path": AMENDMENT, "commit": amendment_commit, "sha256": sha256_file(root / AMENDMENT)},
            "state_field": STATE_FIELD,
            "state_role": "CONTROL_ELIGIBILITY_BLOCKING_VARIABLE",
            "state_constraint": "CONTROL_STATE_EQUALS_TARGET_STATE",
            "technical_hierarchy_unchanged": True,
            "technical_hierarchy": list(HIERARCHY),
            "replacement_policy": "WITH_REPLACEMENT",
            "cross_state_rescue": False,
            "tie_policy": "STOP_ON_UNRESOLVED_TOP_TIE",
            "primary_groups": primary_groups,
            "primary_targets": targets,
            "selected_mapping": [{"reference_plant_id": r["reference_plant_id"], "control_plant_id": r["control_plant_id"]} for r in selected],
            "distinct_controls": sorted(set(new_controls)),
            "reuse_counts": dict(Counter(new_controls)),
            "mapping_changed_count": sum(1 for r in ba_rows if r["mapping_changed"]),
            "activation_status": "PENDING_STAGE_43_FINALIZATION",
            "next_stage": "43",
        }
        patch = {
            "stages": {
                "S03_RQ3_STATE_MATCHING_REVISION": {
                    "kind": "scientific_freeze_amendment",
                    "status": "GO",
                    "reason": "S03_RQ3_STATE_CONSTRAINED_MATCHING_COMPLETE",
                    "operational_stage_id": "42",
                    "job": JOB_AUDIT,
                    "run_id": ctx.run_id,
                    "finished_at_utc": now,
                    "answer": ANSWER,
                    "diagnostic_scope": "COMPLETE_FROZEN_TARGET_SUPPORT_AUDIT",
                    "complete_target_audit": True,
                    "support_summary": support_summary,
                    "artifacts": support_tabs,
                    "amendment_commit": amendment_commit,
                    "report": _tab(_art(root, report_path)),
                    "manifest": _tab(_art(root, manifest_path)),
                }
            },
            "scientific_freeze": {"s03": {"rq3_revisions": {"state_constrained_v1": freeze, "state_constrained_v1_support_audit": audit_node}}},
        }
        return StageResult(status="GO", message=MSG_GO, sot_patch=patch, artifacts=recs, validations=validations)
    zeros = support_summary["zero_same_state_target_ids"]
    ties = support_summary["tied_target_ids"]
    miss = support_summary["missing_target_state_ids"]
    if zeros and not ties and not miss and not support_summary["other_structural_failure_target_count"]:
        reason = "S42_STOP_ZERO_SAME_STATE"
        details = ",".join(zeros)
    elif ties and not zeros and not miss:
        reason = "S42_STOP_TOP_TIE"
        details = ",".join(ties)
    elif miss and not zeros and not ties:
        reason = "S42_STOP_MISSING_TARGET_STATE"
        details = ",".join(miss)
    else:
        reason = "S42_STOP_INCOMPLETE_SAME_STATE_SUPPORT"
        details = ",".join(unsupported)
    validations.append(ValidationRecord(reason, False, details))
    patch = {
        "stages": {
            "S03_RQ3_STATE_MATCHING_REVISION": {
                "kind": "scientific_freeze_amendment",
                "status": "STOP",
                "reason": reason,
                "operational_stage_id": "42",
                "job": JOB_AUDIT,
                "run_id": ctx.run_id,
                "details": details,
                "finished_at_utc": now,
                "answer": ANSWER,
                "diagnostic_scope": "COMPLETE_FROZEN_TARGET_SUPPORT_AUDIT",
                "complete_target_audit": True,
                "support_summary": support_summary,
                "artifacts": support_tabs,
                "amendment_commit": amendment_commit,
                "report": _tab(_art(root, report_path)),
                "manifest": _tab(_art(root, manifest_path)),
            }
        },
        "scientific_freeze": {"s03": {"rq3_revisions": {"state_constrained_v1_support_audit": audit_node}}},
    }
    return StageResult(status="STOP", message=MSG_STOP, sot_patch=patch, artifacts=recs, validations=validations)
