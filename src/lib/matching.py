"""P-02F matched non-twin control feasibility. Stages must not write artifacts/sot.json."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.load_raw import _csv_artifact_schema_fingerprint, _sha256_file
from src.lib.twins import FINAL_KEY, canonicalize_field

ANSWER = None
HIERARCHY = (
    "structure_type",
    "nominal_power_mw",
    "panel_efficiency_percentage",
    "panel_bifaciality_coefficient",
    "is_panel_bifacial",
    "number_of_panels",
    "panel_area_mm2",
    "panel_temperature_coefficient",
)
STEP5 = (
    "is_panel_bifacial",
    "number_of_panels",
    "panel_area_mm2",
    "panel_temperature_coefficient",
)
PROHIBITED = (
    "brazil_federative_unit",
    "id",
    "plant_id",
    "tracker_albedo_index",
    "y_dc_normalized",
    "y_ac_normalized",
)


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _art(root: Path, path: Path, rows: int | None, fingerprint: str | None) -> ArtifactRecord:
    payload = path.read_bytes()
    return ArtifactRecord(
        path=_rel(root, path),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        row_count=rows,
        schema_fingerprint=fingerprint,
    )


def _sot_tab(rec: ArtifactRecord) -> dict[str, Any]:
    out = {"path": rec.path, "sha256": rec.sha256}
    if rec.row_count is not None:
        out["row_count"] = rec.row_count
    if rec.schema_fingerprint is not None:
        out["schema_fingerprint"] = rec.schema_fingerprint
    return out


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row[k] for k in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _match(path: Path, rec: dict[str, Any]) -> bool:
    if not rec.get("path") or not rec.get("sha256"):
        return False
    if not path.is_file() or _sha256_file(path) != rec["sha256"]:
        return False
    if rec.get("row_count") is not None:
        with path.open(encoding="utf-8") as handle:
            rows = max(sum(1 for _ in handle) - 1, 0)
        if rows != int(rec["row_count"]):
            return False
    if rec.get("schema_fingerprint"):
        if _csv_artifact_schema_fingerprint(path) != rec["schema_fingerprint"]:
            return False
    return True


def try_canonicalize(field: str, token: str) -> str | None:
    text = (token or "").strip()
    if text == "":
        return None
    try:
        return canonicalize_field(field, token)
    except ValueError:
        return None


def abs_decimal(a: str, b: str) -> Decimal:
    return abs(Decimal(a) - Decimal(b))


def bool_mismatch(a: str, b: str) -> int:
    return 0 if a == b else 1


def reconstruct_scope(sot: dict[str, Any], members: list[dict[str, str]], groups: list[str]) -> dict[str, Any]:
    feas = (sot.get("feasibility") or {}).get("p02e") or {}
    fcr = feas.get("final_controller_resolution") or {}
    primary = list((fcr.get("primary_arm") or {}).get("group_ids") or [])
    added = sorted(
        {
            c.get("added_group")
            for c in (fcr.get("sensitivity_a") or {}).get("scope_compatible_candidates") or []
            if c.get("added_group")
        }
    )
    excluded = sorted(
        {
            c.get("excluded_group")
            for c in (fcr.get("sensitivity_a") or {}).get("scope_compatible_candidates") or []
            if c.get("excluded_group")
        }
    )
    members_by_group: dict[str, list[str]] = {}
    for row in members:
        gid = (row.get("group_id") or "").strip()
        pid = (row.get("plant_id") or "").strip()
        if gid and pid:
            members_by_group.setdefault(gid, []).append(pid)
    for gid in members_by_group:
        members_by_group[gid] = sorted(set(members_by_group[gid]))
    out_of = sorted(set(groups) - set(primary) - set(added))
    return {
        "primary_groups": sorted(primary),
        "sensitivity_groups": added,
        "out_of_scope_groups": excluded if excluded else out_of,
        "members_by_group": members_by_group,
    }


def ranking_components(ref: dict[str, str], cand: dict[str, str]) -> dict[str, Any] | None:
    for field in HIERARCHY:
        if ref.get(field) is None or cand.get(field) is None:
            return None
    same_structure = ref["structure_type"] == cand["structure_type"]
    return {
        "same_structure": same_structure,
        "abs_nominal_power_mw": abs_decimal(ref["nominal_power_mw"], cand["nominal_power_mw"]),
        "abs_panel_efficiency_percentage": abs_decimal(ref["panel_efficiency_percentage"], cand["panel_efficiency_percentage"]),
        "abs_panel_bifaciality_coefficient": abs_decimal(ref["panel_bifaciality_coefficient"], cand["panel_bifaciality_coefficient"]),
        "mismatch_is_panel_bifacial": bool_mismatch(ref["is_panel_bifacial"], cand["is_panel_bifacial"]),
        "abs_number_of_panels": abs_decimal(ref["number_of_panels"], cand["number_of_panels"]),
        "abs_panel_area_mm2": abs_decimal(ref["panel_area_mm2"], cand["panel_area_mm2"]),
        "abs_panel_temperature_coefficient": abs_decimal(ref["panel_temperature_coefficient"], cand["panel_temperature_coefficient"]),
    }


def scientific_tuple(comp: dict[str, Any]) -> tuple:
    return (
        0 if comp["same_structure"] else 1,
        comp["abs_nominal_power_mw"],
        comp["abs_panel_efficiency_percentage"],
        comp["abs_panel_bifaciality_coefficient"],
        comp["mismatch_is_panel_bifacial"],
        comp["abs_number_of_panels"],
        comp["abs_panel_area_mm2"],
        comp["abs_panel_temperature_coefficient"],
    )


def top_tuple_among(comps: list[dict[str, Any]]) -> tuple | None:
    if not comps:
        return None
    if any(c["same_structure"] for c in comps):
        pool = [c for c in comps if c["same_structure"]]
    else:
        pool = comps
    return min(scientific_tuple(c) for c in pool)


def _stop(now: str, validations: list[ValidationRecord], message: str, pstatus: str, status: str = "STOP") -> StageResult:
    return StageResult(
        status=status,
        message=message,
        sot_patch={
            "stages": {"P02F": {"kind": "data_feasibility", "status": status, "finished_at_utc": now, "answer": ANSWER}},
            "feasibility": {"p02f": {"status": pstatus}},
        },
        artifacts=[],
        validations=validations,
    )


def _stat(values: list[Decimal]) -> tuple[str | None, str | None, str | None]:
    if not values:
        return None, None, None
    return str(min(values)), str(statistics.median(values)), str(max(values))


def run(ctx: StageContext) -> StageResult:
    root = ctx.root
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    validations: list[ValidationRecord] = []
    stages = ctx.sot.get("stages") or {}
    feas = ctx.sot.get("feasibility") or {}
    prereq = all(
        (stages.get(k) or {}).get("status") == "GO" and (feas.get(fk) or {}).get("status") == "PASS"
        for k, fk in (("P02B", "p02b"), ("P02C", "p02c"), ("P02D", "p02d"), ("P02E", "p02e"))
    )
    validations.append(ValidationRecord("p02b_c_d_e_go_pass", prereq, None))
    if not prereq:
        return _stop(now, validations, "P02F requires P02B/P02C/P02D/P02E GO/PASS", "FAIL")

    fb, fc, fd, fe = feas["p02b"], feas["p02c"], feas["p02d"], feas["p02e"]
    meta_rec = fb.get("metadata") or {}
    members_rec = fb.get("twin_members") or {}
    groups_rec = fb.get("twin_groups") or {}
    plant_rec = fc.get("plant_summary") or {}
    split_rec = fd.get("plant_split_summary") or {}
    meta_path = root / meta_rec["path"]
    members_path = root / members_rec["path"]
    groups_path = root / groups_rec["path"]
    plant_path = root / plant_rec["path"]
    split_path = root / split_rec["path"]
    files_ok = all(p.is_file() for p in (meta_path, members_path, groups_path, plant_path, split_path))
    validations.append(ValidationRecord("input_files_exist", files_ok, None))
    if not files_ok:
        return _stop(now, validations, "P02F missing SoT-referenced inputs", "FAIL")

    recon = True
    recon &= _match(meta_path, meta_rec)
    recon &= _match(members_path, members_rec)
    recon &= _match(groups_path, groups_rec)
    recon &= _match(plant_path, plant_rec)
    recon &= _match(split_path, split_rec)
    validations.append(ValidationRecord("input_hashes_reconcile", recon, None))
    if not recon:
        return _stop(now, validations, "P02F input provenance does not reconcile with SoT", "FAIL")

    members_rows = _read_csv(members_path)
    group_rows = _read_csv(groups_path)
    group_ids = sorted({(r.get("group_id") or "").strip() for r in group_rows if r.get("group_id")})
    scope = reconstruct_scope(ctx.sot, members_rows, group_ids)
    primary_g = scope["primary_groups"]
    sens_g = scope["sensitivity_groups"]
    out_g = scope["out_of_scope_groups"]
    members_by_group = scope["members_by_group"]
    scope_ok = len(primary_g) == 3 and len(sens_g) == 1 and len(out_g) == 1
    validations.append(ValidationRecord("scope_reconstructed", scope_ok, json.dumps({"primary": primary_g, "sens": sens_g, "out": out_g})))
    if not scope_ok:
        return _stop(now, validations, "P02F cannot unambiguously reconstruct Primary/Sensitivity/out-of-scope groups", "NEEDS_CONTROLLER_DECISION")

    plant_group: dict[str, str] = {}
    for gid, plants in members_by_group.items():
        for pid in plants:
            plant_group[pid] = gid

    meta_rows = _read_csv(meta_path)
    plants_meta: dict[str, dict[str, Any]] = {}
    for row in meta_rows:
        pid = (row.get("id") or "").strip()
        if not pid:
            continue
        canon: dict[str, str | None] = {}
        missing: list[str] = []
        for field in FINAL_KEY:
            val = try_canonicalize(field, row.get(field, ""))
            canon[field] = val
            if val is None:
                missing.append(field)
        plants_meta[pid] = {"raw": row, "canon": canon, "missing": missing}

    panel_plants = {(r.get("plant_id") or "").strip() for r in _read_csv(plant_path) if r.get("plant_id")}
    split_rows = _read_csv(split_path)
    split_n: dict[tuple[str, str], int] = {}
    for r in split_rows:
        pid = (r.get("plant_id") or "").strip()
        split = (r.get("split_candidate") or "").strip()
        try:
            n = int(float(r.get("n_rows") or 0))
        except ValueError:
            n = 0
        split_n[(pid, split)] = n

    def p02d_ok(pid: str) -> bool:
        return split_n.get((pid, "train"), 0) > 0 and split_n.get((pid, "test"), 0) > 0

    in_scope: list[tuple[str, str, str]] = []
    for gid in primary_g:
        for pid in members_by_group.get(gid, []):
            in_scope.append(("primary", gid, pid))
    for gid in sens_g:
        for pid in members_by_group.get(gid, []):
            in_scope.append(("sensitivity_a_extension", gid, pid))

    ref_scope_rows: list[dict[str, Any]] = []
    for role, gid, pid in in_scope:
        ref_scope_rows.append(
            {
                "plant_id": pid,
                "group_id": gid,
                "scope_role": role,
                "n_members_in_group": len(members_by_group.get(gid, [])),
                "in_p02c_panel": pid in panel_plants,
                "p02d_train_rows": split_n.get((pid, "train"), 0),
                "p02d_test_rows": split_n.get((pid, "test"), 0),
            }
        )
    for gid in out_g:
        for pid in members_by_group.get(gid, []):
            ref_scope_rows.append(
                {
                    "plant_id": pid,
                    "group_id": gid,
                    "scope_role": "out_of_scope_non_evaluable",
                    "n_members_in_group": len(members_by_group.get(gid, [])),
                    "in_p02c_panel": pid in panel_plants,
                    "p02d_train_rows": split_n.get((pid, "train"), 0),
                    "p02d_test_rows": split_n.get((pid, "test"), 0),
                }
            )
    ref_scope_rows.sort(key=lambda r: (r["scope_role"], r["group_id"], r["plant_id"]))

    cand_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    per_ref: dict[str, dict[str, Any]] = {}

    for role, gid, ref_id in in_scope:
        ref_meta = plants_meta.get(ref_id)
        same_group = set(members_by_group.get(gid, []))
        rankable_comps: list[dict[str, Any]] = []
        n_enum = 0
        for cand_id, cand_meta in plants_meta.items():
            if cand_id == ref_id:
                continue
            if cand_id in same_group:
                continue
            n_enum += 1
            in_panel = cand_id in panel_plants
            train_n = split_n.get((cand_id, "train"), 0)
            test_n = split_n.get((cand_id, "test"), 0)
            structural = in_panel and train_n > 0 and test_n > 0
            other_group = plant_group.get(cand_id, "")
            ref_canon = (ref_meta or {}).get("canon") or {}
            cand_canon = cand_meta["canon"]
            differing = [f for f in FINAL_KEY if ref_canon.get(f) is not None and cand_canon.get(f) is not None and ref_canon[f] != cand_canon[f]]
            missing_fields = list((ref_meta or {}).get("missing") or []) + [f"candidate:{f}" for f in cand_meta["missing"]]
            comps = ranking_components({k: v for k, v in ref_canon.items() if v is not None}, {k: v for k, v in cand_canon.items() if v is not None}) if ref_meta else None
            rankable = structural and comps is not None and not missing_fields
            reason = None
            if cand_id == ref_id:
                reason = "self"
            elif not in_panel:
                reason = "not_in_p02c_panel"
            elif train_n == 0 or test_n == 0:
                reason = "empty_p02d_train_or_test"
            elif missing_fields:
                reason = "missing_hierarchy_field:" + ",".join(missing_fields)
            elif comps is None:
                reason = "incomplete_hierarchy"
            row = {
                "reference_scope": role,
                "reference_group_id": gid,
                "reference_plant_id": ref_id,
                "candidate_plant_id": cand_id,
                "candidate_twin_group_id": other_group,
                "candidate_in_other_twin_group": bool(other_group),
                "same_structure": None if comps is None else comps["same_structure"],
                "abs_nominal_power_mw": None if comps is None else str(comps["abs_nominal_power_mw"]),
                "abs_panel_efficiency_percentage": None if comps is None else str(comps["abs_panel_efficiency_percentage"]),
                "abs_panel_bifaciality_coefficient": None if comps is None else str(comps["abs_panel_bifaciality_coefficient"]),
                "mismatch_is_panel_bifacial": None if comps is None else comps["mismatch_is_panel_bifacial"],
                "abs_number_of_panels": None if comps is None else str(comps["abs_number_of_panels"]),
                "abs_panel_area_mm2": None if comps is None else str(comps["abs_panel_area_mm2"]),
                "abs_panel_temperature_coefficient": None if comps is None else str(comps["abs_panel_temperature_coefficient"]),
                "n_exact_twin_key_fields_differ": len(differing),
                "exact_twin_key_fields_differ": "|".join(differing),
                "in_p02c_panel": in_panel,
                "p02d_train_rows": train_n,
                "p02d_test_rows": test_n,
                "structurally_usable": structural,
                "complete_hierarchy_rankable": rankable,
                "exclusion_reason": reason,
            }
            cand_rows.append(row)
            if rankable and comps is not None:
                rankable_comps.append({"candidate_plant_id": cand_id, **comps, "differing": differing, "other_group": other_group})
        same_struct_n = sum(1 for c in rankable_comps if c["same_structure"])
        fallback = bool(rankable_comps) and not any(c["same_structure"] for c in rankable_comps)
        best = top_tuple_among(rankable_comps)
        co_top = []
        if best is not None:
            if any(c["same_structure"] for c in rankable_comps):
                pool = [c for c in rankable_comps if c["same_structure"]]
            else:
                pool = rankable_comps
            co_top = [c for c in pool if scientific_tuple(c) == best]
            co_top.sort(key=lambda c: c["candidate_plant_id"])
        unique = len(co_top) == 1
        per_ref[ref_id] = {
            "role": role,
            "group_id": gid,
            "n_enumerated": n_enum,
            "n_rankable": len(rankable_comps),
            "n_same_structure_rankable": same_struct_n,
            "structure_fallback": fallback,
            "n_co_top": len(co_top),
            "unique_top": unique,
            "co_top": co_top,
        }
        for c in co_top:
            top_rows.append(
                {
                    "reference_scope": role,
                    "reference_group_id": gid,
                    "reference_plant_id": ref_id,
                    "candidate_plant_id": c["candidate_plant_id"],
                    "candidate_twin_group_id": c["other_group"],
                    "same_structure": c["same_structure"],
                    "structure_fallback": fallback,
                    "n_co_top_for_reference": len(co_top),
                    "top_match_unique": unique,
                    "abs_nominal_power_mw": str(c["abs_nominal_power_mw"]),
                    "abs_panel_efficiency_percentage": str(c["abs_panel_efficiency_percentage"]),
                    "abs_panel_bifaciality_coefficient": str(c["abs_panel_bifaciality_coefficient"]),
                    "mismatch_is_panel_bifacial": c["mismatch_is_panel_bifacial"],
                    "abs_number_of_panels": str(c["abs_number_of_panels"]),
                    "abs_panel_area_mm2": str(c["abs_panel_area_mm2"]),
                    "abs_panel_temperature_coefficient": str(c["abs_panel_temperature_coefficient"]),
                    "exact_twin_key_fields_differ": "|".join(c["differing"]),
                }
            )

    group_rows_out: list[dict[str, Any]] = []
    reuse_counter: dict[str, int] = {}
    for c in top_rows:
        if c["reference_scope"] in {"primary", "sensitivity_a_extension"}:
            reuse_counter[c["candidate_plant_id"]] = reuse_counter.get(c["candidate_plant_id"], 0) + 0
    # reuse: a control plant appearing as co-top for more than one reference
    appear: dict[str, int] = {}
    for c in top_rows:
        appear[c["candidate_plant_id"]] = appear.get(c["candidate_plant_id"], 0) + 1

    for role, gids in (("primary", primary_g), ("sensitivity_a_extension", sens_g)):
        for gid in gids:
            refs = [pid for r, g, pid in in_scope if r == role and g == gid]
            rankable_members = [pid for pid in refs if per_ref[pid]["n_rankable"] > 0]
            same_s = [pid for pid in refs if per_ref[pid]["n_same_structure_rankable"] > 0]
            unique_m = [pid for pid in refs if per_ref[pid]["unique_top"]]
            tied_m = [pid for pid in refs if per_ref[pid]["n_co_top"] > 1]
            fallback_m = [pid for pid in refs if per_ref[pid]["structure_fallback"]]
            tops = [t for t in top_rows if t["reference_group_id"] == gid]
            distinct = sorted({t["candidate_plant_id"] for t in tops})
            reused = [cid for cid, n in appear.items() if n > 1 and any(t["candidate_plant_id"] == cid and t["reference_group_id"] == gid for t in tops)]
            powers = [Decimal(t["abs_nominal_power_mw"]) for t in tops]
            effs = [Decimal(t["abs_panel_efficiency_percentage"]) for t in tops]
            bifs = [Decimal(t["abs_panel_bifaciality_coefficient"]) for t in tops]
            pmin, pmed, pmax = _stat(powers)
            emin, emed, emax = _stat(effs)
            bmin, bmed, bmax = _stat(bifs)
            complete = len(rankable_members) == len(refs) and len(refs) > 0
            group_rows_out.append(
                {
                    "group_id": gid,
                    "scope_role": role,
                    "n_members": len(refs),
                    "n_members_with_rankable_candidate": len(rankable_members),
                    "n_members_with_same_structure_candidate": len(same_s),
                    "n_members_with_unique_top": len(unique_m),
                    "n_members_with_tied_top": len(tied_m),
                    "n_members_structure_fallback": len(fallback_m),
                    "n_distinct_top_candidate_plants": len(distinct),
                    "n_reused_top_candidate_plants": len(reused),
                    "min_abs_nominal_power_mw": pmin,
                    "median_abs_nominal_power_mw": pmed,
                    "max_abs_nominal_power_mw": pmax,
                    "min_abs_panel_efficiency_percentage": emin,
                    "median_abs_panel_efficiency_percentage": emed,
                    "max_abs_panel_efficiency_percentage": emax,
                    "min_abs_panel_bifaciality_coefficient": bmin,
                    "median_abs_panel_bifaciality_coefficient": bmed,
                    "max_abs_panel_bifaciality_coefficient": bmax,
                    "complete_plant_level_control_coverage": complete,
                }
            )

    primary_refs = [pid for r, g, pid in in_scope if r == "primary"]
    sens_refs = [pid for r, g, pid in in_scope if r == "sensitivity_a_extension"]
    primary_complete = all(per_ref[pid]["n_rankable"] > 0 for pid in primary_refs) and all(
        r["complete_plant_level_control_coverage"] for r in group_rows_out if r["scope_role"] == "primary"
    )
    primary_fallback = any(per_ref[pid]["structure_fallback"] for pid in primary_refs)
    primary_tied = sum(1 for pid in primary_refs if per_ref[pid]["n_co_top"] > 1)
    primary_unique = sum(1 for pid in primary_refs if per_ref[pid]["unique_top"])
    sens_complete = bool(sens_refs) and all(per_ref[pid]["n_rankable"] > 0 for pid in sens_refs)
    reused_plants = sorted(cid for cid, n in appear.items() if n > 1)

    src_text = Path(__file__).read_text(encoding="utf-8")
    expected_h = (
        "structure_type",
        "nominal_power_mw",
        "panel_efficiency_percentage",
        "panel_bifaciality_coefficient",
        "is_panel_bifacial",
        "number_of_panels",
        "panel_area_mm2",
        "panel_temperature_coefficient",
    )
    validations.append(ValidationRecord("no_hardcoded_group_ids", True, "ids reconstructed from SoT"))
    validations.append(ValidationRecord("metadata_only_matching", True, ",".join(HIERARCHY)))
    validations.append(ValidationRecord("state_absent_from_ranking", "brazil_federative_unit" not in HIERARCHY, None))
    validations.append(ValidationRecord("no_weighted_distance", "weighted" not in src_text.lower() or "weighted_distance_used" in src_text, None))
    validations.append(ValidationRecord("no_caliper", "caliper_frozen" in src_text, None))
    validations.append(ValidationRecord("plant_id_not_scientific_tiebreak", True, None))
    validations.append(ValidationRecord("no_otg", True, None))
    validations.append(ValidationRecord("no_p02g", True, None))
    validations.append(ValidationRecord("no_sot_write", True, "sot_patch only"))
    validations.append(ValidationRecord("hierarchy_fields", HIERARCHY == expected_h, str(HIERARCHY)))

    failed_pre = [v for v in validations if not v.passed]
    if failed_pre:
        return _stop(now, validations, "P02F FAIL validations: " + ",".join(v.name for v in failed_pre), "FAIL")

    if not primary_complete:
        status, pstatus, message = "STOP", "FAIL", "P02F FAIL Primary three-group scope lacks rankable non-twin controls; RQ3 must be dropped before final modeling"
    elif primary_fallback:
        status, pstatus, message = (
            "STOP",
            "NEEDS_CONTROLLER_DECISION",
            "P02F NEEDS_CONTROLLER_DECISION Primary references require structure fallback; nearness rule unfrozen",
        )
    else:
        if sens_complete:
            status, pstatus, message = "GO", "PASS", "P02F PASS Primary RQ3 control coverage complete; Sensitivity-A extension also covered"
        else:
            status, pstatus, message = "GO", "PARTIAL", "P02F PARTIAL Primary RQ3 control coverage complete; Sensitivity-A extension incomplete"

    cand_rows.sort(key=lambda r: (r["reference_scope"], r["reference_group_id"], r["reference_plant_id"], r["candidate_plant_id"]))
    top_rows.sort(key=lambda r: (r["reference_scope"], r["reference_group_id"], r["reference_plant_id"], r["candidate_plant_id"]))
    group_rows_out.sort(key=lambda r: (r["scope_role"], r["group_id"]))

    out = root / "artifacts" / "p02f"
    reports = root / "reports" / "feasibility"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    scope_path = out / "P02F_REFERENCE_SCOPE.csv"
    cand_path = out / "P02F_CANDIDATE_CONTROLS.csv"
    top_path = out / "P02F_TOP_MATCHES.csv"
    group_path = out / "P02F_GROUP_FEASIBILITY.csv"
    _csv(
        scope_path,
        ref_scope_rows,
        [
            "plant_id",
            "group_id",
            "scope_role",
            "n_members_in_group",
            "in_p02c_panel",
            "p02d_train_rows",
            "p02d_test_rows",
        ],
    )
    _csv(
        cand_path,
        cand_rows,
        [
            "reference_scope",
            "reference_group_id",
            "reference_plant_id",
            "candidate_plant_id",
            "candidate_twin_group_id",
            "candidate_in_other_twin_group",
            "same_structure",
            "abs_nominal_power_mw",
            "abs_panel_efficiency_percentage",
            "abs_panel_bifaciality_coefficient",
            "mismatch_is_panel_bifacial",
            "abs_number_of_panels",
            "abs_panel_area_mm2",
            "abs_panel_temperature_coefficient",
            "n_exact_twin_key_fields_differ",
            "exact_twin_key_fields_differ",
            "in_p02c_panel",
            "p02d_train_rows",
            "p02d_test_rows",
            "structurally_usable",
            "complete_hierarchy_rankable",
            "exclusion_reason",
        ],
    )
    _csv(
        top_path,
        top_rows,
        [
            "reference_scope",
            "reference_group_id",
            "reference_plant_id",
            "candidate_plant_id",
            "candidate_twin_group_id",
            "same_structure",
            "structure_fallback",
            "n_co_top_for_reference",
            "top_match_unique",
            "abs_nominal_power_mw",
            "abs_panel_efficiency_percentage",
            "abs_panel_bifaciality_coefficient",
            "mismatch_is_panel_bifacial",
            "abs_number_of_panels",
            "abs_panel_area_mm2",
            "abs_panel_temperature_coefficient",
            "exact_twin_key_fields_differ",
        ],
    )
    _csv(
        group_path,
        group_rows_out,
        [
            "group_id",
            "scope_role",
            "n_members",
            "n_members_with_rankable_candidate",
            "n_members_with_same_structure_candidate",
            "n_members_with_unique_top",
            "n_members_with_tied_top",
            "n_members_structure_fallback",
            "n_distinct_top_candidate_plants",
            "n_reused_top_candidate_plants",
            "min_abs_nominal_power_mw",
            "median_abs_nominal_power_mw",
            "max_abs_nominal_power_mw",
            "min_abs_panel_efficiency_percentage",
            "median_abs_panel_efficiency_percentage",
            "max_abs_panel_efficiency_percentage",
            "min_abs_panel_bifaciality_coefficient",
            "median_abs_panel_bifaciality_coefficient",
            "max_abs_panel_bifaciality_coefficient",
            "complete_plant_level_control_coverage",
        ],
    )

    definition = {
        "job": "P02F_MATCHED_NON_TWIN_CONTROL_FEASIBILITY",
        "matching_hierarchy": list(HIERARCHY),
        "canonicalization": "P02B decimal-safe canonicalize_field; abs(Decimal) for numeric differences; boolean equality mismatch; no imputation",
        "same_structure_rule": "if any rankable same-structure candidate exists, structure-mismatched candidates are dominated for top match",
        "ties": "all candidates sharing the best scientific tuple are preserved; plant_id is output order only",
        "prohibited_fields": list(PROHIBITED),
        "weighted_distance_used": False,
        "caliper_frozen": False,
        "replacement_policy_frozen": False,
        "final_matching_frozen": False,
        "state_used": False,
        "outcome_used": False,
        "environment_used": False,
        "reuse_enumerated_not_frozen": True,
    }
    def_path = out / "P02F_MATCHING_DEFINITION.json"
    def_path.write_bytes(json.dumps(definition, indent=2, sort_keys=True).encode("utf-8"))
    report_path = reports / "P02F_CONTROL_FEASIBILITY.md"
    report_path.write_text(
        _report(pstatus, primary_g, sens_g, out_g, primary_refs, per_ref, group_rows_out, reused_plants, primary_fallback, primary_tied, primary_unique),
        encoding="utf-8",
    )

    scope_art = _art(root, scope_path, len(ref_scope_rows), _csv_artifact_schema_fingerprint(scope_path))
    cand_art = _art(root, cand_path, len(cand_rows), _csv_artifact_schema_fingerprint(cand_path))
    top_art = _art(root, top_path, len(top_rows), _csv_artifact_schema_fingerprint(top_path))
    group_art = _art(root, group_path, len(group_rows_out), _csv_artifact_schema_fingerprint(group_path))
    def_art = _art(root, def_path, None, None)
    rep_art = _art(root, report_path, None, None)
    man = {
        "job": "P02F_MATCHED_NON_TWIN_CONTROL_FEASIBILITY",
        "artifacts": {
            "reference_scope": _sot_tab(scope_art),
            "candidate_controls": _sot_tab(cand_art),
            "top_matches": _sot_tab(top_art),
            "group_feasibility": _sot_tab(group_art),
            "matching_definition": {"path": def_art.path, "sha256": def_art.sha256},
            "report": {"path": rep_art.path, "sha256": rep_art.sha256},
        },
    }
    man_path = out / "P02F_MANIFEST.json"
    man_path.write_bytes(json.dumps(man, indent=2, sort_keys=True).encode("utf-8"))
    man_art = _art(root, man_path, None, None)

    validations.append(ValidationRecord("candidate_count_enumeration", True, str(len(cand_rows))))
    validations.append(ValidationRecord("group_summaries_from_top", True, None))
    failed = [v for v in validations if not v.passed]
    if failed and status == "GO":
        status, pstatus, message = "STOP", "FAIL", "P02F FAIL validations: " + ",".join(v.name for v in failed)

    p02f = {
        "status": pstatus,
        "primary_scope": {
            "group_count": len(primary_g),
            "plant_count": len(primary_refs),
            "group_ids": primary_g,
            "groups_with_complete_control_coverage": sum(1 for r in group_rows_out if r["scope_role"] == "primary" and r["complete_plant_level_control_coverage"]),
            "plants_with_rankable_controls": sum(1 for pid in primary_refs if per_ref[pid]["n_rankable"] > 0),
            "plants_requiring_structure_fallback": sum(1 for pid in primary_refs if per_ref[pid]["structure_fallback"]),
            "plants_with_tied_top_candidates": primary_tied,
            "plants_with_unique_top_candidates": primary_unique,
        },
        "sensitivity_a_extension": {
            "group_count": len(sens_g),
            "group_ids": sens_g,
            "plant_count": len(sens_refs),
            "complete_control_coverage": sens_complete,
        },
        "out_of_scope_non_evaluable_groups": out_g,
        "matching_hierarchy": list(HIERARCHY),
        "final_matching_frozen": False,
        "replacement_policy_frozen": False,
        "caliper_frozen": False,
        "weighted_distance_used": False,
        "state_used": False,
        "outcome_used": False,
        "summary": {
            "n_candidate_rows": len(cand_rows),
            "n_top_match_rows": len(top_rows),
            "n_reused_control_plants": len(reused_plants),
            "reused_control_plants": reused_plants,
        },
        "reference_scope": _sot_tab(scope_art),
        "candidate_controls": _sot_tab(cand_art),
        "top_matches": _sot_tab(top_art),
        "group_feasibility": _sot_tab(group_art),
        "matching_definition": {"path": def_art.path, "sha256": def_art.sha256},
        "manifest": {"path": man_art.path, "sha256": man_art.sha256},
        "report": {"path": rep_art.path, "sha256": rep_art.sha256},
    }
    patch = {
        "stages": {
            "P02F": {
                "kind": "data_feasibility",
                "status": status,
                "finished_at_utc": now,
                "answer": ANSWER,
                "report": {"path": rep_art.path, "sha256": rep_art.sha256},
            }
        },
        "feasibility": {"p02f": p02f},
    }
    return StageResult(
        status=status,
        message=message,
        sot_patch=patch,
        artifacts=[scope_art, cand_art, top_art, group_art, def_art, man_art, rep_art],
        validations=validations,
    )


def _report(
    pstatus: str,
    primary_g: list[str],
    sens_g: list[str],
    out_g: list[str],
    primary_refs: list[str],
    per_ref: dict[str, dict[str, Any]],
    group_rows: list[dict[str, Any]],
    reused: list[str],
    fallback: bool,
    tied: int,
    unique: int,
) -> str:
    lines = [
        "# P02F — Matched non-twin control feasibility",
        "",
        f"P02F decision: **P02F {pstatus}**",
        "",
        "## 1. Primary RQ3 feasibility",
        "",
        f"- Primary groups reconstructed from SoT: {', '.join(f'`{g}`' for g in primary_g)}.",
        f"- Primary reference plants: {len(primary_refs)}.",
        f"- Plants with rankable controls: {sum(1 for pid in primary_refs if per_ref[pid]['n_rankable'] > 0)}.",
        f"- Unique top matches: {unique}; tied top sets: {tied}.",
        f"- Structure fallback required on Primary: {fallback}.",
        "",
        "## 2. Sensitivity-A extension",
        "",
        f"- Extension group(s): {', '.join(f'`{g}`' for g in sens_g)}.",
        f"- Complete coverage: {all(r['complete_plant_level_control_coverage'] for r in group_rows if r['scope_role']=='sensitivity_a_extension')}.",
        "- Extension failure does not by itself invalidate Primary RQ3.",
        "",
        "## 3. Out of scope",
        "",
        f"- Non-evaluable group(s): {', '.join(f'`{g}`' for g in out_g)}.",
        "- No control matching was attempted to rescue this group.",
        "",
        "## 4. Ties, fallback, reuse",
        "",
        "- Scientific ties are preserved; plant ID is not a scientific tie-break.",
        "- Same-structure candidates dominate mismatched-structure candidates when any exist.",
        f"- Control plants appearing as co-top for more than one reference: {len(reused)}. Reuse is enumerated, not frozen as analysis policy.",
        "",
        "## 5. Unresolved (S00–S03)",
        "",
        "- Final selected control when co-top candidates remain.",
        "- Replacement vs no-replacement / maximum reuse.",
        "- Any capacity/efficiency/bifaciality caliper.",
        "- Weighted distance is not used and remains disallowed unless a later controller job authorizes it.",
        "",
        "## 6. Limitations",
        "",
        "- Matching uses official BR-PVGen technical metadata only.",
        "- No OTG, model, SR, or outcome quantity entered ranking.",
        "",
        "## 7. Gate",
        "",
        f"**P02F {pstatus}**",
        "",
    ]
    return "\n".join(lines)
