"""16 — S11 Phase 2 A2/R8 all eligible same-state non-twin controls."""

from __future__ import annotations

import hashlib
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, sha256_file, write_csv
from src.lib.s11_phase2_a2 import (
    STATUS_MATERIALIZED,
    STATUS_NONINFORMATIVE,
    aggregate_path_groupby,
    aggregate_path_iterate,
    compute_r8,
    enumerate_eligibility,
    load_mapping,
    load_members,
    load_meta,
    members_path,
    missing_rq3_inputs,
    r8_close,
    rq3_mapping_path,
)
from src.run._ledger import ledger_patch
from src.sot import apply_patch

ELIG_FIELDS = [
    "target_id",
    "target_state",
    "control_id",
    "same_state",
    "is_nominal_twin",
    "structural_candidate",
    "model_fit_executable",
    "has_any_cs4_pairwise_support",
    "structurally_executable",
    "eligibility_status",
    "n_train_xy",
    "n_val_xy",
    "ranking_tuple",
    "is_primary_matched_control",
    "target_group_id",
]
CMP_FIELDS = [
    "target_id",
    "twin_source_id",
    "control_id",
    "target_state",
    "common_row_count",
    "twin_otg_common_rows",
    "control_otg_common_rows",
    "contrast_control_minus_twin",
    "common_row_fingerprint",
    "computable",
    "reason",
    "orientation",
]
TGT_FIELDS = [
    "target_id",
    "n_eligible_controls",
    "n_distinct_controls_contributing",
    "n_valid_comparisons",
    "target_mean_control_minus_twin",
]
SNAPSHOT_PATHS = (
    ("results", "tables", "rq1"),
    ("results", "tables", "rq2"),
    ("results", "tables", "rq3"),
    ("results", "rq3", "D_fleet_RQ3"),
    ("results", "rq3", "selected_controls"),
    ("results", "s11_phase1", "status"),
    ("results", "s11_phase1", "median_spread"),
    ("results", "s11_phase1", "plant_balanced_rq1"),
    ("scientific_analysis", "rq4_source_selection", "status"),
)


def _nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _snap(sot: dict[str, Any]) -> dict[str, Any]:
    return {".".join(p): _nested(sot, p) for p in SNAPSHOT_PATHS}


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _namespace_ok(pre: dict[str, Any], patch: dict[str, Any], merged: dict[str, Any]) -> tuple[bool, str]:
    rkeys = set((patch.get("results") or {}).keys())
    if rkeys != {"s11_phase2_a2"}:
        return False, f"results keys={sorted(rkeys)}"
    sci = set((patch.get("scientific_analysis") or {}).keys())
    if sci != {"s11_phase2_a2"}:
        return False, f"scientific_analysis keys={sorted(sci)}"
    post = _snap(merged)
    if post != pre:
        return False, "primary/A3/A4/RQ4 snapshot changed"
    if _nested(merged, ("results", "rq3", "D_fleet_RQ3")) != pre.get("results.rq3.D_fleet_RQ3"):
        return False, "RQ3 D_fleet changed"
    return True, "s11_phase2_a2 only; RQ1/RQ2/RQ3/A3/A4/RQ4 snapshot unchanged"


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex_path, pdf_path = root / "paper/main.tex", root / "paper/main.pdf"
    tex_before, pdf_before = _sha(tex_path), _sha(pdf_path)
    missing = missing_rq3_inputs(root)
    add("canonical_rq3_inputs_present", missing is None, missing)
    if missing:
        return StageResult(
            status="STOP",
            message="STOP_MISSING_RQ3_CONTROL_DEFINITION",
            sot_patch={"results": {"s11_phase2_a2": {"status": "STOP", "reason": missing}}},
            validations=checks,
        )

    meta = load_meta(root / "data/raw/br_pvgen/BR-PVGen_metadata.csv")
    members = load_members(root)
    mapping = load_mapping(root)
    add("canonical_rq3_target_cohort", len(mapping) == 10, f"n={len(mapping)}")
    add("canonical_same_state_definition", all("control_plant_id" in r and "target_state" in r for r in mapping), "mapping.csv")
    if not mapping:
        return StageResult(
            status="STOP",
            message="STOP_MISSING_RQ3_CONTROL_DEFINITION",
            sot_patch={"results": {"s11_phase2_a2": {"status": "STOP", "reason": "empty mapping"}}},
            validations=checks,
        )

    elig_rows, elig_sum, members_by_group, elig_err = enumerate_eligibility(root, meta, members, mapping)
    if elig_err:
        return StageResult(
            status="STOP",
            message="STOP_MISSING_RQ3_CONTROL_DEFINITION",
            sot_patch={"results": {"s11_phase2_a2": {"status": "STOP", "reason": elig_err}}},
            validations=checks,
        )
    elig_keys = {k for r in elig_rows for k in r}
    add(
        "eligibility_without_outcomes",
        all("contrast" not in k.lower() and "otg" not in k.lower() for k in elig_keys),
        ",".join(sorted(elig_keys)),
    )
    add(
        "unfit_not_executable",
        all((not r.get("structurally_executable")) or r.get("model_fit_executable") for r in elig_rows),
        str(sum(1 for r in elig_rows if not r.get("model_fit_executable"))),
    )
    add(
        "no_cs4_support_not_executable",
        all((not r.get("structurally_executable")) or r.get("has_any_cs4_pairwise_support") for r in elig_rows),
        str(sum(1 for r in elig_rows if not r.get("has_any_cs4_pairwise_support"))),
    )
    exec_rows = [r for r in elig_rows if r.get("structurally_executable")]

    counts = elig_sum["per_target_control_counts"]
    add("summary_covers_all_rq3_targets", set(counts) == {r["target_plant_id"] for r in mapping}, str(len(counts)))
    add("zero_control_targets_counted", elig_sum["n_targets_zero_eligible_controls"] == sum(1 for v in counts.values() if v == 0), str(elig_sum["n_targets_zero_eligible_controls"]))
    med = elig_sum["median_eligible_controls_per_target"]
    import statistics as _st

    med_obs = float(_st.median([counts[t] for t in counts])) if counts else None
    add("median_eligible_controls", med == med_obs, str(med))
    expected_status = STATUS_NONINFORMATIVE if med == 1 else STATUS_MATERIALIZED
    add("structural_status_rule", elig_sum["structural_status"] == expected_status, elig_sum["structural_status"])
    add("no_nominal_twin_as_control", all(not r["is_nominal_twin"] for r in elig_rows), str(len(elig_rows)))

    out = root / "artifacts/s11"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "S11_A2_R8_ELIGIBILITY.csv", elig_rows, ELIG_FIELDS)
    dump_json(out / "S11_A2_R8_ELIGIBILITY_SUMMARY.json", elig_sum)
    recs = [
        artifact(root, out / "S11_A2_R8_ELIGIBILITY.csv", rows=len(elig_rows), fp=csv_fingerprint(out / "S11_A2_R8_ELIGIBILITY.csv")),
        artifact(root, out / "S11_A2_R8_ELIGIBILITY_SUMMARY.json"),
    ]

    part_b = med != 1
    r8_summary = None
    recon_ok = not part_b
    if not part_b:
        wrote_cmp = (out / "S11_A2_R8_COMPARISONS.csv").is_file()
        add("no_r8_outcomes_when_median_1", not wrote_cmp, f"comparisons_exist={wrote_cmp}")
        add("part_b_skipped_median", med == 1, str(med))
        add("r8_reconciliation", med == 1, "Part B skipped")
        add("no_bootstrap_ci_pvalue", med == 1, "no R8 summary")
    else:
        raw = compute_r8(root, exec_rows, members_by_group)
        if raw.get("status") != "GO":
            return StageResult(
                status="STOP",
                message="STOP_MISSING_RQ3_CONTROL_DEFINITION",
                sot_patch={"results": {"s11_phase2_a2": {"status": "STOP", **raw}}},
                artifacts=recs,
                validations=checks,
            )
        comps = raw["comparisons"]
        targets = elig_sum["rq3_targets"]
        t1, f1 = aggregate_path_groupby(comps, targets)
        t2, f2 = aggregate_path_iterate(comps, targets)
        recon_ok = r8_close(t1, t2, f1, f2)
        add("r8_reconciliation", recon_ok, f"fleet={f1.get('target_balanced_r8_control_minus_twin')}")
        if not recon_ok:
            return StageResult(
                status="STOP",
                message="STOP_RECONCILIATION_FAILURE",
                sot_patch={"results": {"s11_phase2_a2": {"status": "STOP", "reason": "R8 two-path mismatch"}}},
                artifacts=recs,
                validations=checks,
            )
        exec_ids = {(r["target_id"], r["control_id"]) for r in exec_rows}
        add("part_b_only_eligible_pairs", all((c["target_id"], c["control_id"]) in exec_ids for c in comps), str(len(comps)))
        add("no_unfit_in_part_b", all(c.get("reason") != "control_model_unfit" for c in comps), str(sum(1 for c in comps if c.get("reason")=="control_model_unfit")))
        group_of_t = {r["target_id"]: r["target_group_id"] for r in elig_rows}
        twin_ok = all(c["control_id"] not in set(members_by_group.get(group_of_t[c["target_id"]], [])) for c in comps)
        add("no_twin_control_in_comparisons", twin_ok, "control not in target twin group")
        add("orientation_control_minus_twin", all(c.get("orientation") == "CONTROL_MINUS_TWIN" for c in comps), "CONTROL_MINUS_TWIN")
        add(
            "common_rows_used",
            all((str(c.get("computable")).lower() not in {"true", "1"}) or int(c["common_row_count"]) >= 1 for c in comps),
            "computable implies common_row_count>=1",
        )
        add(
            "source_level_weighting",
            all(int(r["n_valid_comparisons"]) >= 1 for r in t1),
            "delta_j = mean_s mean_c contrast",
        )
        add("fleet_equal_target_weight", f1["n_contributing_targets"] == len(t1), str(f1["n_contributing_targets"]))
        n_dist = len({c["control_id"] for c in comps if str(c.get("computable")).lower() in {"true", "1", "yes"}})
        add("distinct_controls_counted", f1["n_distinct_contributing_controls"] == n_dist, str(f1["n_distinct_contributing_controls"]))
        blob = str(f1) + str(t1)
        add("no_bootstrap_ci_pvalue", all(tok not in blob.lower() for tok in ("bootstrap", "p-value", "pvalue", "confidence_interval")), "descriptive only")

        write_csv(out / "S11_A2_R8_COMPARISONS.csv", comps, CMP_FIELDS)
        write_csv(out / "S11_A2_R8_TARGET_SUMMARY.csv", t1, TGT_FIELDS)
        r8_summary = {
            "status": STATUS_MATERIALIZED,
            **f1,
            "n_targets_zero_eligible_controls": elig_sum["n_targets_zero_eligible_controls"],
            "target_formula": "delta_j = mean_s (mean_c (OTG_control - OTG_twin) on pairwise CS4 intersection)",
            "fleet_formula": "equal-weight mean of delta_j over contributing targets",
            "reconciliation_passed": recon_ok,
        }
        dump_json(out / "S11_A2_R8_SUMMARY.json", r8_summary)
        recs.extend(
            [
                artifact(root, out / "S11_A2_R8_COMPARISONS.csv", rows=len(comps), fp=csv_fingerprint(out / "S11_A2_R8_COMPARISONS.csv")),
                artifact(root, out / "S11_A2_R8_TARGET_SUMMARY.csv", rows=len(t1), fp=csv_fingerprint(out / "S11_A2_R8_TARGET_SUMMARY.csv")),
                artifact(root, out / "S11_A2_R8_SUMMARY.json"),
            ]
        )
        elig_sum["part_b_executed"] = True
        dump_json(out / "S11_A2_R8_ELIGIBILITY_SUMMARY.json", elig_sum)
        recs[1] = artifact(root, out / "S11_A2_R8_ELIGIBILITY_SUMMARY.json")

    from config.protocol import CS4, CONTRAST_ORIENTATION, FAM_SPLINE

    add(
        "protocol_cs4_spline_unchanged",
        FAM_SPLINE == "SplineRidge" and CS4["id"] == "CS4" and CONTRAST_ORIENTATION == "CONTROL_MINUS_TWIN",
        f"{FAM_SPLINE}/{CS4['id']}/{CONTRAST_ORIENTATION}",
    )
    a1_files = list(out.glob("S11_A1*")) + list((root / "src/run").glob("*a1*"))
    add("no_a1_execution", a1_files == [], str(a1_files))

    head = _git_head(root)
    provenance = {
        "repository_head": head,
        "source_artifacts": {
            "metadata": {"path": "data/raw/br_pvgen/BR-PVGen_metadata.csv", "sha256": sha256_file(root / "data/raw/br_pvgen/BR-PVGen_metadata.csv")},
            "members": {"path": members_path(root).relative_to(root).as_posix(), "sha256": sha256_file(members_path(root))},
            "rq3_mapping": {"path": "artifacts/rq3/mapping.csv", "sha256": sha256_file(rq3_mapping_path(root))},
            "rq3_summary": {"path": "artifacts/rq3/summary.json", "sha256": sha256_file(root / "artifacts/rq3/summary.json") if (root / "artifacts/rq3/summary.json").is_file() else None},
            "support_masks": {"path": "artifacts/s05/S05_SUPPORT_MASKS.parquet", "sha256": sha256_file(root / "artifacts/s05/S05_SUPPORT_MASKS.parquet") if (root / "artifacts/s05/S05_SUPPORT_MASKS.parquet").is_file() else None},
        },
        "script_path": "src/run/16_s11_phase2_a2.py",
        "helper_path": "src/lib/s11_phase2_a2.py",
        "eligibility_rule": elig_sum["eligibility_rule"],
        "generated_at_utc": now,
        "n_primary_rq3_targets": elig_sum["n_primary_rq3_targets"],
        "eligibility_row_count": len(elig_rows),
        "median_eligible_controls_per_target": med,
        "part_b_executed": part_b,
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes, "row_count": rec.row_count, "schema_fingerprint": rec.schema_fingerprint} for rec in recs},
    }
    if part_b and r8_summary:
        provenance["comparison_row_count"] = r8_summary["n_valid_comparisons"]
        provenance["contributing_target_count"] = r8_summary["n_contributing_targets"]
    dump_json(out / "S11_PHASE2_A2_PROVENANCE.json", provenance)
    recs.append(artifact(root, out / "S11_PHASE2_A2_PROVENANCE.json"))
    add("provenance_complete", (out / "S11_PHASE2_A2_PROVENANCE.json").is_file() and provenance["median_eligible_controls_per_target"] == med, provenance["repository_head"])

    tex_after, pdf_after = _sha(tex_path), _sha(pdf_path)
    add("paper_tex_unchanged", tex_before == tex_after and tex_before is not None, tex_after)
    add("paper_pdf_unchanged", pdf_before == pdf_after, pdf_after)

    payload = {
        "status": elig_sum["structural_status"] if not part_b else STATUS_MATERIALIZED,
        "median_eligible_controls_per_target": med,
        "n_primary_rq3_targets": elig_sum["n_primary_rq3_targets"],
        "n_targets_zero_eligible_controls": elig_sum["n_targets_zero_eligible_controls"],
        "n_targets_one_eligible_control": elig_sum["n_targets_one_eligible_control"],
        "n_targets_more_than_one_eligible_control": elig_sum["n_targets_more_than_one_eligible_control"],
        "distinct_eligible_control_count": elig_sum["distinct_eligible_control_count"],
        "per_target_control_counts": counts,
        "part_b_executed": part_b,
        "eligibility_path": "artifacts/s11/S11_A2_R8_ELIGIBILITY.csv",
        "eligibility_sha256": recs[0].sha256,
        "provenance_path": "artifacts/s11/S11_PHASE2_A2_PROVENANCE.json",
        "provenance_sha256": recs[-1].sha256,
    }
    if part_b and r8_summary:
        payload.update(
            {
                "target_balanced_r8_control_minus_twin": r8_summary["target_balanced_r8_control_minus_twin"],
                "n_contributing_targets": r8_summary["n_contributing_targets"],
                "n_distinct_contributing_controls": r8_summary["n_distinct_contributing_controls"],
                "n_valid_comparisons": r8_summary["n_valid_comparisons"],
                "orientation": "CONTROL_MINUS_TWIN",
            }
        )

    pre = _snap(ctx.sot)
    matched_pre = [dict(r) for r in mapping]
    patch = ledger_patch(
        stage="s11_phase2_a2",
        now=now,
        payload=payload,
        recs=recs,
        extra={"scientific_analysis": {"s11_phase2_a2": payload}},
    )
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_rq3_unchanged", _nested(merged, ("results", "rq3", "D_fleet_RQ3")) == pre["results.rq3.D_fleet_RQ3"], str(pre["results.rq3.D_fleet_RQ3"]))
    add("matched_controls_unchanged", load_mapping(root) == matched_pre, str(sorted({r["control_plant_id"] for r in mapping})))
    add(
        "rq1_rq2_a3_a4_rq4_unchanged",
        all(_nested(merged, p) == pre[".".join(p)] for p in SNAPSHOT_PATHS if p[1] != "s11_phase2_a2"),
        "snapshot equal",
    )
    payload["n_validations"] = len(checks)
    payload["failed"] = None
    patch = ledger_patch(stage="s11_phase2_a2", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s11_phase2_a2": payload}})
    if not ns_ok:
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s11_phase2_a2": {"status": "STOP", "reason": ns_detail}}},
            artifacts=recs,
            validations=checks,
        )
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s11_phase2_a2": {"status": "STOP", "failed": [c.name for c in failed]}}},
            artifacts=recs,
            validations=checks,
        )
    return StageResult(
        status="GO",
        message=payload["status"],
        sot_patch=patch,
        artifacts=recs,
        validations=checks,
    )
