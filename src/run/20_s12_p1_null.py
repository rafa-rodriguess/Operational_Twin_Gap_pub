"""20 — S12 P1 within-plant null calibration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, sha256_file, write_csv
from src.lib.s12_p1_null import (
    STATUS,
    TOL,
    render_report,
    run_science,
    write_eval_parquet,
)
from src.run._ledger import ledger_patch
from src.sot import apply_patch

SNAPSHOT_PATHS = (
    ("results", "tables", "rq1"),
    ("results", "tables", "rq2"),
    ("results", "tables", "rq3"),
    ("results", "rq3", "D_fleet_RQ3"),
    ("results", "rq3", "selected_controls"),
    ("results", "s11_phase1", "status"),
    ("results", "s11_phase1", "median_spread"),
    ("results", "s11_phase1", "plant_balanced_rq1"),
    ("results", "s11_phase2_a2", "status"),
    ("results", "s11_phase2_a2", "target_balanced_r8_control_minus_twin"),
    ("results", "s11_phase3_a1_g3", "status"),
    ("results", "s11_phase3_a1_g3", "rq1_plant_balanced_mean_otg"),
    ("results", "s11_phase4", "status"),
    ("results", "s11_phase4", "decisions"),
    ("results", "s12_phase0", "status"),
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
    if rkeys != {"s12_p1"}:
        return False, f"results keys={sorted(rkeys)}"
    sci = set((patch.get("scientific_analysis") or {}).keys())
    if sci != {"s12_p1"}:
        return False, f"scientific_analysis keys={sorted(sci)}"
    if _snap(merged) != pre:
        return False, "prior scientific snapshot changed"
    return True, "s12_p1 only"


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex_path, pdf_path = root / "paper/main.tex", root / "paper/main.pdf"
    tex_before, pdf_before = _sha(tex_path), _sha(pdf_path)
    s04_before = _sha(root / "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv")
    s07_before = _sha(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv")
    s05_before = _sha(root / "artifacts/s05/S05_SUPPORT_MASKS.parquet")

    out = root / "artifacts/s12/p1"
    models_dir = out / "models"
    sci = run_science(root, models_dir)
    summary = sci["summary"]
    if not summary.get("reconciliation_ok"):
        add("independent_reconciliation", False, str(summary.get("reconciliation_max_abs_discrepancy")))
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s12_p1": {"status": "STOP", "failed": ["independent_reconciliation"], "max_delta": summary.get("reconciliation_max_abs_discrepancy")}}},
            validations=checks,
        )

    out.mkdir(parents=True, exist_ok=True)
    placebo_csv_rows = []
    for r in sci["placebo_rows"]:
        row = dict(r)
        hp = row.get("hyperparameters")
        row["hyperparameters"] = json.dumps(hp, sort_keys=True) if isinstance(hp, dict) else hp
        placebo_csv_rows.append(row)
    p_fields = [
        "plant_id", "group_id", "n_train", "n_test", "n_a_chrono", "n_b_chrono", "n_a_interleaved", "n_b_interleaved",
        "n_blocks_a", "n_blocks_b", "computable_chrono", "computable_interleaved", "reason_chrono", "reason_interleaved",
        "mae_a_chrono", "mae_b_chrono", "otg_placebo_chrono", "mae_a_interleaved", "mae_b_interleaved", "otg_placebo_interleaved",
        "otg_chrono_minus_interleaved", "n_supported_chrono", "n_supported_interleaved", "support_retention_chrono",
        "support_retention_interleaved", "threshold_chrono", "threshold_interleaved", "hyperparameters",
    ]
    h_fields = [
        "group_id", "source_plant_id", "target_plant_id", "model_family", "support_rule", "computable", "reason",
        "n_source_recent_half", "n_target_recent_half", "n_eligible_target_test", "n_supported", "n_supported_days",
        "support_retention", "threshold", "transfer_mae", "local_same_rows_mae", "otg_abs",
    ]
    write_csv(out / "S12_P1_PLACEBO_PLANT_SUMMARY.csv", placebo_csv_rows, p_fields)
    write_csv(out / "S12_P1_OTG_HALF_DIRECTIONAL.csv", sci["half_dir"], h_fields)
    dump_json(out / "S12_P1_SUMMARY.json", summary)
    write_eval_parquet(out / "S12_P1_EVAL_ROWS.parquet", sci["eval_rows"])
    md_path = root / "reports/scientific/S12_P1_NULL_CALIBRATION.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_report(summary), encoding="utf-8")

    recs = [
        artifact(root, out / "S12_P1_PLACEBO_PLANT_SUMMARY.csv", rows=len(placebo_csv_rows), fp=csv_fingerprint(out / "S12_P1_PLACEBO_PLANT_SUMMARY.csv")),
        artifact(root, out / "S12_P1_OTG_HALF_DIRECTIONAL.csv", rows=len(sci["half_dir"]), fp=csv_fingerprint(out / "S12_P1_OTG_HALF_DIRECTIONAL.csv")),
        artifact(root, out / "S12_P1_SUMMARY.json"),
        artifact(root, out / "S12_P1_EVAL_ROWS.parquet", rows=len(sci["eval_rows"])),
        artifact(root, md_path),
    ]
    for mp in sci["model_paths"]:
        recs.append(artifact(root, mp))

    panel_h = sha256_file(root / "artifacts/p02c/P02C_PLANT_PANEL.parquet")
    idx_h = sha256_file(root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet")
    proto_h = sha256_file(root / "config/protocol.py")
    sel_h = sha256_file(root / "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv")
    prov_rel = "artifacts/s12/p1/S12_P1_PROVENANCE.json"
    output_map = {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes, "row_count": rec.row_count} for rec in recs}
    provenance = {
        "repository_head": _git_head(root),
        "canonical_inputs": {
            "artifacts/p02c/P02C_PLANT_PANEL.parquet": panel_h,
            "artifacts/s04/S04_ANALYTIC_INDEX.parquet": idx_h,
            "config/protocol.py": proto_h,
            "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv": sel_h,
            "artifacts/s07/S07_OTG_DIRECTIONAL.csv": s07_before,
            "artifacts/s05/S05_SUPPORT_MASKS.parquet": s05_before,
        },
        "half_split_rule": summary["half_split_rule"],
        "interleave_rule": summary["interleave_rule"],
        "support_semantics": summary["support_semantics"],
        "script_path": "src/run/20_s12_p1_null.py",
        "helper_path": "src/lib/s12_p1_null.py",
        "generated_at_utc": now,
        "accepted_primary_science_not_overwritten": True,
        "manuscript_files_not_edited": True,
        "self_hash_registered_externally": True,
        "outputs": {k: v for k, v in output_map.items() if k != prov_rel},
    }
    dump_json(out / "S12_P1_PROVENANCE.json", provenance)
    recs.append(artifact(root, out / "S12_P1_PROVENANCE.json"))

    plants = sci["plants"]
    add("all_14_primary_plants", len(plants) == 14, str(len(plants)))
    add("chrono_halves_disjoint_in_train", all(r.get("chrono_disjoint") and r.get("chrono_within_train") for r in sci["placebo_rows"]), "chrono A/B")
    add("chrono_rowcount_balanced", all(abs(int(r.get("n_a_chrono") or 0) - int(r.get("n_b_chrono") or 0)) <= 1 for r in sci["placebo_rows"]), "extra row to B")
    add("interleaved_alternating_equal_blocks", all(r.get("n_blocks_a") == r.get("n_blocks_b") and r.get("interleaved_disjoint") for r in sci["placebo_rows"]), "7-day blocks")
    add("hparams_frozen_spline", len(sci["hparams"]) == 14, str(sorted(sci["hparams"])))
    add("independent_reconciliation", summary["reconciliation_ok"], str(summary["reconciliation_max_abs_discrepancy"]))
    add("plant_balanced_two_paths", summary["target_balanced_sensitivity"]["plant_balanced_all_computable_targets"] == summary["target_balanced_sensitivity"]["plant_balanced_iterate_path"], str(summary["otg_half"]["plant_balanced_mean"]))
    add("exceedance_uses_otg_half", summary["null_exceedance"]["uses_otg_half_not_full_train_primary"] is True, "not full-TRAIN OTG")
    add("no_p3_mad", summary["no_p3_mad_fallback"] is True, "degenerate IQR not_computable")
    add("test_evaluation_only", summary["leakage"]["test_in_fit"] is False, str(summary["leakage"]))
    add("expected_transfer_universe", sci["universe_n"] == summary["otg_half"]["n_expected_directional"], str(sci["universe_n"]))
    add("no_p2_p9_files", list((root / "src/run").glob("21_s12*")) == [], "stop after P1")
    add("s04_hparams_unchanged", s04_before == _sha(root / "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv"), s04_before)
    add("s07_unchanged", s07_before == _sha(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"), s07_before)
    add("s05_masks_unchanged", s05_before == _sha(root / "artifacts/s05/S05_SUPPORT_MASKS.parquet"), s05_before)
    add("provenance_omits_self_hash", prov_rel not in provenance["outputs"], str(sorted(provenance["outputs"])[:8]))

    tex_after, pdf_after = _sha(tex_path), _sha(pdf_path)
    add("paper_tex_unchanged", tex_before == tex_after and tex_before is not None, tex_after)
    add("paper_pdf_unchanged", pdf_before == pdf_after, pdf_after)

    payload = {
        "status": STATUS,
        "interpretation_category": summary["interpretation_category"],
        "placebo": summary["placebo"],
        "otg_half": summary["otg_half"],
        "null_exceedance": summary["null_exceedance"],
        "target_balanced_sensitivity": summary["target_balanced_sensitivity"],
        "computability": {
            "n_plants": len(plants),
            "n_placebo_chrono": summary["placebo"]["chrono"]["n_computable"],
            "n_placebo_interleaved": summary["placebo"]["interleaved"]["n_computable"],
            "n_half_expected": summary["otg_half"]["n_expected_directional"],
            "n_half_computable": summary["otg_half"]["n_computable"],
            "noncomputable_half": summary["noncomputable_half"],
            "noncomputable_placebo": summary["noncomputable_placebo"],
        },
        "reconciliation_max_abs_discrepancy": summary["reconciliation_max_abs_discrepancy"],
        "summary_path": "artifacts/s12/p1/S12_P1_SUMMARY.json",
        "summary_sha256": recs[2].sha256,
        "placebo_path": "artifacts/s12/p1/S12_P1_PLACEBO_PLANT_SUMMARY.csv",
        "placebo_sha256": recs[0].sha256,
        "otg_half_path": "artifacts/s12/p1/S12_P1_OTG_HALF_DIRECTIONAL.csv",
        "otg_half_sha256": recs[1].sha256,
        "eval_rows_path": "artifacts/s12/p1/S12_P1_EVAL_ROWS.parquet",
        "eval_rows_sha256": recs[3].sha256,
        "report_path": "reports/scientific/S12_P1_NULL_CALIBRATION.md",
        "report_sha256": recs[4].sha256,
        "provenance_path": prov_rel,
        "provenance_sha256": recs[-1].sha256,
        "primary_science_unchanged": True,
        "failed": None,
    }
    add("sot_provenance_sha_matches_file", payload["provenance_sha256"] == recs[-1].sha256 == _sha(out / "S12_P1_PROVENANCE.json"), recs[-1].sha256)

    pre = _snap(ctx.sot)
    patch = ledger_patch(stage="s12_p1", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p1": payload}})
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_rq_unchanged", _nested(merged, ("results", "rq3", "D_fleet_RQ3")) == pre.get("results.rq3.D_fleet_RQ3") and _nested(merged, ("results", "tables", "rq1")) == pre.get("results.tables.rq1"), str(pre.get("results.rq3.D_fleet_RQ3")))
    add("s11_unchanged", _nested(merged, ("results", "s11_phase4", "decisions")) == pre.get("results.s11_phase4.decisions") and _nested(merged, ("results", "s11_phase1", "median_spread")) == pre.get("results.s11_phase1.median_spread"), "s11")
    add("rq4_unchanged", _nested(merged, ("scientific_analysis", "rq4_source_selection", "status")) == pre.get("scientific_analysis.rq4_source_selection.status"), str(pre.get("scientific_analysis.rq4_source_selection.status")))
    add("s12_phase0_unchanged", _nested(merged, ("results", "s12_phase0", "status")) == pre.get("results.s12_phase0.status"), str(pre.get("results.s12_phase0.status")))
    payload["n_validations"] = len(checks)
    patch = ledger_patch(stage="s12_p1", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p1": payload}})

    if not ns_ok:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p1": {"status": "STOP", "reason": ns_detail, "failed": None}}}, artifacts=recs, validations=checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p1": {"status": "STOP", "failed": [c.name for c in failed]}}}, artifacts=recs, validations=checks)
    return StageResult(status="GO", message=payload["status"], sot_patch=patch, artifacts=recs, validations=checks)
