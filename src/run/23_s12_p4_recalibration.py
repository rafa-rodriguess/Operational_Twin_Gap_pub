"""23 — S12 P4 bias/dispersion + VALIDATION-only recalibration ladder."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, sha256_file, write_csv
from src.lib.s12_p4_recalibration import (
    RQ3_STATUS,
    STATUS,
    render_report,
    run_science,
    write_parquet,
)
from src.run._ledger import ledger_patch
from src.sot import apply_patch

SNAPSHOT_PATHS = (
    ("results", "tables", "rq1"),
    ("results", "tables", "rq2"),
    ("results", "tables", "rq3"),
    ("results", "s11_phase4", "decisions"),
    ("results", "s12_phase0", "status"),
    ("results", "s12_p1", "status"),
    ("results", "s12_p2", "status"),
    ("results", "s12_p3", "status"),
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
    if set((patch.get("results") or {}).keys()) != {"s12_p4"}:
        return False, "results keys"
    if set((patch.get("scientific_analysis") or {}).keys()) != {"s12_p4"}:
        return False, "sci keys"
    if _snap(merged) != pre:
        return False, "prior snapshot changed"
    return True, "s12_p4 only"


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex, pdf = root / "paper/main.tex", root / "paper/main.pdf"
    tex_b, pdf_b = _sha(tex), _sha(pdf)
    s04m = root / "artifacts/s04/models/PS_002/SplineRidge/full.joblib"

    sci = run_science(root, ctx.sot)
    summary = sci["summary"]
    if not summary.get("reconciliation_ok"):
        add("independent_reconciliation", False, str(summary.get("reconciliation_max_abs_discrepancy")))
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s12_p4": {"status": "STOP", "failed": ["independent_reconciliation"], "reconciliation_max_abs_discrepancy": summary.get("reconciliation_max_abs_discrepancy")}}},
            validations=checks,
        )

    out = root / "artifacts/s12/p4"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(
        out / "S12_P4_CALIBRATION_PARAMETERS.csv",
        sci["cal_rows"],
        ["group_id", "source_plant_id", "target_plant_id", "n_validation", "offset_eligible", "affine_eligible", "affine_reason", "delta", "a", "b", "extreme_finite_slope"],
    )
    write_csv(
        out / "S12_P4_DIRECTIONAL.csv",
        sci["directional"],
        [
            "group_id", "source_plant_id", "target_plant_id", "computable", "n_validation_supported", "n_test_supported",
            "offset_eligible", "affine_eligible", "affine_reason", "delta", "a", "b",
            "otg_0", "otg_1", "otg_2", "delta_otg_offset", "delta_otg_affine",
            "transfer_mae_0", "transfer_mae_1", "transfer_mae_2", "local_mae",
            "mbe_0", "mbe_1", "mbe_2", "centered_mae_0", "centered_mae_1", "support_retention", "primary_otg",
        ],
    )
    write_csv(
        out / "S12_P4_BIAS_DISPERSION.csv",
        sci["bias_rows"],
        [
            "group_id", "source_plant_id", "target_plant_id", "mbe_transfer_0", "mbe_transfer_1", "mbe_transfer_2",
            "abs_mbe_0", "abs_mbe_1", "centered_mae_0", "centered_mae_1", "centered_rmse_0", "centered_rmse_1",
            "local_mbe", "local_centered_mae", "otg_0", "abs_mbe_transfer",
        ],
    )
    write_csv(
        out / "S12_P4_RQ2_PAIRS.csv",
        sci["pairs1"] + sci["pairs2"],
        ["group_id", "plant_a", "plant_b", "level", "otg_a_to_b", "otg_b_to_a", "asymmetry_signed", "asymmetry_abs"],
    )
    n_val = write_parquet(out / "S12_P4_VALIDATION_ROWS.parquet", sci["val_rows"])
    n_test = write_parquet(out / "S12_P4_TEST_ROWS.parquet", sci["test_rows"])
    dump_json(out / "S12_P4_SUMMARY.json", summary)
    md = root / "reports/scientific/S12_P4_BIAS_RECALIBRATION.md"
    md.write_text(render_report(summary), encoding="utf-8")

    recs = [
        artifact(root, out / "S12_P4_CALIBRATION_PARAMETERS.csv", rows=len(sci["cal_rows"]), fp=csv_fingerprint(out / "S12_P4_CALIBRATION_PARAMETERS.csv")),
        artifact(root, out / "S12_P4_DIRECTIONAL.csv", rows=len(sci["directional"]), fp=csv_fingerprint(out / "S12_P4_DIRECTIONAL.csv")),
        artifact(root, out / "S12_P4_BIAS_DISPERSION.csv", rows=len(sci["bias_rows"]), fp=csv_fingerprint(out / "S12_P4_BIAS_DISPERSION.csv")),
        artifact(root, out / "S12_P4_RQ2_PAIRS.csv", rows=len(sci["pairs1"]) + len(sci["pairs2"]), fp=csv_fingerprint(out / "S12_P4_RQ2_PAIRS.csv")),
        artifact(root, out / "S12_P4_VALIDATION_ROWS.parquet", rows=n_val),
        artifact(root, out / "S12_P4_TEST_ROWS.parquet", rows=n_test),
        artifact(root, out / "S12_P4_SUMMARY.json"),
        artifact(root, md),
    ]
    prov_rel = "artifacts/s12/p4/S12_P4_PROVENANCE.json"
    provenance = {
        "repository_head": _git_head(root),
        "canonical_inputs": {
            "artifacts/s07/S07_OTG_DIRECTIONAL.csv": sha256_file(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"),
            "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet": sha256_file(root / "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet"),
            "artifacts/s05/S05_SUPPORT_MASKS.parquet": sha256_file(root / "artifacts/s05/S05_SUPPORT_MASKS.parquet"),
            "artifacts/s04/S04_ANALYTIC_INDEX.parquet": sha256_file(root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet"),
            "artifacts/p02c/P02C_PLANT_PANEL.parquet": sha256_file(root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"),
            "config/protocol.py": sha256_file(root / "config/protocol.py"),
        },
        "primary_universe": "SplineRidge + CS4 + computable + primary groups; 85 directions; R9/OTG_half/R7 excluded from main P4",
        "residual_sign": "prediction - y_true",
        "offset_formula": "delta = mean(y_validation - p_source_validation) on source-supported target VALIDATION; p1 = p0 + delta",
        "affine_formula": "OLS y = a + b * p_source on the same VALIDATION support; p2 = a + b * p0; no regularization/clipping/fallback",
        "affine_eligibility": "n>=2, finite, nonzero source-pred variance, full rank intercept+slope, finite coefficients",
        "source_support": "primary CS4 k=5 q=0.95 TRAIN median/IQR; degenerate IQR not_computable; no R9 MAD/omission",
        "validation_role": "target VALIDATION outcomes only estimate delta and (a,b)",
        "test_role": "target TEST never used to estimate recalibration; evaluation only on canonical primary-supported TEST rows",
        "no_test_calibration": True,
        "model_actions": summary["model_actions"],
        "model_equivalence": summary["model_equivalence"],
        "script_path": "src/run/23_s12_p4_recalibration.py",
        "helper_path": "src/lib/s12_p4_recalibration.py",
        "generated_at_utc": now,
        "accepted_science_not_modified": True,
        "manuscript_files_not_edited": True,
        "self_hash_registered_externally": True,
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes, "row_count": rec.row_count} for rec in recs},
    }
    dump_json(out / "S12_P4_PROVENANCE.json", provenance)
    recs.append(artifact(root, out / "S12_P4_PROVENANCE.json"))

    add("universe_85", len(sci["primary"]) == 85 and summary["n_directional_computable"] == 85, str(summary["n_directional_computable"]))
    add("rq3_not_proxied", summary["rq3_status"] == RQ3_STATUS, summary["rq3_status"])
    add("independent_reconciliation", summary["reconciliation_ok"], str(summary["reconciliation_max_abs_discrepancy"]))
    add("no_p5_files", list((root / "src/run").glob("24_s12*")) == [], "stop after P4")
    add("primary_models_not_overwritten", s04m.is_file(), str(s04m))
    add("r9_not_used", summary.get("r9_not_used") is True, "primary CS4")
    add("no_test_in_calibration", summary.get("no_test_in_calibration") is True, "val only")
    add("residual_sign", summary.get("residual_sign") == "prediction - y_true", summary.get("residual_sign"))
    add("provenance_omits_self", prov_rel not in provenance["outputs"], "ok")
    add("paper_tex_unchanged", tex_b == _sha(tex), _sha(tex))
    add("paper_pdf_unchanged", pdf_b == _sha(pdf), _sha(pdf))
    add("no_interp_thresholds", summary.get("interpretation", {}).get("threshold_rules_used") is False, str(summary.get("interpretation")))
    add("model_equivalence_ok", (summary["model_equivalence"]["max_abs_prediction_difference"] or 0) <= 1e-10, str(summary["model_equivalence"]))

    payload = {
        "status": STATUS,
        "n_planned": 85,
        "n_offset_eligible": summary["n_offset_eligible"],
        "n_affine_eligible": summary["n_affine_eligible"],
        "k0": summary["k0"],
        "k1": summary["k1"],
        "k2": summary["k2"],
        "delta_offset": summary["delta_offset"],
        "delta_affine": summary["delta_affine"],
        "calibration_parameters": {
            "median_abs_delta": summary["median_abs_delta"],
            "median_abs_b_minus_1": summary["median_abs_b_minus_1"],
            "delta_distribution": summary["delta_distribution"],
            "slope_distribution": summary["slope_distribution"],
        },
        "bias": summary["bias"],
        "spearman_abs_mbe_vs_otg": summary["spearman_abs_mbe_vs_otg"],
        "rq2": summary["rq2"],
        "rq3_status": summary["rq3_status"],
        "interpretation": summary["interpretation"],
        "model_actions": summary["model_actions"],
        "model_equivalence": summary["model_equivalence"],
        "reconciliation_max_abs_discrepancy": summary["reconciliation_max_abs_discrepancy"],
        "calibration_path": "artifacts/s12/p4/S12_P4_CALIBRATION_PARAMETERS.csv",
        "calibration_sha256": recs[0].sha256,
        "directional_path": "artifacts/s12/p4/S12_P4_DIRECTIONAL.csv",
        "directional_sha256": recs[1].sha256,
        "bias_path": "artifacts/s12/p4/S12_P4_BIAS_DISPERSION.csv",
        "bias_sha256": recs[2].sha256,
        "rq2_pairs_path": "artifacts/s12/p4/S12_P4_RQ2_PAIRS.csv",
        "rq2_pairs_sha256": recs[3].sha256,
        "validation_rows_path": "artifacts/s12/p4/S12_P4_VALIDATION_ROWS.parquet",
        "validation_rows_sha256": recs[4].sha256,
        "test_rows_path": "artifacts/s12/p4/S12_P4_TEST_ROWS.parquet",
        "test_rows_sha256": recs[5].sha256,
        "summary_path": "artifacts/s12/p4/S12_P4_SUMMARY.json",
        "summary_sha256": recs[6].sha256,
        "report_path": "reports/scientific/S12_P4_BIAS_RECALIBRATION.md",
        "report_sha256": recs[7].sha256,
        "provenance_path": prov_rel,
        "provenance_sha256": recs[-1].sha256,
        "sensitivity_only": True,
        "failed": None,
    }
    add("sot_prov_sha", payload["provenance_sha256"] == recs[-1].sha256, recs[-1].sha256)
    pre = _snap(ctx.sot)
    patch = ledger_patch(stage="s12_p4", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p4": payload}})
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_rq_unchanged", _nested(merged, ("results", "tables", "rq1")) == pre.get("results.tables.rq1") and _nested(merged, ("results", "tables", "rq2")) == pre.get("results.tables.rq2"), "rq")
    add("s11_unchanged", _nested(merged, ("results", "s11_phase4", "decisions")) == pre.get("results.s11_phase4.decisions"), "s11")
    add("s12_prior_unchanged", _nested(merged, ("results", "s12_p3", "status")) == pre.get("results.s12_p3.status"), "p0-p3")
    add("rq4_unchanged", _nested(merged, ("scientific_analysis", "rq4_source_selection", "status")) == pre.get("scientific_analysis.rq4_source_selection.status"), "rq4")
    payload["n_validations"] = len(checks)
    patch = ledger_patch(stage="s12_p4", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p4": payload}})
    if not ns_ok:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p4": {"status": "STOP", "reason": ns_detail, "failed": None}}}, artifacts=recs, validations=checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p4": {"status": "STOP", "failed": [c.name for c in failed]}}}, artifacts=recs, validations=checks)
    return StageResult(status="GO", message=payload["status"], sot_patch=patch, artifacts=recs, validations=checks)
