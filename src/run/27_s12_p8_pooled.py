"""27 — S12 P8 leave-target-out pooled fleet baseline."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, sha256_file, write_csv
from src.lib.s12_p8_pooled import (
    PRIMARY_RQ1,
    RQ2_POLICY,
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
    ("results", "s12_p4", "status"),
    ("results", "s12_p5", "status"),
    ("results", "s12_p6", "status"),
    ("results", "s12_p7", "status"),
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
    if set((patch.get("results") or {}).keys()) != {"s12_p8"}:
        return False, "results keys"
    if set((patch.get("scientific_analysis") or {}).keys()) != {"s12_p8"}:
        return False, "sci keys"
    if _snap(merged) != pre:
        return False, "prior snapshot changed"
    return True, "s12_p8 only"


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex, pdf = root / "paper/main.tex", root / "paper/main.pdf"
    tex_b, pdf_b = _sha(tex), _sha(pdf)
    s04_b = _sha(root / "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv")
    s07_b = _sha(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv")
    s05_b = _sha(root / "artifacts/s05/S05_SUPPORT_MASKS.parquet")
    s06_b = _sha(root / "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet")
    p7_b = _sha(root / "artifacts/s12/p7/S12_P7_SUMMARY.json")

    out = root / "artifacts/s12/p8"
    models_dir = out / "models"
    sci = run_science(root, models_dir)
    summary = sci["summary"]
    if not summary.get("reconciliation_ok"):
        add("independent_reconciliation", False, str(summary.get("reconciliation_max_abs_discrepancy")))
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p8": {"status": "STOP", "failed": ["independent_reconciliation"], "reconciliation_max_abs_discrepancy": summary.get("reconciliation_max_abs_discrepancy")}}}, validations=checks)

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "S12_P8_HYPERPARAMETER_CANDIDATES.csv", sci["candidates"])
    write_csv(out / "S12_P8_HYPERPARAMETER_SELECTION.csv", sci["selection"])
    write_csv(out / "S12_P8_TRAINING_DIAGNOSTICS.csv", sci["training"])
    write_csv(out / "S12_P8_DIRECTIONAL.csv", sci["directional"])
    write_csv(out / "S12_P8_TARGET_SUMMARY.csv", sci["target_summary"])
    write_parquet(out / "S12_P8_TEST_ROWS.parquet", sci["test_rows"])
    write_csv(out / "S12_P8_BOOTSTRAP.csv", sci["bootstrap"])
    dump_json(out / "S12_P8_SUMMARY.json", summary)
    md = root / "reports/scientific/S12_P8_POOLED_FLEET_BASELINE.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_report(summary), encoding="utf-8")
    if sci["common_row"]:
        write_csv(out / "S12_P8_COMMON_ROW_BEST_TWIN.csv", sci["common_row"])

    recs = [
        artifact(root, out / "S12_P8_HYPERPARAMETER_CANDIDATES.csv", rows=len(sci["candidates"]), fp=csv_fingerprint(out / "S12_P8_HYPERPARAMETER_CANDIDATES.csv")),
        artifact(root, out / "S12_P8_HYPERPARAMETER_SELECTION.csv", rows=len(sci["selection"]), fp=csv_fingerprint(out / "S12_P8_HYPERPARAMETER_SELECTION.csv")),
        artifact(root, out / "S12_P8_TRAINING_DIAGNOSTICS.csv", rows=len(sci["training"]), fp=csv_fingerprint(out / "S12_P8_TRAINING_DIAGNOSTICS.csv")),
        artifact(root, out / "S12_P8_DIRECTIONAL.csv", rows=len(sci["directional"]), fp=csv_fingerprint(out / "S12_P8_DIRECTIONAL.csv")),
        artifact(root, out / "S12_P8_TARGET_SUMMARY.csv", rows=len(sci["target_summary"]), fp=csv_fingerprint(out / "S12_P8_TARGET_SUMMARY.csv")),
        artifact(root, out / "S12_P8_TEST_ROWS.parquet", rows=len(sci["test_rows"])),
        artifact(root, out / "S12_P8_BOOTSTRAP.csv", rows=len(sci["bootstrap"]), fp=csv_fingerprint(out / "S12_P8_BOOTSTRAP.csv")),
        artifact(root, out / "S12_P8_SUMMARY.json"),
        artifact(root, md),
    ]
    if sci["common_row"]:
        recs.append(artifact(root, out / "S12_P8_COMMON_ROW_BEST_TWIN.csv", rows=len(sci["common_row"]), fp=csv_fingerprint(out / "S12_P8_COMMON_ROW_BEST_TWIN.csv")))
    for mp in sci["model_paths"]:
        recs.append(artifact(root, mp))

    leak = summary["leakage"]
    prov_rel = "artifacts/s12/p8/S12_P8_PROVENANCE.json"
    provenance = {
        "repository_head": _git_head(root),
        "canonical_inputs": {
            "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv": s04_b,
            "artifacts/s04/S04_ANALYTIC_INDEX.parquet": sha256_file(root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet"),
            "artifacts/p02c/P02C_PLANT_PANEL.parquet": sha256_file(root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"),
            "artifacts/s05/S05_SUPPORT_MASKS.parquet": s05_b,
            "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet": s06_b,
            "artifacts/s07/S07_OTG_DIRECTIONAL.csv": s07_b,
            "config/protocol.py": sha256_file(root / "config/protocol.py"),
        },
        "primary_cohort": summary["plants"],
        "donor_exclusion": "leave_target_out_other_13_primary_plants",
        "six_feature_schema": summary["features"],
        "candidate_construction": "unique S04 selected SplineRidge full tuples across 14 primary plants",
        "selection_rule": "equal mean of donor-plant VALIDATION MAE; lexicographic hyperparameter serialization tie-break",
        "row_pooled_fit": "ordinary row-level SplineRidge on donor TRAIN",
        "plant_balanced_weights": "w_ij=1/n_j on donor TRAIN; same vector to scaler__sample_weight and ridge__sample_weight; SplineTransformer uniform knots unweighted; no silent row-pooled fallback",
        "test_row_reuse": "exact accepted S06/S07 CS4 primary TEST identities; no pooled support mask",
        "bootstrap": summary["bootstrap"],
        "script_path": "src/run/27_s12_p8_pooled.py",
        "helper_path": "src/lib/s12_p8_pooled.py",
        "generated_at_utc": now,
        "no_target_leakage": (not leak["target_in_train"]) and (not leak["target_in_val"]) and leak["n_targets_fit"] == 14,
        "accepted_science_not_modified": True,
        "manuscript_files_not_edited": True,
        "self_hash_registered_externally": True,
        "final_model_paths": [p.relative_to(root).as_posix() if p.is_absolute() else str(p) for p in sci["model_paths"]],
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes, "row_count": rec.row_count} for rec in recs},
    }
    dump_json(out / "S12_P8_PROVENANCE.json", provenance)
    recs.append(artifact(root, out / "S12_P8_PROVENANCE.json"))

    add("n_14_loto", leak["n_targets_fit"] == 14 and not leak["donors_not_13"], str(leak))
    add("zero_target_train_val", (not leak["target_in_train"]) and (not leak["target_in_val"]), "loto")
    add("n_6_candidates", summary["n_candidates"] == 6, str(summary["n_candidates"]))
    add("n_28_models", summary["n_models"] == 28, str(summary["n_models"]))
    add("n_85", summary["n_directional"] == 85, str(summary["n_directional"]))
    add("twin_rq1", abs(float(summary["twin_rq1"]["plant_balanced"]) - PRIMARY_RQ1) <= 1e-12, str(summary["twin_rq1"]["plant_balanced"]))
    add("independent_reconciliation", summary["reconciliation_ok"], str(summary["reconciliation_max_abs_discrepancy"]))
    add("rq2_not_redefined", summary["rq2_policy"] == RQ2_POLICY, summary["rq2_policy"])
    add("rq3_not_proxied", summary["rq3_status"] == RQ3_STATUS, summary["rq3_status"])
    add("no_id_features", leak.get("id_features_used") is False, str(summary["features"]))
    add("bootstrap_5000", summary["bootstrap"]["n_boot"] == 5000 and summary["bootstrap"]["seed"] == 20260921, str(summary["bootstrap"]["n_boot"]))
    add("weighting_scaler_and_ridge", summary.get("weighting_ok") is True, str(summary.get("weighting_implementation")))
    add("row_pooled_frozen", summary.get("row_pooled_unchanged") is True, str(summary.get("row_pooled_rq1", {}).get("plant_balanced")))
    add("no_p9", list((root / "src/run").glob("28_s12*")) == [], "stop after P8")
    add("no_thresholds", summary["interpretation"]["threshold_rules_used"] is False, "ok")
    add("s04_unchanged", s04_b == _sha(root / "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv"), s04_b)
    add("s07_unchanged", s07_b == _sha(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"), s07_b)
    add("s05_unchanged", s05_b == _sha(root / "artifacts/s05/S05_SUPPORT_MASKS.parquet"), s05_b)
    add("s06_unchanged", s06_b == _sha(root / "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet"), s06_b)
    add("s12_p7_unchanged", p7_b == _sha(root / "artifacts/s12/p7/S12_P7_SUMMARY.json"), p7_b)
    add("provenance_omits_self", prov_rel not in provenance["outputs"], "ok")
    add("paper_tex_unchanged", tex_b == _sha(tex), _sha(tex))
    add("paper_pdf_unchanged", pdf_b == _sha(pdf), _sha(pdf))

    payload = {
        "status": STATUS,
        "n_candidates": summary["n_candidates"],
        "n_models": summary["n_models"],
        "n_directional": summary["n_directional"],
        "twin_pb_otg": summary["twin_rq1"]["plant_balanced"],
        "row_pooled_pb_otg": summary["row_pooled_rq1"]["plant_balanced"],
        "plant_balanced_pb_otg": summary["plant_balanced_rq1"]["plant_balanced"],
        "pb_gain_row": summary["pb_gain_row"],
        "pb_gain_bal": summary["pb_gain_bal"],
        "gain_row": summary["gain_row"],
        "gain_bal": summary["gain_bal"],
        "bootstrap": summary["bootstrap"],
        "groups": summary["groups"],
        "selection": summary["selection"],
        "leakage": leak,
        "weighting_ok": summary.get("weighting_ok"),
        "weighting_implementation": summary.get("weighting_implementation"),
        "row_pooled_unchanged": summary.get("row_pooled_unchanged"),
        "rq2_policy": RQ2_POLICY,
        "rq3_status": RQ3_STATUS,
        "interpretation": summary["interpretation"],
        "common_row_n_targets": len(summary["common_row_best_twin"]),
        "candidates_path": "artifacts/s12/p8/S12_P8_HYPERPARAMETER_CANDIDATES.csv",
        "candidates_sha256": recs[0].sha256,
        "selection_path": "artifacts/s12/p8/S12_P8_HYPERPARAMETER_SELECTION.csv",
        "selection_sha256": recs[1].sha256,
        "training_path": "artifacts/s12/p8/S12_P8_TRAINING_DIAGNOSTICS.csv",
        "training_sha256": recs[2].sha256,
        "directional_path": "artifacts/s12/p8/S12_P8_DIRECTIONAL.csv",
        "directional_sha256": recs[3].sha256,
        "target_summary_path": "artifacts/s12/p8/S12_P8_TARGET_SUMMARY.csv",
        "target_summary_sha256": recs[4].sha256,
        "test_rows_path": "artifacts/s12/p8/S12_P8_TEST_ROWS.parquet",
        "test_rows_sha256": recs[5].sha256,
        "bootstrap_path": "artifacts/s12/p8/S12_P8_BOOTSTRAP.csv",
        "bootstrap_sha256": recs[6].sha256,
        "summary_path": "artifacts/s12/p8/S12_P8_SUMMARY.json",
        "summary_sha256": recs[7].sha256,
        "report_path": "reports/scientific/S12_P8_POOLED_FLEET_BASELINE.md",
        "report_sha256": recs[8].sha256,
        "provenance_path": prov_rel,
        "provenance_sha256": recs[-1].sha256,
        "failed": None,
    }
    pre = _snap(ctx.sot)
    patch = ledger_patch(stage="s12_p8", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p8": payload}})
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_rq_unchanged", _nested(merged, ("results", "tables", "rq1")) == pre.get("results.tables.rq1"), "rq1")
    add("s12_p7_sot_unchanged", _nested(merged, ("results", "s12_p7", "status")) == pre.get("results.s12_p7.status"), "p7")
    payload["n_validations"] = len(checks)
    patch = ledger_patch(stage="s12_p8", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p8": payload}})
    if not ns_ok:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p8": {"status": "STOP", "reason": ns_detail}}}, artifacts=recs, validations=checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p8": {"status": "STOP", "failed": [c.name for c in failed]}}}, artifacts=recs, validations=checks)
    return StageResult(status="GO", message=payload["status"], sot_patch=patch, artifacts=recs, validations=checks)
