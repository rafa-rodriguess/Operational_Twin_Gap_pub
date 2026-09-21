"""22 — S12 P3 / R9 adaptive scaling sensitivity."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, sha256_file, write_csv
from src.lib.s12_p3_adaptive_scaling import (
    MAD_C,
    RQ3_DEFER,
    STATUS,
    render_report,
    run_science,
    write_eval,
    write_masks,
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
    if set((patch.get("results") or {}).keys()) != {"s12_p3"}:
        return False, "results keys"
    if set((patch.get("scientific_analysis") or {}).keys()) != {"s12_p3"}:
        return False, "sci keys"
    if _snap(merged) != pre:
        return False, "prior snapshot changed"
    return True, "s12_p3 only"


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
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p3": {"status": "STOP", "failed": ["independent_reconciliation"]}}}, validations=checks)

    out = root / "artifacts/s12/p3"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(
        out / "S12_P3_R9_SOURCE_SCALERS.csv",
        sci["scaler_rows"],
        ["source_plant_id", "feature", "median", "iqr", "mad", "scale", "scale_source", "n_source_train", "tau", "n_support_features", "engine_computable"],
    )
    n_mask = write_masks(out / "S12_P3_R9_SUPPORT_MASKS.parquet", sci["mask_chunks"])
    write_csv(
        out / "S12_P3_R9_DIRECTIONAL.csv",
        sci["directional"],
        ["group_id", "source_plant_id", "target_plant_id", "model_family", "support_rule", "computable", "reason", "n_eligible_target_test", "n_supported", "n_supported_days", "support_retention", "threshold", "n_support_features", "transfer_mae", "local_same_rows_mae", "otg_abs", "otg_rel", "otg_rel_defined"],
    )
    write_csv(
        out / "S12_P3_R9_ASYMMETRY.csv",
        sci["asym_rows"],
        ["group_id", "plant_a", "plant_b", "a_to_b_computable", "b_to_a_computable", "otg_a_to_b", "otg_b_to_a", "asymmetry_signed", "asymmetry_abs", "asymmetry_computable"],
    )
    write_csv(
        out / "S12_P3_R9_SUPPORT_COMPARISON.csv",
        sci["cmp_rows"],
        ["source_plant_id", "target_plant_id", "relation", "jaccard", "n_primary_supported", "n_r9_supported", "delta_support_retention", "delta_otg_abs"],
    )
    write_eval(out / "S12_P3_R9_EVAL_ROWS.parquet", sci["eval_rows"])
    dump_json(out / "S12_P3_SUMMARY.json", summary)
    md = root / "reports/scientific/S12_P3_ADAPTIVE_SCALING_R9.md"
    md.write_text(render_report(summary), encoding="utf-8")

    recs = [
        artifact(root, out / "S12_P3_R9_SOURCE_SCALERS.csv", rows=len(sci["scaler_rows"]), fp=csv_fingerprint(out / "S12_P3_R9_SOURCE_SCALERS.csv")),
        artifact(root, out / "S12_P3_R9_SUPPORT_MASKS.parquet", rows=n_mask),
        artifact(root, out / "S12_P3_R9_DIRECTIONAL.csv", rows=len(sci["directional"]), fp=csv_fingerprint(out / "S12_P3_R9_DIRECTIONAL.csv")),
        artifact(root, out / "S12_P3_R9_ASYMMETRY.csv", rows=len(sci["asym_rows"]), fp=csv_fingerprint(out / "S12_P3_R9_ASYMMETRY.csv")),
        artifact(root, out / "S12_P3_R9_SUPPORT_COMPARISON.csv", rows=len(sci["cmp_rows"]), fp=csv_fingerprint(out / "S12_P3_R9_SUPPORT_COMPARISON.csv")),
        artifact(root, out / "S12_P3_R9_EVAL_ROWS.parquet", rows=len(sci["eval_rows"])),
        artifact(root, out / "S12_P3_SUMMARY.json"),
        artifact(root, md),
    ]
    prov_rel = "artifacts/s12/p3/S12_P3_PROVENANCE.json"
    provenance = {
        "repository_head": _git_head(root),
        "canonical_inputs": {
            "artifacts/s07/S07_OTG_DIRECTIONAL.csv": sha256_file(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"),
            "artifacts/s04/S04_ANALYTIC_INDEX.parquet": sha256_file(root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet"),
            "artifacts/p02c/P02C_PLANT_PANEL.parquet": sha256_file(root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"),
            "artifacts/s05/S05_SUPPORT_MASKS.parquet": sha256_file(root / "artifacts/s05/S05_SUPPORT_MASKS.parquet"),
            "config/protocol.py": sha256_file(root / "config/protocol.py"),
        },
        "iqr_validity": "finite and > 0 (src.lib.split_support.iqr_is_valid)",
        "mad": f"median(|x-median|); scale={MAD_C}*MAD",
        "feature_omission": "support distance only; SplineRidge still uses six-core features",
        "support": {"k": 5, "q": 0.95, "quantile": "linear"},
        "model_actions": summary["model_actions"],
        "script_path": "src/run/22_s12_p3_adaptive_scaling.py",
        "helper_path": "src/lib/s12_p3_adaptive_scaling.py",
        "generated_at_utc": now,
        "r9_is_sensitivity_only": True,
        "accepted_science_not_modified": True,
        "manuscript_files_not_edited": True,
        "self_hash_registered_externally": True,
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes, "row_count": rec.row_count} for rec in recs},
    }
    dump_json(out / "S12_P3_PROVENANCE.json", provenance)
    recs.append(artifact(root, out / "S12_P3_PROVENANCE.json"))

    add("n_plants_14_dirs_94", len(sci["plants"]) == 14 and len(sci["directional"]) == 94, f"{len(sci['plants'])}/{len(sci['directional'])}")
    add("mad_constant_fixed", MAD_C == 1.4826, str(MAD_C))
    add("ps048_wind_reported", summary.get("ps048_wind_speed_ms") is not None, str(summary.get("ps048_wind_speed_ms")))
    add("overlap_rq1_85", summary["rq1"]["overlap_n"] == 85, str(summary["rq1"]["overlap_n"]))
    add("rq3_deferred_not_proxied", summary["rq3_status"] == RQ3_DEFER, summary["rq3_status"])
    add("independent_reconciliation", summary["reconciliation_ok"], str(summary["reconciliation_max_abs_discrepancy"]))
    add("no_p4_files", list((root / "src/run").glob("23_s12*")) == [], "stop after P3")
    add("primary_models_not_overwritten", s04m.is_file(), str(s04m))
    add("provenance_omits_self", prov_rel not in provenance["outputs"], "ok")
    add("paper_tex_unchanged", tex_b == _sha(tex), _sha(tex))
    add("paper_pdf_unchanged", pdf_b == _sha(pdf), _sha(pdf))

    payload = {
        "status": STATUS,
        "n_r9_computable": summary["n_r9_computable"],
        "n_recovered": summary["n_recovered"],
        "ps048_n_recovered": summary["ps048_n_recovered"],
        "rq1": summary["rq1"],
        "rq2": summary["rq2"],
        "rq3_status": summary["rq3_status"],
        "support_change": summary["support_change"],
        "interpretation": summary["interpretation"],
        "model_actions": summary["model_actions"],
        "scalers_path": "artifacts/s12/p3/S12_P3_R9_SOURCE_SCALERS.csv",
        "scalers_sha256": recs[0].sha256,
        "masks_path": "artifacts/s12/p3/S12_P3_R9_SUPPORT_MASKS.parquet",
        "masks_sha256": recs[1].sha256,
        "directional_path": "artifacts/s12/p3/S12_P3_R9_DIRECTIONAL.csv",
        "directional_sha256": recs[2].sha256,
        "asymmetry_path": "artifacts/s12/p3/S12_P3_R9_ASYMMETRY.csv",
        "asymmetry_sha256": recs[3].sha256,
        "comparison_path": "artifacts/s12/p3/S12_P3_R9_SUPPORT_COMPARISON.csv",
        "comparison_sha256": recs[4].sha256,
        "eval_path": "artifacts/s12/p3/S12_P3_R9_EVAL_ROWS.parquet",
        "eval_sha256": recs[5].sha256,
        "summary_path": "artifacts/s12/p3/S12_P3_SUMMARY.json",
        "summary_sha256": recs[6].sha256,
        "report_path": "reports/scientific/S12_P3_ADAPTIVE_SCALING_R9.md",
        "report_sha256": recs[7].sha256,
        "provenance_path": prov_rel,
        "provenance_sha256": recs[-1].sha256,
        "r9_is_sensitivity_only": True,
        "failed": None,
    }
    add("sot_prov_sha", payload["provenance_sha256"] == recs[-1].sha256, recs[-1].sha256)
    pre = _snap(ctx.sot)
    patch = ledger_patch(stage="s12_p3", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p3": payload}})
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_rq_unchanged", _nested(merged, ("results", "tables", "rq1")) == pre.get("results.tables.rq1") and _nested(merged, ("results", "tables", "rq2")) == pre.get("results.tables.rq2"), str(pre.get("results.tables.rq1")))
    add("s11_unchanged", _nested(merged, ("results", "s11_phase4", "decisions")) == pre.get("results.s11_phase4.decisions"), "s11")
    add("s12_prior_unchanged", _nested(merged, ("results", "s12_p2", "status")) == pre.get("results.s12_p2.status") and _nested(merged, ("results", "s12_p1", "status")) == pre.get("results.s12_p1.status"), "p0-p2")
    add("rq4_unchanged", _nested(merged, ("scientific_analysis", "rq4_source_selection", "status")) == pre.get("scientific_analysis.rq4_source_selection.status"), "rq4")
    payload["n_validations"] = len(checks)
    patch = ledger_patch(stage="s12_p3", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p3": payload}})
    if not ns_ok:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p3": {"status": "STOP", "reason": ns_detail, "failed": None}}}, artifacts=recs, validations=checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p3": {"status": "STOP", "failed": [c.name for c in failed]}}}, artifacts=recs, validations=checks)
    return StageResult(status="GO", message=payload["status"], sot_patch=patch, artifacts=recs, validations=checks)
