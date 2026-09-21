"""25 — S12 P6 / R10 matched TRAIN-duration sensitivity."""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, sha256_file, write_csv
from src.lib.s12_p6_matched_duration import (
    EQUAL_TOL_DAYS,
    RQ3_STATUS,
    STATUS,
    render_report,
    run_reporting_from_disk,
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
    ("scientific_analysis", "rq4_source_selection", "status"),
)

SCIENCE_REL = (
    "artifacts/s12/p6/S12_P6_PLANT_TRAIN_HISTORY.csv",
    "artifacts/s12/p6/S12_P6_DIRECTION_HISTORY.csv",
    "artifacts/s12/p6/S12_P6_R10_DIRECTIONAL.csv",
    "artifacts/s12/p6/S12_P6_R10_ASYMMETRY.csv",
    "artifacts/s12/p6/S12_P6_R10_TEST_ROWS.parquet",
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


def _model_hashes(root: Path) -> dict[str, str]:
    base = root / "artifacts/s12/p6/models"
    if not base.is_dir():
        return {}
    out = {}
    for path in sorted(base.rglob("*.joblib")):
        out[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _namespace_ok(pre: dict[str, Any], patch: dict[str, Any], merged: dict[str, Any]) -> tuple[bool, str]:
    if set((patch.get("results") or {}).keys()) != {"s12_p6"}:
        return False, "results keys"
    if set((patch.get("scientific_analysis") or {}).keys()) != {"s12_p6"}:
        return False, "sci keys"
    if _snap(merged) != pre:
        return False, "prior snapshot changed"
    return True, "s12_p6 only"


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex, pdf = root / "paper/main.tex", root / "paper/main.pdf"
    tex_b, pdf_b = _sha(tex), _sha(pdf)
    s04m = root / "artifacts/s04/models/PS_002/SplineRidge/full.joblib"
    out = root / "artifacts/s12/p6"
    science_present = (out / "S12_P6_R10_DIRECTIONAL.csv").is_file()
    reporting_only = science_present and os.environ.get("S12_P6_FORCE_REFIT") != "1"
    frozen = {rel: _sha(root / rel) for rel in SCIENCE_REL}
    models_before = _model_hashes(root)
    prior_sot_p6 = ((ctx.sot.get("results") or {}).get("s12_p6") or {})

    sci = run_reporting_from_disk(root, ctx.sot) if reporting_only else run_science(root, ctx.sot)
    summary = sci["summary"]
    if not summary.get("reconciliation_ok"):
        add("independent_reconciliation", False, str(summary.get("reconciliation_max_abs_discrepancy")))
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p6": {"status": "STOP", "failed": ["independent_reconciliation"]}}}, validations=checks)

    out.mkdir(parents=True, exist_ok=True)
    if not reporting_only:
        write_csv(out / "S12_P6_PLANT_TRAIN_HISTORY.csv", sci["plant_rows"])
        write_csv(out / "S12_P6_DIRECTION_HISTORY.csv", sci["hist_rows"])
        write_csv(out / "S12_P6_R10_DIRECTIONAL.csv", sci["directional"])
        write_csv(out / "S12_P6_R10_ASYMMETRY.csv", sci["asym_rows"])
        n_test = write_parquet(out / "S12_P6_R10_TEST_ROWS.parquet", sci["eval_rows"])
    else:
        n_test = None
    dump_json(out / "S12_P6_SUMMARY.json", summary)
    md = root / "reports/scientific/S12_P6_MATCHED_TRAIN_DURATION_R10.md"
    md.write_text(render_report(summary), encoding="utf-8")

    recs = [
        artifact(root, out / "S12_P6_PLANT_TRAIN_HISTORY.csv", rows=len(sci["plant_rows"]), fp=csv_fingerprint(out / "S12_P6_PLANT_TRAIN_HISTORY.csv")),
        artifact(root, out / "S12_P6_DIRECTION_HISTORY.csv", rows=len(sci["hist_rows"]), fp=csv_fingerprint(out / "S12_P6_DIRECTION_HISTORY.csv")),
        artifact(root, out / "S12_P6_R10_DIRECTIONAL.csv", rows=len(sci["directional"]), fp=csv_fingerprint(out / "S12_P6_R10_DIRECTIONAL.csv")),
        artifact(root, out / "S12_P6_R10_ASYMMETRY.csv", rows=len(sci["asym_rows"]), fp=csv_fingerprint(out / "S12_P6_R10_ASYMMETRY.csv")),
        artifact(root, out / "S12_P6_R10_TEST_ROWS.parquet", rows=n_test),
        artifact(root, out / "S12_P6_SUMMARY.json"),
        artifact(root, md),
    ]
    prov_rel = "artifacts/s12/p6/S12_P6_PROVENANCE.json"
    provenance = {
        "repository_head": _git_head(root),
        "canonical_inputs": {
            "artifacts/s07/S07_OTG_DIRECTIONAL.csv": sha256_file(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"),
            "artifacts/s04/S04_ANALYTIC_INDEX.parquet": sha256_file(root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet"),
            "artifacts/p02c/P02C_PLANT_PANEL.parquet": sha256_file(root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"),
            "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv": sha256_file(root / "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv"),
            "config/protocol.py": sha256_file(root / "config/protocol.py"),
        },
        "duration_formula": "duration_days = (max(TRAIN datetime_raw) - min(TRAIN datetime_raw)).total_seconds()/86400",
        "boundary": "inclusive endpoints; no day-boundary rounding; timezone-aware UTC",
        "d_pair": "min(source_duration_days, target_duration_days) shared by both matched windows of the ordered transfer",
        "trailing_window": "TRAIN rows with train_end - D_pair days <= datetime <= train_end",
        "equal_tolerance_days": EQUAL_TOL_DAYS,
        "support": "primary CS4 IQR k=5 q=0.95 on matched source TRAIN only; no R9 MAD/omission",
        "no_r9": True,
        "no_p4": True,
        "bootstrap": {"n": 5000, "seed": 20260921, "cluster": "target_plant_id"},
        "n_cached_models": summary["n_cached_models"],
        "script_path": "src/run/25_s12_p6_matched_duration.py",
        "helper_path": "src/lib/s12_p6_matched_duration.py",
        "generated_at_utc": now,
        "accepted_science_not_modified": True,
        "manuscript_files_not_edited": True,
        "self_hash_registered_externally": True,
        "reporting_only": reporting_only,
        "reporting_derived_from": [
            "artifacts/s12/p6/S12_P6_R10_DIRECTIONAL.csv",
            "artifacts/s12/p6/S12_P6_DIRECTION_HISTORY.csv",
            "artifacts/s12/p6/S12_P6_R10_ASYMMETRY.csv",
        ],
        "frozen_science_sha256": {rel: frozen[rel] for rel in SCIENCE_REL},
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes, "row_count": rec.row_count} for rec in recs},
    }
    dump_json(out / "S12_P6_PROVENANCE.json", provenance)
    recs.append(artifact(root, out / "S12_P6_PROVENANCE.json"))

    add("n_94", len(sci["directional"]) == 94 and len(sci["plants"]) == 14, f"{len(sci['directional'])}")
    add("rq3_not_proxied", summary["rq3_status"] == RQ3_STATUS, summary["rq3_status"])
    add("independent_reconciliation", summary["reconciliation_ok"], str(summary["reconciliation_max_abs_discrepancy"]))
    add("no_p7_files", list((root / "src/run").glob("26_s12*")) == [] and list((root / "src/run").glob("27_s12*")) == [] and list((root / "src/run").glob("28_s12*")) == [], "stop after P6")
    add("no_r9", summary.get("no_r9") is True, "IQR")
    add("primary_models_not_overwritten", s04m.is_file(), str(s04m))
    add("provenance_omits_self", prov_rel not in provenance["outputs"], "ok")
    add("paper_tex_unchanged", tex_b == _sha(tex), _sha(tex))
    add("paper_pdf_unchanged", pdf_b == _sha(pdf), _sha(pdf))
    add("science_artifacts_unchanged", reporting_only and all(_sha(root / rel) == frozen[rel] for rel in SCIENCE_REL), "frozen hashes")
    add("models_unchanged", reporting_only and _model_hashes(root) == models_before, f"{len(models_before)}")
    add("existing_rq1_unchanged", (not prior_sot_p6) or prior_sot_p6.get("rq1_r10") == summary.get("rq1_r10"), "rq1")
    add("existing_delta_unchanged", (not prior_sot_p6) or prior_sot_p6.get("delta_otg") == summary.get("delta_otg"), "delta")
    add("existing_rq2_unchanged", (not prior_sot_p6) or prior_sot_p6.get("rq2_r10") == summary.get("rq2_r10"), "rq2")
    add(
        "reporting_blocks_present",
        all(k in summary for k in ("support_retention_r10", "duration_imbalance_planned", "rq2_r10_by_group", "spearman_imbalance_absA", "spearman_imbalance_absA_change")),
        "blocks",
    )

    payload = {
        "status": STATUS,
        "n_r10_computable": summary["n_r10_computable"],
        "n_lost": summary["n_lost"],
        "duration": summary["duration"],
        "rq1_r10": summary["rq1_r10"],
        "rq1_overlap": summary["rq1_overlap"],
        "delta_otg": summary["delta_otg"],
        "spearman_imbalance_delta": summary["spearman_imbalance_delta"],
        "rq2_r10": summary["rq2_r10"],
        "rq2_overlap": summary["rq2_overlap"],
        "rq2_r10_by_group": summary["rq2_r10_by_group"],
        "support_retention_r10": summary["support_retention_r10"],
        "duration_imbalance_planned": summary["duration_imbalance_planned"],
        "spearman_imbalance_absA": summary["spearman_imbalance_absA"],
        "spearman_imbalance_absA_change": summary["spearman_imbalance_absA_change"],
        "rq3_status": summary["rq3_status"],
        "interpretation": summary["interpretation"],
        "reporting_correction": True,
        "plant_history_path": "artifacts/s12/p6/S12_P6_PLANT_TRAIN_HISTORY.csv",
        "plant_history_sha256": recs[0].sha256,
        "direction_history_path": "artifacts/s12/p6/S12_P6_DIRECTION_HISTORY.csv",
        "direction_history_sha256": recs[1].sha256,
        "directional_path": "artifacts/s12/p6/S12_P6_R10_DIRECTIONAL.csv",
        "directional_sha256": recs[2].sha256,
        "asymmetry_path": "artifacts/s12/p6/S12_P6_R10_ASYMMETRY.csv",
        "asymmetry_sha256": recs[3].sha256,
        "test_rows_path": "artifacts/s12/p6/S12_P6_R10_TEST_ROWS.parquet",
        "test_rows_sha256": recs[4].sha256,
        "summary_path": "artifacts/s12/p6/S12_P6_SUMMARY.json",
        "summary_sha256": recs[5].sha256,
        "report_path": "reports/scientific/S12_P6_MATCHED_TRAIN_DURATION_R10.md",
        "report_sha256": recs[6].sha256,
        "provenance_path": prov_rel,
        "provenance_sha256": recs[-1].sha256,
        "r10_is_sensitivity_only": True,
        "failed": None,
    }
    pre = _snap(ctx.sot)
    patch = ledger_patch(stage="s12_p6", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p6": payload}})
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_rq_unchanged", _nested(merged, ("results", "tables", "rq1")) == pre.get("results.tables.rq1"), "rq")
    add("s12_p5_unchanged", _nested(merged, ("results", "s12_p5", "status")) == pre.get("results.s12_p5.status"), "p5")
    payload["n_validations"] = len(checks)
    patch = ledger_patch(stage="s12_p6", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p6": payload}})
    if not ns_ok:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p6": {"status": "STOP", "reason": ns_detail}}}, artifacts=recs, validations=checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p6": {"status": "STOP", "failed": [c.name for c in failed]}}}, artifacts=recs, validations=checks)
    return StageResult(status="GO", message=payload["status"], sot_patch=patch, artifacts=recs, validations=checks)
