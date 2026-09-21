"""24 — S12 P5 plant-delete jackknife and G2/non-G2 stability."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, sha256_file, write_csv
from src.lib.s12_p5_stability import RQ3_STATUS, STATUS, render_report, run_science
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
    if set((patch.get("results") or {}).keys()) != {"s12_p5"}:
        return False, "results keys"
    if set((patch.get("scientific_analysis") or {}).keys()) != {"s12_p5"}:
        return False, "sci keys"
    if _snap(merged) != pre:
        return False, "prior snapshot changed"
    return True, "s12_p5 only"


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex, pdf = root / "paper/main.tex", root / "paper/main.pdf"
    tex_b, pdf_b = _sha(tex), _sha(pdf)
    lib_src = (root / "src/lib/s12_p5_stability.py").read_text(encoding="utf-8")
    stg_src = (root / "src/run/24_s12_p5_stability.py").read_text(encoding="utf-8")
    add("no_fit", ".fit(" not in lib_src, "no fit in helper")
    add("no_sklearn_joblib", "sklearn" not in lib_src and "joblib" not in lib_src, "metrics-only")
    add("no_p4_r9_tokens_as_input", "S12_P4" not in lib_src and "S12_P3_R9" not in lib_src, "primary S07 only")

    sci = run_science(root, ctx.sot)
    summary = sci["summary"]
    if not summary.get("reconciliation_ok"):
        add("independent_reconciliation", False, str(summary.get("reconciliation_max_abs_discrepancy")))
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p5": {"status": "STOP", "failed": ["independent_reconciliation"]}}}, validations=checks)

    out = root / "artifacts/s12/p5"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "S12_P5_PLANT_DELETE_RQ1.csv", sci["rq1_reps"])
    write_csv(out / "S12_P5_PLANT_DELETE_RQ2.csv", sci["rq2_reps"])
    write_csv(out / "S12_P5_G2_DELETE.csv", sci["g2_del"])
    write_csv(out / "S12_P5_NONG2_DELETE.csv", sci["ng_del"])
    write_csv(out / "S12_P5_SUBGROUP_SUMMARY.csv", sci["subgroup_rows"])
    dump_json(out / "S12_P5_SUMMARY.json", summary)
    md = root / "reports/scientific/S12_P5_PLANT_DELETE_STABILITY.md"
    md.write_text(render_report(summary), encoding="utf-8")

    recs = [
        artifact(root, out / "S12_P5_PLANT_DELETE_RQ1.csv", rows=len(sci["rq1_reps"]), fp=csv_fingerprint(out / "S12_P5_PLANT_DELETE_RQ1.csv")),
        artifact(root, out / "S12_P5_PLANT_DELETE_RQ2.csv", rows=len(sci["rq2_reps"]), fp=csv_fingerprint(out / "S12_P5_PLANT_DELETE_RQ2.csv")),
        artifact(root, out / "S12_P5_G2_DELETE.csv", rows=len(sci["g2_del"]), fp=csv_fingerprint(out / "S12_P5_G2_DELETE.csv")),
        artifact(root, out / "S12_P5_NONG2_DELETE.csv", rows=len(sci["ng_del"]), fp=csv_fingerprint(out / "S12_P5_NONG2_DELETE.csv")),
        artifact(root, out / "S12_P5_SUBGROUP_SUMMARY.csv", rows=len(sci["subgroup_rows"]), fp=csv_fingerprint(out / "S12_P5_SUBGROUP_SUMMARY.csv")),
        artifact(root, out / "S12_P5_SUMMARY.json"),
        artifact(root, md),
    ]
    prov_rel = "artifacts/s12/p5/S12_P5_PROVENANCE.json"
    provenance = {
        "repository_head": _git_head(root),
        "canonical_inputs": {
            "artifacts/s07/S07_OTG_DIRECTIONAL.csv": sha256_file(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"),
            "artifacts/s07/S07_ASYMMETRY.csv": sha256_file(root / "artifacts/s07/S07_ASYMMETRY.csv"),
        },
        "plant_delete_rule": "remove plant from all source and target roles (RQ1) and from both pair members (RQ2)",
        "rq1_balancing": "equal-target: mean of per-target means of directional OTG_abs",
        "rq2_balancing": "plant-balanced mean of per-plant means of |asymmetry| on accepted bidirectional pairs",
        "g2_group_id": "TW_4467a039e00b",
        "nong2_group_ids": ["TW_04b9f5d95694", "TW_88a108921af7"],
        "excluded": ["G3", "R7", "R9", "P4_recalibration", "OTG_half"],
        "no_refit": True,
        "no_new_scoring": True,
        "script_path": "src/run/24_s12_p5_stability.py",
        "helper_path": "src/lib/s12_p5_stability.py",
        "generated_at_utc": now,
        "accepted_science_not_modified": True,
        "manuscript_files_not_edited": True,
        "self_hash_registered_externally": True,
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes, "row_count": rec.row_count} for rec in recs},
    }
    dump_json(out / "S12_P5_PROVENANCE.json", provenance)
    recs.append(artifact(root, out / "S12_P5_PROVENANCE.json"))

    add("n_plants_14", len(sci["plants"]) == 14 and len(sci["rq1_reps"]) == 14 and len(sci["rq2_reps"]) == 14, str(len(sci["rq1_reps"])))
    add("rq3_not_proxied", summary["rq3_status"] == RQ3_STATUS, summary["rq3_status"])
    add("independent_reconciliation", summary["reconciliation_ok"], str(summary["reconciliation_max_abs_discrepancy"]))
    add("no_p6_files", list((root / "src/run").glob("25_s12*")) == [], "stop after P5")
    add("no_interp_thresholds", summary["interpretation"]["threshold_rules_used"] is False, "ok")
    add("provenance_omits_self", prov_rel not in provenance["outputs"], "ok")
    add("paper_tex_unchanged", tex_b == _sha(tex), _sha(tex))
    add("paper_pdf_unchanged", pdf_b == _sha(pdf), _sha(pdf))

    payload = {
        "status": STATUS,
        "n_plants": 14,
        "rq1_full": summary["rq1_full"],
        "rq2_full": summary["rq2_full"],
        "rq1_delete": summary["rq1_delete"],
        "rq2_delete": summary["rq2_delete"],
        "g2_baseline": summary["g2_baseline"],
        "nong2_baseline": summary["nong2_baseline"],
        "g1_baseline": summary["g1_baseline"],
        "g4_baseline": summary["g4_baseline"],
        "g2_delete": summary["g2_delete"],
        "subgroup_vs_full": summary["subgroup_vs_full"],
        "rq3_status": summary["rq3_status"],
        "interpretation": summary["interpretation"],
        "rq1_path": "artifacts/s12/p5/S12_P5_PLANT_DELETE_RQ1.csv",
        "rq1_sha256": recs[0].sha256,
        "rq2_path": "artifacts/s12/p5/S12_P5_PLANT_DELETE_RQ2.csv",
        "rq2_sha256": recs[1].sha256,
        "g2_delete_path": "artifacts/s12/p5/S12_P5_G2_DELETE.csv",
        "g2_delete_sha256": recs[2].sha256,
        "nong2_delete_path": "artifacts/s12/p5/S12_P5_NONG2_DELETE.csv",
        "nong2_delete_sha256": recs[3].sha256,
        "subgroup_path": "artifacts/s12/p5/S12_P5_SUBGROUP_SUMMARY.csv",
        "subgroup_sha256": recs[4].sha256,
        "summary_path": "artifacts/s12/p5/S12_P5_SUMMARY.json",
        "summary_sha256": recs[5].sha256,
        "report_path": "reports/scientific/S12_P5_PLANT_DELETE_STABILITY.md",
        "report_sha256": recs[6].sha256,
        "provenance_path": prov_rel,
        "provenance_sha256": recs[-1].sha256,
        "no_refit": True,
        "failed": None,
    }
    pre = _snap(ctx.sot)
    patch = ledger_patch(stage="s12_p5", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p5": payload}})
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_rq_unchanged", _nested(merged, ("results", "tables", "rq1")) == pre.get("results.tables.rq1"), "rq")
    add("s11_unchanged", _nested(merged, ("results", "s11_phase4", "decisions")) == pre.get("results.s11_phase4.decisions"), "s11")
    add("s12_p4_unchanged", _nested(merged, ("results", "s12_p4", "status")) == pre.get("results.s12_p4.status"), "p4")
    payload["n_validations"] = len(checks)
    patch = ledger_patch(stage="s12_p5", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p5": payload}})
    if not ns_ok:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p5": {"status": "STOP", "reason": ns_detail}}}, artifacts=recs, validations=checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p5": {"status": "STOP", "failed": [c.name for c in failed]}}}, artifacts=recs, validations=checks)
    return StageResult(status="GO", message=payload["status"], sot_patch=patch, artifacts=recs, validations=checks)
