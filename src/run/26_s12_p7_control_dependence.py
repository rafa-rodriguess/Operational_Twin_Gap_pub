"""26 — S12 P7 RQ3 control-identity dependence."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, sha256_file, write_csv
from src.lib.s12_p7_control_dependence import (
    ACCEPTED_R8,
    ORIENTATION,
    PRIMARY_RQ3,
    STATUS,
    render_report,
    run_science,
)
from src.run._ledger import ledger_patch
from src.sot import apply_patch

SNAPSHOT_PATHS = (
    ("results", "tables", "rq1"),
    ("results", "tables", "rq2"),
    ("results", "tables", "rq3"),
    ("results", "s11_phase4", "decisions"),
    ("results", "s11_phase2_a2", "target_balanced_r8_control_minus_twin"),
    ("results", "s12_phase0", "status"),
    ("results", "s12_p1", "status"),
    ("results", "s12_p2", "status"),
    ("results", "s12_p3", "status"),
    ("results", "s12_p4", "status"),
    ("results", "s12_p5", "status"),
    ("results", "s12_p6", "status"),
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
    if set((patch.get("results") or {}).keys()) != {"s12_p7"}:
        return False, "results keys"
    if set((patch.get("scientific_analysis") or {}).keys()) != {"s12_p7"}:
        return False, "sci keys"
    if _snap(merged) != pre:
        return False, "prior snapshot changed"
    return True, "s12_p7 only"


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex, pdf = root / "paper/main.tex", root / "paper/main.pdf"
    tex_b, pdf_b = _sha(tex), _sha(pdf)
    lib_src = (root / "src/lib/s12_p7_control_dependence.py").read_text(encoding="utf-8")
    add("no_model_fit_token", "est.fit" not in lib_src and "sklearn" not in lib_src, "no sklearn")
    add("no_predict_scoring", ".predict(" not in lib_src, "no predict")
    add("no_p4_r9_r10_inputs", "S12_P4" not in lib_src and "S12_P3" not in lib_src and "S12_P6" not in lib_src, "R8/mapping only")

    sci = run_science(root, ctx.sot)
    summary = sci["summary"]
    if not summary.get("reconciliation_ok"):
        add("independent_reconciliation", False, str(summary.get("reconciliation_max_abs_discrepancy")))
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p7": {"status": "STOP", "failed": ["independent_reconciliation"], "reconciliation_max_abs_discrepancy": summary.get("reconciliation_max_abs_discrepancy")}}}, validations=checks)

    out = root / "artifacts/s12/p7"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "S12_P7_TARGET_CONTROL_CELLS.csv", sci["cells"])
    write_csv(out / "S12_P7_TARGET_CONTROL_SENSITIVITY.csv", sci["target_sens"])
    write_csv(out / "S12_P7_STATE_SHARED_ASSIGNMENTS.csv", sci["shared_rows"])
    write_csv(out / "S12_P7_TARGET_SPECIFIC_ASSIGNMENTS.csv", sci["spec_rows"])
    write_csv(out / "S12_P7_LEAVE_ONE_CONTROL_OUT.csv", sci["loco_rows"])
    write_csv(out / "S12_P7_CONTROL_REUSE.csv", sci["reuse"])
    dump_json(out / "S12_P7_SUMMARY.json", summary)
    md = root / "reports/scientific/S12_P7_RQ3_CONTROL_DEPENDENCE.md"
    md.write_text(render_report(summary), encoding="utf-8")

    recs = [
        artifact(root, out / "S12_P7_TARGET_CONTROL_CELLS.csv", rows=len(sci["cells"]), fp=csv_fingerprint(out / "S12_P7_TARGET_CONTROL_CELLS.csv")),
        artifact(root, out / "S12_P7_TARGET_CONTROL_SENSITIVITY.csv", rows=len(sci["target_sens"]), fp=csv_fingerprint(out / "S12_P7_TARGET_CONTROL_SENSITIVITY.csv")),
        artifact(root, out / "S12_P7_STATE_SHARED_ASSIGNMENTS.csv", rows=len(sci["shared_rows"]), fp=csv_fingerprint(out / "S12_P7_STATE_SHARED_ASSIGNMENTS.csv")),
        artifact(root, out / "S12_P7_TARGET_SPECIFIC_ASSIGNMENTS.csv", rows=len(sci["spec_rows"]), fp=csv_fingerprint(out / "S12_P7_TARGET_SPECIFIC_ASSIGNMENTS.csv")),
        artifact(root, out / "S12_P7_LEAVE_ONE_CONTROL_OUT.csv", rows=len(sci["loco_rows"]), fp=csv_fingerprint(out / "S12_P7_LEAVE_ONE_CONTROL_OUT.csv")),
        artifact(root, out / "S12_P7_CONTROL_REUSE.csv", rows=len(sci["reuse"]), fp=csv_fingerprint(out / "S12_P7_CONTROL_REUSE.csv")),
        artifact(root, out / "S12_P7_SUMMARY.json"),
        artifact(root, md),
    ]
    prov_rel = "artifacts/s12/p7/S12_P7_PROVENANCE.json"
    provenance = {
        "repository_head": _git_head(root),
        "canonical_inputs": {
            "artifacts/rq3/mapping.csv": sha256_file(root / "artifacts/rq3/mapping.csv"),
            "artifacts/s11/S11_A2_R8_ELIGIBILITY.csv": sha256_file(root / "artifacts/s11/S11_A2_R8_ELIGIBILITY.csv"),
            "artifacts/s11/S11_A2_R8_COMPARISONS.csv": sha256_file(root / "artifacts/s11/S11_A2_R8_COMPARISONS.csv"),
            "artifacts/s11/S11_A2_R8_TARGET_SUMMARY.csv": sha256_file(root / "artifacts/s11/S11_A2_R8_TARGET_SUMMARY.csv"),
            "artifacts/s11/S11_A2_R8_SUMMARY.json": sha256_file(root / "artifacts/s11/S11_A2_R8_SUMMARY.json"),
        },
        "orientation": ORIENTATION,
        "target_control_rule": "C(t,c)=mean over valid twin-source contrasts contrast_control_minus_twin(t,s,c)",
        "primary_reconstruction_rule": "equal-target mean of C(t, c_primary) for the 10 mapped primary RQ3 targets",
        "state_common_control_rule": "intersection of structurally_executable controls across all primary targets in the state; eligibility only, no outcome filter",
        "state_shared_enumeration": "Cartesian product of one common control per state (RJ, SP, GO); apply that control to every primary target in the state; fleet=equal-target mean of C(t,c)",
        "target_specific_enumeration": "Cartesian product of each target's structurally executable controls independently; fleet=equal-target mean of C(t,c)",
        "loco_rule": "remove one distinct executable control globally; remaining cells averaged equally within target then equally across targets; non-estimable if any target has zero leftover controls",
        "zero_tolerance": 1e-12,
        "no_refit": True,
        "no_scoring": True,
        "no_support_rebuild": True,
        "accepted_science_not_modified": True,
        "manuscript_files_not_edited": True,
        "self_hash_registered_externally": True,
        "script_path": "src/run/26_s12_p7_control_dependence.py",
        "helper_path": "src/lib/s12_p7_control_dependence.py",
        "generated_at_utc": now,
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes, "row_count": rec.row_count} for rec in recs},
    }
    dump_json(out / "S12_P7_PROVENANCE.json", provenance)
    recs.append(artifact(root, out / "S12_P7_PROVENANCE.json"))

    add("n_10_targets", summary["n_primary_targets"] == 10, str(summary["n_primary_targets"]))
    add("orientation", summary["orientation"] == ORIENTATION, summary["orientation"])
    add("n_118_comparisons", summary["n_valid_source_comparisons"] == 118, str(summary["n_valid_source_comparisons"]))
    add("primary_rq3_recon", abs(summary["primary_reconstruction"]["fleet"] - PRIMARY_RQ3) <= 1e-12, str(summary["primary_reconstruction"]["fleet"]))
    add("primary_controls", summary["primary_reconstruction"]["mapping"] == {"RJ": "PS_007", "SP": "PS_008", "GO": "PS_005"}, "map")
    add("r8_accepted_recon", abs(summary["broad_r8"]["accepted_source_level_semantics"] - ACCEPTED_R8) <= 1e-12, str(summary["broad_r8"]["accepted_source_level_semantics"]))
    add("state_shared_32", summary["state_shared"]["n_assignments"] == 32, str(summary["state_shared"]["n_assignments"]))
    add("target_specific_16384", summary["target_specific"]["n_assignments"] == 16384, str(summary["target_specific"]["n_assignments"]))
    add("loco_12", summary["loco"]["n_replicates"] == 12, str(summary["loco"]["n_replicates"]))
    add("independent_reconciliation", summary["reconciliation_ok"], str(summary["reconciliation_max_abs_discrepancy"]))
    add("no_thresholds", summary.get("threshold_rules_used") is False, "ok")
    add("no_p8_p9", list((root / "src/run").glob("27_s12*")) == [] and list((root / "src/run").glob("28_s12*")) == [], "stop after P7")
    add("no_model_files", list((root / "artifacts/s12/p7").glob("**/*.joblib")) == [], "none")
    add("provenance_omits_self", prov_rel not in provenance["outputs"], "ok")
    add("paper_tex_unchanged", tex_b == _sha(tex), _sha(tex))
    add("paper_pdf_unchanged", pdf_b == _sha(pdf), _sha(pdf))

    payload = {
        "status": STATUS,
        "primary_reconstruction": summary["primary_reconstruction"],
        "broad_r8": summary["broad_r8"],
        "state_common_controls": summary["state_common_controls"],
        "state_shared": summary["state_shared"],
        "target_specific": {k: v for k, v in summary["target_specific"].items() if k != "n_controls_by_target"} | {"n_controls_by_target": summary["target_specific"]["n_controls_by_target"]},
        "selected_vs_alternatives": summary["selected_vs_alternatives"],
        "loco": summary["loco"],
        "primary_control_deletions": summary["primary_control_deletions"],
        "sign_stability": summary["sign_stability"],
        "control_reuse_n": summary["control_reuse"]["n_controls"],
        "interpretation": summary["interpretation"],
        "cells_path": "artifacts/s12/p7/S12_P7_TARGET_CONTROL_CELLS.csv",
        "cells_sha256": recs[0].sha256,
        "sensitivity_path": "artifacts/s12/p7/S12_P7_TARGET_CONTROL_SENSITIVITY.csv",
        "sensitivity_sha256": recs[1].sha256,
        "state_shared_path": "artifacts/s12/p7/S12_P7_STATE_SHARED_ASSIGNMENTS.csv",
        "state_shared_sha256": recs[2].sha256,
        "target_specific_path": "artifacts/s12/p7/S12_P7_TARGET_SPECIFIC_ASSIGNMENTS.csv",
        "target_specific_sha256": recs[3].sha256,
        "loco_path": "artifacts/s12/p7/S12_P7_LEAVE_ONE_CONTROL_OUT.csv",
        "loco_sha256": recs[4].sha256,
        "reuse_path": "artifacts/s12/p7/S12_P7_CONTROL_REUSE.csv",
        "reuse_sha256": recs[5].sha256,
        "summary_path": "artifacts/s12/p7/S12_P7_SUMMARY.json",
        "summary_sha256": recs[6].sha256,
        "report_path": "reports/scientific/S12_P7_RQ3_CONTROL_DEPENDENCE.md",
        "report_sha256": recs[7].sha256,
        "provenance_path": prov_rel,
        "provenance_sha256": recs[-1].sha256,
        "failed": None,
    }
    pre = _snap(ctx.sot)
    patch = ledger_patch(stage="s12_p7", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p7": payload}})
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_rq_unchanged", _nested(merged, ("results", "tables", "rq3")) == pre.get("results.tables.rq3"), "rq3")
    add("s11_unchanged", _nested(merged, ("results", "s11_phase4", "decisions")) == pre.get("results.s11_phase4.decisions"), "s11")
    add("s12_p6_unchanged", _nested(merged, ("results", "s12_p6", "status")) == pre.get("results.s12_p6.status"), "p6")
    payload["n_validations"] = len(checks)
    patch = ledger_patch(stage="s12_p7", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p7": payload}})
    if not ns_ok:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p7": {"status": "STOP", "reason": ns_detail}}}, artifacts=recs, validations=checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p7": {"status": "STOP", "failed": [c.name for c in failed]}}}, artifacts=recs, validations=checks)
    return StageResult(status="GO", message=payload["status"], sot_patch=patch, artifacts=recs, validations=checks)
