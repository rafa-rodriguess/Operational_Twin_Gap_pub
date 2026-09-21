"""21 — S12 P2 OTG × support retention / R7."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, sha256_file, write_csv
from src.lib.s12_p2_retention import (
    N_BOOT,
    R7_THRESHOLD,
    RQ3_NC,
    SEED,
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
    ("results", "rq3", "D_fleet_RQ3"),
    ("results", "s11_phase1", "status"),
    ("results", "s11_phase1", "median_spread"),
    ("results", "s11_phase4", "decisions"),
    ("results", "s12_phase0", "status"),
    ("results", "s12_p1", "status"),
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
    if set((patch.get("results") or {}).keys()) != {"s12_p2"}:
        return False, "results keys"
    if set((patch.get("scientific_analysis") or {}).keys()) != {"s12_p2"}:
        return False, "scientific_analysis keys"
    if _snap(merged) != pre:
        return False, "prior scientific snapshot changed"
    return True, "s12_p2 only"


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex_path, pdf_path = root / "paper/main.tex", root / "paper/main.pdf"
    tex_before, pdf_before = _sha(tex_path), _sha(pdf_path)
    s07 = root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"
    s05 = root / "artifacts/s05/S05_SUPPORT_MASKS.parquet"
    s07_h, s05_h = sha256_file(s07), sha256_file(s05)

    src_lib = (root / "src/lib/s12_p2_retention.py").read_text(encoding="utf-8")
    add("no_fit_in_p2_source", "sklearn" not in src_lib and "joblib" not in src_lib and "SplineRidge" not in src_lib, "lib is metrics-only")
    add("no_predict_scoring_call", "y_pred" not in src_lib and "cross_plant" not in src_lib, "no prediction scoring")

    sci = run_science(root, ctx.sot)
    if sci.get("stop"):
        add("science_stop", False, sci.get("detail"))
        return StageResult(
            status="STOP",
            message=sci["stop"],
            sot_patch={"results": {"s12_p2": {"status": "STOP", "reason": sci.get("detail"), "failed": [sci["stop"]]}}},
            validations=checks,
        )
    summary = sci["summary"]
    if not summary.get("reconciliation_ok"):
        add("independent_reconciliation", False, str(summary.get("reconciliation_max_abs_discrepancy")))
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s12_p2": {"status": "STOP", "failed": ["independent_reconciliation"]}}},
            validations=checks,
        )

    out = root / "artifacts/s12/p2"
    out.mkdir(parents=True, exist_ok=True)
    a_fields = [
        "group_id", "source_plant_id", "target_plant_id", "otg_abs", "otg_rel", "otg_rel_defined",
        "support_retention", "local_same_rows_mae", "transfer_mae", "retention_tercile",
        "computable", "model_family", "support_rule",
    ]
    write_csv(out / "S12_P2_ASSOCIATION_ROWS.csv", sci["assoc"], a_fields)
    write_csv(out / "S12_P2_BOOTSTRAP.csv", sci["boot_rows"], ["replicate_id", "rho_abs", "valid_abs", "reason_abs", "rho_rel", "valid_rel", "reason_rel"])
    t_fields = [
        "tercile", "retention_interval", "cut_q_one_third", "cut_q_two_thirds", "n_directional", "n_distinct_targets",
        "median_otg_abs", "p75_otg_abs", "median_otg_rel", "p75_otg_rel", "median_support_retention",
        "plant_balanced_otg_abs", "n_targets_plant_balanced", "group_composition",
    ]
    terc_rows = [{**r, "group_composition": str(r["group_composition"])} for r in sci["terc"]]
    write_csv(out / "S12_P2_RETENTION_TERCILES.csv", terc_rows, t_fields)
    write_csv(
        out / "S12_P2_R7_DIRECTIONAL.csv",
        sci["r7_dir"],
        ["group_id", "source_plant_id", "target_plant_id", "otg_abs", "otg_rel", "support_retention", "local_same_rows_mae"],
    )
    write_csv(
        out / "S12_P2_R7_PAIRS.csv",
        sci["r7_pairs"],
        ["group_id", "plant_a", "plant_b", "support_retention_a_to_b", "support_retention_b_to_a", "asymmetry_abs", "asymmetry_signed"],
    )
    dump_json(out / "S12_P2_SUMMARY.json", summary)
    md = root / "reports/scientific/S12_P2_SUPPORT_RETENTION.md"
    md.write_text(render_report(summary), encoding="utf-8")

    recs = [
        artifact(root, out / "S12_P2_ASSOCIATION_ROWS.csv", rows=len(sci["assoc"]), fp=csv_fingerprint(out / "S12_P2_ASSOCIATION_ROWS.csv")),
        artifact(root, out / "S12_P2_BOOTSTRAP.csv", rows=len(sci["boot_rows"]), fp=csv_fingerprint(out / "S12_P2_BOOTSTRAP.csv")),
        artifact(root, out / "S12_P2_RETENTION_TERCILES.csv", rows=len(terc_rows), fp=csv_fingerprint(out / "S12_P2_RETENTION_TERCILES.csv")),
        artifact(root, out / "S12_P2_R7_DIRECTIONAL.csv", rows=len(sci["r7_dir"]), fp=csv_fingerprint(out / "S12_P2_R7_DIRECTIONAL.csv")),
        artifact(root, out / "S12_P2_R7_PAIRS.csv", rows=len(sci["r7_pairs"]), fp=csv_fingerprint(out / "S12_P2_R7_PAIRS.csv")),
        artifact(root, out / "S12_P2_SUMMARY.json"),
        artifact(root, md),
    ]
    prov_rel = "artifacts/s12/p2/S12_P2_PROVENANCE.json"
    provenance = {
        "repository_head": _git_head(root),
        "canonical_inputs": {
            "artifacts/s07/S07_OTG_DIRECTIONAL.csv": s07_h,
            "artifacts/s07/S07_ASYMMETRY.csv": sha256_file(root / "artifacts/s07/S07_ASYMMETRY.csv"),
            "artifacts/s05/S05_SUPPORT_MASKS.parquet": s05_h,
            "artifacts/rq3/source_contrasts.csv": sha256_file(root / "artifacts/rq3/source_contrasts.csv"),
        },
        "otg_abs_definition": summary["otg_abs_definition"],
        "otg_rel_definition": summary["otg_rel_definition"],
        "bootstrap": {"cluster": "target_plant_id", "n_replicates": N_BOOT, "seed": SEED, "ci": "percentile 2.5/97.5 linear quantile"},
        "tercile_quantile_rule": "numpy.quantile 1/3 and 2/3 method=linear; LOW<q1, MID<q2 else HIGH",
        "r7_threshold": R7_THRESHOLD,
        "rq2_r7_rule": summary["rq2_r7_rule"],
        "rq3_r7": {"status": RQ3_NC, "reason": summary["r7"]["rq3_reason"]},
        "script_path": "src/run/21_s12_p2_retention.py",
        "helper_path": "src/lib/s12_p2_retention.py",
        "generated_at_utc": now,
        "no_refit": True,
        "no_new_prediction_scoring": True,
        "accepted_science_not_edited": True,
        "manuscript_files_not_edited": True,
        "self_hash_registered_externally": True,
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes, "row_count": rec.row_count} for rec in recs},
    }
    dump_json(out / "S12_P2_PROVENANCE.json", provenance)
    recs.append(artifact(root, out / "S12_P2_PROVENANCE.json"))

    from collections import Counter as C

    terc_c = C(r["retention_tercile"] for r in sci["assoc"])
    add("association_n_85", len(sci["assoc"]) == 85, str(len(sci["assoc"])))
    add("relative_otg_accepted_definition", all(r["otg_rel_defined"] for r in sci["assoc"]), "otg_abs/local_mae")
    add("bootstrap_5000_seed", len(sci["boot_rows"]) == N_BOOT and SEED == 20260921, str(len(sci["boot_rows"])))
    add("r7_threshold_exactly_half", R7_THRESHOLD == 0.5 and all(r["support_retention"] >= 0.5 for r in sci["r7_dir"]), str(R7_THRESHOLD))
    add("one_tercile_each", all(r.get("retention_tercile") in {"LOW", "MID", "HIGH"} for r in sci["assoc"]) and sum(terc_c.values()) == 85, str(dict(terc_c)))
    add("rq3_not_proxied", summary["r7"]["rq3_status"] == RQ3_NC, summary["r7"]["rq3_status"])
    add("independent_reconciliation", summary["reconciliation_ok"], str(summary["reconciliation_max_abs_discrepancy"]))
    add("no_p3_files", list((root / "src/run").glob("22_s12*")) == [], "stop after P2")
    add("s07_unchanged", s07_h == sha256_file(s07), s07_h)
    add("s05_unchanged", s05_h == sha256_file(s05), s05_h)
    add("provenance_omits_self", prov_rel not in provenance["outputs"], "ok")
    tex_after, pdf_after = _sha(tex_path), _sha(pdf_path)
    add("paper_tex_unchanged", tex_before == tex_after, tex_after)
    add("paper_pdf_unchanged", pdf_before == pdf_after, pdf_after)

    payload = {
        "status": STATUS,
        "n_association": 85,
        "spearman_abs": summary["spearman_abs"],
        "spearman_rel": summary["spearman_rel"],
        "tercile_cutpoints": summary["tercile_cutpoints"],
        "r7": summary["r7"],
        "association_interpretation": summary["association_interpretation"],
        "association_path": "artifacts/s12/p2/S12_P2_ASSOCIATION_ROWS.csv",
        "association_sha256": recs[0].sha256,
        "bootstrap_path": "artifacts/s12/p2/S12_P2_BOOTSTRAP.csv",
        "bootstrap_sha256": recs[1].sha256,
        "terciles_path": "artifacts/s12/p2/S12_P2_RETENTION_TERCILES.csv",
        "terciles_sha256": recs[2].sha256,
        "r7_directional_path": "artifacts/s12/p2/S12_P2_R7_DIRECTIONAL.csv",
        "r7_directional_sha256": recs[3].sha256,
        "r7_pairs_path": "artifacts/s12/p2/S12_P2_R7_PAIRS.csv",
        "r7_pairs_sha256": recs[4].sha256,
        "summary_path": "artifacts/s12/p2/S12_P2_SUMMARY.json",
        "summary_sha256": recs[5].sha256,
        "report_path": "reports/scientific/S12_P2_SUPPORT_RETENTION.md",
        "report_sha256": recs[6].sha256,
        "provenance_path": prov_rel,
        "provenance_sha256": recs[-1].sha256,
        "no_refit": True,
        "failed": None,
    }
    add("sot_provenance_sha", payload["provenance_sha256"] == recs[-1].sha256, recs[-1].sha256)
    pre = _snap(ctx.sot)
    patch = ledger_patch(stage="s12_p2", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p2": payload}})
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_rq_unchanged", _nested(merged, ("results", "tables", "rq1")) == pre.get("results.tables.rq1") and _nested(merged, ("results", "tables", "rq2")) == pre.get("results.tables.rq2") and _nested(merged, ("results", "tables", "rq3")) == pre.get("results.tables.rq3"), str(pre.get("results.tables.rq1")))
    add("s11_unchanged", _nested(merged, ("results", "s11_phase4", "decisions")) == pre.get("results.s11_phase4.decisions"), "s11")
    add("s12_p0_p1_unchanged", _nested(merged, ("results", "s12_phase0", "status")) == pre.get("results.s12_phase0.status") and _nested(merged, ("results", "s12_p1", "status")) == pre.get("results.s12_p1.status"), "s12 prior")
    add("rq4_unchanged", _nested(merged, ("scientific_analysis", "rq4_source_selection", "status")) == pre.get("scientific_analysis.rq4_source_selection.status"), "rq4")
    payload["n_validations"] = len(checks)
    patch = ledger_patch(stage="s12_p2", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p2": payload}})
    if not ns_ok:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p2": {"status": "STOP", "reason": ns_detail, "failed": None}}}, artifacts=recs, validations=checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p2": {"status": "STOP", "failed": [c.name for c in failed]}}}, artifacts=recs, validations=checks)
    return StageResult(status="GO", message=payload["status"], sot_patch=patch, artifacts=recs, validations=checks)
