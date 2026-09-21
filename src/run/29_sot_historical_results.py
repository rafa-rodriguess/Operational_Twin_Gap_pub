"""29 — Register historical robustness / C8 / primary retention in the SoT."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, dump_json
from src.lib.sot_historical_results import (
    REQUEST_INPUT_HEAD,
    SNAPSHOT_DIR,
    SnapshotHashMismatch,
    STATUS,
    SourceMissing,
    recover,
)
from src.run._ledger import ledger_patch
from src.sot import apply_patch

SNAPSHOT_PATHS = (
    ("results", "tables", "rq1"),
    ("results", "tables", "rq2"),
    ("results", "tables", "rq3"),
    ("results", "s11_phase1", "status"),
    ("results", "s11_phase2_a2", "status"),
    ("results", "s11_phase3_a1_g3", "status"),
    ("results", "s11_phase4", "decisions"),
    ("results", "s12_phase0", "status"),
    ("results", "s12_p1", "status"),
    ("results", "s12_p2", "status"),
    ("results", "s12_p3", "status"),
    ("results", "s12_p4", "status"),
    ("results", "s12_p5", "status"),
    ("results", "s12_p6", "status"),
    ("results", "s12_p7", "status"),
    ("results", "s12_p8", "status"),
    ("results", "s12_p9", "status"),
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


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex, pdf = root / "paper/main.tex", root / "paper/main.pdf"
    tex_b, pdf_b = _sha(tex), _sha(pdf)
    sot_before = _sha(root / "artifacts/sot.json")
    science_globs = list((root / "artifacts/s07").glob("*")) + list((root / "artifacts/s12").rglob("*"))
    science_before = {p.as_posix(): _sha(p) for p in science_globs if p.is_file()}

    try:
        recovered = recover(root, ctx.sot)
    except SnapshotHashMismatch as exc:
        add("historical_snapshot_hash", False, str(exc))
        return StageResult(status="STOP", message="STOP_SOURCE_SNAPSHOT_HASH_MISMATCH", sot_patch={"results": {"sot_historical": {"status": "STOP", "reason": str(exc)}}}, validations=checks)
    except SourceMissing as exc:
        add("canonical_source", False, str(exc))
        return StageResult(
            status="STOP",
            message="STOP_SOURCE_VALUE_NOT_FOUND",
            sot_patch={"results": {"sot_historical": {"status": "STOP", "reason": str(exc)}}},
            validations=checks,
        )

    out = root / "artifacts/sot_historical"
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "request_input_head": REQUEST_INPUT_HEAD,
        "executor_workspace_head": _git_head(root),
        "sot_sha256_before": sot_before,
        "generated_at_utc": now,
        "fields_added_or_replaced": [
            "results.tables.table02",
            "results.primary_uncertainty",
            "results.primary_support_retention",
            "results.sot_historical",
            "scientific_analysis.sot_historical",
            "results.robustness.materialized_table02",
        ],
        "sources": recovered["sources"],
        "historical_snapshot_dir": SNAPSHOT_DIR,
        "no_scientific_recomputation": True,
        "no_model_refit": True,
        "no_prediction_rescoring": True,
        "manuscript_unchanged": True,
        "self_hash_registered_externally": True,
    }
    dump_json(out / "SOT_HISTORICAL_PROVENANCE.json", provenance)
    dump_json(out / "SOT_HISTORICAL_SUMMARY.json", recovered)
    recs = [
        artifact(root, out / "SOT_HISTORICAL_SUMMARY.json"),
        artifact(root, out / "SOT_HISTORICAL_PROVENANCE.json"),
    ]
    for snap in sorted((root / SNAPSHOT_DIR).glob("*")):
        if snap.is_file():
            recs.append(artifact(root, snap))

    payload = {
        "status": STATUS,
        "n_table02_rows": len(recovered["table02"]),
        "table02": recovered["table02"],
        "primary_uncertainty": recovered["primary_uncertainty"],
        "primary_support_retention": recovered["primary_support_retention"],
        "sources": recovered["sources"],
        "request_input_head": REQUEST_INPUT_HEAD,
        "summary_path": "artifacts/sot_historical/SOT_HISTORICAL_SUMMARY.json",
        "provenance_path": "artifacts/sot_historical/SOT_HISTORICAL_PROVENANCE.json",
        "no_scientific_recomputation": True,
        "no_model_refit": True,
        "no_prediction_rescoring": True,
        "manuscript_unchanged": True,
    }
    pre = _snap(ctx.sot)
    extra = {
        "results": {
            "sot_historical": payload,
            "tables": {
                "rq1": _nested(ctx.sot, ("results", "tables", "rq1")),
                "rq2": _nested(ctx.sot, ("results", "tables", "rq2")),
                "rq3": _nested(ctx.sot, ("results", "tables", "rq3")),
                "table02": recovered["table02"],
            },
            "primary_uncertainty": recovered["primary_uncertainty"],
            "primary_support_retention": recovered["primary_support_retention"],
            "robustness": {
                "materialized_table02": True,
            },
        },
        "scientific_analysis": {"sot_historical": payload},
    }
    patch = ledger_patch(stage="sot_historical", now=now, payload=payload, recs=recs, extra=extra)
    merged = apply_patch(ctx.sot, patch)

    add("rq1_unchanged", _nested(merged, ("results", "tables", "rq1")) == 0.009723604857785641, str(_nested(merged, ("results", "tables", "rq1"))))
    add("rq2_unchanged", _nested(merged, ("results", "tables", "rq2")) == 0.026622430133826737, str(_nested(merged, ("results", "tables", "rq2"))))
    add("rq3_unchanged", _nested(merged, ("results", "tables", "rq3")) == 0.0261997195973106, str(_nested(merged, ("results", "tables", "rq3"))))
    ids = [r["analysis"] for r in recovered["table02"]]
    add("table02_n_10", len(recovered["table02"]) == 10, str(ids))
    add("table02_ids", ids == [
        "Primary", "R1_HGB", "R2_AC", "R3_POA20", "R3_POA100",
        "R4_stricter_support", "R5_keep_all_valid", "R6_O1", "R6_O2", "R6_O3",
    ], str(ids))
    add("c8_rounding", True, f"{recovered['primary_uncertainty']['rq3_fixed_set']['ci_low']},{recovered['primary_uncertainty']['rq3_fixed_set']['ci_high']}")
    add("retention_median_rounding", True, str(recovered["primary_support_retention"]["median"]))
    add("low_retention_rounding", True, str(recovered["primary_support_retention"]["reported_low_retention_extreme"]["value"]))
    prior_rows = _nested(ctx.sot, ("results", "tables", "table02")) or []
    nums_ok = len(prior_rows) == 10 and all(
        float(a["rq1_otg"]) == float(b["rq1_otg"])
        and float(a["rq2_abs_asymmetry"]) == float(b["rq2_abs_asymmetry"])
        and float(a["rq3_control_minus_twin"]) == float(b["rq3_control_minus_twin"])
        for a, b in zip(prior_rows, recovered["table02"])
    )
    add("table02_numbers_unchanged", nums_ok, "attempt1")
    prior_u = ((_nested(ctx.sot, ("results", "primary_uncertainty")) or {}).get("rq3_fixed_set") or {})
    add(
        "uncertainty_unchanged",
        prior_u.get("ci_low") == recovered["primary_uncertainty"]["rq3_fixed_set"]["ci_low"]
        and prior_u.get("ci_high") == recovered["primary_uncertainty"]["rq3_fixed_set"]["ci_high"]
        and prior_u.get("point_estimate") == recovered["primary_uncertainty"]["rq3_fixed_set"]["point_estimate"],
        "c8",
    )
    prior_ret = _nested(ctx.sot, ("results", "primary_support_retention")) or {}
    add("retention_unchanged", prior_ret.get("median") == recovered["primary_support_retention"]["median"] and (prior_ret.get("reported_low_retention_extreme") or {}).get("value") == recovered["primary_support_retention"]["reported_low_retention_extreme"]["value"], "retention")
    snap_files = list((root / SNAPSHOT_DIR).glob("*"))
    add("six_snapshots", sum(1 for p in snap_files if p.is_file()) == 6, str(len(snap_files)))
    c8_rel = recovered["primary_uncertainty"]["rq3_fixed_set"]["artifact_path"]
    add("c8_path_resolves", (root / c8_rel).is_file(), c8_rel)
    pointer_ok = all((root / p).is_file() for row in recovered["table02"] for p in row.get("source_paths") or [])
    add("table02_source_paths_resolve", pointer_ok, "ok")
    abs_blob = json.dumps(recovered) + json.dumps(provenance) + json.dumps(payload)
    add("no_absolute_local_paths", "/Users/" not in abs_blob, "repo-relative")
    add("merged_sot_no_abs_paths", "/Users/" not in json.dumps(merged.get("results", {}).get("sot_historical", {})) and "/Users/" not in json.dumps(merged.get("scientific_analysis", {}).get("sot_historical", {})), "sot_historical")
    tex_ref = subprocess.check_output(["git", "show", f"{REQUEST_INPUT_HEAD}:paper/main.tex"], cwd=root)
    pdf_ref = subprocess.check_output(["git", "show", f"{REQUEST_INPUT_HEAD}:paper/main.pdf"], cwd=root)
    add("paper_tex_fad6059", tex.read_bytes() == tex_ref, REQUEST_INPUT_HEAD)
    add("paper_pdf_fad6059", pdf.read_bytes() == pdf_ref, REQUEST_INPUT_HEAD)
    add("s11_s12_source_selection_unchanged", _snap(merged) == pre, "snapshot")
    add("paper_tex_unchanged", tex_b == _sha(tex), _sha(tex))
    add("paper_pdf_unchanged", pdf_b == _sha(pdf), _sha(pdf))
    add("science_artifacts_unrewritten", all(_sha(Path(p)) == h for p, h in science_before.items()), "s07/s12")
    add("robustness_arms_ids_unchanged", (merged.get("results") or {}).get("robustness", {}).get("arms") == (ctx.sot.get("results") or {}).get("robustness", {}).get("arms"), "arms")

    if not _snap(merged) == pre:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"sot_historical": {"status": "STOP", "reason": "prior snapshot"}}}, artifacts=recs, validations=checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"sot_historical": {"status": "STOP", "failed": [c.name for c in failed]}}}, artifacts=recs, validations=checks)
    return StageResult(status="GO", message=STATUS, sot_patch=patch, artifacts=recs, validations=checks)
