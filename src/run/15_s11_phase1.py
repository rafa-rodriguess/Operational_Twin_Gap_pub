"""15 — S11 Phase 1: A3 within-target source spread and A4 group-wise RQ1."""

from __future__ import annotations

import csv
import hashlib
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.protocol import FAM_SPLINE, PRIMARY_GROUP_IDS
from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, sha256_file, write_csv
from src.sot import apply_patch
from src.lib.s11_phase1 import (
    TOL,
    _rows_close,
    a3_path_groupby,
    a3_path_iterate,
    a3_summary,
    a4_close,
    a4_path_groupby,
    a4_path_iterate,
    filter_s07,
    missing_s07_fields,
    plant_balanced_rq1,
)
from src.run._ledger import ledger_patch

A3_FIELDS = [
    "target_id",
    "twin_group",
    "group_id",
    "n_computable_sources",
    "min_otg",
    "min_otg_source_id",
    "max_otg",
    "max_otg_source_id",
    "spread_abs",
    "min_otg_rel",
    "max_otg_rel",
    "spread_rel",
]
A4_GROUP_FIELDS = [
    "twin_group",
    "group_id",
    "n_targets",
    "n_directional_transfers",
    "plant_balanced_rq1",
    "min_target_mean_otg",
    "max_target_mean_otg",
]
A4_TARGET_FIELDS = ["target_id", "twin_group", "group_id", "n_directional_transfers", "rq1_j"]
A3_CLOSE_KEYS = [
    "target_id",
    "twin_group",
    "group_id",
    "n_computable_sources",
    "min_otg",
    "min_otg_source_id",
    "max_otg",
    "max_otg_source_id",
    "spread_abs",
]


PRIMARY_SNAPSHOT_PATHS = (
    ("results", "tables", "rq1"),
    ("results", "tables", "rq2"),
    ("results", "tables", "rq3"),
    ("results", "rq3", "D_fleet_RQ3"),
    ("scientific_analysis", "rq4_source_selection", "status"),
)
PRIMARY_RESULT_KEYS = frozenset(
    {"tables", "rq1_rq2", "rq3", "rq4_spec_freeze", "rq4_preflight", "rq4_execute", "rq4_finalize"}
)


def _nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _primary_snapshot(sot: dict[str, Any]) -> dict[str, Any]:
    return {".".join(path): _nested(sot, path) for path in PRIMARY_SNAPSHOT_PATHS}


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _namespace_ok(pre: dict[str, Any], patch: dict[str, Any], merged: dict[str, Any]) -> tuple[bool, str]:
    result_keys = set((patch.get("results") or {}).keys())
    if result_keys != {"s11_phase1"}:
        return False, f"results keys={sorted(result_keys)}"
    if result_keys & PRIMARY_RESULT_KEYS:
        return False, "patch touches primary results keys"
    sci = patch.get("scientific_analysis") or {}
    if set(sci.keys()) != {"s11_phase1"}:
        return False, f"scientific_analysis keys={sorted(sci)}"
    post = _primary_snapshot(merged)
    if post != pre:
        return False, f"primary snapshot changed {pre} -> {post}"
    if _nested(merged, ("results", "s11_phase1")) is None:
        return False, "missing results.s11_phase1"
    if _nested(merged, ("scientific_analysis", "s11_phase1")) is None:
        return False, "missing scientific_analysis.s11_phase1"
    return True, "s11-only; primary RQ1/RQ2/RQ3/RQ4 snapshot unchanged"


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(a: list[dict], b: list[dict]) -> bool:
    if not _rows_close(a, b, A3_CLOSE_KEYS):
        return False
    for x, y in zip(a, b):
        hx, hy = "spread_rel" in x, "spread_rel" in y
        if hx != hy:
            return False
        if hx and not math.isclose(float(x["spread_rel"]), float(y["spread_rel"]), rel_tol=0.0, abs_tol=TOL):
            return False
    return True


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex_path = root / "paper/main.tex"
    pdf_path = root / "paper/main.pdf"
    tex_before = _sha(tex_path)
    pdf_before = _sha(pdf_path)

    s07_path = root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"
    path_ok = s07_path.is_file()
    rows = None
    if path_ok:
        with s07_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    missing = missing_s07_fields(rows, path_ok)
    add("s07_present", path_ok, str(s07_path.relative_to(root)) if path_ok else "missing")
    add("s07_required_fields", missing is None, missing)
    if missing:
        return StageResult(
            status="STOP",
            message="STOP_MISSING_S07_INPUT",
            sot_patch={"results": {"s11_phase1": {"status": "STOP", "reason": missing}}},
            validations=checks,
        )

    filtered = filter_s07(rows or [])
    add("filter_primary_groups_only", all(r["group_id"] in PRIMARY_GROUP_IDS for r in filtered), str(len(filtered)))
    add("filter_spline_ridge", all(r.get("model_family") == FAM_SPLINE for r in filtered), FAM_SPLINE)
    add("filter_cs4", all(r.get("support_rule") == "CS4" for r in filtered), "CS4")
    add("filter_computable", all(str(r.get("computable")).strip().lower() in {"true", "1", "yes"} for r in filtered), "computable")

    a3_g = a3_path_groupby(filtered)
    a3_i = a3_path_iterate(filtered)
    a3_ok = _close(a3_g, a3_i)
    add("a3_two_path_reconciliation", a3_ok, f"n={len(a3_g)}")
    add("a3_eligible_sources_ge2", all(int(r["n_computable_sources"]) >= 2 for r in a3_g), str(len(a3_g)))
    spread_id = all(
        math.isclose(float(r["spread_abs"]), float(r["max_otg"]) - float(r["min_otg"]), rel_tol=0.0, abs_tol=TOL)
        for r in a3_g
    )
    add("a3_spread_abs_identity", spread_id, "max-min")
    add("a3_spread_non_negative", all(float(r["spread_abs"]) >= -TOL for r in a3_g), "spread>=0")
    summary = a3_summary(a3_g)
    add("a3_summary_count_matches_rows", summary["eligible_target_count"] == len(a3_g), str(summary["eligible_target_count"]))
    if a3_g:
        max_from_rows = max(float(r["spread_abs"]) for r in a3_g)
        max_ok = math.isclose(float(summary["max_spread"]), max_from_rows, rel_tol=0.0, abs_tol=TOL)
    else:
        max_ok = summary["max_spread"] is None
    add("a3_max_spread_matches_row", max_ok, str(summary.get("max_target")))
    n_el = int(summary["eligible_target_count"])
    gate_ok = summary["manuscript_reporting_gate"] == ("MAX_ONLY_PLUS_N" if n_el < 5 else "MEDIAN_AND_MAX")
    add("a3_reporting_gate", gate_ok, str(summary["manuscript_reporting_gate"]))

    a4_g, t_g = a4_path_groupby(filtered)
    a4_i, t_i = a4_path_iterate(filtered)
    a4_ok = a4_close(a4_g, a4_i) and [r["target_id"] for r in t_g] == [r["target_id"] for r in t_i]
    if a4_ok:
        for x, y in zip(t_g, t_i):
            if not math.isclose(float(x["rq1_j"]), float(y["rq1_j"]), rel_tol=0.0, abs_tol=TOL):
                a4_ok = False
                break
    add("a4_two_path_reconciliation", a4_ok, str([r["twin_group"] for r in a4_g]))
    add("a4_groups_complete", [r["twin_group"] for r in a4_g] == ["G1", "G2", "G4"], ",".join(r["twin_group"] for r in a4_g))
    unique_t = {str(r["target_plant_id"]) for r in filtered}
    add("a4_n_targets_sum", sum(int(r["n_targets"]) for r in a4_g) == len(unique_t), str(len(unique_t)))
    band_ok = True
    for r in a4_g:
        mean, lo, hi = r["plant_balanced_rq1"], r["min_target_mean_otg"], r["max_target_mean_otg"]
        if mean is None:
            continue
        if lo - TOL > mean or mean > hi + TOL:
            band_ok = False
    add("a4_min_le_mean_le_max", band_ok, "band")
    by_g_n = {r["twin_group"]: sum(1 for t in t_g if t["twin_group"] == r["twin_group"]) for r in a4_g}
    add("a4_target_means_match_groups", all(int(r["n_targets"]) == by_g_n[r["twin_group"]] for r in a4_g), str(by_g_n))

    computed, n_t = plant_balanced_rq1(filtered)
    sot_rq1 = ((ctx.sot.get("results") or {}).get("tables") or {}).get("rq1")
    rq1_ok = (
        computed is not None
        and sot_rq1 is not None
        and math.isclose(float(computed), float(sot_rq1), rel_tol=0.0, abs_tol=1e-12)
    )
    add("primary_rq1_match", rq1_ok, f"computed={computed} sot={sot_rq1}")
    add("a4_n_targets_equals_unique_filtered_targets", n_t == len(unique_t), str(n_t))

    if not a3_ok or not a4_ok:
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s11_phase1": {"status": "STOP", "reason": "two-path mismatch"}}},
            validations=checks,
        )
    if not rq1_ok:
        return StageResult(
            status="STOP",
            message="STOP_PRIMARY_RQ1_MISMATCH",
            sot_patch={"results": {"s11_phase1": {"status": "STOP", "computed": computed, "sot_rq1": sot_rq1}}},
            validations=checks,
        )

    out = root / "artifacts/s11"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "S11_A3_TARGET_SPREAD.csv", a3_g, A3_FIELDS)
    dump_json(out / "S11_A3_SUMMARY.json", summary)
    write_csv(out / "S11_A4_RQ1_BY_GROUP.csv", a4_g, A4_GROUP_FIELDS)
    write_csv(out / "S11_A4_TARGET_MEANS.csv", t_g, A4_TARGET_FIELDS)

    recs = [
        artifact(root, out / "S11_A3_TARGET_SPREAD.csv", rows=len(a3_g), fp=csv_fingerprint(out / "S11_A3_TARGET_SPREAD.csv")),
        artifact(root, out / "S11_A3_SUMMARY.json"),
        artifact(root, out / "S11_A4_RQ1_BY_GROUP.csv", rows=len(a4_g), fp=csv_fingerprint(out / "S11_A4_RQ1_BY_GROUP.csv")),
        artifact(root, out / "S11_A4_TARGET_MEANS.csv", rows=len(t_g), fp=csv_fingerprint(out / "S11_A4_TARGET_MEANS.csv")),
    ]
    if len(filtered) != 85 or len(unique_t) != 14:
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s11_phase1": {"status": "STOP", "filtered_row_count": len(filtered), "target_count": len(unique_t)}}},
            artifacts=recs,
            validations=checks,
        )
    prior = (ctx.sot.get("results") or {}).get("s11_phase1") or {}
    by_g = {str(r["twin_group"]): r for r in a4_g}
    science_same = True
    if prior.get("status") == "S11_PHASE1_A3_A4_MATERIALIZED":
        science_same = (
            int(prior.get("eligible_target_count") or -1) == int(summary["eligible_target_count"])
            and math.isclose(float(prior["median_spread"]), float(summary["median_spread"]), rel_tol=0.0, abs_tol=TOL)
            and math.isclose(float(prior["max_spread"]), float(summary["max_spread"]), rel_tol=0.0, abs_tol=TOL)
            and prior.get("max_target") == summary["max_target"]
            and math.isclose(float(prior["plant_balanced_rq1"]), float(computed), rel_tol=0.0, abs_tol=1e-12)
        )
        prior_groups = {str(r["twin_group"]): r for r in (prior.get("a4_by_group") or [])}
        for label in ("G1", "G2", "G4"):
            if label not in prior_groups or label not in by_g:
                science_same = False
                break
            if not math.isclose(
                float(prior_groups[label]["plant_balanced_rq1"]),
                float(by_g[label]["plant_balanced_rq1"]),
                rel_tol=0.0,
                abs_tol=TOL,
            ):
                science_same = False
                break
    if not science_same:
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s11_phase1": {"status": "STOP", "reason": "A3/A4 drifted from frozen Phase 1"}}},
            artifacts=recs,
            validations=checks,
        )

    head = _git_head(root)
    provenance = {
        "source_artifact_path": "artifacts/s07/S07_OTG_DIRECTIONAL.csv",
        "source_artifact_sha256": sha256_file(s07_path),
        "repository_head": head,
        "phase1_baseline_commit": "d86bae9ea36df1ab1a478d66eaabf36cb658ca6f",
        "filter": "SplineRidge + CS4 + computable + primary groups",
        "generated_at_utc": now,
        "script_path": "src/run/15_s11_phase1.py",
        "helper_path": "src/lib/s11_phase1.py",
        "filtered_row_count": len(filtered),
        "target_count": len(unique_t),
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes, "row_count": rec.row_count, "schema_fingerprint": rec.schema_fingerprint} for rec in recs},
    }
    dump_json(out / "S11_PHASE1_PROVENANCE.json", provenance)
    recs.append(artifact(root, out / "S11_PHASE1_PROVENANCE.json"))

    tex_after = _sha(tex_path)
    pdf_after = _sha(pdf_path)
    add("paper_tex_unchanged", tex_before == tex_after and tex_before is not None, tex_after)
    add("paper_pdf_unchanged", pdf_before == pdf_after, pdf_after)

    lib_blob = (root / "src/lib/s11_phase1.py").read_text(encoding="utf-8")
    req = (root / "requirements.txt").read_text(encoding="utf-8")
    no_pd = "pandas" not in req.lower() and "pandas" not in lib_blob
    add("no_pandas_dependency", no_pd, "stdlib+csv")

    payload = {
        "status": "S11_PHASE1_A3_A4_MATERIALIZED",
        "eligible_target_count": summary["eligible_target_count"],
        "manuscript_reporting_gate": summary["manuscript_reporting_gate"],
        "median_spread": summary["median_spread"],
        "max_spread": summary["max_spread"],
        "max_target": summary["max_target"],
        "n_eligible_by_twin_group": summary["n_eligible_by_twin_group"],
        "plant_balanced_rq1": computed,
        "sot_rq1": sot_rq1,
        "a4_by_group": a4_g,
        "filtered_row_count": len(filtered),
        "target_count": len(unique_t),
        "provenance_path": "artifacts/s11/S11_PHASE1_PROVENANCE.json",
        "provenance_sha256": recs[-1].sha256,
        "source_artifact_path": provenance["source_artifact_path"],
        "source_artifact_sha256": provenance["source_artifact_sha256"],
        "repository_head": head,
        "filter": provenance["filter"],
        "generated_at_utc": now,
        "script_path": provenance["script_path"],
    }
    pre_primary = _primary_snapshot(ctx.sot)
    patch = ledger_patch(
        stage="s11_phase1",
        now=now,
        payload=payload,
        recs=recs,
        extra={"scientific_analysis": {"s11_phase1": payload}},
    )
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre_primary, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    payload["n_validations"] = len(checks)
    patch = ledger_patch(
        stage="s11_phase1",
        now=now,
        payload=payload,
        recs=recs,
        extra={"scientific_analysis": {"s11_phase1": payload}},
    )

    if len(checks) != 24:
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s11_phase1": {"status": "STOP", "n_validations": len(checks)}}},
            artifacts=recs,
            validations=checks,
        )
    if not ns_ok:
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s11_phase1": {"status": "STOP", "reason": ns_detail}}},
            artifacts=recs,
            validations=checks,
        )
    return StageResult(
        status="GO",
        message="S11_PHASE1_A3_A4_MATERIALIZED",
        sot_patch=patch,
        artifacts=recs,
        validations=checks,
    )
