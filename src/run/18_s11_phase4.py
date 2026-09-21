"""18 — S11 Phase 4: consolidate evidence and select manuscript additions."""

from __future__ import annotations

import hashlib
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, dump_json, sha256_file
from src.lib.s11_phase4 import (
    PHASE13_SOURCES,
    build_decisions,
    collect_headlines,
    inspect_manuscript,
    render_markdown,
    source_inventory,
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
    ("results", "s11_phase3_a1_g3", "rq3_target_balanced_control_minus_twin"),
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


def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return a == b


def _namespace_ok(pre: dict[str, Any], patch: dict[str, Any], merged: dict[str, Any]) -> tuple[bool, str]:
    rkeys = set((patch.get("results") or {}).keys())
    if rkeys != {"s11_phase4"}:
        return False, f"results keys={sorted(rkeys)}"
    sci = set((patch.get("scientific_analysis") or {}).keys())
    if sci != {"s11_phase4"}:
        return False, f"scientific_analysis keys={sorted(sci)}"
    if _snap(merged) != pre:
        return False, "prior scientific snapshot changed"
    return True, "s11_phase4 only"


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex_path, pdf_path = root / "paper/main.tex", root / "paper/main.pdf"
    tex_before, pdf_before = _sha(tex_path), _sha(pdf_path)

    inv = source_inventory(root)
    missing = [p for p, rec in inv.items() if not rec["exists"]]
    add("phase13_artifacts_present", missing == [], ",".join(missing) or "all present")
    if missing:
        return StageResult(
            status="STOP",
            message="STOP_MISSING_PHASE13_ARTIFACTS",
            sot_patch={"results": {"s11_phase4": {"status": "STOP", "missing": missing}}},
            validations=checks,
        )

    head = collect_headlines(root, ctx.sot)
    ms = inspect_manuscript(root)
    res = ctx.sot.get("results") or {}
    p1, p2, p3 = res.get("s11_phase1") or {}, res.get("s11_phase2_a2") or {}, res.get("s11_phase3_a1_g3") or {}

    add(
        "headlines_match_sot",
        _close(head["g3"]["rq1"], p3.get("rq1_plant_balanced_mean_otg"))
        and _close(head["g3"]["rq3"], p3.get("rq3_target_balanced_control_minus_twin"))
        and _close(head["a2"]["target_balanced_r8"], p2.get("target_balanced_r8_control_minus_twin"))
        and _close(head["a3"]["median_spread"], p1.get("median_spread"))
        and _close(head["primary"]["rq1"], p1.get("sot_rq1") or p1.get("plant_balanced_rq1")),
        "SoT vs Phase 1–3 JSON",
    )
    add("g3_separate_from_primary", head["g3"]["group_id"] == "TW_7b33a493697b" and "poa_irradiance_wm2" in head["g3"]["features"] and len(head["g3"]["features"]) == 2, str(head["g3"]["features"]))
    add("a3_scope_g2_only", set((head["a3"]["n_eligible_by_twin_group"] or {}).keys()) == {"G2"}, str(head["a3"]["n_eligible_by_twin_group"]))
    add("a4_between_group", set(head["a4"]["by_group"]) == {"G1", "G2", "G4"}, str(head["a4"]["by_group"]))

    decisions = build_decisions(head, ms)
    add(
        "selection_not_by_sign",
        "estimand sign is not a selection input" in decisions["selection_rule"]
        and decisions["items"]["A2"]["r8_vs_primary_note"]["used_for_selection"] is False
        and decisions["items"]["A4"]["includes_negative_group_mean"] is True,
        decisions["selection_rule"],
    )

    out_s11 = root / "artifacts/s11"
    out_rep = root / "reports/scientific"
    out_s11.mkdir(parents=True, exist_ok=True)
    out_rep.mkdir(parents=True, exist_ok=True)

    consolidation = {
        "status": "S11_PHASE4_DECISION_MATERIALIZED",
        "authoritative_headlines": head,
        "manuscript_context": {k: v for k, v in ms.items() if k != "tex_sha256"},
        "decisions": decisions["items"],
        "recommended_inclusion_set": decisions["recommended_inclusion_set"],
        "recommended_if_space_set": decisions["recommended_if_space_set"],
        "recommended_supplement_or_omit_set": decisions["recommended_supplement_or_omit_set"],
        "selection_rule": decisions["selection_rule"],
        "space_plan": decisions["space_plan"],
        "phase6_payload": decisions["phase6_payload"],
        "synthesis": decisions["synthesis"],
        "no_estimand_recomputed": True,
        "manuscript_not_edited": True,
    }
    dump_json(out_s11 / "S11_PHASE4_CONSOLIDATION.json", consolidation)
    md = render_markdown(head, ms, decisions)
    (out_rep / "S11_PHASE4_INCLUSION_DECISION.md").write_text(md, encoding="utf-8")

    recs = [
        artifact(root, out_s11 / "S11_PHASE4_CONSOLIDATION.json"),
        artifact(root, out_rep / "S11_PHASE4_INCLUSION_DECISION.md"),
    ]

    source_hashes = {rel: rec["sha256"] for rel, rec in inv.items()}
    prov_rel = "artifacts/s11/S11_PHASE4_PROVENANCE.json"
    provenance = {
        "repository_head": _git_head(root),
        "phase13_source_artifacts": source_hashes,
        "manuscript": {"paper/main.tex": tex_before, "paper/main.pdf": pdf_before, "pdf_pages": ms.get("pdf_pages")},
        "script_path": "src/run/18_s11_phase4.py",
        "helper_path": "src/lib/s11_phase4.py",
        "generated_at_utc": now,
        "no_scientific_estimand_recomputed_or_changed": True,
        "manuscript_files_not_edited": True,
        "editorial_decision_only": True,
        "self_hash_registered_externally": True,
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes} for rec in recs},
    }
    dump_json(out_s11 / "S11_PHASE4_PROVENANCE.json", provenance)
    recs.append(artifact(root, out_s11 / "S11_PHASE4_PROVENANCE.json"))
    add(
        "provenance_outputs_omit_self_hash",
        prov_rel not in provenance["outputs"]
        or provenance["outputs"].get(prov_rel, {}).get("sha256") is None,
        str(sorted(provenance["outputs"])),
    )

    add("decision_artifacts_exist", all((root / p).is_file() for p in ("artifacts/s11/S11_PHASE4_CONSOLIDATION.json", "reports/scientific/S11_PHASE4_INCLUSION_DECISION.md", "artifacts/s11/S11_PHASE4_PROVENANCE.json")), "three outputs")
    add("no_estimand_modified", consolidation["no_estimand_recomputed"] and provenance["no_scientific_estimand_recomputed_or_changed"], "synthesis only")

    tex_after, pdf_after = _sha(tex_path), _sha(pdf_path)
    add("paper_tex_unchanged", tex_before == tex_after and tex_before is not None, tex_after)
    add("paper_pdf_unchanged", pdf_before == pdf_after, pdf_after)
    phase56 = list((root / "src/run").glob("19_s11*")) + list((root / "src/run").glob("20_s11*"))
    phase56 += list(out_s11.glob("S11_PHASE5*")) + list(out_s11.glob("S11_PHASE6*"))
    add("no_phase_56_files", phase56 == [], str(phase56))

    payload = {
        "status": "S11_PHASE4_DECISION_MATERIALIZED",
        "decisions": {k: v["decision"] for k, v in decisions["items"].items()},
        "recommended_inclusion_set": decisions["recommended_inclusion_set"],
        "recommended_if_space_set": decisions["recommended_if_space_set"],
        "recommended_supplement_or_omit_set": decisions["recommended_supplement_or_omit_set"],
        "consolidation_path": "artifacts/s11/S11_PHASE4_CONSOLIDATION.json",
        "consolidation_sha256": recs[0].sha256,
        "report_path": "reports/scientific/S11_PHASE4_INCLUSION_DECISION.md",
        "report_sha256": recs[1].sha256,
        "provenance_path": "artifacts/s11/S11_PHASE4_PROVENANCE.json",
        "provenance_sha256": recs[-1].sha256,
        "pdf_pages": ms.get("pdf_pages"),
        "no_estimand_recomputed": True,
        "sequencing": "insert_all_first_then_assess_page_count",
    }
    add("sot_provenance_sha_matches_file", payload["provenance_sha256"] == recs[-1].sha256 == _sha(out_s11 / "S11_PHASE4_PROVENANCE.json"), recs[-1].sha256)
    add(
        "all_four_include_core",
        payload["decisions"] == {"A1": "INCLUDE_CORE", "A2": "INCLUDE_CORE", "A3": "INCLUDE_CORE", "A4": "INCLUDE_CORE"}
        and payload["recommended_inclusion_set"] == ["A1", "A2", "A3", "A4"]
        and payload["recommended_if_space_set"] == []
        and payload["recommended_supplement_or_omit_set"] == [],
        str(payload["decisions"]),
    )
    pre = _snap(ctx.sot)
    matched_pre = _nested(ctx.sot, ("results", "rq3", "selected_controls"))
    p1_pre, p2_pre, p3_pre = dict(p1), dict(p2), dict(p3)
    patch = ledger_patch(stage="s11_phase4", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s11_phase4": payload}})
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_rq_unchanged", _nested(merged, ("results", "rq3", "D_fleet_RQ3")) == pre.get("results.rq3.D_fleet_RQ3") and _nested(merged, ("results", "tables", "rq1")) == pre.get("results.tables.rq1"), str(pre.get("results.rq3.D_fleet_RQ3")))
    add("matched_controls_unchanged", _nested(merged, ("results", "rq3", "selected_controls")) == matched_pre, str(matched_pre))
    add("phase1_unchanged", _nested(merged, ("results", "s11_phase1", "median_spread")) == p1_pre.get("median_spread"), str(p1_pre.get("median_spread")))
    add("phase2_unchanged", _nested(merged, ("results", "s11_phase2_a2", "target_balanced_r8_control_minus_twin")) == p2_pre.get("target_balanced_r8_control_minus_twin"), str(p2_pre.get("target_balanced_r8_control_minus_twin")))
    add("phase3_unchanged", _nested(merged, ("results", "s11_phase3_a1_g3", "rq1_plant_balanced_mean_otg")) == p3_pre.get("rq1_plant_balanced_mean_otg"), str(p3_pre.get("rq1_plant_balanced_mean_otg")))
    add("rq4_unchanged", _nested(merged, ("scientific_analysis", "rq4_source_selection", "status")) == pre.get("scientific_analysis.rq4_source_selection.status"), str(pre.get("scientific_analysis.rq4_source_selection.status")))
    payload["n_validations"] = len(checks)
    payload["failed"] = None
    patch = ledger_patch(stage="s11_phase4", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s11_phase4": payload}})

    if not ns_ok:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s11_phase4": {"status": "STOP", "reason": ns_detail}}}, artifacts=recs, validations=checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s11_phase4": {"status": "STOP", "failed": [c.name for c in failed]}}}, artifacts=recs, validations=checks)
    return StageResult(status="GO", message=payload["status"], sot_patch=patch, artifacts=recs, validations=checks)
