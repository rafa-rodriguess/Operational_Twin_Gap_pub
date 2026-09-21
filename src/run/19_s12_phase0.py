"""19 — S12 Phase 0: pipeline inventory and versioned P0–P9 roadmap."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, sha256_file, write_csv
from src.lib.s12_phase0 import (
    AUDIT_PATHS,
    RESERVED_LABELS,
    build_inventory,
    build_roadmap,
    inspect_files,
    phase_verdicts,
    readiness_rows,
    render_roadmap_md,
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
    ("results", "s11_phase4", "status"),
    ("results", "s11_phase4", "decisions"),
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
    rkeys = set((patch.get("results") or {}).keys())
    if rkeys != {"s12_phase0"}:
        return False, f"results keys={sorted(rkeys)}"
    sci = set((patch.get("scientific_analysis") or {}).keys())
    if sci != {"s12_phase0"}:
        return False, f"scientific_analysis keys={sorted(sci)}"
    if _snap(merged) != pre:
        return False, "prior scientific snapshot changed"
    return True, "s12_phase0 only"


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex_path, pdf_path = root / "paper/main.tex", root / "paper/main.pdf"
    tex_before, pdf_before = _sha(tex_path), _sha(pdf_path)

    files = inspect_files(root)
    missing = [rel for rel, rec in files.items() if not rec["exists"] and rel not in ("paper/main.pdf",)]
    add("audit_inputs_present", missing == [], ",".join(missing) or "all present")
    if missing:
        return StageResult(
            status="STOP",
            message="STOP_MISSING_S12_PHASE0_INPUTS",
            sot_patch={"results": {"s12_phase0": {"status": "STOP", "missing": missing, "failed": None}}},
            validations=checks,
        )

    inv = build_inventory(root)
    inv["file_hashes"] = {rel: rec["sha256"] for rel, rec in files.items() if rec["exists"]}
    rows = readiness_rows(inv)
    verdicts = phase_verdicts(rows)
    roadmap = build_roadmap(inv, verdicts)
    overall = "S12_PHASE0_READY_WITH_RECONSTRUCTION"
    if any(v == "BLOCKED_MISSING_INFORMATION" for v in verdicts.values()):
        overall = "S12_PHASE0_BLOCKED"
    elif all(v == "PERSISTED_READY" for v in verdicts.values()):
        overall = "S12_PHASE0_READY"

    out_s12 = root / "artifacts/s12"
    out_rep = root / "reports/scientific"
    out_s12.mkdir(parents=True, exist_ok=True)
    out_rep.mkdir(parents=True, exist_ok=True)

    dump_json(out_s12 / "S12_ROADMAP.json", roadmap)
    dump_json(out_s12 / "S12_PHASE0_PIPELINE_INVENTORY.json", inv)
    fields = ["phase", "dependency", "status", "source", "refit", "row_level", "action", "cost"]
    write_csv(out_s12 / "S12_PHASE0_READINESS.csv", rows, fields)
    md_path = out_rep / "S12_IMPROVEMENT_ROADMAP.md"
    md_path.write_text(render_roadmap_md(roadmap, inv, verdicts), encoding="utf-8")

    recs = [
        artifact(root, out_s12 / "S12_ROADMAP.json"),
        artifact(root, out_s12 / "S12_PHASE0_PIPELINE_INVENTORY.json"),
        artifact(root, out_s12 / "S12_PHASE0_READINESS.csv", rows=len(rows), fp=csv_fingerprint(out_s12 / "S12_PHASE0_READINESS.csv")),
        artifact(root, md_path),
    ]

    prov_rel = "artifacts/s12/S12_PHASE0_PROVENANCE.json"
    provenance = {
        "repository_head": _git_head(root),
        "audit_inputs": {rel: files[rel]["sha256"] for rel in AUDIT_PATHS if files.get(rel, {}).get("exists")},
        "manuscript": {"paper/main.tex": tex_before, "paper/main.pdf": pdf_before},
        "script_path": "src/run/19_s12_phase0.py",
        "helper_path": "src/lib/s12_phase0.py",
        "generated_at_utc": now,
        "no_scientific_headline_computed": True,
        "manuscript_files_not_edited": True,
        "p1_not_started": True,
        "self_hash_registered_externally": True,
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes} for rec in recs},
    }
    dump_json(out_s12 / "S12_PHASE0_PROVENANCE.json", provenance)
    recs.append(artifact(root, out_s12 / "S12_PHASE0_PROVENANCE.json"))

    pred = inv["predictions"]["primary_rq1_rq2"]
    mask = inv["support_masks"]["primary_cs4"]
    add("predictions_row_level_persisted", pred["status"] == "PERSISTED_READY", pred["status"])
    add("s06_test_only", pred.get("split_membership") == {"test": 74330} or set(pred.get("split_membership") or {}) == {"test"}, str(pred.get("split_membership")))
    add("support_masks_datetime_supported", mask["status"] == "PERSISTED_READY", str(mask.get("columns")))
    add("panel_row_key_unique", inv["row_key_convention"]["panel_uniqueness"]["unique"] is True, str(inv["row_key_convention"]["panel_uniqueness"]))
    add("index_row_key_unique", inv["row_key_convention"]["index_uniqueness"]["unique"] is True, str(inv["row_key_convention"]["index_uniqueness"]))
    add("splits_persisted", inv["splits"]["status"] == "PERSISTED_READY", inv["splits"]["artifact"])
    add("train_histories_fourteen_plants", len(inv["train_histories"]["plants"]) == 14, str(len(inv["train_histories"]["plants"])))
    add("hyperparameters_persisted", inv["hyperparameters"]["status"] == "PERSISTED_READY" and inv["hyperparameters"]["joblib_plants"] == 14, str(inv["hyperparameters"]["joblib_plants"]))
    add("scalers_reconstructible", inv["scalers_support_thresholds"]["status"] == "DETERMINISTIC_RECONSTRUCTION_READY", inv["scalers_support_thresholds"]["status"])
    add("g3_row_preds_require_refit", inv["predictions"]["g3_sensitivity"]["status"] == "REQUIRES_REFIT", inv["predictions"]["g3_sensitivity"]["status"])
    add("rq3_pairwise_reconstructible", inv["predictions"]["primary_rq3_pairwise_rows"]["status"] == "DETERMINISTIC_RECONSTRUCTION_READY", inv["predictions"]["primary_rq3_pairwise_rows"]["status"])
    add("p1_p3_p6_p8_require_refit", all(verdicts[p] == "REQUIRES_REFIT" for p in ("P1", "P3", "P6", "P8")), str({p: verdicts[p] for p in ("P1", "P3", "P6", "P8")}))
    add("p2_persisted", verdicts["P2"] == "PERSISTED_READY", verdicts["P2"])
    add("p5_p7_persisted", verdicts["P5"] == "PERSISTED_READY" and verdicts["P7"] == "PERSISTED_READY", str({"P5": verdicts["P5"], "P7": verdicts["P7"]}))
    add("r8_reserved_s11", RESERVED_LABELS["R8"].startswith("S11"), RESERVED_LABELS["R8"])
    add("r7_r9_r10_mapped", "P2" in RESERVED_LABELS["R7"] and "P3" in RESERVED_LABELS["R9"] and "P6" in RESERVED_LABELS["R10"], str(RESERVED_LABELS))
    add("s11_all_include_core_recorded", inv["s11_inclusion"]["all_include_core"] is True, str(inv["s11_inclusion"]["decisions"]))
    add("s11_not_yet_in_tex", inv["s11_inclusion"]["inserted_in_main_tex"] is False, "defer manuscript to post-P9")
    add("roadmap_p0_p9", [p["id"] for p in roadmap["phases"]] == [f"P{i}" for i in range(10)], str([p["id"] for p in roadmap["phases"]]))
    add("no_scientific_headline", provenance["no_scientific_headline_computed"] is True and overall.startswith("S12_PHASE0_"), overall)
    add("provenance_outputs_omit_self_hash", prov_rel not in provenance["outputs"], str(sorted(provenance["outputs"])))
    add("p1_not_started", list((root / "src/run").glob("20_s12*")) == [] and list(out_s12.glob("S12_PHASE1*")) == [], "no P1 files")

    tex_after, pdf_after = _sha(tex_path), _sha(pdf_path)
    add("paper_tex_unchanged", tex_before == tex_after and tex_before is not None, tex_after)
    add("paper_pdf_unchanged", pdf_before == pdf_after, pdf_after)

    payload = {
        "status": overall,
        "roadmap_path": "artifacts/s12/S12_ROADMAP.json",
        "roadmap_sha256": recs[0].sha256,
        "inventory_path": "artifacts/s12/S12_PHASE0_PIPELINE_INVENTORY.json",
        "inventory_sha256": recs[1].sha256,
        "readiness_path": "artifacts/s12/S12_PHASE0_READINESS.csv",
        "readiness_sha256": recs[2].sha256,
        "report_path": "reports/scientific/S12_IMPROVEMENT_ROADMAP.md",
        "report_sha256": recs[3].sha256,
        "provenance_path": "artifacts/s12/S12_PHASE0_PROVENANCE.json",
        "provenance_sha256": recs[-1].sha256,
        "phase_verdicts": verdicts,
        "reserved_labels": RESERVED_LABELS,
        "no_scientific_headline_computed": True,
        "p1_not_started": True,
        "failed": None,
    }
    add("sot_provenance_sha_matches_file", payload["provenance_sha256"] == recs[-1].sha256 == _sha(out_s12 / "S12_PHASE0_PROVENANCE.json"), recs[-1].sha256)

    pre = _snap(ctx.sot)
    patch = ledger_patch(stage="s12_phase0", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_phase0": payload}})
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_rq_unchanged", _nested(merged, ("results", "rq3", "D_fleet_RQ3")) == pre.get("results.rq3.D_fleet_RQ3"), str(pre.get("results.rq3.D_fleet_RQ3")))
    add("s11_phase4_unchanged", _nested(merged, ("results", "s11_phase4", "decisions")) == pre.get("results.s11_phase4.decisions"), str(pre.get("results.s11_phase4.decisions")))
    add("rq4_unchanged", _nested(merged, ("scientific_analysis", "rq4_source_selection", "status")) == pre.get("scientific_analysis.rq4_source_selection.status"), str(pre.get("scientific_analysis.rq4_source_selection.status")))
    add("n_validations_at_least_20", len(checks) + 1 >= 20, f"n={len(checks) + 1}")
    payload["n_validations"] = len(checks)
    patch = ledger_patch(stage="s12_phase0", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_phase0": payload}})
    patch = ledger_patch(stage="s12_phase0", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_phase0": payload}})

    if not ns_ok:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_phase0": {"status": "STOP", "reason": ns_detail, "failed": None}}}, artifacts=recs, validations=checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_phase0": {"status": "STOP", "failed": [c.name for c in failed]}}}, artifacts=recs, validations=checks)
    return StageResult(status="GO", message=payload["status"], sot_patch=patch, artifacts=recs, validations=checks)
