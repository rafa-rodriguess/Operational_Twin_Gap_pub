"""28 — S12 P9 environmental-distribution distance."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, read_csv, sha256_file, write_csv
import os

from src.lib.s12_p9_environmental_distance import (
    MAX_DIST_ROWS,
    N_BOOT,
    SEED,
    STATUS,
    render_report,
    run_reporting_from_disk,
    run_science,
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
    ("results", "s12_p8", "status"),
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
    if set((patch.get("results") or {}).keys()) != {"s12_p9"}:
        return False, "results keys"
    if set((patch.get("scientific_analysis") or {}).keys()) != {"s12_p9"}:
        return False, "sci keys"
    if _snap(merged) != pre:
        return False, "prior snapshot changed"
    return True, "s12_p9 only"


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex, pdf = root / "paper/main.tex", root / "paper/main.pdf"
    tex_b, pdf_b = _sha(tex), _sha(pdf)
    s07_b = _sha(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv")
    s05_b = _sha(root / "artifacts/s05/S05_SUPPORT_MASKS.parquet")
    p8_b = _sha(root / "artifacts/s12/p8/S12_P8_SUMMARY.json")
    lib_src = (root / "src/lib/s12_p9_environmental_distance.py").read_text(encoding="utf-8")
    add("no_model_fit_token", ".fit(" not in lib_src and "sklearn" not in lib_src, "no sklearn/fit")
    add("no_predict_scoring", ".predict(" not in lib_src, "no predict")

    out = root / "artifacts/s12/p9"
    frozen_rel = (
        "artifacts/s12/p9/S12_P9_SOURCE_SCALERS.csv",
        "artifacts/s12/p9/S12_P9_DIRECTIONAL_DISTANCE.csv",
        "artifacts/s12/p9/S12_P9_TARGET_BALANCED.csv",
        "artifacts/s12/p9/S12_P9_RQ2_DISTANCE_ASYMMETRY.csv",
        "artifacts/s12/p9/S12_P9_BOOTSTRAP.csv",
        "artifacts/s12/p9/S12_P9_DISTANCE_TERCILES.csv",
    )
    frozen_before = {rel: _sha(root / rel) for rel in frozen_rel}
    reporting_only = (root / frozen_rel[1]).is_file() and os.environ.get("S12_P9_FORCE_REFIT") != "1"
    if reporting_only:
        summary = run_reporting_from_disk(root)
        sci = {
            "summary": summary,
            "scalers": read_csv(root / frozen_rel[0]),
            "directional": read_csv(root / frozen_rel[1]),
            "target_balanced": read_csv(root / frozen_rel[2]),
            "pairs": read_csv(root / frozen_rel[3]),
            "bootstrap": read_csv(root / frozen_rel[4]),
            "terciles": read_csv(root / frozen_rel[5]),
        }
    else:
        sci = run_science(root)
        summary = sci["summary"]
    if not summary.get("reconciliation_ok"):
        add("independent_reconciliation", False, str(summary.get("reconciliation_max_abs_discrepancy")))
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p9": {"status": "STOP", "failed": ["independent_reconciliation"], "reconciliation_max_abs_discrepancy": summary.get("reconciliation_max_abs_discrepancy")}}}, validations=checks)

    out.mkdir(parents=True, exist_ok=True)
    if not reporting_only:
        write_csv(out / "S12_P9_SOURCE_SCALERS.csv", sci["scalers"])
        write_csv(out / "S12_P9_DIRECTIONAL_DISTANCE.csv", sci["directional"])
        write_csv(out / "S12_P9_TARGET_BALANCED.csv", sci["target_balanced"])
        write_csv(out / "S12_P9_RQ2_DISTANCE_ASYMMETRY.csv", sci["pairs"])
        write_csv(out / "S12_P9_BOOTSTRAP.csv", sci["bootstrap"])
        write_csv(out / "S12_P9_DISTANCE_TERCILES.csv", sci["terciles"])
    dump_json(out / "S12_P9_SUMMARY.json", summary)
    md = root / "reports/scientific/S12_P9_ENVIRONMENTAL_DISTANCE.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_report(summary), encoding="utf-8")

    recs = [
        artifact(root, out / "S12_P9_SOURCE_SCALERS.csv", rows=len(sci["scalers"]), fp=csv_fingerprint(out / "S12_P9_SOURCE_SCALERS.csv")),
        artifact(root, out / "S12_P9_DIRECTIONAL_DISTANCE.csv", rows=len(sci["directional"]), fp=csv_fingerprint(out / "S12_P9_DIRECTIONAL_DISTANCE.csv")),
        artifact(root, out / "S12_P9_TARGET_BALANCED.csv", rows=len(sci["target_balanced"]), fp=csv_fingerprint(out / "S12_P9_TARGET_BALANCED.csv")),
        artifact(root, out / "S12_P9_RQ2_DISTANCE_ASYMMETRY.csv", rows=len(sci["pairs"]), fp=csv_fingerprint(out / "S12_P9_RQ2_DISTANCE_ASYMMETRY.csv")),
        artifact(root, out / "S12_P9_BOOTSTRAP.csv", rows=len(sci["bootstrap"]), fp=csv_fingerprint(out / "S12_P9_BOOTSTRAP.csv")),
        artifact(root, out / "S12_P9_DISTANCE_TERCILES.csv", rows=len(sci["terciles"]), fp=csv_fingerprint(out / "S12_P9_DISTANCE_TERCILES.csv")),
        artifact(root, out / "S12_P9_SUMMARY.json"),
        artifact(root, md),
    ]
    prov_rel = "artifacts/s12/p9/S12_P9_PROVENANCE.json"
    provenance = {
        "repository_head": _git_head(root),
        "canonical_inputs": {
            "artifacts/p02c/P02C_PLANT_PANEL.parquet": sha256_file(root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"),
            "artifacts/s04/S04_ANALYTIC_INDEX.parquet": sha256_file(root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet"),
            "artifacts/s05/S05_SUPPORT_MASKS.parquet": s05_b,
            "artifacts/s07/S07_OTG_DIRECTIONAL.csv": s07_b,
            "artifacts/s07/S07_ASYMMETRY.csv": sha256_file(root / "artifacts/s07/S07_ASYMMETRY.csv"),
        },
        "six_feature_schema": summary["features"],
        "source_scaler_rule": "median and IQR from canonical source TRAIN only; z=(x-median)/IQR",
        "full_test": "canonical target TEST rows before CS4",
        "supported_test": "accepted S05 CS4 supported target TEST identities; no new mask",
        "sampling": {
            "max_dist_rows": MAX_DIST_ROWS,
            "sens": 1024,
            "master_seed": SEED,
            "child_seed": "sha256(master|identity)[:8] little-endian uint32; identities source_train|pid|cap, target_test_full|pid|cap, target_test_supported|src|tgt|cap",
            "association_stability_1024_vs_2048": "Spearman/partial Spearman recomputed from persisted directional 2048 and cap1024 columns; same average-rank residualization as headline",
        },
        "energy_formula": summary["energy_formula"],
        "mmd_formula": summary["mmd_formula"],
        "bandwidth_rule": "median of positive off-diagonal Euclidean distances of capped source TRAIN sample at that cap",
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "algorithm": "resample target IDs with replacement; multiplicity; rerank/residualize inside partial replicates"},
        "partial_spearman": summary["partial_spearman"],
        "rq2_definitions": "A_OTG=OTG(i->j)-OTG(j->i); A_ED=ED_supported(i->j)-ED_supported(j->i)",
        "script_path": "src/run/28_s12_p9_environmental_distance.py",
        "helper_path": "src/lib/s12_p9_environmental_distance.py",
        "generated_at_utc": now,
        "no_refit": True,
        "no_scoring": True,
        "no_support_rebuild": True,
        "accepted_science_not_modified": True,
        "manuscript_files_not_edited": True,
        "self_hash_registered_externally": True,
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes, "row_count": rec.row_count} for rec in recs},
    }
    dump_json(out / "S12_P9_PROVENANCE.json", provenance)
    recs.append(artifact(root, out / "S12_P9_PROVENANCE.json"))

    add("n_14_scalers", len(sci["scalers"]) == 14, str(len(sci["scalers"])))
    add("n_85", len(sci["directional"]) == 85, str(len(sci["directional"])))
    add("n_38", len(sci["pairs"]) == 38, str(len(sci["pairs"])))
    add("six_features", len(summary["features"]) == 6, str(summary["features"]))
    add("independent_reconciliation", summary["reconciliation_ok"], str(summary["reconciliation_max_abs_discrepancy"]))
    add("p2_otg_retention", summary["p2_reconciliation"]["max_abs_otg"] <= 1e-12 and summary["p2_reconciliation"]["max_abs_retention"] <= 1e-12, str(summary["p2_reconciliation"]))
    add("bootstrap_5000", N_BOOT == 5000 and SEED == 20260921, str(N_BOOT))
    add("no_joblib", list(out.glob("**/*.joblib")) == [], "none")
    add("no_refit", summary["no_refit"] and summary["no_scoring"] and summary["no_support_rebuild"], "ok")
    add("no_thresholds", summary["interpretation"]["threshold_rules_used"] is False, "ok")
    add("s07_unchanged", s07_b == _sha(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"), s07_b)
    add("s05_unchanged", s05_b == _sha(root / "artifacts/s05/S05_SUPPORT_MASKS.parquet"), s05_b)
    add("s12_p8_unchanged", p8_b == _sha(root / "artifacts/s12/p8/S12_P8_SUMMARY.json"), p8_b)
    add("provenance_omits_self", prov_rel not in provenance["outputs"], "ok")
    add("paper_tex_unchanged", tex_b == _sha(tex), _sha(tex))
    add("paper_pdf_unchanged", pdf_b == _sha(pdf), _sha(pdf))
    add("frozen_p9_science_unchanged", reporting_only and all(_sha(root / rel) == frozen_before[rel] for rel in frozen_rel), "byte-identical")
    add("association_stability_block", bool((summary.get("sampling") or {}).get("association_stability_1024_vs_2048", {}).get("comparisons")), "6 comparisons")
    add("n_6_stability", len((summary.get("sampling") or {}).get("association_stability_1024_vs_2048", {}).get("comparisons") or []) == 6, "6")

    payload = {
        "status": STATUS,
        "n_directional": 85,
        "n_pairs": 38,
        "sampling": summary["sampling"],
        "assoc": summary["assoc"],
        "bootstrap": summary["bootstrap"],
        "target_balanced": summary["target_balanced"],
        "groups": summary["groups"],
        "rq2": summary["rq2"],
        "interpretation": summary["interpretation"],
        "p2_reconciliation": summary["p2_reconciliation"],
        "reporting_correction": True,
        "sampling_association_stability": (summary.get("sampling") or {}).get("association_stability_1024_vs_2048"),
        "scalers_path": "artifacts/s12/p9/S12_P9_SOURCE_SCALERS.csv",
        "scalers_sha256": recs[0].sha256,
        "directional_path": "artifacts/s12/p9/S12_P9_DIRECTIONAL_DISTANCE.csv",
        "directional_sha256": recs[1].sha256,
        "target_balanced_path": "artifacts/s12/p9/S12_P9_TARGET_BALANCED.csv",
        "target_balanced_sha256": recs[2].sha256,
        "rq2_path": "artifacts/s12/p9/S12_P9_RQ2_DISTANCE_ASYMMETRY.csv",
        "rq2_sha256": recs[3].sha256,
        "bootstrap_path": "artifacts/s12/p9/S12_P9_BOOTSTRAP.csv",
        "bootstrap_sha256": recs[4].sha256,
        "terciles_path": "artifacts/s12/p9/S12_P9_DISTANCE_TERCILES.csv",
        "terciles_sha256": recs[5].sha256,
        "summary_path": "artifacts/s12/p9/S12_P9_SUMMARY.json",
        "summary_sha256": recs[6].sha256,
        "report_path": "reports/scientific/S12_P9_ENVIRONMENTAL_DISTANCE.md",
        "report_sha256": recs[7].sha256,
        "provenance_path": prov_rel,
        "provenance_sha256": recs[-1].sha256,
        "failed": None,
    }
    pre = _snap(ctx.sot)
    patch = ledger_patch(stage="s12_p9", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p9": payload}})
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_rq_unchanged", _nested(merged, ("results", "tables", "rq1")) == pre.get("results.tables.rq1"), "rq1")
    add("s12_p8_sot_unchanged", _nested(merged, ("results", "s12_p8", "status")) == pre.get("results.s12_p8.status"), "p8")
    payload["n_validations"] = len(checks)
    patch = ledger_patch(stage="s12_p9", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s12_p9": payload}})
    if not ns_ok:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p9": {"status": "STOP", "reason": ns_detail}}}, artifacts=recs, validations=checks)
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(status="STOP", message="STOP_RECONCILIATION_FAILURE", sot_patch={"results": {"s12_p9": {"status": "STOP", "failed": [c.name for c in failed]}}}, artifacts=recs, validations=checks)
    return StageResult(status="GO", message=payload["status"], sot_patch=patch, artifacts=recs, validations=checks)
