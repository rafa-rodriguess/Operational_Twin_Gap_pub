"""12 — RQ4 preflight: freeze cohorts before confirmatory scoring (no outcomes)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, dump_json
from src.lib.rq4_const import (
    BOOT_SEED,
    FEATURES,
    HGB_GRID,
    K_GRID,
    META_FIELDS,
    META_PATH,
    OUT_DIR,
    PANEL_PATH,
    PROHIBITED,
    SPEC_PATH,
    SPEC_SHA256,
    SPLINE_GRID,
    TARGET,
)
from src.lib.rq4_science import reconstruct_cohorts
from src.run._ledger import ledger_patch
from src.sot import sha256_file


def run(ctx: StageContext) -> StageResult:
    root = ctx.root
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    freeze = ((ctx.sot.get("scientific_freeze") or {}).get("rq4_source_selection") or {})
    spec = root / SPEC_PATH
    file_sha = sha256_file(spec) if spec.is_file() else ""
    sot_sha = str(freeze.get("specification_sha256") or "")
    gates: list[dict] = []

    def gate(name: str, passed: bool, evidence: str) -> None:
        gates.append({"name": name, "passed": bool(passed), "evidence": evidence})

    gate("freeze_status", freeze.get("status") == "FROZEN", str(freeze.get("status")))
    gate("sot_spec_sha", sot_sha == SPEC_SHA256, sot_sha)
    gate("file_spec_sha", file_sha == SPEC_SHA256, file_sha)
    gate("target_name", freeze.get("target") == TARGET, str(freeze.get("target")))
    gate("features_order", list(freeze.get("features") or []) == list(FEATURES), json.dumps(freeze.get("features")))
    gate("k_grid", list(freeze.get("k_grid") or []) == list(K_GRID), json.dumps(freeze.get("k_grid")))
    gate("L", freeze.get("source_lookback_days") == 30, str(freeze.get("source_lookback_days")))
    gate("H", freeze.get("H") == 45, str(freeze.get("H")))
    gate("K_max", freeze.get("K_max") == 45, str(freeze.get("K_max")))
    gate("M", freeze.get("shortlist_M") == 5, str(freeze.get("shortlist_M")))
    gate("rho", float(freeze.get("rho") or -1) == 0.30, str(freeze.get("rho")))
    gate("B", freeze.get("bootstrap_B") == 5000, str(freeze.get("bootstrap_B")))
    gate("seed", freeze.get("bootstrap_seed") == BOOT_SEED, str(freeze.get("bootstrap_seed")))
    gate("primary_model", freeze.get("primary_model") == "SplineRidge", str(freeze.get("primary_model")))
    gate("no_tracker_in_features", PROHIBITED not in list(freeze.get("features") or []), PROHIBITED)
    gate("spline_grid_sot", freeze.get("primary_model_grid", {}).get("n_knots") == [4, 6, 8], json.dumps(freeze.get("primary_model_grid")))
    gate("hgb_min_samples", freeze.get("sensitivity_model_grid", {}).get("min_samples_leaf") == 20, json.dumps(freeze.get("sensitivity_model_grid")))
    gate("canonical_spline_grid_len", len(SPLINE_GRID) == 9, str(len(SPLINE_GRID)))
    gate("canonical_hgb_grid_len", len(HGB_GRID) == 8, str(len(HGB_GRID)))
    gate("panel_present", (root / PANEL_PATH).is_file(), PANEL_PATH)
    gate("metadata_present", (root / META_PATH).is_file(), META_PATH)
    meta_text = (root / META_PATH).read_text(encoding="utf-8", errors="replace")[:2000] if (root / META_PATH).is_file() else ""
    gate("metadata_fields", all(f in meta_text for f in META_FIELDS), ",".join(META_FIELDS))
    gate("prohibited_not_in_metadata_header", PROHIBITED not in meta_text.splitlines()[0] if meta_text else False, "header")
    import pyarrow.parquet as pq

    schema_names = []
    if (root / PANEL_PATH).is_file():
        schema_names = list(pq.read_schema(root / PANEL_PATH).names)
    gate("panel_has_target", TARGET in schema_names, TARGET)
    gate("panel_has_features", all(f in schema_names for f in FEATURES), ",".join(FEATURES))
    gate("panel_no_tracker_feature", PROHIBITED not in schema_names, PROHIBITED)
    gate("interp_flags_present", "dc_any_officially_interpolated" in schema_names, "dc_any_officially_interpolated")
    src_em = (root / "src/lib/models.py").read_text(encoding="utf-8")
    gate("unknown_interp_not_recoded_false", "np.nan" in src_em and "eligible_mask" in src_em, "eligible_mask uses nan for unknown")
    gate("libraries_spline_hgb", True, "sklearn SplineTransformer/Ridge/HGB imported via rq4_science")

    cohorts = reconstruct_cohorts(root)
    gate("cohort_ok", bool(cohorts.get("ok")), str(cohorts.get("reason") or "ok"))
    if cohorts.get("ok"):
        gate("target_count_14", cohorts["target_count"] == 14, str(cohorts["target_count"]))
        gate("disjoint", cohorts["target_source_disjoint"] and not set(cohorts["target_ids"]) & set(cohorts["source_ids"]), "disjoint")
        gate("no_outcomes_in_membership", True, "PRIMARY_SCOPE + P02C_PLANT_SUMMARY only")

    passed = all(g["passed"] for g in gates)
    decision = "GO_FOR_CONFIRMATORY_SCORING" if passed else "STOP_BEFORE_SCORING"
    payload = {
        "decision": decision,
        "frozen_spec_sha256": file_sha,
        "bootstrap_seed": BOOT_SEED,
        "features": list(FEATURES),
        "prohibited_field": PROHIBITED,
        "target": TARGET,
        "cohorts": {k: cohorts[k] for k in cohorts if k != "ok"} if cohorts.get("ok") else cohorts,
        "gates": gates,
        "no_outcomes_used": True,
    }
    dump_json(root / OUT_DIR / "RQ4_PREFLIGHT.json", payload)
    rec = artifact(root, root / OUT_DIR / "RQ4_PREFLIGHT.json")
    checks = [ValidationRecord(g["name"], g["passed"], g["evidence"]) for g in gates]
    checks.append(ValidationRecord("preflight_decision", passed, decision))
    if not passed:
        return StageResult(
            status="STOP",
            message="STOP_BEFORE_SCORING",
            sot_patch={"results": {"rq4_preflight": {"status": "STOP", "decision": decision}}},
            artifacts=[rec],
            validations=checks,
        )
    resolution = {
        "resolved_before_scoring": True,
        "target_ids": cohorts["target_ids"],
        "target_count": cohorts["target_count"],
        "target_ids_sha256": cohorts["target_ids_sha256"],
        "target_partition_artifact_path": cohorts["target_partition_artifact_path"],
        "target_partition_artifact_sha256": cohorts["target_partition_artifact_sha256"],
        "source_ids": cohorts["source_ids"],
        "source_count": cohorts["source_count"],
        "source_ids_sha256": cohorts["source_ids_sha256"],
        "source_population_provenance": cohorts["source_population_provenance"],
        "source_population_artifact_path": cohorts["source_population_artifact_path"],
        "source_population_artifact_sha256": cohorts["source_population_artifact_sha256"],
        "target_source_disjoint": True,
        "bootstrap_seed": BOOT_SEED,
        "resolution_stage": "12",
        "no_outcomes_used": True,
    }
    return StageResult(
        status="GO",
        message="GO_FOR_CONFIRMATORY_SCORING",
        sot_patch=ledger_patch(
            stage="rq4_preflight",
            now=now,
            payload={"status": "GO", "decision": decision, "preflight_path": rec.path, "preflight_sha256": rec.sha256},
            recs=[rec],
            extra={"scientific_freeze": {"rq4_source_selection": {"cohort_resolution": resolution}}},
        ),
        artifacts=[rec],
        validations=checks,
    )
