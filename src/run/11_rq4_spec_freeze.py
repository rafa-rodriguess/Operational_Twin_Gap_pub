"""11 — freeze RQ4 source-selection specification into the SoT (no scoring)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact
from src.run._ledger import ledger_patch

SPEC = Path("docs/method_amendments/RQ4_SOURCE_SELECTION_FROZEN_SPEC.md")
PROMPT = Path("docs/prompts/FREEZE_RQ4_SOURCE_SELECTION_PROTOCOL.md")
RQ4 = (
    "RQ4 — How much target-side verification data are needed to select a source "
    "model whose future transfer error is close to the best available source, and "
    "when does the remaining selection regret become no larger than the temporal "
    "instability of the future source ranking itself?"
)


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    spec = root / SPEC
    prompt = root / PROMPT
    if not spec.is_file() or not prompt.is_file():
        return StageResult(
            status="STOP",
            message="missing RQ4 specification or archived prompt",
            sot_patch={"results": {"rq4_spec_freeze": {"status": "STOP"}}},
            validations=[ValidationRecord("spec_files", False, f"spec={spec.is_file()} prompt={prompt.is_file()}")],
        )
    text = spec.read_text(encoding="utf-8")
    recs = [artifact(root, spec), artifact(root, prompt)]
    spec_sha = recs[0].sha256
    freeze = {
        "status": "FROZEN",
        "rq4_wording": RQ4,
        "dataset": "canonical BR-PVGen analytical data in project SoT",
        "target": "y_dc_normalized",
        "features": [
            "poa_irradiance_wm2",
            "ghi_irradiance_wm2",
            "gri_irradiance_wm2",
            "panel_temperature_celsius",
            "ambient_temperature_celsius",
            "wind_speed_ms",
        ],
        "eligibility": {
            "poa_irradiance_wm2_gt": 50,
            "coverage_dc_ge": 0.80,
            "y_finite": True,
            "six_features_finite": True,
            "exclude_interpolated_true": True,
            "unknown_interpolation_not_false": True,
        },
        "target_cohort": "protected confirmatory cohort from project partition artifact; reconstruct and hash-verify",
        "source_cohort": "authorized development/source population; freeze IDs+hash before confirmatory scoring",
        "source_lookback_days": 30,
        "train_validation_split": "chronological_80_20",
        "primary_model": "SplineRidge",
        "primary_model_grid": {"degree": 3, "n_knots": [4, 6, 8], "alpha": [0.1, 1, 10]},
        "sensitivity_model": "HistGradientBoostingRegressor",
        "sensitivity_model_grid": {
            "learning_rate": [0.05, 0.10],
            "max_leaf_nodes": [15, 31],
            "l2_regularization": [0, 1],
            "loss": "absolute_error",
            "max_iter": 300,
            "early_stopping": False,
            "min_samples_leaf": 20,
        },
        "k_grid": [0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45],
        "K_max": 45,
        "H": 45,
        "cut_spacing_days": 45,
        "candidate_pool": "P_ALL",
        "metadata_distance": "gower",
        "metadata_fields": [
            "is_panel_bifacial",
            "nominal_power_mw",
            "number_of_panels",
            "panel_area_mm2",
            "panel_bifaciality_coefficient",
            "panel_efficiency_percentage",
            "panel_temperature_coefficient",
            "structure_type",
        ],
        "shortlist_M": 5,
        "rho": 0.30,
        "support_definition": "CS4 source TRAIN; kNN=5; threshold=0.95 quantile of LOO 5th-neighbour distances",
        "regret_absolute_definition": "L_s_hat(R) - L_star(R)",
        "regret_relative_definition": "Reg_abs / L_star(R) if L_star finite and >0; no epsilon",
        "random_baseline_definition": "mean_s L_s(R) - L_star(R)",
        "metadata_baseline_definition": "expected regret under uniform min-Gower ties at k=0",
        "oracle_definition": "argmin_s L_s(R); oracle regret 0",
        "noise_floor_definition": "mean of A→B and B→A regret on alternating 7-day blocks inside R",
        "tau_primary": 0.05,
        "tau_sensitivity": 0.02,
        "T1": "target-balanced mean Delta_meta = Reg_random - Reg_metadata; 95% target-cluster bootstrap",
        "T2": "target-balanced mean Delta_verify = Reg_metadata - Reg_45; k fixed at 45",
        "T3": "persistent k_tau: earliest k with upper95CI(median Reg_rel(k)) <= tau for that k and all later k",
        "T4": "persistent k_noise: earliest k with upper95CI(median ExcessNoise(k)) <= 0 for that k and all later k",
        "aggregation": "within-target then equal-weight targets",
        "bootstrap_B": 5000,
        "bootstrap_seed": 42,
        "secondary_analyses": [
            "p75_rel_regret",
            "p90_rel_regret",
            "absolute_regret",
            "top1_oracle_recovery",
            "spearman_rank_Wk_vs_R",
            "exact_twin_present",
            "same_state_descriptive",
            "structure_type_descriptive",
            "seasonal_descriptive",
            "chronological_temporal_drift",
            "HGB_sensitivity",
        ],
        "claim_guardrails": "conditional on BR-PVGen fleet, frozen cohorts, horizon, CS4; no universal-day or causal claims",
        "validation_invariants": 51,
        "specification_path": SPEC.as_posix(),
        "specification_sha256": spec_sha,
        "prompt_path": PROMPT.as_posix(),
        "prompt_sha256": recs[1].sha256,
        "frozen_at_utc": now,
    }
    payload = {
        "status": "FROZEN",
        "message": "RQ4 source-selection protocol frozen; no outcome scoring",
        "specification_path": SPEC.as_posix(),
        "specification_sha256": spec_sha,
        "outcomes_computed": False,
    }
    checks = [
        ValidationRecord("exact_rq4_present", RQ4 in text, None),
        ValidationRecord("no_placeholder_TODO", "TODO" not in text and "TBD" not in text, None),
        ValidationRecord("k_grid_explicit", "k ∈ {0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45}" in text or "[0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45]" in text, None),
        ValidationRecord("no_rq4_scoring_artifacts", not (root / "artifacts/rq4").exists(), None),
    ]
    if not all(c.passed for c in checks):
        return StageResult(status="STOP", message="RQ4 freeze validation failed", sot_patch={}, validations=checks)
    return StageResult(
        status="GO",
        message="GO — RQ4 source-selection protocol frozen (no scoring)",
        sot_patch=ledger_patch(
            stage="rq4_spec_freeze",
            now=now,
            payload=payload,
            recs=recs,
            extra={"scientific_freeze": {"rq4_source_selection": freeze}},
        ),
        artifacts=recs,
        validations=checks,
    )
