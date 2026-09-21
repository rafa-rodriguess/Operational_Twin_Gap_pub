"""Stage 50 helper: freeze C8 Bonferroni fleet projection without computing the interval."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import dump_json, sha256_file
from src.lib.rq3_match import snapshot_tree
from src.lib.rq3_mbb import _art, _tab
from src.lib.rq3_align import repo_relative

JOB = "S08_STATE_CONSTRAINED_FLEET_INFERENCE_CONTROLLER_FREEZE"
JOB_PROMPT = "S08_STATE_CONSTRAINED_FLEET_INFERENCE_CONTROLLER_FREEZE_C8"
REVISION = "state_constrained_support_v2"
METHOD = "C8_BONFERRONI_SIMULTANEOUS_TARGET_MBB_PROJECTION"
SCOPE = "FIXED_SET_CONDITIONAL_FLEET_UNCERTAINTY"
P1 = "P1_CONDITIONAL_VALID_DRAW_RESAMPLING"
REASON = "S50_FLEET_FIXED_SET_BONFERRONI_PROJECTION_FROZEN"
MSG_GO = "GO — S08 REVISED RQ3 FLEET C8 BONFERRONI PROJECTION FROZEN"
MSG_STOP = "STOP — S08 FLEET INFERENCE METHOD NOT FROZEN"
OUT = "artifacts/s08_state_constrained_fleet_inference_freeze"
REPORT = "reports/freeze/S08_STATE_CONSTRAINED_FLEET_INFERENCE_CONTROLLER_FREEZE.md"
DECISION_MD = "docs/method_amendments/RQ3_FLEET_FIXED_SET_BONFERRONI_PROJECTION_CONTROLLER_DECISION.md"
ANSWER = None
ALPHA_FAMILY = 0.05
QUANTILE_METHOD = "numpy_percentile_linear"
PERCENTILE_CODE = (
    "np.percentile(target_draws, [100 * alpha_family / (2 * m), "
    "100 * (1 - alpha_family / (2 * m))], method=\"linear\")"
)
NIST = "https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm"
FORBIDDEN_PARSE = (
    "S08_STATE_SOURCE_LEVEL_MATCHED_CONTRASTS.csv",
    "S08_STATE_TARGET_LEVEL_MATCHED_CONTRASTS.csv",
    "S08_STATE_TARGET_LEVEL_BOOTSTRAP_DRAWS.parquet",
    "S08_STATE_TARGET_LEVEL_INFERENCE.csv",
    "S08_STATE_FLEET_DESCRIPTIVE_SUMMARY.json",
    "S08_STATE_GROUP_DESCRIPTIVE_SUMMARY.csv",
    "S08_STATE_HISTORICAL_SENSITIVITY.csv",
    "S08_STATE_INTERPRETATION_FACTS.json",
    "S08_STATE_CONSTRAINED_SCIENTIFIC_RQ3.md",
)
PROTECTED = (
    "artifacts/s03_rq3_state_revision",
    "artifacts/s04",
    "artifacts/s05",
    "artifacts/s06",
    "artifacts/s07",
    "artifacts/s08",
    "artifacts/s08_scientific",
    "artifacts/s08_state_constrained_support_alignment",
    "artifacts/s08_state_constrained_bootstrap_audit",
    "artifacts/s08_state_constrained_bootstrap_freeze",
    "artifacts/s08_state_constrained_scientific",
    "artifacts/s08_state_constrained_fleet_inference_audit",
    "artifacts/s09_scientific",
    "artifacts/s10",
    "paper",
    "paper.md",
)


class S50Stop(Exception):
    def __init__(self, reason: str, details: str = ""):
        super().__init__(details or reason)
        self.reason = reason
        self.details = details or reason


def _nested(sot: dict[str, Any], *keys: str) -> Any:
    cur: Any = sot
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def derive_multiplicity(m: int, alpha_family: float = ALPHA_FAMILY) -> dict[str, float | int]:
    if m <= 0:
        raise S50Stop("S50_STOP_M_NONPOSITIVE", str(m))
    return {
        "m": int(m),
        "alpha_family": float(alpha_family),
        "alpha_target": float(alpha_family) / float(m),
        "q_low": float(alpha_family) / (2.0 * float(m)),
        "q_high": 1.0 - (float(alpha_family) / (2.0 * float(m))),
    }


def _guard_parse(path: Path) -> None:
    name = path.name
    posix = path.as_posix()
    for marker in FORBIDDEN_PARSE:
        if marker in name or marker in posix:
            raise S50Stop("S50_STOP_FORBIDDEN_RESULT_PARSE", posix)


def _read_csv(path: Path) -> list[dict[str, str]]:
    _guard_parse(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _hash_rec(root: Path, rec: dict[str, Any], default: str, label: str) -> Path:
    path = root / (rec.get("path") or default)
    _guard_parse(path)
    if not path.is_file():
        raise S50Stop(f"S50_STOP_MISSING_{label}")
    digest = sha256_file(path)
    if rec.get("sha256") and digest != rec["sha256"]:
        raise S50Stop(f"S50_STOP_HASH_{label}", digest)
    return path


def validate_upstream(sot: dict[str, Any], root: Path) -> dict[str, Any]:
    if _nested(sot, "stages", "S08_STATE_CONSTRAINED_SCIENTIFIC_RQ3", "status") != "GO":
        raise S50Stop("S50_STOP_STAGE45_NOT_GO")
    if _nested(sot, "scientific_analysis", "s08_active_revision") != REVISION:
        raise S50Stop("S50_STOP_ACTIVE_REVISION")
    if _nested(sot, "stages", "S08_STATE_CONSTRAINED_BOOTSTRAP_POLICY_CONTROLLER_FREEZE", "status") != "GO":
        raise S50Stop("S50_STOP_STAGE47_NOT_GO")
    pol = _nested(sot, "scientific_freeze", "s08_rq3_operationalization", "revisions", REVISION, "bootstrap_empty_draw_policy") or {}
    if pol.get("status") != "FROZEN" or pol.get("policy_id") != P1:
        raise S50Stop("S50_STOP_P1_NOT_FROZEN")
    for key in (
        "fixed_source_set_required",
        "equal_source_weights_required",
        "shared_calendar_required",
        "no_proposal_cap",
        "no_source_dropping",
        "no_global_intersection",
        "no_source_specific_calendar",
        "no_model_refit",
    ):
        if pol.get(key) is not True:
            raise S50Stop("S50_STOP_P1_FLAG", key)
    b_boot = int(pol.get("B_boot_retained") or 0)
    if b_boot != 5000:
        raise S50Stop("S50_STOP_P1_BBOOT", str(b_boot))
    st49 = _nested(sot, "stages", "S08_STATE_CONSTRAINED_FLEET_INFERENCE_DEPENDENCY_AUDIT") or {}
    if st49.get("status") != "STOP":
        raise S50Stop("S50_STOP_STAGE49_NOT_STOP", str(st49.get("status")))
    if st49.get("reason") != "S49_FLEET_INFERENCE_CONTROLLER_DECISION_REQUIRED":
        raise S50Stop("S50_STOP_STAGE49_REASON", str(st49.get("reason")))
    audit = _nested(sot, "scientific_freeze", "s08_rq3_operationalization", "revisions", REVISION, "fleet_inference_dependency_audit") or {}
    if audit.get("status") != "CONTROLLER_DECISION_REQUIRED":
        raise S50Stop("S50_STOP_AUDIT_STATUS", str(audit.get("status")))
    if audit.get("selected_method") is not None:
        raise S50Stop("S50_STOP_METHOD_ALREADY_SELECTED", str(audit.get("selected_method")))
    if audit.get("blinded_to_rq3_effects") is not True:
        raise S50Stop("S50_STOP_AUDIT_NOT_BLINDED")
    if audit.get("rq3_effects_accessed") is not False:
        raise S50Stop("S50_STOP_AUDIT_EFFECTS_ACCESSED")
    if audit.get("rq3_effects_computed") is not False:
        raise S50Stop("S50_STOP_AUDIT_EFFECTS_COMPUTED")
    if audit.get("superpopulation_inference_authorized") is not False:
        raise S50Stop("S50_STOP_SUPERPOP_ALREADY_AUTHORIZED")
    arts = st49.get("artifacts") or {}
    summary_path = _hash_rec(root, arts.get("summary") or {}, "artifacts/s08_state_constrained_fleet_inference_audit/S08_STATE_FLEET_INFERENCE_AUDIT_SUMMARY.json", "SUMMARY")
    cand_path = _hash_rec(root, arts.get("candidates") or audit.get("candidate_assessment") or {}, "artifacts/s08_state_constrained_fleet_inference_audit/S08_STATE_FLEET_INFERENCE_CANDIDATE_ASSESSMENT.csv", "CANDIDATES")
    comp_path = _hash_rec(root, arts.get("components") or {}, "artifacts/s08_state_constrained_fleet_inference_audit/S08_STATE_FLEET_DEPENDENCY_COMPONENTS.csv", "COMPONENTS")
    report_path = _hash_rec(root, arts.get("report") or {"path": REPORT.replace("CONTROLLER_FREEZE", "DEPENDENCY_AUDIT")}, "reports/freeze/S08_STATE_CONSTRAINED_FLEET_INFERENCE_DEPENDENCY_AUDIT.md", "REPORT")
    inc_path = _hash_rec(root, arts.get("incidence") or {}, "artifacts/s08_state_constrained_fleet_inference_audit/S08_STATE_FLEET_TARGET_PLANT_NODE_INCIDENCE.csv", "INCIDENCE")
    return {
        "pol": pol,
        "audit": audit,
        "st49": st49,
        "b_boot": b_boot,
        "summary_path": summary_path,
        "cand_path": cand_path,
        "comp_path": comp_path,
        "report_path": report_path,
        "inc_path": inc_path,
    }


def execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    try:
        return _execute(ctx, now, validations)
    except S50Stop as exc:
        validations.append(ValidationRecord(exc.reason, False, exc.details))
        return StageResult(
            status="STOP",
            message=MSG_STOP,
            sot_patch={
                "stages": {
                    JOB: {
                        "kind": "scientific_freeze_amendment",
                        "status": "STOP",
                        "reason": exc.reason,
                        "operational_stage_id": "50",
                        "job": JOB_PROMPT,
                        "details": exc.details,
                        "finished_at_utc": now,
                        "answer": ANSWER,
                        "report": REPORT,
                        "revision_id": REVISION,
                    }
                }
            },
            artifacts=[],
            validations=validations,
        )


def _execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    root = ctx.root
    before = snapshot_tree(root, PROTECTED)
    bundle = validate_upstream(ctx.sot, root)
    validations.append(ValidationRecord("prereqs", True, REVISION))
    incidence = _read_csv(bundle["inc_path"])
    j_fleet = sorted({r["target_plant_id"] for r in incidence if r.get("target_plant_id")})
    m = len(j_fleet)
    if m != 10:
        raise S50Stop("S50_STOP_M_RECONCILIATION", str(m))
    mult = derive_multiplicity(m, ALPHA_FAMILY)
    if mult["alpha_target"] != 0.005 or mult["q_low"] != 0.0025 or mult["q_high"] != 0.9975:
        raise S50Stop("S50_STOP_QUANTILE_RECONCILIATION", json.dumps(mult))
    summary = json.loads(bundle["summary_path"].read_text(encoding="utf-8"))
    if int((summary.get("revised_structure") or {}).get("target_count") or 0) != m:
        raise S50Stop("S50_STOP_SUMMARY_TARGET_COUNT")
    if summary.get("selected_method") is not None:
        raise S50Stop("S50_STOP_SUMMARY_METHOD_SET")
    validations.append(ValidationRecord("multiplicity", True, f"m={m}"))

    decision = {
        "status": "FROZEN",
        "method_id": METHOD,
        "scope": SCOPE,
        "post_result_methodological_amendment": True,
        "fleet_interval_computed_before_freeze": False,
        "stage50_effect_values_accessed": False,
        "stage50_effect_values_computed": False,
        "selection_basis": "STRUCTURAL_DEPENDENCY_AUDIT_PLUS_BONFERRONI_UNION_BOUND",
        "nist_reference": NIST,
        "alpha_family": ALPHA_FAMILY,
        "m_source": "STAGE49_REVISED_STRUCTURAL_TARGET_SET",
        "J_fleet": j_fleet,
        "m": m,
        "alpha_target_formula": "alpha_family / m",
        "two_sided_tail_formula": "alpha_family / (2*m)",
        "alpha_target": mult["alpha_target"],
        "q_low": mult["q_low"],
        "q_high": mult["q_high"],
        "quantile_method": QUANTILE_METHOD,
        "percentile_rule": PERCENTILE_CODE,
        "bootstrap_draw_source": "STAGE45_RETAINED_P1_TARGET_DRAWS",
        "B_boot_retained": bundle["b_boot"],
        "new_bootstrap_draws_authorized": False,
        "target_resampling_authorized": False,
        "component_resampling_authorized": False,
        "joint_component_mbb_authorized": False,
        "global_support_intersection_authorized": False,
        "model_refit_authorized": False,
        "point_estimand": "F1_TARGET_BALANCED_ARITHMETIC_MEAN",
        "projection_lower_formula": "mean(target_bonferroni_lower)",
        "projection_upper_formula": "mean(target_bonferroni_upper)",
        "requires_target_independence": False,
        "requires_component_exchangeability": False,
        "requires_sign_symmetry": False,
        "requires_label_exchangeability": False,
        "superpopulation_inference_authorized": False,
        "fleet_hypothesis_test": False,
        "fleet_p_value": None,
        "primary_target_ci_replaced": False,
        "not_c1_iid_target_resampling": True,
        "not_c2_independent_target_mbb_aggregation": True,
        "not_c3_component_sign_flip": True,
        "not_c4_component_cluster_bootstrap": True,
        "not_c5_joint_component_mbb": True,
        "not_c6_hierarchical_mixed_model": True,
        "not_c7_label_permutation": True,
        "stage51_result_accepted_regardless_of_zero_inclusion_or_width": True,
        "next_action": "RUN_STAGE_51",
    }
    recon = {
        "stage49_summary_sha256": sha256_file(bundle["summary_path"]),
        "stage49_candidates_sha256": sha256_file(bundle["cand_path"]),
        "stage49_components_sha256": sha256_file(bundle["comp_path"]),
        "stage49_report_sha256": sha256_file(bundle["report_path"]),
        "stage49_incidence_sha256": sha256_file(bundle["inc_path"]),
        "p1_policy_id": P1,
        "B_boot_retained": bundle["b_boot"],
        "m": m,
        "J_fleet": j_fleet,
        "selected_method_before": None,
        "selected_method_after": METHOD,
        "rq3_effects_accessed": False,
        "fleet_interval_computed": False,
        "stage51_executed": False,
    }

    decision_md = f"""# RQ3 fleet fixed-set Bonferroni projection — controller decision

## 1. Chronology (post-result methodological amendment)

Stage 45 revised RQ3 point estimates and primary target-level 95% intervals already existed before C8 was selected.
C8 is **not** pre-specified before all RQ3 results and is **not** a preregistered confirmatory analysis.
It is a **post-result methodological amendment** motivated by the same-state comparator/dependency redesign and the Stage 49 structural audit.
This freeze is committed **before any C8 fleet-level interval is computed**. Stage 50 did not read RQ3 effect values.

## 2. Stage 49 structural motivation

The historical fleet-inference graph is not representative of the revised target set. Revised target-level estimands are not IID. Shared plant-data nodes create dependency components. C1/C2 are invalid under that graph. C3/C4/C6/C7 require unverified assumptions. C5 would require a new joint temporal-bootstrap law. C0 remains valid as descriptive synthesis but supplies no fleet interval.

## 3. Selected method

`{METHOD}`

Scope: `{SCOPE}`.

C8 is not C1 (no IID target resampling), not C2 (no independent-target MBB aggregation into a synthetic fleet draw), not C3 (no component sign symmetry), not C4 (no component resampling; F1 equal-target weights preserved), not C5 (no joint component MBB), not C6 (no hierarchical/mixed model), not C7 (no twin/control label permutation).

## 4. Bonferroni inequality

NIST/SEMATECH e-Handbook of Statistical Methods, 7.4.7.3, Bonferroni's method: `{NIST}`

For a finite family of confidence statements (A_1,…,A_m),

P(∩_j A_j) ≥ 1 − Σ_j P(A_j^c).

If each target-level interval has marginal noncoverage at most α/m, simultaneous coverage of all m target parameters is at least 1−α, without requiring independence across targets.

## 5. Projection through F1

The frozen fleet functional is the equal-target mean of the m target contrasts. If simultaneously L_j ≤ Δ_j ≤ U_j for every j, then deterministically mean(L) ≤ D_fleet ≤ mean(U). The rectangular simultaneous set projected through F1 preserves equal target weights. No component weighting is introduced.

The Bonferroni step removes the need to model cross-target dependence. It does not make the target-level temporal MBB intervals exact finite-sample intervals. The fleet interval inherits the validity assumptions/approximation of the already-authorized target-level temporal MBB percentile procedure.

## 6. Exact α allocation

α_family = {ALPHA_FAMILY}.
m is |J_fleet| from the Stage 49 structural incidence table (currently reconciled m={m}).
α_target = α_family / m.
q_low = α_family / (2m); q_high = 1 − α_family / (2m).
Derived at current m: α_target={mult["alpha_target"]}, q_low={mult["q_low"]}, q_high={mult["q_high"]}.

## 7. Exact percentile method

Stage 51 must apply:

```python
{PERCENTILE_CODE}
```

This matches Stage 45 `method="linear"` used for primary target 95% intervals. Those primary 95% intervals remain the target-level primary intervals and are not replaced.

## 8. Existing P1 draws

Use Stage 45 retained P1 target-level draws. B_boot_retained = {bundle["b_boot"]}. No additional proposals, fits, seeds, cross-target resampling, component resampling, synchronization, or global support intersection.

## 9–12. Forbidden assumptions and operations

No target-IID assumption. No superpopulation inference. No p-value (`fleet_p_value` remains null; `fleet_hypothesis_test` is false). No new draws, models, masks, or matching.

## 13. NIST reference

{NIST}

## 14. Interval not computed before freeze

The C8 fleet interval itself was **not** computed before this freeze. Stage 50 computed no endpoints.

## 15. Accepted outcome rule

Stage 51's C8 result is accepted whether the interval excludes zero, includes zero, or is wide/uninformative. Manuscript language may later state zero inclusion/exclusion without translating that statement into an uncomputed p-value.
"""
    (root / DECISION_MD).parent.mkdir(parents=True, exist_ok=True)
    (root / DECISION_MD).write_text(decision_md, encoding="utf-8")

    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "S08_STATE_FLEET_INFERENCE_CONTROLLER_DECISION.json", decision)
    dump_json(out / "S08_STATE_FLEET_INFERENCE_FREEZE_RECONCILIATION.json", recon)
    recs = {
        "decision": _art(root, out / "S08_STATE_FLEET_INFERENCE_CONTROLLER_DECISION.json"),
        "reconciliation": _art(root, out / "S08_STATE_FLEET_INFERENCE_FREEZE_RECONCILIATION.json"),
        "decision_document": _art(root, root / DECISION_MD),
    }
    manifest = {
        "job": JOB_PROMPT,
        "revision_id": REVISION,
        "finished_at_utc": now,
        "run_id": ctx.run_id,
        "selected_method": METHOD,
        "fleet_p_value": None,
        "artifacts": {k: _tab(v) for k, v in recs.items()},
    }
    dump_json(out / "S08_STATE_FLEET_INFERENCE_FREEZE_MANIFEST.json", manifest)
    recs["manifest"] = _art(root, out / "S08_STATE_FLEET_INFERENCE_FREEZE_MANIFEST.json")

    report = f"""# S08 state-constrained fleet inference controller freeze (C8)

## Stage 49 resolution

Stage 49 STOP `{bundle["st49"].get("reason")}` is resolved by freezing `{METHOD}`.

## Why Bonferroni still applies under dependence

The familywise Bonferroni/union-bound step is valid under **arbitrary dependence** among the target confidence events, provided each marginal target interval has noncoverage at most α/m. Cross-target shared plant-data nodes therefore do not invalidate the simultaneous-coverage inequality. They do invalidate IID-target resampling (C1) and independent-target MBB aggregation (C2).

## Exact familywise formulas

α_family = {ALPHA_FAMILY}; m = |J_fleet| = {m} (derived from Stage 49 incidence, not hard-coded as a scientific branch).
α_target = α_family / m = {mult["alpha_target"]}.
q_low = α_family/(2m) = {mult["q_low"]}; q_high = 1 − α_family/(2m) = {mult["q_high"]}.
Percentile: `{PERCENTILE_CODE}`.

## Exact projection

L_fleet^B = mean_j L_j^B; U_fleet^B = mean_j U_j^B. F1 equal-target weighting is preserved. The F1 point estimand is unchanged and was not re-read here.

## Fixed-set scope

`{SCOPE}`. Superpopulation fleet inference is not authorized.

## Post-result amendment chronology

C8 was selected after Stage 45 results existed. It is a methodological amendment motivated by Stage 49 structure plus the Bonferroni coverage argument, not by observed RQ3 direction or magnitude. The C8 interval was not computed before this freeze.

## Not computed / not authorized

No fleet interval was computed. No p-value is authorized. No superpopulation inference is authorized. Primary target-level 95% intervals remain in place.

## Next action

`RUN_STAGE_51`. Stage 51 was not executed.
"""
    (root / REPORT).write_text(report, encoding="utf-8")
    recs["report"] = _art(root, root / REPORT)

    after = snapshot_tree(root, PROTECTED)
    if before != after:
        raise S50Stop("S50_STOP_PROTECTED_MUTATION")
    validations.append(ValidationRecord("blinded_freeze", True, None))
    patch_arts = {k: _tab(v) for k, v in recs.items()}
    freeze_node = {
        "status": "FROZEN",
        "method_id": METHOD,
        "scope": SCOPE,
        "post_result_methodological_amendment": True,
        "fleet_interval_computed_before_freeze": False,
        "stage50_effect_values_accessed": False,
        "stage50_effect_values_computed": False,
        "selection_basis": "STRUCTURAL_DEPENDENCY_AUDIT_PLUS_BONFERRONI_UNION_BOUND",
        "alpha_family": ALPHA_FAMILY,
        "m_source": "STAGE49_REVISED_STRUCTURAL_TARGET_SET",
        "m": m,
        "alpha_target_formula": "alpha_family / m",
        "two_sided_tail_formula": "alpha_family / (2*m)",
        "quantile_method": QUANTILE_METHOD,
        "bootstrap_draw_source": "STAGE45_RETAINED_P1_TARGET_DRAWS",
        "B_boot_retained": bundle["b_boot"],
        "new_bootstrap_draws_authorized": False,
        "target_resampling_authorized": False,
        "component_resampling_authorized": False,
        "joint_component_mbb_authorized": False,
        "global_support_intersection_authorized": False,
        "model_refit_authorized": False,
        "point_estimand": "F1_TARGET_BALANCED_ARITHMETIC_MEAN",
        "projection_lower_formula": "mean(target_bonferroni_lower)",
        "projection_upper_formula": "mean(target_bonferroni_upper)",
        "requires_target_independence": False,
        "requires_component_exchangeability": False,
        "requires_sign_symmetry": False,
        "requires_label_exchangeability": False,
        "superpopulation_inference_authorized": False,
        "fleet_hypothesis_test": False,
        "fleet_p_value": None,
        "primary_target_ci_replaced": False,
        "selected_method": METHOD,
        "next_action": "RUN_STAGE_51",
        "decision": _tab(recs["decision"]),
        "reconciliation": _tab(recs["reconciliation"]),
        "manifest": _tab(recs["manifest"]),
        "decision_document": _tab(recs["decision_document"]),
        "report": REPORT,
    }
    sot_patch = {
        "stages": {
            JOB: {
                "kind": "scientific_freeze_amendment",
                "status": "GO",
                "reason": REASON,
                "operational_stage_id": "50",
                "job": JOB_PROMPT,
                "run_id": ctx.run_id,
                "finished_at_utc": now,
                "answer": ANSWER,
                "report": REPORT,
                "revision_id": REVISION,
                "selected_method": METHOD,
                "next_stage": "51",
                "artifacts": patch_arts,
            }
        },
        "scientific_freeze": {
            "s08_rq3_operationalization": {
                "revisions": {
                    REVISION: {
                        "fleet_inference_dependency_audit": {
                            "resolved_by_stage": "50",
                            "controller_resolution_status": "FROZEN",
                            "controller_selected_method": METHOD,
                            "next_action": "RUN_STAGE_51",
                        },
                        "fleet_inference_controller_freeze": freeze_node,
                    }
                }
            }
        },
    }
    return StageResult(status="GO", message=MSG_GO, sot_patch=sot_patch, artifacts=list(recs.values()), validations=validations)
