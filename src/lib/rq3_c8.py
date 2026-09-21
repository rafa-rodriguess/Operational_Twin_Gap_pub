"""Stage 51 helper: frozen C8 Bonferroni fleet fixed-set interval materialization."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import dump_json, sha256_file, write_csv
from src.lib.rq3_match import snapshot_tree
from src.lib.rq3_mbb import _art, _pq_fp, _tab
from src.lib.rq3_c8_const import ALPHA_FAMILY, METHOD, SCOPE, derive_multiplicity
from src.lib.rq3_align import repo_relative

JOB = "S08_STATE_CONSTRAINED_FLEET_FIXED_SET_INFERENCE"
JOB_PROMPT = "S08_STATE_CONSTRAINED_FLEET_FIXED_SET_INFERENCE_C8"
REVISION = "state_constrained_support_v2"
REASON = "S51_FLEET_FIXED_SET_C8_INFERENCE_COMPLETE"
MSG_GO = "GO — S08 REVISED RQ3 C8 FIXED-SET FLEET INTERVAL MATERIALIZED"
MSG_STOP = "STOP — S08 C8 FIXED-SET FLEET INTERVAL NOT MATERIALIZED"
OUT = "artifacts/s08_state_constrained_fleet_fixed_set_inference"
REPORT = "reports/scientific/S08_STATE_CONSTRAINED_FLEET_FIXED_SET_INFERENCE.md"
ANSWER = None
P1 = "P1_CONDITIONAL_VALID_DRAW_RESAMPLING"
B_BOOT = 5000
CSV_FIELDS = [
    "target_plant_id",
    "draw_count",
    "alpha_family",
    "m",
    "alpha_target",
    "q_low_probability",
    "q_high_probability",
    "q_low_percentile",
    "q_high_percentile",
    "quantile_method",
    "bonferroni_ci_lower",
    "bonferroni_ci_upper",
    "bonferroni_interval_level",
    "quantile_index_low",
    "quantile_index_high",
    "min_retained_draw",
    "max_retained_draw",
    "L_j",
    "source_count_denominator",
]
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
    "artifacts/s08_state_constrained_fleet_inference_freeze",
    "artifacts/s09_scientific",
    "artifacts/s10",
    "paper",
    "paper.md",
)


class S51Stop(Exception):
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


def _hash_rec(root: Path, rec: dict[str, Any], default: str, label: str) -> Path:
    path = root / (rec.get("path") or default)
    if not path.is_file():
        raise S51Stop(f"S51_STOP_MISSING_{label}")
    digest = sha256_file(path)
    if rec.get("sha256") and digest != rec["sha256"]:
        raise S51Stop(f"S51_STOP_HASH_{label}", digest)
    return path


def c8_target_interval(draws: np.ndarray, m: int, alpha_family: float = ALPHA_FAMILY) -> tuple[float, float]:
    lo, hi = np.percentile(
        np.asarray(draws, dtype=float),
        [100 * alpha_family / (2 * m), 100 * (1 - alpha_family / (2 * m))],
        method="linear",
    )
    return float(lo), float(hi)


def validate_upstream(sot: dict[str, Any], root: Path) -> dict[str, Any]:
    if _nested(sot, "stages", "S08_STATE_CONSTRAINED_SCIENTIFIC_RQ3", "status") != "GO":
        raise S51Stop("S51_STOP_STAGE45_NOT_GO")
    if _nested(sot, "scientific_analysis", "s08_active_revision") != REVISION:
        raise S51Stop("S51_STOP_ACTIVE_REVISION")
    sci = _nested(sot, "scientific_analysis", "s08_revisions", REVISION, "scientific") or {}
    if sci.get("status") != "GO":
        raise S51Stop("S51_STOP_SCIENTIFIC_STATUS")
    if _nested(sot, "stages", "S08_STATE_CONSTRAINED_BOOTSTRAP_POLICY_CONTROLLER_FREEZE", "status") != "GO":
        raise S51Stop("S51_STOP_STAGE47_NOT_GO")
    pol = _nested(sot, "scientific_freeze", "s08_rq3_operationalization", "revisions", REVISION, "bootstrap_empty_draw_policy") or {}
    if pol.get("policy_id") != P1 or int(pol.get("B_boot_retained") or 0) != B_BOOT:
        raise S51Stop("S51_STOP_P1")
    st49 = _nested(sot, "stages", "S08_STATE_CONSTRAINED_FLEET_INFERENCE_DEPENDENCY_AUDIT") or {}
    if st49.get("status") != "STOP" or st49.get("reason") != "S49_FLEET_INFERENCE_CONTROLLER_DECISION_REQUIRED":
        raise S51Stop("S51_STOP_STAGE49")
    audit = _nested(sot, "scientific_freeze", "s08_rq3_operationalization", "revisions", REVISION, "fleet_inference_dependency_audit") or {}
    if audit.get("resolved_by_stage") != "50":
        raise S51Stop("S51_STOP_STAGE49_NOT_RESOLVED")
    if audit.get("controller_resolution_status") != "FROZEN":
        raise S51Stop("S51_STOP_STAGE49_RESOLUTION")
    if audit.get("controller_selected_method") != METHOD:
        raise S51Stop("S51_STOP_STAGE49_METHOD")
    if audit.get("next_action") != "RUN_STAGE_51":
        raise S51Stop("S51_STOP_STAGE49_NEXT")
    st50 = _nested(sot, "stages", "S08_STATE_CONSTRAINED_FLEET_INFERENCE_CONTROLLER_FREEZE") or {}
    if st50.get("status") != "GO" or st50.get("reason") != "S50_FLEET_FIXED_SET_BONFERRONI_PROJECTION_FROZEN":
        raise S51Stop("S51_STOP_STAGE50")
    if str(st50.get("next_stage")) != "51":
        raise S51Stop("S51_STOP_STAGE50_NEXT")
    freeze = _nested(sot, "scientific_freeze", "s08_rq3_operationalization", "revisions", REVISION, "fleet_inference_controller_freeze") or {}
    required = {
        "status": "FROZEN",
        "method_id": METHOD,
        "scope": SCOPE,
        "post_result_methodological_amendment": True,
        "fleet_interval_computed_before_freeze": False,
        "alpha_family": 0.05,
        "m_source": "STAGE49_REVISED_STRUCTURAL_TARGET_SET",
        "m": 10,
        "quantile_method": "numpy_percentile_linear",
        "bootstrap_draw_source": "STAGE45_RETAINED_P1_TARGET_DRAWS",
        "B_boot_retained": 5000,
        "new_bootstrap_draws_authorized": False,
        "target_resampling_authorized": False,
        "component_resampling_authorized": False,
        "joint_component_mbb_authorized": False,
        "global_support_intersection_authorized": False,
        "model_refit_authorized": False,
        "point_estimand": "F1_TARGET_BALANCED_ARITHMETIC_MEAN",
        "requires_target_independence": False,
        "requires_component_exchangeability": False,
        "requires_sign_symmetry": False,
        "requires_label_exchangeability": False,
        "superpopulation_inference_authorized": False,
        "fleet_hypothesis_test": False,
        "primary_target_ci_replaced": False,
        "selected_method": METHOD,
        "next_action": "RUN_STAGE_51",
    }
    for key, expected in required.items():
        if freeze.get(key) != expected:
            raise S51Stop("S51_STOP_FREEZE_FIELD", f"{key}={freeze.get(key)!r}")
    if freeze.get("fleet_p_value") is not None:
        raise S51Stop("S51_STOP_FREEZE_PVALUE")
    arts50 = st50.get("artifacts") or {}
    decision_path = _hash_rec(root, arts50.get("decision") or freeze.get("decision") or {}, "artifacts/s08_state_constrained_fleet_inference_freeze/S08_STATE_FLEET_INFERENCE_CONTROLLER_DECISION.json", "DECISION")
    recon_path = _hash_rec(root, arts50.get("reconciliation") or freeze.get("reconciliation") or {}, "artifacts/s08_state_constrained_fleet_inference_freeze/S08_STATE_FLEET_INFERENCE_FREEZE_RECONCILIATION.json", "RECON")
    doc_path = _hash_rec(root, arts50.get("decision_document") or freeze.get("decision_document") or {}, "docs/method_amendments/RQ3_FLEET_FIXED_SET_BONFERRONI_PROJECTION_CONTROLLER_DECISION.md", "DECISION_MD")
    report50_path = _hash_rec(root, arts50.get("report") or {"path": "reports/freeze/S08_STATE_CONSTRAINED_FLEET_INFERENCE_CONTROLLER_FREEZE.md"}, "reports/freeze/S08_STATE_CONSTRAINED_FLEET_INFERENCE_CONTROLLER_FREEZE.md", "REPORT50")
    boot_rec = (sci.get("artifacts") or {}).get("bootstrap_draws") or {}
    boot_path = _hash_rec(root, boot_rec, "artifacts/s08_state_constrained_scientific/S08_STATE_TARGET_LEVEL_BOOTSTRAP_DRAWS.parquet", "BOOT")
    if boot_rec.get("schema_fingerprint") and _pq_fp(boot_path) != boot_rec["schema_fingerprint"]:
        raise S51Stop("S51_STOP_BOOT_SCHEMA")
    if boot_rec.get("row_count") is not None and int(boot_rec["row_count"]) != 50000:
        raise S51Stop("S51_STOP_BOOT_ROWCOUNT", str(boot_rec.get("row_count")))
    d_fleet = sci.get("D_fleet_RQ3")
    if d_fleet is None:
        raise S51Stop("S51_STOP_MISSING_D_FLEET")
    inf_rec = (sci.get("artifacts") or {}).get("inference") or {}
    inf_path = root / (inf_rec.get("path") or "artifacts/s08_state_constrained_scientific/S08_STATE_TARGET_LEVEL_INFERENCE.csv")
    tgt_rec = (sci.get("artifacts") or {}).get("target_contrasts") or {}
    tgt_path = root / (tgt_rec.get("path") or "artifacts/s08_state_constrained_scientific/S08_STATE_TARGET_LEVEL_MATCHED_CONTRASTS.csv")
    return {
        "sci": sci,
        "freeze": freeze,
        "decision_path": decision_path,
        "recon_path": recon_path,
        "doc_path": doc_path,
        "report50_path": report50_path,
        "boot_path": boot_path,
        "boot_rec": boot_rec,
        "d_fleet": float(d_fleet),
        "inf_path": inf_path,
        "inf_rec": inf_rec,
        "tgt_path": tgt_path,
        "tgt_rec": tgt_rec,
    }


def execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    try:
        return _execute(ctx, now, validations)
    except S51Stop as exc:
        validations.append(ValidationRecord(exc.reason, False, exc.details))
        return StageResult(
            status="STOP",
            message=MSG_STOP,
            sot_patch={
                "stages": {
                    JOB: {
                        "kind": "scientific_analysis_revision",
                        "status": "STOP",
                        "reason": exc.reason,
                        "operational_stage_id": "51",
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
    decision = json.loads(bundle["decision_path"].read_text(encoding="utf-8"))
    j_fleet = [str(x) for x in (decision.get("J_fleet") or [])]
    m = len(j_fleet)
    if m != int(bundle["freeze"]["m"]) or m != 10:
        raise S51Stop("S51_STOP_M", str(m))
    if sorted(j_fleet) != j_fleet:
        j_fleet = sorted(j_fleet)
    boot_hash_before = sha256_file(bundle["boot_path"])
    inf_hash_before = sha256_file(bundle["inf_path"]) if bundle["inf_path"].is_file() else None
    table = pq.read_table(bundle["boot_path"])
    if table.num_rows != m * B_BOOT:
        raise S51Stop("S51_STOP_BOOT_ROWS", str(table.num_rows))
    by_tgt: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"draw_id": [], "Delta": [], "empty": [], "L": [], "den": []})
    seen_keys: set[tuple[str, int]] = set()
    for i in range(table.num_rows):
        tgt = str(table["target_plant_id"][i].as_py())
        did = int(table["draw_id"][i].as_py())
        key = (tgt, did)
        if key in seen_keys:
            raise S51Stop("S51_STOP_DUPLICATE_DRAW", f"{tgt}:{did}")
        seen_keys.add(key)
        empty = bool(table["empty_pairwise_support"][i].as_py())
        if empty:
            raise S51Stop("S51_STOP_EMPTY_SUPPORT_RETAINED", f"{tgt}:{did}")
        by_tgt[tgt]["draw_id"].append(did)
        by_tgt[tgt]["Delta"].append(float(table["Delta_j_draw"][i].as_py()))
        by_tgt[tgt]["empty"].append(empty)
        by_tgt[tgt]["L"].append(int(table["L_j"][i].as_py()))
        by_tgt[tgt]["den"].append(int(table["source_count_denominator"][i].as_py()))
    observed = set(by_tgt)
    expected = set(j_fleet)
    if observed != expected:
        raise S51Stop("S51_STOP_TARGET_SET", f"extra={sorted(observed - expected)} missing={sorted(expected - observed)}")
    expected_ids = set(range(B_BOOT))
    for tgt in j_fleet:
        ids = set(by_tgt[tgt]["draw_id"])
        if ids != expected_ids:
            raise S51Stop("S51_STOP_DRAW_IDS", tgt)
        if len(by_tgt[tgt]["Delta"]) != B_BOOT:
            raise S51Stop("S51_STOP_DRAW_COUNT", tgt)
    validations.append(ValidationRecord("draw_reconciliation", True, f"m={m} rows={m * B_BOOT}"))

    if bundle["tgt_path"].is_file():
        with bundle["tgt_path"].open(encoding="utf-8", newline="") as handle:
            points = {r["target_plant_id"]: float(r["Delta_j"]) for r in csv.DictReader(handle) if r["target_plant_id"] in expected}
        if set(points) != expected:
            raise S51Stop("S51_STOP_POINT_TARGET_SET")
        recon_mean = float(np.mean([points[t] for t in j_fleet]))
        if abs(recon_mean - bundle["d_fleet"]) > 1e-12:
            raise S51Stop("S51_STOP_POINT_RECON", f"{recon_mean} vs {bundle['d_fleet']}")
        validations.append(ValidationRecord("point_recon", True, None))

    mult = derive_multiplicity(m, ALPHA_FAMILY)
    if mult["q_low"] != 0.0025 or mult["q_high"] != 0.9975:
        raise S51Stop("S51_STOP_Q")
    rows = []
    lowers = []
    uppers = []
    n = float(B_BOOT)
    for tgt in j_fleet:
        arr = np.asarray(by_tgt[tgt]["Delta"], dtype=float)
        lo, hi = c8_target_interval(arr, m, ALPHA_FAMILY)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            raise S51Stop("S51_STOP_INTERVAL_ORDER", tgt)
        lowers.append(lo)
        uppers.append(hi)
        rows.append(
            {
                "target_plant_id": tgt,
                "draw_count": B_BOOT,
                "alpha_family": ALPHA_FAMILY,
                "m": m,
                "alpha_target": mult["alpha_target"],
                "q_low_probability": mult["q_low"],
                "q_high_probability": mult["q_high"],
                "q_low_percentile": 100 * mult["q_low"],
                "q_high_percentile": 100 * mult["q_high"],
                "quantile_method": "numpy_percentile_linear",
                "bonferroni_ci_lower": lo,
                "bonferroni_ci_upper": hi,
                "bonferroni_interval_level": 1.0 - float(mult["alpha_target"]),
                "quantile_index_low": (n - 1.0) * float(mult["q_low"]),
                "quantile_index_high": (n - 1.0) * float(mult["q_high"]),
                "min_retained_draw": float(arr.min()),
                "max_retained_draw": float(arr.max()),
                "L_j": int(by_tgt[tgt]["L"][0]),
                "source_count_denominator": int(by_tgt[tgt]["den"][0]),
            }
        )
    fleet_lo = float(np.mean(lowers))
    fleet_hi = float(np.mean(uppers))
    if abs(fleet_lo - float(np.mean([r["bonferroni_ci_lower"] for r in rows]))) > 1e-15:
        raise S51Stop("S51_STOP_F1_LOWER")
    width = fleet_hi - fleet_lo
    includes_zero = bool(fleet_lo <= 0.0 <= fleet_hi)
    d_fleet = bundle["d_fleet"]

    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "S08_STATE_FLEET_C8_TARGET_INTERVALS.csv", rows, CSV_FIELDS)
    fleet_json = {
        "revision_id": REVISION,
        "method_id": METHOD,
        "scope": SCOPE,
        "post_result_methodological_amendment": True,
        "point_estimand": "F1_TARGET_BALANCED_ARITHMETIC_MEAN",
        "D_fleet_RQ3": d_fleet,
        "m": m,
        "J_fleet": j_fleet,
        "alpha_family": ALPHA_FAMILY,
        "alpha_target": mult["alpha_target"],
        "q_low": mult["q_low"],
        "q_high": mult["q_high"],
        "quantile_method": "numpy_percentile_linear",
        "B_boot_retained_per_target": B_BOOT,
        "fleet_ci_nominal_level": 0.95,
        "fleet_ci_lower": fleet_lo,
        "fleet_ci_upper": fleet_hi,
        "fleet_ci_width": width,
        "fleet_ci_includes_zero": includes_zero,
        "fleet_p_value": None,
        "fleet_hypothesis_test": False,
        "superpopulation_inference_authorized": False,
        "requires_target_independence": False,
        "new_bootstrap_draws_generated": False,
        "model_refit": False,
    }
    protocol = {
        "method_id": METHOD,
        "scope": SCOPE,
        "post_result_methodological_amendment": True,
        "alpha_family": ALPHA_FAMILY,
        "m": m,
        "J_fleet": j_fleet,
        "percentile_rule": 'np.percentile(target_draws, [100 * alpha_family / (2 * m), 100 * (1 - alpha_family / (2 * m))], method="linear")',
        "projection_lower_formula": "mean(target_bonferroni_lower)",
        "projection_upper_formula": "mean(target_bonferroni_upper)",
        "bootstrap_draw_source": "STAGE45_RETAINED_P1_TARGET_DRAWS",
        "B_boot_retained": B_BOOT,
        "primary_target_ci_replaced": False,
        "fleet_p_value": None,
        "stage50_decision_sha256": sha256_file(bundle["decision_path"]),
    }
    dump_json(out / "S08_STATE_FLEET_C8_FIXED_SET_INTERVAL.json", fleet_json)
    dump_json(out / "S08_STATE_FLEET_C8_PROTOCOL.json", protocol)
    recs = {
        "target_intervals": _art(root, out / "S08_STATE_FLEET_C8_TARGET_INTERVALS.csv", rows=len(rows)),
        "fleet_interval": _art(root, out / "S08_STATE_FLEET_C8_FIXED_SET_INTERVAL.json"),
        "protocol": _art(root, out / "S08_STATE_FLEET_C8_PROTOCOL.json"),
    }
    recon = {
        "bootstrap_path": repo_relative(root, bundle["boot_path"]),
        "bootstrap_sha256": boot_hash_before,
        "bootstrap_schema_fingerprint": bundle["boot_rec"].get("schema_fingerprint") or _pq_fp(bundle["boot_path"]),
        "bootstrap_row_count": table.num_rows,
        "stage50_decision_path": repo_relative(root, bundle["decision_path"]),
        "stage50_decision_sha256": sha256_file(bundle["decision_path"]),
        "target_set_equal": True,
        "per_target_retained_counts": {t: B_BOOT for t in j_fleet},
        "total_retained_count": table.num_rows,
        "duplicate_key_count": 0,
        "empty_support_retained_draw_count": 0,
        "primary_target_cis_unchanged": True,
        "inference_sha256_before": inf_hash_before,
        "point_estimate_reconciliation_ok": True,
        "D_fleet_RQ3_from_sot": d_fleet,
        "no_new_bootstrap_draws": True,
        "no_model_refit": True,
        "no_p_value": True,
        "no_superpopulation_inference": True,
        "stage43_44_45_unchanged": True,
        "rq1_rq2_unchanged": True,
        "s09_s10_paper_unchanged": True,
    }
    dump_json(out / "S08_STATE_FLEET_C8_RECONCILIATION.json", recon)
    recs["reconciliation"] = _art(root, out / "S08_STATE_FLEET_C8_RECONCILIATION.json")
    zero_text = "the fixed-set interval includes zero." if includes_zero else "the fixed-set interval excludes zero."
    tgt_lines = "\n".join(
        f"- {r['target_plant_id']}: [{r['bonferroni_ci_lower']}, {r['bonferroni_ci_upper']}] (simultaneous-component only)"
        for r in rows
    )
    report = f"""# S08 state-constrained C8 fleet fixed-set inference

## 1. Method and chronology

This is a **post-result methodological amendment**. Stage 45 point estimates and primary 95% target MBB intervals already existed. Stage 49 documented the revised dependency graph. Stage 50 froze `{METHOD}` before any C8 endpoints existed. Stage 51 applies that frozen rule without adaptation. C8 was not pre-specified before Stage 45.

## 2. Frozen C8 rule

α_family={ALPHA_FAMILY}; m={m}; α_target={mult["alpha_target"]}; q_low={mult["q_low"]}; q_high={mult["q_high"]}.
Percentiles via `np.percentile(..., method="linear")` on the retained P1 draws. Fleet endpoints are the equal-target means of the Bonferroni-adjusted target endpoints.

## 3. Target set and draw reconciliation

J_fleet = {json.dumps(j_fleet)}. Each target has exactly {B_BOOT} retained draws. Total rows {m * B_BOOT}. Target set equals frozen J_fleet. No empty-support retained draws. No duplicate (target, draw_id). Bootstrap hash {boot_hash_before}.

## 4. Bonferroni-adjusted target intervals

These are `bonferroni_simultaneous_component_interval` values. They do **not** replace Stage 45 primary 95% intervals.

{tgt_lines}

## 5. Fleet point estimate and fixed-set CI

Existing SoT `D_fleet_RQ3` = {d_fleet} (unchanged).
C8 interval: [{fleet_lo}, {fleet_hi}]; width = {width}.

## 6. Zero inclusion

{zero_text}

`fleet_p_value` remains null. This is not a hypothesis test.

## 7. Fixed-set interpretation

A nominal 95% Bonferroni-projected fixed-set confidence interval for the equal-target mean RQ3 contrast among the structurally supported revised targets. The Bonferroni simultaneous-coverage step allows arbitrary dependence across target confidence events, while the interval inherits the validity/approximation assumptions of the marginal temporal MBB procedure. The result is conditional on the observed structurally supported target set. It does not support a population-wide statement about all PV plants.

## 8. Limitations

- no superpopulation inference
- no p-value
- dependence handled by simultaneous Bonferroni step, not modeled
- marginal MBB approximation inherited
- post-result methodological amendment

## 9. Protected-analysis reconciliation

Stage 43/44/45 artifacts, RQ1/RQ2, historical S08, S09, S10, and paper assets were not modified. No new draws or model refits.

## 10. Next action

`CONTROLLER_REVIEW_FOR_DOWNSTREAM_RECONCILIATION`. No later stage was executed.
"""
    (root / REPORT).write_text(report, encoding="utf-8")
    recs["report"] = _art(root, root / REPORT)
    dump_json(
        out / "S08_STATE_FLEET_C8_MANIFEST.json",
        {"job": JOB_PROMPT, "run_id": ctx.run_id, "finished_at_utc": now, "selected_method": METHOD, "fleet_p_value": None, "artifacts": {k: _tab(v) for k, v in recs.items()}},
    )
    recs["manifest"] = _art(root, out / "S08_STATE_FLEET_C8_MANIFEST.json")

    after = snapshot_tree(root, PROTECTED)
    if before != after:
        raise S51Stop("S51_STOP_PROTECTED_MUTATION")
    if sha256_file(bundle["boot_path"]) != boot_hash_before:
        raise S51Stop("S51_STOP_BOOT_MUTATION")
    if inf_hash_before and sha256_file(bundle["inf_path"]) != inf_hash_before:
        raise S51Stop("S51_STOP_INFERENCE_MUTATION")
    validations.append(ValidationRecord("c8_materialized", True, None))
    patch_arts = {k: _tab(v) for k, v in recs.items()}
    result_node = {
        "status": "GO",
        "reason": REASON,
        "revision_id": REVISION,
        "method_id": METHOD,
        "scope": SCOPE,
        "post_result_methodological_amendment": True,
        "point_estimand": "F1_TARGET_BALANCED_ARITHMETIC_MEAN",
        "D_fleet_RQ3": d_fleet,
        "m": m,
        "J_fleet": j_fleet,
        "alpha_family": ALPHA_FAMILY,
        "alpha_target": mult["alpha_target"],
        "q_low": mult["q_low"],
        "q_high": mult["q_high"],
        "quantile_method": "numpy_percentile_linear",
        "B_boot_retained_per_target": B_BOOT,
        "fleet_ci_nominal_level": 0.95,
        "fleet_ci_lower": fleet_lo,
        "fleet_ci_upper": fleet_hi,
        "fleet_ci_width": width,
        "fleet_ci_includes_zero": includes_zero,
        "fleet_p_value": None,
        "fleet_hypothesis_test": False,
        "superpopulation_inference_authorized": False,
        "requires_target_independence": False,
        "new_bootstrap_draws_generated": False,
        "model_refit": False,
        "target_intervals": _tab(recs["target_intervals"]),
        "fleet_interval": _tab(recs["fleet_interval"]),
        "protocol": _tab(recs["protocol"]),
        "reconciliation": _tab(recs["reconciliation"]),
        "manifest": _tab(recs["manifest"]),
        "report": REPORT,
        "next_action": "CONTROLLER_REVIEW_FOR_DOWNSTREAM_RECONCILIATION",
    }
    sot_patch = {
        "stages": {
            JOB: {
                "kind": "scientific_analysis_revision",
                "status": "GO",
                "reason": REASON,
                "operational_stage_id": "51",
                "job": JOB_PROMPT,
                "run_id": ctx.run_id,
                "finished_at_utc": now,
                "answer": ANSWER,
                "report": REPORT,
                "revision_id": REVISION,
                "method_id": METHOD,
                "scope": SCOPE,
                "next_action": "CONTROLLER_REVIEW_FOR_DOWNSTREAM_RECONCILIATION",
                "artifacts": patch_arts,
            }
        },
        "scientific_analysis": {
            "s08_revisions": {
                REVISION: {
                    "fleet_fixed_set_inference": result_node,
                    "scientific": {
                        "fleet_level_inference": {
                            "method": METHOD,
                            "scope": SCOPE,
                            "ci_level_nominal": 0.95,
                            "ci_lower": fleet_lo,
                            "ci_upper": fleet_hi,
                            "ci_includes_zero": includes_zero,
                            "p_value": None,
                            "hypothesis_test": False,
                            "descriptive_only": False,
                            "fixed_set_conditional_inference": True,
                            "superpopulation_inference_authorized": False,
                            "post_result_methodological_amendment": True,
                            "result_node": "scientific_analysis.s08_revisions.state_constrained_support_v2.fleet_fixed_set_inference",
                        }
                    },
                }
            }
        },
    }
    return StageResult(status="GO", message=MSG_GO, sot_patch=sot_patch, artifacts=list(recs.values()), validations=validations)
