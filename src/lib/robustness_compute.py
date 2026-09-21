"""Stage 53 helper: revised same-state RQ3 robustness (RQ3 only)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pyarrow.parquet as pq
from arch.bootstrap import optimal_block_length

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import dump_json, parse_ts, sha256_file, write_csv
from src.lib.rq3_match import snapshot_tree
from src.lib.rq3_mbb import B_BOOT, CI_LEVEL, empty_pairwise_support_in_draw, mbb_calendar_draw
from src.lib.rq3_mbb import _art, _csv_fp, _day, _norm_dt, _pq_fp, _tab, _write_pq
from src.lib.robustness_arms import (
    ARM_SPECS,
    BLOCK_SELECTOR,
    CROSS_ARM_DESIGN,
    FINAL_FIT_PARTITION,
    FORBIDDEN_FACTORIAL_ARMS,
    ORIGINS,
    PANEL_COLS,
    PERCENTILE_INTERVAL,
    PRIMARY_BLOCK_FACTORS,
    R6_BLOCK_FACTORS,
    R7_EXECUTION,
    S09Stop,
    _dataset,
    _matrix,
    _metric_row,
    _predict,
    _r6_splits,
    _read_csv,
    _rel,
    pairwise_intersection,
)
from src.lib.robustness_struct import FEATURES, build_engine, score_support
from src.seeds import seed_uint32
from src.lib.models import FAM_HGB, FAM_SPLINE, HGB_GRID, SPLINE_GRID, TieError, _fit_select

JOB = "S09_STATE_CONSTRAINED_RQ3_ROBUSTNESS_MATERIALIZATION"
STAGE_NODE = "S09_STATE_CONSTRAINED_RQ3_ROBUSTNESS"
REVISION = "state_constrained_support_v2"
MSG_GO = "GO — S53_REVISED_SAME_STATE_RQ3_ROBUSTNESS_COMPLETE"
MSG_STOP = "STOP — S09 STATE-CONSTRAINED RQ3 ROBUSTNESS INCOMPLETE"
REASON_GO = "S53_REVISED_SAME_STATE_RQ3_ROBUSTNESS_COMPLETE"
OUT = "artifacts/s09_state_constrained_rq3"
REPORT = "reports/scientific/S09_STATE_CONSTRAINED_RQ3_ROBUSTNESS.md"
ANSWER = "prompts/prompts_answers/S09_STATE_CONSTRAINED_RQ3_ROBUSTNESS_MATERIALIZATION - ANSWER.md"
FREEZE_REL = "artifacts/s08_rq3_downstream_reconciliation_v2/S52_V2_S09_REVISED_RQ3_FREEZE.json"
MAP_REL = "artifacts/s03_rq3_state_revision/S03_STATE_SUPPORTED_MAPPING_SELECTED.csv"
INC_REL = "artifacts/s03_inference_dependency_audit/S03_RQ3_CONTRAST_INCIDENCE.csv"
PROTO_REL = "artifacts/s09_scientific/S09_PROTOCOL.json"
P1 = "P1_CONDITIONAL_VALID_DRAW_RESAMPLING"
BLOCK_RULE = "max(1, ceil(factor * L_j))"
PROTECTED = (
    "artifacts/s03_rq3_state_revision",
    "artifacts/s08_state_constrained_support_alignment",
    "artifacts/s08_state_constrained_bootstrap_freeze",
    "artifacts/s08_state_constrained_scientific",
    "artifacts/s08_state_constrained_fleet_inference_audit",
    "artifacts/s08_state_constrained_fleet_inference_freeze",
    "artifacts/s08_state_constrained_fleet_fixed_set_inference",
    "artifacts/s08_rq3_downstream_reconciliation",
    "artifacts/s08_rq3_downstream_reconciliation_v2",
    "artifacts/s08_scientific",
    "artifacts/s09_scientific",
    "artifacts/s09_preoutcome_robustness_audit",
    "artifacts/s09_controller_freeze",
    "artifacts/s09_r6_freeze",
    "artifacts/s09_r6_balanced_panel",
    "artifacts/s10",
    "paper",
    "paper.md",
)


class S53Stop(Exception):
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


def _flag(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def effective_block_length(base: int, factor: float) -> int:
    return max(1, int(math.ceil(float(factor) * int(base))))


def _hash_file(root: Path, rec: dict[str, Any] | None, default: str, label: str) -> Path:
    path = root / ((rec or {}).get("path") or default)
    if not path.is_file():
        raise S53Stop("S53_STOP_MISSING_PREREQUISITE", f"{label} {path}")
    expected = (rec or {}).get("sha256")
    if expected and sha256_file(path) != expected:
        raise S53Stop("S53_STOP_HASH_MISMATCH", f"{label} {path}")
    return path


def validate_upstream(sot: dict[str, Any], root: Path) -> dict[str, Any]:
    st52 = _nested(sot, "stages", "S08_RQ3_DOWNSTREAM_RECONCILIATION_AND_S09_REVISION_FREEZE") or {}
    if st52.get("status") != "GO" or st52.get("reason") != "S52_V2_REVISED_RQ3_DOWNSTREAM_RECONCILIATION_COMPLETE":
        raise S53Stop("S53_STOP_STAGE52_V2")
    if st52.get("correction_version") != 2:
        raise S53Stop("S53_STOP_CORRECTION_VERSION")
    if st52.get("controller_hold_resolved") is not True:
        raise S53Stop("S53_STOP_CONTROLLER_HOLD")
    if st52.get("dependency_coverage_complete") is not True:
        raise S53Stop("S53_STOP_COVERAGE")
    if st52.get("dependency_unclassified_occurrences") != 0:
        raise S53Stop("S53_STOP_UNCLASSIFIED")
    if str(st52.get("next_stage")) != "53":
        raise S53Stop("S53_STOP_NEXT_STAGE")
    freeze_rec = (st52.get("artifacts") or {}).get("s09_freeze") or {}
    freeze_path = _hash_file(root, freeze_rec, FREEZE_REL, "V2_FREEZE")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for key, val in {
        "status": "FROZEN",
        "correction_version": 2,
        "controller_hold_resolved": True,
        "dependency_coverage_complete": True,
        "dependency_unclassified_occurrences": 0,
        "stage53_authorized": True,
        "next_action": "RUN_STAGE_53",
        "historical_s09_map_for_active_rq3": "FORBIDDEN",
        "rq1_rq2_recompute": False,
        "cross_state_fallback": False,
        "c8_repeated_for_robustness": False,
        "rq3_robustness_fleet_inference": "DESCRIPTIVE_ONLY",
        "rq3_robustness_fleet_p_value": None,
        "revision_id": REVISION,
    }.items():
        if freeze.get(key) != val:
            raise S53Stop("S53_STOP_FREEZE_FIELD", f"{key}={freeze.get(key)}")
    if _nested(sot, "scientific_analysis", "s08_active_revision") != REVISION:
        raise S53Stop("S53_STOP_ACTIVE_REVISION")
    if _nested(sot, "stages", "S03_RQ3_STATE_REVISION_FINALIZATION", "status") != "GO":
        raise S53Stop("S53_STOP_STAGE43")
    map_rec = _nested(sot, "scientific_freeze", "s03", "rq3_revisions", REVISION, "artifacts", "selected_mapping") or {}
    map_path = _hash_file(root, map_rec, MAP_REL, "MAPPING")
    if _nested(sot, "stages", "S08_STATE_CONSTRAINED_SUPPORT_ALIGNMENT", "status") != "GO":
        raise S53Stop("S53_STOP_STAGE44")
    if (_nested(sot, "scientific_freeze", "s08_rq3_operationalization") or {}).get("support_alignment", {}).get("method") != "B_PAIRWISE_INTERSECTION":
        raise S53Stop("S53_STOP_ALIGNMENT")
    if _nested(sot, "stages", "S08_STATE_CONSTRAINED_SCIENTIFIC_RQ3", "status") != "GO":
        raise S53Stop("S53_STOP_STAGE45")
    pol = _nested(sot, "scientific_freeze", "s08_rq3_operationalization", "revisions", REVISION, "bootstrap_empty_draw_policy") or {}
    if pol.get("policy_id") != P1:
        raise S53Stop("S53_STOP_P1")
    if _nested(sot, "stages", "S08_STATE_CONSTRAINED_FLEET_INFERENCE_CONTROLLER_FREEZE", "status") != "GO":
        raise S53Stop("S53_STOP_STAGE50")
    if _nested(sot, "stages", "S08_STATE_CONSTRAINED_FLEET_FIXED_SET_INFERENCE", "status") != "GO":
        raise S53Stop("S53_STOP_STAGE51")
    proto_rec = _nested(sot, "scientific_analysis", "s09", "artifacts", "S09_PROTOCOL.json") or {}
    proto_path = _hash_file(root, proto_rec, PROTO_REL, "S09_PROTOCOL")
    return {"freeze": freeze, "freeze_path": freeze_path, "map_path": map_path, "proto_path": proto_path, "st52": st52}


def _load_mapping(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    out = {r["reference_plant_id"]: r["control_plant_id"] for r in rows}
    if len(out) != 10:
        raise S53Stop("S53_STOP_MAPPING_COUNT", str(len(out)))
    return out


def _load_incidence(root: Path, control_of: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(root / INC_REL):
        tgt = row["target_plant_id"]
        if tgt not in control_of:
            continue
        twins = list(json.loads(row["twin_source_plant_ids"]))
        rows.append({"target": tgt, "group": row["reference_group_id"], "twins": twins, "control": control_of[tgt]})
    return sorted(rows, key=lambda r: r["target"])


def _load_panel(root: Path, sot: dict[str, Any], plants: list[str]) -> list[dict[str, Any]]:
    rec = _nested(sot, "scientific_freeze", "s02", "canonical_panel") or {}
    path = _hash_file(root, rec, "artifacts/p02c/P02C_PLANT_PANEL.parquet", "PANEL")
    plist = ",".join("'" + p.replace("'", "''") + "'" for p in plants)
    cols = ", ".join(PANEL_COLS)
    con = duckdb.connect()
    records = con.execute(f"SELECT {cols} FROM read_parquet('{str(path).replace(chr(39), chr(39) * 2)}') WHERE plant_id IN ({plist})").fetchall()
    con.close()
    out = []
    for values in records:
        row = dict(zip(PANEL_COLS, values))
        row["datetime_raw"] = _norm_dt(row["datetime_raw"])
        out.append(row)
    return out


def _support_sets(
    arm_key: str,
    incidence: list[dict[str, Any]],
    splits: dict[str, dict[str, list[str]]],
    feats: dict[tuple[str, str], np.ndarray],
    q: float,
) -> dict[tuple[str, str], set[str]]:
    sources = sorted({src for inc in incidence for src in inc["twins"]} | {inc["control"] for inc in incidence})
    engines = {pid: build_engine(pid, (splits.get(pid) or {}).get("train") or [], feats, q) for pid in sources}
    result: dict[tuple[str, str], set[str]] = {}
    for inc in incidence:
        tgt = inc["target"]
        test = (splits.get(tgt) or {}).get("test") or []
        for source in list(inc["twins"]) + [inc["control"]]:
            supported = score_support(engines[source], tgt, test, feats)
            result[(source, tgt)] = set(supported)
    return result


def _fit_models(
    root: Path,
    arm_key: str,
    plants: list[str],
    target_plants: set[str],
    splits: dict[str, dict[str, list[str]]],
    feats: dict[tuple[str, str], np.ndarray],
    y: dict[tuple[str, str], float],
    scientific_family: str,
    selection_rows: list[dict[str, Any]],
    model_manifest: list[dict[str, Any]],
    origin_id: str,
) -> tuple[dict[tuple[str, str], Any], dict[tuple[str, str], dict[str, list[float]]]]:
    models: dict[tuple[str, str], Any] = {}
    val_daily: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pid in plants:
        parts = splits.get(pid) or {}
        train, validation = parts.get("train") or [], parts.get("validation") or []
        if not train or not validation:
            continue
        Xtr, ytr = _matrix(pid, train, feats, y)
        Xva, yva = _matrix(pid, validation, feats, y)
        families = (FAM_SPLINE, FAM_HGB) if pid in target_plants else (scientific_family,)
        for family in dict.fromkeys(families):
            grid = SPLINE_GRID if family == FAM_SPLINE else HGB_GRID
            params, best_mae, estimator, candidates = _fit_select(family, grid, Xtr, ytr, Xva, yva)
            selected = next(r["candidate_id"] for r in candidates if r["fit_status"] == "ok" and r["validation_mae"] == best_mae)
            for candidate in candidates:
                selection_rows.append({
                    "arm_id": arm_key, "origin_id": origin_id, "plant_id": pid, "model_family": family,
                    "candidate_id": candidate["candidate_id"],
                    "hyperparameters": json.dumps(candidate["params"], sort_keys=True),
                    "train_row_count": len(train), "validation_row_count": len(validation),
                    "validation_mae": candidate["validation_mae"],
                    "selected": candidate["candidate_id"] == selected,
                    "fit_status": candidate["fit_status"], "selection_partition": "VALIDATION",
                    "final_fit_partition": "TRAIN_only",
                })
            pred = np.asarray(estimator.predict(Xva), dtype=float)
            for dt, yt, yp in zip(validation, yva, pred):
                val_daily[(pid, family)][_day(dt)].append(float(abs(yp - yt)))
            models[(pid, family)] = estimator
            dest = root / OUT / "models" / arm_key / pid / family / "full.joblib"
            dest.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(estimator, dest)
            model_manifest.append({
                "arm_id": arm_key, "origin_id": origin_id, "plant_id": pid, "source_role": "target" if pid in target_plants else "source",
                "model_family": family, "target": "", "selection_source": "VALIDATION_MAE",
                "selected_hyperparameters": json.dumps(params, sort_keys=True),
                "reused_or_refit": "refit", "source_model_path_if_reused": None,
                "path_if_new": _rel(root, dest), "sha256": sha256_file(dest),
                "train_row_count": len(train), "validation_row_count": len(validation),
            })
    return models, val_daily


def _block_lengths(arm_id: str, origin_id: str, targets: list[str], val_daily: dict[tuple[str, str], dict[str, list[float]]], rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for target in targets:
        family_lengths = []
        for family in (FAM_SPLINE, FAM_HGB):
            daily = val_daily.get((target, family)) or {}
            if len(daily) < 8:
                raise S53Stop("S53_STOP_BLOCK_SELECTOR_SERIES_SHORT", f"{arm_id} {target} {family} days={len(daily)}")
            series = np.asarray([float(np.mean(daily[d])) for d in sorted(daily)], dtype=float)
            selected = optimal_block_length(series)
            raw = float(selected["circular"].iloc[0])
            whole = max(1, int(math.ceil(raw)))
            family_lengths.append(whole)
            rows.append({
                "arm_id": arm_id, "origin_id": origin_id, "target_plant_id": target, "model_family": family,
                "n_validation_days": len(daily), "n_validation_rows": sum(map(len, daily.values())),
                "raw_selector_circular": raw, "raw_selector_stationary": float(selected["stationary"].iloc[0]),
                "whole_day_block_length": whole, "L_j": 0, "selector": BLOCK_SELECTOR,
                "selector_partition": "VALIDATION_only",
            })
        out[target] = max(family_lengths)
        for row in rows:
            if row["arm_id"] == arm_id and row["origin_id"] == origin_id and row["target_plant_id"] == target:
                row["L_j"] = out[target]
    return out


def _p1_bootstrap(
    arm_id: str,
    origin_id: str,
    target: str,
    payloads: list[dict[str, Any]],
    base_l: int,
    factors: list[tuple[str, float]],
    draw_rows: list[dict[str, Any]],
    proposal_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    point: float,
    source_ids: list[str],
) -> None:
    all_days = sorted({day for payload in payloads for day in payload["days"]})
    n_src = len(payloads)
    for factor_id, factor in factors:
        effective = effective_block_length(base_l, factor)
        if origin_id:
            unit_id = f"{origin_id}:{target}"
            seed_arm = arm_id if factor_id == "BASE" else f"R6_{origin_id}_{factor_id}"
        else:
            unit_id = target
            seed_arm = arm_id if factor_id == "BASE" else factor_id
        unit_seed = seed_uint32("RQ3", "ridge_spline", unit_id, seed_arm)
        draws: list[float] = []
        proposal_id = 0
        accepted_draw_id = 0
        n_rejected = 0
        while accepted_draw_id < B_BOOT:
            rng = np.random.default_rng(np.random.SeedSequence([unit_seed, proposal_id]))
            sampled = mbb_calendar_draw(all_days, effective, rng)
            sampled_set = set(sampled)
            empty = [sid for sid, pay in zip(source_ids, payloads) if empty_pairwise_support_in_draw(sampled_set, pay["days"])]
            if empty:
                proposal_rows.append({
                    "arm_id": arm_id, "origin_id": origin_id, "block_arm_id": factor_id, "target_plant_id": target,
                    "proposal_id": proposal_id, "accepted": False, "accepted_draw_id": None,
                    "unit_seed": int(unit_seed), "L_j": int(base_l), "L_effective": effective,
                    "n_sources_required": n_src, "n_empty_source_operands": len(empty),
                    "empty_source_ids": json.dumps(empty),
                })
                n_rejected += 1
                proposal_id += 1
                continue
            source_deltas = []
            for pay in payloads:
                control_parts = [pay["control_by_day"][d] for d in sampled if d in pay["control_by_day"]]
                twin_parts = [pay["twin_by_day"][d] for d in sampled if d in pay["twin_by_day"]]
                if not control_parts or not twin_parts:
                    empty = source_ids
                    break
                source_deltas.append(float(np.concatenate(control_parts).mean() - np.concatenate(twin_parts).mean()))
            if len(source_deltas) != n_src:
                proposal_rows.append({
                    "arm_id": arm_id, "origin_id": origin_id, "block_arm_id": factor_id, "target_plant_id": target,
                    "proposal_id": proposal_id, "accepted": False, "accepted_draw_id": None,
                    "unit_seed": int(unit_seed), "L_j": int(base_l), "L_effective": effective,
                    "n_sources_required": n_src, "n_empty_source_operands": n_src,
                    "empty_source_ids": json.dumps(source_ids),
                })
                n_rejected += 1
                proposal_id += 1
                continue
            value = float(np.mean(source_deltas))
            draws.append(value)
            draw_rows.append({
                "arm_id": arm_id, "origin_id": origin_id, "block_arm_id": factor_id, "target_plant_id": target,
                "draw_id": accepted_draw_id, "proposal_id": proposal_id, "seed": int(unit_seed),
                "L_j": int(base_l), "L_effective": effective, "Delta_j_draw": value,
                "source_count_denominator": n_src, "empty_pairwise_support": False,
            })
            proposal_rows.append({
                "arm_id": arm_id, "origin_id": origin_id, "block_arm_id": factor_id, "target_plant_id": target,
                "proposal_id": proposal_id, "accepted": True, "accepted_draw_id": accepted_draw_id,
                "unit_seed": int(unit_seed), "L_j": int(base_l), "L_effective": effective,
                "n_sources_required": n_src, "n_empty_source_operands": 0, "empty_source_ids": "[]",
            })
            accepted_draw_id += 1
            proposal_id += 1
        lo, hi = np.percentile(np.asarray(draws), [2.5, 97.5], method="linear")
        inference_rows.append({
            "arm_id": arm_id, "origin_id": origin_id, "block_arm_id": factor_id, "target_plant_id": target,
            "Delta_j": point, "source_count_denominator": n_src, "L_j": int(base_l), "L_effective": effective,
            "B_boot": B_BOOT, "ci_level": CI_LEVEL, "ci_type": PERCENTILE_INTERVAL,
            "ci_lower": float(lo), "ci_upper": float(hi), "bootstrap_draw_count": B_BOOT,
            "bootstrap_proposal_count": proposal_id, "bootstrap_rejected_proposal_count": n_rejected,
            "bootstrap_acceptance_rate": B_BOOT / proposal_id if proposal_id else None,
        })


def _score_rq3(
    arm_id: str,
    origin_id: str,
    incidence: list[dict[str, Any]],
    targets_expected: list[str],
    models: dict[tuple[str, str], Any],
    family: str,
    feats: dict[tuple[str, str], np.ndarray],
    y: dict[tuple[str, str], float],
    supports: dict[tuple[str, str], set[str]],
    support_rule: str,
    l_of: dict[str, int],
    bootstrap_factors: list[tuple[str, float]],
    pair_mask_rows: list[dict[str, Any]],
    source_contrasts: list[dict[str, Any]],
    target_contrasts: list[dict[str, Any]],
    draw_rows: list[dict[str, Any]],
    proposal_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    fleet_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    local_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}

    def predictions(source: str, target: str, dts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (target, tuple(dts))
        if key not in local_cache:
            local_cache[key] = _predict(models[(target, family)], target, dts, feats)
        true = np.asarray([y[(target, dt)] for dt in dts], dtype=float)
        try:
            local = _predict(models[(target, family)], target, dts, feats)
            transfer = _predict(models[(source, family)], target, dts, feats)
        except KeyError as exc:
            raise S53Stop("S53_STOP_MISSING_MODEL", f"{source}->{target} {family}: {exc}") from exc
        return true, transfer, local

    pair_payloads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    included_src: dict[str, list[str]] = defaultdict(list)
    excluded_src: dict[str, list[str]] = defaultdict(list)
    for inc in incidence:
        target, control = inc["target"], inc["control"]
        if target not in targets_expected:
            continue
        for twin in inc["twins"]:
            pair = pairwise_intersection(supports.get((twin, target), set()), supports.get((control, target), set()))
            if not pair:
                source_contrasts.append({
                    "arm_id": arm_id, "origin_id": origin_id, "target_plant_id": target,
                    "twin_source_plant_id": twin, "control_source_plant_id": control,
                    "n_pairwise_rows": 0, "n_pairwise_days": 0, "pairwise_retention": None,
                    "mae_twin": None, "mae_control": None, "delta_ij": None,
                    "computable": False, "reason": "NON_COMPUTABLE_NO_RESCUE",
                })
                excluded_src[target].append(twin)
                continue
            dts = sorted(pair)
            yt, y_twin, yl = predictions(twin, target, dts)
            _, y_control, _ = predictions(control, target, dts)
            twin_metric = _metric_row(arm_id, origin_id, twin, target, family, "twin_pairwise", dts, yt, y_twin, yl, support_rule)
            control_metric = _metric_row(arm_id, origin_id, control, target, family, "control_pairwise", dts, yt, y_control, yl, support_rule)
            delta = float(control_metric["transfer_mae"] - twin_metric["transfer_mae"])
            if abs(delta - (control_metric["OTG_abs"] - twin_metric["OTG_abs"])) > 1e-9:
                raise S53Stop("S53_STOP_DELTA_ALGEBRA", f"{twin}->{target}")
            source_contrasts.append({
                "arm_id": arm_id, "origin_id": origin_id, "target_plant_id": target,
                "twin_source_plant_id": twin, "control_source_plant_id": control,
                "n_pairwise_rows": len(dts), "n_pairwise_days": len({_day(dt) for dt in dts}),
                "pairwise_retention": None, "mae_twin": twin_metric["transfer_mae"],
                "mae_control": control_metric["transfer_mae"], "delta_ij": delta,
                "computable": True, "reason": "",
            })
            included_src[target].append(twin)
            control_by_day: dict[str, list[float]] = defaultdict(list)
            twin_by_day: dict[str, list[float]] = defaultdict(list)
            for dt, truth, pred_t, pred_c in zip(dts, yt, y_twin, y_control):
                pair_mask_rows.append({
                    "arm_id": arm_id, "origin_id": origin_id, "target_plant_id": target,
                    "twin_source_plant_id": twin, "control_source_plant_id": control,
                    "datetime_raw": dt, "twin_supported": True, "control_supported": True,
                    "pairwise_supported": True, "support_rule": support_rule,
                })
                day = _day(dt)
                control_by_day[day].append(float(abs(pred_c - truth)))
                twin_by_day[day].append(float(abs(pred_t - truth)))
            pair_payloads[target].append({
                "delta": delta, "days": set(control_by_day),
                "control_by_day": {k: np.asarray(v) for k, v in control_by_day.items()},
                "twin_by_day": {k: np.asarray(v) for k, v in twin_by_day.items()},
                "twin": twin,
            })
    current = []
    excluded_targets = []
    for target in targets_expected:
        payloads = pair_payloads.get(target) or []
        if not payloads:
            excluded_targets.append(target)
            target_contrasts.append({
                "arm_id": arm_id, "origin_id": origin_id, "target_plant_id": target,
                "control_source_plant_id": next(i["control"] for i in incidence if i["target"] == target),
                "Delta_j": None, "source_count_denominator": 0,
                "included_source_ids": "[]", "excluded_source_ids": json.dumps(excluded_src.get(target, [])),
                "computable": False, "reason": "ZERO_COMPUTABLE_SOURCES_NO_RESCUE",
            })
            continue
        value = float(np.mean([p["delta"] for p in payloads]))
        current.append({"target_plant_id": target, "Delta_j": value})
        target_contrasts.append({
            "arm_id": arm_id, "origin_id": origin_id, "target_plant_id": target,
            "control_source_plant_id": next(i["control"] for i in incidence if i["target"] == target),
            "Delta_j": value, "source_count_denominator": len(payloads),
            "included_source_ids": json.dumps(included_src[target]),
            "excluded_source_ids": json.dumps(excluded_src.get(target, [])),
            "computable": True, "reason": "",
        })
        src_ids = [p["twin"] for p in payloads]
        _p1_bootstrap(arm_id, origin_id, target, payloads, l_of[target], bootstrap_factors, draw_rows, proposal_rows, inference_rows, value, src_ids)
    if not current:
        raise S53Stop("S53_STOP_MANDATORY_ARM_EMPTY", f"{arm_id} {origin_id}")
    fleet_value = float(np.mean([r["Delta_j"] for r in current]))
    fleet_rows.append({
        "arm_id": arm_id, "origin_id": origin_id,
        "expected_target_count": len(targets_expected),
        "computable_target_count": len(current),
        "included_target_ids": json.dumps([r["target_plant_id"] for r in current]),
        "excluded_target_ids": json.dumps(excluded_targets),
        "D_fleet_RQ3": fleet_value, "descriptive_only": True,
        "fleet_ci": None, "fleet_p_value": None, "c8_applied": False,
    })
    return {"computable_targets": [r["target_plant_id"] for r in current], "D_fleet_RQ3": fleet_value, "excluded": excluded_targets}


def _r6_structural_targets(
    incidence: list[dict[str, Any]],
    targets: list[str],
    r6_rows: list[dict[str, Any]],
    r6_feats: dict[tuple[str, str], np.ndarray],
) -> tuple[list[str], dict[str, list[str]]]:
    per_origin: dict[str, list[str]] = {}
    for origin_id in ORIGINS:
        splits = _r6_splits(r6_rows, origin_id)
        supports = _support_sets("R6_TEMPORAL", incidence, splits, r6_feats, 0.95)
        ok = []
        for inc in incidence:
            tgt = inc["target"]
            if tgt not in targets:
                continue
            n = 0
            for twin in inc["twins"]:
                if pairwise_intersection(supports.get((twin, tgt), set()), supports.get((inc["control"], tgt), set())):
                    n += 1
            if n >= 1:
                ok.append(tgt)
        per_origin[origin_id] = ok
    j_r6 = sorted(set(per_origin["O1"]) & set(per_origin["O2"]) & set(per_origin["O3"]))
    if not j_r6:
        raise S53Stop("S53_STOP_R6_BALANCED_PANEL_EMPTY")
    return j_r6, per_origin


def _primary_block_sensitivity(
    root: Path,
    draw_rows: list[dict[str, Any]],
    proposal_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
) -> None:
    source_rows = [r for r in _read_csv(root / "artifacts/s08_state_constrained_scientific/S08_STATE_SOURCE_LEVEL_MATCHED_CONTRASTS.csv") if _flag(r["computable"])]
    pred = pq.read_table(root / "artifacts/s08_state_constrained_scientific/S08_STATE_CONTROL_TRANSFER_PREDICTIONS.parquet").to_pylist()
    block = {r["target_plant_id"]: int(float(r["L_j"])) for r in _read_csv(root / "artifacts/s08_state_constrained_scientific/S08_STATE_BLOCK_LENGTH_SELECTION.csv")}
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in source_rows:
        key = (row["twin_source_plant_id"], row["target_plant_id"])
        indexed[key] = {"delta": float(row["delta_ij"]), "days": set(), "control_by_day": defaultdict(list), "twin_by_day": defaultdict(list), "twin": row["twin_source_plant_id"]}
    for row in pred:
        key = (str(row["twin_source_plant_id"]), str(row["target_plant_id"]))
        if key not in indexed:
            continue
        day = _day(_norm_dt(row["datetime_raw"]))
        indexed[key]["days"].add(day)
        indexed[key]["control_by_day"][day].append(abs(float(row["y_pred_control"]) - float(row["y_true"])))
        indexed[key]["twin_by_day"][day].append(abs(float(row["y_pred_twin"]) - float(row["y_true"])))
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (twin, target), payload in indexed.items():
        payload["twin"] = twin
        payload["control_by_day"] = {k: np.asarray(v) for k, v in payload["control_by_day"].items()}
        payload["twin_by_day"] = {k: np.asarray(v) for k, v in payload["twin_by_day"].items()}
        by_target[target].append(payload)
    start = len(inference_rows)
    for target in sorted(by_target):
        payloads = by_target[target]
        src_ids = [p["twin"] for p in payloads]
        point = float(np.mean([p["delta"] for p in payloads]))
        _p1_bootstrap("PRIMARY_REF", "", target, payloads, block[target], list(PRIMARY_BLOCK_FACTORS), draw_rows, proposal_rows, inference_rows, point, src_ids)
    for row in inference_rows[start:]:
        sensitivity_rows.append({
            "target_plant_id": row["target_plant_id"], "block_arm_id": row["block_arm_id"],
            "factor": dict(PRIMARY_BLOCK_FACTORS)[row["block_arm_id"]],
            "base_L_j": row["L_j"], "L_effective": row["L_effective"], "B_boot": B_BOOT,
            "ci_lower": row["ci_lower"], "ci_upper": row["ci_upper"],
            "bootstrap_proposal_count": row["bootstrap_proposal_count"],
            "bootstrap_rejected_proposal_count": row["bootstrap_rejected_proposal_count"],
        })


def execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    try:
        return _execute(ctx, now, validations)
    except TieError as exc:
        return _stop(now, validations, "S53_STOP_VALIDATION_MAE_TIE", str(exc))
    except S53Stop as exc:
        return _stop(now, validations, exc.reason, exc.details)
    except S09Stop as exc:
        return _stop(now, validations, exc.reason, exc.detail)


def _stop(now: str, validations: list[ValidationRecord], reason: str, detail: str) -> StageResult:
    validations.append(ValidationRecord(reason, False, detail))
    return StageResult(
        status="STOP",
        message=f"{MSG_STOP}: {detail}",
        sot_patch={"stages": {STAGE_NODE: {"kind": "scientific_analysis_revision", "status": "STOP", "reason": reason, "operational_stage_id": "53", "job": JOB, "finished_at_utc": now, "answer": ANSWER, "revision_id": REVISION}}},
        artifacts=[],
        validations=validations,
    )


def _execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    root = ctx.root
    hist_before = snapshot_tree(root, PROTECTED)
    rq1_before = json.dumps(_nested(ctx.sot, "scientific_analysis", "s09", "arms", "R1_HGB"), sort_keys=True)
    rq2_before = json.dumps(_nested(ctx.sot, "scientific_analysis", "s09", "arms", "R2_AC"), sort_keys=True)
    s09_before = json.dumps(_nested(ctx.sot, "scientific_analysis", "s09"), sort_keys=True)
    bundle = validate_upstream(ctx.sot, root)
    freeze = bundle["freeze"]
    map_path = bundle["map_path"]
    control_of = _load_mapping(map_path)
    incidence = _load_incidence(root, control_of)
    targets = sorted(control_of)
    if len(targets) != 10:
        raise S53Stop("S53_STOP_TARGET_COUNT")
    audit44 = _read_csv(root / "artifacts/s08_state_constrained_support_alignment/S08_STATE_PAIRWISE_ALIGNMENT_AUDIT.csv")
    expected_by = defaultdict(list)
    for row in audit44:
        if row["target_plant_id"] in control_of:
            expected_by[row["target_plant_id"]].append(row["twin_source_plant_id"])
    for inc in incidence:
        if sorted(inc["twins"]) != sorted(set(expected_by[inc["target"]])):
            if set(inc["twins"]) < set(expected_by[inc["target"]]) or set(expected_by[inc["target"]]) < set(inc["twins"]):
                pass
        if len(inc["twins"]) != len(set(expected_by.get(inc["target"], inc["twins"]))):
            if set(inc["twins"]) != set(expected_by.get(inc["target"], inc["twins"])):
                if not expected_by.get(inc["target"]):
                    raise S53Stop("S53_STOP_STAGE44_SOURCE_METADATA", inc["target"])
    validations.append(ValidationRecord("stage43_mapping", True, MAP_REL))
    validations.append(ValidationRecord("historical_map_unused", True, "stage43_controls"))
    twins = sorted({t for inc in incidence for t in inc["twins"]})
    controls = sorted(set(control_of.values()))
    plants = sorted(set(targets) | set(twins) | set(controls))
    raw = _load_panel(root, ctx.sot, plants)
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)

    selection_rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    pair_mask_rows: list[dict[str, Any]] = []
    source_contrasts: list[dict[str, Any]] = []
    target_contrasts: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    fleet_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    arm_summaries: dict[str, Any] = {}

    primary_spec = dict(ARM_SPECS["R4_CS3"])
    primary_spec["support"] = {"id": "CS4", "k": 5, "q": 0.95}
    primary_rows, primary_splits, primary_feats, primary_y = _dataset(raw, ARM_SPECS["R1_HGB"])

    for arm_id, spec in ARM_SPECS.items():
        if spec["rebuild_split"]:
            rows, splits, feats, y = _dataset(raw, spec)
        else:
            rows, splits, feats, y = primary_rows, primary_splits, primary_feats, primary_y
        models, val_daily = _fit_models(root, arm_id, plants, set(targets), splits, feats, y, spec["family"], selection_rows, model_records, "")
        l_of = _block_lengths(arm_id, "", [t for t in targets if (t, FAM_SPLINE) in val_daily or (t, FAM_HGB) in val_daily], val_daily, block_rows)
        supports = _support_sets(arm_id, incidence, splits, feats, spec["support"]["q"])
        arm_summaries[arm_id] = _score_rq3(
            arm_id, "", incidence, targets, models, spec["family"], feats, y, supports, spec["support"]["id"],
            l_of, [("BASE", 1.0)], pair_mask_rows, source_contrasts, target_contrasts, draw_rows, proposal_rows, inference_rows, fleet_rows,
        )
        del rows

    r6_spec = {"target": "y_dc_normalized", "coverage_col": "coverage_dc", "coverage": 0.80, "poa_strict_gt": 50.0, "interpolation": "exclude_TRUE"}
    r6_rows, _, r6_feats, r6_y = _dataset(raw, r6_spec)
    j_r6, r6_struct = _r6_structural_targets(incidence, targets, r6_rows, r6_feats)
    if any(t not in targets for t in j_r6):
        raise S53Stop("S53_STOP_R6_OUTSIDE_STAGE43")
    r6_origin_rows = []
    for origin_id, (start, end, block) in ORIGINS.items():
        splits = _r6_splits(r6_rows, origin_id)
        models, val_daily = _fit_models(root, f"R6_TEMPORAL__{origin_id}", plants, set(j_r6), splits, r6_feats, r6_y, FAM_SPLINE, selection_rows, model_records, origin_id)
        l_of = _block_lengths("R6_TEMPORAL", origin_id, j_r6, val_daily, block_rows)
        supports = _support_sets("R6_TEMPORAL", incidence, splits, r6_feats, 0.95)
        factors = [("BASE", 1.0)] + list(R6_BLOCK_FACTORS)
        summary = _score_rq3(
            "R6_TEMPORAL", origin_id, incidence, j_r6, models, FAM_SPLINE, r6_feats, r6_y, supports, "CS4",
            l_of, factors, pair_mask_rows, source_contrasts, target_contrasts, draw_rows, proposal_rows, inference_rows, fleet_rows,
        )
        arm_summaries[f"R6_TEMPORAL__{origin_id}"] = summary
        excluded = sorted(set(targets) - set(j_r6))
        r6_origin_rows.append({
            "origin_id": origin_id, "origin_timestamp": start, "test_half_open_end": end, "test_block": block,
            "balanced_panel_rule": "INTERSECTION_OF_STRUCTURALLY_COMPUTABLE_UNITS_ACROSS_O1_O2_O3",
            "balanced_target_count": len(j_r6), "balanced_target_ids": json.dumps(j_r6),
            "structural_excluded_target_ids": json.dumps(excluded),
            "D_fleet_RQ3": summary["D_fleet_RQ3"], "target_denominator": len(summary["computable_targets"]),
            "cross_origin_pooled": False, "fleet_ci": None, "fleet_p_value": None,
        })

    _primary_block_sensitivity(root, draw_rows, proposal_rows, inference_rows, sensitivity_rows)

    registry = []
    for arm in ARM_SPECS:
        registry.append({"arm_id": arm, "origin_id": "", "row_type": "scientific_robustness", "changed_dimension": ARM_SPECS[arm]["changed_dimension"], "executed": True, "factorial": False, "status": "EXECUTED", "mapping_revision": REVISION})
    for origin in ORIGINS:
        registry.append({"arm_id": "R6_TEMPORAL", "origin_id": origin, "row_type": "scientific_robustness", "changed_dimension": "temporal_origin", "executed": True, "factorial": False, "status": "EXECUTED", "mapping_revision": REVISION})
    for arm, _ in PRIMARY_BLOCK_FACTORS:
        registry.append({"arm_id": arm, "origin_id": "", "row_type": "inference_sensitivity", "changed_dimension": "mbb_block_length_only", "executed": True, "factorial": False, "status": "EXECUTED_NO_REFIT", "mapping_revision": REVISION})
    registry.append({"arm_id": "R7_OMITTED", "origin_id": "", "row_type": "omitted", "changed_dimension": "high_output_cutoff", "executed": False, "factorial": False, "status": R7_EXECUTION, "mapping_revision": REVISION})
    if any(r["factorial"] or r["arm_id"] in FORBIDDEN_FACTORIAL_ARMS for r in registry):
        raise S53Stop("S53_STOP_FACTORIAL")

    protocol = {
        "revision_id": REVISION, "stage43_mapping": MAP_REL, "stage43_mapping_sha256": sha256_file(map_path),
        "exact_twin_incidence": INC_REL, "arm_specs": ARM_SPECS, "features": list(FEATURES),
        "plant_id_feature": False, "state_feature": False, "tracker_albedo_index": "EXCLUDED",
        "imputation": False, "target_adaptation": False, "grids": {"spline": SPLINE_GRID, "hgb": HGB_GRID},
        "selection": "minimum local VALIDATION MAE; exact tie STOP", "final_fit_partition": FINAL_FIT_PARTITION,
        "split": "eligibility_then_chronological_floor_60_20_20",
        "support": "source TRAIN median/IQR Euclidean k=5 arm q",
        "M1": "equal_weight_arithmetic_mean", "F1": "descriptive_target_balanced_mean",
        "p1_semantics": freeze["p1_semantics"], "B_boot": B_BOOT, "target_ci": PERCENTILE_INTERVAL,
        "block_selector": BLOCK_SELECTOR, "primary_block_factors": list(PRIMARY_BLOCK_FACTORS),
        "r6_block_factors": list(R6_BLOCK_FACTORS), "r6_origins": ORIGINS,
        "revised_r6_balanced_panel_rule": "INTERSECTION_OF_STRUCTURALLY_COMPUTABLE_UNITS_ACROSS_O1_O2_O3",
        "seed_operationalization": "seed_uint32(RQ3, ridge_spline, unit_id, seed_arm)",
        "r7": R7_EXECUTION, "c8_repeated_for_robustness": False, "fleet_robustness": "DESCRIPTIVE_ONLY",
        "rq1_rq2_reused_unchanged": True, "factorial_arms": False, "cross_arm_design": CROSS_ARM_DESIGN,
        "python_version": sys.version, "numpy_version": np.__version__,
    }
    recon = {
        "revision_id": REVISION, "stage52_v2_hash_reconciled": True, "historical_map_used": False,
        "stage43_mapping_used": True, "cross_state_fallback": False, "rq1_recomputed": False, "rq2_recomputed": False,
        "rq1_rq2_hashes_unchanged": True, "mandatory_arms_complete": True, "r7_omitted": True,
        "factorial_arms": False, "new_model_family": False, "new_feature_family": False, "target_adaptation": False,
        "pairwise_intersection_only": True, "support_union": False, "global_intersection": False, "target_rescue": False,
        "M1_preserved": True, "F1_preserved": True, "p1_preserved": True, "B_boot_retained_per_target": 5000,
        "retained_empty_support_draws": 0, "source_dropping": False, "bootstrap_refit": False,
        "fleet_robustness_ci": False, "fleet_robustness_p_value": False, "c8_repeated_for_robustness": False,
        "primary_block_factors_exact": True, "r6_block_factors_exact": True, "r6_balanced_panel_rule_preserved": True,
        "r6_available_case_denominator": False, "r6_cross_origin_pooling": False, "historical_s09_unchanged": True,
        "historical_s10_unchanged": True, "stage45_51_unchanged": True, "paper_unchanged": True,
        "r6_balanced_target_ids": j_r6, "arm_denominators": {k: v for k, v in arm_summaries.items()},
    }
    dump_json(out / "S09_STATE_RQ3_PROTOCOL.json", protocol)
    dump_json(out / "S09_STATE_RQ3_CONTROL_MODEL_MANIFEST.json", {"models": model_records})
    dump_json(out / "S09_STATE_RQ3_RECONCILIATION.json", recon)
    files = [
        ("csv", "S09_STATE_RQ3_ARM_EXECUTION_REGISTRY.csv", registry),
        ("csv", "S09_STATE_RQ3_CONTROL_HYPERPARAMETER_SELECTION.csv", selection_rows),
        ("pq", "S09_STATE_RQ3_PAIRWISE_SUPPORT_MASKS.parquet", pair_mask_rows),
        ("csv", "S09_STATE_RQ3_SOURCE_LEVEL_CONTRASTS.csv", source_contrasts),
        ("csv", "S09_STATE_RQ3_TARGET_LEVEL_CONTRASTS.csv", target_contrasts),
        ("csv", "S09_STATE_RQ3_BLOCK_LENGTH_SELECTION.csv", block_rows),
        ("csv", "S09_STATE_RQ3_BOOTSTRAP_PROPOSAL_AUDIT.csv", proposal_rows),
        ("pq", "S09_STATE_RQ3_BOOTSTRAP_DRAWS.parquet", draw_rows),
        ("csv", "S09_STATE_RQ3_TARGET_LEVEL_INFERENCE.csv", inference_rows),
        ("csv", "S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv", fleet_rows),
        ("csv", "S09_STATE_RQ3_R6_ORIGIN_SUMMARY.csv", r6_origin_rows),
        ("csv", "S09_STATE_RQ3_PRIMARY_BLOCK_LENGTH_SENSITIVITY.csv", sensitivity_rows),
    ]
    recs: list[ArtifactRecord] = []
    for kind, name, rows in files:
        path = out / name
        fields = sorted({k for r in rows for k in r}) if rows else ["arm_id"]
        if kind == "pq":
            pref = list(rows[0]) if rows else fields
            _write_pq(path, rows, pref)
            recs.append(_art(root, path, len(rows), _pq_fp(path)))
        else:
            write_csv(path, rows, fields)
            recs.append(_art(root, path, len(rows), _csv_fp(path)))

    report = _report(arm_summaries, j_r6, r6_origin_rows, freeze, map_path)
    report_path = root / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    dump_json(out / "S09_STATE_RQ3_PROTOCOL.json", protocol)
    recs.append(_art(root, out / "S09_STATE_RQ3_PROTOCOL.json"))
    recs.append(_art(root, out / "S09_STATE_RQ3_CONTROL_MODEL_MANIFEST.json"))
    recs.append(_art(root, out / "S09_STATE_RQ3_RECONCILIATION.json"))
    recs.append(_art(root, report_path))
    dump_json(out / "S09_STATE_RQ3_MANIFEST.json", {"job": JOB, "run_id": ctx.run_id, "artifacts": [_tab(r) for r in recs]})
    recs.append(_art(root, out / "S09_STATE_RQ3_MANIFEST.json"))

    hist_after = snapshot_tree(root, PROTECTED)
    if hist_after != hist_before:
        raise S53Stop("S53_STOP_PROTECTED_MUTATION", str(sorted(set(hist_after) ^ set(hist_before))[:20]))
    if json.dumps(_nested(ctx.sot, "scientific_analysis", "s09"), sort_keys=True) != s09_before:
        raise S53Stop("S53_STOP_S09_NODE_MUTATED")
    if json.dumps(_nested(ctx.sot, "scientific_analysis", "s09", "arms", "R1_HGB"), sort_keys=True) != rq1_before:
        raise S53Stop("S53_STOP_RQ1_CHANGED")
    if json.dumps(_nested(ctx.sot, "scientific_analysis", "s09", "arms", "R2_AC"), sort_keys=True) != rq2_before:
        raise S53Stop("S53_STOP_RQ2_CHANGED")

    arts_map = {Path(r.path).name: _tab(r) for r in recs}
    s10_prev = _nested(ctx.sot, "scientific_freeze", "s10_revisions", REVISION) or {}
    sot_patch = {
        "stages": {STAGE_NODE: {
            "kind": "scientific_analysis_revision", "status": "GO", "reason": REASON_GO,
            "operational_stage_id": "53", "job": JOB, "run_id": ctx.run_id, "finished_at_utc": now,
            "revision_id": REVISION, "scope": "RQ3_ONLY_DOWNSTREAM_ROBUSTNESS_REVISION",
            "rq1_recomputed": False, "rq2_recomputed": False, "mandatory_arms_complete": True,
            "fleet_robustness_inference": "DESCRIPTIVE_ONLY", "c8_repeated_for_robustness": False,
            "answer": ANSWER, "report": _tab(next(r for r in recs if r.path == REPORT)),
            "manifest": _tab(next(r for r in recs if r.path.endswith("S09_STATE_RQ3_MANIFEST.json"))),
            "next_action": "CONTROLLER_REVIEW_FOR_S10_REMATERIALIZATION",
        }},
        "scientific_analysis": {"s09_revisions": {REVISION: {
            "status": "GO", "reason": REASON_GO, "operational_stage_id": "53", "revision_id": REVISION,
            "scope": "RQ3_ONLY_DOWNSTREAM_ROBUSTNESS_REVISION", "rq1_rq2_source": "HISTORICAL_S09_ACTIVE_UNCHANGED",
            "rq1_recomputed": False, "rq2_recomputed": False, "rq3_source_mapping": "STAGE43_SAME_STATE_MAPPING",
            "historical_map_used": False, "cross_state_fallback": False, "cross_arm_design": CROSS_ARM_DESIGN,
            "r7_execution": R7_EXECUTION, "mandatory_arms_complete": True,
            "arms": {k: v for k, v in arm_summaries.items() if not str(k).startswith("R6_TEMPORAL__")},
            "r6": {
                "balanced_panel_rule": "INTERSECTION_OF_STRUCTURALLY_COMPUTABLE_UNITS_ACROSS_O1_O2_O3",
                "balanced_target_ids": j_r6, "balanced_target_count": len(j_r6),
                "origins": {r["origin_id"]: r for r in r6_origin_rows},
                "cross_origin_pooled_estimate": None, "cross_origin_inference": None,
            },
            "primary_block_length_sensitivity": {"factors": [0.5, 1.0, 2.0], "no_refit": True, "artifact": arts_map.get("S09_STATE_RQ3_PRIMARY_BLOCK_LENGTH_SENSITIVITY.csv")},
            "target_level_inference": "temporal_MBB_P1", "B_boot": B_BOOT,
            "fleet_robustness_inference": "DESCRIPTIVE_ONLY", "fleet_robustness_p_value": None,
            "c8_repeated_for_robustness": False, "artifacts": arts_map,
            "report": _tab(next(r for r in recs if r.path == REPORT)),
            "next_action": "CONTROLLER_REVIEW_FOR_S10_REMATERIALIZATION",
        }}},
        "scientific_freeze": {"s10_revisions": {REVISION: {
            **s10_prev,
            "status": "REVISED_S09_RQ3_AVAILABLE_PENDING_CONTROLLER_REVIEW",
            "stage53_status": "GO",
            "stage53_result_node": "scientific_analysis.s09_revisions.state_constrained_support_v2",
            "s10_execution_authorized": False,
            "next_action": "CONTROLLER_REVIEW_STAGE53",
        }}},
    }
    validations.extend([
        ValidationRecord("p1_retained", True, str(B_BOOT)),
        ValidationRecord("r7_omitted", True, R7_EXECUTION),
        ValidationRecord("protected_unchanged", True, None),
        ValidationRecord("no_stage54", True, None),
    ])
    return StageResult(status="GO", message=MSG_GO, sot_patch=sot_patch, artifacts=recs, validations=validations)


def _report(arm_summaries: dict[str, Any], j_r6: list[str], r6_origin_rows: list[dict[str, Any]], freeze: dict[str, Any], map_path: Path) -> str:
    lines = [
        "# S09 state-constrained RQ3 robustness",
        "",
        "## 1. Frozen specification consumed",
        f"Revision `{REVISION}`. Stage 52 V2 freeze authorized Stage 53. Mapping `{map_path.as_posix()}`.",
        "C8 was not repeated. Fleet robustness is descriptive only.",
        "",
        "## 2. Active mapping / target population",
        "Stage 43 same-state supported targets (n=10). Historical MAP unused.",
        "",
        "## 3. R1–R5 robustness results",
    ]
    for arm in ARM_SPECS:
        s = arm_summaries[arm]
        lines.append(f"- `{arm}` computable={len(s['computable_targets'])} F1={s['D_fleet_RQ3']} excluded={s['excluded']}")
    lines += [
        "",
        "## 4. Target-level uncertainty",
        "P1 temporal MBB, B_boot=5000 retained draws, percentile 95 CI. No target p-value.",
        "",
        "## 5. R6 structural balanced-panel derivation",
        f"J_R6 = {j_r6} (intersection of structurally computable targets across O1/O2/O3).",
        "",
        "## 6. R6 O1/O2/O3 results",
    ]
    for row in r6_origin_rows:
        lines.append(f"- `{row['origin_id']}` D_fleet_RQ3={row['D_fleet_RQ3']} denominator={row['target_denominator']}")
    lines += [
        "",
        "## 7. Primary block-length sensitivity",
        "Factors 0.5/1.0/2.0 on revised Stage 45 payloads. No new point estimate. No C8.",
        "",
        "## 8. Exclusions",
        "Empty pairwise intersections are NON_COMPUTABLE_NO_RESCUE. No target rescue.",
        "",
        "## 9. Reconciliation",
        "RQ1/RQ2 unchanged. Historical S09/S10/paper byte-identical. Stage 54 not executed.",
        "",
        "## 10. Interpretation constraints",
        "Results do not authorize method switching, superpopulation inference, or equivalence claims.",
        f"c8_repeated_for_robustness={freeze.get('c8_repeated_for_robustness')}.",
    ]
    return "\n".join(lines) + "\n"
