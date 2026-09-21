"""Stage 45 helper: revised same-state RQ3 scientific execution."""

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
from src.lib.split_support import FEATURE_INTERP, assign_post_eligibility_split, dump_json, parse_ts, sha256_file, write_csv
from src.lib.rq3_match import snapshot_tree
from src.lib.rq3_mbb import (
    B_BOOT,
    CI_LEVEL,
    FAM_TOKEN,
    _art,
    _csv_fp,
    _day,
    _norm_dt,
    _pq_fp,
    _tab,
    _write_pq,
    empty_pairwise_support_in_draw,
    mbb_calendar_draw,
)
from src.lib.rq3_align import repo_relative
from src.seeds import seed_uint32
from src.lib.models import (
    FEATURES,
    FAM_HGB,
    FAM_SPLINE,
    SPLINE_GRID,
    TARGET,
    _bool01,
    _fit_select,
    eligible_mask,
    TieError,
)

JOB = "S08_STATE_CONSTRAINED_SCIENTIFIC_RQ3"
REVISION = "state_constrained_support_v2"
MSG_GO = "GO — REVISED SAME-STATE RQ3 EXECUTED UNDER FROZEN DESIGN"
MSG_STOP = "STOP — S08 STATE-CONSTRAINED SCIENTIFIC RQ3 INCOMPLETE"
OUT = "artifacts/s08_state_constrained_scientific"
REPORT = "reports/scientific/S08_STATE_CONSTRAINED_SCIENTIFIC_RQ3.md"
ANSWER = None
P1_POLICY = "P1_CONDITIONAL_VALID_DRAW_RESAMPLING"
P1_DECISION_SHA = "0a00780e62c9413d0bd6b3b6557ccbb998b7c7c0b8f153b5d8ad00b32b40d380"
PROTECTED = (
    "artifacts/s03_finalization",
    "artifacts/s03_inference_dependency_audit",
    "artifacts/s04",
    "artifacts/s05",
    "artifacts/s06",
    "artifacts/s07",
    "artifacts/s08",
    "artifacts/s08_alignment_audit",
    "artifacts/s08_scientific",
    "artifacts/s08_state_constrained_support_alignment",
    "artifacts/s08_state_constrained_bootstrap_audit",
    "artifacts/s08_state_constrained_bootstrap_freeze",
    "artifacts/s09_scientific",
    "artifacts/s10",
    "paper",
    "paper.md",
)


class S45Stop(Exception):
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_upstream(sot: dict[str, Any], root: Path | None = None) -> None:
    if _nested(sot, "stages", "S03_RQ3_STATE_REVISION_FINALIZATION", "status") != "GO":
        raise S45Stop("S45_STOP_STAGE43_NOT_GO")
    if _nested(sot, "scientific_freeze", "s03", "active_rq3_control_revision") != REVISION:
        raise S45Stop("S45_STOP_ACTIVE_REVISION")
    if _nested(sot, "scientific_freeze", "s03", "rq3_revisions", REVISION, "status") != "FROZEN":
        raise S45Stop("S45_STOP_REVISION_NOT_FROZEN")
    if _nested(sot, "stages", "S08_STATE_CONSTRAINED_SUPPORT_ALIGNMENT", "status") != "GO":
        raise S45Stop("S45_STOP_STAGE44_NOT_GO")
    align = _nested(sot, "scientific_analysis", "s08_revisions", REVISION, "support_alignment") or {}
    if align.get("status") != "GO":
        raise S45Stop("S45_STOP_SUPPORT_ALIGN_NOT_GO")
    if align.get("scientific_results_computed") is not False:
        raise S45Stop("S45_STOP_SUPPORT_ALREADY_COMPUTED")
    if align.get("outcomes_accessed") is not False:
        raise S45Stop("S45_STOP_SUPPORT_OUTCOMES_ACCESSED")
    for key in ("S04", "S05", "S06", "S07"):
        if (_nested(sot, "stages", key, "status") != "GO") and (_nested(sot, "scientific_analysis", key.lower(), "status") not in {"GO", None}):
            st = _nested(sot, "stages", key) or {}
            sa = _nested(sot, "scientific_analysis", key.lower()) or {}
            if st.get("status") != "GO" and sa.get("status") != "GO":
                raise S45Stop("S45_STOP_UPSTREAM_NOT_GO", key)
    op = _nested(sot, "scientific_freeze", "s08_rq3_operationalization") or {}
    if (op.get("support_alignment") or {}).get("method") != "B_PAIRWISE_INTERSECTION":
        raise S45Stop("S45_STOP_ALIGNMENT_NOT_B")
    if (op.get("target_level_aggregation") or {}).get("method") != "M1_EQUAL_WEIGHT_ARITHMETIC_MEAN":
        raise S45Stop("S45_STOP_AGG_NOT_M1")
    if (op.get("contrast_orientation") or {}).get("orientation") != "CONTROL_MINUS_TWIN":
        raise S45Stop("S45_STOP_ORIENTATION")
    if (op.get("fleet_level_synthesis") or {}).get("method") != "F1_TARGET_BALANCED_ARITHMETIC_MEAN":
        raise S45Stop("S45_STOP_FLEET_NOT_F1")
    st47 = _nested(sot, "stages", "S08_STATE_CONSTRAINED_BOOTSTRAP_POLICY_CONTROLLER_FREEZE") or {}
    if st47.get("status") != "GO":
        raise S45Stop("S45_STOP_STAGE47_NOT_GO")
    pol = _nested(sot, "scientific_freeze", "s08_rq3_operationalization", "revisions", REVISION, "bootstrap_empty_draw_policy") or {}
    if pol.get("status") != "FROZEN" or pol.get("policy_id") != P1_POLICY:
        raise S45Stop("S45_STOP_P1_NOT_FROZEN", str(pol.get("policy_id")))
    if pol.get("outcomes_accessed_for_decision") is not False:
        raise S45Stop("S45_STOP_P1_OUTCOMES_FOR_DECISION")
    if pol.get("effects_computed_for_decision") is not False:
        raise S45Stop("S45_STOP_P1_EFFECTS_FOR_DECISION")
    st48 = _nested(sot, "stages", "S47_SOT_NEXT_ACTION_RECONCILIATION") or {}
    if st48.get("status") != "GO":
        raise S45Stop("S45_STOP_STAGE48_NOT_GO")
    audit = _nested(sot, "scientific_freeze", "s08_rq3_operationalization", "revisions", REVISION, "bootstrap_empty_draw_audit") or {}
    if audit.get("next_action") != "RERUN_STAGE_45" or pol.get("next_action") != "RERUN_STAGE_45":
        raise S45Stop("S45_STOP_NEXT_ACTION")
    if int(pol.get("B_boot_retained") or 0) != 5000:
        raise S45Stop("S45_STOP_P1_BBOOT")
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
            raise S45Stop("S45_STOP_P1_INVARIANT", key)
    dec = pol.get("decision") or {}
    dec_path = (root or Path(".")) / (dec.get("path") or "artifacts/s08_state_constrained_bootstrap_freeze/S08_STATE_BOOTSTRAP_POLICY_CONTROLLER_DECISION.json")
    expected = dec.get("sha256") or P1_DECISION_SHA
    if sha256_file(dec_path) != expected or expected != P1_DECISION_SHA:
        raise S45Stop("S45_STOP_P1_DECISION_HASH")


def execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    try:
        return _execute(ctx, now, validations)
    except S45Stop as exc:
        validations.append(ValidationRecord(exc.reason, False, exc.details))
        return StageResult(
            status="STOP",
            message=MSG_STOP,
            sot_patch={
                "stages": {
                    "S08_STATE_CONSTRAINED_SCIENTIFIC_RQ3": {
                        "kind": "scientific_analysis_revision",
                        "status": "STOP",
                        "reason": exc.reason,
                        "operational_stage_id": "45",
                        "job": JOB,
                        "details": exc.details,
                        "finished_at_utc": now,
                        "answer": ANSWER,
                        "revision_id": REVISION,
                    }
                }
            },
            artifacts=[],
            validations=validations,
        )
    except TieError as exc:
        return execute(
            StageContext(ctx.root, ctx.stage_id, ctx.run_id, ctx.sot),
            now,
            validations + [ValidationRecord("S45_STOP_VALIDATION_MAE_TIE", False, str(exc))],
        ) if False else StageResult(
            status="STOP",
            message=MSG_STOP,
            sot_patch={
                "stages": {
                    "S08_STATE_CONSTRAINED_SCIENTIFIC_RQ3": {
                        "kind": "scientific_analysis_revision",
                        "status": "STOP",
                        "reason": "S45_STOP_VALIDATION_MAE_TIE",
                        "operational_stage_id": "45",
                        "job": JOB,
                        "details": str(exc),
                        "finished_at_utc": now,
                        "answer": ANSWER,
                        "revision_id": REVISION,
                    }
                }
            },
            artifacts=[],
            validations=validations + [ValidationRecord("S45_STOP_VALIDATION_MAE_TIE", False, str(exc))],
        )


def _execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    root = ctx.root
    hist_before = snapshot_tree(root, PROTECTED)
    validate_upstream(ctx.sot, root)
    validations.append(ValidationRecord("upstream_revision", True, REVISION))
    if seed_uint32("RQ3", FAM_TOKEN, "PS_002", "L_j") != 1676118836:
        raise S45Stop("S45_STOP_SEED_EXAMPLE_MISMATCH")
    validations.append(ValidationRecord("seed_example", True, "1676118836"))

    rev = _nested(ctx.sot, "scientific_freeze", "s03", "rq3_revisions", REVISION) or {}
    mapping_rows = list(rev.get("selected_mapping") or [])
    control_of = {r["reference_plant_id"]: r["control_plant_id"] for r in mapping_rows}
    supported = list(rev.get("supported_targets") or [])
    if sorted(control_of) != sorted(supported):
        raise S45Stop("S45_STOP_MAPPING_SUPPORT")
    align = _nested(ctx.sot, "scientific_analysis", "s08_revisions", REVISION, "support_alignment") or {}
    stage44 = _nested(ctx.sot, "stages", "S08_STATE_CONSTRAINED_SUPPORT_ALIGNMENT") or {}
    arts44 = stage44.get("artifacts") or {}
    flow_rec = align.get("support_flow") or arts44.get("support_flow") or {}
    flow_path = root / (flow_rec.get("path") or "artifacts/s08_state_constrained_support_alignment/S08_STATE_SUPPORT_FLOW.json")
    if flow_rec.get("sha256") and sha256_file(flow_path) != flow_rec["sha256"]:
        raise S45Stop("S45_STOP_FLOW_HASH")
    flow = json.loads(flow_path.read_text(encoding="utf-8"))
    j_comp_struct = list(flow.get("J_comp_structural") or [])
    pair_mask_rec = arts44.get("pairwise_masks") or {}
    pair_mask_path = root / (pair_mask_rec.get("path") or "artifacts/s08_state_constrained_support_alignment/S08_STATE_PAIRWISE_SUPPORT_MASKS.parquet")
    if pair_mask_rec.get("sha256") and sha256_file(pair_mask_path) != pair_mask_rec["sha256"]:
        raise S45Stop("S45_STOP_PAIRWISE_MASK_HASH")
    audit_path = root / ((arts44.get("pairwise_audit") or align.get("pairwise_alignment_summary") or {}).get("path") or "artifacts/s08_state_constrained_support_alignment/S08_STATE_PAIRWISE_ALIGNMENT_AUDIT.csv")
    audit_rows = _read_csv(audit_path)
    group_of: dict[str, str] = {}
    expected_twins: dict[str, list[str]] = defaultdict(list)
    pairwise_items = []
    source_exclusions = []
    mask_table = pq.read_table(pair_mask_path)
    dts_map: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for twin, tgt, cid, dt, ok in zip(
        mask_table["twin_source_plant_id"].to_pylist(),
        mask_table["target_plant_id"].to_pylist(),
        mask_table["control_source_plant_id"].to_pylist(),
        mask_table["datetime_raw"].to_pylist(),
        mask_table["pairwise_supported"].to_pylist(),
    ):
        if ok:
            dts_map[(str(twin), str(tgt), str(cid))].add(_norm_dt(dt))
    for row in audit_rows:
        tgt = row["target_plant_id"]
        twin = row["twin_source_plant_id"]
        cid = row["control_source_plant_id"]
        group_of[tgt] = row["reference_group_id"]
        expected_twins[tgt].append(twin)
        if cid != control_of.get(tgt):
            raise S45Stop("S45_STOP_AUDIT_CONTROL_MISMATCH", tgt)
        if _flag(row.get("computable")):
            dts = dts_map.get((twin, tgt, cid), set())
            if len(dts) != int(float(row["n_pairwise_intersection"])):
                raise S45Stop("S45_STOP_MASK_N_MISMATCH", f"{twin}->{tgt}")
            if not dts:
                raise S45Stop("S45_STOP_COMPUTABLE_EMPTY_MASK", f"{twin}->{tgt}")
            pairwise_items.append({"target": tgt, "twin": twin, "control": cid, "group": row["reference_group_id"], "dts": dts})
        else:
            source_exclusions.append({"target": tgt, "twin": twin, "reason": row.get("reason") or "NONCOMPUTABLE"})

    targets = sorted({item["target"] for item in pairwise_items})
    if sorted(targets) != sorted(j_comp_struct):
        raise S45Stop("S45_STOP_JCOMP_STRUCT_MISMATCH", str(sorted(targets)))
    controls = sorted(set(control_of[t] for t in targets))
    validations.append(ValidationRecord("stage44_masks_consumed", True, str(len(pairwise_items))))

    sa = ctx.sot.get("scientific_analysis") or {}
    s04 = sa.get("s04") or {}
    s06 = sa.get("s06") or {}
    panel_path = root / ((_nested(ctx.sot, "scientific_freeze", "s02", "canonical_panel") or {}).get("path") or "artifacts/p02c/P02C_PLANT_PANEL.parquet")
    s06_pred = root / ((s06.get("predictions") or s06.get("cross_plant_predictions") or {}).get("path") or "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet")
    s04_test = root / ((s04.get("local_test_predictions") or {}).get("path") or "artifacts/s04/S04_LOCAL_TEST_PREDICTIONS.parquet")
    s04_val = root / ((s04.get("validation_predictions") or {}).get("path") or "artifacts/s04/S04_VALIDATION_PREDICTIONS.parquet")
    for p in (panel_path, s06_pred, s04_test, s04_val, pair_mask_path):
        if not p.is_file():
            raise S45Stop("S45_STOP_MISSING_INPUT", str(p))

    plist = ",".join("'" + p + "'" for p in controls)
    con = duckdb.connect()
    table = con.execute(
        f"""
        SELECT plant_id, CAST(datetime_raw AS VARCHAR) AS datetime_raw, y_dc_normalized, coverage_dc, poa_irradiance_wm2,
               ghi_irradiance_wm2, gri_irradiance_wm2, panel_temperature_celsius,
               ambient_temperature_celsius, wind_speed_ms, dc_any_officially_interpolated,
               interpolated_keys_poa_irradiance_wm2, interpolated_keys_ghi_irradiance_wm2,
               interpolated_keys_gri_irradiance_wm2, interpolated_keys_panel_temperature_celsius,
               interpolated_keys_ambient_temperature_celsius, interpolated_keys_wind_speed_ms
        FROM read_parquet('{str(panel_path).replace("'", "''")}')
        WHERE plant_id IN ({plist})
          AND y_dc_normalized IS NOT NULL AND poa_irradiance_wm2 > 50 AND coverage_dc >= 0.80
          AND ghi_irradiance_wm2 IS NOT NULL AND gri_irradiance_wm2 IS NOT NULL
          AND panel_temperature_celsius IS NOT NULL AND ambient_temperature_celsius IS NOT NULL
          AND wind_speed_ms IS NOT NULL
          AND (dc_any_officially_interpolated IS NULL OR dc_any_officially_interpolated IS FALSE)
          AND (interpolated_keys_poa_irradiance_wm2 IS NULL OR interpolated_keys_poa_irradiance_wm2 IS FALSE)
          AND (interpolated_keys_ghi_irradiance_wm2 IS NULL OR interpolated_keys_ghi_irradiance_wm2 IS FALSE)
          AND (interpolated_keys_gri_irradiance_wm2 IS NULL OR interpolated_keys_gri_irradiance_wm2 IS FALSE)
          AND (interpolated_keys_panel_temperature_celsius IS NULL OR interpolated_keys_panel_temperature_celsius IS FALSE)
          AND (interpolated_keys_ambient_temperature_celsius IS NULL OR interpolated_keys_ambient_temperature_celsius IS FALSE)
          AND (interpolated_keys_wind_speed_ms IS NULL OR interpolated_keys_wind_speed_ms IS FALSE)
        """
    ).to_arrow_table()
    con.close()
    by_plant: dict[str, list[int]] = {p: [] for p in controls}
    for i, pid in enumerate(table["plant_id"].to_pylist()):
        by_plant[str(pid)].append(i)
    y_all = np.array(table[TARGET].to_pylist(), dtype=float)
    poa_all = np.array(table["poa_irradiance_wm2"].to_pylist(), dtype=float)
    cov_all = np.array(table["coverage_dc"].to_pylist(), dtype=float)
    feat_all = {f: np.array(table[f].to_pylist(), dtype=float) for f in FEATURES}
    dc_all = _bool01(table["dc_any_officially_interpolated"].to_pylist())
    flag_all = {f: _bool01(table[FEATURE_INTERP[f]].to_pylist()) for f in FEATURES}
    dt_all = [_norm_dt(x) for x in table["datetime_raw"].to_pylist()]
    ts_all = [parse_ts(x) for x in dt_all]
    analytic_rows = []
    sel_rows = []
    models: dict[str, Any] = {}
    out = root / OUT
    models_dir = out / "models"
    out.mkdir(parents=True, exist_ok=True)
    for pid in controls:
        ix = np.array(by_plant[pid], dtype=int)
        elig = eligible_mask(poa_all[ix], cov_all[ix], y_all[ix], {k: v[ix] for k, v in feat_all.items()}, dc_all[ix], {k: v[ix] for k, v in flag_all.items()})
        local_ts = [ts_all[int(j)] for j in ix]
        local_dt = [dt_all[int(j)] for j in ix]
        order = [int(k) for k in np.where(elig)[0]]
        sentinel = datetime.max.replace(tzinfo=timezone.utc)
        order.sort(key=lambda k: (local_ts[k] if local_ts[k] is not None else sentinel, local_dt[k]))
        n = len(order)
        labels = [""] * len(ix)
        for rank, k in enumerate(order):
            labels[k] = assign_post_eligibility_split(n, rank)
        train_k = [k for k, lab in enumerate(labels) if lab == "train"]
        val_k = [k for k, lab in enumerate(labels) if lab == "validation"]
        test_k = [k for k, lab in enumerate(labels) if lab == "test"]
        if not train_k or not val_k or not test_k:
            raise S45Stop("S45_STOP_EMPTY_CONTROL_PARTITION", pid)
        for k, lab in enumerate(labels):
            analytic_rows.append({"control_plant_id": pid, "datetime_raw": local_dt[k], "split": lab or None})
        Xtr = np.column_stack([feat_all[f][ix[train_k]] for f in FEATURES])
        ytr = y_all[ix[train_k]]
        Xva = np.column_stack([feat_all[f][ix[val_k]] for f in FEATURES])
        yva = y_all[ix[val_k]]
        _params, mae, est, cand = _fit_select(FAM_SPLINE, SPLINE_GRID, Xtr, ytr, Xva, yva)
        cid_sel = [r["candidate_id"] for r in cand if r["fit_status"] == "ok" and r["validation_mae"] == mae][0]
        for r in cand:
            sel_rows.append(
                {
                    "control_plant_id": pid,
                    "model_family": FAM_SPLINE,
                    "candidate_id": r["candidate_id"],
                    "hyperparameters": json.dumps(r["params"], sort_keys=True),
                    "train_row_count": len(train_k),
                    "validation_row_count": len(val_k),
                    "test_row_count": len(test_k),
                    "validation_mae": r["validation_mae"],
                    "selected": r["candidate_id"] == cid_sel,
                    "fit_status": r["fit_status"],
                }
            )
        dest = models_dir / pid / FAM_SPLINE
        dest.mkdir(parents=True, exist_ok=True)
        joblib.dump(est, dest / "full.joblib")
        models[pid] = est
    validations.append(ValidationRecord("control_models_train_val_only", True, None))

    tlist = ",".join("'" + p + "'" for p in targets)
    con = duckdb.connect()
    tgt_tbl = con.execute(
        f"""
        SELECT plant_id, CAST(datetime_raw AS VARCHAR) AS datetime_raw, y_dc_normalized,
               poa_irradiance_wm2, ghi_irradiance_wm2, gri_irradiance_wm2,
               panel_temperature_celsius, ambient_temperature_celsius, wind_speed_ms
        FROM read_parquet('{str(panel_path).replace("'", "''")}')
        WHERE plant_id IN ({tlist})
        """
    ).to_arrow_table()
    con.close()
    feat_by: dict[tuple[str, str], np.ndarray] = {}
    for i in range(tgt_tbl.num_rows):
        pid = str(tgt_tbl["plant_id"][i].as_py())
        dt = _norm_dt(tgt_tbl["datetime_raw"][i].as_py())
        vals = [tgt_tbl[f][i].as_py() for f in FEATURES]
        if any(v is None for v in vals):
            continue
        feat_by[(pid, dt)] = np.array(vals, dtype=float)

    twin_pred: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    tp = pq.read_table(s06_pred)
    for i in range(tp.num_rows):
        if str(tp["model_family"][i].as_py()) != FAM_SPLINE:
            continue
        twin_pred[(str(tp["source_plant_id"][i].as_py()), str(tp["target_plant_id"][i].as_py()), _norm_dt(tp["datetime_raw"][i].as_py()))] = (
            float(tp["y_true"][i].as_py()),
            float(tp["y_pred_transfer"][i].as_py()),
            float(tp["y_pred_local"][i].as_py()),
        )
    local_pred: dict[tuple[str, str], tuple[float, float]] = {}
    lp = pq.read_table(s04_test)
    for i in range(lp.num_rows):
        if str(lp["model_family"][i].as_py()) != FAM_SPLINE:
            continue
        local_pred[(str(lp["plant_id"][i].as_py()), _norm_dt(lp["datetime_raw"][i].as_py()))] = (float(lp["y_true"][i].as_py()), float(lp["y_pred"][i].as_py()))

    ctrl_pred_rows = []
    source_rows = []
    source_payload: dict[tuple[str, str], dict[str, Any]] = {}
    for item in pairwise_items:
        tgt, twin, cid = item["target"], item["twin"], item["control"]
        dts = sorted(item["dts"])
        missing = [dt for dt in dts if (twin, tgt, dt) not in twin_pred or (tgt, dt) not in local_pred or (tgt, dt) not in feat_by]
        if missing:
            raise S45Stop("S45_STOP_MISSING_PREDICTION_OR_FEATURE", f"{twin}->{tgt} {missing[0]}")
        X = np.stack([feat_by[(tgt, dt)] for dt in dts])
        yhat = np.asarray(models[cid].predict(X), dtype=float)
        e_twin, e_loc, e_ctrl = [], [], []
        by_day_ctrl: dict[str, list[float]] = defaultdict(list)
        by_day_twin: dict[str, list[float]] = defaultdict(list)
        for dt, y_c in zip(dts, yhat):
            y_true, y_tw, y_loc = twin_pred[(twin, tgt, dt)]
            y_true_l, _yhat_l = local_pred[(tgt, dt)]
            if abs(y_true - y_true_l) > 1e-12:
                raise S45Stop("S45_STOP_TRUTH_MISALIGN", f"{tgt} {dt}")
            et = abs(y_tw - y_true)
            el = abs(y_loc - y_true)
            ec = abs(float(y_c) - y_true)
            e_twin.append(et)
            e_loc.append(el)
            e_ctrl.append(ec)
            day = _day(dt)
            by_day_ctrl[day].append(ec)
            by_day_twin[day].append(et)
            ctrl_pred_rows.append({"control_source_plant_id": cid, "target_plant_id": tgt, "twin_source_plant_id": twin, "datetime_raw": dt, "y_true": y_true, "y_pred_control": float(y_c), "y_pred_local": y_loc, "y_pred_twin": y_tw})
        mae_t = float(np.mean(e_twin))
        mae_c = float(np.mean(e_ctrl))
        mae_l = float(np.mean(e_loc))
        otg_t = mae_t - mae_l
        otg_c = mae_c - mae_l
        delta = otg_c - otg_t
        if abs(delta - (mae_c - mae_t)) > 1e-12:
            raise S45Stop("S45_STOP_DELTA_ALGEBRA")
        days = {_day(d) for d in dts}
        audit = next(r for r in audit_rows if r["twin_source_plant_id"] == twin and r["target_plant_id"] == tgt)
        source_rows.append(
            {
                "reference_group_id": item["group"],
                "twin_source_plant_id": twin,
                "target_plant_id": tgt,
                "control_source_plant_id": cid,
                "n_pairwise_rows": len(dts),
                "n_pairwise_days": len(days),
                "pairwise_retention_vs_target": float(audit["pairwise_retention_vs_target"]),
                "twin_transfer_mae": mae_t,
                "control_transfer_mae": mae_c,
                "local_same_rows_mae": mae_l,
                "twin_OTG_abs": otg_t,
                "control_OTG_abs": otg_c,
                "delta_ij": delta,
                "computable": True,
                "reason": "",
            }
        )
        source_payload[(twin, tgt)] = {
            "dts": dts,
            "days": days,
            "by_day_ctrl": {k: np.asarray(v, dtype=float) for k, v in by_day_ctrl.items()},
            "by_day_twin": {k: np.asarray(v, dtype=float) for k, v in by_day_twin.items()},
        }
    source_rows.sort(key=lambda r: (r["target_plant_id"], r["twin_source_plant_id"]))

    by_tgt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_tgt[row["target_plant_id"]].append(row)
    target_rows = []
    excluded_targets = []
    excluded_reasons: dict[str, str] = {}
    for tgt in sorted(supported):
        rows = by_tgt.get(tgt) or []
        if tgt not in j_comp_struct:
            excluded_targets.append(tgt)
            excluded_reasons[tgt] = "NOT_IN_J_COMP_STRUCTURAL"
            continue
        if not rows:
            excluded_targets.append(tgt)
            excluded_reasons[tgt] = "ZERO_SCIENTIFIC_SOURCE_CONTRASTS"
            continue
        Delta = float(np.mean([r["delta_ij"] for r in rows]))
        target_rows.append(
            {
                "target_plant_id": tgt,
                "reference_group_id": group_of[tgt],
                "matched_control_source_plant_id": control_of[tgt],
                "n_expected_exact_twin_sources": len(expected_twins[tgt]),
                "n_pairwise_computable_exact_twin_sources": len(rows),
                "source_ids_included": json.dumps(sorted(r["twin_source_plant_id"] for r in rows)),
                "Delta_j": Delta,
                "point_denominator": len(rows),
            }
        )
    if not target_rows:
        raise S45Stop("S45_STOP_NO_SCIENTIFIC_TARGET")

    val = pq.read_table(s04_val)
    val_ae: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for i in range(val.num_rows):
        if not bool(val["selected_model_only"][i].as_py()):
            continue
        pid = str(val["plant_id"][i].as_py())
        fam = str(val["model_family"][i].as_py())
        dt = _norm_dt(val["datetime_raw"][i].as_py())
        ae = float(val["abs_error"][i].as_py())
        val_ae[(pid, fam)][_day(dt)].append(ae)
    block_rows = []
    L_of: dict[str, int] = {}
    for tgt_row in target_rows:
        tgt = tgt_row["target_plant_id"]
        Ls = []
        for fam in (FAM_SPLINE, FAM_HGB):
            daily = val_ae[(tgt, fam)]
            if len(daily) < 8:
                raise S45Stop("S45_STOP_BLOCK_LENGTH_SERIES", tgt)
            series = np.array([float(np.mean(daily[d])) for d in sorted(daily)], dtype=float)
            obl = optimal_block_length(series)
            raw = float(obl["circular"].iloc[0])
            whole = max(1, int(math.ceil(raw)))
            Ls.append(whole)
            block_rows.append({"target_plant_id": tgt, "model_family": fam, "n_validation_days": len(daily), "n_validation_rows": sum(len(v) for v in daily.values()), "raw_selector_circular": raw, "raw_selector_stationary": float(obl["stationary"].iloc[0]), "whole_day_block_length": whole, "selector": "arch.bootstrap.optimal_block_length_PPW2009", "L_j": 0})
        L_of[tgt] = max(Ls)
        for r in block_rows:
            if r["target_plant_id"] == tgt:
                r["L_j"] = L_of[tgt]
    validations.append(ValidationRecord("block_length_validation_only", True, None))

    boot_rows = []
    proposal_rows = []
    inf_rows = []
    for tgt_row in target_rows:
        tgt = tgt_row["target_plant_id"]
        srcs = by_tgt[tgt]
        src_ids = [r["twin_source_plant_id"] for r in srcs]
        all_days = sorted({d for r in srcs for d in source_payload[(r["twin_source_plant_id"], tgt)]["days"]})
        L = L_of[tgt]
        draws = []
        unit_seed = seed_uint32("RQ3", FAM_TOKEN, tgt, "L_j")
        pays = [source_payload[(sid, tgt)] for sid in src_ids]
        proposal_id = 0
        accepted_draw_id = 0
        n_rejected = 0
        while accepted_draw_id < B_BOOT:
            rng = np.random.default_rng(np.random.SeedSequence([unit_seed, proposal_id]))
            sampled = mbb_calendar_draw(all_days, L, rng)
            sampled_set = set(sampled)
            empty = [sid for sid, pay in zip(src_ids, pays) if empty_pairwise_support_in_draw(sampled_set, pay["days"])]
            if empty:
                proposal_rows.append(
                    {
                        "target_plant_id": tgt,
                        "proposal_id": proposal_id,
                        "accepted": False,
                        "accepted_draw_id": None,
                        "unit_seed": int(unit_seed),
                        "L_j": L,
                        "n_sources_required": len(src_ids),
                        "n_empty_source_operands": len(empty),
                        "empty_source_ids": json.dumps(empty),
                    }
                )
                n_rejected += 1
                proposal_id += 1
                continue
            deltas_b = []
            for pay in pays:
                parts_c = []
                parts_t = []
                for day in sampled:
                    if day in pay["by_day_ctrl"]:
                        parts_c.append(pay["by_day_ctrl"][day])
                        parts_t.append(pay["by_day_twin"][day])
                if not parts_c:
                    empty = src_ids
                    break
                mae_c = float(np.concatenate(parts_c).mean())
                mae_t = float(np.concatenate(parts_t).mean())
                deltas_b.append(mae_c - mae_t)
            if len(deltas_b) != len(src_ids):
                proposal_rows.append(
                    {
                        "target_plant_id": tgt,
                        "proposal_id": proposal_id,
                        "accepted": False,
                        "accepted_draw_id": None,
                        "unit_seed": int(unit_seed),
                        "L_j": L,
                        "n_sources_required": len(src_ids),
                        "n_empty_source_operands": len(src_ids),
                        "empty_source_ids": json.dumps(src_ids),
                    }
                )
                n_rejected += 1
                proposal_id += 1
                continue
            Delta_b = float(np.mean(deltas_b))
            draws.append(Delta_b)
            boot_rows.append(
                {
                    "target_plant_id": tgt,
                    "draw_id": accepted_draw_id,
                    "proposal_id": proposal_id,
                    "seed": int(unit_seed),
                    "L_j": L,
                    "Delta_j_draw": Delta_b,
                    "source_count_denominator": len(src_ids),
                    "empty_pairwise_support": False,
                }
            )
            proposal_rows.append(
                {
                    "target_plant_id": tgt,
                    "proposal_id": proposal_id,
                    "accepted": True,
                    "accepted_draw_id": accepted_draw_id,
                    "unit_seed": int(unit_seed),
                    "L_j": L,
                    "n_sources_required": len(src_ids),
                    "n_empty_source_operands": 0,
                    "empty_source_ids": "[]",
                }
            )
            accepted_draw_id += 1
            proposal_id += 1
        lo, hi = np.percentile(np.asarray(draws, dtype=float), [2.5, 97.5], method="linear")
        rec = dict(tgt_row)
        rec.update(
            {
                "L_j": L,
                "B_boot": B_BOOT,
                "ci_level": CI_LEVEL,
                "ci_type": "two_sided_percentile",
                "ci_lower": float(lo),
                "ci_upper": float(hi),
                "bootstrap_draw_count": B_BOOT,
                "bootstrap_empty_support_draw_count": 0,
                "bootstrap_proposal_count": proposal_id,
                "bootstrap_rejected_proposal_count": n_rejected,
                "bootstrap_acceptance_rate": B_BOOT / proposal_id if proposal_id else None,
            }
        )
        inf_rows.append(rec)
    validations.append(ValidationRecord("p1_retained_draws", True, str(B_BOOT)))

    jcomp = [r["target_plant_id"] for r in inf_rows]
    D_fleet = float(np.mean([r["Delta_j"] for r in inf_rows]))
    fleet = {
        "method": "F1_TARGET_BALANCED_ARITHMETIC_MEAN",
        "D_fleet_RQ3": D_fleet,
        "n_stage43_supported": len(supported),
        "n_j_comp_structural": len(j_comp_struct),
        "n_computable_targets": len(jcomp),
        "actual_denominator": len(jcomp),
        "j_comp_scientific_ids": jcomp,
        "excluded_target_ids": excluded_targets,
        "excluded_reasons": excluded_reasons,
        "fleet_level_ci": None,
        "fleet_level_p_value": None,
        "descriptive_only": True,
        "superpopulation_claim": False,
    }
    group_rows = []
    for g in sorted(set(group_of[t] for t in jcomp)):
        members_expected = [t for t, grp in group_of.items() if grp == g and t in supported]
        comp = [r for r in inf_rows if r["reference_group_id"] == g]
        group_rows.append({"reference_group_id": g, "n_expected_targets": len(members_expected), "n_computable_targets": len(comp), "included_target_ids": json.dumps([r["target_plant_id"] for r in comp]), "D_g": float(np.mean([r["Delta_j"] for r in comp])) if comp else None, "group_level_ci": None, "group_level_p_value": None})

    hist_inf = _read_csv(root / "artifacts/s08_scientific/S08_TARGET_LEVEL_INFERENCE.csv")
    hist_fleet = json.loads((root / "artifacts/s08_scientific/S08_FLEET_DESCRIPTIVE_SUMMARY.json").read_text(encoding="utf-8"))
    hist_map = {r["target_plant_id"]: r for r in hist_inf}
    old_map = {r["reference_plant_id"]: r["control_plant_id"] for r in (_nested(ctx.sot, "scientific_freeze", "s03", "corrective_reaudit", "control_matching", "selected_mapping") or [])}
    sens_rows = []
    for r in inf_rows:
        tgt = r["target_plant_id"]
        h = hist_map.get(tgt) or {}
        hist_d = float(h["Delta_j"]) if h.get("Delta_j") not in (None, "") else None
        rev_d = float(r["Delta_j"])
        sens_rows.append(
            {
                "target_plant_id": tgt,
                "reference_group_id": r["reference_group_id"],
                "historical_control_source_plant_id": old_map.get(tgt, h.get("matched_control_source_plant_id")),
                "revised_control_source_plant_id": r["matched_control_source_plant_id"],
                "historical_Delta_j": hist_d,
                "revised_Delta_j": rev_d,
                "revised_minus_historical_Delta_j": None if hist_d is None else rev_d - hist_d,
                "historical_ci_lower": h.get("ci_lower"),
                "historical_ci_upper": h.get("ci_upper"),
                "revised_ci_lower": r["ci_lower"],
                "revised_ci_upper": r["ci_upper"],
            }
        )
    hist_D = hist_fleet.get("D_fleet_RQ3")
    def _sign(x: float | None) -> str:
        if x is None:
            return "undefined"
        if x > 0:
            return "positive"
        if x < 0:
            return "negative"
        return "zero"

    facts = {
        "revised_D_fleet_RQ3": D_fleet,
        "revised_D_fleet_RQ3_sign": _sign(D_fleet),
        "revised_denominator": len(jcomp),
        "n_targets_Delta_positive": sum(1 for r in inf_rows if r["Delta_j"] > 0),
        "n_targets_Delta_negative": sum(1 for r in inf_rows if r["Delta_j"] < 0),
        "n_targets_Delta_zero": sum(1 for r in inf_rows if r["Delta_j"] == 0),
        "historical_spec_role": "HISTORICAL_TECHNICAL_ONLY_COMPARATOR_GEOGRAPHY_SENSITIVITY",
        "historical_D_fleet_RQ3": hist_D,
        "historical_denominator": hist_fleet.get("actual_denominator"),
        "historical_D_fleet_RQ3_sign": _sign(None if hist_D is None else float(hist_D)),
        "revised_and_historical_fleet_signs": "same" if _sign(D_fleet) == _sign(None if hist_D is None else float(hist_D)) else "different",
        "revised_specification_primary_regardless_of_direction": True,
        "fleet_level_ci": None,
        "fleet_level_p_value": None,
        "old_vs_new_significance_test": False,
        "supported_vs_unsupported_outcome_test": False,
    }

    hist_after = snapshot_tree(root, PROTECTED)
    if hist_after != hist_before:
        raise S45Stop("S45_STOP_HISTORICAL_MUTATION", str(sorted(set(hist_after) ^ set(hist_before))[:12]))

    _write_pq(out / "S08_STATE_CONTROL_ANALYTIC_INDEX.parquet", analytic_rows, ["control_plant_id", "datetime_raw", "split"])
    write_csv(out / "S08_STATE_CONTROL_HYPERPARAMETER_SELECTION.csv", sel_rows, ["control_plant_id", "model_family", "candidate_id", "hyperparameters", "train_row_count", "validation_row_count", "test_row_count", "validation_mae", "selected", "fit_status"])
    model_manifest = []
    for pid in controls:
        p = models_dir / pid / FAM_SPLINE / "full.joblib"
        model_manifest.append({"control_plant_id": pid, "path": repo_relative(root, p), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    dump_json(out / "S08_STATE_CONTROL_MODEL_MANIFEST.json", {"models": model_manifest, "primary_family": FAM_SPLINE})
    _write_pq(out / "S08_STATE_CONTROL_TRANSFER_PREDICTIONS.parquet", ctrl_pred_rows, ["control_source_plant_id", "target_plant_id", "twin_source_plant_id", "datetime_raw", "y_true", "y_pred_control", "y_pred_local", "y_pred_twin"])
    src_fields = ["reference_group_id", "twin_source_plant_id", "target_plant_id", "control_source_plant_id", "n_pairwise_rows", "n_pairwise_days", "pairwise_retention_vs_target", "twin_transfer_mae", "control_transfer_mae", "local_same_rows_mae", "twin_OTG_abs", "control_OTG_abs", "delta_ij", "computable", "reason"]
    write_csv(out / "S08_STATE_SOURCE_LEVEL_MATCHED_CONTRASTS.csv", source_rows, src_fields)
    tgt_fields = ["target_plant_id", "reference_group_id", "matched_control_source_plant_id", "n_expected_exact_twin_sources", "n_pairwise_computable_exact_twin_sources", "source_ids_included", "Delta_j"]
    write_csv(out / "S08_STATE_TARGET_LEVEL_MATCHED_CONTRASTS.csv", [{k: r[k] for k in tgt_fields} for r in target_rows], tgt_fields)
    write_csv(out / "S08_STATE_BLOCK_LENGTH_SELECTION.csv", block_rows, ["target_plant_id", "model_family", "n_validation_days", "n_validation_rows", "raw_selector_circular", "raw_selector_stationary", "whole_day_block_length", "L_j", "selector"])
    write_csv(
        out / "S08_STATE_BOOTSTRAP_PROPOSAL_AUDIT.csv",
        proposal_rows,
        ["target_plant_id", "proposal_id", "accepted", "accepted_draw_id", "unit_seed", "L_j", "n_sources_required", "n_empty_source_operands", "empty_source_ids"],
    )
    _write_pq(out / "S08_STATE_TARGET_LEVEL_BOOTSTRAP_DRAWS.parquet", boot_rows, ["target_plant_id", "draw_id", "proposal_id", "seed", "L_j", "Delta_j_draw", "source_count_denominator", "empty_pairwise_support"])
    inf_fields = tgt_fields + ["L_j", "B_boot", "ci_level", "ci_type", "ci_lower", "ci_upper", "point_denominator", "bootstrap_draw_count", "bootstrap_empty_support_draw_count", "bootstrap_proposal_count", "bootstrap_rejected_proposal_count", "bootstrap_acceptance_rate"]
    write_csv(out / "S08_STATE_TARGET_LEVEL_INFERENCE.csv", inf_rows, inf_fields)
    dump_json(out / "S08_STATE_FLEET_DESCRIPTIVE_SUMMARY.json", fleet)
    write_csv(out / "S08_STATE_GROUP_DESCRIPTIVE_SUMMARY.csv", group_rows, ["reference_group_id", "n_expected_targets", "n_computable_targets", "included_target_ids", "D_g", "group_level_ci", "group_level_p_value"])
    write_csv(out / "S08_STATE_HISTORICAL_SENSITIVITY.csv", sens_rows, ["target_plant_id", "reference_group_id", "historical_control_source_plant_id", "revised_control_source_plant_id", "historical_Delta_j", "revised_Delta_j", "revised_minus_historical_Delta_j", "historical_ci_lower", "historical_ci_upper", "revised_ci_lower", "revised_ci_upper"])
    dump_json(out / "S08_STATE_INTERPRETATION_FACTS.json", facts)
    protocol = {
        "revision_id": REVISION,
        "support_alignment": "B_PAIRWISE_INTERSECTION",
        "contrast_orientation": "CONTROL_MINUS_TWIN",
        "target_aggregation": "M1_EQUAL_WEIGHT_ARITHMETIC_MEAN",
        "fleet_synthesis": "F1_TARGET_BALANCED_ARITHMETIC_MEAN",
        "primary_model": FAM_SPLINE,
        "support": "CS4_k5_q0.95",
        "B_boot": B_BOOT,
        "ci": "two_sided_percentile_95",
        "block_selector": "arch.bootstrap.optimal_block_length Patton-Politis-White 2009 circular column for MBB",
        "seed_rule": "Operational_Twin_Gap|proposal_v1.3|{estimand}|{model_family}|{unit_id}|{arm_id}",
        "seed_family_token": FAM_TOKEN,
        "no_refit": True,
        "bootstrap_empty_draw_policy": P1_POLICY,
        "bootstrap_policy_freeze_stage": 47,
        "bootstrap_policy_decision_path": "artifacts/s08_state_constrained_bootstrap_freeze/S08_STATE_BOOTSTRAP_POLICY_CONTROLLER_DECISION.json",
        "bootstrap_policy_decision_sha256": P1_DECISION_SHA,
        "B_boot_retained": B_BOOT,
        "proposal_cap": None,
        "rejected_proposals_are_effect_draws": False,
        "stage44_pairwise_mask": repo_relative(root, pair_mask_path),
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }
    dump_json(out / "S08_STATE_PROTOCOL.json", protocol)
    recon = {
        "n_source_contrasts_expected_from_stage44_computable": len(pairwise_items),
        "n_scientifically_computable_source_contrasts": len(source_rows),
        "n_source_exclusions": len(source_exclusions),
        "source_exclusions": source_exclusions,
        "matching_recomputed": False,
        "twin_models_refit": False,
        "masks_recomputed": False,
        "fleet_ci": False,
        "permutation": False,
        "sign_flip": False,
        "s09_rematerialized": False,
        "historical_s08_unchanged": True,
        "delta_algebra_ok": True,
        "empty_bootstrap_draws": 0,
        "bootstrap_empty_draw_policy": P1_POLICY,
        "stage47_decision_sha256": P1_DECISION_SHA,
        "stage48_reconciled": True,
        "bootstrap_total_retained_draws": len(boot_rows),
        "bootstrap_total_proposals": len(proposal_rows),
        "bootstrap_total_rejected_proposals": sum(1 for r in proposal_rows if not _flag(r["accepted"])),
        "retained_empty_support_draws": 0,
        "fixed_source_set_preserved": True,
        "equal_source_weights_preserved": True,
        "shared_calendar_preserved": True,
        "no_source_dropping": True,
        "no_global_intersection": True,
        "no_source_specific_calendars": True,
        "no_model_refit": True,
        "rq1_rq2_unchanged": True,
        "revised_primary_regardless_of_direction": True,
        "historical_snapshot_sha256": hashlib.sha256(json.dumps(hist_before, sort_keys=True).encode()).hexdigest(),
    }
    dump_json(out / "S08_STATE_RECONCILIATION.json", recon)
    report_path = root / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    inf_lines = [
        f"| {r['target_plant_id']} | {r['matched_control_source_plant_id']} | {r['n_pairwise_computable_exact_twin_sources']} | {r['Delta_j']:.6g} | {r['ci_lower']:.6g} | {r['ci_upper']:.6g} | {r['L_j']} |"
        for r in inf_rows
    ]
    report_path.write_text(
        "# Revised same-state RQ3 (Stage 45)\n\n"
        "## A. Frozen specification consumed\n\n"
        f"- active revision `{REVISION}`\n"
        f"- mapping `{repo_relative(root, root / 'artifacts/s03_rq3_state_revision/S03_STATE_SUPPORTED_MAPPING_SELECTED.csv')}`\n"
        f"- Stage 44 pairwise `{repo_relative(root, pair_mask_path)}`\n"
        f"- primary model `{FAM_SPLINE}`; contrast CONTROL_MINUS_TWIN; aggregation M1; fleet F1; bootstrap B={B_BOOT} retained valid draws, P1_CONDITIONAL_VALID_DRAW_RESAMPLING (frozen at Stage 47 before revised RQ3 results existed); two-sided percentile 95%\n"
        f"- rejected proposals={sum(1 for r in proposal_rows if not _flag(r['accepted']))}; total proposals={len(proposal_rows)}\n\n"
        "## B. Scientific flow\n\n"
        f"J_frozen={flow['J_frozen_count']} → J_state_support={flow['J_state_support_count']} → "
        f"J_comp_structural={flow['J_comp_structural_count']} → J_comp_scientific={len(jcomp)}\n"
        f"Excluded from scientific J_comp: {json.dumps(excluded_reasons)}\n\n"
        "## C. Source-level revised RQ3\n\n"
        f"See `S08_STATE_SOURCE_LEVEL_MATCHED_CONTRASTS.csv` (n={len(source_rows)}). Stage 44 noncomputable sources remain excluded: {json.dumps(source_exclusions)}.\n\n"
        "## D. Target-level revised RQ3\n\n"
        "| target | control | n_src | Delta_j | CI_low | CI_up | L_j |\n|---|---|---|---|---|---|---|\n"
        + "\n".join(inf_lines)
        + "\n\n## E. Fleet descriptive result\n\n"
        f"`D_fleet_RQ3` = {D_fleet:.6g}; actual denominator = {len(jcomp)}. No fleet CI. No fleet p-value.\n\n"
        "## F. Historical sensitivity\n\n"
        "Historical technical-only RQ3 is `HISTORICAL_TECHNICAL_ONLY_COMPARATOR_GEOGRAPHY_SENSITIVITY`. "
        f"Historical `D_fleet_RQ3` = {hist_D} (denominator {hist_fleet.get('actual_denominator')}). "
        "The revised same-state specification is primary by prior controller decision, not by result direction.\n\n"
        "## G. Scope of the scientific claim\n\n"
        "Revised RQ3 tests whether nominal equivalence carries additional information about model portability relative to a same-state near-but-non-identical comparator. "
        "Same-state blocking removes the systematic cross-state comparator imbalance at a coarse level. "
        "It does not fully control climate, exact geography, operator, altitude, or microclimate. "
        "OTG reflects the observed operational system, not pure physical plant differences.\n\n"
        "## H. Non-actions\n\n"
        "No rematching, support-mask redesign, new model, new threshold, fleet inference, new state estimand, supported-vs-unsupported outcome test, Stage 49, or S09/S10/paper rematerialization.\n",
        encoding="utf-8",
    )
    recs = [
        _art(root, out / "S08_STATE_CONTROL_ANALYTIC_INDEX.parquet", rows=len(analytic_rows), fp=_pq_fp(out / "S08_STATE_CONTROL_ANALYTIC_INDEX.parquet")),
        _art(root, out / "S08_STATE_CONTROL_HYPERPARAMETER_SELECTION.csv", rows=len(sel_rows), fp=_csv_fp(out / "S08_STATE_CONTROL_HYPERPARAMETER_SELECTION.csv")),
        _art(root, out / "S08_STATE_CONTROL_MODEL_MANIFEST.json"),
        _art(root, out / "S08_STATE_CONTROL_TRANSFER_PREDICTIONS.parquet", rows=len(ctrl_pred_rows), fp=_pq_fp(out / "S08_STATE_CONTROL_TRANSFER_PREDICTIONS.parquet")),
        _art(root, out / "S08_STATE_SOURCE_LEVEL_MATCHED_CONTRASTS.csv", rows=len(source_rows), fp=_csv_fp(out / "S08_STATE_SOURCE_LEVEL_MATCHED_CONTRASTS.csv")),
        _art(root, out / "S08_STATE_TARGET_LEVEL_MATCHED_CONTRASTS.csv", rows=len(target_rows), fp=_csv_fp(out / "S08_STATE_TARGET_LEVEL_MATCHED_CONTRASTS.csv")),
        _art(root, out / "S08_STATE_BLOCK_LENGTH_SELECTION.csv", rows=len(block_rows), fp=_csv_fp(out / "S08_STATE_BLOCK_LENGTH_SELECTION.csv")),
        _art(root, out / "S08_STATE_BOOTSTRAP_PROPOSAL_AUDIT.csv", rows=len(proposal_rows), fp=_csv_fp(out / "S08_STATE_BOOTSTRAP_PROPOSAL_AUDIT.csv")),
        _art(root, out / "S08_STATE_TARGET_LEVEL_BOOTSTRAP_DRAWS.parquet", rows=len(boot_rows), fp=_pq_fp(out / "S08_STATE_TARGET_LEVEL_BOOTSTRAP_DRAWS.parquet")),
        _art(root, out / "S08_STATE_TARGET_LEVEL_INFERENCE.csv", rows=len(inf_rows), fp=_csv_fp(out / "S08_STATE_TARGET_LEVEL_INFERENCE.csv")),
        _art(root, out / "S08_STATE_FLEET_DESCRIPTIVE_SUMMARY.json"),
        _art(root, out / "S08_STATE_GROUP_DESCRIPTIVE_SUMMARY.csv", rows=len(group_rows), fp=_csv_fp(out / "S08_STATE_GROUP_DESCRIPTIVE_SUMMARY.csv")),
        _art(root, out / "S08_STATE_HISTORICAL_SENSITIVITY.csv", rows=len(sens_rows), fp=_csv_fp(out / "S08_STATE_HISTORICAL_SENSITIVITY.csv")),
        _art(root, out / "S08_STATE_INTERPRETATION_FACTS.json"),
        _art(root, out / "S08_STATE_PROTOCOL.json"),
        _art(root, out / "S08_STATE_RECONCILIATION.json"),
        _art(root, report_path),
    ]
    dump_json(out / "S08_STATE_MANIFEST.json", {"job": JOB, "run_id": ctx.run_id, "artifacts": [_tab(r) for r in recs]})
    recs.append(_art(root, out / "S08_STATE_MANIFEST.json"))
    named = {
        "control_index": recs[0],
        "control_hyper": recs[1],
        "control_models": recs[2],
        "control_preds": recs[3],
        "source_contrasts": recs[4],
        "target_contrasts": recs[5],
        "block_length": recs[6],
        "bootstrap_proposal_audit": recs[7],
        "bootstrap_draws": recs[8],
        "inference": recs[9],
        "fleet": recs[10],
        "groups": recs[11],
        "historical_sensitivity": recs[12],
        "interpretation_facts": recs[13],
        "protocol": recs[14],
        "reconciliation": recs[15],
        "report": recs[16],
        "manifest": recs[17],
        "stage44_pairwise_masks_reference": {"path": repo_relative(root, pair_mask_path), "sha256": sha256_file(pair_mask_path), "row_count": pair_mask_rec.get("row_count"), "schema_fingerprint": pair_mask_rec.get("schema_fingerprint")},
    }
    arts_sot = {k: (_tab(v) if isinstance(v, ArtifactRecord) else v) for k, v in named.items()}
    patch = {
        "stages": {
            "S08_STATE_CONSTRAINED_SCIENTIFIC_RQ3": {
                "kind": "scientific_analysis_revision",
                "status": "GO",
                "reason": "S45_REVISED_SAME_STATE_RQ3_EXECUTED",
                "operational_stage_id": "45",
                "job": JOB,
                "run_id": ctx.run_id,
                "finished_at_utc": now,
                "answer": ANSWER,
                "report": REPORT,
                "revision_id": REVISION,
                "artifacts": arts_sot,
            }
        },
        "scientific_analysis": {
            "s08_active_revision": REVISION,
            "s08_revisions": {
                REVISION: {
                    "scientific": {
                        "status": "GO",
                        "reason": "S45_REVISED_SAME_STATE_RQ3_EXECUTED",
                        "revision_id": REVISION,
                        "rq": "RQ3",
                        "primary_for_rq3": True,
                        "scientific_results_computed": True,
                        "primary_model_family": FAM_SPLINE,
                        "primary_support": {"id": "CS4", "k": 5, "q": 0.95},
                        "support_alignment": "B_PAIRWISE_INTERSECTION",
                        "contrast_orientation": "CONTROL_MINUS_TWIN",
                        "target_aggregation": "M1_EQUAL_WEIGHT_ARITHMETIC_MEAN",
                        "fleet_synthesis": "F1_TARGET_BALANCED_ARITHMETIC_MEAN",
                        "target_level_inference": {"method": "temporal_MBB", "B_boot": B_BOOT, "ci": "two_sided_percentile_95", "p_value": False},
                        "fleet_level_inference": {"ci": None, "p_value": None, "descriptive_only": True},
                        "j_frozen_count": flow["J_frozen_count"],
                        "j_state_support_count": flow["J_state_support_count"],
                        "j_comp_structural_count": flow["J_comp_structural_count"],
                        "j_comp_scientific_count": len(jcomp),
                        "j_comp_scientific_ids": jcomp,
                        "excluded_targets": excluded_targets,
                        "excluded_reasons": excluded_reasons,
                        "n_expected_source_contrasts": len(audit_rows),
                        "n_scientifically_computable_source_contrasts": len(source_rows),
                        "source_level_exclusions": source_exclusions,
                        "revised_controls": controls,
                        "D_fleet_RQ3": D_fleet,
                        "fleet_actual_denominator": len(jcomp),
                        "fleet_level_ci": None,
                        "fleet_level_p_value": None,
                        "descriptive_only": True,
                        "historical_spec_role": "HISTORICAL_TECHNICAL_ONLY_COMPARATOR_GEOGRAPHY_SENSITIVITY",
                        "historical_fleet_scalar_reference": {"D_fleet_RQ3": hist_D, "actual_denominator": hist_fleet.get("actual_denominator")},
                        "historical_sensitivity": arts_sot["historical_sensitivity"],
                        "artifacts": arts_sot,
                        "report": REPORT,
                        "bootstrap_empty_draw_policy": P1_POLICY,
                        "bootstrap_policy_freeze_stage": "47",
                        "bootstrap_policy_decision": {"path": "artifacts/s08_state_constrained_bootstrap_freeze/S08_STATE_BOOTSTRAP_POLICY_CONTROLLER_DECISION.json", "sha256": P1_DECISION_SHA},
                        "bootstrap_total_retained_draws": len(boot_rows),
                        "bootstrap_total_proposals": len(proposal_rows),
                        "bootstrap_total_rejected_proposals": sum(1 for r in proposal_rows if not _flag(r["accepted"])),
                        "next_stage": "49",
                    }
                }
            },
        },
    }
    validations.append(ValidationRecord("no_sot_write", True, "sot_patch only"))
    validations.append(ValidationRecord("historical_s08_untouched", True, None))
    return StageResult(status="GO", message=MSG_GO, sot_patch=patch, artifacts=recs, validations=validations)
