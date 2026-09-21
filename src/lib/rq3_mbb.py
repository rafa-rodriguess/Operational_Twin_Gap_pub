"""Frozen-spec S08 scientific execution helpers. Imported by stage_25; no independent run()."""

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
import pyarrow as pa
import pyarrow.parquet as pq
from arch.bootstrap import optimal_block_length

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import FEATURE_INTERP, assign_post_eligibility_split, dump_json, parse_ts, sha256_file, write_csv
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

ANSWER = None
OUT = "artifacts/s08_scientific"
REPORT = "reports/scientific/S08_TWIN_VS_MATCHED_NON_TWIN_REEXECUTION.md"
REPORT_HIST = "reports/scientific/S08_TWIN_VS_MATCHED_NON_TWIN.md"
JOB = "S08_SCIENTIFIC_REEXECUTION_AFTER_SPEC_FREEZE"
MSG_GO = "GO — S08 RQ3 TWIN VS MATCHED NON-TWIN EXECUTED UNDER FROZEN SPECIFICATION"
B_BOOT = 5000
CI_LEVEL = 0.95
FAM_TOKEN = "ridge_spline"
READY = "FULLY_SPECIFIED_READY_FOR_S08_SCIENTIFIC_REEXECUTION"
class S08ExecStop(Exception):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


def _norm_dt(value: Any) -> str:
    ts = parse_ts(str(value))
    if ts is None:
        return str(value)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _art(root: Path, path: Path, rows: int | None = None, fp: str | None = None) -> ArtifactRecord:
    payload = path.read_bytes()
    return ArtifactRecord(path=_rel(root, path), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), row_count=rows, schema_fingerprint=fp)


def _tab(rec: ArtifactRecord) -> dict[str, Any]:
    out: dict[str, Any] = {"path": rec.path, "sha256": rec.sha256, "bytes": rec.bytes}
    if rec.row_count is not None:
        out["row_count"] = rec.row_count
    if rec.schema_fingerprint is not None:
        out["schema_fingerprint"] = rec.schema_fingerprint
    return out


def _csv_fp(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    return hashlib.sha256("|".join(header).encode("utf-8")).hexdigest()


def _pq_fp(path: Path) -> str:
    return hashlib.sha256("|".join(pq.read_schema(path).names).encode("utf-8")).hexdigest()


def _check(root: Path, rec: dict[str, Any] | None, fallback: str) -> Path:
    path = root / ((rec or {}).get("path") or fallback)
    if not path.is_file():
        raise S08ExecStop("S08_STOP_MISSING_UPSTREAM_ARTIFACT", f"STOP — missing {fallback}")
    expected = (rec or {}).get("sha256")
    if expected and sha256_file(path) != expected:
        raise S08ExecStop("S08_STOP_HASH_MISMATCH", f"STOP — hash mismatch {path.as_posix()}")
    return path


def _day(dt: str) -> str:
    ts = parse_ts(dt)
    return ts.date().isoformat() if ts else str(dt)[:10]


def mbb_calendar_draw(days: list[str], L: int, rng: np.random.Generator) -> list[str]:
    n = len(days)
    if n == 0:
        return []
    L = max(1, min(int(L), n))
    n_starts = max(1, n - L + 1)
    out: list[int] = []
    while len(out) < n:
        start = int(rng.integers(0, n_starts))
        out.extend(range(start, start + L))
    return [days[i] for i in out[:n]]


def empty_pairwise_support_in_draw(sampled_days: set[str], pair_days: set[str]) -> bool:
    return sampled_days.isdisjoint(pair_days)


def _write_pq(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for f in fields:
        vals = [r.get(f) for r in rows]
        if f in {"pairwise_supported", "empty_pairwise_support", "computable"}:
            arrays[f] = pa.array([bool(v) if v is not None else False for v in vals], type=pa.bool_())
        elif f in {"draw_id", "proposal_id", "n_pairwise_rows", "n_pairwise_days", "L_j", "B_boot", "seed", "source_count_denominator", "bootstrap_draw_count", "bootstrap_empty_support_draw_count", "point_denominator", "n_expected_exact_twin_sources", "n_pairwise_computable_exact_twin_sources"}:
            arrays[f] = pa.array([None if v is None else int(v) for v in vals], type=pa.int64())
        elif f in {"y_true", "y_pred_control", "y_pred_local", "y_pred_twin", "Delta_j_draw", "ci_level"}:
            arrays[f] = pa.array([None if v is None else float(v) for v in vals], type=pa.float64())
        else:
            arrays[f] = pa.array([None if v is None else str(v) for v in vals], type=pa.string())
    pq.write_table(pa.table(arrays), path)


def _stop_result(now: str, validations: list[ValidationRecord], reason: str, message: str) -> StageResult:
    return StageResult(
        status="STOP",
        message=message,
        sot_patch={
            "stages": {
                "S08": {
                    "kind": "scientific_analysis",
                    "status": "STOP",
                    "reason": reason,
                    "operational_stage_id": "25",
                    "job": JOB,
                    "finished_at_utc": now,
                    "answer": ANSWER,
                }
            },
            "scientific_analysis": {
                "s08": {
                    "status": "STOP",
                    "operational_stage_id": "25",
                    "rq": "RQ3",
                    "reexecution_after_specification_freeze": True,
                    "reason": reason,
                }
            },
        },
        artifacts=[],
        validations=validations,
    )


def execute_scientific(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    try:
        return _execute_scientific_body(ctx, now, validations)
    except S08ExecStop as exc:
        return _stop_result(now, validations, exc.reason, exc.message)
    except TieError as exc:
        return _stop_result(now, validations, "S08_STOP_VALIDATION_MAE_TIE", str(exc))


def _execute_scientific_body(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    root = ctx.root
    sot = ctx.sot
    freeze = sot.get("scientific_freeze") or {}
    op = freeze.get("s08_rq3_operationalization") or {}
    sa = sot.get("scientific_analysis") or {}
    if (op.get("support_alignment") or {}).get("method") != "B_PAIRWISE_INTERSECTION":
        raise S08ExecStop("S08_STOP_SUPPORT_ALIGNMENT_NOT_B", "STOP — support alignment is not B_PAIRWISE_INTERSECTION")
    if (op.get("target_level_aggregation") or {}).get("method") != "M1_EQUAL_WEIGHT_ARITHMETIC_MEAN":
        raise S08ExecStop("S08_STOP_AGGREGATION_NOT_M1", "STOP — target aggregation is not M1")
    if (op.get("contrast_orientation") or {}).get("orientation") != "CONTROL_MINUS_TWIN":
        raise S08ExecStop("S08_STOP_ORIENTATION_NOT_CONTROL_MINUS_TWIN", "STOP — contrast orientation is not CONTROL_MINUS_TWIN")
    if (op.get("fleet_level_synthesis") or {}).get("method") != "F1_TARGET_BALANCED_ARITHMETIC_MEAN":
        raise S08ExecStop("S08_STOP_FLEET_SYNTHESIS_NOT_F1", "STOP — fleet synthesis is not F1")
    comp_rec = ((op.get("specification_readiness") or {}).get("completeness") or {})
    comp_path = _check(root, comp_rec, "artifacts/s08_fleet_synthesis_freeze/S08_RQ3_SPECIFICATION_COMPLETENESS.json")
    completeness = json.loads(comp_path.read_text(encoding="utf-8"))
    if completeness.get("missing") != [] or completeness.get("scientific_results_computed") is not False or completeness.get("s09_authorized") is not False:
        raise S08ExecStop("S08_STOP_STAGE31_COMPLETENESS", "STOP — Stage 31 completeness does not reconcile")
    if not all((v or {}).get("frozen") for v in (completeness.get("items") or {}).values()):
        raise S08ExecStop("S08_STOP_STAGE31_ITEM_NOT_FROZEN", "STOP — Stage 31 completeness item not frozen")
    if seed_uint32("RQ3", FAM_TOKEN, "PS_002", "L_j") != 1676118836:
        raise S08ExecStop("S08_STOP_SEED_EXAMPLE_MISMATCH", "STOP — frozen seed example 1676118836 did not reproduce")
    validations.append(ValidationRecord("stage31_readiness", True, None))
    validations.append(ValidationRecord("seed_example", True, "1676118836"))

    hist = {
        "preflight": root / "artifacts/s08/S08_SPECIFICATION_PREFLIGHT.json",
        "reconciliation": root / "artifacts/s08/S08_RECONCILIATION.json",
        "manifest": root / "artifacts/s08/S08_MANIFEST.json",
        "report": root / REPORT_HIST,
    }
    hist_bytes = {k: p.read_bytes() for k, p in hist.items()}

    scope = json.loads((root / "artifacts/s03_finalization/S03_FINAL_SCOPE.json").read_text(encoding="utf-8"))
    mapping = list(scope["matched_control_mapping"])
    control_of = {r["reference_plant_id"]: r["control_plant_id"] for r in mapping}
    controls = sorted(set(control_of.values()))
    if controls != ["PS_006", "PS_010", "PS_013"]:
        raise S08ExecStop("S08_STOP_CONTROL_MAPPING", f"STOP — unexpected controls {controls}")
    group_of: dict[str, str] = {}
    twins_of: dict[str, list[str]] = {}
    with (root / "artifacts/s03_inference_dependency_audit/S03_RQ3_CONTRAST_INCIDENCE.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            group_of[row["target_plant_id"]] = row["reference_group_id"]
            twins_of[row["target_plant_id"]] = json.loads(row["twin_source_plant_ids"])
    targets = sorted(control_of)

    s04 = sa.get("s04") or {}
    s05 = sa.get("s05") or {}
    s06 = sa.get("s06") or {}
    s07 = sa.get("s07") or {}
    for rec, fb in (
        (s04.get("local_test_predictions"), "artifacts/s04/S04_LOCAL_TEST_PREDICTIONS.parquet"),
        (s05.get("support_masks"), "artifacts/s05/S05_SUPPORT_MASKS.parquet"),
        (s06.get("predictions") or s06.get("cross_plant_predictions"), "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet"),
        (s07.get("directional"), "artifacts/s07/S07_OTG_DIRECTIONAL.csv"),
    ):
        _check(root, rec if rec else {"path": fb}, fb)

    align = ((freeze.get("s08_specification_audits") or {}).get("common_support_alignment") or {})
    ctrl_mask_path = _check(root, align.get("control_cs4_masks"), "artifacts/s08_alignment_audit/S08_CONTROL_CS4_SUPPORT_MASKS.parquet")
    pair_audit_path = _check(root, align.get("pairwise_audit"), "artifacts/s08_alignment_audit/S08_PAIRWISE_ALIGNMENT_AUDIT.csv")
    s05_masks = _check(root, s05.get("support_masks"), "artifacts/s05/S05_SUPPORT_MASKS.parquet")
    s06_pred = _check(root, s06.get("predictions") or s06.get("cross_plant_predictions"), "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet")
    s04_test = _check(root, s04.get("local_test_predictions"), "artifacts/s04/S04_LOCAL_TEST_PREDICTIONS.parquet")
    s04_val = _check(root, s04.get("validation_predictions"), "artifacts/s04/S04_VALIDATION_PREDICTIONS.parquet")
    panel_path = _check(root, (freeze.get("s02") or {}).get("canonical_panel"), "artifacts/p02c/P02C_PLANT_PANEL.parquet")

    twin_sup: dict[tuple[str, str], set[str]] = defaultdict(set)
    t5 = pq.read_table(s05_masks, columns=["source_plant_id", "target_plant_id", "rule_id", "datetime_raw", "supported"])
    for src, tgt, rid, dt, ok in zip(t5["source_plant_id"].to_pylist(), t5["target_plant_id"].to_pylist(), t5["rule_id"].to_pylist(), t5["datetime_raw"].to_pylist(), t5["supported"].to_pylist()):
        if rid == "CS4" and ok:
            twin_sup[(str(src), str(tgt))].add(_norm_dt(dt))
    ctrl_sup: dict[tuple[str, str], set[str]] = defaultdict(set)
    tc = pq.read_table(ctrl_mask_path, columns=["control_source_plant_id", "target_plant_id", "rule_id", "datetime_raw", "supported"])
    for cid, tgt, rid, dt, ok in zip(tc["control_source_plant_id"].to_pylist(), tc["target_plant_id"].to_pylist(), tc["rule_id"].to_pylist(), tc["datetime_raw"].to_pylist(), tc["supported"].to_pylist()):
        if rid == "CS4" and ok:
            ctrl_sup[(str(cid), str(tgt))].add(_norm_dt(dt))

    pairwise: list[dict[str, Any]] = []
    mask_rows = []
    for tgt in targets:
        cid = control_of[tgt]
        for twin in twins_of[tgt]:
            inter = twin_sup[(twin, tgt)] & ctrl_sup[(cid, tgt)]
            if not inter:
                continue
            pairwise.append({"target": tgt, "twin": twin, "control": cid, "group": group_of[tgt], "dts": inter})
            for dt in sorted(inter):
                mask_rows.append({"target_plant_id": tgt, "twin_source_plant_id": twin, "control_source_plant_id": cid, "datetime_raw": dt, "pairwise_supported": True, "support_rule": "CS4", "alignment": "B_PAIRWISE_INTERSECTION"})
    if len(pairwise) != 85:
        raise S08ExecStop("S08_STOP_PAIRWISE_COUNT", f"STOP — pairwise count {len(pairwise)} != 85")
    validations.append(ValidationRecord("pairwise_85", True, None))

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
            raise S08ExecStop("S08_STOP_EMPTY_CONTROL_PARTITION", f"STOP — empty control partition {pid}")
        for k, lab in enumerate(labels):
            analytic_rows.append({"control_plant_id": pid, "datetime_raw": local_dt[k], "split": lab or None})
        Xtr = np.column_stack([feat_all[f][ix[train_k]] for f in FEATURES])
        ytr = y_all[ix[train_k]]
        Xva = np.column_stack([feat_all[f][ix[val_k]] for f in FEATURES])
        yva = y_all[ix[val_k]]
        params, mae, est, cand = _fit_select(FAM_SPLINE, SPLINE_GRID, Xtr, ytr, Xva, yva)
        cid_sel = [r["candidate_id"] for r in cand if r["fit_status"] == "ok" and r["validation_mae"] == mae][0]
        for r in cand:
            sel_rows.append({"control_plant_id": pid, "model_family": FAM_SPLINE, "candidate_id": r["candidate_id"], "hyperparameters": json.dumps(r["params"], sort_keys=True), "train_row_count": len(train_k), "validation_row_count": len(val_k), "test_row_count": len(test_k), "validation_mae": r["validation_mae"], "selected": r["candidate_id"] == cid_sel, "fit_status": r["fit_status"]})
        dest = models_dir / pid / FAM_SPLINE
        dest.mkdir(parents=True, exist_ok=True)
        joblib.dump(est, dest / "full.joblib")
        models[pid] = est
    validations.append(ValidationRecord("control_models_train_only", True, None))

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
    for item in pairwise:
        tgt, twin, cid = item["target"], item["twin"], item["control"]
        dts = sorted(item["dts"])
        missing = [dt for dt in dts if (twin, tgt, dt) not in twin_pred or (tgt, dt) not in local_pred or (tgt, dt) not in feat_by]
        if missing:
            raise S08ExecStop("S08_STOP_MISSING_PREDICTION_OR_FEATURE", f"STOP — missing twin/local/feature {twin}->{tgt} {missing[0]}")
        X = np.stack([feat_by[(tgt, dt)] for dt in dts])
        yhat = np.asarray(models[cid].predict(X), dtype=float)
        e_twin, e_loc, e_ctrl = [], [], []
        by_day_ctrl: dict[str, list[float]] = defaultdict(list)
        by_day_twin: dict[str, list[float]] = defaultdict(list)
        for dt, y_c in zip(dts, yhat):
            y_true, y_tw, y_loc = twin_pred[(twin, tgt, dt)]
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
            raise S08ExecStop("S08_STOP_DELTA_ALGEBRA", "STOP — delta algebra failed")
        days = {_day(d) for d in dts}
        row = {
            "reference_group_id": item["group"],
            "twin_source_plant_id": twin,
            "target_plant_id": tgt,
            "control_source_plant_id": cid,
            "n_pairwise_rows": len(dts),
            "n_pairwise_days": len(days),
            "pairwise_retention_vs_target": float("nan"),
            "twin_transfer_mae": mae_t,
            "control_transfer_mae": mae_c,
            "local_same_rows_mae": mae_l,
            "twin_OTG_abs": otg_t,
            "control_OTG_abs": otg_c,
            "delta_ij": delta,
            "computable": True,
            "reason": "",
        }
        source_rows.append(row)
        source_payload[(twin, tgt)] = {
            "dts": dts,
            "days": days,
            "by_day_ctrl": {k: np.asarray(v, dtype=float) for k, v in by_day_ctrl.items()},
            "by_day_twin": {k: np.asarray(v, dtype=float) for k, v in by_day_twin.items()},
        }
    with pair_audit_path.open(encoding="utf-8", newline="") as handle:
        audit = {(r["twin_source_plant_id"], r["target_plant_id"]): r for r in csv.DictReader(handle)}
    for row in source_rows:
        ar = audit.get((row["twin_source_plant_id"], row["target_plant_id"]))
        if ar:
            row["pairwise_retention_vs_target"] = float(ar["pairwise_retention_vs_target"])
            if int(ar["n_pairwise_intersection"]) != row["n_pairwise_rows"]:
                raise S08ExecStop("S08_STOP_PAIRWISE_N_MISMATCH", "STOP — pairwise n mismatch vs Stage 26")

    by_tgt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_tgt[row["target_plant_id"]].append(row)
    target_rows = []
    for tgt in targets:
        rows = by_tgt[tgt]
        if not rows:
            raise S08ExecStop("S08_STOP_TARGET_NO_COMPUTABLE_SOURCES", f"STOP — no computable sources for {tgt}")
        Delta = float(np.mean([r["delta_ij"] for r in rows]))
        target_rows.append({
            "target_plant_id": tgt,
            "reference_group_id": group_of[tgt],
            "matched_control_source_plant_id": control_of[tgt],
            "n_expected_exact_twin_sources": len(twins_of[tgt]),
            "n_pairwise_computable_exact_twin_sources": len(rows),
            "source_ids_included": json.dumps(sorted(r["twin_source_plant_id"] for r in rows)),
            "Delta_j": Delta,
            "point_denominator": len(rows),
        })

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
    for tgt in targets:
        Ls = []
        for fam in (FAM_SPLINE, FAM_HGB):
            daily = val_ae[(tgt, fam)]
            if len(daily) < 8:
                raise S08ExecStop("S08_STOP_BLOCK_LENGTH_SELECTOR_IMPLEMENTATION_UNRESOLVED", "STOP — block-length selector series too short")
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
    inf_rows = []
    for tgt_row in target_rows:
        tgt = tgt_row["target_plant_id"]
        srcs = by_tgt[tgt]
        all_days = sorted({d for r in srcs for d in source_payload[(r["twin_source_plant_id"], tgt)]["days"]})
        L = L_of[tgt]
        draws = []
        unit_seed = seed_uint32("RQ3", FAM_TOKEN, tgt, "L_j")
        pays = [source_payload[(r["twin_source_plant_id"], tgt)] for r in srcs]
        for b in range(B_BOOT):
            rng = np.random.default_rng(np.random.SeedSequence([unit_seed, b]))
            sampled = mbb_calendar_draw(all_days, L, rng)
            sampled_set = set(sampled)
            deltas_b = []
            for pay in pays:
                if empty_pairwise_support_in_draw(sampled_set, pay["days"]):
                    raise S08ExecStop("S08_STOP_BOOTSTRAP_DRAW_EMPTY_PAIRWISE_SUPPORT_UNSPECIFIED", f"STOP — empty pairwise support in bootstrap target={tgt} draw={b}")
                parts_c = []
                parts_t = []
                for day in sampled:
                    if day in pay["by_day_ctrl"]:
                        parts_c.append(pay["by_day_ctrl"][day])
                        parts_t.append(pay["by_day_twin"][day])
                if not parts_c:
                    raise S08ExecStop("S08_STOP_BOOTSTRAP_DRAW_EMPTY_PAIRWISE_SUPPORT_UNSPECIFIED", f"STOP — empty pairwise support in bootstrap target={tgt} draw={b}")
                mae_c = float(np.concatenate(parts_c).mean())
                mae_t = float(np.concatenate(parts_t).mean())
                deltas_b.append(mae_c - mae_t)
            Delta_b = float(np.mean(deltas_b))
            draws.append(Delta_b)
            boot_rows.append({"target_plant_id": tgt, "draw_id": b, "seed": int(unit_seed), "L_j": L, "Delta_j_draw": Delta_b, "source_count_denominator": len(srcs), "empty_pairwise_support": False})
        lo, hi = np.percentile(np.asarray(draws, dtype=float), [2.5, 97.5], method="linear")
        rec = dict(tgt_row)
        rec.update({"L_j": L, "B_boot": B_BOOT, "ci_level": CI_LEVEL, "ci_type": "two_sided_percentile", "ci_lower": float(lo), "ci_upper": float(hi), "bootstrap_draw_count": B_BOOT, "bootstrap_empty_support_draw_count": 0})
        inf_rows.append(rec)
    validations.append(ValidationRecord("no_empty_bootstrap_support", True, None))

    jcomp = [r for r in inf_rows if r["n_pairwise_computable_exact_twin_sources"] > 0]
    D_fleet = float(np.mean([r["Delta_j"] for r in jcomp])) if jcomp else None
    fleet = {
        "method": "F1_TARGET_BALANCED_ARITHMETIC_MEAN",
        "D_fleet_RQ3": D_fleet,
        "n_expected_targets": len(targets),
        "n_computable_targets": len(jcomp),
        "actual_denominator": len(jcomp),
        "excluded_target_ids": [],
        "excluded_reasons": {},
        "fleet_level_ci": None,
        "fleet_level_p_value": None,
        "descriptive_only": True,
        "superpopulation_claim": False,
    }
    group_rows = []
    for g in sorted(set(group_of.values())):
        members = [r for r in inf_rows if r["reference_group_id"] == g]
        comp = [r for r in members if r["n_pairwise_computable_exact_twin_sources"] > 0]
        group_rows.append({"reference_group_id": g, "n_expected_targets": len(members), "n_computable_targets": len(comp), "included_target_ids": json.dumps([r["target_plant_id"] for r in comp]), "D_g": float(np.mean([r["Delta_j"] for r in comp])) if comp else None, "group_level_ci": None, "group_level_p_value": None})

    _write_pq(out / "S08_CONTROL_ANALYTIC_INDEX.parquet", analytic_rows, ["control_plant_id", "datetime_raw", "split"])
    write_csv(out / "S08_CONTROL_HYPERPARAMETER_SELECTION.csv", sel_rows, ["control_plant_id", "model_family", "candidate_id", "hyperparameters", "train_row_count", "validation_row_count", "test_row_count", "validation_mae", "selected", "fit_status"])
    model_manifest = []
    for pid in controls:
        p = models_dir / pid / FAM_SPLINE / "full.joblib"
        model_manifest.append({"control_plant_id": pid, "path": _rel(root, p), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    dump_json(out / "S08_CONTROL_MODEL_MANIFEST.json", {"models": model_manifest})
    _write_pq(out / "S08_PAIRWISE_SUPPORT_MASKS.parquet", mask_rows, ["target_plant_id", "twin_source_plant_id", "control_source_plant_id", "datetime_raw", "pairwise_supported", "support_rule", "alignment"])
    _write_pq(out / "S08_CONTROL_TRANSFER_PREDICTIONS.parquet", ctrl_pred_rows, ["control_source_plant_id", "target_plant_id", "twin_source_plant_id", "datetime_raw", "y_true", "y_pred_control", "y_pred_local", "y_pred_twin"])
    src_fields = ["reference_group_id", "twin_source_plant_id", "target_plant_id", "control_source_plant_id", "n_pairwise_rows", "n_pairwise_days", "pairwise_retention_vs_target", "twin_transfer_mae", "control_transfer_mae", "local_same_rows_mae", "twin_OTG_abs", "control_OTG_abs", "delta_ij", "computable", "reason"]
    write_csv(out / "S08_SOURCE_LEVEL_MATCHED_CONTRASTS.csv", source_rows, src_fields)
    tgt_fields = ["target_plant_id", "reference_group_id", "matched_control_source_plant_id", "n_expected_exact_twin_sources", "n_pairwise_computable_exact_twin_sources", "source_ids_included", "Delta_j"]
    write_csv(out / "S08_TARGET_LEVEL_MATCHED_CONTRASTS.csv", [{k: r[k] for k in tgt_fields} for r in target_rows], tgt_fields)
    write_csv(out / "S08_BLOCK_LENGTH_SELECTION.csv", block_rows, ["target_plant_id", "model_family", "n_validation_days", "n_validation_rows", "raw_selector_circular", "raw_selector_stationary", "whole_day_block_length", "L_j", "selector"])
    _write_pq(out / "S08_TARGET_LEVEL_BOOTSTRAP_DRAWS.parquet", boot_rows, ["target_plant_id", "draw_id", "seed", "L_j", "Delta_j_draw", "source_count_denominator", "empty_pairwise_support"])
    inf_fields = tgt_fields + ["L_j", "B_boot", "ci_level", "ci_type", "ci_lower", "ci_upper", "point_denominator", "bootstrap_draw_count", "bootstrap_empty_support_draw_count"]
    write_csv(out / "S08_TARGET_LEVEL_INFERENCE.csv", inf_rows, inf_fields)
    dump_json(out / "S08_FLEET_DESCRIPTIVE_SUMMARY.json", fleet)
    write_csv(out / "S08_GROUP_DESCRIPTIVE_SUMMARY.csv", group_rows, ["reference_group_id", "n_expected_targets", "n_computable_targets", "included_target_ids", "D_g", "group_level_ci", "group_level_p_value"])
    protocol = {
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
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }
    dump_json(out / "S08_PROTOCOL.json", protocol)
    recon = {
        "pairwise_n": len(pairwise),
        "n_targets": len(targets),
        "n_source_contrasts": len(source_rows),
        "matching_recomputed": False,
        "twin_models_refit": False,
        "fleet_ci": False,
        "permutation": False,
        "sign_flip": False,
        "s09_executed": False,
        "historical_s08_bytes_unchanged": True,
        "delta_algebra_ok": True,
        "empty_bootstrap_draws": 0,
    }
    dump_json(out / "S08_RECONCILIATION.json", recon)
    for k, p in hist.items():
        if p.read_bytes() != hist_bytes[k]:
            raise S08ExecStop("S08_STOP_HISTORICAL_S08_MUTATED", f"STOP — historical {k} mutated")
    report_path = root / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_sci_report(inf_rows, fleet, group_rows, source_rows), encoding="utf-8")
    recs = [
        _art(root, out / "S08_CONTROL_ANALYTIC_INDEX.parquet", rows=len(analytic_rows), fp=_pq_fp(out / "S08_CONTROL_ANALYTIC_INDEX.parquet")),
        _art(root, out / "S08_CONTROL_HYPERPARAMETER_SELECTION.csv", rows=len(sel_rows), fp=_csv_fp(out / "S08_CONTROL_HYPERPARAMETER_SELECTION.csv")),
        _art(root, out / "S08_CONTROL_MODEL_MANIFEST.json"),
        _art(root, out / "S08_PAIRWISE_SUPPORT_MASKS.parquet", rows=len(mask_rows), fp=_pq_fp(out / "S08_PAIRWISE_SUPPORT_MASKS.parquet")),
        _art(root, out / "S08_CONTROL_TRANSFER_PREDICTIONS.parquet", rows=len(ctrl_pred_rows), fp=_pq_fp(out / "S08_CONTROL_TRANSFER_PREDICTIONS.parquet")),
        _art(root, out / "S08_SOURCE_LEVEL_MATCHED_CONTRASTS.csv", rows=len(source_rows), fp=_csv_fp(out / "S08_SOURCE_LEVEL_MATCHED_CONTRASTS.csv")),
        _art(root, out / "S08_TARGET_LEVEL_MATCHED_CONTRASTS.csv", rows=len(target_rows), fp=_csv_fp(out / "S08_TARGET_LEVEL_MATCHED_CONTRASTS.csv")),
        _art(root, out / "S08_BLOCK_LENGTH_SELECTION.csv", rows=len(block_rows), fp=_csv_fp(out / "S08_BLOCK_LENGTH_SELECTION.csv")),
        _art(root, out / "S08_TARGET_LEVEL_BOOTSTRAP_DRAWS.parquet", rows=len(boot_rows), fp=_pq_fp(out / "S08_TARGET_LEVEL_BOOTSTRAP_DRAWS.parquet")),
        _art(root, out / "S08_TARGET_LEVEL_INFERENCE.csv", rows=len(inf_rows), fp=_csv_fp(out / "S08_TARGET_LEVEL_INFERENCE.csv")),
        _art(root, out / "S08_FLEET_DESCRIPTIVE_SUMMARY.json"),
        _art(root, out / "S08_GROUP_DESCRIPTIVE_SUMMARY.csv", rows=len(group_rows), fp=_csv_fp(out / "S08_GROUP_DESCRIPTIVE_SUMMARY.csv")),
        _art(root, out / "S08_PROTOCOL.json"),
        _art(root, out / "S08_RECONCILIATION.json"),
        _art(root, report_path),
    ]
    dump_json(out / "S08_MANIFEST.json", {"job": JOB, "artifacts": [_tab(r) for r in recs]})
    recs.append(_art(root, out / "S08_MANIFEST.json"))
    prior = {k: {"path": _rel(root, p), "sha256": hashlib.sha256(hist_bytes[k]).hexdigest(), "bytes": len(hist_bytes[k])} for k, p in hist.items()}
    names = ["control_index", "control_hyper", "control_models", "pairwise_masks", "control_preds", "source_contrasts", "target_contrasts", "block_length", "bootstrap_draws", "inference", "fleet", "groups", "protocol", "reconciliation", "report", "manifest"]
    arts_sot = {names[i]: _tab(recs[i]) for i in range(len(names))}
    patch = {
        "stages": {
            "S08": {
                "kind": "scientific_analysis",
                "status": "GO",
                "reason": "S08_RQ3_EXECUTED_FROZEN_SPECIFICATION",
                "operational_stage_id": "25",
                "job": JOB,
                "finished_at_utc": now,
                "answer": ANSWER,
                "report": _tab(recs[14]),
            }
        },
        "scientific_analysis": {
            "s08": {
                "status": "GO",
                "operational_stage_id": "25",
                "rq": "RQ3",
                "reexecution_after_specification_freeze": True,
                "specification_readiness": READY,
                "primary_model_family": FAM_SPLINE,
                "primary_support": "CS4_k5_q0.95",
                "support_alignment": "B_PAIRWISE_INTERSECTION",
                "contrast_orientation": "CONTROL_MINUS_TWIN",
                "target_aggregation": "M1_EQUAL_WEIGHT_ARITHMETIC_MEAN",
                "fleet_synthesis": "F1_TARGET_BALANCED_ARITHMETIC_MEAN",
                "target_level_inference": "TEMPORAL_MBB_5000_PERCENTILE_95",
                "fleet_level_inference": "NONE_DESCRIPTIVE_ONLY",
                "n_targets": len(targets),
                "n_source_contrasts": len(source_rows),
                "n_computable_targets": len(jcomp),
                "D_fleet_RQ3": D_fleet,
                "prior_stop_reference": prior,
                **arts_sot,
            }
        },
    }
    validations.append(ValidationRecord("historical_s08_unchanged", True, None))
    return StageResult(status="GO", message=MSG_GO, sot_patch=patch, artifacts=recs, validations=validations)


def _sci_report(inf_rows, fleet, group_rows, source_rows) -> str:
    lines = [
        "# S08 twin vs matched non-twin (scientific re-execution)",
        "",
        "Frozen specification consumed: CS4, B pairwise intersection, CONTROL_MINUS_TWIN, M1 equal-source mean, F1 target-balanced descriptive fleet mean, target-level temporal MBB (B=5000, 95% percentile), no fleet inference.",
        "",
        f"Source-level computable contrasts: {len(source_rows)}. Targets: {len(inf_rows)}.",
        "",
        "| target | group | control | n_src | Delta_j | CI_low | CI_up | L_j |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in inf_rows:
        lines.append(
            f"| {r['target_plant_id']} | {r['reference_group_id']} | {r['matched_control_source_plant_id']} | {r['n_pairwise_computable_exact_twin_sources']} | {r['Delta_j']:.6g} | {r['ci_lower']:.6g} | {r['ci_upper']:.6g} | {r['L_j']} |"
        )
    lines.extend(
        [
            "",
            f"Fleet F1 descriptive scalar: `{fleet['D_fleet_RQ3']}` with denominator {fleet['actual_denominator']} of {fleet['n_expected_targets']} expected targets. No fleet CI/p-value.",
            "",
            "Group companion (descriptive mean only):",
            "",
        ]
    )
    for g in group_rows:
        lines.append(f"- `{g['reference_group_id']}`: {g['n_computable_targets']}/{g['n_expected_targets']} computable, D_g={g['D_g']}")
    lines.extend(
        [
            "",
            "Block lengths from Politis-White + Patton-Politis-White (`arch.bootstrap.optimal_block_length`, circular column) on daily mean VALIDATION absolute error; L_j = max across SplineRidge and HistGradientBoosting, ceil to whole days, min 1.",
            "",
            "No equivalence classification. No causal interpretation. No S09 robustness.",
            "",
            f"**{MSG_GO}**",
            "",
        ]
    )
    return "\n".join(lines)
