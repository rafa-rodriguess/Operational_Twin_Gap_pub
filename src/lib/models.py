"""S04 local models and Gself (operational stage 21). No OTG, CS, or cross-plant transfer."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import (
    FEATURE_INTERP,
    FEATURE_STRATEGIES,
    assign_post_eligibility_split,
    dump_json,
    parse_ts,
    sha256_file,
    write_csv,
)

ANSWER = None
OUT = "artifacts/s04"
REPORT = "reports/scientific/S04_LOCAL_MODELS_GSELF.md"
FEATURES = FEATURE_STRATEGIES["six_core"]
TARGET = "y_dc_normalized"
FAM_SPLINE = "SplineRidge"
FAM_HGB = "HistGradientBoosting"
SPLINE_GRID = [{"n_knots": k, "alpha": a} for k, a in product((4, 6, 8), (0.1, 1, 10))]
HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaf, "l2_regularization": l2}
    for lr, leaf, l2 in product((0.05, 0.10), (15, 31), (0, 1))
]


def _csv_fp(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    return hashlib.sha256("|".join(header).encode("utf-8")).hexdigest()


def _pq_fp(path: Path) -> str:
    schema = pq.read_schema(path)
    return hashlib.sha256("|".join(schema.names).encode("utf-8")).hexdigest()


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


def _bool01(arr) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i, val in enumerate(arr):
        if val is None:
            continue
        if val is True or val is np.True_:
            out[i] = 1.0
        elif val is False or val is np.False_:
            out[i] = 0.0
    return out


def eligible_mask(poa, cov, y, feats: dict[str, np.ndarray], dc_flag, feat_flags: dict[str, np.ndarray]) -> np.ndarray:
    n = len(y)
    mask = np.isfinite(y) & np.isfinite(poa) & np.isfinite(cov) & (poa > 50.0) & (cov >= 0.80)
    for name in FEATURES:
        mask &= np.isfinite(feats[name])
    any_true = dc_flag == 1.0
    for name in FEATURES:
        any_true = any_true | (feat_flags[name] == 1.0)
    return mask & (~any_true)


def gself_block_ids(train_ts: list[datetime]) -> list[str]:
    t0, t1 = min(train_ts), max(train_ts)
    span = t1 - t0
    d = span / 3
    e1, e2 = t0 + d, t0 + 2 * d
    out = []
    for t in train_ts:
        if t < e1:
            out.append("B1")
        elif t < e2:
            out.append("B2")
        else:
            out.append("B3")
    return out


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    resid = y_pred - y_true
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": float(np.mean(resid))}


def _spline(params: dict) -> Pipeline:
    return Pipeline(
        [
            ("spline", SplineTransformer(n_knots=params["n_knots"], degree=3)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=params["alpha"])),
        ]
    )


def _hgb(params: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"],
        l2_regularization=params["l2_regularization"],
        loss="absolute_error",
        max_iter=300,
        early_stopping=False,
    )


def _fit_select(family: str, grid: list[dict], Xtr, ytr, Xva, yva) -> tuple[dict, float, Any, list[dict]]:
    rows = []
    best = None
    for i, params in enumerate(grid):
        cid = f"{family}_{i:02d}"
        est = _spline(params) if family == FAM_SPLINE else _hgb(params)
        status = "ok"
        mae = float("nan")
        try:
            est.fit(Xtr, ytr)
            pred = np.asarray(est.predict(Xva), dtype=float)
            if not np.all(np.isfinite(pred)):
                status = "nonfinite_val_pred"
            else:
                mae = float(np.mean(np.abs(pred - yva)))
        except Exception as exc:  # noqa: BLE001
            status = f"fit_error:{type(exc).__name__}"
            est = None
        rows.append({"candidate_id": cid, "params": params, "validation_mae": mae, "fit_status": status, "estimator": est})
    finite = [r for r in rows if r["fit_status"] == "ok" and np.isfinite(r["validation_mae"])]
    if not finite:
        raise RuntimeError(f"no trainable {family} candidate")
    min_mae = min(r["validation_mae"] for r in finite)
    winners = [r for r in finite if r["validation_mae"] == min_mae]
    if len(winners) > 1:
        raise TieError(family, [w["candidate_id"] for w in winners], min_mae)
    best = winners[0]
    return best["params"], min_mae, best["estimator"], rows


class TieError(Exception):
    def __init__(self, family, cids, mae):
        super().__init__(f"exact validation-MAE tie family={family} mae={mae} candidates={cids}")
        self.family = family
        self.cids = cids


def run(ctx: StageContext) -> StageResult:
    root = ctx.root
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    validations: list[ValidationRecord] = []
    from config.protocol import PRIMARY_GROUP_IDS
    src = Path(__file__).read_text(encoding="utf-8")
    validations.append(ValidationRecord("no_tracker_albedo_feature", "tracker_albedo_index" not in FEATURES, None))
    validations.append(ValidationRecord("no_Lj_realization", ("realized_" + "L_j") not in src, None))

    scope_path = root / "artifacts/scope/PRIMARY_SCOPE.json"
    if scope_path.is_file():
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        plants = list(scope.get("primary_plants") or [])
    else:
        members_path = root / "artifacts/twins/members.csv"
        import csv as _csv
        plants = []
        if members_path.is_file():
            with members_path.open(encoding="utf-8", newline="") as handle:
                for row in _csv.DictReader(handle):
                    if row.get("group_id") in PRIMARY_GROUP_IDS:
                        plants.append(row["plant_id"])
        plants = sorted(set(plants))
        scope = {"primary_plants": plants, "gself_structural_executability": []}
    if len(plants) != 14:
        return _halt(now, validations, "STOP", f"primary plant count is {len(plants)}, expected 14")

    panel_path = root / "artifacts/panel/plant_panel.parquet"
    if not panel_path.is_file():
        panel_path = root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"
    if not panel_path.is_file():
        return _halt(now, validations, "STOP", "canonical panel missing")

    plist = ",".join("'" + p.replace("'", "''") + "'" for p in plants)
    import duckdb

    con = duckdb.connect()
    table = con.execute(
        f"""
        SELECT plant_id, datetime_raw, y_dc_normalized, coverage_dc, poa_irradiance_wm2,
               ghi_irradiance_wm2, gri_irradiance_wm2, panel_temperature_celsius,
               ambient_temperature_celsius, wind_speed_ms, dc_any_officially_interpolated,
               interpolated_keys_poa_irradiance_wm2, interpolated_keys_ghi_irradiance_wm2,
               interpolated_keys_gri_irradiance_wm2, interpolated_keys_panel_temperature_celsius,
               interpolated_keys_ambient_temperature_celsius, interpolated_keys_wind_speed_ms
        FROM read_parquet('{str(panel_path).replace("'", "''")}')
        WHERE plant_id IN ({plist})
          AND y_dc_normalized IS NOT NULL
          AND poa_irradiance_wm2 > 50
          AND coverage_dc >= 0.80
          AND ghi_irradiance_wm2 IS NOT NULL
          AND gri_irradiance_wm2 IS NOT NULL
          AND panel_temperature_celsius IS NOT NULL
          AND ambient_temperature_celsius IS NOT NULL
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

    out_dir = root / OUT
    models_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    analytic = []
    sel_rows = []
    test_pred = []
    val_pred = []
    gself_pred = []
    metrics_rows = []
    gself_comp = []
    gself_sum = []
    model_files = []
    protocol_defaults: dict[str, Any] = {}

    split_audit = root / "artifacts/s03_corrective_v2/S03_CORRECTED_SPLIT_AUDIT.csv"
    split_expect: dict[str, dict[str, int]] = {}
    if split_audit.is_file():
        with split_audit.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("plant_id") in set(plants) and row.get("feature_strategy") == "six_core" and row.get("interpolation_policy") == "exclude_TRUE":
                    try:
                        if float(row.get("irradiance_wm2") or 0) == 50 and float(row.get("coverage_dc_min") or 0) == 0.8:
                            split_expect[row["plant_id"]] = {
                                "n_train": int(float(row["n_train"])),
                                "n_validation": int(float(row["n_validation"])),
                                "n_test": int(float(row["n_test"])),
                            }
                    except (TypeError, ValueError):
                        pass

    by_plant: dict[str, list[int]] = {p: [] for p in plants}
    pids = table["plant_id"].to_pylist()
    for i, pid in enumerate(pids):
        if pid in by_plant:
            by_plant[pid].append(i)

    def col(name: str) -> list:
        return table[name].to_pylist()

    y_all = np.array(col(TARGET), dtype=float)
    poa_all = np.array(col("poa_irradiance_wm2"), dtype=float)
    cov_all = np.array(col("coverage_dc"), dtype=float)
    feat_all = {f: np.array(col(f), dtype=float) for f in FEATURES}
    dc_all = _bool01(col("dc_any_officially_interpolated"))
    flag_all = {f: _bool01(col(FEATURE_INTERP[f])) for f in FEATURES}
    dt_all = col("datetime_raw")
    ts_all = [parse_ts(x) if isinstance(x, str) else (x if isinstance(x, datetime) else None) for x in dt_all]

    for pid in plants:
        ix = np.array(by_plant[pid], dtype=int)
        if ix.size == 0:
            return _halt(now, validations, "STOP", f"S04 STOP: no panel rows for {pid}")
        elig = eligible_mask(poa_all[ix], cov_all[ix], y_all[ix], {k: v[ix] for k, v in feat_all.items()}, dc_all[ix], {k: v[ix] for k, v in flag_all.items()})
        local_ts = [ts_all[int(j)] for j in ix]
        local_dt = [str(dt_all[int(j)] or "") for j in ix]
        order = [int(k) for k in np.where(elig)[0]]
        sentinel = datetime.max.replace(tzinfo=timezone.utc)

        def skey(k: int):
            t = local_ts[k]
            return (t if t is not None else sentinel, local_dt[k])

        order.sort(key=skey)
        n = len(order)
        labels = [""] * len(ix)
        for rank, k in enumerate(order):
            labels[k] = assign_post_eligibility_split(n, rank)
        train_k = [k for k, lab in enumerate(labels) if lab == "train"]
        val_k = [k for k, lab in enumerate(labels) if lab == "validation"]
        test_k = [k for k, lab in enumerate(labels) if lab == "test"]
        if not train_k or not val_k or not test_k:
            return _halt(now, validations, "STOP", f"S04 STOP: empty partition for {pid}")
        t_tr = [local_ts[k] for k in train_k if local_ts[k]]
        t_va = [local_ts[k] for k in val_k if local_ts[k]]
        t_te = [local_ts[k] for k in test_k if local_ts[k]]
        if max(t_tr) >= min(t_va) or max(t_va) >= min(t_te):
            return _halt(now, validations, "REDESIGN", f"FATAL: chronology leak {pid}")
        if split_expect.get(pid) and (
            split_expect[pid]["n_train"] != len(train_k) or split_expect[pid]["n_validation"] != len(val_k) or split_expect[pid]["n_test"] != len(test_k)
        ):
            return _halt(now, validations, "STOP", f"S04 STOP: split counts disagree with S03 for {pid}")

        blocks = gself_block_ids(t_tr)
        if sorted(set(blocks)) != ["B1", "B2", "B3"]:
            return _halt(now, validations, "STOP", f"S04 STOP: Gself blocks incomplete for {pid}")
        block_of = {train_k[i]: blocks[i] for i in range(len(train_k))}
        for k, lab in enumerate(labels):
            gi = int(ix[k])
            analytic.append({"plant_id": pid, "datetime_raw": local_dt[k], "split": lab or None, "train_gself_block_id": block_of.get(k)})

        def mat(ks):
            X = np.column_stack([feat_all[f][ix[ks]] for f in FEATURES])
            y = y_all[ix[ks]]
            dts = [local_dt[k] for k in ks]
            return X, y, dts

        Xtr, ytr, _ = mat(train_k)
        Xva, yva, dtva = mat(val_k)
        Xte, yte, dtte = mat(test_k)

        for family, grid in ((FAM_SPLINE, SPLINE_GRID), (FAM_HGB, HGB_GRID)):
            try:
                params, _, full_est, cand_rows = _fit_select(family, grid, Xtr, ytr, Xva, yva)
            except TieError as exc:
                return _halt(now, validations, "STOP", f"S04 STOP: {exc}")
            except RuntimeError as exc:
                return _halt(now, validations, "STOP", f"S04 STOP: {pid} {exc}")
            selected_id = next(r["candidate_id"] for r in cand_rows if r["estimator"] is full_est or (r["fit_status"] == "ok" and r["params"] == params and r["validation_mae"] == min(x["validation_mae"] for x in cand_rows if x["fit_status"] == "ok")))
            selected_id = min((r for r in cand_rows if r["fit_status"] == "ok"), key=lambda r: r["validation_mae"])["candidate_id"]
            for r in cand_rows:
                sel_rows.append(
                    {
                        "plant_id": pid,
                        "model_family": family,
                        "fit_scope": "full",
                        "candidate_id": r["candidate_id"],
                        "hyperparameters": json.dumps(r["params"], sort_keys=True),
                        "train_row_count": len(train_k),
                        "validation_row_count": len(val_k),
                        "validation_mae": r["validation_mae"],
                        "selected": r["candidate_id"] == selected_id,
                        "fit_status": r["fit_status"],
                    }
                )
            pred_va = np.asarray(full_est.predict(Xva), dtype=float)
            pred_te = np.asarray(full_est.predict(Xte), dtype=float)
            if not np.all(np.isfinite(pred_va)) or not np.all(np.isfinite(pred_te)):
                return _halt(now, validations, "STOP", f"S04 STOP: nonfinite predictions {pid} {family}")
            for dt, yt, yp in zip(dtva, yva, pred_va):
                val_pred.append({"plant_id": pid, "datetime_raw": dt, "model_family": family, "selected_model_only": True, "y_true": float(yt), "y_pred": float(yp), "residual": float(yp - yt), "abs_error": float(abs(yp - yt))})
            for dt, yt, yp in zip(dtte, yte, pred_te):
                test_pred.append({"plant_id": pid, "datetime_raw": dt, "model_family": family, "y_true": float(yt), "y_pred": float(yp), "residual": float(yp - yt), "abs_error": float(abs(yp - yt))})
            m = metrics(yte, pred_te)
            metrics_rows.append({"plant_id": pid, "model_family": family, "n_train": len(train_k), "n_validation": len(val_k), "n_test": len(test_k), "test_mae": m["mae"], "test_rmse": m["rmse"], "test_r2": m["r2"], "test_bias": m["bias"], "selected_candidate_id": selected_id})
            dest = models_dir / pid / family
            dest.mkdir(parents=True, exist_ok=True)
            full_path = dest / "full.joblib"
            joblib.dump(full_est, full_path)
            model_files.append(full_path)
            protocol_defaults[f"{family}_full_get_params"] = _safe_params(full_est)
            full_mae = m["mae"]
            comps = {}
            for bid in ("B1", "B2", "B3"):
                keep = [k for k in train_k if block_of[k] != bid]
                if not keep:
                    return _halt(now, validations, "STOP", f"S04 STOP: empty TRAIN_minus_{bid} for {pid}")
                Xr, yr, _ = mat(keep)
                try:
                    rparams, _, rest_est, rcands = _fit_select(family, grid, Xr, yr, Xva, yva)
                except TieError as exc:
                    return _halt(now, validations, "STOP", f"S04 STOP: {pid} {bid} {exc}")
                except RuntimeError as exc:
                    return _halt(now, validations, "STOP", f"S04 STOP: {pid} {bid} {exc}")
                rcid = min((r for r in rcands if r["fit_status"] == "ok"), key=lambda r: r["validation_mae"])["candidate_id"]
                for r in rcands:
                    sel_rows.append({"plant_id": pid, "model_family": family, "fit_scope": f"restricted_{bid}", "candidate_id": r["candidate_id"], "hyperparameters": json.dumps(r["params"], sort_keys=True), "train_row_count": len(keep), "validation_row_count": len(val_k), "validation_mae": r["validation_mae"], "selected": r["candidate_id"] == rcid, "fit_status": r["fit_status"]})
                pred_r = np.asarray(rest_est.predict(Xte), dtype=float)
                if not np.all(np.isfinite(pred_r)):
                    return _halt(now, validations, "STOP", f"S04 STOP: nonfinite restricted pred {pid} {bid}")
                rmae = float(np.mean(np.abs(pred_r - yte)))
                gap = rmae - full_mae
                comps[bid] = gap
                gself_comp.append({"plant_id": pid, "model_family": family, "block_id": bid, "n_train_full": len(train_k), "n_train_restricted": len(keep), "n_validation": len(val_k), "n_test": len(test_k), "full_test_mae": full_mae, "restricted_test_mae": rmae, "gself_component": gap, "selected_restricted_candidate_id": rcid})
                for dt, yt, yf, yr_ in zip(dtte, yte, pred_te, pred_r):
                    gself_pred.append({"plant_id": pid, "datetime_raw": dt, "model_family": family, "block_id": bid, "y_true": float(yt), "y_pred_full": float(yf), "y_pred_restricted": float(yr_), "abs_error_full": float(abs(yf - yt)), "abs_error_restricted": float(abs(yr_ - yt))})
                rpath = dest / f"restricted_{bid}.joblib"
                joblib.dump(rest_est, rpath)
                model_files.append(rpath)
            gself_sum.append({"plant_id": pid, "model_family": family, "gself_b1": comps["B1"], "gself_b2": comps["B2"], "gself_b3": comps["B3"], "gself_mean_signed": float(np.mean([comps["B1"], comps["B2"], comps["B3"]])), "full_test_mae": full_mae, "all_three_components_present": True})

    validations.append(ValidationRecord("plants_modeled", len({r["plant_id"] for r in metrics_rows}) == 14, None))
    validations.append(ValidationRecord("families_complete", len(metrics_rows) == 28, None))

    def write_pq(path: Path, rows: list[dict], fields: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for f in fields:
            vals = [r.get(f) for r in rows]
            if f in {"y_true", "y_pred", "residual", "abs_error", "y_pred_full", "y_pred_restricted", "abs_error_full", "abs_error_restricted"}:
                arrays[f] = pa.array(vals, type=pa.float64())
            elif f == "selected_model_only":
                arrays[f] = pa.array(vals, type=pa.bool_())
            else:
                arrays[f] = pa.array([None if v is None else str(v) for v in vals], type=pa.string())
        pq.write_table(pa.table(arrays), path)

    idx_p = out_dir / "S04_ANALYTIC_INDEX.parquet"
    test_p = out_dir / "S04_LOCAL_TEST_PREDICTIONS.parquet"
    val_p = out_dir / "S04_VALIDATION_PREDICTIONS.parquet"
    gs_p = out_dir / "S04_GSELF_TEST_PREDICTIONS.parquet"
    write_pq(idx_p, analytic, ["plant_id", "datetime_raw", "split", "train_gself_block_id"])
    write_pq(test_p, test_pred, ["plant_id", "datetime_raw", "model_family", "y_true", "y_pred", "residual", "abs_error"])
    write_pq(val_p, val_pred, ["plant_id", "datetime_raw", "model_family", "selected_model_only", "y_true", "y_pred", "residual", "abs_error"])
    write_pq(gs_p, gself_pred, ["plant_id", "datetime_raw", "model_family", "block_id", "y_true", "y_pred_full", "y_pred_restricted", "abs_error_full", "abs_error_restricted"])
    write_csv(out_dir / "S04_HYPERPARAMETER_SELECTION.csv", sel_rows, ["plant_id", "model_family", "fit_scope", "candidate_id", "hyperparameters", "train_row_count", "validation_row_count", "validation_mae", "selected", "fit_status"])
    write_csv(out_dir / "S04_LOCAL_METRICS.csv", metrics_rows, ["plant_id", "model_family", "n_train", "n_validation", "n_test", "test_mae", "test_rmse", "test_r2", "test_bias", "selected_candidate_id"])
    write_csv(out_dir / "S04_GSELF_COMPONENTS.csv", gself_comp, ["plant_id", "model_family", "block_id", "n_train_full", "n_train_restricted", "n_validation", "n_test", "full_test_mae", "restricted_test_mae", "gself_component", "selected_restricted_candidate_id"])
    write_csv(out_dir / "S04_GSELF_SUMMARY.csv", gself_sum, ["plant_id", "model_family", "gself_b1", "gself_b2", "gself_b3", "gself_mean_signed", "full_test_mae", "all_three_components_present"])

    protocol = {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pyarrow_version": pa.__version__,
        "scikit_learn_version": __import__("sklearn").__version__,
        "feature_order": list(FEATURES),
        "target": TARGET,
        "primary_eligibility": "poa>50, coverage_dc>=0.80, finite y and six-core, exclude_TRUE interpolation (dc or active features); UNKNOWN not recoded FALSE",
        "split_rule": "eligibility_then_chronological_floor_60_20_20",
        "spline_grid": SPLINE_GRID,
        "hgb_grid": HGB_GRID,
        "unspecified_effective_params": protocol_defaults,
        "validation_selection": "minimum finite VALIDATION MAE; exact ties STOP",
        "final_fit_partition": "TRAIN_only",
        "gself_blocks": "three equal-duration calendar intervals on TRAIN span; last closed on the right",
        "test_used_for_selection": False,
        "cross_plant_data_used": False,
        "common_support_or_otg_calculated": False,
        "block_length_selector_realization_computed": False,
    }
    dump_json(out_dir / "S04_MODEL_PROTOCOL.json", protocol)
    model_manifest = {"models": []}
    for path in sorted(model_files, key=lambda p: str(p)):
        rec = _art(root, path)
        model_manifest["models"].append(_tab(rec))
    dump_json(out_dir / "S04_MODEL_MANIFEST.json", model_manifest)

    recs = [
        _art(root, idx_p, rows=len(analytic), fp=_pq_fp(idx_p)),
        _art(root, test_p, rows=len(test_pred), fp=_pq_fp(test_p)),
        _art(root, val_p, rows=len(val_pred), fp=_pq_fp(val_p)),
        _art(root, gs_p, rows=len(gself_pred), fp=_pq_fp(gs_p)),
        _art(root, out_dir / "S04_HYPERPARAMETER_SELECTION.csv", rows=len(sel_rows), fp=_csv_fp(out_dir / "S04_HYPERPARAMETER_SELECTION.csv")),
        _art(root, out_dir / "S04_LOCAL_METRICS.csv", rows=len(metrics_rows), fp=_csv_fp(out_dir / "S04_LOCAL_METRICS.csv")),
        _art(root, out_dir / "S04_GSELF_COMPONENTS.csv", rows=len(gself_comp), fp=_csv_fp(out_dir / "S04_GSELF_COMPONENTS.csv")),
        _art(root, out_dir / "S04_GSELF_SUMMARY.csv", rows=len(gself_sum), fp=_csv_fp(out_dir / "S04_GSELF_SUMMARY.csv")),
        _art(root, out_dir / "S04_MODEL_PROTOCOL.json"),
        _art(root, out_dir / "S04_MODEL_MANIFEST.json"),
    ]
    report_path = root / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(metrics_rows, gself_sum, sel_rows), encoding="utf-8")
    recs.append(_art(root, report_path))
    man = {"job": "S04_LOCAL_MODELS_GSELF", "artifacts": {r.path.split("/")[-1]: _tab(r) for r in recs}}
    dump_json(out_dir / "S04_MANIFEST.json", man)
    recs.append(_art(root, out_dir / "S04_MANIFEST.json"))

    validations.append(ValidationRecord("no_sot_write", True, "sot_patch only"))
    validations.append(ValidationRecord("no_otg", True, None))
    patch = {
        "stages": {
            "S04": {
                "kind": "scientific_analysis",
                "status": "GO",
                "reason": "S04_LOCAL_MODELS_GSELF_COMPLETE",
                "operational_stage_id": "21",
                "job": "S04_LOCAL_MODELS_GSELF",
                "finished_at_utc": now,
                "answer": ANSWER,
                "report": _tab(recs[-2]),
            }
        },
        "scientific_analysis": {
            "s04": {
                "status": "GO",
                "operational_stage_id": "21",
                "scope": {
                    "primary_plant_count": 14,
                    "primary_plants": plants,
                    "model_families": [FAM_SPLINE, FAM_HGB],
                    "target": TARGET,
                    "feature_strategy": "six_core",
                },
                "model_protocol": _tab(recs[8]),
                "local_metrics": _tab(recs[5]),
                "local_test_predictions": _tab(recs[1]),
                "validation_predictions": _tab(recs[2]),
                "hyperparameter_selection": _tab(recs[4]),
                "gself": {
                    "components": _tab(recs[6]),
                    "summary": _tab(recs[7]),
                    "formula": "mean_b[MAE(restricted_b,TEST)-MAE(full,TEST)]",
                    "signed": True,
                    "block_count": 3,
                    "role": "within_asset_internal_portability_reference_not_physical_noise",
                },
                "model_manifest": _tab(recs[9]),
                "manifest": _tab(recs[-1]),
                "inference_inputs": {
                    "validation_absolute_error_series": _tab(recs[2]),
                    "block_length_selector_rule": "FROZEN_IN_S03_REALIZATION_DEFERRED_TO_INFERENCE_STAGE",
                },
                "cross_plant_results_computed": False,
                "otg_computed": False,
                "bootstrap_draws_generated": False,
            }
        },
    }
    return StageResult(status="GO", message="GO — S04 LOCAL MODELS AND GSELF COMPLETE", sot_patch=patch, artifacts=recs, validations=validations)


def _safe_params(est) -> dict[str, Any]:
    try:
        raw = est.get_params(deep=False)
    except Exception:
        return {}
    out = {}
    for k, v in raw.items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = type(v).__name__
    return out


def _halt(now: str, validations: list[ValidationRecord], status: str, message: str) -> StageResult:
    return StageResult(
        status=status,  # type: ignore[arg-type]
        message=message,
        sot_patch={"stages": {"S04": {"kind": "scientific_analysis", "status": status, "finished_at_utc": now, "reason": message, "answer": ANSWER, "job": "S04_LOCAL_MODELS_GSELF", "operational_stage_id": "21"}}},
        artifacts=[],
        validations=validations,
    )


def _report(metrics_rows, gself_sum, sel_rows) -> str:
    lines = [
        "# S04 — Local models and Gself",
        "",
        "Values below are copied from S04 artifacts. Gself is an internal portability reference, not physical noise. Families are not ranked.",
        "",
        "## Local TEST metrics",
        "",
        "| plant | family | n_test | MAE | RMSE | R2 | bias | selected |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(metrics_rows, key=lambda x: (x["plant_id"], x["model_family"])):
        lines.append(f"| {r['plant_id']} | {r['model_family']} | {r['n_test']} | {r['test_mae']:.8g} | {r['test_rmse']:.8g} | {r['test_r2']:.8g} | {r['test_bias']:.8g} | {r['selected_candidate_id']} |")
    lines += ["", "## Gself signed mean", "", "| plant | family | B1 | B2 | B3 | mean |", "|---|---|---:|---:|---:|---:|"]
    for r in sorted(gself_sum, key=lambda x: (x["plant_id"], x["model_family"])):
        lines.append(f"| {r['plant_id']} | {r['model_family']} | {r['gself_b1']:.8g} | {r['gself_b2']:.8g} | {r['gself_b3']:.8g} | {r['gself_mean_signed']:.8g} |")
    lines += ["", "No common-support, OTG, cross-plant, bootstrap, or permutation results were computed.", "", "**GO — S04 LOCAL MODELS AND GSELF COMPLETE**", ""]
    return "\n".join(lines)
