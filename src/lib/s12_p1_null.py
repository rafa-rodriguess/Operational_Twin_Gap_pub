"""S12 P1: within-plant null calibration (isolated; does not overwrite primary models)."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors

from config.protocol import CS4, FAM_SPLINE, PRIMARY_GROUP_IDS, TARGET_PRIMARY
from src.lib.models import _spline
from src.lib.split_support import (
    FEATURE_STRATEGIES,
    NN_ALGORITHM,
    iqr_is_valid,
    loo_kth_distances,
    parse_ts,
    q_linear,
    source_median_iqr,
    standardize,
)

FEATURES = FEATURE_STRATEGIES["six_core"]
WEEK = timedelta(days=7)
HALF_RULE = "chrono_order_by_datetime_raw; A=first floor(n/2); extra odd row assigned to later B"
INTERLEAVE_RULE = "7-day blocks from first canonical TRAIN timestamp; even->A odd->B; drop final unmatched block if n_blocks odd"
TOL = 1e-15
STATUS = "S12_P1_NULL_CALIBRATION_MATERIALIZED"


def _flag(val: Any) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def _days(dts: list[str]) -> int:
    out = set()
    for dt in dts:
        ts = parse_ts(dt)
        out.add(ts.date().isoformat() if ts else dt[:10])
    return len(out)


def _dist(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None, "mean": None}
    arr = np.asarray(vals, dtype=float)
    q = np.quantile(arr, [0.25, 0.5, 0.75, 0.95], method="linear")
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(q[0]),
        "median": float(q[1]),
        "p75": float(q[2]),
        "p95": float(q[3]),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def plant_balanced(rows: list[dict[str, Any]], field: str = "otg_abs") -> tuple[float | None, int]:
    by_t: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("computable"):
            by_t[str(r["target_plant_id"])].append(float(r[field]))
    if not by_t:
        return None, 0
    means = [sum(v) / len(v) for v in by_t.values()]
    return float(sum(means) / len(means)), len(by_t)


def plant_balanced_iterate(rows: list[dict[str, Any]], field: str = "otg_abs") -> tuple[float | None, int]:
    targets: list[str] = []
    for r in rows:
        if r.get("computable"):
            t = str(r["target_plant_id"])
            if t not in targets:
                targets.append(t)
    if not targets:
        return None, 0
    acc_means = []
    for t in targets:
        s, n = 0.0, 0
        for r in rows:
            if r.get("computable") and str(r["target_plant_id"]) == t:
                s += float(r[field])
                n += 1
        acc_means.append(s / n)
    return float(sum(acc_means) / len(acc_means)), len(acc_means)


def chrono_halves(dts: list[str]) -> tuple[list[str], list[str]]:
    n_a = len(dts) // 2
    return dts[:n_a], dts[n_a:]


def interleave_halves(dts: list[str]) -> tuple[list[str], list[str], dict[str, Any]]:
    if not dts:
        return [], [], {"n_blocks_before_drop": 0, "dropped_final_block": None, "n_blocks_a": 0, "n_blocks_b": 0}
    t0 = parse_ts(dts[0])
    if t0 is None:
        raise ValueError("unparseable first TRAIN timestamp")
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    bids = []
    for dt in dts:
        ts = parse_ts(dt)
        if ts is None:
            raise ValueError(dt)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        bids.append(int((ts - t0).total_seconds() // WEEK.total_seconds()))
    n_cal = max(bids) + 1
    dropped = None
    if n_cal % 2 == 1:
        dropped = n_cal - 1
        n_cal -= 1
    keep_max = n_cal - 1
    a, b = [], []
    occupied_a, occupied_b = set(), set()
    for dt, bid in zip(dts, bids):
        if bid > keep_max:
            continue
        if bid % 2 == 0:
            a.append(dt)
            occupied_a.add(bid)
        else:
            b.append(dt)
            occupied_b.add(bid)
    return a, b, {
        "n_calendar_blocks_before_drop": max(bids) + 1,
        "dropped_final_block": dropped,
        "n_blocks_a": n_cal // 2,
        "n_blocks_b": n_cal // 2,
        "n_occupied_blocks_a": len(occupied_a),
        "n_occupied_blocks_b": len(occupied_b),
        "anchor_ts": dts[0],
    }


def load_scope(root: Path) -> tuple[list[str], dict[str, str]]:
    scope = json.loads((root / "artifacts/scope/PRIMARY_SCOPE.json").read_text(encoding="utf-8"))
    plants = list(scope["primary_plants"])
    members: dict[str, str] = {}
    path = root / "artifacts/twins/members.csv"
    if not path.is_file():
        path = root / "artifacts/p02b/P02B_TWIN_MEMBERS.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("plant_id") in set(plants):
                members[row["plant_id"]] = row["group_id"]
    return plants, members


def load_hparams(root: Path, plants: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with (root / "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model_family") != FAM_SPLINE or row.get("fit_scope") != "full":
                continue
            if not _flag(row.get("selected")):
                continue
            if row["plant_id"] in plants:
                out[row["plant_id"]] = json.loads(row["hyperparameters"])
    return out


def load_directional_universe(root: Path) -> list[dict[str, str]]:
    with (root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_tables(root: Path, plants: list[str]):
    plist = ",".join("'" + p.replace("'", "''") + "'" for p in plants)
    feats_sql = ", ".join(f"p.{f}" for f in FEATURES)
    panel = root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"
    idx = root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet"
    con = duckdb.connect()
    table = con.execute(
        f"""
        SELECT i.plant_id, CAST(i.datetime_raw AS VARCHAR) AS datetime_raw, i.split,
               p.{TARGET_PRIMARY} AS y, {feats_sql}
        FROM read_parquet('{str(idx).replace("'", "''")}') i
        INNER JOIN read_parquet('{str(panel).replace("'", "''")}') p
          ON p.plant_id = i.plant_id AND CAST(p.datetime_raw AS VARCHAR) = CAST(i.datetime_raw AS VARCHAR)
        WHERE i.plant_id IN ({plist})
        """
    ).to_arrow_table()
    con.close()
    feat_by: dict[tuple[str, str], np.ndarray] = {}
    y_map: dict[tuple[str, str], float] = {}
    splits: dict[str, dict[str, list[str]]] = {p: {"train": [], "validation": [], "test": []} for p in plants}
    pids = table["plant_id"].to_pylist()
    dts = table["datetime_raw"].to_pylist()
    labs = table["split"].to_pylist()
    ys = table["y"].to_pylist()
    feat_cols = {f: table[f].to_pylist() for f in FEATURES}
    for i, pid in enumerate(pids):
        pid = str(pid)
        dt = str(dts[i])
        splits[pid][str(labs[i])].append(dt)
        yv = ys[i]
        if yv is not None and np.isfinite(float(yv)):
            y_map[(pid, dt)] = float(yv)
        vals = []
        ok = True
        for name in FEATURES:
            raw = feat_cols[name][i]
            if raw is None or not np.isfinite(float(raw)):
                ok = False
                break
            vals.append(float(raw))
        if ok:
            feat_by[(pid, dt)] = np.array(vals, dtype=float)
    for pid in plants:
        for part in splits[pid]:
            splits[pid][part].sort(key=lambda x: (parse_ts(x) or x, x))
    return feat_by, y_map, splits


def build_engine(pid: str, train_dts: list[str], feat_by: dict) -> dict[str, Any]:
    rows = []
    for dt in train_dts:
        vec = feat_by.get((pid, dt))
        if vec is None or not np.all(np.isfinite(vec)):
            return {"computable": False, "reason": "missing_or_nonfinite_train_features"}
        rows.append(vec)
    if len(rows) <= int(CS4["k"]):
        return {"computable": False, "reason": "insufficient_train_rows_for_k5"}
    X = np.vstack(rows)
    medians, iqrs = [], []
    for m, name in enumerate(FEATURES):
        med, iqr = source_median_iqr(X[:, m])
        if not iqr_is_valid(iqr):
            return {"computable": False, "reason": f"degenerate_iqr:{name}"}
        medians.append(med)
        iqrs.append(iqr)
    z = np.column_stack([standardize(X[:, m], medians[m], iqrs[m]) for m in range(len(FEATURES))])
    dist = loo_kth_distances(z, int(CS4["k"]))
    tau = q_linear(dist, float(CS4["q"]))
    if not np.isfinite(tau):
        return {"computable": False, "reason": "nonfinite_threshold"}
    nn = NearestNeighbors(n_neighbors=int(CS4["k"]), algorithm=NN_ALGORITHM, metric="euclidean")
    nn.fit(z)
    return {"computable": True, "reason": None, "medians": medians, "iqrs": iqrs, "nn": nn, "tau": float(tau), "n_source": len(rows)}


def score_support(engine: dict[str, Any], tgt: str, test_dts: list[str], feat_by: dict) -> list[tuple[str, float, float, bool]]:
    if not engine.get("computable"):
        return []
    k = int(CS4["k"])
    rows, keep = [], []
    for dt in test_dts:
        vec = feat_by.get((tgt, dt))
        if vec is None or not np.all(np.isfinite(vec)):
            continue
        rows.append(vec)
        keep.append(dt)
    if not rows:
        return []
    X = np.vstack(rows)
    z = np.column_stack([standardize(X[:, m], engine["medians"][m], engine["iqrs"][m]) for m in range(len(FEATURES))])
    dist, _ = engine["nn"].kneighbors(z, n_neighbors=k, return_distance=True)
    d5 = dist[:, k - 1]
    tau = float(engine["tau"])
    return [(dt, float(d5[i]), tau, bool(d5[i] <= tau)) for i, dt in enumerate(keep)]


def fit_frozen(pid: str, dts: list[str], feat_by, y_map, params: dict[str, Any]):
    X, y, used = [], [], []
    for dt in dts:
        vec = feat_by.get((pid, dt))
        yv = y_map.get((pid, dt))
        if vec is None or yv is None:
            continue
        X.append(vec)
        y.append(yv)
        used.append(dt)
    if len(X) < 2:
        return None, used, "insufficient_fit_rows"
    est = _spline(params)
    est.fit(np.vstack(X), np.array(y, dtype=float))
    return est, used, None


def predict_mae(est_a, est_b, tgt: str, supported: list[str], feat_by, y_map):
    yt, pa, pb, rows = [], [], [], []
    for dt in supported:
        vec = feat_by.get((tgt, dt))
        yv = y_map.get((tgt, dt))
        if vec is None or yv is None:
            continue
        ya = float(est_a.predict(vec.reshape(1, -1))[0])
        yb = float(est_b.predict(vec.reshape(1, -1))[0])
        yt.append(yv)
        pa.append(ya)
        pb.append(yb)
        rows.append((dt, yv, ya, yb))
    if not yt:
        return None, None, None, []
    arr_t = np.array(yt)
    mae_a = float(np.mean(np.abs(np.array(pa) - arr_t)))
    mae_b = float(np.mean(np.abs(np.array(pb) - arr_t)))
    return mae_a, mae_b, mae_a - mae_b, rows


def interpret(med_half, med_c, med_i, pb_half, p95_c, p95_i) -> str:
    if None in (med_half, med_c, med_i, pb_half, p95_c, p95_i):
        return "insufficient_computable_sample"
    placebo_med = max(med_c, med_i)
    if med_half <= placebo_med or pb_half <= placebo_med:
        return "placebo_comparable_to_or_larger_than_OTG_half"
    if pb_half > p95_c and pb_half > p95_i:
        return "clear_separation"
    return "substantial_overlap"



def _eval_row(arm, pid, src, tgt, dt, yv, ya, yb, d, tau, params):
    return {
        "arm": arm,
        "plant_id": pid,
        "source_plant_id": src,
        "target_plant_id": tgt,
        "datetime_raw": dt,
        "split": "test",
        "supported": True,
        "y_true": yv,
        "y_pred_a_or_source": ya,
        "y_pred_b_or_local": yb,
        "abs_error_a_or_source": abs(ya - yv),
        "abs_error_b_or_local": abs(yb - yv),
        "kth_distance": d,
        "threshold": tau,
        "hyperparameters": json.dumps(params, sort_keys=True),
    }


def run_science(root: Path, models_dir: Path) -> dict[str, Any]:
    plants, members = load_scope(root)
    hparams = load_hparams(root, plants)
    universe = load_directional_universe(root)
    feat_by, y_map, splits = load_tables(root, plants)
    models_dir.mkdir(parents=True, exist_ok=True)
    placebo_rows, eval_rows = [], []
    leakage = {"test_in_fit": False}
    inter_meta = {}
    model_paths = []

    def persist(est, name: str):
        if est is None:
            return
        path = models_dir / f"{name}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(est, path)
        model_paths.append(path)

    half_models: dict[str, Any] = {}
    half_n: dict[str, int] = {}

    for pid in plants:
        train, test, params = splits[pid]["train"], splits[pid]["test"], hparams.get(pid)
        rec = {"plant_id": pid, "group_id": members.get(pid), "n_train": len(train), "n_test": len(test), "hyperparameters": params}
        if params is None:
            rec.update({"computable_chrono": False, "reason_chrono": "missing_selected_spline_hparams"})
            placebo_rows.append(rec)
            continue
        a_c, b_c = chrono_halves(train)
        a_i, b_i, imeta = interleave_halves(train)
        inter_meta[pid] = imeta
        rec.update({
            "n_a_chrono": len(a_c), "n_b_chrono": len(b_c),
            "chrono_disjoint": set(a_c).isdisjoint(b_c),
            "chrono_within_train": set(a_c).issubset(train) and set(b_c).issubset(train),
            "n_a_interleaved": len(a_i), "n_b_interleaved": len(b_i),
            "n_blocks_a": imeta["n_blocks_a"], "n_blocks_b": imeta["n_blocks_b"],
            "interleaved_disjoint": set(a_i).isdisjoint(b_i),
            "interleaved_within_train": set(a_i).issubset(train) and set(b_i).issubset(train),
        })
        if (set(a_c)|set(b_c)|set(a_i)|set(b_i)) & set(test):
            leakage["test_in_fit"] = True

        def run_arm(label, a_dts, b_dts, reason_key, otg_key, mae_a_key, mae_b_key):
            eng = build_engine(pid, a_dts, feat_by)
            if not eng.get("computable"):
                rec[reason_key] = eng.get("reason")
                rec[f"computable_{label}"] = False
                return
            scored = score_support(eng, pid, test, feat_by)
            supported = [dt for dt, _d, _t, ok in scored if ok]
            rec[f"n_supported_{label}"] = len(supported)
            rec[f"support_retention_{label}"] = (len(supported) / len(test)) if test else None
            rec[f"threshold_{label}"] = eng.get("tau")
            est_a, used_a, err_a = fit_frozen(pid, a_dts, feat_by, y_map, params)
            est_b, used_b, err_b = fit_frozen(pid, b_dts, feat_by, y_map, params)
            persist(est_a, f"{pid}/{label}_A")
            persist(est_b, f"{pid}/{label}_B")
            if set(used_a) & set(test) or set(used_b) & set(test):
                leakage["test_in_fit"] = True
            if err_a or err_b or est_a is None or est_b is None:
                rec[reason_key] = err_a or err_b
                rec[f"computable_{label}"] = False
                return
            if not supported:
                rec[reason_key] = "empty_cs4_support"
                rec[f"computable_{label}"] = False
                return
            mae_a, mae_b, otg, pred_rows = predict_mae(est_a, est_b, pid, supported, feat_by, y_map)
            if otg is None:
                rec[reason_key] = "missing_predictions_on_support"
                rec[f"computable_{label}"] = False
                return
            rec[f"computable_{label}"] = True
            rec[reason_key] = None
            rec[mae_a_key] = mae_a
            rec[mae_b_key] = mae_b
            rec[otg_key] = otg
            dist_map = {dt: (d, tau) for dt, d, tau, ok in scored if ok}
            arm_name = "chrono_placebo" if label == "chrono" else "interleaved_placebo"
            for dt, yv, ya, yb in pred_rows:
                d, tau = dist_map[dt]
                eval_rows.append(_eval_row(arm_name, pid, pid, pid, dt, yv, ya, yb, d, tau, params))

        run_arm("chrono", a_c, b_c, "reason_chrono", "otg_placebo_chrono", "mae_a_chrono", "mae_b_chrono")
        run_arm("interleaved", a_i, b_i, "reason_interleaved", "otg_placebo_interleaved", "mae_a_interleaved", "mae_b_interleaved")
        rec["otg_chrono_minus_interleaved"] = (
            rec["otg_placebo_chrono"] - rec["otg_placebo_interleaved"]
            if rec.get("computable_chrono") and rec.get("computable_interleaved") else None
        )
        recent = b_c
        half_n[pid] = len(recent)
        he = build_engine(pid, recent, feat_by)
        hm, used_h, herr = fit_frozen(pid, recent, feat_by, y_map, params)
        persist(hm, f"{pid}/otg_half")
        if set(used_h) & set(test):
            leakage["test_in_fit"] = True
        half_models[pid] = (hm, herr, he, params)
        placebo_rows.append(rec)

    half_dir = []
    for row in universe:
        src, tgt, gid = row["source_plant_id"], row["target_plant_id"], row["group_id"]
        rec = {
            "group_id": gid, "source_plant_id": src, "target_plant_id": tgt,
            "model_family": FAM_SPLINE, "support_rule": "CS4",
            "n_source_recent_half": half_n.get(src), "n_target_recent_half": half_n.get(tgt), "computable": False,
        }
        sm, tm = half_models.get(src), half_models.get(tgt)
        if sm is None or tm is None:
            rec["reason"] = "missing_half_model"
            half_dir.append(rec)
            continue
        est_s, err_s, eng_s, par_s = sm
        est_t, err_t, _eng_t, par_t = tm
        if err_s or err_t or est_s is None or est_t is None:
            rec["reason"] = err_s or err_t or "unfit"
            half_dir.append(rec)
            continue
        if not eng_s.get("computable"):
            rec["reason"] = eng_s.get("reason")
            half_dir.append(rec)
            continue
        test = splits[tgt]["test"]
        scored = score_support(eng_s, tgt, test, feat_by)
        supported = [dt for dt, _d, _t, ok in scored if ok]
        rec["n_eligible_target_test"] = len(test)
        rec["n_supported"] = len(supported)
        rec["support_retention"] = (len(supported) / len(test)) if test else None
        rec["n_supported_days"] = _days(supported) if supported else 0
        rec["threshold"] = eng_s.get("tau")
        if not supported:
            rec["reason"] = "empty_cs4_support"
            half_dir.append(rec)
            continue
        mae_s, mae_t, otg, pred_rows = predict_mae(est_s, est_t, tgt, supported, feat_by, y_map)
        if otg is None:
            rec["reason"] = "missing_predictions_on_support"
            half_dir.append(rec)
            continue
        rec.update({"computable": True, "reason": None, "transfer_mae": mae_s, "local_same_rows_mae": mae_t, "otg_abs": otg})
        dist_map = {dt: (d, tau) for dt, d, tau, ok in scored if ok}
        for dt, yv, ya, yb in pred_rows:
            d, tau = dist_map[dt]
            eval_rows.append(_eval_row("otg_half", tgt, src, tgt, dt, yv, ya, yb, d, tau, {"source": par_s, "target": par_t}))
        half_dir.append(rec)

    chrono_ok = [r for r in placebo_rows if r.get("computable_chrono")]
    inter_ok = [r for r in placebo_rows if r.get("computable_interleaved")]
    both_ok = [r for r in placebo_rows if r.get("computable_chrono") and r.get("computable_interleaved")]
    half_ok = [r for r in half_dir if r.get("computable")]
    d_chrono = _dist([float(r["otg_placebo_chrono"]) for r in chrono_ok])
    d_inter = _dist([float(r["otg_placebo_interleaved"]) for r in inter_ok])
    d_delta = _dist([float(r["otg_chrono_minus_interleaved"]) for r in both_ok])
    half_vals = [float(r["otg_abs"]) for r in half_ok]
    d_half = _dist(half_vals)
    pb, n_t = plant_balanced(half_ok)
    pb2, n_t2 = plant_balanced_iterate(half_ok)
    g2_id = "TW_4467a039e00b"
    pb_g2, _ = plant_balanced([r for r in half_ok if r.get("group_id") == g2_id])
    pb_nong2, _ = plant_balanced([r for r in half_ok if r.get("group_id") != g2_id])
    p95_c, p95_i = d_chrono["p95"], d_inter["p95"]
    n_ex_c = sum(1 for v in half_vals if p95_c is not None and v > p95_c)
    n_ex_i = sum(1 for v in half_vals if p95_i is not None and v > p95_i)
    frac_c = (n_ex_c / len(half_vals)) if half_vals else None
    frac_i = (n_ex_i / len(half_vals)) if half_vals else None
    by_t_ex, by_t_exi = defaultdict(list), defaultdict(list)
    for r in half_ok:
        t, v = str(r["target_plant_id"]), float(r["otg_abs"])
        by_t_ex[t].append(p95_c is not None and v > p95_c)
        by_t_exi[t].append(p95_i is not None and v > p95_i)

    def _mean_frac(d):
        return float(sum(sum(v) / len(v) for v in d.values()) / len(d)) if d else None

    category = interpret(d_half["median"], d_chrono["median"], d_inter["median"], pb, p95_c, p95_i)
    notes = {
        "clear_separation": "OTG_half plant-balanced mean exceeds both placebo P95 values and medians; supports a cross-plant component beyond this within-plant null, not complete causal attribution.",
        "substantial_overlap": "OTG_half sits above placebo medians but does not exceed both P95 thresholds; residual-penalty interpretation should be qualified.",
        "placebo_comparable_to_or_larger_than_OTG_half": "RQ1 cannot cleanly attribute the full observed transfer gap to cross-plant operational difference alone under this training-size-matched null.",
    }
    sr_vals = [float(r["support_retention"]) for r in half_ok if r.get("support_retention") is not None]
    recon_max = 0.0

    def recon_placebo(arm, field, ok_rows):
        nonlocal recon_max
        by = defaultdict(lambda: {"a": [], "b": []})
        for r in eval_rows:
            if r["arm"] == arm:
                by[r["plant_id"]]["a"].append(r["abs_error_a_or_source"])
                by[r["plant_id"]]["b"].append(r["abs_error_b_or_local"])
        lookup = {r["plant_id"]: r for r in placebo_rows}
        for pid in [r["plant_id"] for r in ok_rows]:
            ea, eb = by[pid]["a"], by[pid]["b"]
            if not ea:
                recon_max = max(recon_max, 1.0)
                continue
            recon_max = max(recon_max, abs(float(np.mean(ea) - np.mean(eb)) - float(lookup[pid][field])))

    recon_placebo("chrono_placebo", "otg_placebo_chrono", chrono_ok)
    recon_placebo("interleaved_placebo", "otg_placebo_interleaved", inter_ok)
    by_tr = defaultdict(lambda: {"s": [], "l": []})
    for r in eval_rows:
        if r["arm"] == "otg_half":
            key = (r["source_plant_id"], r["target_plant_id"])
            by_tr[key]["s"].append(r["abs_error_a_or_source"])
            by_tr[key]["l"].append(r["abs_error_b_or_local"])
    for r in half_ok:
        recb = by_tr.get((r["source_plant_id"], r["target_plant_id"]))
        if not recb or not recb["s"]:
            recon_max = max(recon_max, 1.0)
            continue
        recon_max = max(recon_max, abs(float(np.mean(recb["s"]) - np.mean(recb["l"])) - float(r["otg_abs"])))
    if pb is not None and pb2 is not None:
        recon_max = max(recon_max, abs(pb - pb2))

    summary = {
        "status": STATUS,
        "n_primary_plants": len(plants),
        "primary_plants": plants,
        "placebo": {
            "chrono": {**d_chrono, "n_computable": len(chrono_ok), "n_planned": len(plants)},
            "interleaved": {**d_inter, "n_computable": len(inter_ok), "n_planned": len(plants)},
            "chrono_minus_interleaved": {**d_delta, "n_paired": len(both_ok)},
        },
        "otg_half": {
            **d_half,
            "directional_mean": d_half["mean"],
            "plant_balanced_mean": pb,
            "n_targets_in_plant_balanced": n_t,
            "n_expected_directional": len(universe),
            "n_computable": len(half_ok),
            "n_noncomputable": len(half_dir) - len(half_ok),
            "support_retention": _dist(sr_vals),
        },
        "null_exceedance": {
            "p95_placebo_chrono": p95_c,
            "p95_placebo_interleaved": p95_i,
            "n_otg_half_exceeding_p95_chrono": n_ex_c,
            "fraction_otg_half_exceeding_p95_chrono": frac_c,
            "n_otg_half_exceeding_p95_interleaved": n_ex_i,
            "fraction_otg_half_exceeding_p95_interleaved": frac_i,
            "uses_otg_half_not_full_train_primary": True,
        },
        "target_balanced_sensitivity": {
            "plant_balanced_all_computable_targets": pb,
            "plant_balanced_iterate_path": pb2,
            "n_targets": n_t2,
            "plant_balanced_G2_only": pb_g2,
            "plant_balanced_non_G2": pb_nong2,
            "mean_per_target_exceedance_frac_p95_chrono": _mean_frac(by_t_ex),
            "mean_per_target_exceedance_frac_p95_interleaved": _mean_frac(by_t_exi),
            "directional_unweighted_mean": d_half["mean"],
            "note": "Plant-balanced = equal weight per target (RQ1 synthesis). Unweighted directional mean can be G2-heavy.",
        },
        "interpretation_category": category,
        "interpretation_note": notes.get(category, category),
        "primary_otg_unchanged": True,
        "primary_otg_not_training_size_matched_to_placebo": True,
        "half_split_rule": HALF_RULE,
        "interleave_rule": INTERLEAVE_RULE,
        "support_semantics": "CS4 k=5 q=0.95 source-history median/IQR; LOO d5 threshold; degenerate IQR = not_computable; no P3 MAD fallback",
        "features": list(FEATURES),
        "target": TARGET_PRIMARY,
        "reconciliation_max_abs_discrepancy": recon_max,
        "reconciliation_ok": recon_max <= TOL,
        "leakage": leakage,
        "interleave_meta": inter_meta,
        "no_p3_mad_fallback": True,
        "noncomputable_half": [
            {"source_plant_id": r["source_plant_id"], "target_plant_id": r["target_plant_id"], "reason": r.get("reason")}
            for r in half_dir if not r.get("computable")
        ],
        "noncomputable_placebo": [
            {"plant_id": r["plant_id"], "reason_chrono": r.get("reason_chrono"), "reason_interleaved": r.get("reason_interleaved")}
            for r in placebo_rows if not r.get("computable_chrono") or not r.get("computable_interleaved")
        ],
    }
    return {
        "plants": plants, "members": members, "hparams": hparams, "placebo_rows": placebo_rows,
        "half_dir": half_dir, "eval_rows": eval_rows, "summary": summary, "model_paths": model_paths,
        "universe_n": len(universe), "primary_groups": list(PRIMARY_GROUP_IDS), "splits": splits,
    }


def write_eval_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "arm", "plant_id", "source_plant_id", "target_plant_id", "datetime_raw", "split", "supported",
        "y_true", "y_pred_a_or_source", "y_pred_b_or_local", "abs_error_a_or_source", "abs_error_b_or_local",
        "kth_distance", "threshold", "hyperparameters",
    ]
    arrays = {}
    for f in fields:
        vals = [r.get(f) for r in rows]
        if f in {"y_true", "y_pred_a_or_source", "y_pred_b_or_local", "abs_error_a_or_source", "abs_error_b_or_local", "kth_distance", "threshold"}:
            arrays[f] = pa.array(vals, type=pa.float64())
        elif f == "supported":
            arrays[f] = pa.array(vals, type=pa.bool_())
        else:
            arrays[f] = pa.array([None if v is None else str(v) for v in vals], type=pa.string())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), path)


def render_report(summary: dict[str, Any]) -> str:
    p, h, e = summary["placebo"], summary["otg_half"], summary["null_exceedance"]
    return "\n".join([
        "# S12 P1 — Within-plant null calibration",
        "",
        "Secondary calibration only. Accepted primary OTG / RQ1–RQ3 / S11 / source-selection are unchanged and are not training-size matched to these half-history placebos.",
        "",
        f"Status: `{summary['status']}`. Interpretation: **{summary['interpretation_category']}**.",
        "",
        str(summary["interpretation_note"]),
        "",
        "## Placebo (one value per computable plant)",
        "",
        f"- Chronological: n={p['chrono']['n_computable']}/{p['chrono']['n_planned']}, median={p['chrono']['median']}, P75={p['chrono']['p75']}, P95={p['chrono']['p95']}, min={p['chrono']['min']}, max={p['chrono']['max']}.",
        f"- Interleaved 7-day: n={p['interleaved']['n_computable']}/{p['interleaved']['n_planned']}, median={p['interleaved']['median']}, P75={p['interleaved']['p75']}, P95={p['interleaved']['p95']}, min={p['interleaved']['min']}, max={p['interleaved']['max']}.",
        f"- Paired chrono−interleaved: n={p['chrono_minus_interleaved']['n_paired']}, median={p['chrono_minus_interleaved']['median']}.",
        "",
        "## OTG_half (training-size-matched real transfer)",
        "",
        f"- Expected directional={h['n_expected_directional']}, computable={h['n_computable']}, noncomputable={h['n_noncomputable']}.",
        f"- Directional mean={h['directional_mean']}, median={h['median']}, plant-balanced mean={h['plant_balanced_mean']} (n_targets={h['n_targets_in_plant_balanced']}).",
        f"- P75={h['p75']}, P95={h['p95']}, min={h['min']}, max={h['max']}.",
        f"- Support retention among computable: median={h['support_retention']['median']}.",
        "",
        "## Null exceedance (OTG_half vs placebo P95, not full-TRAIN primary OTG)",
        "",
        f"- P95 chrono={e['p95_placebo_chrono']}; {e['n_otg_half_exceeding_p95_chrono']} / {h['n_computable']} = {e['fraction_otg_half_exceeding_p95_chrono']}.",
        f"- P95 interleaved={e['p95_placebo_interleaved']}; {e['n_otg_half_exceeding_p95_interleaved']} / {h['n_computable']} = {e['fraction_otg_half_exceeding_p95_interleaved']}.",
        "",
        "## Target-balanced sensitivity",
        "",
        json.dumps(summary["target_balanced_sensitivity"], indent=2, sort_keys=True),
        "",
        f"Independent reconciliation max |Δ| = {summary['reconciliation_max_abs_discrepancy']}.",
        "",
    ])
