"""S11 Phase 3 A1: G3 sensitivity cohort (POA+GHI reduced features)."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from sklearn.neighbors import NearestNeighbors

from config.protocol import (
    CS4,
    FAM_SPLINE,
    FEATURES_POA_GHI,
    SENSITIVITY_A_GROUP_ID,
    TARGET_PRIMARY,
)
from src.lib.matching import ranking_components
from src.lib.models import SPLINE_GRID, _bool01, _fit_select
from src.lib.rq3_science import load_meta
from src.lib.split_support import (
    FEATURE_INTERP,
    FEATURE_STRATEGIES,
    NN_ALGORITHM,
    assign_post_eligibility_split,
    iqr_is_valid,
    loo_kth_distances,
    parse_ts,
    q_linear,
    source_median_iqr,
    standardize,
)

TOL = 1e-15
G3_ID = SENSITIVITY_A_GROUP_ID
FEATURES = FEATURES_POA_GHI
STATUS_MATERIALIZED = "S11_PHASE3_A1_G3_MATERIALIZED"
STATUS_FEASIBILITY = "S11_PHASE3_A1_G3_DESCRIPTIVE_FEASIBILITY_ONLY"
STATUS_NO_RQ3 = "NO_EXECUTABLE_SAME_STATE_CONTROLS"


def _flag(val: object) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def _norm_dt(value: Any) -> str:
    ts = parse_ts(str(value))
    if ts is None:
        return str(value)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_g3(root: Path) -> tuple[list[str], str | None]:
    if tuple(FEATURES) != tuple(FEATURE_STRATEGIES["POA_GHI"]):
        return [], "FEATURES_POA_GHI != FEATURE_STRATEGIES['POA_GHI']"
    members_path = root / "artifacts/twins/members.csv"
    if not members_path.is_file():
        members_path = root / "artifacts/p02b/P02B_TWIN_MEMBERS.csv"
    if not members_path.is_file():
        return [], "twins members missing"
    plants = []
    with members_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("group_id") == G3_ID:
                plants.append(row["plant_id"])
    plants = sorted(set(plants))
    if len(plants) != 8:
        return plants, f"G3 plant count {len(plants)} != 8"
    return plants, None


def _panel_path(root: Path) -> Path:
    p = root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"
    if p.is_file():
        return p
    return root / "artifacts/panel/plant_panel.parquet"


def load_panel(root: Path, plants: list[str]):
    path = _panel_path(root)
    if not path.is_file():
        return None
    plist = ",".join("'" + p.replace("'", "''") + "'" for p in plants)
    con = duckdb.connect()
    table = con.execute(
        f"""
        SELECT plant_id, CAST(datetime_raw AS VARCHAR) AS datetime_raw, y_dc_normalized, coverage_dc,
               poa_irradiance_wm2, ghi_irradiance_wm2, gri_irradiance_wm2, panel_temperature_celsius,
               ambient_temperature_celsius, wind_speed_ms, dc_any_officially_interpolated,
               interpolated_keys_poa_irradiance_wm2, interpolated_keys_ghi_irradiance_wm2,
               interpolated_keys_gri_irradiance_wm2, interpolated_keys_panel_temperature_celsius,
               interpolated_keys_ambient_temperature_celsius, interpolated_keys_wind_speed_ms
        FROM read_parquet('{str(path).replace("'", "''")}')
        WHERE plant_id IN ({plist})
        """
    ).to_arrow_table()
    con.close()
    return table


def _eligible_g3(poa, cov, y, feats, dc_flag, feat_flags) -> np.ndarray:
    mask = np.isfinite(y) & np.isfinite(poa) & np.isfinite(cov) & (poa > 50.0) & (cov >= 0.80)
    for name in FEATURES:
        mask &= np.isfinite(feats[name])
    any_true = dc_flag == 1.0
    for name in FEATURES:
        any_true = any_true | (feat_flags[name] == 1.0)
    return mask & (~any_true)


def splits_and_index(table, plants: list[str]):
    pids = table["plant_id"].to_pylist()
    dts = [_norm_dt(x) for x in table["datetime_raw"].to_pylist()]
    ts_all = [parse_ts(x) for x in dts]
    y_all = np.array(table[TARGET_PRIMARY].to_pylist(), dtype=float)
    poa_all = np.array(table["poa_irradiance_wm2"].to_pylist(), dtype=float)
    cov_all = np.array(table["coverage_dc"].to_pylist(), dtype=float)
    feat_all = {f: np.array(table[f].to_pylist(), dtype=float) for f in FEATURES}
    dc_all = _bool01(table["dc_any_officially_interpolated"].to_pylist())
    flag_all = {f: _bool01(table[FEATURE_INTERP[f]].to_pylist()) for f in FEATURES}
    by_plant: dict[str, list[int]] = {p: [] for p in plants}
    for i, pid in enumerate(pids):
        if pid in by_plant:
            by_plant[str(pid)].append(i)
    feat_by: dict[tuple[str, str], np.ndarray] = {}
    y_map: dict[tuple[str, str], float] = {}
    nfeat = len(FEATURES)
    for i, pid in enumerate(pids):
        dt = dts[i]
        vals = [feat_all[f][i] for f in FEATURES]
        if any(v is None or not np.isfinite(v) for v in vals):
            continue
        key = (str(pid), dt)
        feat_by[key] = np.array(vals, dtype=float)
        if np.isfinite(y_all[i]):
            y_map[key] = float(y_all[i])
    splits: dict[str, dict[str, list[str]]] = {}
    sentinel = datetime.max.replace(tzinfo=timezone.utc)
    for pid in plants:
        ix = np.array(by_plant[pid], dtype=int)
        if ix.size == 0:
            splits[pid] = {"train": [], "validation": [], "test": []}
            continue
        elig = _eligible_g3(
            poa_all[ix], cov_all[ix], y_all[ix], {k: v[ix] for k, v in feat_all.items()}, dc_all[ix], {k: v[ix] for k, v in flag_all.items()}
        )
        local_ts = [ts_all[int(j)] for j in ix]
        local_dt = [dts[int(j)] for j in ix]
        order = [int(k) for k in np.where(elig)[0]]
        order.sort(key=lambda k: (local_ts[k] if local_ts[k] is not None else sentinel, local_dt[k]))
        n = len(order)
        buckets = {"train": [], "validation": [], "test": []}
        for rank, k in enumerate(order):
            buckets[assign_post_eligibility_split(n, rank)].append(local_dt[k])
        splits[pid] = buckets
    return feat_by, y_map, splits, nfeat


def build_engine_n(pid: str, train_dts: list[str], feat_by: dict, q: float, nfeat: int, names: tuple[str, ...]):
    rows = []
    for dt in train_dts:
        vec = feat_by.get((pid, dt))
        if vec is None or not np.all(np.isfinite(vec)):
            return {"computable": False, "reason": "missing_or_nonfinite_train_features"}
        rows.append(vec)
    if len(rows) < 5:
        return {"computable": False, "reason": "insufficient_train_rows_for_k5"}
    X = np.vstack(rows)
    medians, iqrs = [], []
    for m, name in enumerate(names):
        med, iqr = source_median_iqr(X[:, m])
        if not iqr_is_valid(iqr):
            return {"computable": False, "reason": f"degenerate_iqr:{name}"}
        medians.append(med)
        iqrs.append(iqr)
    z = np.column_stack([standardize(X[:, m], medians[m], iqrs[m]) for m in range(nfeat)])
    dist = loo_kth_distances(z, 5)
    tau = q_linear(dist, q)
    if not np.isfinite(tau):
        return {"computable": False, "reason": "nonfinite_threshold"}
    nn = NearestNeighbors(n_neighbors=5, algorithm=NN_ALGORITHM, metric="euclidean")
    nn.fit(z)
    return {"computable": True, "reason": "", "medians": medians, "iqrs": iqrs, "nn": nn, "tau": float(tau), "nfeat": nfeat}


def score_support_n(engine: dict[str, Any], tgt: str, test_dts: list[str], feat_by: dict) -> set[str]:
    if not engine.get("computable"):
        return set()
    nfeat = int(engine["nfeat"])
    rows, keep = [], []
    for dt in test_dts:
        vec = feat_by.get((tgt, dt))
        if vec is None or not np.all(np.isfinite(vec)):
            continue
        rows.append(vec)
        keep.append(dt)
    if not rows:
        return set()
    X = np.vstack(rows)
    z = np.column_stack([standardize(X[:, m], engine["medians"][m], engine["iqrs"][m]) for m in range(nfeat)])
    dist, _ = engine["nn"].kneighbors(z, n_neighbors=5, return_distance=True)
    d5 = dist[:, 4]
    return {keep[i] for i, ok in enumerate(d5 <= engine["tau"]) if ok}


def _xy(pid: str, dts: list[str], feat_by, y_map):
    X, y = [], []
    for dt in dts:
        vec = feat_by.get((pid, dt))
        yv = y_map.get((pid, dt))
        if vec is None or yv is None:
            continue
        X.append(vec)
        y.append(yv)
    return X, y


def fit_models(plants: list[str], splits, feat_by, y_map) -> dict[str, Any]:
    models = {}
    for pid in plants:
        Xtr, ytr = _xy(pid, splits[pid]["train"], feat_by, y_map)
        Xva, yva = _xy(pid, splits[pid]["validation"], feat_by, y_map)
        if len(Xtr) < 8 or len(Xva) < 4:
            models[pid] = None
            continue
        try:
            _p, _m, est, _c = _fit_select(FAM_SPLINE, SPLINE_GRID, np.vstack(Xtr), np.array(ytr), np.vstack(Xva), np.array(yva))
            models[pid] = est
        except Exception:
            models[pid] = None
    return models


def build_engines(plants: list[str], splits, feat_by) -> dict[str, Any]:
    q = float(CS4["q"])
    nfeat = len(FEATURES)
    return {pid: build_engine_n(pid, splits[pid]["train"], feat_by, q, nfeat, FEATURES) for pid in plants}


def compute_directional(plants: list[str], splits, feat_by, y_map, models, engines) -> list[dict[str, Any]]:
    rows = []
    for src, tgt in product(plants, plants):
        if src == tgt:
            continue
        rec = {
            "group_id": G3_ID,
            "source_plant_id": src,
            "target_plant_id": tgt,
            "model_family": FAM_SPLINE,
            "support_rule": "CS4",
            "computable": False,
        }
        if models.get(src) is None or models.get(tgt) is None:
            rec["reason"] = "model_unfit"
            rows.append(rec)
            continue
        sup = score_support_n(engines[src], tgt, splits[tgt]["test"], feat_by)
        if not sup:
            rec["reason"] = "empty_cs4_support"
            rec["n_supported"] = 0
            rows.append(rec)
            continue
        y_true, y_tr, y_loc = [], [], []
        for dt in sorted(sup):
            vec = feat_by.get((tgt, dt))
            yv = y_map.get((tgt, dt))
            if vec is None or yv is None:
                continue
            y_true.append(yv)
            y_tr.append(float(models[src].predict(vec.reshape(1, -1))[0]))
            y_loc.append(float(models[tgt].predict(vec.reshape(1, -1))[0]))
        if len(y_true) < 1:
            rec["reason"] = "missing_predictions_on_support"
            rec["n_supported"] = len(sup)
            rows.append(rec)
            continue
        yt = np.array(y_true)
        tmae = float(np.mean(np.abs(np.array(y_tr) - yt)))
        lmae = float(np.mean(np.abs(np.array(y_loc) - yt)))
        otg = tmae - lmae
        rec.update(
            {
                "computable": True,
                "reason": "ok",
                "n_supported": len(y_true),
                "transfer_mae": tmae,
                "local_same_rows_mae": lmae,
                "otg_abs": otg,
                "otg_rel": (otg / lmae) if lmae != 0 else None,
                "otg_rel_defined": lmae != 0,
            }
        )
        rows.append(rec)
    return rows


def rq1_paths(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float | None]:
    ok = [r for r in rows if r.get("computable")]
    by_t: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        by_t[str(r["target_plant_id"])].append(float(r["otg_abs"]))
    trows = []
    for t in sorted(by_t):
        vals = by_t[t]
        trows.append({"target_id": t, "n_computable_sources": len(vals), "rq1_j": float(sum(vals) / len(vals)), "min_otg": min(vals), "max_otg": max(vals)})
    fleet = float(sum(r["rq1_j"] for r in trows) / len(trows)) if trows else None
    t2 = []
    targets = []
    for r in ok:
        t = str(r["target_plant_id"])
        if t not in targets:
            targets.append(t)
    for t in sorted(targets):
        acc, n = 0.0, 0
        mn = mx = None
        for r in ok:
            if str(r["target_plant_id"]) != t:
                continue
            v = float(r["otg_abs"])
            acc += v
            n += 1
            mn = v if mn is None or v < mn else mn
            mx = v if mx is None or v > mx else mx
        t2.append({"target_id": t, "n_computable_sources": n, "rq1_j": acc / n, "min_otg": mn, "max_otg": mx})
    if [r["target_id"] for r in trows] != [r["target_id"] for r in t2]:
        return trows, None
    for a, b in zip(trows, t2):
        if not math.isclose(a["rq1_j"], b["rq1_j"], rel_tol=0.0, abs_tol=TOL):
            return trows, None
    return trows, fleet


def asymmetry_rows(directional: list[dict[str, Any]], plants: list[str]) -> list[dict[str, Any]]:
    lookup = {(r["source_plant_id"], r["target_plant_id"]): r for r in directional}
    out = []
    order = sorted(plants)
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            ab, ba = lookup.get((a, b)), lookup.get((b, a))
            ab_ok = bool(ab and ab.get("computable"))
            ba_ok = bool(ba and ba.get("computable"))
            both = ab_ok and ba_ok
            signed = (float(ab["otg_abs"]) - float(ba["otg_abs"])) if both else None
            out.append(
                {
                    "group_id": G3_ID,
                    "plant_a": a,
                    "plant_b": b,
                    "a_to_b_computable": ab_ok,
                    "b_to_a_computable": ba_ok,
                    "otg_a_to_b": float(ab["otg_abs"]) if ab_ok else None,
                    "otg_b_to_a": float(ba["otg_abs"]) if ba_ok else None,
                    "asymmetry_signed": signed,
                    "asymmetry_abs": abs(signed) if signed is not None else None,
                    "asymmetry_computable": both,
                }
            )
    return out


def rq2_paths(asym: list[dict[str, Any]]) -> tuple[float | None, bool]:
    ok = [r for r in asym if r.get("asymmetry_computable")]
    by_p: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        v = float(r["asymmetry_abs"])
        by_p[str(r["plant_a"])].append(v)
        by_p[str(r["plant_b"])].append(v)
    if not by_p:
        return None, True
    p1 = float(sum(sum(v) / len(v) for v in by_p.values()) / len(by_p))
    plants = []
    for r in ok:
        for p in (str(r["plant_a"]), str(r["plant_b"])):
            if p not in plants:
                plants.append(p)
    acc_p = []
    for p in plants:
        vals = []
        for r in ok:
            if str(r["plant_a"]) == p or str(r["plant_b"]) == p:
                vals.append(float(r["asymmetry_abs"]))
        acc_p.append(sum(vals) / len(vals))
    p2 = float(sum(acc_p) / len(acc_p))
    return p1, math.isclose(p1, p2, rel_tol=0.0, abs_tol=TOL)


def enumerate_g3_rq3_candidates(meta, plants: list[str], group_of: dict[str, str]) -> list[dict[str, Any]]:
    all_ids = sorted(meta.keys())
    rows = []
    for tgt in plants:
        tstate = meta.get(tgt, {}).get("state")
        tcanon = meta.get(tgt, {}).get("canon") or {}
        gid = group_of.get(tgt)
        for cand in all_ids:
            if cand == tgt:
                continue
            if group_of.get(cand) == gid:
                continue
            cstate = meta.get(cand, {}).get("state")
            if not tstate or cstate != tstate:
                continue
            comp = ranking_components(tcanon, meta.get(cand, {}).get("canon") or {})
            if comp is None:
                continue
            rows.append({"target_id": tgt, "control_id": cand, "target_state": tstate, "structural_candidate": True, "target_group_id": gid})
    return rows


def members_file(root: Path) -> Path:
    path = root / "artifacts/twins/members.csv"
    if path.is_file():
        return path
    return root / "artifacts/p02b/P02B_TWIN_MEMBERS.csv"


def load_members(root: Path) -> list[dict[str, str]]:
    with members_file(root).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_maps(members: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    group_of: dict[str, str] = {}
    members_by_group: dict[str, list[str]] = defaultdict(list)
    for row in members:
        pid, gid = row["plant_id"], row["group_id"]
        group_of[pid] = gid
        members_by_group[gid].append(pid)
    return group_of, dict(members_by_group)


def xy_counts(pid: str, splits, feat_by, y_map) -> tuple[int, int]:
    ntr = len(_xy(pid, splits.get(pid, {}).get("train") or [], feat_by, y_map)[0])
    nva = len(_xy(pid, splits.get(pid, {}).get("validation") or [], feat_by, y_map)[0])
    return ntr, nva


def ingest_plants(root: Path, plants: list[str]):
    table = load_panel(root, plants)
    if table is None:
        return None, None, None, "panel missing"
    feat_by, y_map, splits, _nfeat = splits_and_index(table, plants)
    return feat_by, y_map, splits, None


def merge_index(a, b):
    if a is None:
        return b
    if b is None:
        return a
    out = dict(a)
    out.update(b)
    return out


def annotate_g3_rq3(cands: list[dict[str, Any]], plants: list[str], members_by_group, engines, splits, feat_by, y_map) -> list[dict[str, Any]]:
    out = []
    for rec in cands:
        tgt = rec["target_id"]
        ctrl = rec["control_id"]
        ntr, nva = xy_counts(ctrl, splits, feat_by, y_map)
        twins = [p for p in members_by_group.get(G3_ID, plants) if p != tgt]
        test_dts = (splits.get(tgt) or {}).get("test") or []
        ctrl_eng = engines.get(ctrl) or {"computable": False}
        ctrl_sup = score_support_n(ctrl_eng, tgt, test_dts, feat_by)
        has_sup = False
        for src in twins:
            src_eng = engines.get(src) or {"computable": False}
            twin_sup = score_support_n(src_eng, tgt, test_dts, feat_by)
            if twin_sup & ctrl_sup:
                has_sup = True
                break
        fit_ok = ntr >= 8 and nva >= 4
        row = dict(rec)
        row["n_train_xy"] = ntr
        row["n_val_xy"] = nva
        row["model_fit_executable"] = fit_ok
        row["has_any_cs4_pairwise_support"] = has_sup
        row["structurally_executable"] = bool(fit_ok and has_sup)
        row["eligibility_status"] = (
            "executable" if row["structurally_executable"] else ("unfit" if not fit_ok else "no_cs4_support")
        )
        out.append(row)
    return out


def _mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def compute_g3_rq3(
    exec_rows: list[dict[str, Any]],
    plants: list[str],
    members_by_group,
    engines,
    splits,
    feat_by,
    y_map,
    models,
) -> list[dict[str, Any]]:
    rows = []
    twins_all = members_by_group.get(G3_ID, plants)
    for rec in exec_rows:
        tgt = rec["target_id"]
        ctrl = rec["control_id"]
        test_dts = (splits.get(tgt) or {}).get("test") or []
        ctrl_sup = score_support_n(engines.get(ctrl) or {}, tgt, test_dts, feat_by)
        for src in [p for p in twins_all if p != tgt]:
            twin_sup = score_support_n(engines.get(src) or {}, tgt, test_dts, feat_by)
            inter = twin_sup & ctrl_sup
            base = {
                "target_id": tgt,
                "twin_source_id": src,
                "control_id": ctrl,
                "target_state": rec.get("target_state"),
                "orientation": "CONTROL_MINUS_TWIN",
            }
            if models.get(ctrl) is None or models.get(src) is None or models.get(tgt) is None:
                rows.append({**base, "computable": False, "reason": "model_unfit", "common_row_count": len(inter)})
                continue
            if not inter:
                rows.append({**base, "computable": False, "reason": "empty_pairwise_intersection", "common_row_count": 0})
                continue
            y_true, y_twin, y_loc, y_ctrl = [], [], [], []
            for dt in sorted(inter):
                vec = feat_by.get((tgt, dt))
                yv = y_map.get((tgt, dt))
                if vec is None or yv is None:
                    continue
                y_true.append(yv)
                y_twin.append(float(models[src].predict(vec.reshape(1, -1))[0]))
                y_loc.append(float(models[tgt].predict(vec.reshape(1, -1))[0]))
                y_ctrl.append(float(models[ctrl].predict(vec.reshape(1, -1))[0]))
            if len(y_true) < 1:
                rows.append({**base, "computable": False, "reason": "missing_predictions_on_intersection", "common_row_count": len(inter)})
                continue
            yt = np.array(y_true)
            otg_twin = _mae(yt, y_twin) - _mae(yt, y_loc)
            otg_ctrl = _mae(yt, y_ctrl) - _mae(yt, y_loc)
            rows.append(
                {
                    **base,
                    "computable": True,
                    "reason": "ok",
                    "common_row_count": len(y_true),
                    "twin_otg_common_rows": otg_twin,
                    "control_otg_common_rows": otg_ctrl,
                    "contrast_control_minus_twin": otg_ctrl - otg_twin,
                }
            )
    return rows


def _ok(row: dict[str, Any], key: str = "computable") -> bool:
    val = row.get(key)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"true", "1", "yes"}


def noncomputable_reasons(directional: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in directional:
        if _ok(row):
            continue
        reason = str(row.get("reason") or "unspecified")
        out[reason] = out.get(reason, 0) + 1
    return dict(sorted(out.items()))


def n_bidirectional(asym: list[dict[str, Any]]) -> int:
    return sum(1 for row in asym if _ok(row, "asymmetry_computable"))


def rq3_target_summary_rows(
    comps: list[dict[str, Any]],
    exec_rows: list[dict[str, Any]],
    plants: list[str],
) -> list[dict[str, Any]]:
    exec_by: dict[str, set[str]] = defaultdict(set)
    for row in exec_rows:
        exec_by[str(row["target_id"])].add(str(row["control_id"]))
    out = []
    for tgt in plants:
        valid = [r for r in comps if str(r["target_id"]) == tgt and _ok(r)]
        if not valid:
            continue
        by_src: dict[str, list[float]] = defaultdict(list)
        controls: set[str] = set()
        for row in valid:
            by_src[str(row["twin_source_id"])].append(float(row["contrast_control_minus_twin"]))
            controls.add(str(row["control_id"]))
        src_means = [sum(vals) / len(vals) for _, vals in sorted(by_src.items())]
        out.append(
            {
                "target_id": tgt,
                "n_executable_controls": len(exec_by.get(tgt, set())),
                "n_distinct_controls_contributing": len(controls),
                "n_twin_sources_contributing": len(by_src),
                "n_valid_source_control_comparisons": len(valid),
                "target_mean_control_minus_twin": float(sum(src_means) / len(src_means)),
            }
        )
    return out


def fleet_from_target_rows(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return float(sum(float(r["target_mean_control_minus_twin"]) for r in rows) / len(rows))


PROVENANCE_REQUIRED_KEYS = (
    "repository_head",
    "canonical_sources",
    "group_id",
    "plants",
    "features",
    "target",
    "split",
    "model_family",
    "support_rule",
    "otg_definition",
    "script_path",
    "helper_path",
    "reused_upstream_artifacts",
    "newly_computed_artifacts",
    "generated_at_utc",
    "n_expected_directed_transfers",
    "n_computable_directed_transfers",
    "n_noncomputable_directed_transfers",
    "n_bidirectionally_computable_unordered_pairs",
    "rq3_n_executable_target_control_pairs",
    "rq3_n_valid_source_control_comparisons",
    "outputs",
)
