"""Outcome-blind structural helpers for S09 Stage 34. No scoring of models or OTG."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.lib.split_support import (
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

FEATURES = FEATURE_STRATEGIES["six_core"]
INTERP_COLS = (
    "dc_any_officially_interpolated",
    "interpolated_keys_poa_irradiance_wm2",
    "interpolated_keys_ghi_irradiance_wm2",
    "interpolated_keys_gri_irradiance_wm2",
    "interpolated_keys_panel_temperature_celsius",
    "interpolated_keys_ambient_temperature_celsius",
    "interpolated_keys_wind_speed_ms",
)
MAP = {
    "PS_002": "PS_013",
    "PS_003": "PS_013",
    "PS_035": "PS_006",
    "PS_039": "PS_006",
    "PS_042": "PS_010",
    "PS_043": "PS_010",
    "PS_044": "PS_010",
    "PS_045": "PS_010",
    "PS_046": "PS_010",
    "PS_047": "PS_010",
    "PS_048": "PS_010",
    "PS_049": "PS_010",
    "PS_050": "PS_010",
    "PS_051": "PS_010",
}


def _false_or_null(val: Any) -> bool:
    return val is None or val is False


def row_eligible(row: dict[str, Any], *, target: str, coverage_col: str, coverage_thr: float, poa: float, interp: str) -> bool:
    if row.get(target) is None:
        return False
    if row.get("poa_irradiance_wm2") is None or not (float(row["poa_irradiance_wm2"]) > poa):
        return False
    cov = row.get(coverage_col)
    if cov is None or float(cov) < coverage_thr:
        return False
    for name in FEATURES:
        if row.get(name) is None:
            return False
    if interp == "exclude_TRUE":
        for col in INTERP_COLS:
            if not _false_or_null(row.get(col)):
                return False
    return True


def reconstruct_splits(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[str(row["plant_id"])].append(row)
    out: dict[str, dict[str, list[str]]] = {}
    sentinel = datetime.max.replace(tzinfo=timezone.utc)
    for pid, items in by.items():
        order = sorted(items, key=lambda r: (parse_ts(str(r["datetime_raw"])) or sentinel, str(r["datetime_raw"])))
        n = len(order)
        buckets = {"train": [], "validation": [], "test": []}
        for rank, rec in enumerate(order):
            buckets[assign_post_eligibility_split(n, rank)].append(str(rec["datetime_raw"]))
        out[pid] = buckets
    return out


def plant_audit(pid: str, rows: list[dict[str, Any]], splits: dict[str, list[str]], *, arm_id: str) -> dict[str, Any]:
    dts = [str(r["datetime_raw"]) for r in rows]
    tss = [parse_ts(dt) for dt in dts]
    days = {t.date() for t in tss if t is not None}
    stamps = sorted(t for t in tss if t is not None)
    feat_ok = all(all(r.get(name) is not None for name in FEATURES) for r in rows) if rows else False
    interp_avail = all(any(c in r for c in INTERP_COLS) for r in rows) if rows else False
    parts = {}
    nonempty = True
    for name in ("train", "validation", "test"):
        keys = splits.get(name) or []
        pts = [parse_ts(x) for x in keys]
        ds = {t.date() for t in pts if t is not None}
        finite = [t for t in pts if t is not None]
        parts[f"n_{name}"] = len(keys)
        parts[f"n_{name}_days"] = len(ds)
        parts[f"first_{name}_ts"] = min(finite).isoformat() if finite else ""
        parts[f"last_{name}_ts"] = max(finite).isoformat() if finite else ""
        parts[f"{name}_nonempty"] = bool(keys)
        nonempty = nonempty and bool(keys)
    chrono = True
    tr, va, te = splits.get("train") or [], splits.get("validation") or [], splits.get("test") or []
    def last(xs: list[str]):
        ts = [parse_ts(x) for x in xs]
        ts = [t for t in ts if t is not None]
        return max(ts) if ts else None
    def first(xs: list[str]):
        ts = [parse_ts(x) for x in xs]
        ts = [t for t in ts if t is not None]
        return min(ts) if ts else None
    if last(tr) and first(va) and last(tr) > first(va):
        chrono = False
    if last(va) and first(te) and last(va) > first(te):
        chrono = False
    return {
        "arm_id": arm_id,
        "plant_id": pid,
        "n_target_observed": len(rows),
        "n_eligible": len(rows),
        "n_eligible_days": len(days),
        "first_eligible_ts": stamps[0].isoformat() if stamps else "",
        "last_eligible_ts": stamps[-1].isoformat() if stamps else "",
        "required_feature_complete": feat_ok,
        "official_interpolation_flag_available": interp_avail,
        "partitions_nonempty": nonempty,
        "chronology_ok": chrono,
        **parts,
    }


def feat_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    for r in rows:
        vec = np.array([float(r[name]) for name in FEATURES], dtype=float)
        out[(str(r["plant_id"]), str(r["datetime_raw"]))] = vec
    return out


def build_engine(pid: str, train_dts: list[str], feat_by: dict[tuple[str, str], np.ndarray], q: float) -> dict[str, Any]:
    rows = []
    for dt in train_dts:
        vec = feat_by.get((pid, dt))
        if vec is None or not np.all(np.isfinite(vec)):
            return {"computable": False, "reason": "missing_or_nonfinite_train_features", "n_train": len(train_dts)}
        rows.append(vec)
    if len(rows) < 5:
        return {"computable": False, "reason": "insufficient_train_rows_for_k5", "n_train": len(rows)}
    X = np.vstack(rows)
    medians = []
    iqrs = []
    for m, name in enumerate(FEATURES):
        med, iqr = source_median_iqr(X[:, m])
        if not iqr_is_valid(iqr):
            return {"computable": False, "reason": f"degenerate_iqr:{name}", "n_train": len(rows)}
        medians.append(med)
        iqrs.append(iqr)
    z = np.column_stack([standardize(X[:, m], medians[m], iqrs[m]) for m in range(6)])
    dist = loo_kth_distances(z, 5)
    tau = q_linear(dist, q)
    if not np.isfinite(tau):
        return {"computable": False, "reason": "nonfinite_threshold", "n_train": len(rows)}
    nn = NearestNeighbors(n_neighbors=5, algorithm=NN_ALGORITHM, metric="euclidean")
    nn.fit(z)
    return {"computable": True, "reason": "", "n_train": len(rows), "medians": medians, "iqrs": iqrs, "nn": nn, "tau": float(tau)}


def score_support(engine: dict[str, Any], tgt: str, test_dts: list[str], feat_by: dict[tuple[str, str], np.ndarray]) -> set[str]:
    if not engine.get("computable"):
        return set()
    rows = []
    keep: list[str] = []
    for dt in test_dts:
        vec = feat_by.get((tgt, dt))
        if vec is None or not np.all(np.isfinite(vec)):
            continue
        rows.append(vec)
        keep.append(dt)
    if not rows:
        return set()
    X = np.vstack(rows)
    z = np.column_stack([standardize(X[:, m], engine["medians"][m], engine["iqrs"][m]) for m in range(6)])
    dist, _ = engine["nn"].kneighbors(z, n_neighbors=5, return_distance=True)
    d5 = dist[:, 4]
    tau = engine["tau"]
    return {keep[i] for i, ok in enumerate(d5 <= tau) if ok}


def n_days(keys: set[str] | list[str]) -> int:
    days = set()
    for dt in keys:
        t = parse_ts(str(dt))
        if t is not None:
            days.add(t.date())
    return len(days)
