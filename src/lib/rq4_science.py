"""RQ4 confirmatory scoring helpers (frozen protocol)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.io import dump_json, sha256_file, write_csv
from src.lib.models import eligible_mask
from src.lib.split_support import FEATURE_INTERP
from src.lib.rq4_const import (
    BOOT_B,
    BOOT_SEED,
    CS4_K,
    CS4_Q,
    CUT_SPACING,
    FEATURES,
    H_DAYS,
    HGB_GRID,
    K_GRID,
    K_MAX,
    L_DAYS,
    M_SHORTLIST,
    META_BOOL,
    META_CAT,
    META_FIELDS,
    META_NUMERIC,
    META_PATH,
    OUT_DIR,
    PANEL_PATH,
    PARTITION_PATH,
    RHO,
    SPLINE_GRID,
    SUMMARY_PATH,
    TARGET,
    TAU_PRIMARY,
    TAU_STRICT,
    TRAIN_FRAC,
)
from src.lib.split_support import NN_ALGORITHM, QUANTILE_METHOD, iqr_is_valid, parse_ts, q_linear, source_median_iqr, standardize


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode("utf-8")).hexdigest()


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


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def day0(ts: datetime) -> datetime:
    ts = _as_utc(ts)
    return datetime(ts.year, ts.month, ts.day, tzinfo=timezone.utc)


def to_ns(ts: datetime) -> np.datetime64:
    return np.datetime64(_as_utc(ts).replace(tzinfo=None), "ns")


@dataclass
class PlantData:
    plant_id: str
    ts: np.ndarray
    y: np.ndarray
    poa: np.ndarray
    cov: np.ndarray
    feats: dict[str, np.ndarray]
    elig: np.ndarray
    dcov: np.ndarray
    first: datetime
    last: datetime


def reconstruct_cohorts(root: Path) -> dict[str, Any]:
    part = root / PARTITION_PATH
    summary = root / SUMMARY_PATH
    if not part.is_file() or not summary.is_file():
        return {"ok": False, "reason": "missing_partition_or_summary"}
    scope = json.loads(part.read_text(encoding="utf-8"))
    targets = list(scope.get("primary_plants") or [])
    if sorted(targets) != targets:
        targets = sorted(targets)
    if len(targets) != 14 or len(set(targets)) != 14:
        return {"ok": False, "reason": f"target_count={len(targets)}"}
    with summary.open(encoding="utf-8", newline="") as handle:
        all_ids = [row["plant_id"] for row in csv.DictReader(handle) if row.get("plant_id")]
    all_ids = sorted(set(all_ids))
    tset = set(targets)
    sources = [p for p in all_ids if p not in tset]
    if not sources:
        return {"ok": False, "reason": "empty_source_cohort"}
    if tset & set(sources):
        return {"ok": False, "reason": "target_source_overlap"}
    return {
        "ok": True,
        "target_ids": targets,
        "source_ids": sources,
        "target_count": len(targets),
        "source_count": len(sources),
        "target_ids_sha256": ids_sha256(targets),
        "source_ids_sha256": ids_sha256(sources),
        "target_partition_artifact_path": PARTITION_PATH,
        "target_partition_artifact_sha256": sha256_file(part),
        "source_population_artifact_path": SUMMARY_PATH,
        "source_population_artifact_sha256": sha256_file(summary),
        "source_population_provenance": "P02C_PLANT_SUMMARY plants minus PRIMARY_SCOPE confirmatory targets",
        "target_source_disjoint": True,
    }


def load_metadata(root: Path, plant_ids: list[str]) -> dict[str, dict[str, Any]]:
    path = root / META_PATH
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pid = row.get("id") or row.get("plant_id")
            if pid in plant_ids:
                rec: dict[str, Any] = {}
                for name in META_FIELDS:
                    raw = row.get(name)
                    if raw is None or str(raw).strip() == "":
                        rec[name] = None
                    elif name in META_BOOL:
                        rec[name] = str(raw).strip().lower() in {"true", "1", "yes"}
                    elif name in META_NUMERIC:
                        rec[name] = float(raw)
                    else:
                        rec[name] = str(raw).strip()
                out[pid] = rec
    return out


def gower_matrix(meta: dict[str, dict[str, Any]], ids: list[str]) -> dict[tuple[str, str], float]:
    ranges: dict[str, tuple[float, float]] = {}
    for name in META_NUMERIC:
        vals = [meta[p][name] for p in ids if p in meta and meta[p].get(name) is not None and np.isfinite(meta[p][name])]
        if vals:
            ranges[name] = (min(vals), max(vals))
    dist: dict[tuple[str, str], float] = {}
    for a in ids:
        for b in ids:
            if a == b:
                continue
            if a not in meta or b not in meta:
                continue
            parts: list[float] = []
            ma, mb = meta[a], meta[b]
            for name in META_NUMERIC:
                va, vb = ma.get(name), mb.get(name)
                if va is None or vb is None or not np.isfinite(va) or not np.isfinite(vb):
                    continue
                lo, hi = ranges.get(name, (va, va))
                if hi == lo:
                    parts.append(0.0)
                else:
                    parts.append(abs(va - vb) / (hi - lo))
            for name in META_BOOL:
                va, vb = ma.get(name), mb.get(name)
                if va is None or vb is None:
                    continue
                parts.append(0.0 if va == vb else 1.0)
            for name in META_CAT:
                va, vb = ma.get(name), mb.get(name)
                if va is None or vb is None:
                    continue
                parts.append(0.0 if va == vb else 1.0)
            if not parts:
                continue
            dist[(a, b)] = float(np.mean(parts))
    return dist


def load_plants(root: Path, plant_ids: list[str]) -> dict[str, PlantData]:
    import duckdb

    path = root / PANEL_PATH
    plist = ",".join("'" + p.replace("'", "''") + "'" for p in plant_ids)
    cols = [
        "plant_id",
        "datetime_raw",
        TARGET,
        "coverage_dc",
        *FEATURES,
        "dc_any_officially_interpolated",
        *[FEATURE_INTERP[f] for f in FEATURES],
    ]
    con = duckdb.connect()
    table = con.execute(
        f"""
        SELECT {", ".join(cols)}
        FROM read_parquet('{str(path).replace("'", "''")}')
        WHERE plant_id IN ({plist})
        ORDER BY plant_id, datetime_raw
        """
    ).to_arrow_table()
    con.close()
    pids = table["plant_id"].to_pylist()
    dt_raw = table["datetime_raw"].to_pylist()
    ts_list = []
    for x in dt_raw:
        if isinstance(x, datetime):
            ts_list.append(_as_utc(x))
        else:
            parsed = parse_ts(str(x) if x is not None else "")
            ts_list.append(_as_utc(parsed) if parsed else None)
    y = np.array(table[TARGET].to_pylist(), dtype=float)
    poa = np.array(table["poa_irradiance_wm2"].to_pylist(), dtype=float)
    cov = np.array(table["coverage_dc"].to_pylist(), dtype=float)
    feats = {f: np.array(table[f].to_pylist(), dtype=float) for f in FEATURES}
    dc = _bool01(table["dc_any_officially_interpolated"].to_pylist())
    flags = {f: _bool01(table[FEATURE_INTERP[f]].to_pylist()) for f in FEATURES}
    elig = eligible_mask(poa, cov, y, feats, dc, flags)
    dcov = np.isfinite(poa) & np.isfinite(cov) & (poa > 50.0) & (cov >= 0.80)
    for name in FEATURES:
        dcov &= np.isfinite(feats[name])
    any_true = dc == 1.0
    for name in FEATURES:
        any_true = any_true | (flags[name] == 1.0)
    dcov = dcov & (~any_true)
    by: dict[str, list[int]] = {p: [] for p in plant_ids}
    for i, pid in enumerate(pids):
        if pid in by:
            by[pid].append(i)
    ns = np.array(
        [np.datetime64("NaT")] * len(ts_list),
        dtype="datetime64[ns]",
    )
    for i, t in enumerate(ts_list):
        if t is not None:
            ns[i] = to_ns(t)
    out: dict[str, PlantData] = {}
    for pid, idx in by.items():
        if not idx:
            continue
        ix = np.array(idx, dtype=int)
        tsl = ns[ix]
        finite = ~np.isnat(tsl)
        if not np.any(finite):
            continue
        first = tsl[finite].min().astype("datetime64[us]").astype(datetime).replace(tzinfo=timezone.utc)
        last = tsl[finite].max().astype("datetime64[us]").astype(datetime).replace(tzinfo=timezone.utc)
        out[pid] = PlantData(
            plant_id=pid,
            ts=tsl,
            y=y[ix],
            poa=poa[ix],
            cov=cov[ix],
            feats={k: v[ix] for k, v in feats.items()},
            elig=elig[ix],
            dcov=dcov[ix],
            first=first,
            last=last,
        )
    return out


def cuts_for_target(plant: PlantData) -> list[datetime]:
    start = day0(plant.first) + timedelta(days=L_DAYS)
    last_c = day0(plant.last) - timedelta(days=K_MAX + H_DAYS)
    out: list[datetime] = []
    c = start
    while c <= last_c:
        r0 = c + timedelta(days=H_DAYS)
        r1 = c + timedelta(days=H_DAYS + H_DAYS)
        in_r = (plant.ts >= to_ns(r0)) & (plant.ts < to_ns(r1)) & plant.elig
        d0 = (plant.ts >= to_ns(c)) & (plant.ts < to_ns(c + timedelta(days=K_MAX))) & plant.dcov
        if int(in_r.sum()) >= 1 and int(d0.sum()) >= 1:
            out.append(c)
        c = c + timedelta(days=CUT_SPACING)
    return out


def chrono_split(mask: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    idx = np.where(mask)[0]
    if idx.size < 2:
        return None
    order = idx[np.argsort(ts[idx], kind="mergesort")]
    n = len(order)
    cut = int(math.floor(TRAIN_FRAC * n))
    if cut <= 0 or cut >= n:
        return None
    train = np.zeros(len(mask), dtype=bool)
    val = np.zeros(len(mask), dtype=bool)
    train[order[:cut]] = True
    val[order[cut:]] = True
    if ts[train].max() >= ts[val].min():
        return None
    return train, val


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
        min_samples_leaf=20,
    )


def xmat(plant: PlantData, mask: np.ndarray) -> np.ndarray:
    return np.column_stack([plant.feats[f][mask] for f in FEATURES])


def fit_family(family: str, plant: PlantData, train: np.ndarray, val: np.ndarray) -> dict[str, Any] | None:
    grid = SPLINE_GRID if family == "SplineRidge" else HGB_GRID
    Xtr, ytr = xmat(plant, train), plant.y[train]
    Xva, yva = xmat(plant, val), plant.y[val]
    if Xtr.shape[0] < 8 or Xva.shape[0] < 1:
        return None
    rows = []
    for params in grid:
        est = _spline(params) if family == "SplineRidge" else _hgb(params)
        mae = float("nan")
        status = "ok"
        try:
            est.fit(Xtr, ytr)
            pred = np.asarray(est.predict(Xva), dtype=float)
            if not np.all(np.isfinite(pred)):
                status = "nonfinite"
                est = None
            else:
                mae = float(np.mean(np.abs(pred - yva)))
        except Exception:  # noqa: BLE001
            status = "fit_error"
            est = None
        rows.append({"params": params, "mae": mae, "status": status, "est": est})
    ok = [r for r in rows if r["status"] == "ok" and np.isfinite(r["mae"])]
    if not ok:
        return None
    min_mae = min(r["mae"] for r in ok)
    winners = [r for r in ok if r["mae"] == min_mae]
    if family == "SplineRidge":
        winners.sort(key=lambda r: (r["params"]["n_knots"], r["params"]["alpha"]))
    else:
        winners.sort(
            key=lambda r: (
                r["params"]["learning_rate"],
                r["params"]["max_leaf_nodes"],
                r["params"]["l2_regularization"],
            )
        )
    best = winners[0]
    return {"params": best["params"], "val_mae": min_mae, "estimator": best["est"]}


def cs4_engine(plant: PlantData, train: np.ndarray) -> dict[str, Any] | None:
    n = int(train.sum())
    if n <= CS4_K:
        return None
    medians: dict[str, float] = {}
    iqrs: dict[str, float] = {}
    cols = []
    for name in FEATURES:
        vals = plant.feats[name][train]
        if vals.size == 0 or not np.all(np.isfinite(vals)):
            return None
        med, iqr = source_median_iqr(vals)
        if not iqr_is_valid(iqr):
            return None
        medians[name] = med
        iqrs[name] = iqr
        cols.append(standardize(vals, med, iqr))
    z = np.column_stack(cols)
    nn = NearestNeighbors(n_neighbors=CS4_K + 1, algorithm=NN_ALGORITHM, metric="euclidean")
    nn.fit(z)
    dist, _ = nn.kneighbors(z, n_neighbors=CS4_K + 1, return_distance=True)
    loo = dist[:, CS4_K]
    thr = q_linear(loo, CS4_Q)
    if not np.isfinite(thr):
        return None
    return {"medians": medians, "iqrs": iqrs, "threshold": float(thr), "nn": nn}


def in_cs4(engine: dict[str, Any], plant: PlantData, mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) == 0:
        return np.zeros(len(mask), dtype=bool)
    cols = []
    for name in FEATURES:
        cols.append(standardize(plant.feats[name][mask], engine["medians"][name], engine["iqrs"][name]))
    z = np.column_stack(cols)
    dist, _ = engine["nn"].kneighbors(z, n_neighbors=CS4_K, return_distance=True)
    d5 = dist[:, CS4_K - 1]
    out = np.zeros(len(mask), dtype=bool)
    out[np.where(mask)[0]] = np.isfinite(d5) & (d5 <= engine["threshold"])
    return out


def mae_on(est, plant: PlantData, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    pred = np.asarray(est.predict(xmat(plant, mask)), dtype=float)
    if not np.all(np.isfinite(pred)):
        return float("nan")
    return float(np.mean(np.abs(pred - plant.y[mask])))


def predict_on(est, plant: PlantData, mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) == 0:
        return np.array([], dtype=float)
    return np.asarray(est.predict(xmat(plant, mask)), dtype=float)


def _write_pq(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pq.write_table(pa.table({name: pa.array([], type=pa.string()) for name in fields}), path, compression="zstd")
        return
    arrays = []
    for name in fields:
        col = [r.get(name) for r in rows]
        if not col:
            arrays.append(pa.array([], type=pa.null()))
            continue
        if all(isinstance(v, (bool, np.bool_)) or v is None for v in col):
            arrays.append(pa.array(col, type=pa.bool_()))
        elif all(isinstance(v, (int, np.integer)) or v is None for v in col) and not any(isinstance(v, bool) for v in col):
            arrays.append(pa.array(col, type=pa.int64()))
        elif all(isinstance(v, (int, float, np.floating, np.integer)) or v is None for v in col) and not any(
            isinstance(v, bool) for v in col
        ):
            arrays.append(pa.array([None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v) for v in col], type=pa.float64()))
        else:
            arrays.append(pa.array([None if v is None else str(v) for v in col], type=pa.string()))
    pq.write_table(pa.table(dict(zip(fields, arrays))), path, compression="zstd")


def _empty_num(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else v


def target_balanced(values_by_target: dict[str, list[float]], how: str) -> float | None:
    summaries = []
    for _t, vals in sorted(values_by_target.items()):
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        if not finite:
            continue
        if how == "mean":
            summaries.append(float(np.mean(finite)))
        elif how == "p75":
            summaries.append(float(np.quantile(finite, 0.75, method=QUANTILE_METHOD)))
        elif how == "p90":
            summaries.append(float(np.quantile(finite, 0.90, method=QUANTILE_METHOD)))
        else:
            summaries.append(float(np.median(finite)))
    if not summaries:
        return None
    if how == "mean":
        return float(np.mean(summaries))
    return float(np.median(summaries))


def bootstrap_ci(values_by_target: dict[str, list[float]], how: str, rng: np.random.Generator) -> tuple[float | None, float | None, float | None]:
    keys = [k for k, v in sorted(values_by_target.items()) if any(x is not None and np.isfinite(x) for x in v)]
    point = target_balanced(values_by_target, how)
    if len(keys) < 1:
        return point, None, None
    draws = np.empty(BOOT_B, dtype=float)
    n = len(keys)
    for b in range(BOOT_B):
        pick = rng.integers(0, n, size=n)
        resampled = {f"{i}": values_by_target[keys[int(j)]] for i, j in enumerate(pick)}
        val = target_balanced(resampled, how)
        draws[b] = val if val is not None else np.nan
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return point, None, None
    lo = float(np.quantile(finite, 0.025, method=QUANTILE_METHOD))
    hi = float(np.quantile(finite, 0.975, method=QUANTILE_METHOD))
    return point, lo, hi


def persistent_k(curve: dict[int, tuple[float | None, float | None]], threshold: float) -> int | None:
    ks = list(K_GRID)
    for i, k in enumerate(ks):
        hi = curve.get(k, (None, None))[1]
        if hi is None or not np.isfinite(hi) or hi > threshold:
            continue
        later_ok = True
        for k2 in ks[i:]:
            hi2 = curve.get(k2, (None, None))[1]
            if hi2 is None or not np.isfinite(hi2) or hi2 > threshold:
                later_ok = False
                break
        if later_ok:
            return int(k)
    return None


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    from scipy.stats import spearmanr

    rho, _p = spearmanr(a, b, nan_policy="omit")
    if rho is None or not np.isfinite(rho):
        return None
    return float(rho)


def execute_confirmatory(root: Path) -> dict[str, Any]:
    cohorts = reconstruct_cohorts(root)
    if not cohorts["ok"]:
        raise RuntimeError(cohorts["reason"])
    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "RQ4_TARGET_COHORT.json", {k: cohorts[k] for k in ("target_ids", "target_count", "target_ids_sha256")})
    dump_json(out / "RQ4_SOURCE_COHORT.json", {k: cohorts[k] for k in ("source_ids", "source_count", "source_ids_sha256", "source_population_provenance")})
    all_ids = cohorts["target_ids"] + cohorts["source_ids"]
    meta = load_metadata(root, all_ids)
    gower = gower_matrix(meta, all_ids)
    plants = load_plants(root, all_ids)
    members: dict[str, str] = {}
    mem_path = root / "artifacts/twins/members.csv"
    if mem_path.is_file():
        with mem_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("plant_id"):
                    members[row["plant_id"]] = row.get("group_id") or ""
    summary_state: dict[str, str] = {}
    with (root / SUMMARY_PATH).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            summary_state[row["plant_id"]] = row.get("brazil_federative_unit") or ""

    model_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    cov_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    hgb_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}

    def source_fit(sid: str, c: datetime, family: str) -> dict[str, Any] | None:
        key = (sid, c.isoformat(), family)
        if key in cache:
            return cache[key]
        plant = plants.get(sid)
        if plant is None:
            cache[key] = None
            return None
        w0, w1 = c - timedelta(days=L_DAYS), c
        win = (plant.ts >= to_ns(w0)) & (plant.ts < to_ns(w1)) & plant.elig
        split = chrono_split(win, plant.ts)
        if split is None:
            cache[key] = None
            return None
        train, val = split
        fitted = fit_family(family, plant, train, val)
        if fitted is None:
            cache[key] = None
            return None
        engine = cs4_engine(plant, train)
        rec = {
            "source_id": sid,
            "cut_c": c.isoformat(),
            "family": family,
            "n_train": int(train.sum()),
            "n_validation": int(val.sum()),
            "params_json": json.dumps(fitted["params"], sort_keys=True),
            "val_mae": fitted["val_mae"],
            "cs4_ok": engine is not None,
            "cs4_threshold": None if engine is None else engine["threshold"],
            "estimator": fitted["estimator"],
            "engine": engine,
            "train_max_ns": int(plant.ts[train].max().astype("datetime64[ns]").astype(np.int64)),
            "val_max_ns": int(plant.ts[val].max().astype("datetime64[ns]").astype(np.int64)),
            "cut_ns": int(to_ns(c).astype("datetime64[ns]").astype(np.int64)),
            "train_end_before_c": bool(plant.ts[train].max() < to_ns(c)),
            "val_end_before_c": bool(plant.ts[val].max() < to_ns(c)),
            "train_frac": float(int(train.sum()) / max(1, int(train.sum()) + int(val.sum()))),
        }
        cache[key] = rec
        return rec

    reason_counts: dict[str, int] = {}

    def bump(reason: str) -> None:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for j in cohorts["target_ids"]:
        tgt = plants.get(j)
        if tgt is None:
            bump("missing_target_panel")
            continue
        cuts = cuts_for_target(tgt)
        print(f"RQ4 target {j} n_cuts={len(cuts)}", flush=True)
        for c in cuts:
            r0 = c + timedelta(days=H_DAYS)
            r1 = r0 + timedelta(days=H_DAYS)
            d0 = (tgt.ts >= to_ns(c)) & (tgt.ts < to_ns(c + timedelta(days=K_MAX))) & tgt.dcov
            denom = int(d0.sum())
            universe = [s for s in cohorts["source_ids"] if gower.get((j, s)) is not None and np.isfinite(gower[(j, s)])]
            universe.sort(key=lambda s: (gower[(j, s)], s))
            gower_short = universe[:M_SHORTLIST] if len(universe) > M_SHORTLIST else universe
            cand: list[str] = []
            engines = {}
            ests = {}
            hgb_ests = {}
            for s in gower_short:
                rec = source_fit(s, c, "SplineRidge")
                coverage = float("nan")
                pass_rho = False
                if rec is not None and rec["engine"] is not None and denom:
                    coverage = float(in_cs4(rec["engine"], tgt, d0).sum() / denom)
                    pass_rho = bool(np.isfinite(coverage) and coverage >= RHO)
                cov_rows.append(
                    {
                        "target_id": j,
                        "source_id": s,
                        "cut_c": c.isoformat(),
                        "coverage": coverage,
                        "rho": RHO,
                        "denominator": denom,
                        "pass_rho": pass_rho,
                        "gower_rank": universe.index(s) if s in universe else None,
                    }
                )
                if rec is None or rec["engine"] is None or not pass_rho:
                    continue
                cand.append(s)
                engines[s] = rec["engine"]
                ests[s] = rec["estimator"]
                hgb_rec = source_fit(s, c, "HistGradientBoosting")
                if hgb_rec is not None:
                    hgb_ests[s] = hgb_rec["estimator"]
            short = cand
            if len(short) < 2:
                bump("lt2_after_support_coverage")
                instance_rows.append(
                    {
                        "target_id": j,
                        "cut_c": c.isoformat(),
                        "informative": False,
                        "reason": "lt2_after_support_coverage",
                        "n_candidates_pre_shortlist": len(universe),
                        "n_gower_shortlist": len(gower_short),
                        "n_candidates": len(short),
                    }
                )
                continue
            # common support intersection on eligible y rows in R
            r_elig = (tgt.ts >= to_ns(r0)) & (tgt.ts < to_ns(r1)) & tgt.elig
            common_r = r_elig.copy()
            for s in short:
                common_r &= in_cs4(engines[s], tgt, r_elig)
            if int(common_r.sum()) < 1:
                bump("empty_common_R")
                instance_rows.append(
                    {
                        "target_id": j,
                        "cut_c": c.isoformat(),
                        "informative": False,
                        "reason": "empty_common_R",
                        "n_candidates": len(short),
                    }
                )
                continue
            loc = np.where(common_r)[0]
            loc_sorted = loc[np.argsort(tgt.ts[loc], kind="mergesort")]
            fp_src = "|".join(str(int(x)) for x in tgt.ts[loc_sorted].astype("datetime64[ns]").astype(np.int64))
            common_r_fp = hashlib.sha256(fp_src.encode()).hexdigest()
            losses = {s: mae_on(ests[s], tgt, common_r) for s in short}
            if any(not np.isfinite(v) for v in losses.values()):
                bump("nonfinite_R_loss")
                continue
            lstar = min(losses.values())
            oracle = sorted((s for s in short), key=lambda s: (losses[s], gower[(j, s)], s))[0]
            mean_l = float(np.mean(list(losses.values())))
            min_g = min(gower[(j, s)] for s in short)
            ties = [s for s in short if gower[(j, s)] == min_g]
            l_meta = float(np.mean([losses[s] for s in ties]))
            # noise floor
            r_days = (tgt.ts[common_r].astype("datetime64[D]") - np.datetime64(r0.date()))
            day_idx = r_days.astype(int)
            block = np.zeros(len(tgt.ts), dtype=object)
            block[:] = ""
            loc = np.where(common_r)[0]
            ab = np.where((day_idx // 7) % 2 == 0, "A", "B")
            block[loc] = ab
            mask_a = common_r & (block == "A")
            mask_b = common_r & (block == "B")
            nf_abs = None
            nf_rel = None
            reg_ab = None
            reg_ba = None
            if int(mask_a.sum()) >= 1 and int(mask_b.sum()) >= 1:
                la = {s: mae_on(ests[s], tgt, mask_a) for s in short}
                lb = {s: mae_on(ests[s], tgt, mask_b) for s in short}
                if all(np.isfinite(v) for v in list(la.values()) + list(lb.values())):
                    sa = min(short, key=lambda s: (la[s], gower[(j, s)], s))
                    sb = min(short, key=lambda s: (lb[s], gower[(j, s)], s))
                    lstar_b = min(lb.values())
                    lstar_a = min(la.values())
                    reg_ab = lb[sa] - lstar_b
                    reg_ba = la[sb] - lstar_a
                    nf_abs = float((reg_ab + reg_ba) / 2.0)
                    if lstar > 0 and np.isfinite(lstar):
                        # relative noise uses R L* when defined; also store block-wise
                        nf_rel_parts = []
                        if lstar_b > 0:
                            nf_rel_parts.append(reg_ab / lstar_b)
                        if lstar_a > 0:
                            nf_rel_parts.append(reg_ba / lstar_a)
                        nf_rel = float(np.mean(nf_rel_parts)) if nf_rel_parts else None
            # chronological half of the same R
            mid = len(loc_sorted) // 2
            first_half = np.zeros(len(tgt.ts), dtype=bool)
            second_half = np.zeros(len(tgt.ts), dtype=bool)
            first_half[loc_sorted[:mid]] = True
            second_half[loc_sorted[mid:]] = True
            chrono_12 = None
            chrono_21 = None
            if int(first_half.sum()) >= 1 and int(second_half.sum()) >= 1:
                l1 = {s: mae_on(ests[s], tgt, first_half) for s in short}
                l2 = {s: mae_on(ests[s], tgt, second_half) for s in short}
                if all(np.isfinite(v) for v in list(l1.values()) + list(l2.values())):
                    s1 = min(short, key=lambda s: (l1[s], gower[(j, s)], s))
                    s2 = min(short, key=lambda s: (l2[s], gower[(j, s)], s))
                    chrono_12 = float(l2[s1] - min(l2.values()))
                    chrono_21 = float(l1[s2] - min(l1.values()))
            def _mask_fp(mask: np.ndarray) -> tuple[str | None, int]:
                loc_m = np.where(mask)[0]
                if loc_m.size == 0:
                    return None, 0
                loc_m = loc_m[np.argsort(tgt.ts[loc_m], kind="mergesort")]
                payload = "|".join(str(int(x)) for x in tgt.ts[loc_m].astype("datetime64[ns]").astype(np.int64))
                return hashlib.sha256(payload.encode()).hexdigest(), int(loc_m.size)

            fp_a, n_a = _mask_fp(mask_a)
            fp_b, n_b = _mask_fp(mask_b)
            fp_1, n_1 = _mask_fp(first_half)
            fp_2, n_2 = _mask_fp(second_half)
            audit_rows.append(
                {
                    "target_id": j,
                    "cut_c": c.isoformat(),
                    "common_r_fp": common_r_fp,
                    "fp_A": fp_a,
                    "fp_B": fp_b,
                    "fp_first_half": fp_1,
                    "fp_second_half": fp_2,
                    "n_common_R": int(common_r.sum()),
                    "n_A": n_a,
                    "n_B": n_b,
                    "n_first_half": n_1,
                    "n_second_half": n_2,
                    "reg_A_to_B": reg_ab,
                    "reg_B_to_A": reg_ba,
                    "chrono_first_to_second_abs": chrono_12,
                    "chrono_second_to_first_abs": chrono_21,
                    "noise_floor_abs": None if reg_ab is None or reg_ba is None else float((reg_ab + reg_ba) / 2.0),
                }
            )
            noise_rows.append(
                {
                    "target_id": j,
                    "cut_c": c.isoformat(),
                    "noise_floor_abs": nf_abs,
                    "noise_floor_rel": nf_rel,
                    "n_common_R": int(common_r.sum()),
                    "n_A": n_a,
                    "n_B": n_b,
                    "chrono_first_to_second_abs": chrono_12,
                    "chrono_second_to_first_abs": chrono_21,
                    "common_r_fp": common_r_fp,
                }
            )
            exact_twin = False
            gj = members.get(j)
            if gj:
                exact_twin = any(members.get(s) == gj for s in short)
            instance_rows.append(
                {
                    "target_id": j,
                    "cut_c": c.isoformat(),
                    "informative": True,
                    "reason": "",
                    "n_candidates_pre_shortlist": len(universe),
                    "n_gower_shortlist": len(gower_short),
                    "n_candidates": len(short),
                    "shortlist": ",".join(short),
                    "oracle_source": oracle,
                    "L_star": lstar,
                    "L_random": mean_l,
                    "L_metadata": l_meta,
                    "n_metadata_ties": len(ties),
                    "display_metadata_source": min(ties),
                    "exact_twin_in_shortlist": exact_twin,
                    "same_state_any": any(summary_state.get(s) == summary_state.get(j) for s in short),
                    "same_structure_any": any(
                        (meta.get(s) or {}).get("structure_type") == (meta.get(j) or {}).get("structure_type") for s in short
                    ),
                    "season": c.month,
                    "noise_floor_abs": nf_abs,
                    "common_r_fp": common_r_fp,
                    "r0": r0.isoformat(),
                    "r1": r1.isoformat(),
                }
            )
            for s in short:
                rec = cache[(s, c.isoformat(), "SplineRidge")]
                model_rows.append({k: rec[k] for k in rec if k not in {"estimator", "engine"}})
                loss_rows.append(
                    {
                        "target_id": j,
                        "source_id": s,
                        "cut_c": c.isoformat(),
                        "family": "SplineRidge",
                        "gower": gower[(j, s)],
                        "L_R": losses[s],
                        "is_oracle": s == oracle,
                    }
                )
            for k in K_GRID:
                if k == 0:
                    lhat = l_meta
                    shat = min(ties)
                    spear = None
                    recov = 1.0 if shat == oracle and len(ties) == 1 else (1.0 / len(ties) if oracle in ties else 0.0)
                else:
                    w1 = c + timedelta(days=k)
                    wmask = (tgt.ts >= to_ns(c)) & (tgt.ts < to_ns(w1)) & tgt.elig
                    common_w = wmask.copy()
                    for s in short:
                        common_w &= in_cs4(engines[s], tgt, wmask)
                    if int(common_w.sum()) < 1:
                        regret_rows.append(
                            {
                                "target_id": j,
                                "cut_c": c.isoformat(),
                                "k": k,
                                "informative_k": False,
                                "reason": "empty_common_Wk",
                            }
                        )
                        continue
                    wloss = {s: mae_on(ests[s], tgt, common_w) for s in short}
                    if any(not np.isfinite(v) for v in wloss.values()):
                        continue
                    shat = min(short, key=lambda s: (wloss[s], gower[(j, s)], s))
                    lhat = losses[shat]
                    recov = 1.0 if shat == oracle else 0.0
                    spear = _spearman([wloss[s] for s in short], [losses[s] for s in short])
                reg_abs = lhat - lstar
                if np.isfinite(lstar) and lstar > 0:
                    reg_rel = reg_abs / lstar
                    inv = 0
                else:
                    reg_rel = None
                    inv = 1
                excess = None if nf_abs is None else reg_abs - nf_abs
                regret_rows.append(
                    {
                        "target_id": j,
                        "cut_c": c.isoformat(),
                        "k": k,
                        "informative_k": True,
                        "s_hat": shat,
                        "L_hat": lhat,
                        "Reg_abs": reg_abs,
                        "Reg_rel": reg_rel,
                        "invalid_rel": inv,
                        "Reg_random": mean_l - lstar,
                        "Reg_metadata": l_meta - lstar,
                        "ExcessNoise": excess,
                        "top1_oracle": recov,
                        "spearman": spear,
                        "n_candidates": len(short),
                        "L_star": lstar,
                        "common_r_fp": common_r_fp,
                    }
                )
            def _hgb_k_loop() -> None:
                if len(hgb_ests) != len(short):
                    return
                hlosses = {s: mae_on(hgb_ests[s], tgt, common_r) for s in short}
                if not all(np.isfinite(v) for v in hlosses.values()):
                    return
                hlstar = min(hlosses.values())
                hties = [s for s in short if gower[(j, s)] == min_g]
                hl_meta = float(np.mean([hlosses[s] for s in hties]))
                for k in K_GRID:
                    if k == 0:
                        hlhat = hl_meta
                    else:
                        wmask = (tgt.ts >= to_ns(c)) & (tgt.ts < to_ns(c + timedelta(days=k))) & tgt.elig
                        common_w = wmask.copy()
                        for s in short:
                            common_w &= in_cs4(engines[s], tgt, wmask)
                        if int(common_w.sum()) < 1:
                            continue
                        wloss = {s: mae_on(hgb_ests[s], tgt, common_w) for s in short}
                        if not all(np.isfinite(v) for v in wloss.values()):
                            continue
                        hshat = min(short, key=lambda s: (wloss[s], gower[(j, s)], s))
                        hlhat = hlosses[hshat]
                    hreg = hlhat - hlstar
                    hgb_rows.append(
                        {
                            "target_id": j,
                            "cut_c": c.isoformat(),
                            "k": k,
                            "Reg_abs": hreg,
                            "Reg_rel": hreg / hlstar if hlstar > 0 else None,
                        }
                    )

            _hgb_k_loop()

    _write_pq(out / "RQ4_SOURCE_MODELS.parquet", model_rows, ["source_id", "cut_c", "family", "n_train", "n_validation", "params_json", "val_mae", "cs4_ok", "cs4_threshold", "train_end_before_c", "val_end_before_c", "train_max_ns", "val_max_ns", "cut_ns", "train_frac"])
    inst_fields = [
        "target_id",
        "cut_c",
        "informative",
        "reason",
        "n_candidates_pre_shortlist",
        "n_gower_shortlist",
        "n_candidates",
        "shortlist",
        "oracle_source",
        "L_star",
        "L_random",
        "L_metadata",
        "n_metadata_ties",
        "display_metadata_source",
        "exact_twin_in_shortlist",
        "same_state_any",
        "same_structure_any",
        "season",
        "noise_floor_abs",
        "common_r_fp",
        "r0",
        "r1",
    ]
    _write_pq(out / "RQ4_DECISION_INSTANCES.parquet", instance_rows, inst_fields)
    _write_pq(out / "RQ4_CANDIDATE_LOSSES.parquet", loss_rows, ["target_id", "source_id", "cut_c", "family", "gower", "L_R", "is_oracle"])
    _write_pq(out / "RQ4_SUPPORT_COVERAGE.parquet", cov_rows, ["target_id", "source_id", "cut_c", "coverage", "rho", "denominator", "pass_rho", "gower_rank"])
    _write_pq(out / "RQ4_REGRET_BY_INSTANCE.parquet", regret_rows, ["target_id", "cut_c", "k", "informative_k", "s_hat", "L_hat", "Reg_abs", "Reg_rel", "invalid_rel", "Reg_random", "Reg_metadata", "ExcessNoise", "top1_oracle", "spearman", "n_candidates", "L_star", "common_r_fp", "reason"] if regret_rows else ["target_id", "cut_c", "k", "informative_k"])
    _write_pq(out / "RQ4_TEMPORAL_NOISE.parquet", noise_rows, ["target_id", "cut_c", "noise_floor_abs", "noise_floor_rel", "n_common_R", "n_A", "n_B", "chrono_first_to_second_abs", "chrono_second_to_first_abs", "common_r_fp"])
    _write_pq(
        out / "RQ4_TEMPORAL_AUDIT.parquet",
        audit_rows,
        [
            "target_id",
            "cut_c",
            "common_r_fp",
            "fp_A",
            "fp_B",
            "fp_first_half",
            "fp_second_half",
            "n_common_R",
            "n_A",
            "n_B",
            "n_first_half",
            "n_second_half",
            "reg_A_to_B",
            "reg_B_to_A",
            "chrono_first_to_second_abs",
            "chrono_second_to_first_abs",
            "noise_floor_abs",
        ],
    )
    _write_pq(out / "RQ4_HGB_SENSITIVITY.parquet", hgb_rows, ["target_id", "cut_c", "k", "Reg_abs", "Reg_rel"] if hgb_rows else ["target_id", "cut_c", "k", "Reg_abs", "Reg_rel"])
    n_inf = sum(1 for r in instance_rows if r.get("informative"))
    return {
        "decision_instance_total": len(instance_rows),
        "decision_instance_informative": n_inf,
        "decision_instance_noninformative": len(instance_rows) - n_inf,
        "noninformative_reason_counts": reason_counts,
        "n_regret_rows": len(regret_rows),
        "n_hgb_rows": len(hgb_rows),
    }


def _group(rows: list[dict[str, Any]], field: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in rows:
        if r.get(field) is None:
            continue
        try:
            v = float(r[field])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        out.setdefault(str(r["target_id"]), []).append(v)
    return out


def finalize_from_artifacts(root: Path, execution_commit: str, spec_sha: str, cohorts: dict[str, Any]) -> dict[str, Any]:
    out = root / OUT_DIR
    regret = pq.read_table(out / "RQ4_REGRET_BY_INSTANCE.parquet").to_pylist()
    inst = pq.read_table(out / "RQ4_DECISION_INSTANCES.parquet").to_pylist()
    noise = pq.read_table(out / "RQ4_TEMPORAL_NOISE.parquet").to_pylist()
    hgb = pq.read_table(out / "RQ4_HGB_SENSITIVITY.parquet").to_pylist() if (out / "RQ4_HGB_SENSITIVITY.parquet").is_file() else []
    informative = [r for r in inst if r.get("informative")]
    rng = np.random.default_rng(BOOT_SEED)
    rng_t1 = np.random.default_rng(BOOT_SEED)
    rng_t2 = np.random.default_rng(BOOT_SEED + 1)

    def rows_k(k: int) -> list[dict[str, Any]]:
        return [r for r in regret if r.get("k") == k and r.get("informative_k")]

    t1_map: dict[str, list[float]] = {}
    t2_map: dict[str, list[float]] = {}
    for r in rows_k(0):
        d = None
        if r.get("Reg_random") is not None and r.get("Reg_metadata") is not None:
            d = float(r["Reg_random"]) - float(r["Reg_metadata"])
        if d is not None and np.isfinite(d):
            t1_map.setdefault(str(r["target_id"]), []).append(d)
    for r in rows_k(45):
        if r.get("Reg_metadata") is None or r.get("Reg_abs") is None:
            continue
        d = float(r["Reg_metadata"]) - float(r["Reg_abs"])
        if np.isfinite(d):
            t2_map.setdefault(str(r["target_id"]), []).append(d)
    t1_mean, t1_lo, t1_hi = bootstrap_ci(t1_map, "mean", rng_t1)
    t1_med, _, _ = bootstrap_ci(t1_map, "median", np.random.default_rng(BOOT_SEED + 2))
    t2_mean, t2_lo, t2_hi = bootstrap_ci(t2_map, "mean", rng_t2)
    t2_med, _, _ = bootstrap_ci(t2_map, "median", np.random.default_rng(BOOT_SEED + 3))

    primary_curve = []
    rel_hi: dict[int, tuple[float | None, float | None]] = {}
    excess_hi: dict[int, tuple[float | None, float | None]] = {}
    boot_draws = []
    for k in K_GRID:
        rk = rows_k(k)
        rel_map = _group(rk, "Reg_rel")
        abs_map = _group(rk, "Reg_abs")
        rec_map = _group(rk, "top1_oracle")
        sp_map = _group(rk, "spearman")
        ex_map = _group(rk, "ExcessNoise")
        cand_map = _group(rk, "n_candidates")
        med_rel, lo, hi = bootstrap_ci(rel_map, "median", np.random.default_rng(BOOT_SEED + 100 + k))
        med_abs, _, _ = bootstrap_ci(abs_map, "median", np.random.default_rng(BOOT_SEED + 200 + k))
        med_ex, elo, ehi = bootstrap_ci(ex_map, "median", np.random.default_rng(BOOT_SEED + 300 + k))
        rel_hi[k] = (med_rel, hi)
        excess_hi[k] = (med_ex, ehi)
        # p75/p90 of target summaries
        summaries = []
        p75s, p90s = [], []
        for t, vals in sorted(rel_map.items()):
            summaries.append(float(np.mean(vals)))
            p75s.append(float(np.quantile(vals, 0.75, method=QUANTILE_METHOD)))
            p90s.append(float(np.quantile(vals, 0.90, method=QUANTILE_METHOD)))
        entry = {
            "k": k,
            "valid_targets": len(rel_map),
            "valid_decision_instances": len(rk),
            "median_candidate_count": target_balanced(cand_map, "median"),
            "median_relative_regret": med_rel,
            "relative_regret_ci95_low": lo,
            "relative_regret_ci95_high": hi,
            "p75_relative_regret": target_balanced(rel_map, "p75"),
            "p90_relative_regret": target_balanced(rel_map, "p90"),
            "median_absolute_regret": med_abs,
            "top1_oracle_recovery": target_balanced(rec_map, "mean"),
            "rank_spearman": target_balanced(sp_map, "median"),
            "invalid_relative_regret_count": int(sum(int(r.get("invalid_rel") or 0) for r in rk)),
            "median_excess_noise": med_ex,
            "excess_noise_ci95_low": elo,
            "excess_noise_ci95_high": ehi,
        }
        primary_curve.append(entry)
        keys = list(rel_map.keys())
        n = len(keys)
        if n:
            local = np.random.default_rng(BOOT_SEED + 400 + k)
            for b in range(BOOT_B):
                pick = local.integers(0, n, size=n)
                resampled = {str(i): rel_map[keys[int(j)]] for i, j in enumerate(pick)}
                boot_draws.append({"k": k, "b": b, "stat": "median_reg_rel", "value": target_balanced(resampled, "median")})

    t1_keys = [tk for tk, v in sorted(t1_map.items()) if any(x is not None and np.isfinite(x) for x in v)]
    n_t1 = len(t1_keys)
    if n_t1:
        rng_t1_draw = np.random.default_rng(BOOT_SEED + 500)
        for b in range(BOOT_B):
            pick = rng_t1_draw.integers(0, n_t1, size=n_t1)
            resampled = {str(i): t1_map[t1_keys[int(j)]] for i, j in enumerate(pick)}
            boot_draws.append({"k": 0, "b": b, "stat": "T1_Delta_meta_mean", "value": target_balanced(resampled, "mean")})

    k_tau = persistent_k(rel_hi, TAU_PRIMARY)
    k_tau_strict = persistent_k(rel_hi, TAU_STRICT)
    k_noise = persistent_k(excess_hi, 0.0)
    if k_tau is not None and k_noise is not None:
        joint = "both_exist"
    elif k_tau is not None:
        joint = "k_tau_only"
    elif k_noise is not None:
        joint = "k_noise_only"
    else:
        joint = "neither"

    nf_map = _group(noise, "noise_floor_abs")
    nf_mean = target_balanced(nf_map, "mean")
    cand_all = [int(r["n_candidates"]) for r in informative if r.get("n_candidates") is not None]
    cand_dist = None
    if cand_all:
        cand_dist = {
            "p25": float(np.quantile(cand_all, 0.25, method=QUANTILE_METHOD)),
            "median": float(np.median(cand_all)),
            "p75": float(np.quantile(cand_all, 0.75, method=QUANTILE_METHOD)),
        }

    table_rows = []
    def add_row(label: str, k, extra: dict[str, Any]):
        table_rows.append({"row": label, "k": k, **extra})

    rnd = rows_k(0)

    def rel_from_abs(field: str, rows: list[dict[str, Any]]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for r in rows:
            lstar = r.get("L_star")
            val = r.get(field)
            if lstar is None or val is None:
                continue
            ls, vv = float(lstar), float(val)
            if np.isfinite(ls) and ls > 0 and np.isfinite(vv):
                out.setdefault(str(r["target_id"]), []).append(vv / ls)
        return out

    TABLE_COLS = [
        "valid_targets",
        "valid_decision_instances",
        "median_candidate_count",
        "median_relative_regret",
        "p75_relative_regret",
        "p90_relative_regret",
        "median_absolute_regret",
        "top1_oracle_recovery",
        "rank_spearman",
        "relative_regret_ci95_low",
        "relative_regret_ci95_high",
    ]

    def pack_from_curve(entry: dict[str, Any]) -> dict[str, Any]:
        return {c: entry.get(c) for c in TABLE_COLS}

    rel_random_map = rel_from_abs("Reg_random", rnd)
    meta_rel_map = rel_from_abs("Reg_metadata", rnd)
    add_row("random_baseline", 0, {
        **{c: None for c in TABLE_COLS},
        "valid_targets": len(rel_random_map),
        "valid_decision_instances": len(rnd),
        "median_candidate_count": primary_curve[0]["median_candidate_count"] if primary_curve else None,
        "median_relative_regret": target_balanced(rel_random_map, "median"),
        "p75_relative_regret": target_balanced(rel_random_map, "p75"),
        "p90_relative_regret": target_balanced(rel_random_map, "p90"),
        "median_absolute_regret": target_balanced(_group(rnd, "Reg_random"), "median"),
        "top1_oracle_recovery": None,
        "rank_spearman": None,
        "relative_regret_ci95_low": None,
        "relative_regret_ci95_high": None,
    })
    add_row("metadata_k0", 0, pack_from_curve(primary_curve[0]) if primary_curve else {})
    for entry in primary_curve:
        if entry["k"] == 0:
            continue
        add_row(f"k_{entry['k']}", entry["k"], pack_from_curve(entry))
    add_row("temporal_noise_reference", None, {
        **{c: None for c in TABLE_COLS},
        "median_absolute_regret": target_balanced(nf_map, "median"),
        "median_relative_regret": target_balanced(_group(noise, "noise_floor_rel"), "median"),
    })
    add_row("oracle", None, {**{c: None for c in TABLE_COLS}, "median_relative_regret": 0.0, "median_absolute_regret": 0.0})

    write_csv(out / "RQ4_PRIMARY_TABLE.csv", table_rows)
    _write_pq(out / "RQ4_PRIMARY_CURVE.parquet", primary_curve, list(primary_curve[0].keys()) if primary_curve else ["k"])
    _write_pq(out / "RQ4_BOOTSTRAP_DRAWS.parquet", boot_draws, ["k", "b", "stat", "value"])

    k45 = rows_k(45)
    inst_idx = {(r["target_id"], r["cut_c"]): r for r in informative}

    def split_rel(pred) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        yes: dict[str, list[float]] = {}
        no: dict[str, list[float]] = {}
        for r in k45:
            inst_r = inst_idx.get((r["target_id"], r["cut_c"]))
            if not inst_r or r.get("Reg_rel") is None:
                continue
            bucket = yes if pred(inst_r) else no
            bucket.setdefault(str(r["target_id"]), []).append(float(r["Reg_rel"]))
        return yes, no

    twin_yes, twin_no = split_rel(lambda i: bool(i.get("exact_twin_in_shortlist")))
    st_yes, st_no = split_rel(lambda i: bool(i.get("same_state_any")))
    su_yes, su_no = split_rel(lambda i: bool(i.get("same_structure_any")))
    secondary = [
        {"name": "exact_twin_present_median_reg_rel_k45", "value": target_balanced(twin_yes, "median")},
        {"name": "exact_twin_absent_median_reg_rel_k45", "value": target_balanced(twin_no, "median")},
        {"name": "same_state_median_reg_rel_k45", "value": target_balanced(st_yes, "median")},
        {"name": "diff_state_median_reg_rel_k45", "value": target_balanced(st_no, "median")},
        {"name": "same_structure_median_reg_rel_k45", "value": target_balanced(su_yes, "median")},
        {"name": "diff_structure_median_reg_rel_k45", "value": target_balanced(su_no, "median")},
        {"name": "chrono_first_to_second_median_abs", "value": target_balanced(_group(noise, "chrono_first_to_second_abs"), "median")},
        {"name": "chrono_second_to_first_median_abs", "value": target_balanced(_group(noise, "chrono_second_to_first_abs"), "median")},
    ]
    by_season: dict[int, dict[str, list[float]]] = {}
    for r in k45:
        inst_r = inst_idx.get((r["target_id"], r["cut_c"]))
        if inst_r and r.get("Reg_rel") is not None:
            season = int(inst_r.get("season") or 0)
            by_season.setdefault(season, {}).setdefault(str(r["target_id"]), []).append(float(r["Reg_rel"]))
    for season, mmap in sorted(by_season.items()):
        secondary.append({"name": f"seasonal_median_reg_rel_k45_month_{season}", "value": target_balanced(mmap, "median")})
    hgb_sum = {}
    hgb_ks = sorted({int(r["k"]) for r in hgb if r.get("k") is not None})
    for k in hgb_ks:
        hk = [r for r in hgb if r.get("k") == k]
        hgb_sum[f"k{k}_median_reg_rel"] = target_balanced(_group(hk, "Reg_rel"), "median")
    hgb_sum["k_grid"] = hgb_ks
    _write_pq(out / "RQ4_SECONDARY_RESULTS.parquet", secondary, ["name", "value"])

    # figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [e["k"] for e in primary_curve]
    ys = [e["median_relative_regret"] if e["median_relative_regret"] is not None else np.nan for e in primary_curve]
    lo = [e["relative_regret_ci95_low"] if e["relative_regret_ci95_low"] is not None else np.nan for e in primary_curve]
    hi = [e["relative_regret_ci95_high"] if e["relative_regret_ci95_high"] is not None else np.nan for e in primary_curve]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax = axes[0]
    ax.plot(xs, ys, color="black", marker="o", label="target-balanced median relative regret")
    ax.fill_between(xs, lo, hi, color="0.7", alpha=0.4, label="95% target-cluster CI")
    ax.axhline(TAU_PRIMARY, color="tab:red", ls="--", label="tau=0.05")
    ax.axhline(TAU_STRICT, color="tab:orange", ls=":", label="tau=0.02")
    nf_rel = target_balanced(_group(noise, "noise_floor_rel"), "median")
    if nf_rel is not None:
        ax.axhline(nf_rel, color="tab:blue", ls="-.", label="empirical temporal noise (rel.)")
    ax.set_ylabel("target-balanced relative regret")
    ax.legend(fontsize=7)
    ax2 = axes[1]
    yabs = [e["median_absolute_regret"] if e["median_absolute_regret"] is not None else np.nan for e in primary_curve]
    ax2.plot(xs, yabs, color="black", marker="s", label="target-balanced median absolute regret")
    ax2.set_xlabel("verification days k")
    ax2.set_ylabel("absolute regret")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "RQ4_PRIMARY_FIGURE.png", dpi=160)
    plt.close(fig)

    n_inf = sum(1 for r in inst if r.get("informative"))
    n_non = len(inst) - n_inf
    status = "RQ4_SOURCE_SELECTION_RESULTS_MATERIALIZED" if n_inf else "RQ4_SOURCE_SELECTION_RESULTS_STRUCTURALLY_NONINFORMATIVE"
    invalid_rel = int(sum(int(r.get("invalid_rel") or 0) for r in regret if r.get("informative_k")))
    reasons: dict[str, int] = {}
    for r in inst:
        if not r.get("informative"):
            reasons[str(r.get("reason") or "unknown")] = reasons.get(str(r.get("reason") or "unknown"), 0) + 1

    t1_claim = bool(t1_lo is not None and t1_hi is not None and t1_lo > 0)
    analysis = {
        "status": status,
        "protocol_specification_path": "docs/method_amendments/RQ4_SOURCE_SELECTION_FROZEN_SPEC.md",
        "protocol_specification_sha256": spec_sha,
        "execution_commit": execution_commit,
        "target_cohort": cohorts["target_ids"],
        "source_cohort": cohorts["source_ids"],
        "target_count": cohorts["target_count"],
        "source_count": cohorts["source_count"],
        "decision_instance_total": len(inst),
        "decision_instance_informative": n_inf,
        "decision_instance_noninformative": n_non,
        "noninformative_reason_counts": reasons,
        "candidate_count_distribution": cand_dist,
        "invalid_relative_regret_count": invalid_rel,
        "primary_model": "SplineRidge",
        "sensitivity_model": "HistGradientBoostingRegressor",
        "k_grid": list(K_GRID),
        "primary_curve": primary_curve,
        "random_baseline": {"median_absolute_regret": target_balanced(_group(rnd, "Reg_random"), "median"), "median_relative_regret": target_balanced(rel_random_map, "median")},
        "metadata_baseline": {"median_absolute_regret": target_balanced(_group(rnd, "Reg_metadata"), "median"), "median_relative_regret": target_balanced(meta_rel_map, "median")},
        "oracle_baseline": {"regret": 0},
        "temporal_noise_reference": {"median_abs": target_balanced(nf_map, "median"), "median_rel": nf_rel, "mean_abs": nf_mean},
        "T1": {
            "Delta_meta_mean": t1_mean,
            "Delta_meta_median": t1_med,
            "ci95_low": t1_lo,
            "ci95_high": t1_hi,
            "claim_metadata_beats_random": t1_claim,
            "rule": "claim only if CI fully above zero",
        },
        "T2": {
            "Delta_verify_mean": t2_mean,
            "Delta_verify_median": t2_med,
            "ci95_low": t2_lo,
            "ci95_high": t2_hi,
            "k": 45,
        },
        "T3": {"tau": TAU_PRIMARY, "k_tau": k_tau, "curve": {str(k): {"median": rel_hi[k][0], "upper95": rel_hi[k][1]} for k in K_GRID}},
        "T4": {"k_noise": k_noise, "curve": {str(k): {"median": excess_hi[k][0], "upper95": excess_hi[k][1]} for k in K_GRID}},
        "T3_T4_joint_case": joint,
        "tau_primary": TAU_PRIMARY,
        "tau_sensitivity": TAU_STRICT,
        "k_tau": k_tau,
        "k_tau_strict": k_tau_strict,
        "k_noise": k_noise,
        "secondary_summary": {row["name"]: row["value"] for row in secondary},
        "HGB_sensitivity_summary": hgb_sum,
        "no_design_change_after_outcomes": True,
    }
    # scientific content hash without timestamps
    content = json.dumps({"primary_curve": primary_curve, "T1": analysis["T1"], "T2": analysis["T2"], "k_tau": k_tau, "k_noise": k_noise}, sort_keys=True, default=str)
    analysis["_scientific_content_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    dump_json(out / "RQ4_SUMMARY.json", {k: analysis[k] for k in analysis if not k.startswith("_")})
    return analysis
