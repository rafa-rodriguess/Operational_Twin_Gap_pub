"""S12 P6 / R10: matched TRAIN elapsed-duration sensitivity. Isolated refits; primary CS4 IQR."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from config.protocol import FAM_SPLINE, PRIMARY_GROUP_IDS
from src.lib.s12_p1_null import (
    _days,
    build_engine,
    fit_frozen,
    load_directional_universe,
    load_hparams,
    load_scope,
    load_tables,
    plant_balanced,
    plant_balanced_iterate,
    score_support,
)
from src.io import read_csv
from src.lib.s12_p2_retention import percentile_ci, spearman_rho
from src.lib.s12_p3_adaptive_scaling import plant_balanced_asym
from src.lib.split_support import parse_ts

STATUS = "S12_P6_R10_MATCHED_DURATION_MATERIALIZED"
RQ3_STATUS = "R10_RQ3_NOT_RUN_MATCHED_DURATION_SCOPE"
TOL = 1e-12
EQUAL_TOL_DAYS = 1.0 / 86400.0  # 1 second
BOOT_SEED = 20260921
N_BOOT = 5000
DAY = timedelta(days=1)


def _flag(val: Any) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


GROUP_LABELS = {
    "TW_04b9f5d95694": "G1",
    "TW_4467a039e00b": "G2",
    "TW_88a108921af7": "G4",
}


def _finite(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def dist_pack(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None, "mean": None}
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "min": float(np.min(a)),
        "p25": float(np.quantile(a, 0.25, method="linear")),
        "median": float(np.median(a)),
        "p75": float(np.quantile(a, 0.75, method="linear")),
        "p95": float(np.quantile(a, 0.95, method="linear")),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def reporting_blocks(directional: list[dict[str, Any]], hist_rows: list[dict[str, Any]], asym_rows: list[dict[str, Any]]) -> dict[str, Any]:
    r10 = [r for r in directional if _flag(r.get("computable"))]
    retention = [v for r in r10 if (v := _finite(r.get("support_retention"))) is not None]
    imbalance = [v for r in hist_rows if (v := _finite(r.get("duration_imbalance"))) is not None]
    by_group: dict[str, Any] = {}
    for gid, label in GROUP_LABELS.items():
        pairs = [r for r in asym_rows if str(r.get("group_id")) == gid and _flag(r.get("asymmetry_computable"))]
        if not pairs:
            by_group[label] = {"group_id": gid, "estimable": False, "n_pairs": 0, "n_plants": 0, "plant_balanced": None, "median_abs": None}
            continue
        pb, npl = plant_balanced_asym(pairs)
        by_group[label] = {
            "group_id": gid,
            "estimable": True,
            "n_pairs": len(pairs),
            "n_plants": npl,
            "plant_balanced": pb,
            "median_abs": float(np.median([float(r["asymmetry_abs"]) for r in pairs])),
        }
    overlap_pairs = [r for r in asym_rows if _flag(r.get("asymmetry_computable")) and _flag(r.get("primary_both_computable"))]
    xs_a, ys_a, xs_c, ys_c = [], [], [], []
    for r in overlap_pairs:
        imb = _finite(r.get("duration_imbalance"))
        ap = _finite(r.get("abs_A_primary"))
        ch = _finite(r.get("abs_A_change"))
        if imb is not None and ap is not None:
            xs_a.append(imb)
            ys_a.append(ap)
        if imb is not None and ch is not None:
            xs_c.append(imb)
            ys_c.append(ch)
    rho_a, rho_a_reason = spearman_rho(xs_a, ys_a) if len(xs_a) >= 3 else (None, "n_lt_3")
    rho_c, rho_c_reason = spearman_rho(xs_c, ys_c) if len(xs_c) >= 3 else (None, "n_lt_3")
    return {
        "support_retention_r10": dist_pack(retention),
        "duration_imbalance_planned": dist_pack(imbalance),
        "rq2_r10_by_group": by_group,
        "spearman_imbalance_absA": {"rho": rho_a, "reason": rho_a_reason, "n": len(xs_a)},
        "spearman_imbalance_absA_change": {"rho": rho_c, "reason": rho_c_reason, "n": len(xs_c)},
    }


def run_reporting_from_disk(root: Path, sot: dict[str, Any]) -> dict[str, Any]:
    out = root / "artifacts/s12/p6"
    directional = read_csv(out / "S12_P6_R10_DIRECTIONAL.csv")
    hist_rows = read_csv(out / "S12_P6_DIRECTION_HISTORY.csv")
    asym_rows = read_csv(out / "S12_P6_R10_ASYMMETRY.csv")
    plant_rows = read_csv(out / "S12_P6_PLANT_TRAIN_HISTORY.csv")
    summary = json.loads((out / "S12_P6_SUMMARY.json").read_text(encoding="utf-8"))
    extra = reporting_blocks(directional, hist_rows, asym_rows)
    summary.update(extra)
    summary["interpretation"] = interpret(summary)
    summary["reporting_correction"] = True
    summary["r10_is_sensitivity_only"] = True
    return {
        "plants": [r["plant_id"] for r in plant_rows],
        "plant_rows": plant_rows,
        "hist_rows": hist_rows,
        "directional": directional,
        "asym_rows": asym_rows,
        "eval_rows": [],
        "summary": _jsonable(summary),
        "reporting_only": True,
    }


def _aware(ts):
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def duration_days(start, end) -> float:
    return (end - start).total_seconds() / 86400.0


def plant_history(pid: str, train_dts: list[str]) -> dict[str, Any]:
    tss = [_aware(parse_ts(dt)) for dt in train_dts]
    tss = [t for t in tss if t is not None]
    start, end = min(tss), max(tss)
    days = {t.date().isoformat() for t in tss}
    return {
        "plant_id": pid,
        "train_start": start.isoformat(),
        "train_end": end.isoformat(),
        "duration_days": duration_days(start, end),
        "n_train": len(train_dts),
        "n_calendar_days": len(days),
        "_start": start,
        "_end": end,
    }


def match_window(train_dts: list[str], train_end, d_pair: float) -> tuple[list[str], Any, Any]:
    lo = train_end - (d_pair * DAY)
    kept = []
    tss = []
    for dt in train_dts:
        ts = _aware(parse_ts(dt))
        if ts is None:
            continue
        if lo <= ts <= train_end:
            kept.append(dt)
            tss.append(ts)
    if not kept:
        return [], None, None
    return kept, min(tss), max(tss)


def predict_many(est, plant_id: str, dts: list[str], feat_by, y_map):
    xs, ys, keep = [], [], []
    for dt in dts:
        vec = feat_by.get((plant_id, dt))
        yv = y_map.get((plant_id, dt))
        if vec is None or yv is None:
            continue
        xs.append(vec)
        ys.append(yv)
        keep.append(dt)
    if not xs:
        return [], np.array([]), np.array([])
    pred = np.asarray(est.predict(np.vstack(xs)), dtype=float)
    return keep, np.asarray(ys, dtype=float), pred


def get_model(cache: dict, pid: str, dts: list[str], feat_by, y_map, params, root: Path):
    key = (pid, dts[0] if dts else "", dts[-1] if dts else "", len(dts))
    if key in cache:
        return cache[key]
    est, used, err = fit_frozen(pid, dts, feat_by, y_map, params)
    rec = {"est": est, "error": err, "n": len(used), "key": key}
    if est is not None:
        dest = root / "artifacts/s12/p6/models" / pid / f"{len(dts)}_{dts[0].replace(':', '')}_{dts[-1].replace(':', '')}.joblib"
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(est, dest)
        rec["path"] = str(dest)
    cache[key] = rec
    return rec


def cluster_bootstrap(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[float | None, float | None, int]:
    targets = sorted({str(r["target_plant_id"]) for r in rows})
    by_t: dict[str, list] = defaultdict(list)
    for r in rows:
        by_t[str(r["target_plant_id"])].append(r)
    rng = np.random.default_rng(BOOT_SEED)
    valid = []
    for i in range(N_BOOT):
        sampled = rng.choice(targets, size=len(targets), replace=True)
        xs, ys = [], []
        for t in sampled:
            for r in by_t[str(t)]:
                if r.get(x_key) is None or r.get(y_key) is None:
                    continue
                xs.append(float(r[x_key]))
                ys.append(float(r[y_key]))
        rho, _ = spearman_rho(xs, ys)
        if rho is not None:
            valid.append(rho)
    lo, hi = percentile_ci(valid)
    return lo, hi, len(valid)


def rq1_from(rows: list[dict[str, Any]], field: str = "otg_abs") -> dict[str, Any]:
    ok = [r for r in rows if r.get("computable") and r.get(field) is not None]
    tagged = [{**r, "otg_abs": r[field], "computable": True} for r in ok]
    pb, nt = plant_balanced(tagged)
    pb2, _ = plant_balanced_iterate(tagged)
    abs_v = [float(r[field]) for r in ok]
    rel = [float(r["otg_rel"]) for r in ok if r.get("otg_rel_defined")]
    by_g = defaultdict(list)
    for r in tagged:
        by_g[r["group_id"]].append(r)
    return {
        "n": len(ok),
        "n_targets": nt,
        "n_sources": len({r["source_plant_id"] for r in ok}),
        "directional_mean": float(np.mean(abs_v)) if abs_v else None,
        "directional_median": float(np.median(abs_v)) if abs_v else None,
        "plant_balanced": pb,
        "plant_balanced_iterate": pb2,
        "rel_median": float(np.median(rel)) if rel else None,
        "rel_p75": float(np.quantile(rel, 0.75, method="linear")) if rel else None,
        "group_plant_balanced": {g: plant_balanced(v)[0] for g, v in sorted(by_g.items())},
        "recon": abs((pb or 0) - (pb2 or 0)) if pb is not None else 0.0,
    }


def interpret(s: dict[str, Any]) -> dict[str, Any]:
    ov = s["rq1_overlap"]

    def rho_txt(val: Any) -> Any:
        if isinstance(val, dict):
            return val.get("rho")
        return val

    sa = rho_txt(s.get("spearman_imbalance_absA"))
    sc = rho_txt(s.get("spearman_imbalance_absA_change"))
    text = (
        f"R10 computable {s['n_r10_computable']}/94 (primary 85); lost {s['n_lost']}; newly computable {s['n_new']}. "
        f"RQ1 PB primary {s['rq1_primary']} vs R10 {s['rq1_r10']['plant_balanced']} (Δ {s['rq1_r10']['difference_vs_primary']}). "
        f"Overlap n={ov['n']} PB primary {ov['primary_pb']} vs R10 {ov['r10_pb']} (Δ {ov['difference']}). "
        f"Median Delta_OTG_duration {s['delta_otg']['median']} (positive = R10 reduced penalty); "
        f"{s['delta_otg']['n_positive']}/{s['delta_otg']['n']} positive, {s['delta_otg']['n_negative']}/{s['delta_otg']['n']} negative. "
        f"Spearman |duration imbalance| vs Delta_OTG rho={s['spearman_imbalance_delta']['rho']} "
        f"CI [{s['spearman_imbalance_delta']['ci95_low']}, {s['spearman_imbalance_delta']['ci95_high']}]. "
        f"RQ2 PB |A| primary {s['rq2_primary']} vs R10 {s['rq2_r10']['plant_balanced']} (overlap pairs {s['rq2_overlap']['n']}). "
        f"Spearman |duration imbalance| vs |A_primary| rho={sa} ; vs |A| change rho={sc}. "
        "R10 remains a sensitivity arm, not a new primary protocol."
    )
    return {"text": text, "threshold_rules_used": False, "r10_is_sensitivity_only": True}


def run_science(root: Path, sot: dict[str, Any]) -> dict[str, Any]:
    plants, members = load_scope(root)
    hparams = load_hparams(root, plants)
    universe = [r for r in load_directional_universe(root) if r.get("model_family") == FAM_SPLINE and r.get("support_rule") == "CS4"]
    feat_by, y_map, splits = load_tables(root, plants)
    hist = {pid: {**plant_history(pid, splits[pid]["train"]), "group_id": members[pid]} for pid in plants}
    primary_ok = {(r["source_plant_id"], r["target_plant_id"]) for r in universe if _flag(r.get("computable"))}
    primary_otg = {(r["source_plant_id"], r["target_plant_id"]): float(r["otg_abs"]) for r in universe if _flag(r.get("computable"))}
    tables = (sot.get("results") or {}).get("tables") or {}
    prim_rq1, prim_rq2 = tables.get("rq1"), tables.get("rq2")
    recon_max = abs(len(plants) - 14) + abs(len(universe) - 94)

    cache: dict = {}
    plant_rows = [{k: v for k, v in hist[pid].items() if not str(k).startswith("_")} for pid in plants]
    hist_rows, directional, eval_rows = [], [], []
    engines_ok = {}

    for row in universe:
        src, tgt, gid = row["source_plant_id"], row["target_plant_id"], row["group_id"]
        hs, ht = hist[src], hist[tgt]
        d_pair = min(hs["duration_days"], ht["duration_days"])
        imb = abs(hs["duration_days"] - ht["duration_days"])
        if abs(hs["duration_days"] - ht["duration_days"]) <= EQUAL_TOL_DAYS:
            shorter = "equal"
        elif hs["duration_days"] < ht["duration_days"]:
            shorter = "source"
        else:
            shorter = "target"
        src_dts, src_lo, src_hi = match_window(splits[src]["train"], hs["_end"], d_pair)
        tgt_dts, tgt_lo, tgt_hi = match_window(splits[tgt]["train"], ht["_end"], d_pair)
        src_span = duration_days(src_lo, src_hi) if src_lo else None
        tgt_span = duration_days(tgt_lo, tgt_hi) if tgt_lo else None
        recon_max = max(recon_max, abs(d_pair - min(hs["duration_days"], ht["duration_days"])))
        rec = {
            "group_id": gid,
            "source_plant_id": src,
            "target_plant_id": tgt,
            "model_family": FAM_SPLINE,
            "support_rule": "CS4_R10",
            "computable": False,
            "primary_computable": (src, tgt) in primary_ok,
            "d_pair": d_pair,
            "source_full_duration": hs["duration_days"],
            "target_full_duration": ht["duration_days"],
            "duration_imbalance": imb,
            "shorter_side": shorter,
            "n_source_matched": len(src_dts),
            "n_target_matched": len(tgt_dts),
            "source_matched_start": src_lo.isoformat() if src_lo else None,
            "source_matched_end": src_hi.isoformat() if src_hi else None,
            "target_matched_start": tgt_lo.isoformat() if tgt_lo else None,
            "target_matched_end": tgt_hi.isoformat() if tgt_hi else None,
            "source_matched_elapsed": src_span,
            "target_matched_elapsed": tgt_span,
            "matched_row_ratio": (len(src_dts) / len(tgt_dts)) if tgt_dts else None,
        }
        hist_rows.append(dict(rec))
        if not src_dts or not tgt_dts:
            rec["reason"] = "empty_matched_train_window"
            directional.append(rec)
            continue
        ms = get_model(cache, src, src_dts, feat_by, y_map, hparams[src], root)
        mt = get_model(cache, tgt, tgt_dts, feat_by, y_map, hparams[tgt], root)
        if ms["est"] is None or mt["est"] is None:
            rec["reason"] = ms["error"] or mt["error"] or "model_unfit"
            directional.append(rec)
            continue
        eng_key = (src, rec["source_matched_start"], rec["source_matched_end"], len(src_dts))
        if eng_key not in engines_ok:
            engines_ok[eng_key] = build_engine(src, src_dts, feat_by)
        eng = engines_ok[eng_key]
        if not eng.get("computable"):
            rec["reason"] = eng.get("reason")
            directional.append(rec)
            continue
        test = splits[tgt]["test"]
        scored = score_support(eng, tgt, test, feat_by)
        supported = [dt for dt, _d, _t, ok in scored if ok]
        rec["n_eligible_target_test"] = len(test)
        rec["n_supported"] = len(supported)
        rec["support_retention"] = (len(supported) / len(test)) if test else None
        rec["n_supported_days"] = _days(supported) if supported else 0
        rec["threshold"] = eng.get("tau")
        if not supported:
            rec["reason"] = "empty_cs4_support"
            directional.append(rec)
            continue
        keep, y, p_src = predict_many(ms["est"], tgt, supported, feat_by, y_map)
        keep2, y2, p_loc = predict_many(mt["est"], tgt, supported, feat_by, y_map)
        if len(keep) != len(keep2) or keep != keep2:
            rec["reason"] = "source_local_row_mismatch"
            directional.append(rec)
            continue
        if not keep:
            rec["reason"] = "missing_predictions_on_support"
            directional.append(rec)
            continue
        tmae = float(np.mean(np.abs(p_src - y)))
        lmae = float(np.mean(np.abs(p_loc - y)))
        otg = tmae - lmae
        rec.update(
            {
                "computable": True,
                "reason": None,
                "transfer_mae": tmae,
                "local_same_rows_mae": lmae,
                "otg_abs": otg,
                "otg_rel": (otg / lmae) if lmae != 0 else None,
                "otg_rel_defined": lmae != 0,
            }
        )
        directional.append(rec)
        dist_map = {dt: (d, tau) for dt, d, tau, ok in scored if ok}
        for i, dt in enumerate(keep):
            d, tau = dist_map[dt]
            eval_rows.append(
                {
                    "group_id": gid,
                    "source_plant_id": src,
                    "target_plant_id": tgt,
                    "datetime_raw": dt,
                    "split": "test",
                    "supported": True,
                    "y_true": float(y[i]),
                    "p_source": float(p_src[i]),
                    "p_local": float(p_loc[i]),
                    "ae_source": abs(float(p_src[i] - y[i])),
                    "ae_local": abs(float(p_loc[i] - y[i])),
                    "kth_distance": d,
                    "threshold": tau,
                    "d_pair": d_pair,
                }
            )

    r10_ok = [r for r in directional if r.get("computable")]
    r10_ids = {(r["source_plant_id"], r["target_plant_id"]) for r in r10_ok}
    lost = sorted(primary_ok - r10_ids)
    new = sorted(r10_ids - primary_ok)
    overlap_ids = sorted(primary_ok & r10_ids)
    ps048 = [r for r in directional if r["source_plant_id"] == "PS_048"]

    rq1_r10 = rq1_from(r10_ok)
    rq1_r10["primary"] = prim_rq1
    rq1_r10["difference_vs_primary"] = (rq1_r10["plant_balanced"] - float(prim_rq1)) if rq1_r10["plant_balanced"] is not None and prim_rq1 is not None else None
    recon_max = max(recon_max, rq1_r10["recon"])

    overlap_r10 = [r for r in r10_ok if (r["source_plant_id"], r["target_plant_id"]) in primary_ok]
    overlap_pri = [{"group_id": r["group_id"], "source_plant_id": r["source_plant_id"], "target_plant_id": r["target_plant_id"], "otg_abs": primary_otg[(r["source_plant_id"], r["target_plant_id"])], "computable": True, "otg_rel": r.get("otg_rel"), "otg_rel_defined": False} for r in overlap_r10]
    ov_r10 = rq1_from(overlap_r10)
    ov_pri = rq1_from(overlap_pri)
    deltas = []
    for r in overlap_r10:
        key = (r["source_plant_id"], r["target_plant_id"])
        dlt = primary_otg[key] - float(r["otg_abs"])
        r["delta_otg_duration"] = dlt
        r["abs_duration_imbalance"] = abs(float(r["duration_imbalance"]))
        deltas.append(dlt)
    d_pb, _ = plant_balanced([{**r, "otg_abs": r["delta_otg_duration"], "computable": True} for r in overlap_r10]) if overlap_r10 else (None, 0)
    delta_pack = {
        "n": len(deltas),
        "mean": float(np.mean(deltas)) if deltas else None,
        "median": float(np.median(deltas)) if deltas else None,
        "p25": float(np.quantile(deltas, 0.25, method="linear")) if deltas else None,
        "p75": float(np.quantile(deltas, 0.75, method="linear")) if deltas else None,
        "min": float(min(deltas)) if deltas else None,
        "max": float(max(deltas)) if deltas else None,
        "plant_balanced": d_pb,
        "n_positive": int(sum(v > 0 for v in deltas)),
        "n_negative": int(sum(v < 0 for v in deltas)),
        "frac_positive": (sum(v > 0 for v in deltas) / len(deltas)) if deltas else None,
        "frac_negative": (sum(v < 0 for v in deltas) / len(deltas)) if deltas else None,
    }
    rho, rho_r = spearman_rho([r["abs_duration_imbalance"] for r in overlap_r10], [r["delta_otg_duration"] for r in overlap_r10]) if len(overlap_r10) >= 3 else (None, "n_lt_3")
    lo, hi, nval = cluster_bootstrap(overlap_r10, "abs_duration_imbalance", "delta_otg_duration") if overlap_r10 else (None, None, 0)

    lookup = {(r["source_plant_id"], r["target_plant_id"]): r for r in directional}
    plants_by_g: dict[str, set[str]] = defaultdict(set)
    for r in universe:
        plants_by_g[r["group_id"]].add(r["source_plant_id"])
        plants_by_g[r["group_id"]].add(r["target_plant_id"])
    asym_rows = []
    for gid, ps in plants_by_g.items():
        order = sorted(ps)
        for i, a in enumerate(order):
            for b in order[i + 1 :]:
                ab, ba = lookup.get((a, b)), lookup.get((b, a))
                both = bool(ab and ab.get("computable") and ba and ba.get("computable"))
                oa = ab["otg_abs"] if ab and ab.get("computable") else None
                ob = ba["otg_abs"] if ba and ba.get("computable") else None
                signed = (oa - ob) if both else None
                prim_both = (a, b) in primary_ok and (b, a) in primary_ok
                asym_rows.append(
                    {
                        "group_id": gid,
                        "plant_a": a,
                        "plant_b": b,
                        "a_to_b_computable": bool(ab and ab.get("computable")),
                        "b_to_a_computable": bool(ba and ba.get("computable")),
                        "otg_a_to_b": oa,
                        "otg_b_to_a": ob,
                        "asymmetry_signed": signed,
                        "asymmetry_abs": abs(signed) if signed is not None else None,
                        "asymmetry_computable": both,
                        "primary_both_computable": prim_both,
                        "duration_imbalance": abs(hist[a]["duration_days"] - hist[b]["duration_days"]),
                    }
                )
    r10_pairs = [r for r in asym_rows if r["asymmetry_computable"]]
    rq2_r10, npl = plant_balanced_asym(r10_pairs) if r10_pairs else (None, 0)
    overlap_pairs = [r for r in r10_pairs if r["primary_both_computable"]]
    # primary overlap pair |A|
    prim_pair_lookup = {}
    import csv as _csv
    with (root / "artifacts/s07/S07_ASYMMETRY.csv").open(encoding="utf-8", newline="") as handle:
        for prow in _csv.DictReader(handle):
            if prow.get("group_id") in PRIMARY_GROUP_IDS and _flag(prow.get("asymmetry_computable")):
                prim_pair_lookup[(prow["plant_a"], prow["plant_b"])] = float(prow["asymmetry_abs"])
                prim_pair_lookup[(prow["plant_b"], prow["plant_a"])] = float(prow["asymmetry_abs"])
    ov_pair_pri = []
    for r in overlap_pairs:
        pa, pb = r["plant_a"], r["plant_b"]
        ap = prim_pair_lookup.get((pa, pb))
        if ap is None:
            continue
        ov_pair_pri.append({**r, "asymmetry_abs": ap})
        r["abs_A_primary"] = ap
        r["abs_A_change"] = ap - float(r["asymmetry_abs"])
    rq2_ov_r10, _ = plant_balanced_asym(overlap_pairs) if overlap_pairs else (None, 0)
    rq2_ov_pri, _ = plant_balanced_asym(ov_pair_pri) if ov_pair_pri else (None, 0)
    rho_a, _ = spearman_rho([r["duration_imbalance"] for r in overlap_pairs], [r["abs_A_primary"] for r in overlap_pairs if r.get("abs_A_primary") is not None]) if len(overlap_pairs) >= 3 else (None, None)
    rho_c, _ = spearman_rho([r["duration_imbalance"] for r in overlap_pairs], [r["abs_A_change"] for r in overlap_pairs if r.get("abs_A_change") is not None]) if len(overlap_pairs) >= 3 else (None, None)

    # row-level OTG recon
    by_dir = defaultdict(list)
    for e in eval_rows:
        by_dir[(e["source_plant_id"], e["target_plant_id"])].append(e)
    for r in r10_ok:
        chunk = by_dir[(r["source_plant_id"], r["target_plant_id"])]
        tmae = float(np.mean([c["ae_source"] for c in chunk]))
        lmae = float(np.mean([c["ae_local"] for c in chunk]))
        recon_max = max(recon_max, abs(tmae - r["transfer_mae"]), abs((tmae - lmae) - r["otg_abs"]))

    durs = [hist[p]["duration_days"] for p in plants]
    dpairs = [r["d_pair"] for r in directional]
    n_src_short = sum(1 for r in directional if r["shorter_side"] == "source")
    n_tgt_short = sum(1 for r in directional if r["shorter_side"] == "target")
    n_eq = sum(1 for r in directional if r["shorter_side"] == "equal")

    summary = {
        "status": STATUS,
        "n_plants": 14,
        "n_planned": 94,
        "n_primary_computable": 85,
        "n_r10_computable": len(r10_ok),
        "n_r10_noncomputable": 94 - len(r10_ok),
        "n_lost": len(lost),
        "n_new": len(new),
        "lost_identities": [{"source_plant_id": a, "target_plant_id": b} for a, b in lost],
        "new_identities": [{"source_plant_id": a, "target_plant_id": b} for a, b in new],
        "ps048": [{"target_plant_id": r["target_plant_id"], "computable": r.get("computable"), "reason": r.get("reason")} for r in ps048],
        "duration": {
            "plant_min": float(min(durs)),
            "plant_median": float(np.median(durs)),
            "plant_max": float(max(durs)),
            "d_pair_min": float(min(dpairs)),
            "d_pair_median": float(np.median(dpairs)),
            "d_pair_max": float(max(dpairs)),
            "n_source_shorter": n_src_short,
            "n_target_shorter": n_tgt_short,
            "n_equal": n_eq,
            "equal_tolerance_days": EQUAL_TOL_DAYS,
        },
        "rq1_primary": prim_rq1,
        "rq1_r10": rq1_r10,
        "rq1_overlap": {"n": len(overlap_r10), "primary_pb": ov_pri["plant_balanced"], "r10_pb": ov_r10["plant_balanced"], "difference": (ov_r10["plant_balanced"] - ov_pri["plant_balanced"]) if ov_r10["plant_balanced"] is not None else None},
        "delta_otg": delta_pack,
        "spearman_imbalance_delta": {"rho": rho, "reason": rho_r, "ci95_low": lo, "ci95_high": hi, "n_boot_valid": nval, "seed": BOOT_SEED, "n_replicates": N_BOOT},
        "rq2_primary": prim_rq2,
        "rq2_r10": {
            "n_planned_pairs": 47,
            "n_primary_pairs": 38,
            "n_r10_pairs": len(r10_pairs),
            "plant_balanced": rq2_r10,
            "n_plants": npl,
            "median_abs": float(np.median([r["asymmetry_abs"] for r in r10_pairs])) if r10_pairs else None,
            "difference_vs_primary": (rq2_r10 - float(prim_rq2)) if rq2_r10 is not None and prim_rq2 is not None else None,
        },
        "rq2_overlap": {"n": len(overlap_pairs), "primary_pb": rq2_ov_pri, "r10_pb": rq2_ov_r10, "difference": (rq2_ov_r10 - rq2_ov_pri) if rq2_ov_r10 is not None and rq2_ov_pri is not None else None},
        "rq3_status": RQ3_STATUS,
        "n_cached_models": len(cache),
        "noncomputable": [{"source_plant_id": r["source_plant_id"], "target_plant_id": r["target_plant_id"], "reason": r.get("reason")} for r in directional if not r.get("computable")],
        "reconciliation_max_abs_discrepancy": recon_max,
        "reconciliation_ok": recon_max <= 1e-9,
        "r10_is_sensitivity_only": True,
        "no_r9": True,
        "no_p4": True,
    }
    summary.update(reporting_blocks(directional, hist_rows, asym_rows))
    summary["interpretation"] = interpret(summary)
    summary = _jsonable(summary)
    return {
        "plants": plants,
        "plant_rows": plant_rows,
        "hist_rows": hist_rows,
        "directional": directional,
        "asym_rows": asym_rows,
        "eval_rows": eval_rows,
        "overlap_r10": overlap_r10,
        "summary": summary,
        "boot_rows": [{"replicate_id": i} for i in range(0)],
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        pq.write_table(pa.table({"_empty": pa.array([], type=pa.int8())}), path)
        return 0
    cols = list(rows[0].keys())
    pq.write_table(pa.table({c: [r.get(c) for r in rows] for c in cols}), path)
    return len(rows)


def render_report(s: dict[str, Any]) -> str:
    d = s["duration"]
    sr = s.get("support_retention_r10") or {}
    di = s.get("duration_imbalance_planned") or {}
    g = s.get("rq2_r10_by_group") or {}
    sa = s.get("spearman_imbalance_absA")
    sc = s.get("spearman_imbalance_absA_change")

    def rho_txt(val: Any) -> Any:
        if isinstance(val, dict):
            return val.get("rho")
        return val

    def grp(label: str) -> str:
        rec = g.get(label) or {}
        if not rec.get("estimable"):
            return f"{label} not estimable (n_pairs={rec.get('n_pairs', 0)})"
        return (
            f"{label} n_pairs={rec['n_pairs']} n_plants={rec['n_plants']} "
            f"PB |A|={rec['plant_balanced']} median |A|={rec['median_abs']}"
        )

    return "\n".join(
        [
            "# S12 P6 / R10 — matched TRAIN elapsed-duration sensitivity",
            "",
            "Secondary arm. Canonical TEST evaluation. Strict primary CS4 IQR (no R9). Trailing matched windows; D_pair = min(source, target) TRAIN elapsed days; both endpoints included. Equal-duration tolerance = 1 second.",
            "",
            f"Status: `{s['status']}`. R10 computable {s['n_r10_computable']}/94 (primary 85); lost {s['n_lost']}; new {s['n_new']}.",
            f"Plant TRAIN duration days min/median/max {d['plant_min']}/{d['plant_median']}/{d['plant_max']}. D_pair min/median/max {d['d_pair_min']}/{d['d_pair_median']}/{d['d_pair_max']}. Shorter: source {d['n_source_shorter']}, target {d['n_target_shorter']}, equal {d['n_equal']}.",
            f"RQ1 PB primary {s['rq1_primary']} vs R10 {s['rq1_r10']['plant_balanced']} (Δ {s['rq1_r10']['difference_vs_primary']}); overlap n={s['rq1_overlap']['n']} Δ {s['rq1_overlap']['difference']}.",
            f"Delta_OTG_duration median {s['delta_otg']['median']}; {s['delta_otg']['n_positive']}/{s['delta_otg']['n']} positive. Spearman |imbalance| vs Delta rho={s['spearman_imbalance_delta']['rho']} CI [{s['spearman_imbalance_delta']['ci95_low']}, {s['spearman_imbalance_delta']['ci95_high']}].",
            f"RQ2 PB |A| primary {s['rq2_primary']} vs R10 {s['rq2_r10']['plant_balanced']} (n_pairs {s['rq2_r10']['n_r10_pairs']}); overlap pairs {s['rq2_overlap']['n']}.",
            f"R10 support-retention (n={sr.get('n')} computable directions) min/P25/median/P75/P95/max/mean {sr.get('min')}/{sr.get('p25')}/{sr.get('median')}/{sr.get('p75')}/{sr.get('p95')}/{sr.get('max')}/{sr.get('mean')}.",
            f"Full-history duration imbalance days (n={di.get('n')} planned directions) min/P25/median/P75/P95/max/mean {di.get('min')}/{di.get('p25')}/{di.get('median')}/{di.get('p75')}/{di.get('p95')}/{di.get('max')}/{di.get('mean')}.",
            f"Group-level R10 RQ2: {grp('G1')}; {grp('G2')}; {grp('G4')}.",
            f"Spearman |full-history duration difference| vs primary |A| rho={rho_txt(sa)}; vs |A_primary|-|A_R10| rho={rho_txt(sc)}.",
            f"RQ3: `{s['rq3_status']}`. Cached matched-window models: {s['n_cached_models']}.",
            f"Interpretation: {s['interpretation']['text']}",
            f"Reconciliation max |Δ|={s['reconciliation_max_abs_discrepancy']} ok={s['reconciliation_ok']}.",
            "",
        ]
    )
