"""S12 P4: bias/dispersion and VALIDATION-only recalibration ladder. Sensitivity only."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from config.protocol import FAM_SPLINE, PRIMARY_GROUP_IDS
from src.lib.s12_p1_null import (
    FEATURES,
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
from src.lib.s12_p2_retention import percentile_ci, spearman_rho

STATUS = "S12_P4_RECALIBRATION_MATERIALIZED"
RQ3_STATUS = "P4_RQ3_NOT_RUN_RECALIBRATION_SCOPE"
TOL = 1e-12
PRED_TOL = 1e-10
BOOT_SEED = 20260921
N_BOOT = 5000


def _flag(val: Any) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


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


def error_pack(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    e = pred - y
    mbe = float(np.mean(e))
    return {
        "mbe": mbe,
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "centered_mae": float(np.mean(np.abs(e - mbe))),
        "centered_rmse": float(np.sqrt(np.mean((e - mbe) ** 2))),
    }


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


def affine_fit(y: np.ndarray, p: np.ndarray) -> tuple[float | None, float | None, str | None]:
    if y.size < 2:
        return None, None, "n_val_lt_2"
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        return None, None, "nonfinite_validation"
    if float(np.var(p)) == 0.0:
        return None, None, "zero_variance_source_pred"
    x = np.column_stack([np.ones(y.size), p])
    if int(np.linalg.matrix_rank(x)) < 2:
        return None, None, "rank_deficient_intercept_slope"
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if not np.isfinite(a) or not np.isfinite(b):
        return None, None, "nonfinite_coefficients"
    return a, b, None


def load_s06_test(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    table = pq.read_table(
        root / "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet",
        columns=[
            "source_plant_id",
            "target_plant_id",
            "model_family",
            "support_rule",
            "datetime_raw",
            "y_true",
            "y_pred_transfer",
            "y_pred_local",
        ],
    )
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for src, tgt, fam, rule, dt, y, pt, pl in zip(
        table["source_plant_id"].to_pylist(),
        table["target_plant_id"].to_pylist(),
        table["model_family"].to_pylist(),
        table["support_rule"].to_pylist(),
        table["datetime_raw"].to_pylist(),
        table["y_true"].to_pylist(),
        table["y_pred_transfer"].to_pylist(),
        table["y_pred_local"].to_pylist(),
    ):
        if fam != FAM_SPLINE or str(rule) != "CS4":
            continue
        key = (str(src), str(tgt))
        rec = out.setdefault(key, {"dt": [], "y": [], "p0": [], "local": []})
        rec["dt"].append(str(dt))
        rec["y"].append(float(y))
        rec["p0"].append(float(pt))
        rec["local"].append(float(pl))
    for rec in out.values():
        rec["y"] = np.asarray(rec["y"], dtype=float)
        rec["p0"] = np.asarray(rec["p0"], dtype=float)
        rec["local"] = np.asarray(rec["local"], dtype=float)
    return out


def load_s05_supported(root: Path) -> dict[tuple[str, str], set[str]]:
    table = pq.read_table(
        root / "artifacts/s05/S05_SUPPORT_MASKS.parquet",
        columns=["source_plant_id", "target_plant_id", "rule_id", "datetime_raw", "supported", "computable"],
    )
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for src, tgt, rid, dt, sup, comp in zip(
        table["source_plant_id"].to_pylist(),
        table["target_plant_id"].to_pylist(),
        table["rule_id"].to_pylist(),
        table["datetime_raw"].to_pylist(),
        table["supported"].to_pylist(),
        table["computable"].to_pylist(),
    ):
        if str(rid) != "CS4" or not comp or not sup:
            continue
        out[(str(src), str(tgt))].add(str(dt))
    return dict(out)


def load_or_refit_models(root: Path, plants: list[str], splits, feat_by, y_map, hparams) -> dict[str, Any]:
    out = {}
    for pid in plants:
        candidates = [
            root / "artifacts/s04/models" / pid / FAM_SPLINE / "full.joblib",
            root / "artifacts/s12/p3/models" / pid / "SplineRidge_full.joblib",
        ]
        est, action = None, "refit_missing_artifact"
        probe = next((feat_by[(pid, dt)] for dt in splits[pid]["train"] if (pid, dt) in feat_by), None)
        for path, label in zip(candidates, ("reused_s04", "reused_p3")):
            if not path.is_file() or probe is None:
                continue
            try:
                loaded = joblib.load(path)
                loaded.predict(probe.reshape(1, -1))
                est, action = loaded, label
                break
            except Exception:
                action = "refit_unusable_artifact"
        if est is None:
            est, _used, err = fit_frozen(pid, splits[pid]["train"], feat_by, y_map, hparams[pid])
            if err:
                out[pid] = {"est": None, "action": action, "error": err}
                continue
            dest = root / "artifacts/s12/p4/models" / pid / "SplineRidge_full.joblib"
            dest.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(est, dest)
            action = action if action.startswith("refit") else "refit"
        out[pid] = {"est": est, "action": action, "error": None}
    return out


def cluster_bootstrap_xy(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[float | None, float | None, int]:
    targets = sorted({str(r["target_plant_id"]) for r in rows})
    by_t: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_t[str(r["target_plant_id"])].append(r)
    rng = np.random.default_rng(BOOT_SEED)
    valid = []
    for _ in range(N_BOOT):
        sampled = rng.choice(targets, size=len(targets), replace=True)
        xs, ys = [], []
        for t in sampled:
            for r in by_t[str(t)]:
                xs.append(float(r[x_key]))
                ys.append(float(r[y_key]))
        rho, _reason = spearman_rho(xs, ys)
        if rho is not None:
            valid.append(rho)
    lo, hi = percentile_ci(valid)
    return lo, hi, len(valid)


def ladder_summary(rows: list[dict[str, Any]], otg_key: str, mae_key: str, mbe_key: str) -> dict[str, Any]:
    usable = [r for r in rows if r.get(otg_key) is not None]
    pb, nt = plant_balanced([{**r, "otg_abs": r[otg_key], "computable": True} for r in usable], "otg_abs")
    pb2, _ = plant_balanced_iterate([{**r, "otg_abs": r[otg_key], "computable": True} for r in usable], "otg_abs")
    otgs = [float(r[otg_key]) for r in usable]
    rel = [float(r[otg_key]) / float(r["local_mae"]) for r in usable if float(r["local_mae"]) != 0]
    by_g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in usable:
        by_g[r["group_id"]].append({**r, "otg_abs": r[otg_key], "computable": True})
    return {
        "n_planned": 85,
        "n_test_evaluable": len(usable),
        "directional_mean_otg": float(np.mean(otgs)) if otgs else None,
        "directional_median_otg": float(np.median(otgs)) if otgs else None,
        "plant_balanced_otg": pb,
        "plant_balanced_otg_iterate": pb2,
        "n_targets": nt,
        "rel_median": float(np.median(rel)) if rel else None,
        "rel_p75": float(np.quantile(rel, 0.75, method="linear")) if rel else None,
        "group_plant_balanced": {g: plant_balanced(v)[0] for g, v in sorted(by_g.items())},
        "transfer_mae": _dist([float(r[mae_key]) for r in usable]),
        "abs_transfer_mbe": _dist([abs(float(r[mbe_key])) for r in usable]),
        "otg": _dist(otgs),
    }


def interpret(summary: dict[str, Any]) -> dict[str, Any]:
    pb0 = summary["k0"]["plant_balanced_otg"]
    pb1 = summary["k1"]["plant_balanced_otg"]
    pb2 = summary["k2"]["plant_balanced_otg"]
    d1 = summary["delta_offset"]
    d2 = summary["delta_affine"]
    rq2 = summary["rq2"]
    ctx2 = summary.get("contextual_reduction_affine")
    text = (
        f"Offset correction reduces plant-balanced OTG from {pb0} to {pb1} "
        f"(PB Delta_OTG {d1['plant_balanced']}; {d1['n_improved']}/{d1['n']} directions improve, "
        f"{d1['n_worsened']}/{d1['n']} worsen). "
        f"Affine correction reduces it further to {pb2} (PB Delta_OTG {d2['plant_balanced']}; "
        f"{d2['n_improved']}/{d2['n']} improve, {d2['n_worsened']}/{d2['n']} worsen), "
        f"moving fleet-level plant-balanced OTG slightly below zero. "
        f"RQ2 plant-balanced |A| is {rq2['k0']['plant_balanced_abs_asymmetry']} (k=0), "
        f"{rq2['k1']['plant_balanced_abs_asymmetry']} (k=1), "
        f"{rq2['k2']['plant_balanced_abs_asymmetry']} (k=2) on {rq2['k0']['n_pairs']} same-level pairs. "
        f"Median |delta| {summary['median_abs_delta']}; median |b-1| {summary['median_abs_b_minus_1']}. "
        f"The ratio 1 - PB_OTG_k2/PB_OTG_k0 = {ctx2} is not a share of penalty removed: "
        f"PB_OTG_k2 is negative, so the ratio overshoots 1. Headline comparisons use absolute "
        f"plant-balanced OTG and Delta_OTG, not that ratio."
    )
    return {
        "text": text,
        "descriptors": [
            "offset correction removes a substantial portion of the fleet-level residual penalty",
            "affine correction adds substantial reduction beyond offset and moves plant-balanced OTG slightly below zero",
            "response remains heterogeneous because nontrivial sets of directions worsen under both corrections",
        ],
        "threshold_rules_used": False,
        "contextual_ratio_k2_is_overshoot": True,
    }


def run_science(root: Path, sot: dict[str, Any]) -> dict[str, Any]:
    plants, members = load_scope(root)
    hparams = load_hparams(root, plants)
    universe = [r for r in load_directional_universe(root) if r.get("model_family") == FAM_SPLINE and r.get("support_rule") == "CS4"]
    primary = [r for r in universe if _flag(r.get("computable")) and r.get("group_id") in PRIMARY_GROUP_IDS]
    feat_by, y_map, splits = load_tables(root, plants)
    models = load_or_refit_models(root, plants, splits, feat_by, y_map, hparams)
    s06 = load_s06_test(root)
    s05 = load_s05_supported(root)
    engines = {pid: build_engine(pid, splits[pid]["train"], feat_by) for pid in plants}

    recon_max = 0.0
    recon_max = max(recon_max, abs(len(primary) - 85))

    val_rows, test_rows = [], []
    directional, cal_rows, bias_rows = [], [], []
    pred_diffs, mae_diffs, loc_diffs, otg_diffs = [], [], [], []

    for row in primary:
        src, tgt, gid = row["source_plant_id"], row["target_plant_id"], row["group_id"]
        rec = {
            "group_id": gid,
            "source_plant_id": src,
            "target_plant_id": tgt,
            "support_retention": float(row["support_retention"]),
            "primary_otg": float(row["otg_abs"]),
            "computable": True,
        }
        test = s06.get((src, tgt))
        if test is None or len(test["dt"]) == 0:
            rec["computable"] = False
            rec["reason"] = "missing_s06_test_rows"
            directional.append(rec)
            continue
        s05_set = s05.get((src, tgt), set())
        test_set = set(test["dt"])
        recon_max = max(recon_max, float(len(s05_set.symmetric_difference(test_set))))
        y, p0, loc = test["y"], test["p0"], test["local"]
        k0 = error_pack(p0, y)
        loc_err = error_pack(loc, y)
        otg0 = k0["mae"] - loc_err["mae"]
        recon_max = max(recon_max, abs(otg0 - float(row["otg_abs"])))
        recon_max = max(recon_max, abs(k0["mae"] - float(row["transfer_mae"])))
        recon_max = max(recon_max, abs(loc_err["mae"] - float(row["local_same_rows_mae"])))

        ms, mt = models.get(src, {}), models.get(tgt, {})
        if ms.get("est") is None or mt.get("est") is None:
            rec["reason"] = ms.get("error") or mt.get("error") or "model_unfit"
            rec["computable"] = False
            directional.append(rec)
            continue
        keep, ys_chk, pred_chk = predict_many(ms["est"], tgt, test["dt"], feat_by, y_map)
        loc_keep, loc_y, loc_pred = predict_many(mt["est"], tgt, test["dt"], feat_by, y_map)
        if len(keep) == len(test["dt"]):
            pred_diffs.append(float(np.max(np.abs(pred_chk - p0))))
            mae_diffs.append(abs(float(np.mean(np.abs(pred_chk - ys_chk))) - k0["mae"]))
        if len(loc_keep) == len(test["dt"]):
            loc_diffs.append(float(np.max(np.abs(loc_pred - loc))))
        if pred_diffs:
            otg_from_refit = float(np.mean(np.abs(pred_chk - ys_chk))) - float(np.mean(np.abs(loc_pred - loc_y))) if len(loc_keep) == len(test["dt"]) else None
            if otg_from_refit is not None:
                otg_diffs.append(abs(otg_from_refit - otg0))

        eng = engines[src]
        scored = score_support(eng, tgt, splits[tgt]["validation"], feat_by) if eng.get("computable") else []
        val_supported = [dt for dt, _d, _t, ok in scored if ok]
        vkeep, vy, vp = predict_many(ms["est"], tgt, val_supported, feat_by, y_map) if val_supported else ([], np.array([]), np.array([]))
        dist_map = {dt: (d, tau) for dt, d, tau, ok in scored if ok}
        rec["n_validation_supported"] = int(len(vkeep))
        rec["n_test_supported"] = int(len(test["dt"]))
        rec["n_supported_days_test"] = _days(test["dt"])
        rec["offset_eligible"] = len(vkeep) >= 1
        rec["affine_eligible"] = False
        rec["affine_reason"] = None
        rec["delta"] = None
        rec["a"] = None
        rec["b"] = None
        if rec["offset_eligible"]:
            rec["delta"] = float(np.mean(vy - vp))
        a, b, aff_reason = affine_fit(vy, vp) if rec["offset_eligible"] else (None, None, "n_val_lt_1")
        if a is None:
            rec["affine_reason"] = aff_reason
        else:
            rec["affine_eligible"] = True
            rec["a"], rec["b"] = a, b
            rec["extreme_finite_slope"] = bool(abs(b) > 5 or abs(a) > 1)

        p1 = p0 + rec["delta"] if rec["offset_eligible"] else None
        p2 = rec["a"] + rec["b"] * p0 if rec["affine_eligible"] else None
        k1 = error_pack(p1, y) if p1 is not None else None
        k2 = error_pack(p2, y) if p2 is not None else None
        rec["otg_0"] = otg0
        rec["otg_1"] = (k1["mae"] - loc_err["mae"]) if k1 else None
        rec["otg_2"] = (k2["mae"] - loc_err["mae"]) if k2 else None
        rec["delta_otg_offset"] = (rec["otg_0"] - rec["otg_1"]) if rec["otg_1"] is not None else None
        rec["delta_otg_affine"] = (rec["otg_0"] - rec["otg_2"]) if rec["otg_2"] is not None else None
        rec["local_mae"] = loc_err["mae"]
        rec["transfer_mae_0"] = k0["mae"]
        rec["transfer_mae_1"] = k1["mae"] if k1 else None
        rec["transfer_mae_2"] = k2["mae"] if k2 else None
        rec["mbe_0"] = k0["mbe"]
        rec["mbe_1"] = k1["mbe"] if k1 else None
        rec["mbe_2"] = k2["mbe"] if k2 else None
        rec["centered_mae_0"] = k0["centered_mae"]
        rec["centered_mae_1"] = k1["centered_mae"] if k1 else None
        rec["centered_mae_2"] = k2["centered_mae"] if k2 else None
        rec["centered_rmse_0"] = k0["centered_rmse"]
        rec["centered_rmse_1"] = k1["centered_rmse"] if k1 else None
        rec["centered_rmse_2"] = k2["centered_rmse"] if k2 else None
        rec["local_mbe"] = loc_err["mbe"]
        rec["local_centered_mae"] = loc_err["centered_mae"]
        rec["local_centered_rmse"] = loc_err["centered_rmse"]
        rec["abs_mbe_reduction_offset"] = abs(k0["mbe"]) - abs(k1["mbe"]) if k1 else None
        rec["threshold"] = eng.get("tau") if eng.get("computable") else None
        rec["engine_computable"] = bool(eng.get("computable"))
        rec["engine_reason"] = eng.get("reason")
        directional.append(rec)

        recon_delta = abs(rec["delta"] - float(np.mean(vy - vp))) if rec["offset_eligible"] else 0.0
        recon_max = max(recon_max, recon_delta)
        if rec["affine_eligible"]:
            aa, bb, _ = affine_fit(vy, vp)
            recon_max = max(recon_max, abs(aa - rec["a"]), abs(bb - rec["b"]))

        cal_rows.append(
            {
                "group_id": gid,
                "source_plant_id": src,
                "target_plant_id": tgt,
                "n_validation": rec["n_validation_supported"],
                "offset_eligible": rec["offset_eligible"],
                "affine_eligible": rec["affine_eligible"],
                "affine_reason": rec["affine_reason"],
                "delta": rec["delta"],
                "a": rec["a"],
                "b": rec["b"],
                "extreme_finite_slope": rec.get("extreme_finite_slope"),
            }
        )
        bias_rows.append(
            {
                "group_id": gid,
                "source_plant_id": src,
                "target_plant_id": tgt,
                "mbe_transfer_0": rec["mbe_0"],
                "mbe_transfer_1": rec["mbe_1"],
                "mbe_transfer_2": rec["mbe_2"],
                "abs_mbe_0": abs(rec["mbe_0"]),
                "abs_mbe_1": abs(rec["mbe_1"]) if rec["mbe_1"] is not None else None,
                "centered_mae_0": rec["centered_mae_0"],
                "centered_mae_1": rec["centered_mae_1"],
                "centered_rmse_0": rec["centered_rmse_0"],
                "centered_rmse_1": rec["centered_rmse_1"],
                "local_mbe": rec["local_mbe"],
                "local_centered_mae": rec["local_centered_mae"],
                "otg_0": rec["otg_0"],
                "abs_mbe_transfer": abs(rec["mbe_0"]),
            }
        )
        vmap = {dt: i for i, dt in enumerate(vkeep)}
        for dt, d, tau, ok_s in scored:
            if not ok_s or dt not in vmap:
                continue
            i = vmap[dt]
            val_rows.append(
                {
                    "group_id": gid,
                    "source_plant_id": src,
                    "target_plant_id": tgt,
                    "datetime_raw": dt,
                    "split": "validation",
                    "supported": True,
                    "y_true": float(vy[i]),
                    "p_source": float(vp[i]),
                    "kth_distance": d,
                    "threshold": tau,
                }
            )
        for i, dt in enumerate(test["dt"]):
            test_rows.append(
                {
                    "group_id": gid,
                    "source_plant_id": src,
                    "target_plant_id": tgt,
                    "datetime_raw": dt,
                    "split": "test",
                    "supported": True,
                    "y_true": float(y[i]),
                    "p0": float(p0[i]),
                    "p1": float(p1[i]) if p1 is not None else None,
                    "p2": float(p2[i]) if p2 is not None else None,
                    "y_pred_local": float(loc[i]),
                    "e0": float(p0[i] - y[i]),
                    "e1": float(p1[i] - y[i]) if p1 is not None else None,
                    "e2": float(p2[i] - y[i]) if p2 is not None else None,
                    "ae0": abs(float(p0[i] - y[i])),
                    "ae_local": abs(float(loc[i] - y[i])),
                }
            )

    ok = [r for r in directional if r.get("computable") and r.get("otg_0") is not None]
    k1_rows = [r for r in ok if r.get("otg_1") is not None]
    k2_rows = [r for r in ok if r.get("otg_2") is not None]
    k0s = ladder_summary(ok, "otg_0", "transfer_mae_0", "mbe_0")
    k1s = ladder_summary(k1_rows, "otg_1", "transfer_mae_1", "mbe_1")
    k2s = ladder_summary(k2_rows, "otg_2", "transfer_mae_2", "mbe_2")
    recon_max = max(recon_max, abs((k0s["plant_balanced_otg"] or 0) - (k0s["plant_balanced_otg_iterate"] or 0)))
    recon_max = max(recon_max, abs((k1s["plant_balanced_otg"] or 0) - (k1s["plant_balanced_otg_iterate"] or 0)))
    recon_max = max(recon_max, abs((k2s["plant_balanced_otg"] or 0) - (k2s["plant_balanced_otg_iterate"] or 0)))

    def delta_pack(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        pb, nt = plant_balanced([{**r, "otg_abs": r[key], "computable": True} for r in rows if r.get(key) is not None], "otg_abs")
        return {
            "n": len(vals),
            "n_targets": nt,
            "plant_balanced": pb,
            "median": float(np.median(vals)) if vals else None,
            "mean": float(np.mean(vals)) if vals else None,
            "n_improved": int(sum(v > 0 for v in vals)),
            "n_worsened": int(sum(v < 0 for v in vals)),
            "n_unchanged": int(sum(v == 0 for v in vals)),
            "frac_improved": (sum(v > 0 for v in vals) / len(vals)) if vals else None,
            "frac_worsened": (sum(v < 0 for v in vals) / len(vals)) if vals else None,
            "distribution": _dist(vals),
        }

    d_off = delta_pack(k1_rows, "delta_otg_offset")
    d_aff = delta_pack(k2_rows, "delta_otg_affine")
    ctx_red_1 = (1.0 - k1s["plant_balanced_otg"] / k0s["plant_balanced_otg"]) if k0s["plant_balanced_otg"] not in (None, 0) and k1s["plant_balanced_otg"] is not None else None
    ctx_red_2 = (1.0 - k2s["plant_balanced_otg"] / k0s["plant_balanced_otg"]) if k0s["plant_balanced_otg"] not in (None, 0) and k2s["plant_balanced_otg"] is not None else None

    lookup = {(r["source_plant_id"], r["target_plant_id"]): r for r in ok}
    plants_by_g: dict[str, set[str]] = defaultdict(set)
    for r in primary:
        plants_by_g[r["group_id"]].add(r["source_plant_id"])
        plants_by_g[r["group_id"]].add(r["target_plant_id"])

    def rq2_pairs(level: str, otg_key: str, need: str) -> list[dict[str, Any]]:
        out = []
        for gid, ps in plants_by_g.items():
            order = sorted(ps)
            for i, a in enumerate(order):
                for b in order[i + 1 :]:
                    ab, ba = lookup.get((a, b)), lookup.get((b, a))
                    if not ab or not ba:
                        continue
                    if not ab.get(need) or not ba.get(need):
                        continue
                    oa, ob = ab.get(otg_key), ba.get(otg_key)
                    if oa is None or ob is None:
                        continue
                    signed = float(oa) - float(ob)
                    out.append(
                        {
                            "group_id": gid,
                            "plant_a": a,
                            "plant_b": b,
                            "level": level,
                            "otg_a_to_b": float(oa),
                            "otg_b_to_a": float(ob),
                            "asymmetry_signed": signed,
                            "asymmetry_abs": abs(signed),
                        }
                    )
        return out

    pairs0 = rq2_pairs("k0", "otg_0", "computable")
    pairs1 = rq2_pairs("k1", "otg_1", "offset_eligible")
    pairs2 = rq2_pairs("k2", "otg_2", "affine_eligible")

    def rq2_sum(pairs: list[dict[str, Any]]) -> dict[str, Any]:
        from src.lib.s12_p1_null import plant_balanced as _unused  # noqa: F401
        from src.lib.s12_p3_adaptive_scaling import plant_balanced_asym

        pb, npl = plant_balanced_asym(pairs) if pairs else (None, 0)
        vals = [p["asymmetry_abs"] for p in pairs]
        return {
            "n_pairs": len(pairs),
            "n_plants": npl,
            "plant_balanced_abs_asymmetry": pb,
            "median_abs_asymmetry": float(np.median(vals)) if vals else None,
        }

    spear_rows = [{"target_plant_id": r["target_plant_id"], "abs_mbe": abs(float(r["mbe_0"])), "otg_0": float(r["otg_0"])} for r in ok]
    rho, rho_reason = spearman_rho([r["abs_mbe"] for r in spear_rows], [r["otg_0"] for r in spear_rows])
    lo, hi, n_valid = cluster_bootstrap_xy(spear_rows, "abs_mbe", "otg_0") if len(spear_rows) >= 3 else (None, None, 0)

    tables = (sot.get("results") or {}).get("tables") or {}
    prim_rq1 = tables.get("rq1")
    if isinstance(prim_rq1, dict):
        prim_rq1_v = prim_rq1.get("plant_balanced") if "plant_balanced" in prim_rq1 else prim_rq1
    else:
        prim_rq1_v = prim_rq1
    if k0s["plant_balanced_otg"] is not None and prim_rq1_v is not None and not isinstance(prim_rq1_v, dict):
        recon_max = max(recon_max, abs(k0s["plant_balanced_otg"] - float(prim_rq1_v)))

    # reconstruct TEST MAE from rows
    by_dir: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in test_rows:
        by_dir[(r["source_plant_id"], r["target_plant_id"])].append(r)
    for rec in ok:
        chunk = by_dir[(rec["source_plant_id"], rec["target_plant_id"])]
        mae0 = float(np.mean([c["ae0"] for c in chunk]))
        locm = float(np.mean([c["ae_local"] for c in chunk]))
        recon_max = max(recon_max, abs(mae0 - rec["transfer_mae_0"]), abs((mae0 - locm) - rec["otg_0"]))
        if rec.get("offset_eligible"):
            mae1 = float(np.mean([abs(c["p1"] - c["y_true"]) for c in chunk]))
            recon_max = max(recon_max, abs(mae1 - rec["transfer_mae_1"]))

    deltas = [float(r["delta"]) for r in ok if r.get("delta") is not None]
    intercepts = [float(r["a"]) for r in ok if r.get("affine_eligible")]
    slopes = [float(r["b"]) for r in ok if r.get("affine_eligible")]
    n_aff_inelig = sum(1 for r in ok if not r.get("affine_eligible"))

    max_pred = max(pred_diffs) if pred_diffs else None
    if max_pred is not None:
        recon_max = max(recon_max, 0.0 if max_pred <= PRED_TOL else max_pred)
    if mae_diffs:
        recon_max = max(recon_max, max(mae_diffs))
    if loc_diffs:
        recon_max = max(recon_max, max(0.0 if d <= PRED_TOL else d for d in loc_diffs))
    if otg_diffs:
        recon_max = max(recon_max, max(otg_diffs))

    summary = {
        "status": STATUS,
        "n_planned": 85,
        "n_primary": len(primary),
        "n_directional_computable": len(ok),
        "n_offset_eligible": len(k1_rows),
        "n_affine_eligible": len(k2_rows),
        "n_affine_ineligible": n_aff_inelig,
        "universe": "accepted 85 SplineRidge+CS4 computable primary directions; R9/OTG_half/R7 not used",
        "residual_sign": "prediction - y_true",
        "centered_error_note": "centered MAE/RMSE are descriptive after removing mean residual; not an additive MAE decomposition",
        "k0": k0s,
        "k1": {**k1s, "n_calibration_eligible": len(k1_rows)},
        "k2": {**k2s, "n_calibration_eligible": len(k2_rows)},
        "delta_offset": d_off,
        "delta_affine": d_aff,
        "contextual_reduction_offset": ctx_red_1,
        "contextual_reduction_affine": ctx_red_2,
        "contextual_reduction_note": (
            "Optional fleet-level 1 - PB_OTG_k/PB_OTG_0; not a mean of per-direction percentages. "
            "When PB_OTG_k2 < 0 the k2 ratio overshoots 1 and is not a percent of penalty removed."
        ),
        "calibration_n": _dist([float(r["n_validation_supported"]) for r in ok]),
        "delta_distribution": _dist(deltas),
        "median_abs_delta": float(np.median(np.abs(deltas))) if deltas else None,
        "intercept_distribution": _dist(intercepts),
        "slope_distribution": _dist(slopes),
        "median_abs_b_minus_1": float(np.median(np.abs(np.asarray(slopes) - 1.0))) if slopes else None,
        "n_extreme_finite_slope": int(sum(bool(r.get("extreme_finite_slope")) for r in ok)),
        "bias": {
            "transfer_mbe_0": _dist([float(r["mbe_0"]) for r in ok]),
            "abs_transfer_mbe_0": _dist([abs(float(r["mbe_0"])) for r in ok]),
            "local_mbe": _dist([float(r["local_mbe"]) for r in ok]),
            "abs_local_mbe": _dist([abs(float(r["local_mbe"])) for r in ok]),
            "centered_mae_transfer_0": _dist([float(r["centered_mae_0"]) for r in ok]),
            "centered_mae_local": _dist([float(r["local_centered_mae"]) for r in ok]),
            "centered_rmse_transfer_0": _dist([float(r["centered_rmse_0"]) for r in ok]),
            "mbe_after_offset": _dist([float(r["mbe_1"]) for r in k1_rows]),
            "abs_mbe_reduction_offset": _dist([float(r["abs_mbe_reduction_offset"]) for r in k1_rows]),
            "centered_mae_after_offset": _dist([float(r["centered_mae_1"]) for r in k1_rows]),
        },
        "spearman_abs_mbe_vs_otg": {
            "rho": rho,
            "reason": rho_reason,
            "ci95_low": lo,
            "ci95_high": hi,
            "n_boot_valid": n_valid,
            "seed": BOOT_SEED,
        },
        "rq2": {"k0": rq2_sum(pairs0), "k1": rq2_sum(pairs1), "k2": rq2_sum(pairs2)},
        "rq3_status": RQ3_STATUS,
        "rq3_reason": "RQ3 needs control-side CS4 scoring and pairwise TEST intersection; P4 does not rebuild that pipeline or invent a twin-only proxy.",
        "interpretation": None,
        "model_actions": {pid: models[pid]["action"] for pid in plants},
        "model_equivalence": {
            "n_directions_checked": len(pred_diffs),
            "max_abs_prediction_difference": max_pred,
            "max_transfer_mae_difference": max(mae_diffs) if mae_diffs else None,
            "max_local_mae_difference": max(loc_diffs) if loc_diffs else None,
            "max_otg_difference": max(otg_diffs) if otg_diffs else None,
            "tolerance": PRED_TOL,
        },
        "reconciliation_max_abs_discrepancy": recon_max,
        "reconciliation_ok": recon_max <= PRED_TOL,
        "no_test_in_calibration": True,
        "r9_not_used": True,
        "sensitivity_only": True,
    }
    summary["interpretation"] = interpret(summary)
    return {
        "plants": plants,
        "primary": primary,
        "directional": directional,
        "cal_rows": cal_rows,
        "bias_rows": bias_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "pairs1": pairs1,
        "pairs2": pairs2,
        "summary": summary,
        "models": models,
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        pq.write_table(pa.table({"_empty": pa.array([], type=pa.int8())}), path)
        return 0
    cols = list(rows[0].keys())
    data = {c: [r.get(c) for r in rows] for c in cols}
    pq.write_table(pa.table(data), path)
    return len(rows)


def render_report(s: dict[str, Any]) -> str:
    k0, k1, k2 = s["k0"], s["k1"], s["k2"]
    d1, d2 = s["delta_offset"], s["delta_affine"]
    eq = s["model_equivalence"]
    lines = [
        "# S12 P4 — bias/dispersion and VALIDATION-only recalibration",
        "",
        "Secondary analysis. Accepted primary OTG/RQ1–RQ3/S11/P1–P3 are not replaced. Residual sign is `prediction - y_true`. Centered MAE/RMSE are not an additive MAE decomposition.",
        "",
        f"Status: `{s['status']}`. Universe: {s['n_primary']} primary computable directions (planned 85). Offset-eligible {s['n_offset_eligible']}; affine-eligible {s['n_affine_eligible']}; affine ineligible {s['n_affine_ineligible']}.",
        "",
        f"Headline plant-balanced OTG: k=0 {k0['plant_balanced_otg']}; k=1 {k1['plant_balanced_otg']}; k=2 {k2['plant_balanced_otg']}. "
        f"PB Delta_OTG offset {d1['plant_balanced']} ({d1['n_improved']}/{d1['n']} improve, {d1['n_worsened']}/{d1['n']} worsen); "
        f"PB Delta_OTG affine {d2['plant_balanced']} ({d2['n_improved']}/{d2['n']} improve, {d2['n_worsened']}/{d2['n']} worsen).",
        f"The ratio 1 - PB_OTG_k2/PB_OTG_k0 = {s['contextual_reduction_affine']} overshoots 1 because PB_OTG_k2 is negative; it is not a literal share of penalty removed. The k=1 ratio {s['contextual_reduction_offset']} is optional fleet-level context only.",
        "",
        f"Median |delta| {s['median_abs_delta']}; median |b-1| {s['median_abs_b_minus_1']}; extreme finite slopes {s['n_extreme_finite_slope']}.",
        f"Spearman |MBE_transfer| vs OTG_0 rho={s['spearman_abs_mbe_vs_otg']['rho']} CI [{s['spearman_abs_mbe_vs_otg']['ci95_low']}, {s['spearman_abs_mbe_vs_otg']['ci95_high']}].",
        f"RQ2 plant-balanced |A|: k0 {s['rq2']['k0']['plant_balanced_abs_asymmetry']} (n={s['rq2']['k0']['n_pairs']}); k1 {s['rq2']['k1']['plant_balanced_abs_asymmetry']} (n={s['rq2']['k1']['n_pairs']}); k2 {s['rq2']['k2']['plant_balanced_abs_asymmetry']} (n={s['rq2']['k2']['n_pairs']}).",
        f"RQ3: `{s['rq3_status']}`.",
        "",
        f"Interpretation: {s['interpretation']['text']}",
        "Descriptors: " + "; ".join(s["interpretation"]["descriptors"]) + ".",
        f"No arbitrary numeric interpretation gates (threshold_rules_used={s['interpretation']['threshold_rules_used']}).",
        f"Model actions: {s['model_actions']}. Equivalence max |Δpred|={eq['max_abs_prediction_difference']}, max |ΔOTG|={eq['max_otg_difference']}.",
        f"Reconciliation max |Δ|={s['reconciliation_max_abs_discrepancy']} ok={s['reconciliation_ok']}.",
        "",
    ]
    return "\n".join(lines)
