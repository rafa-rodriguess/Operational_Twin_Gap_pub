"""S12 P8: leave-target-out pooled SplineRidge fleet baseline (isolated)."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from config.protocol import FAM_SPLINE, PRIMARY_GROUP_IDS
from src.lib.models import _spline
from src.lib.s12_p1_null import (
    FEATURES,
    _days,
    load_directional_universe,
    load_hparams,
    load_scope,
    load_tables,
    plant_balanced,
    plant_balanced_iterate,
)
from src.lib.s12_p2_retention import percentile_ci

STATUS = "S12_P8_POOLED_FLEET_BASELINE_MATERIALIZED"
RQ3_STATUS = "P8_RQ3_NOT_RUN_POOLED_BASELINE_SCOPE"
RQ2_POLICY = "not_redefined_pooled_is_target_specific_not_source_asymmetry"
PRIMARY_RQ1 = 0.009723604857785641
SEED = 20260921
N_BOOT = 5000
TOL = 1e-12
ZERO_GAIN_TOL = 1e-15
REF_ROW_PB_OTG = -0.00497426794148001
REF_ROW_PB_GAIN = 0.014697872799265651
REF_ROW_CI = (0.010391494156256828, 0.0183922243087188)
REF_ROW_NPOS = 70
REF_ROW_NNEG = 15
G1 = "TW_04b9f5d95694"
G2 = "TW_4467a039e00b"
G4 = "TW_88a108921af7"
GROUP_LABELS = {G1: "G1", G2: "G2", G4: "G4"}
VARIANTS = ("row_pooled", "plant_balanced")


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


def _flag(val: Any) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def _finite(val: Any) -> float | None:
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def hp_key(params: dict[str, Any]) -> str:
    return json.dumps({"alpha": params["alpha"], "n_knots": params["n_knots"]}, sort_keys=True)


def dist_pack(vals: list[float]) -> dict[str, Any]:
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


def load_candidates(root: Path, plants: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[str, dict[str, Any]] = {}
    rows = []
    with (root / "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model_family") != FAM_SPLINE or row.get("fit_scope") != "full":
                continue
            if not _flag(row.get("selected")):
                continue
            if row["plant_id"] not in plants:
                continue
            params = json.loads(row["hyperparameters"])
            key = hp_key(params)
            rec = by_key.setdefault(key, {"candidate_id": f"S04_{len(by_key):02d}", "params": {"alpha": params["alpha"], "n_knots": params["n_knots"]}, "source_plants": []})
            rec["source_plants"].append(row["plant_id"])
            rows.append({"plant_id": row["plant_id"], "candidate_key": key, "alpha": params["alpha"], "n_knots": params["n_knots"]})
    cands = [by_key[k] for k in sorted(by_key)]
    for i, c in enumerate(cands):
        c["candidate_id"] = f"S04_{i:02d}"
        c["source_plants"] = sorted(set(c["source_plants"]))
    return cands, rows


def stack_split(plants: list[str], dts_map: dict[str, list[str]], feat_by, y_map):
    xs, ys, pids, dts = [], [], [], []
    for pid in plants:
        for dt in dts_map[pid]:
            vec = feat_by.get((pid, dt))
            yv = y_map.get((pid, dt))
            if vec is None or yv is None:
                continue
            xs.append(vec)
            ys.append(yv)
            pids.append(pid)
            dts.append(dt)
    if not xs:
        return np.empty((0, len(FEATURES))), np.array([]), np.array([], dtype=object), []
    return np.vstack(xs), np.asarray(ys, dtype=float), np.asarray(pids, dtype=object), dts


def donor_mae(pred: np.ndarray, y: np.ndarray, plants: np.ndarray) -> tuple[float | None, dict[str, float]]:
    by: dict[str, list[float]] = defaultdict(list)
    for p, a, b in zip(plants, pred, y):
        by[str(p)].append(abs(float(a) - float(b)))
    maes = {k: float(np.mean(v)) for k, v in by.items() if v}
    if not maes:
        return None, {}
    return float(sum(maes.values()) / len(maes)), maes


def fit_pooled(params: dict[str, Any], X, y, plants: np.ndarray | None, variant: str):
    est = _spline(params)
    if variant == "row_pooled":
        est.fit(X, y)
        w = np.ones(len(y), dtype=float)
        w_by = {str(p): float(np.sum(w[plants == p])) for p in np.unique(plants)}
        meta = {
            "scaler_sample_weight": False,
            "ridge_sample_weight": False,
            "same_weight_vector_scaler_ridge": True,
            "spline_transformer_sample_weight": False,
            "spline_transformer_knots": "uniform_accepted",
        }
        return est, w, w_by, meta
    counts = Counter(str(p) for p in plants)
    w = np.array([1.0 / counts[str(p)] for p in plants], dtype=float)
    est.fit(X, y, scaler__sample_weight=w, ridge__sample_weight=w)
    w_by = {str(p): float(np.sum(w[plants == p])) for p in np.unique(plants)}
    meta = {
        "scaler_sample_weight": True,
        "ridge_sample_weight": True,
        "same_weight_vector_scaler_ridge": True,
        "spline_transformer_sample_weight": False,
        "spline_transformer_knots": "uniform_accepted",
    }
    return est, w, w_by, meta


def load_s07_primary(root: Path) -> list[dict[str, Any]]:
    out = []
    for row in load_directional_universe(root):
        if row.get("model_family") != FAM_SPLINE or row.get("support_rule") != "CS4":
            continue
        if row.get("group_id") not in PRIMARY_GROUP_IDS:
            continue
        if not _flag(row.get("computable")):
            continue
        rec = dict(row)
        rec["n_supported"] = int(row["n_supported"])
        rec["support_retention"] = float(row["support_retention"])
        rec["transfer_mae"] = float(row["transfer_mae"])
        rec["local_same_rows_mae"] = float(row["local_same_rows_mae"])
        rec["otg_abs"] = float(row["otg_abs"])
        rec["otg_rel"] = _finite(row.get("otg_rel"))
        rec["computable"] = True
        out.append(rec)
    return out


def load_s06_and_s05(root: Path):
    s06 = pq.read_table(
        root / "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet",
        columns=[
            "source_plant_id", "target_plant_id", "model_family", "support_rule",
            "datetime_raw", "y_true", "y_pred_transfer", "y_pred_local",
        ],
    )
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for src, tgt, fam, rule, dt, y, pt, pl in zip(
        s06["source_plant_id"].to_pylist(),
        s06["target_plant_id"].to_pylist(),
        s06["model_family"].to_pylist(),
        s06["support_rule"].to_pylist(),
        s06["datetime_raw"].to_pylist(),
        s06["y_true"].to_pylist(),
        s06["y_pred_transfer"].to_pylist(),
        s06["y_pred_local"].to_pylist(),
    ):
        if fam != FAM_SPLINE or str(rule) != "CS4":
            continue
        rec = pairs.setdefault((str(src), str(tgt)), {"dt": [], "y": [], "twin": [], "local": []})
        rec["dt"].append(str(dt))
        rec["y"].append(float(y))
        rec["twin"].append(float(pt))
        rec["local"].append(float(pl))
    s05 = pq.read_table(
        root / "artifacts/s05/S05_SUPPORT_MASKS.parquet",
        columns=["source_plant_id", "target_plant_id", "rule_id", "datetime_raw", "supported", "computable", "kth_distance", "threshold"],
    )
    meta: dict[tuple[str, str, str], tuple[float, float]] = {}
    supported: dict[tuple[str, str], set[str]] = defaultdict(set)
    for src, tgt, rid, dt, sup, comp, d, tau in zip(
        s05["source_plant_id"].to_pylist(),
        s05["target_plant_id"].to_pylist(),
        s05["rule_id"].to_pylist(),
        s05["datetime_raw"].to_pylist(),
        s05["supported"].to_pylist(),
        s05["computable"].to_pylist(),
        s05["kth_distance"].to_pylist(),
        s05["threshold"].to_pylist(),
    ):
        if str(rid) != "CS4" or not comp or not sup:
            continue
        key_dt = str(dt)
        supported[(str(src), str(tgt))].add(key_dt)
        if d is not None and tau is not None:
            meta[(str(src), str(tgt), key_dt)] = (float(d), float(tau))
    aligned = {}
    for key, rec in pairs.items():
        keep = supported.get(key, set())
        dts, ys, tw, loc = [], [], [], []
        for dt, y, a, b in zip(rec["dt"], rec["y"], rec["twin"], rec["local"]):
            if dt in keep:
                dts.append(dt)
                ys.append(y)
                tw.append(a)
                loc.append(b)
        aligned[key] = {
            "dt": dts,
            "y": np.asarray(ys, dtype=float),
            "twin": np.asarray(tw, dtype=float),
            "local": np.asarray(loc, dtype=float),
        }
    return aligned, meta


def rq1_block(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    pb, nt = plant_balanced(rows, field)
    pb2, nt2 = plant_balanced_iterate(rows, field)
    vals = [float(r[field]) for r in rows if r.get("computable")]
    rels = []
    for r in rows:
        loc = float(r["local_same_rows_mae"])
        if loc != 0:
            rels.append(float(r[field]) / loc)
    return {
        "n_directional": len(rows),
        "n_targets": nt,
        "directional_mean": float(np.mean(vals)) if vals else None,
        "directional_median": float(np.median(vals)) if vals else None,
        "plant_balanced": pb,
        "plant_balanced_iterate": pb2,
        "n_targets_iterate": nt2,
        "rel_median": float(np.median(rels)) if rels else None,
        "rel_p75": float(np.quantile(rels, 0.75, method="linear")) if rels else None,
    }


def gain_pack(vals: list[float], pb: float | None) -> dict[str, Any]:
    pack = dist_pack(vals)
    n = len(vals)
    n_pos = int(sum(v > ZERO_GAIN_TOL for v in vals))
    n_neg = int(sum(v < -ZERO_GAIN_TOL for v in vals))
    n_zero = n - n_pos - n_neg
    pack.update({
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "frac_pos": (n_pos / n) if n else None,
        "frac_neg": (n_neg / n) if n else None,
        "frac_zero": (n_zero / n) if n else None,
        "plant_balanced_mean": pb,
    })
    return pack


def bootstrap_pb_gain(rows: list[dict[str, Any]], field: str) -> tuple[list[dict[str, Any]], float | None, float | None]:
    targets = sorted({str(r["target_plant_id"]) for r in rows})
    by_t: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_t[str(r["target_plant_id"])].append(float(r[field]))
    rng = np.random.default_rng(SEED)
    reps = []
    stats = []
    for i in range(N_BOOT):
        sampled = rng.choice(targets, size=len(targets), replace=True)
        means = [float(np.mean(by_t[str(t)])) for t in sampled]
        stat = float(np.mean(means))
        reps.append({"replicate_id": i, "field": field, "pb_gain": stat})
        stats.append(stat)
    lo, hi = percentile_ci(stats)
    return reps, lo, hi


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        pq.write_table(pa.table({"_empty": pa.array([], type=pa.int8())}), path)
        return 0
    cols = list(rows[0].keys())
    pq.write_table(pa.table({c: [r.get(c) for r in rows] for c in cols}), path)
    return len(rows)


def interpret(s: dict[str, Any]) -> dict[str, Any]:
    notes = []
    gr = s["pb_gain_row"]
    gb = s["pb_gain_bal"]
    cir = s["bootstrap"]["row_pooled"]
    cib = s["bootstrap"]["plant_balanced"]
    if cir["ci_lo"] is not None and cir["ci_lo"] > 0 and cib["ci_lo"] is not None and cib["ci_lo"] > 0:
        notes.append("pooled_outperforms_twin: plant-balanced GAIN is positive for both variants and both bootstrap CIs lie above zero. Nominal twin equivalence remains scientifically informative but is not the strongest zero-shot deployment strategy among those tested.")
    elif cir["ci_hi"] is not None and cir["ci_hi"] < 0 and cib["ci_hi"] is not None and cib["ci_hi"] < 0:
        notes.append("twin_specificity_adds_value: pooled plant-balanced GAIN is negative and both bootstrap CIs lie below zero.")
    else:
        notes.append("similar_fleet_performance_with_heterogeneous_targets: fleet-level GAIN is not uniformly signed by bootstrap CIs while target-level gains vary.")
    n_diff = int(s["selection"]["n_targets_differing_hparams"])
    d_otg = abs(float(s["row_pooled_rq1"]["plant_balanced"]) - float(s["plant_balanced_rq1"]["plant_balanced"]))
    d_gain = abs(float(s["pb_gain_row"]) - float(s["pb_gain_bal"]))
    notes.append(
        f"row- and plant-balanced pooling select different configurations on {n_diff}/14 targets; "
        f"their PB OTGs differ by {d_otg} and PB GAINs differ by {d_gain}; "
        f"therefore weighting choice affects the pooled baseline to that observed degree."
    )
    g2 = s["groups"]["G2"]["gain_row"]["plant_balanced_mean"]
    ng = s["groups"]["nonG2"]["gain_row"]["plant_balanced_mean"]
    if g2 is not None and ng is not None and ((g2 > 0) != (ng > 0)):
        notes.append("group_dependent_result: G2 and non-G2 plant-balanced GAIN signs differ (non-G2 is a small descriptive cohort).")
    elif g2 is not None and ng is not None:
        notes.append("G2 versus non-G2 plant-balanced GAIN is reported descriptively; non-G2 contains few plants.")
    text = " ".join(notes)
    return {
        "text": text,
        "notes": notes,
        "threshold_rules_used": False,
        "pb_gain_row": gr,
        "pb_gain_bal": gb,
    }


def run_science(root: Path, models_dir: Path) -> dict[str, Any]:
    plants, members = load_scope(root)
    hparams = load_hparams(root, plants)
    candidates, cand_plant_rows = load_candidates(root, plants)
    feat_by, y_map, splits = load_tables(root, plants)
    s07 = load_s07_primary(root)
    aligned, s05_meta = load_s06_and_s05(root)
    deltas: list[float] = []

    def note(a, b, name: str) -> None:
        if a is None or b is None:
            deltas.append(float("inf"))
            return
        deltas.append(abs(float(a) - float(b)))

    cand_csv = []
    for c in candidates:
        cand_csv.append({
            "candidate_id": c["candidate_id"],
            "alpha": c["params"]["alpha"],
            "n_knots": c["params"]["n_knots"],
            "hyperparameters": hp_key(c["params"]),
            "n_s04_source_plants": len(c["source_plants"]),
            "s04_source_plants": ",".join(c["source_plants"]),
            "features": ",".join(FEATURES),
        })

    sel_rows = []
    train_rows = []
    models: dict[str, dict[str, Any]] = {v: {} for v in VARIANTS}
    model_paths: list[Path] = []
    leakage = {
        "target_in_train": False,
        "target_in_val": False,
        "n_targets_fit": 0,
        "donors_not_13": [],
        "features": list(FEATURES),
        "id_features_used": False,
        "weighting": {
            "plant_balanced_w_ij": "1/n_j",
            "scaler_sample_weight": True,
            "ridge_sample_weight": True,
            "same_weight_vector_scaler_ridge": True,
            "spline_transformer_sample_weight": False,
            "spline_transformer_knots": "uniform_accepted",
            "no_target_derived_preprocessing": True,
        },
    }

    for tgt in plants:
        donors = [p for p in plants if p != tgt]
        if len(donors) != 13:
            leakage["donors_not_13"].append(tgt)
        tr_map = {p: splits[p]["train"] for p in donors}
        va_map = {p: splits[p]["validation"] for p in donors}
        Xtr, ytr, ptr, dtr = stack_split(donors, tr_map, feat_by, y_map)
        Xva, yva, pva, _dva = stack_split(donors, va_map, feat_by, y_map)
        if any(str(pid) == tgt for pid in ptr):
            leakage["target_in_train"] = True
        if any(pid == tgt for pid in pva):
            leakage["target_in_val"] = True
        n_tr_by = Counter(str(p) for p in ptr)
        n_va_by = Counter(str(p) for p in pva)
        for variant in VARIANTS:
            scored = []
            for cand in candidates:
                try:
                    est, w, w_by, wmeta = fit_pooled(cand["params"], Xtr, ytr, ptr, variant)
                    pred = np.asarray(est.predict(Xva), dtype=float)
                    score, per = donor_mae(pred, yva, pva)
                    status = "ok" if score is not None and np.all(np.isfinite(pred)) else "nonfinite"
                except Exception as exc:  # noqa: BLE001
                    est, w, w_by, wmeta, score, per, status = None, None, {}, {}, None, {}, f"fit_error:{type(exc).__name__}"
                scored.append({"cand": cand, "est": est, "score": score, "per": per, "status": status, "w_by": w_by, "wmeta": wmeta})
                sel_rows.append({
                    "target_plant_id": tgt,
                    "variant": variant,
                    "candidate_id": cand["candidate_id"],
                    "hyperparameters": hp_key(cand["params"]),
                    "alpha": cand["params"]["alpha"],
                    "n_knots": cand["params"]["n_knots"],
                    "donor_equal_mean_val_mae": score,
                    "fit_status": status,
                    "selected": False,
                    "n_donor_train": int(len(ytr)),
                    "n_donor_val": int(len(yva)),
                })
            ok = [r for r in scored if r["status"] == "ok" and r["score"] is not None]
            if not ok:
                raise RuntimeError(f"no trainable candidate for {tgt} {variant}")
            ok.sort(key=lambda r: (float(r["score"]), hp_key(r["cand"]["params"])))
            best, runner = ok[0], ok[1] if len(ok) > 1 else None
            est, w, w_by, wmeta = fit_pooled(best["cand"]["params"], Xtr, ytr, ptr, variant)
            models[variant][tgt] = est
            path = models_dir / variant / f"{tgt}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(est, path)
            model_paths.append(path)
            for row in sel_rows:
                if row["target_plant_id"] == tgt and row["variant"] == variant and row["candidate_id"] == best["cand"]["candidate_id"]:
                    row["selected"] = True
                    row["runner_up_val_mae"] = runner["score"] if runner else None
                    row["selection_gap"] = (float(runner["score"]) - float(best["score"])) if runner else None
            w_vals = list(w_by.values())
            w_equal = (max(w_vals) - min(w_vals)) <= TOL if w_vals else False
            if variant == "plant_balanced":
                note(True, w_equal, "w")
                if not w_equal:
                    deltas.append(max(w_vals) - min(w_vals) if w_vals else float("inf"))
            train_rows.append({
                "target_plant_id": tgt,
                "variant": variant,
                "n_donor_plants": len(donors),
                "donors": ",".join(donors),
                "n_donor_train_total": int(len(ytr)),
                "n_donor_val_total": int(len(yva)),
                "train_rows_by_plant": json.dumps(dict(sorted(n_tr_by.items())), sort_keys=True),
                "val_rows_by_plant": json.dumps(dict(sorted(n_va_by.items())), sort_keys=True),
                "effective_total_weight_by_plant": json.dumps({k: w_by[k] for k in sorted(w_by)}, sort_keys=True),
                "balanced_weights_equal": w_equal if variant == "plant_balanced" else "",
                "selected_candidate_id": best["cand"]["candidate_id"],
                "selected_hyperparameters": hp_key(best["cand"]["params"]),
                "selected_val_mae": best["score"],
                "runner_up_val_mae": runner["score"] if runner else None,
                "selection_gap": (float(runner["score"]) - float(best["score"])) if runner else None,
                "held_out_target_train_rows": 0,
                "held_out_target_val_rows": 0,
                "scaler_sample_weight": wmeta.get("scaler_sample_weight"),
                "ridge_sample_weight": wmeta.get("ridge_sample_weight"),
                "same_weight_vector_scaler_ridge": wmeta.get("same_weight_vector_scaler_ridge"),
                "spline_transformer_sample_weight": wmeta.get("spline_transformer_sample_weight"),
            })
        leakage["n_targets_fit"] += 1

    test_rows = []
    directional = []
    for rec in s07:
        src, tgt = rec["source_plant_id"], rec["target_plant_id"]
        pack = aligned.get((src, tgt))
        if pack is None or len(pack["dt"]) == 0:
            raise RuntimeError(f"missing S06/S05 rows for {src}->{tgt}")
        note(len(pack["dt"]), rec["n_supported"], "nsup")
        X = []
        keep_idx = []
        for i, dt in enumerate(pack["dt"]):
            vec = feat_by.get((tgt, dt))
            if vec is None:
                continue
            X.append(vec)
            keep_idx.append(i)
        if len(keep_idx) != len(pack["dt"]):
            deltas.append(abs(len(keep_idx) - len(pack["dt"])))
        Xm = np.vstack(X)
        y = pack["y"][keep_idx]
        twin = pack["twin"][keep_idx]
        loc = pack["local"][keep_idx]
        pr = np.asarray(models["row_pooled"][tgt].predict(Xm), dtype=float)
        pb = np.asarray(models["plant_balanced"][tgt].predict(Xm), dtype=float)
        twin_mae = float(np.mean(np.abs(twin - y)))
        loc_mae = float(np.mean(np.abs(loc - y)))
        row_mae = float(np.mean(np.abs(pr - y)))
        bal_mae = float(np.mean(np.abs(pb - y)))
        note(twin_mae, rec["transfer_mae"], "twin")
        note(loc_mae, rec["local_same_rows_mae"], "loc")
        otg_twin = twin_mae - loc_mae
        otg_row = row_mae - loc_mae
        otg_bal = bal_mae - loc_mae
        gain_row = twin_mae - row_mae
        gain_bal = twin_mae - bal_mae
        note(otg_twin, rec["otg_abs"], "otg")
        dts_keep = [pack["dt"][i] for i in keep_idx]
        directional.append({
            "group_id": rec["group_id"],
            "source_plant_id": src,
            "target_plant_id": tgt,
            "model_family": FAM_SPLINE,
            "support_rule": "CS4",
            "n_supported": int(len(dts_keep)),
            "n_supported_days": _days(dts_keep),
            "support_retention": rec["support_retention"],
            "twin_mae": twin_mae,
            "local_same_rows_mae": loc_mae,
            "pool_row_mae": row_mae,
            "pool_bal_mae": bal_mae,
            "otg_twin": otg_twin,
            "otg_pool_row": otg_row,
            "otg_pool_bal": otg_bal,
            "gain_row_vs_twin": gain_row,
            "gain_bal_vs_twin": gain_bal,
            "computable": True,
            "selected_row_hparams": next(r["selected_hyperparameters"] for r in train_rows if r["target_plant_id"] == tgt and r["variant"] == "row_pooled"),
            "selected_bal_hparams": next(r["selected_hyperparameters"] for r in train_rows if r["target_plant_id"] == tgt and r["variant"] == "plant_balanced"),
        })
        for i, dt in enumerate(dts_keep):
            dtau = s05_meta.get((src, tgt, dt), (None, None))
            test_rows.append({
                "group_id": rec["group_id"],
                "source_plant_id": src,
                "target_plant_id": tgt,
                "datetime_raw": dt,
                "y_true": float(y[i]),
                "y_pred_twin": float(twin[i]),
                "y_pred_local": float(loc[i]),
                "y_pred_pool_row": float(pr[i]),
                "y_pred_pool_bal": float(pb[i]),
                "abs_err_twin": abs(float(twin[i]) - float(y[i])),
                "abs_err_local": abs(float(loc[i]) - float(y[i])),
                "abs_err_pool_row": abs(float(pr[i]) - float(y[i])),
                "abs_err_pool_bal": abs(float(pb[i]) - float(y[i])),
                "kth_distance": dtau[0],
                "threshold": dtau[1],
                "supported": True,
                "support_rule": "CS4",
            })

    # row-level recon of direction MAEs
    by_dir: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in test_rows:
        by_dir[(r["source_plant_id"], r["target_plant_id"])].append(r)
    for d in directional:
        sub = by_dir[(d["source_plant_id"], d["target_plant_id"])]
        note(d["pool_row_mae"], float(np.mean([x["abs_err_pool_row"] for x in sub])), "rowmae")
        note(d["pool_bal_mae"], float(np.mean([x["abs_err_pool_bal"] for x in sub])), "balmae")
        note(d["twin_mae"], float(np.mean([x["abs_err_twin"] for x in sub])), "twinr")
        note(d["local_same_rows_mae"], float(np.mean([x["abs_err_local"] for x in sub])), "locr")
        note(d["n_supported"], len(sub), "nrow")

    for d in directional:
        d["otg_abs"] = d["otg_twin"]
        d["computable"] = True
    twin_rq1 = rq1_block(directional, "otg_twin")
    row_rq1 = rq1_block(directional, "otg_pool_row")
    bal_rq1 = rq1_block(directional, "otg_pool_bal")
    note(twin_rq1["plant_balanced"], PRIMARY_RQ1, "rq1")
    note(twin_rq1["plant_balanced"], twin_rq1["plant_balanced_iterate"], "twin_it")
    note(row_rq1["plant_balanced"], row_rq1["plant_balanced_iterate"], "row_it")
    note(bal_rq1["plant_balanced"], bal_rq1["plant_balanced_iterate"], "bal_it")
    note(row_rq1["plant_balanced"], REF_ROW_PB_OTG, "row_otg_ref")

    g_row_pb, _ = plant_balanced([{**r, "computable": True} for r in directional], "gain_row_vs_twin")
    g_bal_pb, _ = plant_balanced([{**r, "computable": True} for r in directional], "gain_bal_vs_twin")
    g_row_pb2, _ = plant_balanced_iterate([{**r, "computable": True} for r in directional], "gain_row_vs_twin")
    g_bal_pb2, _ = plant_balanced_iterate([{**r, "computable": True} for r in directional], "gain_bal_vs_twin")
    note(g_row_pb, g_row_pb2, "grow")
    note(g_bal_pb, g_bal_pb2, "gbal")
    pb_gain_row = float(twin_rq1["plant_balanced"] - row_rq1["plant_balanced"])
    pb_gain_bal = float(twin_rq1["plant_balanced"] - bal_rq1["plant_balanced"])
    note(pb_gain_row, g_row_pb, "pbgr")
    note(pb_gain_bal, g_bal_pb, "pbgb")
    note(pb_gain_row, REF_ROW_PB_GAIN, "row_gain_ref")

    gain_row_vals = [float(r["gain_row_vs_twin"]) for r in directional]
    gain_bal_vals = [float(r["gain_bal_vs_twin"]) for r in directional]
    gain_row_sum = gain_pack(gain_row_vals, g_row_pb)
    gain_bal_sum = gain_pack(gain_bal_vals, g_bal_pb)

    boot_row, lo_r, hi_r = bootstrap_pb_gain(directional, "gain_row_vs_twin")
    boot_bal, lo_b, hi_b = bootstrap_pb_gain(directional, "gain_bal_vs_twin")
    boot_rows = boot_row + boot_bal
    note(lo_r, REF_ROW_CI[0], "row_ci_lo")
    note(hi_r, REF_ROW_CI[1], "row_ci_hi")
    note(gain_row_sum["n_pos"], REF_ROW_NPOS, "row_npos")
    note(gain_row_sum["n_neg"], REF_ROW_NNEG, "row_nneg")

    def group_block(ids: tuple[str, ...] | str, label: str) -> dict[str, Any]:
        if isinstance(ids, str):
            ids = (ids,)
        sub = [r for r in directional if r["group_id"] in ids]
        if not sub:
            return {"label": label, "n_directional": 0, "estimable": False}
        tw = rq1_block(sub, "otg_twin")
        rw = rq1_block(sub, "otg_pool_row")
        bw = rq1_block(sub, "otg_pool_bal")
        gr, _ = plant_balanced(sub, "gain_row_vs_twin")
        gb, _ = plant_balanced(sub, "gain_bal_vs_twin")
        return {
            "label": label,
            "n_directional": len(sub),
            "n_targets": tw["n_targets"],
            "estimable": True,
            "twin": tw,
            "row_pooled": rw,
            "plant_balanced": bw,
            "gain_row": gain_pack([float(r["gain_row_vs_twin"]) for r in sub], gr),
            "gain_bal": gain_pack([float(r["gain_bal_vs_twin"]) for r in sub], gb),
            "caution_small_n": label != "G2",
        }

    groups = {
        "G1": group_block(G1, "G1"),
        "G2": group_block(G2, "G2"),
        "G4": group_block(G4, "G4"),
        "nonG2": group_block((G1, G4), "nonG2"),
    }

    target_sum = []
    common_rows = []
    by_tgt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in directional:
        by_tgt[r["target_plant_id"]].append(r)
    dts_by_pair = {(r["source_plant_id"], r["target_plant_id"]): [x["datetime_raw"] for x in by_dir[(r["source_plant_id"], r["target_plant_id"])]] for r in directional}
    for tgt in plants:
        sub = by_tgt.get(tgt, [])
        n = len(sub)
        twin_m = [float(r["twin_mae"]) for r in sub]
        row_m = [float(r["pool_row_mae"]) for r in sub]
        bal_m = [float(r["pool_bal_mae"]) for r in sub]
        gr = [float(r["gain_row_vs_twin"]) for r in sub]
        gb = [float(r["gain_bal_vs_twin"]) for r in sub]
        rec = {
            "target_plant_id": tgt,
            "group_id": members.get(tgt),
            "n_computable_twin_directions": n,
            "mean_twin_mae": float(np.mean(twin_m)) if twin_m else None,
            "median_twin_mae": float(np.median(twin_m)) if twin_m else None,
            "mean_pool_row_mae": float(np.mean(row_m)) if row_m else None,
            "mean_pool_bal_mae": float(np.mean(bal_m)) if bal_m else None,
            "equal_source_mean_gain_row": float(np.mean(gr)) if gr else None,
            "equal_source_mean_gain_bal": float(np.mean(gb)) if gb else None,
            "n_pool_row_beats": int(sum(v > ZERO_GAIN_TOL for v in gr)),
            "n_pool_row_loses": int(sum(v < -ZERO_GAIN_TOL for v in gr)),
            "n_pool_bal_beats": int(sum(v > ZERO_GAIN_TOL for v in gb)),
            "n_pool_bal_loses": int(sum(v < -ZERO_GAIN_TOL for v in gb)),
            "pool_row_beats_every": bool(n and all(v > ZERO_GAIN_TOL for v in gr)),
            "pool_row_loses_every": bool(n and all(v < -ZERO_GAIN_TOL for v in gr)),
            "pool_bal_beats_every": bool(n and all(v > ZERO_GAIN_TOL for v in gb)),
            "pool_bal_loses_every": bool(n and all(v < -ZERO_GAIN_TOL for v in gb)),
        }
        note(rec["n_pool_row_beats"] + rec["n_pool_row_loses"] + int(sum(abs(v) <= ZERO_GAIN_TOL for v in gr)), n, "tgtc")
        target_sum.append(rec)
        sources = [r["source_plant_id"] for r in sub]
        if len(sources) >= 2:
            sets = [set(dts_by_pair[(s, tgt)]) for s in sources]
            inter = set.intersection(*sets) if sets else set()
            n_int = len(inter)
            if n_int > 0:
                y_map_dt = {}
                twin_err = {s: [] for s in sources}
                row_err, bal_err, ys = [], [], []
                for s in sources:
                    for x in by_dir[(s, tgt)]:
                        if x["datetime_raw"] in inter:
                            twin_err[s].append(x["abs_err_twin"])
                            y_map_dt[x["datetime_raw"]] = x
                # unique dts for pooled (same across sources)
                for dt in sorted(inter):
                    x = y_map_dt[dt]
                    row_err.append(x["abs_err_pool_row"])
                    bal_err.append(x["abs_err_pool_bal"])
                    ys.append(x["y_true"])
                twin_maes = {s: float(np.mean(v)) for s, v in twin_err.items() if len(v) == n_int}
                if twin_maes:
                    best_s = min(twin_maes, key=lambda s: (twin_maes[s], s))
                    common_rows.append({
                        "target_plant_id": tgt,
                        "n_sources": len(sources),
                        "n_common_rows": n_int,
                        "best_twin_source": best_s,
                        "best_twin_mae": twin_maes[best_s],
                        "pool_row_mae": float(np.mean(row_err)),
                        "pool_bal_mae": float(np.mean(bal_err)),
                        "gain_row_vs_best_twin": twin_maes[best_s] - float(np.mean(row_err)),
                        "gain_bal_vs_best_twin": twin_maes[best_s] - float(np.mean(bal_err)),
                        "headline": False,
                        "diagnostic": "optional_common_row_best_twin",
                    })

    bal_train = [r for r in train_rows if r["variant"] == "plant_balanced"]
    weighting_ok = (
        len(bal_train) == 14
        and all(int(r["n_donor_plants"]) == 13 for r in bal_train)
        and all(int(r["held_out_target_train_rows"]) == 0 and int(r["held_out_target_val_rows"]) == 0 for r in bal_train)
        and all(r.get("balanced_weights_equal") is True for r in bal_train)
        and all(r.get("scaler_sample_weight") is True and r.get("ridge_sample_weight") is True for r in bal_train)
        and all(r.get("same_weight_vector_scaler_ridge") is True for r in bal_train)
        and all(r.get("spline_transformer_sample_weight") is False for r in bal_train)
    )
    leakage["weighting"]["all_targets_validated"] = weighting_ok

    freq_row: dict[str, int] = Counter()
    freq_bal: dict[str, int] = Counter()
    n_diff = 0
    selected = {}
    for r in train_rows:
        selected.setdefault(r["target_plant_id"], {})[r["variant"]] = r["selected_hyperparameters"]
        if r["variant"] == "row_pooled":
            freq_row[r["selected_hyperparameters"]] += 1
        else:
            freq_bal[r["selected_hyperparameters"]] += 1
    for tgt in plants:
        if selected[tgt]["row_pooled"] != selected[tgt]["plant_balanced"]:
            n_diff += 1

    finite_d = [d for d in deltas if np.isfinite(d)]
    row_pooled_unchanged = (
        abs(float(row_rq1["plant_balanced"]) - REF_ROW_PB_OTG) <= TOL
        and abs(float(pb_gain_row) - REF_ROW_PB_GAIN) <= TOL
        and abs(float(lo_r) - REF_ROW_CI[0]) <= TOL
        and abs(float(hi_r) - REF_ROW_CI[1]) <= TOL
        and int(gain_row_sum["n_pos"]) == REF_ROW_NPOS
        and int(gain_row_sum["n_neg"]) == REF_ROW_NNEG
    )
    recon_ok = (
        bool(finite_d)
        and max(finite_d) <= TOL
        and leakage["n_targets_fit"] == 14
        and not leakage["target_in_train"]
        and not leakage["target_in_val"]
        and not leakage["donors_not_13"]
        and len(directional) == 85
        and weighting_ok
        and row_pooled_unchanged
    )
    summary = {
        "status": STATUS if recon_ok else "STOP_RECONCILIATION_FAILURE",
        "n_plants": len(plants),
        "plants": plants,
        "n_candidates": len(candidates),
        "candidates": [{"candidate_id": c["candidate_id"], "params": c["params"], "source_plants": c["source_plants"]} for c in candidates],
        "n_models": len(model_paths),
        "leakage": leakage,
        "n_directional": len(directional),
        "n_test_rows": len(test_rows),
        "twin_rq1": twin_rq1,
        "row_pooled_rq1": row_rq1,
        "plant_balanced_rq1": bal_rq1,
        "accepted_primary_rq1": PRIMARY_RQ1,
        "pb_gain_row": pb_gain_row,
        "pb_gain_bal": pb_gain_bal,
        "gain_row": gain_row_sum,
        "gain_bal": gain_bal_sum,
        "bootstrap": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "algorithm": "resample_target_ids_with_replacement; equal-target mean GAIN with multiplicity",
            "row_pooled": {"ci_lo": lo_r, "ci_hi": hi_r, "n_replicates": len(boot_row)},
            "plant_balanced": {"ci_lo": lo_b, "ci_hi": hi_b, "n_replicates": len(boot_bal)},
        },
        "groups": groups,
        "selection": {
            "tie_break": "lexicographic json hyperparameters",
            "score": "equal mean of donor-plant validation MAE",
            "final_fit": "donor TRAIN only",
            "frequency_row_pooled": dict(freq_row),
            "frequency_plant_balanced": dict(freq_bal),
            "n_targets_differing_hparams": n_diff,
            "selected_by_target": selected,
        },
        "common_row_best_twin": common_rows,
        "row_pooled_unchanged": row_pooled_unchanged,
        "weighting_ok": weighting_ok,
        "weighting_implementation": {
            "w_ij": "1/n_j",
            "spline_transformer": "uniform knots; no sample_weight in this implementation",
            "standard_scaler": "scaler__sample_weight=w",
            "ridge": "ridge__sample_weight=w",
            "same_vector": True,
        },
        "rq2_policy": RQ2_POLICY,
        "rq3_status": RQ3_STATUS,
        "reconciliation_ok": recon_ok,
        "reconciliation_max_abs_discrepancy": float(max(finite_d)) if finite_d else None,
        "no_p9": True,
        "features": list(FEATURES),
        "support_reuse": "exact S06/S07 CS4 primary TEST identities; no pooled support mask",
    }
    summary["interpretation"] = interpret(summary)
    return {
        "summary": _jsonable(summary),
        "candidates": cand_csv,
        "selection": sel_rows,
        "training": train_rows,
        "directional": directional,
        "target_summary": target_sum,
        "test_rows": test_rows,
        "bootstrap": boot_rows,
        "common_row": common_rows,
        "model_paths": model_paths,
        "plants": plants,
        "hparams_s04": hparams,
        "cand_plant_rows": cand_plant_rows,
    }


def render_report(s: dict[str, Any]) -> str:
    tw, rw, bw = s["twin_rq1"], s["row_pooled_rq1"], s["plant_balanced_rq1"]
    br, bb = s["bootstrap"]["row_pooled"], s["bootstrap"]["plant_balanced"]
    lines = [
        "# S12 P8 — Leave-target-out pooled fleet baseline",
        "",
        "Secondary benchmark only. Accepted primary RQ1/RQ2/RQ3, S11, source-selection, and S12 P0–P7 are unchanged. Headline comparison uses the 85 accepted primary CS4 SplineRidge directions and the exact S06/S07 TEST support rows. No pooled support mask is defined. RQ2 is not redefined. RQ3 is out of scope.",
        "",
        f"Status: `{s['status']}`.",
        "",
        s["interpretation"]["text"],
        "",
        "## Setup",
        "",
        f"- 14 primary plants; 6 unique S04 SplineRidge full selected tuples; leave-target-out donors = 13.",
        f"- Features: {', '.join(s['features'])} (no plant/state/group IDs).",
        f"- Selection: equal mean of donor-plant VALIDATION MAE; lexicographic hyperparameter tie-break; final fit on donor TRAIN only.",
        f"- P8-B weights `w_ij = 1/n_j`. SplineTransformer uses accepted uniform knots (no sample-weight fit). The same weight vector is passed as `scaler__sample_weight` and `ridge__sample_weight`, so each donor has equal total row weight before preprocessing and Ridge.",
        f"- Row-pooled algorithm unchanged; frozen recon vs first P8 run: `{s.get('row_pooled_unchanged')}`.",
        "",
        "## Fleet RQ1-style OTG (equal-target)",
        "",
        f"- Twin PB OTG = {tw['plant_balanced']} (accepted {s['accepted_primary_rq1']}; n_dir={tw['n_directional']}, n_targets={tw['n_targets']}).",
        f"- Row-pooled PB OTG = {rw['plant_balanced']}.",
        f"- Plant-balanced pooled PB OTG = {bw['plant_balanced']}.",
        f"- PB GAIN row vs twin = {s['pb_gain_row']}.",
        f"- PB GAIN balanced vs twin = {s['pb_gain_bal']}.",
        "",
        "## Paired GAIN vs twin (85 directions)",
        "",
        f"- Row-pooled: mean={s['gain_row']['mean']}, median={s['gain_row']['median']}, P25={s['gain_row']['p25']}, P75={s['gain_row']['p75']}, P95={s['gain_row']['p95']}, min={s['gain_row']['min']}, max={s['gain_row']['max']}; n>0={s['gain_row']['n_pos']}, n<0={s['gain_row']['n_neg']}, n=0={s['gain_row']['n_zero']}; PB={s['gain_row']['plant_balanced_mean']}.",
        f"- Plant-balanced: mean={s['gain_bal']['mean']}, median={s['gain_bal']['median']}, P25={s['gain_bal']['p25']}, P75={s['gain_bal']['p75']}, P95={s['gain_bal']['p95']}, min={s['gain_bal']['min']}, max={s['gain_bal']['max']}; n>0={s['gain_bal']['n_pos']}, n<0={s['gain_bal']['n_neg']}, n=0={s['gain_bal']['n_zero']}; PB={s['gain_bal']['plant_balanced_mean']}.",
        "",
        "## Target-cluster bootstrap (5000, seed 20260921)",
        "",
        f"- Row-pooled PB GAIN 95% percentile CI = [{br['ci_lo']}, {br['ci_hi']}].",
        f"- Plant-balanced PB GAIN 95% percentile CI = [{bb['ci_lo']}, {bb['ci_hi']}].",
        "",
        "## Groups (descriptive)",
        "",
        f"- G2: n_dir={s['groups']['G2']['n_directional']}, PB GAIN row={s['groups']['G2']['gain_row']['plant_balanced_mean']}, bal={s['groups']['G2']['gain_bal']['plant_balanced_mean']}.",
        f"- non-G2 (G1+G4, small): n_dir={s['groups']['nonG2']['n_directional']}, PB GAIN row={s['groups']['nonG2']['gain_row']['plant_balanced_mean']}, bal={s['groups']['nonG2']['gain_bal']['plant_balanced_mean']}.",
        f"- G1: n_dir={s['groups']['G1']['n_directional']}; G4: n_dir={s['groups']['G4']['n_directional']}.",
        "",
        "## Selection",
        "",
        f"- Candidate set size = {s['n_candidates']}.",
        f"- Targets with different row vs balanced selected tuples = {s['selection']['n_targets_differing_hparams']}.",
        f"- Frequency row-pooled: {json.dumps(s['selection']['frequency_row_pooled'], sort_keys=True)}.",
        f"- Frequency plant-balanced: {json.dumps(s['selection']['frequency_plant_balanced'], sort_keys=True)}.",
        "",
        f"Optional common-row best-twin diagnostic: n_targets={len(s['common_row_best_twin'])} (not headline).",
        "",
        f"RQ2 policy: `{s['rq2_policy']}`. RQ3: `{s['rq3_status']}`.",
        "",
        f"Independent reconciliation max |Δ| = {s['reconciliation_max_abs_discrepancy']}. Models persisted: {s['n_models']}.",
        "",
    ]
    return "\n".join(lines)
