"""S12 P3 / R9: adaptive support scaling (IQR -> 1.4826*MAD -> omit feature). Sensitivity only."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors

from config.protocol import CS4, FAM_SPLINE, PRIMARY_GROUP_IDS
from src.lib.models import _spline
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
from src.lib.split_support import (
    NN_ALGORITHM,
    iqr_is_valid,
    loo_kth_distances,
    q_linear,
    source_median_iqr,
    standardize,
)

MAD_C = 1.4826
STATUS = "S12_P3_R9_ADAPTIVE_SCALING_MATERIALIZED"
RQ3_DEFER = "R9_RQ3_DEFERRED_NOT_REQUIRED_FOR_SCALER_SENSITIVITY"
TOL = 1e-12


def _flag(val: Any) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def r9_feature_scales(X: np.ndarray) -> dict[str, Any]:
    kept, medians, scales, sources = [], [], [], []
    detail = []
    for m, name in enumerate(FEATURES):
        col = X[:, m]
        med, iqr = source_median_iqr(col)
        rec = {"feature": name, "median": med, "iqr": iqr, "mad": None, "scale": None, "scale_source": None}
        if iqr_is_valid(iqr):
            rec["scale"] = iqr
            rec["scale_source"] = "IQR"
            kept.append(m)
            medians.append(med)
            scales.append(iqr)
            sources.append("IQR")
        else:
            mad = float(np.median(np.abs(col - med)))
            rec["mad"] = mad
            scale = MAD_C * mad
            if iqr_is_valid(scale):
                rec["scale"] = scale
                rec["scale_source"] = "MAD"
                kept.append(m)
                medians.append(med)
                scales.append(scale)
                sources.append("MAD")
            else:
                rec["scale_source"] = "OMITTED_DEGENERATE"
        detail.append(rec)
    return {"kept": kept, "medians": medians, "scales": scales, "sources": sources, "detail": detail}


def build_r9_engine(pid: str, train_dts: list[str], feat_by: dict) -> dict[str, Any]:
    rows = []
    for dt in train_dts:
        vec = feat_by.get((pid, dt))
        if vec is None or not np.all(np.isfinite(vec)):
            return {"computable": False, "reason": "missing_or_nonfinite_train_features", "n_source": len(train_dts)}
        rows.append(vec)
    if len(rows) <= int(CS4["k"]):
        return {"computable": False, "reason": "insufficient_train_rows_for_k5", "n_source": len(rows)}
    X = np.vstack(rows)
    sc = r9_feature_scales(X)
    if not sc["kept"]:
        return {"computable": False, "reason": "all_support_features_degenerate", "n_source": len(rows), "scaler_detail": sc["detail"]}
    z = np.column_stack([standardize(X[:, j], sc["medians"][i], sc["scales"][i]) for i, j in enumerate(sc["kept"])])
    dist = loo_kth_distances(z, int(CS4["k"]))
    tau = q_linear(dist, float(CS4["q"]))
    if not np.isfinite(tau):
        return {"computable": False, "reason": "nonfinite_threshold", "n_source": len(rows), "scaler_detail": sc["detail"]}
    nn = NearestNeighbors(n_neighbors=int(CS4["k"]), algorithm=NN_ALGORITHM, metric="euclidean")
    nn.fit(z)
    return {
        "computable": True,
        "reason": None,
        "kept": sc["kept"],
        "medians": sc["medians"],
        "scales": sc["scales"],
        "sources": sc["sources"],
        "scaler_detail": sc["detail"],
        "nn": nn,
        "tau": float(tau),
        "n_source": len(rows),
        "n_support_features": len(sc["kept"]),
    }


def score_r9(engine: dict[str, Any], tgt: str, test_dts: list[str], feat_by: dict) -> list[tuple[str, float, float, bool]]:
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
    z = np.column_stack([standardize(X[:, j], engine["medians"][i], engine["scales"][i]) for i, j in enumerate(engine["kept"])])
    dist, _ = engine["nn"].kneighbors(z, n_neighbors=k, return_distance=True)
    d5 = dist[:, k - 1]
    tau = float(engine["tau"])
    return [(dt, float(d5[i]), tau, bool(d5[i] <= tau)) for i, dt in enumerate(keep)]


def load_or_refit_models(root: Path, plants: list[str], splits, feat_by, y_map, hparams) -> dict[str, Any]:
    out = {}
    for pid in plants:
        path = root / "artifacts/s04/models" / pid / FAM_SPLINE / "full.joblib"
        action = "reused"
        est = None
        if path.is_file():
            try:
                est = joblib.load(path)
                probe = next((feat_by[(pid, dt)] for dt in splits[pid]["train"] if (pid, dt) in feat_by), None)
                if probe is None:
                    raise RuntimeError("no train vector")
                est.predict(probe.reshape(1, -1))
            except Exception:
                est = None
                action = "refit_unusable_artifact"
        else:
            action = "refit_missing_artifact"
        if est is None:
            est, _used, err = fit_frozen(pid, splits[pid]["train"], feat_by, y_map, hparams[pid])
            action = action if action.startswith("refit") else "refit"
            if err:
                out[pid] = {"est": None, "action": action, "error": err}
                continue
            dest = root / "artifacts/s12/p3/models" / pid / "SplineRidge_full.joblib"
            dest.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(est, dest)
        out[pid] = {"est": est, "action": "reused" if action == "reused" else action, "error": None, "path": str(path)}
    return out


def predict_on(est, pid: str, dts: list[str], feat_by, y_map):
    rows = []
    for dt in dts:
        vec = feat_by.get((pid, dt))
        yv = y_map.get((pid, dt))
        if vec is None or yv is None:
            continue
        yp = float(est.predict(vec.reshape(1, -1))[0])
        rows.append((dt, yv, yp))
    return rows


def plant_balanced_asym(pairs: list[dict[str, Any]]) -> tuple[float | None, int]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in pairs:
        v = float(r["asymmetry_abs"])
        by[str(r["plant_a"])].append(v)
        by[str(r["plant_b"])].append(v)
    if not by:
        return None, 0
    means = [sum(v) / len(v) for v in by.values()]
    return float(sum(means) / len(means)), len(by)


def load_primary_cs4_sets(root: Path) -> dict[tuple[str, str], set[str]]:
    path = root / "artifacts/s05/S05_SUPPORT_MASKS.parquet"
    table = pq.read_table(path, columns=["source_plant_id", "target_plant_id", "rule_id", "datetime_raw", "supported", "computable"])
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


def run_science(root: Path, sot: dict[str, Any]) -> dict[str, Any]:
    plants, members = load_scope(root)
    hparams = load_hparams(root, plants)
    universe = load_directional_universe(root)
    feat_by, y_map, splits = load_tables(root, plants)
    models = load_or_refit_models(root, plants, splits, feat_by, y_map, hparams)
    primary_sets = load_primary_cs4_sets(root)
    primary_ok = {(r["source_plant_id"], r["target_plant_id"]) for r in universe if _flag(r.get("computable"))}
    primary_otg = {(r["source_plant_id"], r["target_plant_id"]): float(r["otg_abs"]) for r in universe if _flag(r.get("computable"))}

    scaler_rows = []
    engines = {}
    for pid in plants:
        eng = build_r9_engine(pid, splits[pid]["train"], feat_by)
        engines[pid] = eng
        det = eng.get("scaler_detail") or []
        if not det and splits[pid]["train"]:
            Xrows = [feat_by[(pid, dt)] for dt in splits[pid]["train"] if (pid, dt) in feat_by]
            if Xrows:
                det = r9_feature_scales(np.vstack(Xrows))["detail"]
        for rec in det:
            scaler_rows.append({"source_plant_id": pid, "n_source_train": eng.get("n_source"), "tau": eng.get("tau"), "n_support_features": eng.get("n_support_features"), "engine_computable": eng.get("computable"), **rec})
        if not det:
            scaler_rows.append({"source_plant_id": pid, "n_source_train": eng.get("n_source"), "tau": None, "n_support_features": 0, "engine_computable": False, "feature": None, "scale_source": None, "reason_engine": eng.get("reason")})

    directional, eval_rows, mask_chunks = [], [], {k: [] for k in ["group_id", "source_plant_id", "target_plant_id", "datetime_raw", "kth_distance", "threshold", "supported", "computable", "reason"]}
    for row in universe:
        src, tgt, gid = row["source_plant_id"], row["target_plant_id"], row["group_id"]
        rec = {"group_id": gid, "source_plant_id": src, "target_plant_id": tgt, "model_family": FAM_SPLINE, "support_rule": "CS4_R9", "computable": False}
        eng = engines.get(src) or {}
        ms, mt = models.get(src, {}), models.get(tgt, {})
        if not eng.get("computable"):
            rec["reason"] = eng.get("reason")
            directional.append(rec)
            continue
        if ms.get("est") is None or mt.get("est") is None:
            rec["reason"] = ms.get("error") or mt.get("error") or "model_unfit"
            directional.append(rec)
            continue
        test = splits[tgt]["test"]
        scored = score_r9(eng, tgt, test, feat_by)
        supported = [dt for dt, _d, _t, ok in scored if ok]
        rec["n_eligible_target_test"] = len(test)
        rec["n_supported"] = len(supported)
        rec["support_retention"] = (len(supported) / len(test)) if test else None
        rec["n_supported_days"] = _days(supported) if supported else 0
        rec["threshold"] = eng.get("tau")
        rec["n_support_features"] = eng.get("n_support_features")
        for dt, d, tau, ok in scored:
            mask_chunks["group_id"].append(gid)
            mask_chunks["source_plant_id"].append(src)
            mask_chunks["target_plant_id"].append(tgt)
            mask_chunks["datetime_raw"].append(dt)
            mask_chunks["kth_distance"].append(d)
            mask_chunks["threshold"].append(tau)
            mask_chunks["supported"].append(ok)
            mask_chunks["computable"].append(True)
            mask_chunks["reason"].append(None)
        if not supported:
            rec["reason"] = "empty_cs4_support"
            directional.append(rec)
            continue
        y_true, y_tr, y_loc = [], [], []
        for dt in supported:
            vec = feat_by.get((tgt, dt))
            yv = y_map.get((tgt, dt))
            if vec is None or yv is None:
                continue
            ya = float(ms["est"].predict(vec.reshape(1, -1))[0])
            yb = float(mt["est"].predict(vec.reshape(1, -1))[0])
            y_true.append(yv)
            y_tr.append(ya)
            y_loc.append(yb)
            eval_rows.append({"source_plant_id": src, "target_plant_id": tgt, "datetime_raw": dt, "y_true": yv, "y_pred_transfer": ya, "y_pred_local": yb, "abs_error_transfer": abs(ya - yv), "abs_error_local": abs(yb - yv)})
        if not y_true:
            rec["reason"] = "missing_predictions_on_support"
            directional.append(rec)
            continue
        tmae = float(np.mean(np.abs(np.array(y_tr) - np.array(y_true))))
        lmae = float(np.mean(np.abs(np.array(y_loc) - np.array(y_true))))
        otg = tmae - lmae
        rec.update({"computable": True, "reason": None, "transfer_mae": tmae, "local_same_rows_mae": lmae, "otg_abs": otg, "otg_rel": (otg / lmae) if lmae != 0 else None, "otg_rel_defined": lmae != 0})
        directional.append(rec)

    r9_ok = [r for r in directional if r.get("computable")]
    r9_ids = {(r["source_plant_id"], r["target_plant_id"]) for r in r9_ok}
    recovered = sorted(r9_ids - primary_ok)
    lost = sorted(primary_ok - r9_ids)
    unchanged = sorted(r9_ids & primary_ok)
    ps048 = [r for r in directional if r["source_plant_id"] == "PS_048"]

    pb, nt = plant_balanced(r9_ok)
    pb2, _ = plant_balanced_iterate(r9_ok)
    overlap_rows = [r for r in r9_ok if (r["source_plant_id"], r["target_plant_id"]) in primary_ok]
    pb_ov, nt_ov = plant_balanced(overlap_rows)
    tables = (sot.get("results") or {}).get("tables") or {}
    prim_rq1, prim_rq2 = tables.get("rq1"), tables.get("rq2")
    abs_vals = [float(r["otg_abs"]) for r in r9_ok]
    rel_vals = [float(r["otg_rel"]) for r in r9_ok if r.get("otg_rel_defined")]
    sr_vals = [float(r["support_retention"]) for r in r9_ok if r.get("support_retention") is not None]
    by_g = defaultdict(list)
    for r in r9_ok:
        by_g[r["group_id"]].append(r)
    group_rq1 = {g: plant_balanced(rows)[0] for g, rows in sorted(by_g.items())}

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
                signed = (float(ab["otg_abs"]) - float(ba["otg_abs"])) if both else None
                asym_rows.append({"group_id": gid, "plant_a": a, "plant_b": b, "a_to_b_computable": bool(ab and ab.get("computable")), "b_to_a_computable": bool(ba and ba.get("computable")), "otg_a_to_b": float(ab["otg_abs"]) if ab and ab.get("computable") else None, "otg_b_to_a": float(ba["otg_abs"]) if ba and ba.get("computable") else None, "asymmetry_signed": signed, "asymmetry_abs": abs(signed) if signed is not None else None, "asymmetry_computable": both})
    r9_pairs = [r for r in asym_rows if r["asymmetry_computable"]]
    primary_pairs = set()
    with (root / "artifacts/s07/S07_ASYMMETRY.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if _flag(row.get("asymmetry_computable")) and row.get("group_id") in PRIMARY_GROUP_IDS:
                primary_pairs.add(tuple(sorted((row["plant_a"], row["plant_b"]))))
    r9_pair_ids = {tuple(sorted((r["plant_a"], r["plant_b"]))) for r in r9_pairs}
    overlap_pairs = [r for r in r9_pairs if tuple(sorted((r["plant_a"], r["plant_b"]))) in primary_pairs]
    rq2, npl = plant_balanced_asym(r9_pairs)
    rq2_ov, npl_ov = plant_balanced_asym(overlap_pairs)

    cmp_rows = []
    jacs, d_sr, d_otg = [], [], []
    n_ident = n_super = n_sub = n_part = 0
    for src, tgt in sorted(primary_ok):
        prim = primary_sets.get((src, tgt), set())
        r9s = {dt for dt, _d, _t, ok in score_r9(engines[src], tgt, splits[tgt]["test"], feat_by) if ok}
        inter, uni = len(prim & r9s), len(prim | r9s)
        jac = (inter / uni) if uni else (1.0 if not prim and not r9s else 0.0)
        if prim == r9s:
            rel = "identical"
            n_ident += 1
        elif prim < r9s:
            rel = "r9_superset"
            n_super += 1
        elif r9s < prim:
            rel = "r9_subset"
            n_sub += 1
        else:
            rel = "partial"
            n_part += 1
        r9rec = lookup.get((src, tgt))
        sr_p = next((float(r["support_retention"]) for r in universe if r["source_plant_id"] == src and r["target_plant_id"] == tgt), None)
        sr_r = r9rec.get("support_retention") if r9rec else None
        otg_r = r9rec.get("otg_abs") if r9rec and r9rec.get("computable") else None
        otg_p = primary_otg.get((src, tgt))
        cmp_rows.append({"source_plant_id": src, "target_plant_id": tgt, "relation": rel, "jaccard": jac, "n_primary_supported": len(prim), "n_r9_supported": len(r9s), "delta_support_retention": (None if sr_p is None or sr_r is None else sr_r - sr_p), "delta_otg_abs": (None if otg_p is None or otg_r is None else otg_r - otg_p)})
        jacs.append(jac)
        if sr_p is not None and sr_r is not None:
            d_sr.append(abs(sr_r - sr_p))
        if otg_p is not None and otg_r is not None:
            d_otg.append(abs(otg_r - otg_p))

    recon_max = 0.0
    if pb is not None and pb2 is not None:
        recon_max = max(recon_max, abs(pb - pb2))
    by_tr = defaultdict(lambda: {"t": [], "l": []})
    for r in eval_rows:
        by_tr[(r["source_plant_id"], r["target_plant_id"])]["t"].append(r["abs_error_transfer"])
        by_tr[(r["source_plant_id"], r["target_plant_id"])]["l"].append(r["abs_error_local"])
    for r in r9_ok:
        recb = by_tr.get((r["source_plant_id"], r["target_plant_id"]))
        if not recb or not recb["t"]:
            recon_max = max(recon_max, 1.0)
            continue
        otg = float(np.mean(recb["t"]) - np.mean(recb["l"]))
        recon_max = max(recon_max, abs(otg - float(r["otg_abs"])))

    ps048_wind = next((r for r in scaler_rows if r.get("source_plant_id") == "PS_048" and r.get("feature") == "wind_speed_ms"), None)
    recovered_n = sum(1 for r in ps048 if r.get("computable"))
    interpretation = []
    if recovered_n == 9:
        interpretation.append("R9 recovers all 9 PS_048 source directions.")
    elif recovered_n == 0:
        interpretation.append("R9 does not recover PS_048; MAD/omission does not resolve source-history degeneracy.")
    else:
        interpretation.append(f"R9 recovers {recovered_n}/9 PS_048 source directions.")
    if pb is not None and prim_rq1 is not None:
        interpretation.append(f"RQ1 plant-balanced primary {prim_rq1} vs R9 {pb} (difference {pb - float(prim_rq1)}). R9 remains a sensitivity arm, not a new primary protocol.")
    if jacs and float(np.median(jacs)) < 0.99 and pb is not None and prim_rq1 is not None and abs(pb - float(prim_rq1)) < 0.002:
        interpretation.append("Support masks can change while headline RQ1 stays close: distinguish row-level support sensitivity from aggregate robustness.")

    summary = {
        "status": STATUS,
        "r9_rule": "IQR if valid else 1.4826*MAD if valid else omit feature from support distance only; all degenerate -> all_support_features_degenerate",
        "mad_constant": MAD_C,
        "n_planned": 94,
        "n_primary_computable": 85,
        "n_r9_computable": len(r9_ok),
        "n_recovered": len(recovered),
        "n_lost": len(lost),
        "n_unchanged_identities": len(unchanged),
        "recovered_identities": [{"source_plant_id": a, "target_plant_id": b} for a, b in recovered],
        "lost_identities": [{"source_plant_id": a, "target_plant_id": b} for a, b in lost],
        "ps048_source_directions": [{"target_plant_id": r["target_plant_id"], "computable": r.get("computable"), "reason": r.get("reason")} for r in ps048],
        "ps048_n_recovered": recovered_n,
        "ps048_wind_speed_ms": ps048_wind,
        "model_actions": {pid: models[pid]["action"] for pid in plants},
        "rq1": {
            "n": len(r9_ok),
            "directional_mean": float(np.mean(abs_vals)) if abs_vals else None,
            "median": float(np.median(abs_vals)) if abs_vals else None,
            "rel_median": float(np.median(rel_vals)) if rel_vals else None,
            "rel_p75": float(np.quantile(rel_vals, 0.75, method="linear")) if rel_vals else None,
            "plant_balanced": pb,
            "n_targets": nt,
            "group_plant_balanced": group_rq1,
            "support_retention_median": float(np.median(sr_vals)) if sr_vals else None,
            "primary": prim_rq1,
            "difference_vs_primary": (pb - float(prim_rq1)) if pb is not None and prim_rq1 is not None else None,
            "overlap_n": len(overlap_rows),
            "overlap_plant_balanced": pb_ov,
            "overlap_n_targets": nt_ov,
            "overlap_difference_vs_primary": (pb_ov - float(prim_rq1)) if pb_ov is not None and prim_rq1 is not None else None,
        },
        "rq2": {
            "n_unordered_primary_pairs": len(asym_rows),
            "n_bidirectional_primary": len(primary_pairs),
            "n_bidirectional_r9": len(r9_pairs),
            "n_newly_recovered_pairs": len(r9_pair_ids - primary_pairs),
            "plant_balanced_abs_asymmetry": rq2,
            "n_plants": npl,
            "median_abs_asymmetry": float(np.median([r["asymmetry_abs"] for r in r9_pairs])) if r9_pairs else None,
            "primary": prim_rq2,
            "difference_vs_primary": (rq2 - float(prim_rq2)) if rq2 is not None and prim_rq2 is not None else None,
            "overlap_n_pairs": len(overlap_pairs),
            "overlap_plant_balanced": rq2_ov,
            "overlap_difference_vs_primary": (rq2_ov - float(prim_rq2)) if rq2_ov is not None and prim_rq2 is not None else None,
        },
        "rq3_status": RQ3_DEFER,
        "rq3_reason": "Canonical RQ3 needs R9 support on both twin and matched-control source histories plus pairwise TEST intersection. Control models/scoring are a separate pipeline; P3 does not rebuild it or invent a twin-only proxy.",
        "support_change": {
            "n_overlap_directions": len(primary_ok),
            "identical": n_ident,
            "r9_superset": n_super,
            "r9_subset": n_sub,
            "partial": n_part,
            "median_jaccard": float(np.median(jacs)) if jacs else None,
            "min_jaccard": float(np.min(jacs)) if jacs else None,
            "median_abs_delta_retention": float(np.median(d_sr)) if d_sr else None,
            "median_abs_delta_otg": float(np.median(d_otg)) if d_otg else None,
        },
        "interpretation": " ".join(interpretation),
        "r9_is_sensitivity_only": True,
        "reconciliation_max_abs_discrepancy": recon_max,
        "reconciliation_ok": recon_max <= TOL,
        "noncomputable": [{"source_plant_id": r["source_plant_id"], "target_plant_id": r["target_plant_id"], "reason": r.get("reason")} for r in directional if not r.get("computable")],
    }
    return {
        "plants": plants,
        "scaler_rows": scaler_rows,
        "directional": directional,
        "asym_rows": asym_rows,
        "cmp_rows": cmp_rows,
        "eval_rows": eval_rows,
        "mask_chunks": mask_chunks,
        "summary": summary,
        "models": models,
    }


def write_masks(path: Path, chunks: dict[str, list]) -> int:
    n = len(chunks["datetime_raw"])
    table = pa.table(
        {
            "group_id": pa.array(chunks["group_id"], pa.string()),
            "source_plant_id": pa.array(chunks["source_plant_id"], pa.string()),
            "target_plant_id": pa.array(chunks["target_plant_id"], pa.string()),
            "datetime_raw": pa.array(chunks["datetime_raw"], pa.string()),
            "kth_distance": pa.array(chunks["kth_distance"], pa.float64()),
            "threshold": pa.array(chunks["threshold"], pa.float64()),
            "supported": pa.array(chunks["supported"], pa.bool_()),
            "computable": pa.array(chunks["computable"], pa.bool_()),
            "reason": pa.array(chunks["reason"], pa.string()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return n


def write_eval(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["source_plant_id", "target_plant_id", "datetime_raw", "y_true", "y_pred_transfer", "y_pred_local", "abs_error_transfer", "abs_error_local"]
    arrays = {}
    for f in fields:
        vals = [r.get(f) for r in rows]
        if f in {"y_true", "y_pred_transfer", "y_pred_local", "abs_error_transfer", "abs_error_local"}:
            arrays[f] = pa.array(vals, type=pa.float64())
        else:
            arrays[f] = pa.array([None if v is None else str(v) for v in vals], type=pa.string())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), path)


def render_report(summary: dict[str, Any]) -> str:
    q1, q2 = summary["rq1"], summary["rq2"]
    sc = summary["support_change"]
    return "\n".join(
        [
            "# S12 P3 — R9 adaptive scaling sensitivity",
            "",
            "Sensitivity arm only. IQR → 1.4826×MAD → omit feature from the support metric. Predictive SplineRidge features are unchanged. Accepted primary protocol is not replaced.",
            "",
            f"Status: `{summary['status']}`.",
            "",
            f"Computability: primary 85/94; R9 {summary['n_r9_computable']}/94; recovered {summary['n_recovered']}; lost {summary['n_lost']}; PS_048 recovered {summary['ps048_n_recovered']}/9.",
            "",
            f"RQ1 plant-balanced: primary {q1['primary']} vs R9 {q1['plant_balanced']} (Δ {q1['difference_vs_primary']}); overlap-only n={q1['overlap_n']} plant-balanced {q1['overlap_plant_balanced']} (Δ {q1['overlap_difference_vs_primary']}).",
            f"RQ2 plant-balanced |A|: primary {q2['primary']} vs R9 {q2['plant_balanced_abs_asymmetry']} (Δ {q2['difference_vs_primary']}); overlap pairs {q2['overlap_n_pairs']} → {q2['overlap_plant_balanced']}.",
            f"RQ3: `{summary['rq3_status']}`.",
            "",
            f"Support change on 85 primary identities: identical {sc['identical']}, R9 superset {sc['r9_superset']}, subset {sc['r9_subset']}, partial {sc['partial']}; median Jaccard {sc['median_jaccard']}, min {sc['min_jaccard']}.",
            "",
            summary["interpretation"],
            "",
            f"Reconciliation max |Δ|={summary['reconciliation_max_abs_discrepancy']}.",
            "",
        ]
    )
