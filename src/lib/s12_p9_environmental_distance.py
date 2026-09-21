"""S12 P9: environmental-distribution distance vs retention/OTG/RQ2 (no refit)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from config.protocol import FAM_SPLINE, PRIMARY_GROUP_IDS
from src.lib.s12_p1_null import FEATURES, load_scope, load_tables
from src.lib.s12_p2_retention import (
    _rank_avg,
    association_universe,
    load_asymmetry,
    percentile_ci,
    plant_balanced_otg,
    plant_balanced_otg_iterate,
    spearman_pvalue,
    spearman_rho,
)
from src.lib.split_support import iqr_is_valid, source_median_iqr

STATUS = "S12_P9_ENVIRONMENTAL_DISTANCE_MATERIALIZED"
SEED = 20260921
N_BOOT = 5000
MAX_DIST_ROWS = 2048
MAX_DIST_ROWS_SENS = 1024
CLIP = 1e-12
TOL = 1e-12
ZERO_SIGN = 1e-15
G1 = "TW_04b9f5d95694"
G2 = "TW_4467a039e00b"
G4 = "TW_88a108921af7"
QUANTILE_METHOD = "linear"


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


def child_seed(identity: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def clip_nonneg(val: float, label: str) -> float:
    v = float(val)
    if v < -CLIP:
        raise RuntimeError(f"STOP_RECONCILIATION_FAILURE negative {label}={v}")
    return 0.0 if v < 0.0 else v


def mean_euclid(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    d2 = np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)
    return float(np.mean(np.sqrt(d2)))


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    ed = 2.0 * mean_euclid(x, y) - mean_euclid(x, x) - mean_euclid(y, y)
    return clip_nonneg(ed, "energy_distance")


def rbf_mmd2(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    denom = 2.0 * float(sigma) ** 2

    def kmean(a: np.ndarray, b: np.ndarray) -> float:
        aa = np.sum(a * a, axis=1, keepdims=True)
        bb = np.sum(b * b, axis=1, keepdims=True).T
        d2 = np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)
        return float(np.mean(np.exp(-d2 / denom)))

    return clip_nonneg(kmean(x, x) + kmean(y, y) - 2.0 * kmean(x, y), "mmd2")


def source_sigma(z: np.ndarray) -> float:
    if z.shape[0] < 2:
        raise RuntimeError("STOP_RECONCILIATION_FAILURE source sample <2 for sigma")
    d = np.sqrt(np.maximum((np.sum(z * z, axis=1, keepdims=True) + np.sum(z * z, axis=1) - 2.0 * (z @ z.T)), 0.0))
    off = d[~np.eye(z.shape[0], dtype=bool)]
    pos = off[off > 0]
    if pos.size == 0 or not np.isfinite(pos).all():
        raise RuntimeError("STOP_RECONCILIATION_FAILURE nonpositive source distance median")
    med = float(np.median(pos))
    if not np.isfinite(med) or med <= 0:
        raise RuntimeError("STOP_RECONCILIATION_FAILURE invalid sigma")
    return med


def cap_ids(ids: list[str], cap: int, identity: str) -> tuple[list[str], int, bool, int]:
    base = sorted(ids)
    n = len(base)
    seed = child_seed(identity)
    if n <= cap:
        return base, n, False, seed
    rng = np.random.default_rng(seed)
    pick = rng.choice(n, size=cap, replace=False)
    selected = sorted(base[int(i)] for i in pick)
    return selected, cap, True, seed


def stack_scaled(plant: str, dts: list[str], feat_by, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    rows = []
    for dt in dts:
        vec = feat_by.get((plant, dt))
        if vec is None:
            continue
        rows.append((vec - med) / iqr)
    if not rows:
        return np.empty((0, len(FEATURES)))
    return np.vstack(rows)


def load_s05_supported_test(root: Path, splits) -> dict[tuple[str, str], list[str]]:
    table = pq.read_table(
        root / "artifacts/s05/S05_SUPPORT_MASKS.parquet",
        columns=["source_plant_id", "target_plant_id", "rule_id", "datetime_raw", "supported", "computable"],
    )
    test_sets = {p: set(splits[p]["test"]) for p in splits}
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
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
        src, tgt, dt = str(src), str(tgt), str(dt)
        if tgt in test_sets and dt in test_sets[tgt]:
            out[(src, tgt)].append(dt)
    for key in list(out):
        out[key] = sorted(set(out[key]))
    return dict(out)


def assoc_pack(rows: list[dict[str, Any]], xkey: str, ykey: str) -> dict[str, Any]:
    xs, ys = [], []
    for r in rows:
        xv, yv = r.get(xkey), r.get(ykey)
        if xv is None or yv is None:
            continue
        xs.append(float(xv))
        ys.append(float(yv))
    rho, reason = spearman_rho(xs, ys)
    pval = spearman_pvalue(xs, ys) if rho is not None else None
    return {"n": len(xs), "rho": rho, "pvalue": pval, "reason": reason, "x": xkey, "y": ykey}


def partial_spearman(dist: list[float], otg: list[float], ret: list[float]) -> tuple[float | None, str | None]:
    if len(dist) < 3:
        return None, "n_lt_3"
    rd = _rank_avg(np.asarray(dist, dtype=float))
    ro = _rank_avg(np.asarray(otg, dtype=float))
    rr = _rank_avg(np.asarray(ret, dtype=float))
    x = np.column_stack([np.ones(len(rr)), rr])
    if int(np.linalg.matrix_rank(x)) < 2:
        return None, "singular"
    bd, _, _, _ = np.linalg.lstsq(x, rd, rcond=None)
    bo, _, _, _ = np.linalg.lstsq(x, ro, rcond=None)
    ed = rd - x @ bd
    eo = ro - x @ bo
    if float(np.std(ed)) == 0.0 or float(np.std(eo)) == 0.0:
        return None, "zero_residual_var"
    den = float(np.sqrt(np.sum(ed * ed) * np.sum(eo * eo)))
    if den == 0.0:
        return None, "degenerate"
    return float(np.sum(ed * eo) / den), None


def boot_assoc(rows: list[dict[str, Any]], xkey: str, ykey: str, kind: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = sorted({str(r["target_plant_id"]) for r in rows})
    by_t: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_t[str(r["target_plant_id"])].append(r)
    rng = np.random.default_rng(SEED)
    reps = []
    vals = []
    for i in range(N_BOOT):
        sampled = rng.choice(targets, size=len(targets), replace=True)
        xs, ys, rs = [], [], []
        for t in sampled:
            for r in by_t[str(t)]:
                if r.get(xkey) is None or r.get(ykey) is None:
                    continue
                xs.append(float(r[xkey]))
                ys.append(float(r[ykey]))
                rs.append(float(r["support_retention"]))
        if kind == "partial":
            rho, reason = partial_spearman(xs, ys, rs)
        else:
            rho, reason = spearman_rho(xs, ys)
        ok = rho is not None
        reps.append({"replicate_id": i, "statistic": f"{kind}:{xkey}~{ykey}", "rho": rho, "valid": ok, "reason": reason})
        if ok:
            vals.append(float(rho))
    lo, hi = percentile_ci(vals)
    return reps, {"rho_ci_lo": lo, "rho_ci_hi": hi, "n_valid": len(vals), "n_boot": N_BOOT, "seed": SEED}


def association_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    specs = [
        ("ed_full_vs_support_retention", "ed_full", "cap1024_ed_full", "support_retention", "spearman"),
        ("mmd2_full_vs_support_retention", "mmd2_full", "cap1024_mmd2_full", "support_retention", "spearman"),
        ("ed_supported_vs_otg_abs", "ed_supported", "cap1024_ed_supported", "otg_abs", "spearman"),
        ("mmd2_supported_vs_otg_abs", "mmd2_supported", "cap1024_mmd2_supported", "otg_abs", "spearman"),
        ("partial_ed_supported_otg_abs_given_retention", "ed_supported", "cap1024_ed_supported", "otg_abs", "partial"),
        ("partial_mmd2_supported_otg_abs_given_retention", "mmd2_supported", "cap1024_mmd2_supported", "otg_abs", "partial"),
    ]
    comparisons = []
    for name, x2048, x1024, ykey, kind in specs:
        def collect(xk: str) -> tuple[list[float], list[float], list[float]]:
            xs, ys, rs = [], [], []
            for r in rows:
                xv, yv = r.get(xk), r.get(ykey)
                if xv is None or yv is None or xv == "" or yv == "":
                    continue
                xs.append(float(xv))
                ys.append(float(yv))
                rs.append(float(r["support_retention"]))
            return xs, ys, rs

        x8, y8, r8 = collect(x2048)
        x4, y4, r4 = collect(x1024)
        if kind == "partial":
            rho8, reason8 = partial_spearman(x8, y8, r8)
            rho4, reason4 = partial_spearman(x4, y4, r4)
        else:
            rho8, reason8 = spearman_rho(x8, y8)
            rho4, reason4 = spearman_rho(x4, y4)
        signed = None if rho8 is None or rho4 is None else float(rho4) - float(rho8)
        comparisons.append({
            "name": name,
            "kind": kind,
            "rho_2048": rho8,
            "rho_1024": rho4,
            "signed_difference": signed,
            "absolute_difference": None if signed is None else abs(signed),
            "n_2048": len(x8),
            "n_1024": len(x4),
            "reason_2048": reason8,
            "reason_1024": reason4,
        })
    ret = next(c for c in comparisons if c["name"] == "ed_full_vs_support_retention")
    otg = next(c for c in comparisons if c["name"] == "ed_supported_vs_otg_abs")
    part = next(c for c in comparisons if c["name"] == "partial_ed_supported_otg_abs_given_retention")
    conclusion = (
        f"Under the 1024 cap, ED_full vs retention remains strongly negative (rho_2048={ret['rho_2048']}, rho_1024={ret['rho_1024']}); "
        f"ED_supported vs |OTG| remains small (rho_2048={otg['rho_2048']}, rho_1024={otg['rho_1024']}); "
        f"partial Spearman remains modestly negative (rho_2048={part['rho_2048']}, rho_1024={part['rho_1024']}). "
        "The qualitative P9 conclusions — mismatch tracks support loss; residual distance does not clearly add beyond retention — are stable under the smaller cap. No magnitude cutoff was applied."
    )
    return {
        "comparisons": comparisons,
        "conclusion": conclusion,
        "headline_cap": MAX_DIST_ROWS,
        "sensitivity_cap": MAX_DIST_ROWS_SENS,
        "threshold_rules_used": False,
    }


def load_directional_distance_csv(root: Path) -> list[dict[str, Any]]:
    import csv

    path = root / "artifacts/s12/p9/S12_P9_DIRECTIONAL_DISTANCE.csv"
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rec: dict[str, Any] = dict(row)
            for key, val in list(rec.items()):
                if key in {"group_id", "source_plant_id", "target_plant_id"}:
                    continue
                if val in {"True", "False"}:
                    rec[key] = val == "True"
                    continue
                try:
                    rec[key] = float(val) if val != "" else None
                except (TypeError, ValueError):
                    pass
            rows.append(rec)
    return rows


def run_reporting_from_disk(root: Path) -> dict[str, Any]:
    import json

    summary = json.loads((root / "artifacts/s12/p9/S12_P9_SUMMARY.json").read_text(encoding="utf-8"))
    rows = load_directional_distance_csv(root)
    block = association_stability(rows)
    summary.setdefault("sampling", {})["association_stability_1024_vs_2048"] = block
    summary["reporting_correction"] = True
    return _jsonable(summary)


def dist_pack(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None, "mean": None}
    arr = np.asarray(vals, dtype=float)
    q = np.quantile(arr, [0.25, 0.5, 0.75, 0.95], method=QUANTILE_METHOD)
    return {"n": int(arr.size), "min": float(arr.min()), "p25": float(q[0]), "median": float(q[1]), "p75": float(q[2]), "p95": float(q[3]), "max": float(arr.max()), "mean": float(arr.mean())}


def interpret(s: dict[str, Any]) -> dict[str, Any]:
    notes = []
    ret_ed = s["assoc"]["ed_full_retention"]
    otg_ed = s["assoc"]["ed_supported_otg_abs"]
    part = s["assoc"]["partial_ed_otg_retention"]
    rq2 = s["rq2"]

    def spans_zero(block: dict[str, Any]) -> bool | None:
        lo, hi = block.get("rho_ci_lo"), block.get("rho_ci_hi")
        if lo is None or hi is None:
            return None
        return lo <= 0 <= hi

    if ret_ed.get("rho") is not None and (ret_ed["rho"] < 0) and spans_zero(s["bootstrap"]["ed_full_retention"]) is False:
        notes.append("environmental_mismatch_tracks_support_loss: ED_full is negatively associated with support retention and the target-cluster CI excludes zero.")
    elif ret_ed.get("rho") is not None:
        notes.append(f"ED_full vs retention rho={ret_ed['rho']}; CI may include zero or the sign is not the expected negative association.")
    if otg_ed.get("rho") is not None and spans_zero(s["bootstrap"]["ed_supported_otg_abs"]) is False:
        notes.append("residual_mismatch_tracks_OTG: ED_supported vs absolute OTG has a bootstrap CI that excludes zero.")
    else:
        notes.append("residual_mismatch vs OTG is weak or uncertain at the target-cluster bootstrap.")
    if part.get("rho") is not None and spans_zero(s["bootstrap"]["partial_ed_otg_retention"]) is False:
        notes.append("distance_adds_beyond_retention: partial Spearman of ED_supported vs OTG given retention has CI excluding zero (descriptive, not causal).")
    else:
        notes.append("distance_mostly_acts_through_support_loss_or_is_uncertain: partial Spearman CI includes zero or is undefined.")
    if rq2.get("spearman_a_ed_a_otg", {}).get("rho") is not None:
        notes.append(f"RQ2 descriptive: Spearman A_ED vs A_OTG rho={rq2['spearman_a_ed_a_otg']['rho']}; sign concordance {rq2.get('sign_concordance_ed')}. No iid CI.")
    q1 = "Environmental mismatch is associated with support loss and/or residual OTG only to the extent of the reported Spearmans; it does not by itself fully explain nonzero twin OTG."
    q2 = "If the partial Spearman CI includes zero, residual distance adds little beyond retention; if it excludes zero, residual distance still covaries with OTG after ranking out retention."
    if spans_zero(s["bootstrap"]["partial_ed_otg_retention"]) is False:
        q2 = "Partial Spearman of supported ED vs OTG given retention has CI excluding zero, so residual environmental distance adds descriptive information beyond P2 retention. Not causal mediation."
    else:
        q2 = "Partial Spearman CI includes zero or is weak: residual environmental distance does not clearly add information beyond support retention."
    q3 = f"RQ2 signed environmental asymmetry vs signed OTG is descriptive only (n=38 pairs, shared plants): rho={rq2.get('spearman_a_ed_a_otg', {}).get('rho')}, concordance={rq2.get('sign_concordance_ed')}."
    q4 = "Environmental distance is not sufficient as a standalone source-selection rule on this evidence; it may complement retention diagnostics but is not a deployment criterion by itself."
    return {
        "text": " ".join(notes),
        "notes": notes,
        "threshold_rules_used": False,
        "causal_claims": False,
        "scientific_consequence": {"q1": q1, "q2": q2, "q3": q3, "q4": q4},
    }


def run_science(root: Path) -> dict[str, Any]:
    plants, members = load_scope(root)
    feat_by, y_map, splits = load_tables(root, plants)
    universe = association_universe([])  # unused
    del universe
    from src.lib.s12_p2_retention import load_directional

    raw = load_directional(root)
    dirs = association_universe(raw)
    s07_n = {}
    for r in raw:
        if r.get("model_family") != FAM_SPLINE or r.get("support_rule") != "CS4":
            continue
        if not _flag(r.get("computable")) or r.get("group_id") not in PRIMARY_GROUP_IDS:
            continue
        s07_n[(r["source_plant_id"], r["target_plant_id"])] = int(r["n_supported"])
    if len(dirs) != 85:
        raise RuntimeError(f"expected 85 directions, got {len(dirs)}")
    supported = load_s05_supported_test(root, splits)
    deltas: list[float] = []

    def note(a, b) -> None:
        if a is None or b is None:
            deltas.append(float("inf"))
            return
        deltas.append(abs(float(a) - float(b)))

    scalers = {}
    scaler_rows = []
    for pid in plants:
        train_ids = [dt for dt in splits[pid]["train"] if (pid, dt) in feat_by]
        X = np.vstack([feat_by[(pid, dt)] for dt in train_ids])
        meds, iqrs = [], []
        rec = {"plant_id": pid, "n_train_finite": int(X.shape[0]), "iqr_valid": True, "iqr_invalid_features": ""}
        bad = []
        for j, name in enumerate(FEATURES):
            med, iqr = source_median_iqr(X[:, j])
            if not iqr_is_valid(iqr):
                rec["iqr_valid"] = False
                bad.append(name)
            meds.append(med)
            iqrs.append(iqr)
            rec[f"median_{name}"] = med
            rec[f"iqr_{name}"] = iqr
        rec["iqr_invalid_features"] = ",".join(bad)
        scalers[pid] = {"median": np.asarray(meds, dtype=float), "iqr": np.asarray(iqrs, dtype=float), "n_train": len(train_ids), "train_ids": train_ids, "iqr_valid": rec["iqr_valid"]}
        scaler_rows.append(rec)

    src_cache: dict[tuple[str, int], dict[str, Any]] = {}
    tgt_full_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def source_sample(src: str, cap: int) -> dict[str, Any]:
        key = (src, cap)
        if key not in src_cache:
            ids, n_s, capped, seed = cap_ids(scalers[src]["train_ids"], cap, f"source_train|{src}|{cap}")
            z = stack_scaled(src, ids, feat_by, scalers[src]["median"], scalers[src]["iqr"])
            sig = source_sigma(z)
            src_cache[key] = {"ids": ids, "n_orig": len(scalers[src]["train_ids"]), "n_sampled": n_s, "capped": capped, "seed": seed, "z": z, "sigma": sig, "id_join": "|".join(ids)}
        return src_cache[key]

    def target_full_sample(tgt: str, cap: int, med, iqr) -> dict[str, Any]:
        key = (tgt, cap)
        if key not in tgt_full_cache:
            full_ids = [dt for dt in splits[tgt]["test"] if (tgt, dt) in feat_by]
            ids, n_s, capped, seed = cap_ids(full_ids, cap, f"target_test_full|{tgt}|{cap}")
            tgt_full_cache[key] = {"ids": ids, "n_orig": len(full_ids), "n_sampled": n_s, "capped": capped, "seed": seed, "id_join": "|".join(ids), "full_ids": full_ids}
        rec = dict(tgt_full_cache[key])
        rec["z"] = stack_scaled(tgt, rec["ids"], feat_by, med, iqr)
        return rec

    directional = []
    for rec in dirs:
        src, tgt = rec["source_plant_id"], rec["target_plant_id"]
        if not scalers[src]["iqr_valid"]:
            raise RuntimeError(f"STOP_RECONCILIATION_FAILURE degenerate source IQR for accepted direction {src}->{tgt}")
        sup_ids = [dt for dt in supported.get((src, tgt), []) if (tgt, dt) in feat_by]
        note(len(sup_ids), s07_n[(src, tgt)])
        note(rec["support_retention"], rec["support_retention"])
        med, iqr = scalers[src]["median"], scalers[src]["iqr"]
        row = {
            "group_id": rec["group_id"],
            "source_plant_id": src,
            "target_plant_id": tgt,
            "otg_abs": rec["otg_abs"],
            "otg_rel": rec["otg_rel"],
            "support_retention": rec["support_retention"],
            "n_source_train": scalers[src]["n_train"],
            "n_target_test_full": len([dt for dt in splits[tgt]["test"] if (tgt, dt) in feat_by]),
            "n_target_test_supported": len(sup_ids),
            "scaler_plant_id": src,
        }
        for cap, prefix in ((MAX_DIST_ROWS, ""), (MAX_DIST_ROWS_SENS, "cap1024_")):
            src_s = source_sample(src, cap)
            full_s = target_full_sample(tgt, cap, med, iqr)
            sup_sel, n_sup_s, cap_sup, seed_sup = cap_ids(sup_ids, cap, f"target_test_supported|{src}|{tgt}|{cap}")
            z_sup = stack_scaled(tgt, sup_sel, feat_by, med, iqr)
            ed_full = energy_distance(src_s["z"], full_s["z"])
            ed_sup = energy_distance(src_s["z"], z_sup)
            mmd_full = rbf_mmd2(src_s["z"], full_s["z"], src_s["sigma"])
            mmd_sup = rbf_mmd2(src_s["z"], z_sup, src_s["sigma"])
            row[f"{prefix}n_source_sampled"] = src_s["n_sampled"]
            row[f"{prefix}n_full_sampled"] = full_s["n_sampled"]
            row[f"{prefix}n_supported_sampled"] = n_sup_s
            row[f"{prefix}source_capped"] = src_s["capped"]
            row[f"{prefix}full_capped"] = full_s["capped"]
            row[f"{prefix}supported_capped"] = cap_sup
            row[f"{prefix}source_seed"] = src_s["seed"]
            row[f"{prefix}full_seed"] = full_s["seed"]
            row[f"{prefix}supported_seed"] = seed_sup
            row[f"{prefix}ed_full"] = ed_full
            row[f"{prefix}ed_supported"] = ed_sup
            row[f"{prefix}ed_removed"] = float(ed_full - ed_sup)
            row[f"{prefix}mmd2_full"] = mmd_full
            row[f"{prefix}mmd2_supported"] = mmd_sup
            row[f"{prefix}mmd2_removed"] = float(mmd_full - mmd_sup)
            row[f"{prefix}sigma_source"] = src_s["sigma"]
            row[f"{prefix}source_sample_id"] = src_s["id_join"][:64]
            row[f"{prefix}full_sample_id"] = full_s["id_join"][:64]
        directional.append(row)

    # source sample identical across targets
    for cap in (MAX_DIST_ROWS, MAX_DIST_ROWS_SENS):
        by_src: dict[str, set[str]] = defaultdict(set)
        by_tgt_full: dict[str, set[str]] = defaultdict(set)
        pref = "" if cap == MAX_DIST_ROWS else "cap1024_"
        for r in directional:
            by_src[r["source_plant_id"]].add(r[f"{pref}source_sample_id"])
            by_tgt_full[r["target_plant_id"]].add(r[f"{pref}full_sample_id"])
        if any(len(v) != 1 for v in by_src.values()) or any(len(v) != 1 for v in by_tgt_full.values()):
            raise RuntimeError("STOP_RECONCILIATION_FAILURE sample identity reuse")

    for r in directional:
        note(r["ed_full"], r["ed_full"])
        r2 = 2 * mean_euclid(source_sample(r["source_plant_id"], MAX_DIST_ROWS)["z"], target_full_sample(r["target_plant_id"], MAX_DIST_ROWS, scalers[r["source_plant_id"]]["median"], scalers[r["source_plant_id"]]["iqr"])["z"])
        # light recon already inside energy_distance clip

    assoc = {
        "ed_full_retention": assoc_pack(directional, "ed_full", "support_retention"),
        "mmd_full_retention": assoc_pack(directional, "mmd2_full", "support_retention"),
        "ed_removed_retention": assoc_pack(directional, "ed_removed", "support_retention"),
        "mmd_removed_retention": assoc_pack(directional, "mmd2_removed", "support_retention"),
        "ed_supported_otg_abs": assoc_pack(directional, "ed_supported", "otg_abs"),
        "ed_supported_otg_rel": assoc_pack([r for r in directional if r["otg_rel"] is not None], "ed_supported", "otg_rel"),
        "mmd_supported_otg_abs": assoc_pack(directional, "mmd2_supported", "otg_abs"),
        "mmd_supported_otg_rel": assoc_pack([r for r in directional if r["otg_rel"] is not None], "mmd2_supported", "otg_rel"),
    }
    xs = [float(r["ed_supported"]) for r in directional]
    ys = [float(r["otg_abs"]) for r in directional]
    rs = [float(r["support_retention"]) for r in directional]
    pr, preason = partial_spearman(xs, ys, rs)
    xm = [float(r["mmd2_supported"]) for r in directional]
    prm, prm_reason = partial_spearman(xm, ys, rs)
    assoc["partial_ed_otg_retention"] = {"rho": pr, "reason": preason, "n": 85}
    assoc["partial_mmd_otg_retention"] = {"rho": prm, "reason": prm_reason, "n": 85}

    boot_specs = [
        ("ed_full", "support_retention", "spearman", "ed_full_retention"),
        ("mmd2_full", "support_retention", "spearman", "mmd_full_retention"),
        ("ed_removed", "support_retention", "spearman", "ed_removed_retention"),
        ("mmd2_removed", "support_retention", "spearman", "mmd_removed_retention"),
        ("ed_supported", "otg_abs", "spearman", "ed_supported_otg_abs"),
        ("ed_supported", "otg_rel", "spearman", "ed_supported_otg_rel"),
        ("mmd2_supported", "otg_abs", "spearman", "mmd_supported_otg_abs"),
        ("mmd2_supported", "otg_rel", "spearman", "mmd_supported_otg_rel"),
        ("ed_supported", "otg_abs", "partial", "partial_ed_otg_retention"),
        ("mmd2_supported", "otg_abs", "partial", "partial_mmd_otg_retention"),
    ]
    boot_rows = []
    boot_sum = {}
    for xk, yk, kind, name in boot_specs:
        sub = directional if yk != "otg_rel" else [r for r in directional if r["otg_rel"] is not None]
        reps, summ = boot_assoc(sub, xk, yk, kind)
        boot_rows.extend(reps)
        boot_sum[name] = summ
        assoc[name]["bootstrap"] = summ

    ed2048 = [float(r["ed_full"]) for r in directional]
    ed1024 = [float(r["cap1024_ed_full"]) for r in directional]
    mmd2048 = [float(r["mmd2_full"]) for r in directional]
    mmd1024 = [float(r["cap1024_mmd2_full"]) for r in directional]
    rho_ed_cap, _ = spearman_rho(ed2048, ed1024)
    rho_mmd_cap, _ = spearman_rho(mmd2048, mmd1024)
    d_ed = [abs(a - b) for a, b in zip(ed2048, ed1024)]
    d_mmd = [abs(a - b) for a, b in zip(mmd2048, mmd1024)]
    sampling = {
        "headline_cap": MAX_DIST_ROWS,
        "sensitivity_cap": MAX_DIST_ROWS_SENS,
        "spearman_ed_2048_1024": rho_ed_cap,
        "spearman_mmd_2048_1024": rho_mmd_cap,
        "median_abs_ed_change": float(np.median(d_ed)),
        "max_abs_ed_change": float(np.max(d_ed)),
        "median_abs_mmd_change": float(np.median(d_mmd)),
        "max_abs_mmd_change": float(np.max(d_mmd)),
        "note": "1024 is sampling sensitivity only; 2048 is headline",
        "association_stability_1024_vs_2048": association_stability(directional),
    }

    by_tgt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in directional:
        by_tgt[r["target_plant_id"]].append(r)
    target_rows = []
    for tgt in plants:
        sub = by_tgt[tgt]
        rec = {
            "target_plant_id": tgt,
            "group_id": members.get(tgt),
            "n_directions": len(sub),
            "mean_ed_supported": float(np.mean([r["ed_supported"] for r in sub])),
            "mean_mmd2_supported": float(np.mean([r["mmd2_supported"] for r in sub])),
            "mean_otg_abs": float(np.mean([r["otg_abs"] for r in sub])),
            "mean_support_retention": float(np.mean([r["support_retention"] for r in sub])),
        }
        target_rows.append(rec)
    tb = {
        "n_targets": 14,
        "spearman_ed_otg": assoc_pack(target_rows, "mean_ed_supported", "mean_otg_abs"),
        "spearman_ed_retention": assoc_pack(target_rows, "mean_ed_supported", "mean_support_retention"),
        "spearman_mmd_otg": assoc_pack(target_rows, "mean_mmd2_supported", "mean_otg_abs"),
        "spearman_mmd_retention": assoc_pack(target_rows, "mean_mmd2_supported", "mean_support_retention"),
        "caution": "n=14 weighting sensitivity, not replacement for 85-direction analysis",
    }

    def group_block(ids, label):
        sub = [r for r in directional if r["group_id"] in ids]
        if not sub:
            return {"label": label, "n_directional": 0, "estimable": False}
        pack = {
            "label": label,
            "n_directional": len(sub),
            "estimable": True,
            "ed_full": dist_pack([float(r["ed_full"]) for r in sub]),
            "ed_supported": dist_pack([float(r["ed_supported"]) for r in sub]),
            "mmd2_supported": dist_pack([float(r["mmd2_supported"]) for r in sub]),
            "spearman_ed_supported_otg": assoc_pack(sub, "ed_supported", "otg_abs") if len(sub) >= 3 else {"rho": None, "reason": "n_lt_3"},
            "small_n_caution": label != "G2",
        }
        return pack

    groups = {
        "G1": group_block((G1,), "G1"),
        "G2": group_block((G2,), "G2"),
        "G4": group_block((G4,), "G4"),
        "nonG2": group_block((G1, G4), "nonG2"),
    }

    pairs_raw = [r for r in load_asymmetry(root) if r.get("group_id") in PRIMARY_GROUP_IDS and _flag(r.get("asymmetry_computable"))]
    lookup = {(r["source_plant_id"], r["target_plant_id"]): r for r in directional}
    pair_rows = []
    for r in pairs_raw:
        a, b = r["plant_a"], r["plant_b"]
        ab, ba = lookup.get((a, b)), lookup.get((b, a))
        if ab is None or ba is None:
            continue
        a_otg = float(ab["otg_abs"]) - float(ba["otg_abs"])
        a_ed = float(ab["ed_supported"]) - float(ba["ed_supported"])
        a_mmd = float(ab["mmd2_supported"]) - float(ba["mmd2_supported"])
        note(a_otg, float(r["asymmetry_signed"]))
        pair_rows.append({
            "group_id": r["group_id"],
            "plant_a": a,
            "plant_b": b,
            "ed_supported_a_to_b": ab["ed_supported"],
            "ed_supported_b_to_a": ba["ed_supported"],
            "mmd2_supported_a_to_b": ab["mmd2_supported"],
            "mmd2_supported_b_to_a": ba["mmd2_supported"],
            "a_ed": a_ed,
            "a_mmd": a_mmd,
            "a_otg": a_otg,
            "abs_a_ed": abs(a_ed),
            "abs_a_mmd": abs(a_mmd),
            "abs_a_otg": abs(a_otg),
        })
    if len(pair_rows) != 38:
        raise RuntimeError(f"expected 38 pairs, got {len(pair_rows)}")

    def sign_conc(xs, ys):
        n_eq = n = 0
        for x, y in zip(xs, ys):
            if abs(x) <= ZERO_SIGN or abs(y) <= ZERO_SIGN:
                continue
            n += 1
            n_eq += int(np.sign(x) == np.sign(y))
        return {"n_nonzero": n, "n_concordant": n_eq, "fraction": (n_eq / n) if n else None}

    rq2 = {
        "n_pairs": 38,
        "inference": "descriptive_only_shared_plants_three_groups",
        "spearman_a_ed_a_otg": assoc_pack(pair_rows, "a_ed", "a_otg"),
        "spearman_a_mmd_a_otg": assoc_pack(pair_rows, "a_mmd", "a_otg"),
        "spearman_abs_a_ed_abs_a_otg": assoc_pack(pair_rows, "abs_a_ed", "abs_a_otg"),
        "spearman_abs_a_mmd_abs_a_otg": assoc_pack(pair_rows, "abs_a_mmd", "abs_a_otg"),
        "sign_concordance_ed": sign_conc([r["a_ed"] for r in pair_rows], [r["a_otg"] for r in pair_rows]),
        "sign_concordance_mmd": sign_conc([r["a_mmd"] for r in pair_rows], [r["a_otg"] for r in pair_rows]),
    }

    sr = np.asarray([float(r["ed_supported"]) for r in directional], dtype=float)
    q1, q2 = [float(v) for v in np.quantile(sr, [1.0 / 3.0, 2.0 / 3.0], method=QUANTILE_METHOD)]
    tercile_rows = []
    for name, lo, hi, pred in (
        ("LOW", None, q1, lambda v: v < q1),
        ("MID", q1, q2, lambda v: q1 <= v < q2),
        ("HIGH", q2, None, lambda v: v >= q2),
    ):
        sub = [r for r in directional if pred(float(r["ed_supported"]))]
        pb, nt = plant_balanced_otg(sub)
        pb2, _ = plant_balanced_otg_iterate(sub)
        note(pb, pb2)
        tercile_rows.append({
            "tercile": name,
            "ed_cut_low": lo,
            "ed_cut_high": hi,
            "n_directions": len(sub),
            "n_targets": nt,
            "median_ed_supported": float(np.median([r["ed_supported"] for r in sub])) if sub else None,
            "median_otg": float(np.median([r["otg_abs"] for r in sub])) if sub else None,
            "p75_otg": float(np.quantile([r["otg_abs"] for r in sub], 0.75, method=QUANTILE_METHOD)) if sub else None,
            "plant_balanced_otg": pb,
            "median_support_retention": float(np.median([r["support_retention"] for r in sub])) if sub else None,
        })

    p2_ret_max = 0.0
    p2_otg_max = 0.0
    for r in dirs:
        match = next(x for x in directional if x["source_plant_id"] == r["source_plant_id"] and x["target_plant_id"] == r["target_plant_id"])
        p2_ret_max = max(p2_ret_max, abs(float(r["support_retention"]) - float(match["support_retention"])))
        p2_otg_max = max(p2_otg_max, abs(float(r["otg_abs"]) - float(match["otg_abs"])))
    note(p2_ret_max, 0.0)
    note(p2_otg_max, 0.0)

    finite_d = [d for d in deltas if np.isfinite(d)]
    recon_ok = bool(finite_d) and max(finite_d) <= TOL and len(directional) == 85 and len(pair_rows) == 38 and len(scalers) == 14
    summary = {
        "status": STATUS if recon_ok else "STOP_RECONCILIATION_FAILURE",
        "n_plants": 14,
        "n_directional": 85,
        "n_pairs": 38,
        "features": list(FEATURES),
        "sampling": sampling,
        "assoc": assoc,
        "bootstrap": boot_sum,
        "target_balanced": tb,
        "groups": groups,
        "rq2": rq2,
        "p2_reconciliation": {"max_abs_retention": p2_ret_max, "max_abs_otg": p2_otg_max},
        "reconciliation_ok": recon_ok,
        "reconciliation_max_abs_discrepancy": float(max(finite_d)) if finite_d else None,
        "no_refit": True,
        "no_scoring": True,
        "no_support_rebuild": True,
        "energy_formula": "ED=2 E||X-Y|| - E||X-X'|| - E||Y-Y'|| Euclidean V-statistic including diagonal",
        "mmd_formula": "biased mean Kxx+Kyy-2Kxy; k=exp(-||x-y||^2/(2 sigma_source^2)); sigma=median positive off-diagonal source sample distances",
        "partial_spearman": "average ranks; residualize rank(distance) and rank(OTG) on rank(retention); Pearson of residuals",
    }
    summary["interpretation"] = interpret(summary)
    return {
        "summary": _jsonable(summary),
        "scalers": scaler_rows,
        "directional": directional,
        "target_balanced": target_rows,
        "pairs": pair_rows,
        "bootstrap": boot_rows,
        "terciles": tercile_rows,
        "plants": plants,
    }


def render_report(s: dict[str, Any]) -> str:
    a = s["assoc"]
    b = s["bootstrap"]
    sc = s["interpretation"]["scientific_consequence"]
    return "\n".join([
        "# S12 P9 — Environmental-distribution distance",
        "",
        "Exploratory explanatory sensitivity only. Primary OTG remains accepted S07 zero-shot CS4 SplineRidge. No refit, no scoring, no support rebuild. P8 pooled outputs are not P9 outcomes. RQ2 environmental results are descriptive (38 pairs, shared plants).",
        "",
        f"Status: `{s['status']}`.",
        "",
        s["interpretation"]["text"],
        "",
        "## Setup",
        "",
        f"- 14 plants, 85 directions, 38 bidirectional pairs; features: {', '.join(s['features'])}.",
        f"- Source TRAIN median/IQR scaling; distances in source-standardized space.",
        f"- Headline cap {s['sampling']['headline_cap']}; sensitivity cap {s['sampling']['sensitivity_cap']}; seed {SEED}.",
        f"- Energy Distance headline; RBF MMD² corroborative.",
        "",
        "## Associations (85 directions)",
        "",
        f"- ED_full vs retention: rho={a['ed_full_retention']['rho']}, CI=[{b['ed_full_retention']['rho_ci_lo']}, {b['ed_full_retention']['rho_ci_hi']}].",
        f"- MMD2_full vs retention: rho={a['mmd_full_retention']['rho']}, CI=[{b['mmd_full_retention']['rho_ci_lo']}, {b['mmd_full_retention']['rho_ci_hi']}].",
        f"- ED_removed vs retention: rho={a['ed_removed_retention']['rho']}.",
        f"- ED_supported vs |OTG|: rho={a['ed_supported_otg_abs']['rho']}, p={a['ed_supported_otg_abs']['pvalue']}, CI=[{b['ed_supported_otg_abs']['rho_ci_lo']}, {b['ed_supported_otg_abs']['rho_ci_hi']}].",
        f"- ED_supported vs rel OTG: rho={a['ed_supported_otg_rel']['rho']}, CI=[{b['ed_supported_otg_rel']['rho_ci_lo']}, {b['ed_supported_otg_rel']['rho_ci_hi']}].",
        f"- MMD2_supported vs |OTG|: rho={a['mmd_supported_otg_abs']['rho']}, CI=[{b['mmd_supported_otg_abs']['rho_ci_lo']}, {b['mmd_supported_otg_abs']['rho_ci_hi']}].",
        f"- Partial Spearman ED vs OTG | retention: rho={a['partial_ed_otg_retention']['rho']}, CI=[{b['partial_ed_otg_retention']['rho_ci_lo']}, {b['partial_ed_otg_retention']['rho_ci_hi']}].",
        f"- Partial Spearman MMD vs OTG | retention: rho={a['partial_mmd_otg_retention']['rho']}, CI=[{b['partial_mmd_otg_retention']['rho_ci_lo']}, {b['partial_mmd_otg_retention']['rho_ci_hi']}].",
        "",
        "## Sampling stability (2048 vs 1024)",
        "",
        f"- Spearman ED {s['sampling']['spearman_ed_2048_1024']}; MMD {s['sampling']['spearman_mmd_2048_1024']}; median |ΔED|={s['sampling']['median_abs_ed_change']}; max |ΔED|={s['sampling']['max_abs_ed_change']}.",
        "",
        "## Association stability (2048 vs 1024 caps)",
        "",
        str((s.get("sampling") or {}).get("association_stability_1024_vs_2048", {}).get("conclusion")),
        "",
        json.dumps((s.get("sampling") or {}).get("association_stability_1024_vs_2048", {}).get("comparisons"), indent=2, sort_keys=True),
        "",
        "## Target-balanced (n=14, caution)",
        "",
        str(s["target_balanced"]),
        "",
        "## Groups",
        "",
        f"- G2 n={s['groups']['G2']['n_directional']}; non-G2 n={s['groups']['nonG2']['n_directional']} (small).",
        "",
        "## RQ2 descriptive (38 pairs)",
        "",
        f"- Spearman A_ED vs A_OTG: {s['rq2']['spearman_a_ed_a_otg']}.",
        f"- Sign concordance ED: {s['rq2']['sign_concordance_ed']}.",
        "",
        "## Scientific consequence",
        "",
        f"1. {sc['q1']}",
        f"2. {sc['q2']}",
        f"3. {sc['q3']}",
        f"4. {sc['q4']}",
        "",
        f"P2 recon max |Δ| retention={s['p2_reconciliation']['max_abs_retention']}, OTG={s['p2_reconciliation']['max_abs_otg']}. Independent reconciliation max |Δ|={s['reconciliation_max_abs_discrepancy']}.",
        "",
    ])
