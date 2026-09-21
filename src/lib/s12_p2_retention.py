"""S12 P2: OTG vs support retention and R7 (persisted artifacts only; no refit)."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from config.protocol import FAM_SPLINE, PRIMARY_GROUP_IDS

SEED = 20260921
N_BOOT = 5000
R7_THRESHOLD = 0.5
QUANTILE_METHOD = "linear"
OTG_ABS_DEF = "transfer_mae - local_same_rows_mae (S07 otg_abs)"
OTG_REL_DEF = "otg_abs / local_same_rows_mae when local_same_rows_mae != 0 (S07 otg_rel)"
STATUS = "S12_P2_SUPPORT_RETENTION_MATERIALIZED"
RQ3_NC = "R7_RQ3_NOT_COMPUTABLE_FROM_PERSISTED_ARTIFACTS"
TOL = 1e-15


def _flag(val: Any) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def _f(val: Any) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def load_directional(root: Path) -> list[dict[str, str]]:
    path = root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_asymmetry(root: Path) -> list[dict[str, str]]:
    with (root / "artifacts/s07/S07_ASYMMETRY.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def association_universe(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("model_family") != FAM_SPLINE or r.get("support_rule") != "CS4":
            continue
        if not _flag(r.get("computable")):
            continue
        if r.get("group_id") not in PRIMARY_GROUP_IDS:
            continue
        rel_ok = _flag(r.get("otg_rel_defined"))
        out.append(
            {
                "group_id": r["group_id"],
                "source_plant_id": r["source_plant_id"],
                "target_plant_id": r["target_plant_id"],
                "otg_abs": float(r["otg_abs"]),
                "otg_rel": _f(r["otg_rel"]) if rel_ok else None,
                "otg_rel_defined": rel_ok,
                "support_retention": float(r["support_retention"]),
                "local_same_rows_mae": float(r["local_same_rows_mae"]),
                "transfer_mae": float(r["transfer_mae"]),
                "computable": True,
                "model_family": r["model_family"],
                "support_rule": r["support_rule"],
            }
        )
    return out


def _rank_avg(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> tuple[float | None, str | None]:
    if len(xs) < 3:
        return None, "n_lt_3"
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return None, "degenerate_constant"
    rx, ry = _rank_avg(x), _rank_avg(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if den == 0.0:
        return None, "degenerate_rank"
    return float(np.sum(rx * ry) / den), None


def spearman_pvalue(xs: list[float], ys: list[float]) -> float | None:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None
    rho, p = spearmanr(xs, ys, nan_policy="omit")
    if p is None or not np.isfinite(p):
        return None
    return float(p)


def plant_balanced_otg(rows: list[dict[str, Any]]) -> tuple[float | None, int]:
    by_t: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_t[str(r["target_plant_id"])].append(float(r["otg_abs"]))
    if not by_t:
        return None, 0
    means = [sum(v) / len(v) for v in by_t.values()]
    return float(sum(means) / len(means)), len(by_t)


def plant_balanced_otg_iterate(rows: list[dict[str, Any]]) -> tuple[float | None, int]:
    targets: list[str] = []
    for r in rows:
        t = str(r["target_plant_id"])
        if t not in targets:
            targets.append(t)
    if not targets:
        return None, 0
    means = []
    for t in targets:
        vals = [float(r["otg_abs"]) for r in rows if str(r["target_plant_id"]) == t]
        means.append(sum(vals) / len(vals))
    return float(sum(means) / len(means)), len(means)


def plant_balanced_asym(pairs: list[dict[str, Any]]) -> tuple[float | None, int]:
    by_p: dict[str, list[float]] = defaultdict(list)
    for r in pairs:
        v = float(r["asymmetry_abs"])
        by_p[str(r["plant_a"])].append(v)
        by_p[str(r["plant_b"])].append(v)
    if not by_p:
        return None, 0
    means = [sum(v) / len(v) for v in by_p.values()]
    return float(sum(means) / len(means)), len(by_p)


def cluster_bootstrap(rows: list[dict[str, Any]], y_key: str, seed: int, n_boot: int) -> list[dict[str, Any]]:
    targets = sorted({str(r["target_plant_id"]) for r in rows})
    by_t = defaultdict(list)
    for r in rows:
        by_t[str(r["target_plant_id"])].append(r)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_boot):
        sampled = rng.choice(targets, size=len(targets), replace=True)
        xs, ys = [], []
        for t in sampled:
            for r in by_t[str(t)]:
                yv = r.get(y_key)
                if yv is None:
                    continue
                xs.append(float(r["support_retention"]))
                ys.append(float(yv))
        rho, reason = spearman_rho(xs, ys)
        out.append({"replicate_id": i, "rho": rho, "valid": rho is not None, "reason": reason, "n_rows": len(xs)})
    return out


def percentile_ci(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float)
    lo, hi = np.quantile(arr, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def terciles(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sr = np.asarray([float(r["support_retention"]) for r in rows], dtype=float)
    q1, q2 = [float(v) for v in np.quantile(sr, [1.0 / 3.0, 2.0 / 3.0], method=QUANTILE_METHOD)]
    labeled = []
    for r in rows:
        v = float(r["support_retention"])
        if v < q1:
            t = "LOW"
        elif v < q2:
            t = "MID"
        else:
            t = "HIGH"
        rec = dict(r)
        rec["retention_tercile"] = t
        labeled.append(rec)
    return labeled, {"q_one_third": q1, "q_two_thirds": q2, "quantile_method": QUANTILE_METHOD}


def tercile_summary(labeled: list[dict[str, Any]], cuts: dict[str, Any]) -> list[dict[str, Any]]:
    q1, q2 = cuts["q_one_third"], cuts["q_two_thirds"]
    intervals = {"LOW": f"[min, {q1})", "MID": f"[{q1}, {q2})", "HIGH": f"[{q2}, max]"}
    out = []
    for name in ("LOW", "MID", "HIGH"):
        sub = [r for r in labeled if r["retention_tercile"] == name]
        abs_v = [float(r["otg_abs"]) for r in sub]
        rel_v = [float(r["otg_rel"]) for r in sub if r.get("otg_rel") is not None]
        sr_v = [float(r["support_retention"]) for r in sub]
        targets = sorted({r["target_plant_id"] for r in sub})
        groups = Counter(r["group_id"] for r in sub)
        pb, n_t = plant_balanced_otg(sub)
        out.append(
            {
                "tercile": name,
                "retention_interval": intervals[name],
                "cut_q_one_third": q1,
                "cut_q_two_thirds": q2,
                "n_directional": len(sub),
                "n_distinct_targets": len(targets),
                "median_otg_abs": float(np.median(abs_v)) if abs_v else None,
                "p75_otg_abs": float(np.quantile(abs_v, 0.75, method="linear")) if abs_v else None,
                "median_otg_rel": float(np.median(rel_v)) if rel_v else None,
                "p75_otg_rel": float(np.quantile(rel_v, 0.75, method="linear")) if rel_v else None,
                "median_support_retention": float(np.median(sr_v)) if sr_v else None,
                "plant_balanced_otg_abs": pb,
                "n_targets_plant_balanced": n_t,
                "group_composition": dict(groups),
            }
        )
    return out


def r7_rq2_pairs(assoc: list[dict[str, Any]], asym: list[dict[str, str]]) -> list[dict[str, Any]]:
    sr = {(r["source_plant_id"], r["target_plant_id"]): float(r["support_retention"]) for r in assoc}
    out = []
    for row in asym:
        if row.get("group_id") not in PRIMARY_GROUP_IDS:
            continue
        if not _flag(row.get("asymmetry_computable")):
            continue
        a, b = row["plant_a"], row["plant_b"]
        sab, sba = sr.get((a, b)), sr.get((b, a))
        if sab is None or sba is None:
            continue
        if sab >= R7_THRESHOLD and sba >= R7_THRESHOLD:
            rec = {
                "group_id": row["group_id"],
                "plant_a": a,
                "plant_b": b,
                "support_retention_a_to_b": sab,
                "support_retention_b_to_a": sba,
                "asymmetry_abs": float(row["asymmetry_abs"]),
                "asymmetry_signed": float(row["asymmetry_signed"]),
            }
            out.append(rec)
    return out


def interpret_rho(rho: float | None, lo: float | None, hi: float | None) -> str:
    if rho is None:
        return "weak_or_no_clear_association"
    crosses_zero = lo is not None and hi is not None and lo <= 0.0 <= hi
    if rho < 0 and not crosses_zero:
        return "negative_otg_retention_association"
    if rho > 0 and not crosses_zero:
        return "positive_association"
    return "weak_or_no_clear_association"


def run_science(root: Path, sot: dict[str, Any]) -> dict[str, Any]:
    directional = load_directional(root)
    asym = load_asymmetry(root)
    assoc = association_universe(directional)
    if len(assoc) != 85:
        return {"stop": "STOP_RECONCILIATION_FAILURE", "detail": f"n_assoc={len(assoc)} != 85"}
    if any(not r["otg_rel_defined"] or r["otg_rel"] is None for r in assoc):
        return {"stop": "STOP_MISSING_RELATIVE_OTG_DEFINITION", "detail": "undefined otg_rel in computable universe"}
    for r in assoc:
        recon = r["otg_abs"] / r["local_same_rows_mae"] if r["local_same_rows_mae"] != 0 else None
        if recon is None or abs(recon - float(r["otg_rel"])) > 1e-12:
            return {"stop": "STOP_MISSING_RELATIVE_OTG_DEFINITION", "detail": "otg_rel != otg_abs/local_mae"}

    xs = [r["support_retention"] for r in assoc]
    yabs = [r["otg_abs"] for r in assoc]
    yrel = [float(r["otg_rel"]) for r in assoc]
    rho_abs, _ = spearman_rho(xs, yabs)
    rho_rel, _ = spearman_rho(xs, yrel)
    p_abs = spearman_pvalue(xs, yabs)
    p_rel = spearman_pvalue(xs, yrel)
    boot_abs = cluster_bootstrap(assoc, "otg_abs", SEED, N_BOOT)
    boot_rel = cluster_bootstrap(assoc, "otg_rel", SEED, N_BOOT)
    valid_abs = [float(r["rho"]) for r in boot_abs if r["valid"]]
    valid_rel = [float(r["rho"]) for r in boot_rel if r["valid"]]
    ci_abs = percentile_ci(valid_abs)
    ci_rel = percentile_ci(valid_rel)
    labeled, cuts = terciles(assoc)
    terc = tercile_summary(labeled, cuts)

    r7_dir = [r for r in assoc if r["support_retention"] >= R7_THRESHOLD]
    r7_pb, r7_nt = plant_balanced_otg(r7_dir)
    r7_pb2, _ = plant_balanced_otg_iterate(r7_dir)
    r7_pairs = r7_rq2_pairs(assoc, asym)
    r7_as, r7_np = plant_balanced_asym(r7_pairs)
    by_g = defaultdict(list)
    for r in r7_dir:
        by_g[r["group_id"]].append(r)
    group_rq1 = {g: plant_balanced_otg(rows)[0] for g, rows in sorted(by_g.items())}
    by_gp = defaultdict(list)
    for r in r7_pairs:
        by_gp[r["group_id"]].append(r)
    group_rq2 = {g: plant_balanced_asym(rows)[0] for g, rows in sorted(by_gp.items())}

    tables = (sot.get("results") or {}).get("tables") or {}
    prim_rq1 = tables.get("rq1")
    prim_rq2 = tables.get("rq2")
    prim_rq3 = tables.get("rq3")
    r7_abs = [float(r["otg_abs"]) for r in r7_dir]
    r7_rel = [float(r["otg_rel"]) for r in r7_dir]
    r7_as_vals = [float(r["asymmetry_abs"]) for r in r7_pairs]

    recon_max = abs((r7_pb or 0) - (r7_pb2 or 0)) if r7_pb is not None else 0.0
    recon_n = abs(len(assoc) - 85)
    recon_max = max(recon_max, float(recon_n))
    ci2 = percentile_ci(valid_abs)
    if ci_abs[0] is not None and ci2[0] is not None:
        recon_max = max(recon_max, abs(ci_abs[0] - ci2[0]), abs(ci_abs[1] - ci2[1]))

    assoc_interp = interpret_rho(rho_abs, ci_abs[0], ci_abs[1])
    notes = {
        "negative_otg_retention_association": "Lower support retention tends to accompany larger absolute transfer penalties (bootstrap CI excludes 0).",
        "weak_or_no_clear_association": "The absolute OTG tail is not explained simply by retention under the target-cluster bootstrap (CI includes 0 or rho near 0).",
        "positive_association": "Larger absolute OTG occurs with higher support retention; a different explanation is required.",
    }
    r7_note = None
    if r7_pb is not None and prim_rq1 is not None:
        dlt = r7_pb - float(prim_rq1)
        r7_note = f"R7 RQ1 plant-balanced {r7_pb} vs primary {prim_rq1} (signed difference {dlt}). Support coverage is a modifier to the extent this numerical shift matters; R7 is not labeled better or worse."

    summary = {
        "status": STATUS,
        "n_association": len(assoc),
        "otg_abs_definition": OTG_ABS_DEF,
        "otg_rel_definition": OTG_REL_DEF,
        "spearman_abs": {
            "rho": rho_abs,
            "n": len(assoc),
            "p_value_descriptive": p_abs,
            "bootstrap_seed": SEED,
            "n_replicates": N_BOOT,
            "n_valid": len(valid_abs),
            "n_invalid": N_BOOT - len(valid_abs),
            "ci95_low": ci_abs[0],
            "ci95_high": ci_abs[1],
        },
        "spearman_rel": {
            "rho": rho_rel,
            "n": len(assoc),
            "p_value_descriptive": p_rel,
            "bootstrap_seed": SEED,
            "n_replicates": N_BOOT,
            "n_valid": len(valid_rel),
            "n_invalid": N_BOOT - len(valid_rel),
            "ci95_low": ci_rel[0],
            "ci95_high": ci_rel[1],
            "interpreted_separately_from_abs": True,
        },
        "tercile_cutpoints": cuts,
        "terciles": terc,
        "r7": {
            "threshold": R7_THRESHOLD,
            "n_directional": len(r7_dir),
            "n_distinct_targets": len({r["target_plant_id"] for r in r7_dir}),
            "n_distinct_sources": len({r["source_plant_id"] for r in r7_dir}),
            "rq1_plant_balanced": r7_pb,
            "rq1_directional_median": float(np.median(r7_abs)) if r7_abs else None,
            "rq1_rel_median": float(np.median(r7_rel)) if r7_rel else None,
            "rq1_rel_p75": float(np.quantile(r7_rel, 0.75, method="linear")) if r7_rel else None,
            "rq1_group_plant_balanced": group_rq1,
            "rq1_primary": prim_rq1,
            "rq1_abs_difference": (r7_pb - float(prim_rq1)) if r7_pb is not None and prim_rq1 is not None else None,
            "rq2_n_eligible_unordered_pairs": len(r7_pairs),
            "rq2_plant_balanced_abs_asymmetry": r7_as,
            "rq2_n_plants": r7_np,
            "rq2_median_abs_asymmetry": float(np.median(r7_as_vals)) if r7_as_vals else None,
            "rq2_group_plant_balanced": group_rq2,
            "rq2_primary": prim_rq2,
            "rq2_abs_difference": (r7_as - float(prim_rq2)) if r7_as is not None and prim_rq2 is not None else None,
            "rq3_status": RQ3_NC,
            "rq3_reason": "RQ3 uses pairwise TEST intersection of twin CS4 and control CS4 (n_pairwise in source_contrasts.csv) without persisted control-side or intersection retention; S07 support_retention is twin-transfer only. Filtering only the twin leg would be a one-sided proxy. No refit/new scoring in P2.",
            "rq3_primary": prim_rq3,
        },
        "association_interpretation": assoc_interp,
        "association_interpretation_note": notes[assoc_interp],
        "r7_interpretation_note": r7_note,
        "reconciliation_max_abs_discrepancy": recon_max,
        "reconciliation_ok": recon_max <= 1e-12,
        "no_refit": True,
        "no_new_prediction_scoring": True,
        "bootstrap_cluster": "target_plant_id with replacement; all source→target rows of a sampled target included with multiplicity",
        "rq2_r7_rule": "unordered pair eligible iff both directions computable in primary and both support_retention >= 0.5",
    }
    boot_rows = []
    for a, b in zip(boot_abs, boot_rel):
        boot_rows.append(
            {
                "replicate_id": a["replicate_id"],
                "rho_abs": a["rho"],
                "valid_abs": a["valid"],
                "reason_abs": a["reason"],
                "rho_rel": b["rho"],
                "valid_rel": b["valid"],
                "reason_rel": b["reason"],
            }
        )
    return {
        "assoc": labeled,
        "boot_rows": boot_rows,
        "terc": terc,
        "r7_dir": r7_dir,
        "r7_pairs": r7_pairs,
        "summary": summary,
        "rho_abs": rho_abs,
        "cuts": cuts,
    }


def render_report(summary: dict[str, Any]) -> str:
    a, r, t = summary["spearman_abs"], summary["r7"], summary["r7"]
    lines = [
        "# S12 P2 — OTG × support retention (R7)",
        "",
        "Persisted-data diagnostic only. No refit and no new prediction scoring. Accepted primary RQ1–RQ3 / S11 / S12 P0–P1 unchanged.",
        "",
        f"Status: `{summary['status']}`.",
        "",
        "## Association (n=85 computable primary directional transfers)",
        "",
        f"- Absolute OTG vs retention Spearman rho={a['rho']}; descriptive p={a['p_value_descriptive']}; target-cluster bootstrap 95% percentile CI [{a['ci95_low']}, {a['ci95_high']}] (valid {a['n_valid']}/{a['n_replicates']}, seed {a['bootstrap_seed']}).",
        f"- Relative OTG vs retention rho={summary['spearman_rel']['rho']}; CI [{summary['spearman_rel']['ci95_low']}, {summary['spearman_rel']['ci95_high']}] (valid {summary['spearman_rel']['n_valid']}/5000). Relative OTG is interpreted separately.",
        "",
        f"Interpretation: **{summary['association_interpretation']}**. {summary['association_interpretation_note']}",
        "",
        "## Retention terciles",
        "",
        f"Cutpoints (numpy quantile method=linear): 1/3={summary['tercile_cutpoints']['q_one_third']}, 2/3={summary['tercile_cutpoints']['q_two_thirds']}.",
        "",
    ]
    for row in summary["terciles"]:
        lines.append(
            f"- {row['tercile']} {row['retention_interval']}: n={row['n_directional']}, targets={row['n_distinct_targets']}, median OTG_abs={row['median_otg_abs']}, P75 abs={row['p75_otg_abs']}, median rel={row['median_otg_rel']}, plant-balanced abs={row['plant_balanced_otg_abs']}, groups={row['group_composition']}."
        )
    lines += [
        "",
        "## R7 support_retention >= 0.5",
        "",
        f"- RQ1: n_dir={r['n_directional']}, targets={r['n_distinct_targets']}, sources={r['n_distinct_sources']}; plant-balanced OTG={r['rq1_plant_balanced']} vs primary {r['rq1_primary']} (difference {r['rq1_abs_difference']}).",
        f"- RQ2: n_pairs={r['rq2_n_eligible_unordered_pairs']}; plant-balanced |asymmetry|={r['rq2_plant_balanced_abs_asymmetry']} vs primary {r['rq2_primary']} (difference {r['rq2_abs_difference']}).",
        f"- RQ3: `{r['rq3_status']}`. {r['rq3_reason']}",
        "",
        str(summary.get("r7_interpretation_note")),
        "",
        f"Reconciliation max |Δ|={summary['reconciliation_max_abs_discrepancy']}.",
        "",
    ]
    return "\n".join(lines)
