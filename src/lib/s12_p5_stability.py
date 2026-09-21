"""S12 P5: plant-delete jackknife and G2/non-G2 stability. Persisted primary artifacts only."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from config.protocol import FAM_SPLINE, PRIMARY_GROUP_IDS, SENSITIVITY_A_GROUP_ID
from src.lib.s12_p1_null import load_scope
from src.lib.s12_p2_retention import (
    association_universe,
    load_asymmetry,
    load_directional,
    plant_balanced_asym,
    plant_balanced_otg,
    plant_balanced_otg_iterate,
)

STATUS = "S12_P5_PLANT_DELETE_STABILITY_MATERIALIZED"
RQ3_STATUS = "P5_RQ3_NOT_RUN_STABILITY_SCOPE"
G2 = "TW_4467a039e00b"
G1 = "TW_04b9f5d95694"
G4 = "TW_88a108921af7"
NONG2 = (G1, G4)
TOL = 1e-12


def _flag(val: Any) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def load_pairs(root: Path) -> list[dict[str, Any]]:
    out = []
    for row in load_asymmetry(root):
        if row.get("group_id") not in PRIMARY_GROUP_IDS:
            continue
        if row.get("group_id") == SENSITIVITY_A_GROUP_ID:
            continue
        if not _flag(row.get("asymmetry_computable")):
            continue
        out.append(
            {
                "group_id": row["group_id"],
                "plant_a": row["plant_a"],
                "plant_b": row["plant_b"],
                "asymmetry_abs": float(row["asymmetry_abs"]),
                "asymmetry_signed": float(row["asymmetry_signed"]),
            }
        )
    return out


def plant_balanced_asym_iterate(pairs: list[dict[str, Any]]) -> tuple[float | None, int]:
    plants: list[str] = []
    for r in pairs:
        for p in (str(r["plant_a"]), str(r["plant_b"])):
            if p not in plants:
                plants.append(p)
    if not plants:
        return None, 0
    means = []
    for p in plants:
        vals = []
        for r in pairs:
            if str(r["plant_a"]) == p or str(r["plant_b"]) == p:
                vals.append(float(r["asymmetry_abs"]))
        means.append(sum(vals) / len(vals))
    return float(sum(means) / len(means)), len(means)


def rq1_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pb, nt = plant_balanced_otg(rows)
    pb2, nt2 = plant_balanced_otg_iterate(rows)
    abs_v = [float(r["otg_abs"]) for r in rows]
    rel_v = [float(r["otg_rel"]) for r in rows if r.get("otg_rel") is not None]
    sources = {str(r["source_plant_id"]) for r in rows}
    targets = {str(r["target_plant_id"]) for r in rows}
    return {
        "n_directional": len(rows),
        "n_targets": nt,
        "n_sources": len(sources),
        "n_plants": len(sources | targets),
        "plant_balanced": pb,
        "plant_balanced_iterate": pb2,
        "n_targets_iterate": nt2,
        "directional_mean": float(np.mean(abs_v)) if abs_v else None,
        "directional_median": float(np.median(abs_v)) if abs_v else None,
        "rel_median": float(np.median(rel_v)) if rel_v else None,
        "rel_p75": float(np.quantile(rel_v, 0.75, method="linear")) if rel_v else None,
        "recon_pb": abs((pb or 0) - (pb2 or 0)) if pb is not None else 0.0,
    }


def rq2_block(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    pb, npl = plant_balanced_asym(pairs)
    pb2, npl2 = plant_balanced_asym_iterate(pairs)
    vals = [float(r["asymmetry_abs"]) for r in pairs]
    return {
        "n_pairs": len(pairs),
        "n_plants": npl,
        "plant_balanced": pb,
        "plant_balanced_iterate": pb2,
        "n_plants_iterate": npl2,
        "median_abs": float(np.median(vals)) if vals else None,
        "estimable": bool(pairs),
        "recon_pb": abs((pb or 0) - (pb2 or 0)) if pb is not None else 0.0,
        "reason": None if pairs else "no_remaining_bidirectional_pair",
    }


def drop_plant_rows(rows: list[dict[str, Any]], plant: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["source_plant_id"] != plant and r["target_plant_id"] != plant]


def drop_plant_pairs(pairs: list[dict[str, Any]], plant: str) -> list[dict[str, Any]]:
    return [r for r in pairs if r["plant_a"] != plant and r["plant_b"] != plant]


def filter_group_rows(rows: list[dict[str, Any]], groups: tuple[str, ...] | str) -> list[dict[str, Any]]:
    if isinstance(groups, str):
        groups = (groups,)
    return [r for r in rows if r["group_id"] in groups]


def filter_group_pairs(pairs: list[dict[str, Any]], groups: tuple[str, ...] | str) -> list[dict[str, Any]]:
    if isinstance(groups, str):
        groups = (groups,)
    return [r for r in pairs if r["group_id"] in groups]


def influence_summary(reps: list[dict[str, Any]], field: str, full: float | None) -> dict[str, Any]:
    usable = [r for r in reps if r.get(field) is not None]
    vals = [float(r[field]) for r in usable]
    if not vals:
        return {"n": 0}
    deltas = [float(r["delta"]) for r in usable]
    abs_d = [abs(d) for d in deltas]
    i_max_pos = max(usable, key=lambda r: float(r["delta"]))
    i_max_neg = min(usable, key=lambda r: float(r["delta"]))
    i_max_abs = max(usable, key=lambda r: abs(float(r["delta"])))
    ranking = sorted(usable, key=lambda r: -abs(float(r["delta"])))
    same_sign = None
    if full is not None:
        same_sign = int(sum(np.sign(float(r[field])) == np.sign(full) for r in usable))
    return {
        "n": len(vals),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "median": float(np.median(vals)),
        "range": float(max(vals) - min(vals)),
        "max_abs_shift": float(max(abs_d)),
        "plant_max_positive_shift": i_max_pos["deleted_plant"],
        "delta_max_positive_shift": float(i_max_pos["delta"]),
        "plant_max_negative_shift": i_max_neg["deleted_plant"],
        "delta_max_negative_shift": float(i_max_neg["delta"]),
        "plant_max_abs_shift": i_max_abs["deleted_plant"],
        "n_same_sign_as_full": same_sign,
        "ranking_abs_shift": [
            {"deleted_plant": r["deleted_plant"], "deleted_group": r.get("deleted_group"), "value": r[field], "delta": r["delta"], "abs_delta": abs(float(r["delta"]))}
            for r in ranking
        ],
    }


def interpret(summary: dict[str, Any]) -> dict[str, Any]:
    rq1 = summary["rq1_delete"]
    rq2 = summary["rq2_delete"]
    g2 = summary["g2_baseline"]["rq1"]
    ng = summary["nong2_baseline"]["rq1"]
    g1 = summary["g1_baseline"]["rq1"]
    g4 = summary["g4_baseline"]["rq1"]
    text = (
        f"Leave-one-plant-out RQ1 over 14 deletes ranges [{rq1['min']}, {rq1['max']}] "
        f"(median {rq1['median']}; range {rq1['range']}) versus full primary {summary['rq1_full']}. "
        f"Largest absolute shift is {rq1['max_abs_shift']} when deleting {rq1['plant_max_abs_shift']}; "
        f"max positive shift {rq1['delta_max_positive_shift']} ({rq1['plant_max_positive_shift']}), "
        f"max negative shift {rq1['delta_max_negative_shift']} ({rq1['plant_max_negative_shift']}). "
        f"{rq1['n_same_sign_as_full']}/14 replicates keep the sign of full RQ1. "
        f"RQ2 PB |A| ranges [{rq2['min']}, {rq2['max']}] (median {rq2['median']}) versus full {summary['rq2_full']}; "
        f"largest absolute shift {rq2['max_abs_shift']} ({rq2['plant_max_abs_shift']}). "
        f"G2-only RQ1 {g2['plant_balanced']} (n_dir {g2['n_directional']}) versus non-G2 pooled {ng['plant_balanced']} "
        f"(G1 {g1['plant_balanced']}, G4 {g4['plant_balanced']}). "
        f"These are descriptive influence diagnostics on a transfer network, not iid jackknife intervals."
    )
    descriptors = [
        "leave-one-plant-out RQ1/RQ2 remain defined for all 14 primary deletions",
        "influence is ranked by observed absolute shift, without evaluative plant scores",
        "G2-only versus G1/G4 baselines are subgroup contrasts, not delete-one replicates",
        "non-G2 is a small descriptive cohort; pair estimability can fail when a two-plant group is broken",
    ]
    return {"text": text, "descriptors": descriptors, "threshold_rules_used": False, "jackknife_ci": False}


def run_science(root: Path, sot: dict[str, Any]) -> dict[str, Any]:
    plants, members = load_scope(root)
    rows = association_universe(load_directional(root))
    pairs = load_pairs(root)
    recon_max = 0.0
    recon_max = max(recon_max, abs(len(plants) - 14), abs(len(rows) - 85), abs(len(pairs) - 38))

    full_rq1 = rq1_block(rows)
    full_rq2 = rq2_block(pairs)
    tables = (sot.get("results") or {}).get("tables") or {}
    sot_rq1 = tables.get("rq1")
    sot_rq2 = tables.get("rq2")
    if isinstance(sot_rq1, dict):
        sot_rq1 = sot_rq1.get("plant_balanced", sot_rq1)
    if isinstance(sot_rq2, dict):
        sot_rq2 = sot_rq2.get("plant_balanced", sot_rq2)
    if sot_rq1 is not None:
        recon_max = max(recon_max, abs(float(full_rq1["plant_balanced"]) - float(sot_rq1)))
    if sot_rq2 is not None:
        recon_max = max(recon_max, abs(float(full_rq2["plant_balanced"]) - float(sot_rq2)))
    recon_max = max(recon_max, full_rq1["recon_pb"], full_rq2["recon_pb"])

    rq1_reps, rq2_reps = [], []
    for p in plants:
        r1 = drop_plant_rows(rows, p)
        r2 = drop_plant_pairs(pairs, p)
        assert all(x["source_plant_id"] != p and x["target_plant_id"] != p for x in r1)
        assert all(x["plant_a"] != p and x["plant_b"] != p for x in r2)
        b1 = rq1_block(r1)
        b2 = rq2_block(r2)
        recon_max = max(recon_max, b1["recon_pb"], b2["recon_pb"])
        d1 = b1["plant_balanced"] - full_rq1["plant_balanced"]
        d2 = (b2["plant_balanced"] - full_rq2["plant_balanced"]) if b2["plant_balanced"] is not None else None
        rq1_reps.append(
            {
                "deleted_plant": p,
                "deleted_group": members[p],
                "remaining_n_directional": b1["n_directional"],
                "remaining_n_targets": b1["n_targets"],
                "remaining_n_sources": b1["n_sources"],
                "rq1": b1["plant_balanced"],
                "directional_mean": b1["directional_mean"],
                "directional_median": b1["directional_median"],
                "rel_median": b1["rel_median"],
                "rel_p75": b1["rel_p75"],
                "delta": d1,
                "abs_delta": abs(d1),
                "sign": int(np.sign(b1["plant_balanced"])) if b1["plant_balanced"] is not None else None,
            }
        )
        rq2_reps.append(
            {
                "deleted_plant": p,
                "deleted_group": members[p],
                "remaining_n_pairs": b2["n_pairs"],
                "remaining_n_plants": b2["n_plants"],
                "rq2": b2["plant_balanced"],
                "median_abs": b2["median_abs"],
                "delta": d2,
                "abs_delta": abs(d2) if d2 is not None else None,
                "estimable": b2["estimable"],
            }
        )

    g2_rows, g2_pairs = filter_group_rows(rows, G2), filter_group_pairs(pairs, G2)
    ng_rows, ng_pairs = filter_group_rows(rows, NONG2), filter_group_pairs(pairs, NONG2)
    g1_rows, g1_pairs = filter_group_rows(rows, G1), filter_group_pairs(pairs, G1)
    g4_rows, g4_pairs = filter_group_rows(rows, G4), filter_group_pairs(pairs, G4)
    g2_b, ng_b = rq1_block(g2_rows), rq1_block(ng_rows)
    g2_a, ng_a = rq2_block(g2_pairs), rq2_block(ng_pairs)
    g1_b, g4_b = rq1_block(g1_rows), rq1_block(g4_rows)
    g1_a, g4_a = rq2_block(g1_pairs), rq2_block(g4_pairs)
    recon_max = max(recon_max, g2_b["recon_pb"], ng_b["recon_pb"], g1_b["recon_pb"], g4_b["recon_pb"])
    recon_max = max(recon_max, g2_a["recon_pb"], ng_a["recon_pb"], g1_a["recon_pb"], g4_a["recon_pb"])

    g2_plants = [p for p in plants if members[p] == G2]
    ng_plants = [p for p in plants if members[p] in NONG2]
    g2_del, ng_del = [], []
    for p in g2_plants:
        b1 = rq1_block(drop_plant_rows(g2_rows, p))
        b2 = rq2_block(drop_plant_pairs(g2_pairs, p))
        recon_max = max(recon_max, b1["recon_pb"], b2["recon_pb"])
        g2_del.append(
            {
                "deleted_plant": p,
                "deleted_group": G2,
                "scope": "G2_only",
                "rq1": b1["plant_balanced"],
                "n_directional": b1["n_directional"],
                "n_targets": b1["n_targets"],
                "delta_rq1": b1["plant_balanced"] - g2_b["plant_balanced"],
                "rq2": b2["plant_balanced"],
                "n_pairs": b2["n_pairs"],
                "delta_rq2": (b2["plant_balanced"] - g2_a["plant_balanced"]) if b2["plant_balanced"] is not None else None,
                "rq2_estimable": b2["estimable"],
            }
        )
    for p in ng_plants:
        b1 = rq1_block(drop_plant_rows(ng_rows, p))
        b2 = rq2_block(drop_plant_pairs(ng_pairs, p))
        recon_max = max(recon_max, b1["recon_pb"], b2["recon_pb"] if b2["estimable"] else 0.0)
        ng_del.append(
            {
                "deleted_plant": p,
                "deleted_group": members[p],
                "scope": "nonG2_pooled",
                "rq1": b1["plant_balanced"],
                "n_directional": b1["n_directional"],
                "n_targets": b1["n_targets"],
                "delta_rq1": (b1["plant_balanced"] - ng_b["plant_balanced"]) if b1["plant_balanced"] is not None else None,
                "rq2": b2["plant_balanced"],
                "n_pairs": b2["n_pairs"],
                "delta_rq2": (b2["plant_balanced"] - ng_a["plant_balanced"]) if b2["plant_balanced"] is not None and ng_a["plant_balanced"] is not None else None,
                "rq2_estimable": b2["estimable"],
                "rq2_reason": b2["reason"],
            }
        )

    # reconstruct min/max identities from tables
    rq1_inf = influence_summary(rq1_reps, "rq1", full_rq1["plant_balanced"])
    rq2_inf = influence_summary(rq2_reps, "rq2", full_rq2["plant_balanced"])
    from_table_max = max(rq1_reps, key=lambda r: abs(r["delta"]))["deleted_plant"]
    recon_max = max(recon_max, 0.0 if from_table_max == rq1_inf["plant_max_abs_shift"] else 1.0)

    subgroup_rows = []
    for name, b1, b2, npl in (
        ("full_primary", full_rq1, full_rq2, 14),
        ("G2", g2_b, g2_a, len(g2_plants)),
        ("nonG2_pooled", ng_b, ng_a, len(ng_plants)),
        ("G1", g1_b, g1_a, 2),
        ("G4", g4_b, g4_a, 2),
    ):
        subgroup_rows.append(
            {
                "subset": name,
                "n_plants": npl,
                "n_directional": b1["n_directional"],
                "n_targets": b1["n_targets"],
                "rq1": b1["plant_balanced"],
                "rq1_median": b1["directional_median"],
                "rq1_rel_median": b1["rel_median"],
                "rq1_rel_p75": b1["rel_p75"],
                "n_pairs": b2["n_pairs"],
                "rq2": b2["plant_balanced"],
                "rq2_median_abs": b2["median_abs"],
                "delta_rq1_vs_full": (b1["plant_balanced"] - full_rq1["plant_balanced"]) if name != "full_primary" else 0.0,
                "delta_rq2_vs_full": (b2["plant_balanced"] - full_rq2["plant_balanced"]) if name != "full_primary" and b2["plant_balanced"] is not None else (0.0 if name == "full_primary" else None),
            }
        )

    g2_del_inf = influence_summary(
        [{"deleted_plant": r["deleted_plant"], "deleted_group": G2, "rq1": r["rq1"], "delta": r["delta_rq1"]} for r in g2_del],
        "rq1",
        g2_b["plant_balanced"],
    )

    payload = {
        "rq1_full": full_rq1["plant_balanced"],
        "rq2_full": full_rq2["plant_balanced"],
        "rq1_full_block": full_rq1,
        "rq2_full_block": full_rq2,
        "rq1_delete": rq1_inf,
        "rq2_delete": rq2_inf,
        "g2_baseline": {"rq1": g2_b, "rq2": g2_a, "n_plants": len(g2_plants)},
        "nong2_baseline": {"rq1": ng_b, "rq2": ng_a, "n_plants": len(ng_plants)},
        "g1_baseline": {"rq1": g1_b, "rq2": g1_a},
        "g4_baseline": {"rq1": g4_b, "rq2": g4_a},
        "g2_delete": g2_del_inf,
        "membership": {"G2": G2, "G1": G1, "G4": G4, "plants": members},
    }
    summary = {
        "status": STATUS,
        "n_plants": 14,
        "n_directional_primary": len(rows),
        "n_pairs_primary": len(pairs),
        "rq1_full": full_rq1["plant_balanced"],
        "rq2_full": full_rq2["plant_balanced"],
        "rq1_delete": rq1_inf,
        "rq2_delete": rq2_inf,
        "g2_baseline": payload["g2_baseline"],
        "nong2_baseline": payload["nong2_baseline"],
        "g1_baseline": payload["g1_baseline"],
        "g4_baseline": payload["g4_baseline"],
        "g2_delete": g2_del_inf,
        "nong2_delete_rows": ng_del,
        "subgroup_vs_full": {
            "g2_minus_full_rq1": g2_b["plant_balanced"] - full_rq1["plant_balanced"],
            "nong2_minus_full_rq1": ng_b["plant_balanced"] - full_rq1["plant_balanced"],
            "g2_minus_full_rq2": g2_a["plant_balanced"] - full_rq2["plant_balanced"],
            "nong2_minus_full_rq2": (ng_a["plant_balanced"] - full_rq2["plant_balanced"]) if ng_a["plant_balanced"] is not None else None,
        },
        "rq3_status": RQ3_STATUS,
        "rq3_reason": "Canonical RQ3 needs control CS4 pairwise intersection; P5 does not rebuild that pipeline or plant-delete it.",
        "no_refit": True,
        "no_new_scoring": True,
        "excluded": ["G3", "R7", "R9", "P4_recalibration", "OTG_half"],
        "jackknife_ci": False,
        "reconciliation_max_abs_discrepancy": recon_max,
        "reconciliation_ok": recon_max <= TOL,
    }
    summary["interpretation"] = interpret(summary)
    return {
        "plants": plants,
        "members": members,
        "rq1_reps": rq1_reps,
        "rq2_reps": rq2_reps,
        "g2_del": g2_del,
        "ng_del": ng_del,
        "subgroup_rows": subgroup_rows,
        "summary": summary,
    }


def render_report(s: dict[str, Any]) -> str:
    r1, r2 = s["rq1_delete"], s["rq2_delete"]
    return "\n".join(
        [
            "# S12 P5 — plant-delete influence and G2/non-G2 stability",
            "",
            "Persisted primary SplineRidge+CS4 artifacts only. No refit, no new scoring, no R7/R9/P4/G3. Descriptive network influence, not an iid jackknife CI.",
            "",
            f"Status: `{s['status']}`. Full primary RQ1 {s['rq1_full']} (n={s['n_directional_primary']}); RQ2 PB |A| {s['rq2_full']} (n_pairs={s['n_pairs_primary']}).",
            "",
            f"14-plant RQ1 LOPO: min {r1['min']}, max {r1['max']}, median {r1['median']}, range {r1['range']}; max |shift| {r1['max_abs_shift']} ({r1['plant_max_abs_shift']}); same-sign {r1['n_same_sign_as_full']}/14.",
            f"14-plant RQ2 LOPO: min {r2['min']}, max {r2['max']}, median {r2['median']}, range {r2['range']}; max |shift| {r2['max_abs_shift']} ({r2['plant_max_abs_shift']}).",
            "",
            f"G2-only RQ1 {s['g2_baseline']['rq1']['plant_balanced']} (n_dir {s['g2_baseline']['rq1']['n_directional']}); non-G2 pooled {s['nong2_baseline']['rq1']['plant_balanced']}; G1 {s['g1_baseline']['rq1']['plant_balanced']}; G4 {s['g4_baseline']['rq1']['plant_balanced']}.",
            f"G2-only RQ2 {s['g2_baseline']['rq2']['plant_balanced']}; non-G2 RQ2 {s['nong2_baseline']['rq2']['plant_balanced']}.",
            f"G2-delete max |RQ1 shift| {s['g2_delete']['max_abs_shift']} ({s['g2_delete']['plant_max_abs_shift']}).",
            "",
            f"RQ3: `{s['rq3_status']}`.",
            f"Interpretation: {s['interpretation']['text']}",
            "Descriptors: " + "; ".join(s["interpretation"]["descriptors"]) + ".",
            f"threshold_rules_used={s['interpretation']['threshold_rules_used']}. Reconciliation max |Δ|={s['reconciliation_max_abs_discrepancy']} ok={s['reconciliation_ok']}.",
            "",
        ]
    )
