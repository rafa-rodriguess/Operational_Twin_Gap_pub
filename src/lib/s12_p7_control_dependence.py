"""S12 P7: RQ3 control-identity sensitivity from persisted S11 R8 / mapping artifacts."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from src.io import read_csv

STATUS = "S12_P7_RQ3_CONTROL_DEPENDENCE_MATERIALIZED"
ORIENTATION = "CONTROL_MINUS_TWIN"
PRIMARY_RQ3 = 0.0261997195973106
ACCEPTED_R8 = 0.038537039716828335
TOL = 1e-12
ZERO_TOL = 1e-12
PRIMARY_BY_STATE = {"RJ": "PS_007", "SP": "PS_008", "GO": "PS_005"}
STATES = ("RJ", "SP", "GO")


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


def _close(a: Any, b: Any, tol: float = TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _sd(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    return float(statistics.stdev(vals))


def dist_pack(vals: list[float], primary: float | None = None) -> dict[str, Any]:
    if not vals:
        return {
            "n": 0, "min": None, "p05": None, "p25": None, "median": None, "mean": None,
            "p75": None, "p95": None, "max": None, "sd": None,
            "n_positive": 0, "n_negative": 0, "n_zero": 0,
            "frac_positive": None, "frac_negative": None, "frac_zero": None,
            "n_below_primary": None, "n_equal_primary": None, "n_above_primary": None,
            "frac_below_primary": None, "frac_equal_primary": None, "frac_above_primary": None,
        }
    a = np.asarray(vals, dtype=float)
    n = int(a.size)
    pos = int(np.sum(a > ZERO_TOL))
    neg = int(np.sum(a < -ZERO_TOL))
    zero = int(np.sum(np.abs(a) <= ZERO_TOL))
    out = {
        "n": n,
        "min": float(np.min(a)),
        "p05": float(np.quantile(a, 0.05, method="linear")),
        "p25": float(np.quantile(a, 0.25, method="linear")),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p75": float(np.quantile(a, 0.75, method="linear")),
        "p95": float(np.quantile(a, 0.95, method="linear")),
        "max": float(np.max(a)),
        "sd": float(np.std(a, ddof=1)) if n > 1 else 0.0,
        "n_positive": pos,
        "n_negative": neg,
        "n_zero": zero,
        "frac_positive": pos / n,
        "frac_negative": neg / n,
        "frac_zero": zero / n,
    }
    if primary is not None:
        below = int(np.sum(a < primary - ZERO_TOL))
        equal = int(np.sum(np.abs(a - primary) <= ZERO_TOL))
        above = int(np.sum(a > primary + ZERO_TOL))
        out.update(
            {
                "n_below_primary": below,
                "n_equal_primary": equal,
                "n_above_primary": above,
                "frac_below_primary": below / n,
                "frac_equal_primary": equal / n,
                "frac_above_primary": above / n,
            }
        )
    return out


def _parse_rank(text: str) -> tuple:
    parts = []
    for p in str(text).split("|"):
        try:
            parts.append(float(p))
        except ValueError:
            parts.append(p)
    return tuple(parts)


def load_inputs(root: Path) -> dict[str, Any]:
    mapping = read_csv(root / "artifacts/rq3/mapping.csv")
    elig = read_csv(root / "artifacts/s11/S11_A2_R8_ELIGIBILITY.csv")
    comps = read_csv(root / "artifacts/s11/S11_A2_R8_COMPARISONS.csv")
    tgt_sum = read_csv(root / "artifacts/s11/S11_A2_R8_TARGET_SUMMARY.csv")
    r8_sum = json.loads((root / "artifacts/s11/S11_A2_R8_SUMMARY.json").read_text(encoding="utf-8"))
    return {"mapping": mapping, "elig": elig, "comps": comps, "tgt_sum": tgt_sum, "r8_sum": r8_sum}


def executable_map(elig: list[dict[str, str]]) -> dict[str, list[str]]:
    by: dict[str, list[str]] = defaultdict(list)
    for r in elig:
        if not _flag(r.get("structurally_executable")):
            continue
        t, c = r["target_id"], r["control_id"]
        if c not in by[t]:
            by[t].append(c)
    for t in by:
        by[t] = sorted(by[t])
    return dict(by)


def build_cells(comps: list[dict[str, str]], mapping: list[dict[str, str]], elig: list[dict[str, str]]) -> list[dict[str, Any]]:
    meta = {r["target_plant_id"]: r for r in mapping}
    primary = {r["target_plant_id"]: r["control_plant_id"] for r in mapping}
    valid = [r for r in comps if _flag(r.get("computable"))]
    by: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in valid:
        by[(r["target_id"], r["control_id"])].append(r)
    cells = []
    for (t, c), rows in sorted(by.items()):
        contrasts = [float(x["contrast_control_minus_twin"]) for x in rows]
        rows_n = [float(x["common_row_count"]) for x in rows]
        m = meta.get(t, {})
        mean_c = _mean(contrasts)
        mean_c2 = float(np.mean(np.asarray(contrasts, dtype=float)))
        if not _close(mean_c, mean_c2):
            raise ValueError(f"cell mean recon {t} {c}")
        cells.append(
            {
                "target_id": t,
                "target_state": m.get("target_state") or rows[0].get("target_state"),
                "target_group": m.get("target_group_id"),
                "control_id": c,
                "is_primary_matched_control": c == primary.get(t),
                "n_twin_sources": len(contrasts),
                "target_control_mean": mean_c,
                "min_source_contrast": min(contrasts),
                "max_source_contrast": max(contrasts),
                "sd_source_contrast": _sd(contrasts),
                "mean_common_row_count": _mean(rows_n),
                "min_common_row_count": min(rows_n),
                "max_common_row_count": max(rows_n),
                "orientation": ORIENTATION,
            }
        )
    return cells


def reconstruct_s11_r8(comps: list[dict[str, str]], targets: list[str]) -> tuple[float | None, list[dict[str, Any]], int]:
    valid = [r for r in comps if _flag(r.get("computable"))]
    by_t_src: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in valid:
        by_t_src[(r["target_id"], r["twin_source_id"])].append(float(r["contrast_control_minus_twin"]))
    means = []
    rows = []
    for t in targets:
        src_means = [sum(v) / len(v) for (tt, _s), v in sorted(by_t_src.items()) if tt == t]
        if not src_means:
            continue
        tm = float(sum(src_means) / len(src_means))
        means.append(tm)
        rows.append({"target_id": t, "target_mean_control_minus_twin": tm, "n_sources": len(src_means)})
    fleet = _mean(means)
    return fleet, rows, len(valid)


def reconstruct_s11_r8_iterate(comps: list[dict[str, str]], targets: list[str]) -> float | None:
    means = []
    for t in targets:
        recs = [r for r in comps if r["target_id"] == t and _flag(r.get("computable"))]
        sources = []
        for r in recs:
            if r["twin_source_id"] not in sources:
                sources.append(r["twin_source_id"])
        src_means = []
        for src in sources:
            acc, n = 0.0, 0
            for r in recs:
                if r["twin_source_id"] != src:
                    continue
                acc += float(r["contrast_control_minus_twin"])
                n += 1
            if n:
                src_means.append(acc / n)
        if src_means:
            means.append(sum(src_means) / len(src_means))
    return _mean(means)


def cell_lookup(cells: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    return {(r["target_id"], r["control_id"]): float(r["target_control_mean"]) for r in cells}


def equal_control_within_target(cells: list[dict[str, Any]], targets: list[str]) -> tuple[float | None, dict[str, float]]:
    by_t: dict[str, list[float]] = defaultdict(list)
    for r in cells:
        by_t[r["target_id"]].append(float(r["target_control_mean"]))
    tmeans = {}
    for t in targets:
        tmeans[t] = _mean(by_t.get(t, []))
    fleet = _mean([v for v in tmeans.values() if v is not None])
    return fleet, tmeans


def assignment_fleet(targets: list[str], chosen: dict[str, str], lookup: dict[tuple[str, str], float]) -> float | None:
    vals = []
    for t in targets:
        key = (t, chosen[t])
        if key not in lookup:
            return None
        vals.append(lookup[key])
    return _mean(vals)


def interpret(s: dict[str, Any]) -> dict[str, Any]:
    ss = s["state_shared"]
    ts = s["target_specific"]
    loco = s["loco"]
    signs = s["sign_stability"]
    rng = ss["distribution"]["max"] - ss["distribution"]["min"]
    frac_pos = ss["distribution"]["frac_positive"]
    text = (
        f"Primary RQ3 reconstructed at {s['primary_reconstruction']['fleet']} "
        f"(accepted {PRIMARY_RQ3}). State-shared admissible assignments n={ss['n_assignments']}, "
        f"range [{ss['distribution']['min']}, {ss['distribution']['max']}] (width {rng}); "
        f"{ss['distribution']['n_positive']}/{ss['n_assignments']} positive "
        f"({frac_pos}); fraction equal to primary {ss['distribution']['frac_equal_primary']}; "
        f"below {ss['distribution']['frac_below_primary']}; above {ss['distribution']['frac_above_primary']}. "
        f"Target-specific envelope n={ts['n_assignments']}, range [{ts['distribution']['min']}, {ts['distribution']['max']}], "
        f"frac_positive {ts['distribution']['frac_positive']}. "
        f"Per-target signs: all-positive {signs['targets_all_positive']}, all-negative {signs['targets_all_negative']}, "
        f"mixed {signs['targets_mixed']}; largest control range {s['per_target_summary']['largest_range']} "
        f"on {s['per_target_summary']['largest_range_target']}. "
        f"Selected vs alternatives fleet {s['selected_vs_alternatives']['fleet_selected']} vs "
        f"{s['selected_vs_alternatives']['fleet_alternatives']} (Δ {s['selected_vs_alternatives']['fleet_difference']}). "
        f"LOCO n_estimable={loco['n_estimable']}/{loco['n_replicates']}, estimates "
        f"[{loco['min']}, {loco['max']}], max |shift| {loco['max_abs_shift']} removing {loco['max_abs_shift_control']}; "
        f"frac_positive {loco['frac_positive']}. Primary-control deletions: {s['primary_control_deletions']}. "
        "These are exact enumerations over structurally executable controls, not inferential tests."
    )
    return {"text": text, "threshold_rules_used": False}


def run_science(root: Path, sot: dict[str, Any]) -> dict[str, Any]:
    recon_max = 0.0
    data = load_inputs(root)
    mapping = data["mapping"]
    targets = [r["target_plant_id"] for r in mapping]
    recon_max = max(recon_max, abs(len(targets) - 10))
    by_state: dict[str, list[str]] = {st: [] for st in STATES}
    primary_map = {}
    for r in mapping:
        primary_map[r["target_plant_id"]] = r["control_plant_id"]
        by_state[r["target_state"]].append(r["target_plant_id"])
        expected = PRIMARY_BY_STATE[r["target_state"]]
        if r["control_plant_id"] != expected:
            raise ValueError(f"mapping control {r['target_plant_id']}")

    elig_exec = executable_map(data["elig"])
    cells = build_cells(data["comps"], mapping, data["elig"])
    lookup = cell_lookup(cells)
    valid_comps = [r for r in data["comps"] if _flag(r.get("computable"))]
    recon_max = max(recon_max, abs(len(valid_comps) - 118))
    if any(r.get("orientation") != ORIENTATION for r in data["comps"]):
        raise ValueError("orientation")

    # primary reconstruction two paths
    prim_vals = [lookup[(t, primary_map[t])] for t in targets]
    prim_fleet = _mean(prim_vals)
    prim_fleet2 = float(np.mean(np.asarray(prim_vals, dtype=float)))
    recon_max = max(recon_max, abs(prim_fleet - prim_fleet2), abs(prim_fleet - PRIMARY_RQ3))
    state_primary = {st: _mean([lookup[(t, primary_map[t])] for t in by_state[st]]) for st in STATES}

    r8_accepted_path, r8_rows, n_valid = reconstruct_s11_r8(data["comps"], targets)
    r8_iter = reconstruct_s11_r8_iterate(data["comps"], targets)
    recon_max = max(recon_max, abs((r8_accepted_path or 0) - (r8_iter or 0)), abs((r8_accepted_path or 0) - ACCEPTED_R8), abs(n_valid - 118))
    r8_equal, r8_tmeans = equal_control_within_target(cells, targets)
    r8_equal2 = _mean([_mean([c["target_control_mean"] for c in cells if c["target_id"] == t]) for t in targets])
    recon_max = max(recon_max, abs((r8_equal or 0) - (r8_equal2 or 0)))
    r8_identical = _close(r8_equal, r8_accepted_path)

    # common controls from eligibility, not outcomes
    common: dict[str, list[str]] = {}
    for st in STATES:
        sets = [set(elig_exec[t]) for t in by_state[st]]
        inter = set.intersection(*sets) if sets else set()
        common[st] = sorted(inter)
        if PRIMARY_BY_STATE[st] not in inter:
            raise ValueError(f"primary not in common {st}")
    n_shared = 1
    for st in STATES:
        n_shared *= len(common[st])

    # state-shared assignments
    shared_rows = []
    shared_fleets = []
    for combo in product(*[common[st] for st in STATES]):
        chosen_state = dict(zip(STATES, combo))
        chosen = {t: chosen_state[next(st for st in STATES if t in by_state[st])] for t in targets}
        fleet = assignment_fleet(targets, chosen, lookup)
        if fleet is None:
            raise ValueError(f"missing cell for {chosen}")
        recon = _mean([lookup[(t, chosen[t])] for t in targets])
        recon_max = max(recon_max, abs(fleet - recon))
        shared_fleets.append(fleet)
        shared_rows.append(
            {
                "assignment_id": len(shared_rows),
                "control_RJ": chosen_state["RJ"],
                "control_SP": chosen_state["SP"],
                "control_GO": chosen_state["GO"],
                "fleet": fleet,
                "is_primary_assignment": chosen_state == PRIMARY_BY_STATE,
                "orientation": ORIENTATION,
            }
        )
    recon_max = max(recon_max, abs(len(shared_rows) - n_shared))
    shared_dist = dist_pack(shared_fleets, PRIMARY_RQ3)
    primary_shared_id = next(r["assignment_id"] for r in shared_rows if r["is_primary_assignment"])

    # target-specific
    ctrl_lists = [elig_exec[t] for t in targets]
    n_specific = 1
    for lst in ctrl_lists:
        n_specific *= len(lst)
    spec_rows = []
    spec_fleets = []
    for combo in product(*ctrl_lists):
        chosen = dict(zip(targets, combo))
        fleet = assignment_fleet(targets, chosen, lookup)
        if fleet is None:
            raise ValueError("missing target-specific cell")
        recon = _mean([lookup[(t, chosen[t])] for t in targets])
        recon_max = max(recon_max, abs(fleet - recon))
        spec_fleets.append(fleet)
        rec = {"assignment_id": len(spec_rows), "fleet": fleet, "is_primary_assignment": chosen == primary_map, "orientation": ORIENTATION}
        for t in targets:
            rec[f"control_{t}"] = chosen[t]
        spec_rows.append(rec)
    recon_max = max(recon_max, abs(len(spec_rows) - n_specific))
    spec_dist = dist_pack(spec_fleets, PRIMARY_RQ3)

    # per-target
    tgt_sens = []
    all_pos, all_neg, mixed = [], [], []
    largest_range, largest_t = -1.0, None
    for t in targets:
        ctrls = elig_exec[t]
        vals = [lookup[(t, c)] for c in ctrls]
        prim = lookup[(t, primary_map[t])]
        alts = [lookup[(t, c)] for c in ctrls if c != primary_map[t]]
        pack = dist_pack(vals, prim)
        same_sign = (pack["n_positive"] == pack["n"]) or (pack["n_negative"] == pack["n"]) or (pack["n_zero"] == pack["n"])
        rng = pack["max"] - pack["min"]
        if rng > largest_range:
            largest_range, largest_t = rng, t
        sign_pat = "all_positive" if pack["n_positive"] == pack["n"] else "all_negative" if pack["n_negative"] == pack["n"] else "mixed"
        if sign_pat == "all_positive":
            all_pos.append(t)
        elif sign_pat == "all_negative":
            all_neg.append(t)
        else:
            mixed.append(t)
        elig_rows = [r for r in data["elig"] if r["target_id"] == t and _flag(r.get("structurally_executable"))]
        ranked = sorted(elig_rows, key=lambda r: _parse_rank(r.get("ranking_tuple") or ""))
        prim_rank = next((i + 1 for i, r in enumerate(ranked) if r["control_id"] == primary_map[t]), None)
        tgt_sens.append(
            {
                "target_id": t,
                "target_state": next(r["target_state"] for r in mapping if r["target_plant_id"] == t),
                "n_eligible_controls": len(ctrls),
                "primary_control": primary_map[t],
                "primary_target_mean": prim,
                "all_control_mean": _mean(vals),
                "alternative_only_mean": _mean(alts),
                "selected_minus_alternatives": (prim - _mean(alts)) if alts else None,
                "min": pack["min"],
                "p25": pack["p25"],
                "median": pack["median"],
                "p75": pack["p75"],
                "max": pack["max"],
                "range": rng,
                "sd": pack["sd"],
                "n_positive": pack["n_positive"],
                "n_negative": pack["n_negative"],
                "n_zero": pack["n_zero"],
                "all_same_sign": same_sign,
                "sign_pattern": sign_pat,
                "frac_controls_below_primary": pack["frac_below_primary"],
                "frac_controls_equal_primary": pack["frac_equal_primary"],
                "frac_controls_above_primary": pack["frac_above_primary"],
                "primary_structural_rank": prim_rank,
                "primary_best_structural_rank": prim_rank == 1,
                "n_structural_ties_at_primary": sum(1 for r in ranked if _parse_rank(r.get("ranking_tuple") or "") == _parse_rank(next(x["ranking_tuple"] for x in ranked if x["control_id"] == primary_map[t]))),
            }
        )

    sel_vals = [lookup[(t, primary_map[t])] for t in targets]
    alt_vals = []
    for t in targets:
        alts = [lookup[(t, c)] for c in elig_exec[t] if c != primary_map[t]]
        if not alts:
            alt_vals.append(None)
        else:
            alt_vals.append(_mean(alts))
    fleet_sel = _mean(sel_vals)
    fleet_alt = _mean([v for v in alt_vals if v is not None])
    recon_max = max(recon_max, abs((fleet_sel or 0) - PRIMARY_RQ3))
    sva_state = {}
    for st in STATES:
        sva_state[st] = {
            "selected": _mean([lookup[(t, primary_map[t])] for t in by_state[st]]),
            "alternatives": _mean([_mean([lookup[(t, c)] for c in elig_exec[t] if c != primary_map[t]]) for t in by_state[st]]),
        }

    # LOCO
    distinct = sorted({c for lst in elig_exec.values() for c in lst})
    recon_max = max(recon_max, abs(len(distinct) - 12))
    loco_rows = []
    for removed in distinct:
        tmeans = []
        affected = []
        ok = True
        remaining_c = set()
        for t in targets:
            leftover = [lookup[(t, c)] for c in elig_exec[t] if c != removed]
            leftover_ids = [c for c in elig_exec[t] if c != removed]
            remaining_c.update(leftover_ids)
            if removed in elig_exec[t]:
                affected.append(t)
            if not leftover:
                ok = False
                tmeans = None
                break
            tmeans.append(_mean(leftover))
        if not ok:
            rec = {
                "removed_control": removed,
                "estimable": False,
                "states_affected": ",".join(sorted({next(r["target_state"] for r in mapping if r["target_plant_id"] == t) for t in affected})) if affected else "",
                "targets_affected": ",".join(affected),
                "n_remaining_distinct_controls": None,
                "n_targets_retained": 0,
                "fleet": None,
                "shift_from_equal_control_r8": None,
                "sign": None,
            }
        else:
            fleet = _mean(tmeans)
            rec = {
                "removed_control": removed,
                "estimable": True,
                "states_affected": ",".join(sorted({next(r["target_state"] for r in mapping if r["target_plant_id"] == t) for t in affected})) if affected else "",
                "targets_affected": ",".join(affected),
                "n_remaining_distinct_controls": len(remaining_c),
                "n_targets_retained": 10,
                "fleet": fleet,
                "shift_from_equal_control_r8": (fleet - r8_equal) if r8_equal is not None else None,
                "sign": "positive" if fleet > ZERO_TOL else "negative" if fleet < -ZERO_TOL else "zero",
            }
        loco_rows.append(rec)
    estimable_loco = [r for r in loco_rows if r["estimable"]]
    loco_fleets = [float(r["fleet"]) for r in estimable_loco]
    shifts = [abs(float(r["shift_from_equal_control_r8"])) for r in estimable_loco]
    max_shift_i = max(range(len(estimable_loco)), key=lambda i: shifts[i]) if shifts else None
    loco_sum = {
        "n_replicates": 12,
        "n_estimable": len(estimable_loco),
        "n_nonestimable": 12 - len(estimable_loco),
        "min": min(loco_fleets) if loco_fleets else None,
        "max": max(loco_fleets) if loco_fleets else None,
        "median": float(np.median(loco_fleets)) if loco_fleets else None,
        "max_abs_shift": max(shifts) if shifts else None,
        "max_abs_shift_control": estimable_loco[max_shift_i]["removed_control"] if max_shift_i is not None else None,
        "n_positive": sum(1 for v in loco_fleets if v > ZERO_TOL),
        "frac_positive": (sum(1 for v in loco_fleets if v > ZERO_TOL) / len(loco_fleets)) if loco_fleets else None,
        "sign_stable_all_positive": bool(loco_fleets) and all(v > ZERO_TOL for v in loco_fleets),
    }
    # independent loco path: skip-control rebuild
    for r in estimable_loco:
        alt = []
        for t in targets:
            leftover = [lookup[(t, c)] for c in elig_exec[t] if c != r["removed_control"]]
            alt.append(_mean(leftover))
        recon_max = max(recon_max, abs(_mean(alt) - float(r["fleet"])))

    primary_del = {cid: next((r for r in loco_rows if r["removed_control"] == cid), None) for cid in ("PS_005", "PS_007", "PS_008")}

    reuse = []
    ctrl_state = {}
    for r in data["elig"]:
        if _flag(r.get("structurally_executable")):
            ctrl_state[r["control_id"]] = r["target_state"]
    for cid in distinct:
        tcells = [c for c in cells if c["control_id"] == cid]
        ts = sorted({c["target_id"] for c in tcells})
        vals = [c["target_control_mean"] for c in tcells]
        reuse.append(
            {
                "control_id": cid,
                "state": ctrl_state.get(cid),
                "n_rq3_targets_executable": sum(1 for t in targets if cid in elig_exec[t]),
                "n_target_control_cells": len(tcells),
                "mean_target_control_contrast": _mean(vals),
                "min_target_control_contrast": min(vals) if vals else None,
                "max_target_control_contrast": max(vals) if vals else None,
                "is_primary_matched_control": cid in PRIMARY_BY_STATE.values(),
            }
        )
    recon_max = max(recon_max, abs(sum(r["n_target_control_cells"] for r in reuse) - len(cells)))

    # sign from persisted-like tables
    recon_max = max(
        recon_max,
        abs(len(all_pos) + len(all_neg) + len(mixed) - 10),
        abs(shared_dist["n_positive"] / shared_dist["n"] - sum(1 for v in shared_fleets if v > ZERO_TOL) / len(shared_fleets)),
    )

    sot_rq3 = ((sot.get("results") or {}).get("tables") or {}).get("rq3")
    recon_max = max(recon_max, abs(float(sot_rq3) - PRIMARY_RQ3) if sot_rq3 is not None else 0.0)

    n_elig_expected = {"PS_002": 2, "PS_003": 2, "PS_035": 8, "PS_039": 8, "PS_042": 2, "PS_043": 2, "PS_044": 2, "PS_048": 2, "PS_049": 2, "PS_050": 2}
    for t, n in n_elig_expected.items():
        recon_max = max(recon_max, abs(len(elig_exec[t]) - n))

    summary = {
        "status": STATUS,
        "orientation": ORIENTATION,
        "n_primary_targets": 10,
        "n_valid_source_comparisons": n_valid,
        "n_target_control_cells": len(cells),
        "n_distinct_controls": len(distinct),
        "primary_reconstruction": {
            "fleet": prim_fleet,
            "accepted": PRIMARY_RQ3,
            "difference": prim_fleet - PRIMARY_RQ3,
            "state_means": state_primary,
            "mapping": {"RJ": "PS_007", "SP": "PS_008", "GO": "PS_005"},
        },
        "broad_r8": {
            "accepted_source_level_semantics": r8_accepted_path,
            "accepted_value": ACCEPTED_R8,
            "equal_control_within_target": r8_equal,
            "numerically_identical": r8_identical,
            "weighting_note": (
                "identical"
                if r8_identical
                else "Accepted S11 R8 averages, for each target, source-level means of contrasts across controls (mean_s mean_c). R8_equal_control_within_target averages target-control cell means equally (mean_c then mean_t), so controls are not weighted by how many twin-source rows they have."
            ),
        },
        "state_common_controls": common,
        "state_shared": {
            "n_common_RJ": len(common["RJ"]),
            "n_common_SP": len(common["SP"]),
            "n_common_GO": len(common["GO"]),
            "n_assignments": len(shared_rows),
            "distribution": shared_dist,
            "primary_assignment_id": primary_shared_id,
            "primary_fleet": PRIMARY_RQ3,
        },
        "target_specific": {
            "n_controls_by_target": {t: len(elig_exec[t]) for t in targets},
            "n_assignments": len(spec_rows),
            "distribution": spec_dist,
            "label": "broader target-specific envelope, not the closest perturbation to the primary design",
        },
        "selected_vs_alternatives": {
            "fleet_selected": fleet_sel,
            "fleet_alternatives": fleet_alt,
            "fleet_difference": (fleet_sel - fleet_alt) if fleet_sel is not None and fleet_alt is not None else None,
            "by_state": sva_state,
            "note": "Primary mapping is structural/upstream; this contrast is descriptive of selected vs remaining executable controls.",
        },
        "loco": loco_sum,
        "primary_control_deletions": {
            cid: {
                "estimable": rec["estimable"] if rec else False,
                "fleet": rec["fleet"] if rec else None,
                "shift_from_equal_control_r8": rec["shift_from_equal_control_r8"] if rec else None,
                "sign": rec["sign"] if rec else None,
            }
            for cid, rec in primary_del.items()
        },
        "sign_stability": {
            "targets_all_positive": all_pos,
            "targets_all_negative": all_neg,
            "targets_mixed": mixed,
            "frac_state_shared_positive": shared_dist["frac_positive"],
            "frac_target_specific_positive": spec_dist["frac_positive"],
            "frac_loco_positive": loco_sum["frac_positive"],
        },
        "per_target_summary": {
            "largest_range": largest_range,
            "largest_range_target": largest_t,
        },
        "control_reuse": {
            "n_controls": len(reuse),
            "rows": reuse,
        },
        "reconciliation_max_abs_discrepancy": recon_max,
        "reconciliation_ok": recon_max <= 1e-9,
        "threshold_rules_used": False,
        "no_refit": True,
        "no_scoring": True,
        "no_support_rebuild": True,
    }
    summary["interpretation"] = interpret(summary)
    return {
        "targets": targets,
        "cells": cells,
        "target_sens": tgt_sens,
        "shared_rows": shared_rows,
        "spec_rows": spec_rows,
        "loco_rows": loco_rows,
        "reuse": reuse,
        "summary": _jsonable(summary),
    }


def render_report(s: dict[str, Any]) -> str:
    ss, ts, loco = s["state_shared"], s["target_specific"], s["loco"]
    return "\n".join(
        [
            "# S12 P7 — RQ3 control-identity dependence",
            "",
            "Persisted-artifact analysis. Orientation CONTROL_MINUS_TWIN. No refit, scoring, or support rebuild. G3/R7/R9/P4/R10 unused.",
            "",
            f"Status: `{s['status']}`.",
            f"Primary reconstruction {s['primary_reconstruction']['fleet']} (accepted {s['primary_reconstruction']['accepted']}, Δ {s['primary_reconstruction']['difference']}). State primary means {s['primary_reconstruction']['state_means']}. Mapping RJ/SP/GO = PS_007/PS_008/PS_005.",
            f"Accepted S11 R8 (source-level semantics) {s['broad_r8']['accepted_source_level_semantics']} vs {s['broad_r8']['accepted_value']}. Equal-control-within-target {s['broad_r8']['equal_control_within_target']}. Identical={s['broad_r8']['numerically_identical']}. {s['broad_r8']['weighting_note']}",
            f"Common executable controls: RJ {s['state_common_controls']['RJ']}; SP {s['state_common_controls']['SP']}; GO {s['state_common_controls']['GO']}.",
            f"State-shared assignments n={ss['n_assignments']} (headline). Dist min/P05/P25/median/mean/P75/P95/max {ss['distribution']['min']}/{ss['distribution']['p05']}/{ss['distribution']['p25']}/{ss['distribution']['median']}/{ss['distribution']['mean']}/{ss['distribution']['p75']}/{ss['distribution']['p95']}/{ss['distribution']['max']}; SD {ss['distribution']['sd']}; frac+/−/0 {ss['distribution']['frac_positive']}/{ss['distribution']['frac_negative']}/{ss['distribution']['frac_zero']}; vs primary below/eq/above {ss['distribution']['frac_below_primary']}/{ss['distribution']['frac_equal_primary']}/{ss['distribution']['frac_above_primary']}. Primary assignment_id={ss['primary_assignment_id']}.",
            f"Target-specific envelope n={ts['n_assignments']} ({ts['label']}). Dist min/median/mean/max {ts['distribution']['min']}/{ts['distribution']['median']}/{ts['distribution']['mean']}/{ts['distribution']['max']}; frac+ {ts['distribution']['frac_positive']}.",
            f"Selected vs alternatives fleet {s['selected_vs_alternatives']['fleet_selected']} vs {s['selected_vs_alternatives']['fleet_alternatives']} (Δ {s['selected_vs_alternatives']['fleet_difference']}). By state {s['selected_vs_alternatives']['by_state']}.",
            f"LOCO {loco['n_estimable']}/{loco['n_replicates']} estimable; min/median/max {loco['min']}/{loco['median']}/{loco['max']}; max |shift| {loco['max_abs_shift']} ({loco['max_abs_shift_control']}); frac+ {loco['frac_positive']}. Primary deletions {s['primary_control_deletions']}.",
            f"Sign stability targets +/{s['sign_stability']['targets_all_positive']} −/{s['sign_stability']['targets_all_negative']} mixed/{s['sign_stability']['targets_mixed']}.",
            f"Interpretation: {s['interpretation']['text']}",
            f"Reconciliation max |Δ|={s['reconciliation_max_abs_discrepancy']} ok={s['reconciliation_ok']}.",
            "",
        ]
    )
