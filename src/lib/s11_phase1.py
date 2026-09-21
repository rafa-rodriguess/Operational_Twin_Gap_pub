"""S11 Phase 1: A3 within-target spread and A4 group-wise plant-balanced RQ1."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from config.protocol import FAM_SPLINE, PRIMARY_GROUP_IDS

GROUP_LABEL = {
    "TW_04b9f5d95694": "G1",
    "TW_4467a039e00b": "G2",
    "TW_88a108921af7": "G4",
}
REQUIRED_S07 = (
    "group_id",
    "source_plant_id",
    "target_plant_id",
    "model_family",
    "support_rule",
    "otg_abs",
    "computable",
)
TOL = 1e-15


def _flag(val: object) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def missing_s07_fields(rows: list[dict[str, str]] | None, path_ok: bool) -> str | None:
    if not path_ok:
        return "artifacts/s07/S07_OTG_DIRECTIONAL.csv"
    if rows is None:
        return "S07_OTG_DIRECTIONAL.csv empty/unreadable"
    if not rows:
        return "S07_OTG_DIRECTIONAL.csv has no data rows"
    missing = [c for c in REQUIRED_S07 if c not in rows[0]]
    if missing:
        return ",".join(missing)
    return None


def filter_s07(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if row.get("group_id") not in PRIMARY_GROUP_IDS:
            continue
        if row.get("model_family") != FAM_SPLINE:
            continue
        if row.get("support_rule") != "CS4":
            continue
        if not _flag(row.get("computable")):
            continue
        out.append(row)
    return out


def _rel_ok(row: dict[str, str]) -> bool:
    return _flag(row.get("otg_rel_defined")) and row.get("otg_rel") not in {None, ""}


def a3_path_groupby(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_t: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_t[str(row["target_plant_id"])].append(row)
    out: list[dict[str, Any]] = []
    for target, recs in sorted(by_t.items()):
        sources = {str(r["source_plant_id"]) for r in recs}
        if len(sources) < 2:
            continue
        ranked = sorted(recs, key=lambda r: (float(r["otg_abs"]), str(r["source_plant_id"])))
        lo, hi = ranked[0], ranked[-1]
        rec: dict[str, Any] = {
            "target_id": target,
            "twin_group": GROUP_LABEL[str(lo["group_id"])],
            "group_id": str(lo["group_id"]),
            "n_computable_sources": len(sources),
            "min_otg": float(lo["otg_abs"]),
            "min_otg_source_id": str(lo["source_plant_id"]),
            "max_otg": float(hi["otg_abs"]),
            "max_otg_source_id": str(hi["source_plant_id"]),
            "spread_abs": float(hi["otg_abs"]) - float(lo["otg_abs"]),
        }
        rels = [r for r in recs if _rel_ok(r)]
        if len(rels) >= 2:
            rlo = min(rels, key=lambda r: (float(r["otg_rel"]), str(r["source_plant_id"])))
            rhi = max(rels, key=lambda r: (float(r["otg_rel"]), str(r["source_plant_id"])))
            rec["min_otg_rel"] = float(rlo["otg_rel"])
            rec["max_otg_rel"] = float(rhi["otg_rel"])
            rec["spread_rel"] = float(rhi["otg_rel"]) - float(rlo["otg_rel"])
        out.append(rec)
    return out


def a3_path_iterate(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    targets = []
    seen = set()
    for row in rows:
        t = str(row["target_plant_id"])
        if t not in seen:
            seen.add(t)
            targets.append(t)
    out: list[dict[str, Any]] = []
    for target in sorted(targets):
        recs = [r for r in rows if str(r["target_plant_id"]) == target]
        sources = []
        for r in recs:
            sid = str(r["source_plant_id"])
            if sid not in sources:
                sources.append(sid)
        if len(sources) < 2:
            continue
        min_otg = max_otg = None
        min_src = max_src = None
        gid = str(recs[0]["group_id"])
        for r in recs:
            val = float(r["otg_abs"])
            sid = str(r["source_plant_id"])
            if min_otg is None or val < min_otg or (val == min_otg and sid < str(min_src)):
                min_otg, min_src = val, sid
            if max_otg is None or val > max_otg or (val == max_otg and sid > str(max_src)):
                max_otg, max_src = val, sid
        rec: dict[str, Any] = {
            "target_id": target,
            "twin_group": GROUP_LABEL[gid],
            "group_id": gid,
            "n_computable_sources": len(sources),
            "min_otg": min_otg,
            "min_otg_source_id": min_src,
            "max_otg": max_otg,
            "max_otg_source_id": max_src,
            "spread_abs": max_otg - min_otg,
        }
        rels = [r for r in recs if _rel_ok(r)]
        if len(rels) >= 2:
            rmin = rmax = None
            smin = smax = None
            for r in rels:
                val = float(r["otg_rel"])
                sid = str(r["source_plant_id"])
                if rmin is None or val < rmin or (val == rmin and sid < str(smin)):
                    rmin, smin = val, sid
                if rmax is None or val > rmax or (val == rmax and sid > str(smax)):
                    rmax, smax = val, sid
            rec["min_otg_rel"] = rmin
            rec["max_otg_rel"] = rmax
            rec["spread_rel"] = rmax - rmin
        out.append(rec)
    return out


def _rows_close(a: list[dict[str, Any]], b: list[dict[str, Any]], keys: list[str]) -> bool:
    if [r["target_id"] for r in a] != [r["target_id"] for r in b]:
        return False
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        for k in keys:
            vx, vy = x.get(k), y.get(k)
            if isinstance(vx, float) or isinstance(vy, float):
                if vx is None or vy is None:
                    return False
                if not math.isclose(float(vx), float(vy), rel_tol=0.0, abs_tol=TOL):
                    return False
            elif vx != vy:
                return False
    return True


def a3_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    spreads = [float(r["spread_abs"]) for r in rows]
    by_g: dict[str, int] = defaultdict(int)
    for r in rows:
        by_g[str(r["twin_group"])] += 1
    max_row = max(rows, key=lambda r: (float(r["spread_abs"]), str(r["target_id"]))) if rows else None
    return {
        "eligible_target_count": n,
        "median_spread": statistics.median(spreads) if spreads else None,
        "max_spread": float(max_row["spread_abs"]) if max_row else None,
        "max_target": max_row["target_id"] if max_row else None,
        "n_eligible_by_twin_group": {k: by_g[k] for k in sorted(by_g)},
        "manuscript_reporting_gate": "MAX_ONLY_PLUS_N" if n < 5 else "MEDIAN_AND_MAX",
    }


def a4_path_groupby(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_t: dict[str, list[float]] = defaultdict(list)
    gid_of: dict[str, str] = {}
    n_dir: dict[str, int] = defaultdict(int)
    for row in rows:
        t = str(row["target_plant_id"])
        by_t[t].append(float(row["otg_abs"]))
        gid_of[t] = str(row["group_id"])
        n_dir[t] += 1
    target_means = []
    for t in sorted(by_t):
        vals = by_t[t]
        target_means.append(
            {
                "target_id": t,
                "twin_group": GROUP_LABEL[gid_of[t]],
                "group_id": gid_of[t],
                "n_directional_transfers": n_dir[t],
                "rq1_j": float(sum(vals) / len(vals)),
            }
        )
    by_g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in target_means:
        by_g[rec["twin_group"]].append(rec)
    groups = []
    for label in ("G1", "G2", "G4"):
        recs = by_g.get(label, [])
        means = [float(r["rq1_j"]) for r in recs]
        groups.append(
            {
                "twin_group": label,
                "group_id": next((k for k, v in GROUP_LABEL.items() if v == label), ""),
                "n_targets": len(recs),
                "n_directional_transfers": int(sum(int(r["n_directional_transfers"]) for r in recs)),
                "plant_balanced_rq1": float(sum(means) / len(means)) if means else None,
                "min_target_mean_otg": min(means) if means else None,
                "max_target_mean_otg": max(means) if means else None,
            }
        )
    return groups, target_means


def a4_path_iterate(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = []
    seen = set()
    for row in rows:
        t = str(row["target_plant_id"])
        if t not in seen:
            seen.add(t)
            targets.append(t)
    target_means = []
    for t in sorted(targets):
        recs = [r for r in rows if str(r["target_plant_id"]) == t]
        acc = 0.0
        for r in recs:
            acc += float(r["otg_abs"])
        mean = acc / len(recs)
        gid = str(recs[0]["group_id"])
        target_means.append(
            {
                "target_id": t,
                "twin_group": GROUP_LABEL[gid],
                "group_id": gid,
                "n_directional_transfers": len(recs),
                "rq1_j": mean,
            }
        )
    groups = []
    for label in ("G1", "G2", "G4"):
        recs = [r for r in target_means if r["twin_group"] == label]
        acc = 0.0
        n_tr = 0
        mn = mx = None
        for r in recs:
            v = float(r["rq1_j"])
            acc += v
            n_tr += int(r["n_directional_transfers"])
            mn = v if mn is None or v < mn else mn
            mx = v if mx is None or v > mx else mx
        groups.append(
            {
                "twin_group": label,
                "group_id": next((k for k, v in GROUP_LABEL.items() if v == label), ""),
                "n_targets": len(recs),
                "n_directional_transfers": n_tr,
                "plant_balanced_rq1": (acc / len(recs)) if recs else None,
                "min_target_mean_otg": mn,
                "max_target_mean_otg": mx,
            }
        )
    return groups, target_means


def plant_balanced_rq1(rows: list[dict[str, str]]) -> tuple[float | None, int]:
    by_t: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_t[str(row["target_plant_id"])].append(float(row["otg_abs"]))
    if not by_t:
        return None, 0
    means = [sum(v) / len(v) for v in by_t.values()]
    return float(sum(means) / len(means)), len(by_t)


def a4_close(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    if [r["twin_group"] for r in a] != [r["twin_group"] for r in b]:
        return False
    keys = ["n_targets", "n_directional_transfers", "plant_balanced_rq1", "min_target_mean_otg", "max_target_mean_otg"]
    for x, y in zip(a, b):
        for k in keys:
            vx, vy = x.get(k), y.get(k)
            if vx is None and vy is None:
                continue
            if isinstance(vx, float) or isinstance(vy, float):
                if vx is None or vy is None:
                    return False
                if not math.isclose(float(vx), float(vy), rel_tol=0.0, abs_tol=TOL):
                    return False
            elif vx != vy:
                return False
    return True
