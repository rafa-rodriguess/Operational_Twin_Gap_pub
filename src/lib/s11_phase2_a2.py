"""S11 Phase 2 A2/R8: all structurally eligible same-state non-twin controls."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from config.protocol import CS4, FAM_SPLINE, PRIMARY_GROUP_IDS
from src.lib.matching import ranking_components, scientific_tuple
from src.lib.models import FEATURES, SPLINE_GRID, _fit_select
from src.lib.robustness_struct import build_engine, score_support
from src.lib.rq3_science import (
    _feat_index,
    _load_panel,
    _mae,
    _norm_dt,
    _splits_from_table,
    load_meta,
    select_same_state_controls,
)

TOL = 1e-15
STATUS_EXECUTABLE = "ELIGIBLE_STRUCTURALLY_EXECUTABLE"
STATUS_UNFIT = "STRUCTURAL_CANDIDATE_MODEL_UNFIT"
STATUS_NO_SUPPORT = "STRUCTURAL_CANDIDATE_NO_CS4_PAIRWISE_SUPPORT"
STATUS_UNFIT_NO_SUPPORT = "STRUCTURAL_CANDIDATE_UNFIT_AND_NO_CS4_SUPPORT"
STATUS_NONINFORMATIVE = "S11_PHASE2_A2_STRUCTURALLY_NONINFORMATIVE_EQUIVALENT_TO_PRIMARY"
STATUS_MATERIALIZED = "S11_PHASE2_A2_R8_MATERIALIZED"


def members_path(root: Path) -> Path:
    p = root / "artifacts/twins/members.csv"
    if p.is_file():
        return p
    return root / "artifacts/p02b/P02B_TWIN_MEMBERS.csv"


def rq3_mapping_path(root: Path) -> Path:
    return root / "artifacts/rq3/mapping.csv"


def missing_rq3_inputs(root: Path) -> str | None:
    missing = []
    if not (root / "data/raw/br_pvgen/BR-PVGen_metadata.csv").is_file():
        missing.append("data/raw/br_pvgen/BR-PVGen_metadata.csv")
    if not members_path(root).is_file():
        missing.append("artifacts/twins/members.csv")
    if not rq3_mapping_path(root).is_file():
        missing.append("artifacts/rq3/mapping.csv")
    return ",".join(missing) if missing else None


def load_members(root: Path) -> list[dict[str, str]]:
    import csv

    with members_path(root).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_mapping(root: Path) -> list[dict[str, str]]:
    import csv

    with rq3_mapping_path(root).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_xy_counts(pid: str, splits: dict[str, dict[str, list[str]]], feat_by: dict, y_map: dict) -> tuple[int, int]:
    ntr = nva = 0
    for dt in splits[pid]["train"]:
        if feat_by.get((pid, dt)) is not None and y_map.get((pid, dt)) is not None:
            ntr += 1
    for dt in splits[pid]["validation"]:
        if feat_by.get((pid, dt)) is not None and y_map.get((pid, dt)) is not None:
            nva += 1
    return ntr, nva


def _structure_context(root: Path, plants: list[str], controls: list[str]) -> dict[str, Any] | str:
    panel_path = root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"
    if not panel_path.is_file():
        panel_path = root / "artifacts/panel/plant_panel.parquet"
    mask_path = root / "artifacts/s05/S05_SUPPORT_MASKS.parquet"
    idx_path = root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet"
    if not (panel_path.is_file() and mask_path.is_file() and idx_path.is_file()):
        return "missing_panel_or_s04_s05"
    table = _load_panel(panel_path, plants)
    feat_by, y_map = _feat_index(table)
    control_splits = _splits_from_table(table, controls)
    idx = pq.read_table(idx_path)
    test_by: dict[str, list[str]] = defaultdict(list)
    for pid, dt, split in zip(idx["plant_id"].to_pylist(), idx["datetime_raw"].to_pylist(), idx["split"].to_pylist()):
        if str(split) == "test":
            test_by[str(pid)].append(_norm_dt(dt))
    twin_sup: dict[tuple[str, str], set[str]] = defaultdict(set)
    masks = pq.read_table(mask_path)
    for src, tgt, rid, dt, ok, supported in zip(
        masks["source_plant_id"].to_pylist(),
        masks["target_plant_id"].to_pylist(),
        masks["rule_id"].to_pylist(),
        masks["datetime_raw"].to_pylist(),
        masks["computable"].to_pylist(),
        masks["supported"].to_pylist(),
    ):
        if str(rid) != CS4["id"] or not ok or not supported:
            continue
        twin_sup[(str(src), str(tgt))].add(_norm_dt(dt))
    q = float(CS4["q"])
    engines = {}
    fit_ok: dict[str, bool] = {}
    n_tv: dict[str, tuple[int, int]] = {}
    for pid in controls:
        train_dts = control_splits[pid]["train"]
        engines[pid] = build_engine(pid, train_dts, feat_by, q)
        ntr, nva = _split_xy_counts(pid, control_splits, feat_by, y_map)
        n_tv[pid] = (ntr, nva)
        fit_ok[pid] = ntr >= 8 and nva >= 4
    return {
        "feat_by": feat_by,
        "y_map": y_map,
        "control_splits": control_splits,
        "test_by": test_by,
        "twin_sup": twin_sup,
        "engines": engines,
        "fit_ok": fit_ok,
        "n_tv": n_tv,
        "panel_path": panel_path,
    }


def annotate_executability(
    root: Path,
    rows: list[dict[str, Any]],
    members_by_group: dict[str, list[str]],
) -> str | None:
    """Fill executable flags using splits/support only (no OTG, no predictions)."""
    if not rows:
        return None
    targets = sorted({r["target_id"] for r in rows})
    controls = sorted({r["control_id"] for r in rows})
    plants = sorted(set(targets) | set(controls) | {p for g in members_by_group.values() for p in g})
    ctx = _structure_context(root, plants, controls)
    if isinstance(ctx, str):
        return ctx
    for rec in rows:
        tgt = rec["target_id"]
        ctrl = rec["control_id"]
        gid = rec["target_group_id"]
        ntr, nva = ctx["n_tv"][ctrl]
        rec["n_train_xy"] = ntr
        rec["n_val_xy"] = nva
        rec["structural_candidate"] = True
        rec["model_fit_executable"] = bool(ctx["fit_ok"][ctrl])
        twins_src = [p for p in members_by_group.get(gid, []) if p != tgt]
        test_dts = ctx["test_by"].get(tgt) or []
        ctrl_sup = score_support(ctx["engines"].get(ctrl) or {}, tgt, test_dts, ctx["feat_by"])
        has_sup = False
        for src in twins_src:
            inter = ctx["twin_sup"].get((src, tgt), set()) & ctrl_sup
            if inter:
                has_sup = True
                break
        rec["has_any_cs4_pairwise_support"] = has_sup
        rec["structurally_executable"] = rec["model_fit_executable"] and has_sup
        if rec["structurally_executable"]:
            rec["eligibility_status"] = STATUS_EXECUTABLE
        elif not rec["model_fit_executable"] and not has_sup:
            rec["eligibility_status"] = STATUS_UNFIT_NO_SUPPORT
        elif not rec["model_fit_executable"]:
            rec["eligibility_status"] = STATUS_UNFIT
        else:
            rec["eligibility_status"] = STATUS_NO_SUPPORT
    return None


def _eligibility_summary(
    rows: list[dict[str, Any]],
    mapping: list[dict[str, str]],
) -> dict[str, Any]:
    matched = {r["target_plant_id"]: r["control_plant_id"] for r in mapping}
    targets = [r["target_plant_id"] for r in mapping]
    counts = {t: 0 for t in targets}
    for rec in rows:
        if rec.get("structurally_executable"):
            counts[str(rec["target_id"])] += 1
    vals = [counts[t] for t in targets]
    n_zero = sum(1 for v in vals if v == 0)
    n_one = sum(1 for v in vals if v == 1)
    n_gt1 = sum(1 for v in vals if v > 1)
    median = float(statistics.median(vals)) if vals else None
    informative = median is not None and median != 1
    executable_ids = {r["control_id"] for r in rows if r.get("structurally_executable")}
    return {
        "n_primary_rq3_targets": len(targets),
        "n_structural_candidates": len(rows),
        "n_targets_zero_eligible_controls": n_zero,
        "n_targets_one_eligible_control": n_one,
        "n_targets_more_than_one_eligible_control": n_gt1,
        "distinct_eligible_control_count": len(executable_ids),
        "per_target_control_counts": {t: counts[t] for t in targets},
        "median_eligible_controls_per_target": median,
        "part_b_executed": False,
        "structural_status": STATUS_MATERIALIZED if informative else STATUS_NONINFORMATIVE,
        "eligibility_rule": (
            "structural_candidate = same-state AND not nominal twin AND ranking_components defined; "
            "structurally_executable = candidate AND n_train_xy>=8 AND n_val_xy>=4 AND nonempty CS4 pairwise "
            "intersection with at least one twin source on target test rows. Gate uses structurally_executable only. "
            "Eligibility does not read transfer MAE, local MAE, or CONTROL minus TWIN contrasts."
        ),
        "primary_matched_controls": matched,
        "rq3_targets": targets,
        "primary_group_ids": list(PRIMARY_GROUP_IDS),
    }


def enumerate_eligibility(
    root: Path,
    meta: dict[str, dict[str, Any]],
    members: list[dict[str, str]],
    mapping: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[str]], str | None]:
    _selected, _excluded, group_of, members_by_group = select_same_state_controls(meta, members)
    del _selected, _excluded
    matched = {r["target_plant_id"]: r["control_plant_id"] for r in mapping}
    targets = [r["target_plant_id"] for r in mapping]
    all_ids = sorted(meta.keys())
    rows: list[dict[str, Any]] = []
    for tgt in targets:
        tstate = meta.get(tgt, {}).get("state")
        tcanon = meta.get(tgt, {}).get("canon") or {}
        gid = group_of.get(tgt)
        for cand in all_ids:
            if cand == tgt:
                continue
            is_twin = group_of.get(cand) == gid and gid is not None
            cstate = meta.get(cand, {}).get("state")
            same_state = bool(tstate) and cstate == tstate
            if not same_state or is_twin:
                continue
            ccanon = meta.get(cand, {}).get("canon") or {}
            comp = ranking_components(tcanon, ccanon)
            if comp is None:
                continue
            tup = scientific_tuple(comp)
            rows.append(
                {
                    "target_id": tgt,
                    "target_state": tstate,
                    "control_id": cand,
                    "same_state": True,
                    "is_nominal_twin": False,
                    "structural_candidate": True,
                    "model_fit_executable": False,
                    "has_any_cs4_pairwise_support": False,
                    "structurally_executable": False,
                    "eligibility_status": STATUS_UNFIT,
                    "ranking_tuple": "|".join(str(x) for x in tup),
                    "is_primary_matched_control": cand == matched.get(tgt),
                    "target_group_id": gid,
                    "n_train_xy": None,
                    "n_val_xy": None,
                }
            )
    err = annotate_executability(root, rows, members_by_group)
    summary = _eligibility_summary(rows, mapping)
    return rows, summary, members_by_group, err


def _row_fp(dts: list[str]) -> str:
    payload = "\n".join(sorted(dts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_r8(
    root: Path,
    elig_rows: list[dict[str, Any]],
    members_by_group: dict[str, list[str]],
) -> dict[str, Any]:
    panel_path = root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"
    if not panel_path.is_file():
        panel_path = root / "artifacts/panel/plant_panel.parquet"
    mask_path = root / "artifacts/s05/S05_SUPPORT_MASKS.parquet"
    pred_path = root / "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet"
    idx_path = root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet"
    if not (panel_path.is_file() and mask_path.is_file() and pred_path.is_file() and idx_path.is_file()):
        return {"status": "STOP", "reason": "missing_upstream_s04_s05_s06_or_panel"}

    targets = sorted({r["target_id"] for r in elig_rows})
    controls = sorted({r["control_id"] for r in elig_rows})
    plants = sorted(set(targets) | set(controls) | {p for g in members_by_group.values() for p in g})
    table = _load_panel(panel_path, plants)
    feat_by, y_map = _feat_index(table)
    control_splits = _splits_from_table(table, controls)

    idx = pq.read_table(idx_path)
    test_by: dict[str, list[str]] = defaultdict(list)
    for pid, dt, split in zip(idx["plant_id"].to_pylist(), idx["datetime_raw"].to_pylist(), idx["split"].to_pylist()):
        if str(split) == "test":
            test_by[str(pid)].append(_norm_dt(dt))

    twin_sup: dict[tuple[str, str], set[str]] = defaultdict(set)
    masks = pq.read_table(mask_path)
    for src, tgt, rid, dt, ok, supported in zip(
        masks["source_plant_id"].to_pylist(),
        masks["target_plant_id"].to_pylist(),
        masks["rule_id"].to_pylist(),
        masks["datetime_raw"].to_pylist(),
        masks["computable"].to_pylist(),
        masks["supported"].to_pylist(),
    ):
        if str(rid) != CS4["id"] or not ok or not supported:
            continue
        twin_sup[(str(src), str(tgt))].add(_norm_dt(dt))

    twin_pred: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    tp = pq.read_table(pred_path)
    fam = tp["model_family"].to_pylist() if "model_family" in tp.column_names else [FAM_SPLINE] * tp.num_rows
    for i in range(tp.num_rows):
        if str(fam[i]) != FAM_SPLINE:
            continue
        src = str(tp["source_plant_id"][i].as_py())
        tgt = str(tp["target_plant_id"][i].as_py())
        dt = _norm_dt(tp["datetime_raw"][i].as_py())
        twin_pred[(src, tgt, dt)] = (
            float(tp["y_true"][i].as_py()),
            float(tp["y_pred_transfer"][i].as_py()),
            float(tp["y_pred_local"][i].as_py()),
        )

    engines = {}
    models = {}
    q = float(CS4["q"])
    for pid in controls:
        train_dts = control_splits[pid]["train"]
        val_dts = control_splits[pid]["validation"]
        engines[pid] = build_engine(pid, train_dts, feat_by, q)
        Xtr, ytr, Xva, yva = [], [], [], []
        for dt in train_dts:
            vec = feat_by.get((pid, dt))
            yv = y_map.get((pid, dt))
            if vec is None or yv is None:
                continue
            Xtr.append(vec)
            ytr.append(yv)
        for dt in val_dts:
            vec = feat_by.get((pid, dt))
            yv = y_map.get((pid, dt))
            if vec is None or yv is None:
                continue
            Xva.append(vec)
            yva.append(yv)
        if len(Xtr) < 8 or len(Xva) < 4:
            models[pid] = None
            continue
        _params, _mae_val, est, _cand = _fit_select(
            FAM_SPLINE, SPLINE_GRID, np.vstack(Xtr), np.array(ytr), np.vstack(Xva), np.array(yva)
        )
        models[pid] = est

    comparisons: list[dict[str, Any]] = []
    for rec in elig_rows:
        tgt = rec["target_id"]
        ctrl = rec["control_id"]
        gid = rec["target_group_id"]
        twins_src = [p for p in members_by_group.get(gid, []) if p != tgt]
        test_dts = test_by.get(tgt) or []
        ctrl_sup = score_support(engines.get(ctrl) or {}, tgt, test_dts, feat_by)
        for src in twins_src:
            inter = twin_sup.get((src, tgt), set()) & ctrl_sup
            base = {
                "target_id": tgt,
                "twin_source_id": src,
                "control_id": ctrl,
                "target_state": rec["target_state"],
                "orientation": "CONTROL_MINUS_TWIN",
            }
            if not inter or models.get(ctrl) is None:
                comparisons.append(
                    {
                        **base,
                        "computable": False,
                        "reason": "empty_pairwise_intersection" if models.get(ctrl) is not None else "control_model_unfit",
                        "common_row_count": len(inter),
                        "twin_otg_common_rows": None,
                        "control_otg_common_rows": None,
                        "contrast_control_minus_twin": None,
                        "common_row_fingerprint": _row_fp(sorted(inter)) if inter else None,
                    }
                )
                continue
            y_true, y_twin, y_local, y_ctrl, used = [], [], [], [], []
            for dt in sorted(inter):
                trip = twin_pred.get((src, tgt, dt))
                vec = feat_by.get((tgt, dt))
                if trip is None or vec is None:
                    continue
                y_true.append(trip[0])
                y_twin.append(trip[1])
                y_local.append(trip[2])
                y_ctrl.append(float(models[ctrl].predict(vec.reshape(1, -1))[0]))
                used.append(dt)
            if len(y_true) < 1:
                comparisons.append(
                    {
                        **base,
                        "computable": False,
                        "reason": "missing_predictions_on_intersection",
                        "common_row_count": len(inter),
                        "twin_otg_common_rows": None,
                        "control_otg_common_rows": None,
                        "contrast_control_minus_twin": None,
                        "common_row_fingerprint": _row_fp(sorted(inter)),
                    }
                )
                continue
            yt = np.array(y_true)
            otg_twin = _mae(yt, np.array(y_twin)) - _mae(yt, np.array(y_local))
            otg_control = _mae(yt, np.array(y_ctrl)) - _mae(yt, np.array(y_local))
            comparisons.append(
                {
                    **base,
                    "computable": True,
                    "reason": "ok",
                    "common_row_count": len(y_true),
                    "twin_otg_common_rows": otg_twin,
                    "control_otg_common_rows": otg_control,
                    "contrast_control_minus_twin": otg_control - otg_twin,
                    "common_row_fingerprint": _row_fp(used),
                }
            )
    return {"status": "GO", "comparisons": comparisons}


def _flag(val: object) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def aggregate_path_groupby(comparisons: list[dict[str, Any]], targets: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = [r for r in comparisons if _flag(r.get("computable"))]
    by_t_src: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_t_ctrl: dict[str, set[str]] = defaultdict(set)
    n_comp: dict[str, int] = defaultdict(int)
    n_elig: dict[str, set[str]] = defaultdict(set)
    for r in comparisons:
        n_elig[str(r["target_id"])].add(str(r["control_id"]))
    for r in valid:
        t, s, c = str(r["target_id"]), str(r["twin_source_id"]), str(r["control_id"])
        by_t_src[(t, s)].append(float(r["contrast_control_minus_twin"]))
        by_t_ctrl[t].add(c)
        n_comp[t] += 1
    target_rows = []
    for t in targets:
        src_means = [sum(v) / len(v) for (tt, _s), v in sorted(by_t_src.items()) if tt == t]
        if not src_means:
            continue
        target_rows.append(
            {
                "target_id": t,
                "n_eligible_controls": len(n_elig.get(t, set())),
                "n_distinct_controls_contributing": len(by_t_ctrl.get(t, set())),
                "n_valid_comparisons": n_comp.get(t, 0),
                "target_mean_control_minus_twin": float(sum(src_means) / len(src_means)),
            }
        )
    return target_rows, _fleet(target_rows, targets, valid)


def aggregate_path_iterate(comparisons: list[dict[str, Any]], targets: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_rows = []
    valid_all = []
    for t in targets:
        recs = [r for r in comparisons if str(r["target_id"]) == t]
        elig_c = []
        for r in recs:
            cid = str(r["control_id"])
            if cid not in elig_c:
                elig_c.append(cid)
        sources = []
        for r in recs:
            sid = str(r["twin_source_id"])
            if sid not in sources:
                sources.append(sid)
        src_means = []
        n_valid = 0
        contrib = []
        for src in sources:
            acc = 0.0
            n = 0
            for r in recs:
                if str(r["twin_source_id"]) != src or not _flag(r.get("computable")):
                    continue
                acc += float(r["contrast_control_minus_twin"])
                n += 1
                cid = str(r["control_id"])
                if cid not in contrib:
                    contrib.append(cid)
                valid_all.append(r)
            if n:
                src_means.append(acc / n)
                n_valid += n
        if not src_means:
            continue
        target_rows.append(
            {
                "target_id": t,
                "n_eligible_controls": len(elig_c),
                "n_distinct_controls_contributing": len(contrib),
                "n_valid_comparisons": n_valid,
                "target_mean_control_minus_twin": float(sum(src_means) / len(src_means)),
            }
        )
    return target_rows, _fleet(target_rows, targets, valid_all)


def _fleet(target_rows: list[dict[str, Any]], targets: list[str], valid: list[dict[str, Any]]) -> dict[str, Any]:
    means = [float(r["target_mean_control_minus_twin"]) for r in target_rows]
    fleet = float(sum(means) / len(means)) if means else None
    return {
        "target_balanced_r8_control_minus_twin": fleet,
        "n_contributing_targets": len(target_rows),
        "n_primary_targets": len(targets),
        "n_distinct_contributing_controls": len({str(r["control_id"]) for r in valid}),
        "n_valid_comparisons": len(valid),
        "min_target_contrast": min(means) if means else None,
        "max_target_contrast": max(means) if means else None,
        "orientation": "CONTROL_MINUS_TWIN",
    }


def r8_close(a: list[dict[str, Any]], b: list[dict[str, Any]], fa: dict[str, Any], fb: dict[str, Any]) -> bool:
    if [r["target_id"] for r in a] != [r["target_id"] for r in b]:
        return False
    for x, y in zip(a, b):
        if x["n_valid_comparisons"] != y["n_valid_comparisons"]:
            return False
        if x["n_distinct_controls_contributing"] != y["n_distinct_controls_contributing"]:
            return False
        if not math.isclose(
            float(x["target_mean_control_minus_twin"]),
            float(y["target_mean_control_minus_twin"]),
            rel_tol=0.0,
            abs_tol=TOL,
        ):
            return False
    if fa["n_distinct_contributing_controls"] != fb["n_distinct_contributing_controls"]:
        return False
    va, vb = fa["target_balanced_r8_control_minus_twin"], fb["target_balanced_r8_control_minus_twin"]
    if va is None or vb is None:
        return va is None and vb is None
    return math.isclose(float(va), float(vb), rel_tol=0.0, abs_tol=TOL)
