"""RQ3 same-state matching, pairwise CS4 intersection, and control−twin OTG."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq

from config.protocol import CS4, FAM_SPLINE, PRIMARY_GROUP_IDS, STATE_FIELD
from src.io import dump_json, write_csv
from src.lib.matching import ranking_components, scientific_tuple, try_canonicalize
from src.lib.models import FEATURES, SPLINE_GRID, TARGET, _bool01, _fit_select, eligible_mask
from src.lib.robustness_struct import build_engine, score_support
from src.lib.split_support import FEATURE_INTERP, assign_post_eligibility_split, parse_ts
from src.lib.twins import FINAL_KEY


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _norm_dt(value: Any) -> str:
    ts = parse_ts(str(value))
    if ts is None:
        return str(value)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def load_meta(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _read_csv(path):
        pid = (row.get("id") or "").strip()
        if not pid:
            continue
        canon = {f: try_canonicalize(f, row.get(f, "")) for f in FINAL_KEY}
        out[pid] = {"state": (row.get(STATE_FIELD) or "").strip(), "canon": canon, "raw": row}
    return out


def select_same_state_controls(
    meta: dict[str, dict[str, Any]],
    members: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], dict[str, list[str]]]:
    group_of: dict[str, str] = {}
    members_by_group: dict[str, list[str]] = defaultdict(list)
    for row in members:
        pid, gid = row["plant_id"], row["group_id"]
        group_of[pid] = gid
        members_by_group[gid].append(pid)
    primary_plants = sorted(pid for pid, gid in group_of.items() if gid in PRIMARY_GROUP_IDS)
    all_ids = sorted(meta.keys())
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for tgt in primary_plants:
        tstate = meta.get(tgt, {}).get("state")
        tcanon = meta.get(tgt, {}).get("canon") or {}
        ranked = []
        for cand in all_ids:
            if cand == tgt:
                continue
            if group_of.get(cand) == group_of.get(tgt):
                continue
            cstate = meta.get(cand, {}).get("state")
            if not tstate or cstate != tstate:
                continue
            ccanon = meta.get(cand, {}).get("canon") or {}
            comp = ranking_components(tcanon, ccanon)
            if comp is None:
                continue
            ranked.append({"control_plant_id": cand, "tuple": scientific_tuple(comp)})
        ranked.sort(key=lambda r: r["tuple"])
        if not ranked:
            excluded.append({"target_plant_id": tgt, "reason": "NO_SAME_STATE_CONTROL"})
            continue
        top = ranked[0]["tuple"]
        ties = [r for r in ranked if r["tuple"] == top]
        if len(ties) != 1:
            excluded.append({"target_plant_id": tgt, "reason": "UNRESOLVED_TOP_TIE", "n_ties": len(ties)})
            continue
        selected.append(
            {
                "target_plant_id": tgt,
                "target_group_id": group_of[tgt],
                "target_state": tstate,
                "control_plant_id": ties[0]["control_plant_id"],
                "control_state": tstate,
            }
        )
    return selected, excluded, group_of, members_by_group


def _load_panel(path: Path, plants: list[str]):
    plist = ",".join("'" + p.replace("'", "''") + "'" for p in plants)
    con = duckdb.connect()
    table = con.execute(
        f"""
        SELECT plant_id, CAST(datetime_raw AS VARCHAR) AS datetime_raw, y_dc_normalized, coverage_dc, poa_irradiance_wm2,
               ghi_irradiance_wm2, gri_irradiance_wm2, panel_temperature_celsius,
               ambient_temperature_celsius, wind_speed_ms, dc_any_officially_interpolated,
               interpolated_keys_poa_irradiance_wm2, interpolated_keys_ghi_irradiance_wm2,
               interpolated_keys_gri_irradiance_wm2, interpolated_keys_panel_temperature_celsius,
               interpolated_keys_ambient_temperature_celsius, interpolated_keys_wind_speed_ms
        FROM read_parquet('{str(path).replace("'", "''")}')
        WHERE plant_id IN ({plist})
          AND y_dc_normalized IS NOT NULL
          AND poa_irradiance_wm2 > 50
          AND coverage_dc >= 0.80
          AND ghi_irradiance_wm2 IS NOT NULL
          AND gri_irradiance_wm2 IS NOT NULL
          AND panel_temperature_celsius IS NOT NULL
          AND ambient_temperature_celsius IS NOT NULL
          AND wind_speed_ms IS NOT NULL
          AND (dc_any_officially_interpolated IS NULL OR dc_any_officially_interpolated IS FALSE)
          AND (interpolated_keys_poa_irradiance_wm2 IS NULL OR interpolated_keys_poa_irradiance_wm2 IS FALSE)
          AND (interpolated_keys_ghi_irradiance_wm2 IS NULL OR interpolated_keys_ghi_irradiance_wm2 IS FALSE)
          AND (interpolated_keys_gri_irradiance_wm2 IS NULL OR interpolated_keys_gri_irradiance_wm2 IS FALSE)
          AND (interpolated_keys_panel_temperature_celsius IS NULL OR interpolated_keys_panel_temperature_celsius IS FALSE)
          AND (interpolated_keys_ambient_temperature_celsius IS NULL OR interpolated_keys_ambient_temperature_celsius IS FALSE)
          AND (interpolated_keys_wind_speed_ms IS NULL OR interpolated_keys_wind_speed_ms IS FALSE)
        """
    ).to_arrow_table()
    con.close()
    return table


def _splits_from_table(table, plants: list[str]) -> dict[str, dict[str, list[str]]]:
    by_plant: dict[str, list[int]] = {p: [] for p in plants}
    pids = table["plant_id"].to_pylist()
    for i, pid in enumerate(pids):
        if pid in by_plant:
            by_plant[str(pid)].append(i)
    dt_all = [_norm_dt(x) for x in table["datetime_raw"].to_pylist()]
    ts_all = [parse_ts(x) for x in dt_all]
    y_all = np.array(table[TARGET].to_pylist(), dtype=float)
    poa_all = np.array(table["poa_irradiance_wm2"].to_pylist(), dtype=float)
    cov_all = np.array(table["coverage_dc"].to_pylist(), dtype=float)
    feat_all = {f: np.array(table[f].to_pylist(), dtype=float) for f in FEATURES}
    dc_all = _bool01(table["dc_any_officially_interpolated"].to_pylist())
    flag_all = {f: _bool01(table[FEATURE_INTERP[f]].to_pylist()) for f in FEATURES}
    out: dict[str, dict[str, list[str]]] = {}
    sentinel = datetime.max.replace(tzinfo=timezone.utc)
    for pid in plants:
        ix = np.array(by_plant[pid], dtype=int)
        if ix.size == 0:
            out[pid] = {"train": [], "validation": [], "test": []}
            continue
        elig = eligible_mask(
            poa_all[ix],
            cov_all[ix],
            y_all[ix],
            {k: v[ix] for k, v in feat_all.items()},
            dc_all[ix],
            {k: v[ix] for k, v in flag_all.items()},
        )
        local_ts = [ts_all[int(j)] for j in ix]
        local_dt = [dt_all[int(j)] for j in ix]
        order = [int(k) for k in np.where(elig)[0]]
        order.sort(key=lambda k: (local_ts[k] if local_ts[k] is not None else sentinel, local_dt[k]))
        n = len(order)
        buckets = {"train": [], "validation": [], "test": []}
        for rank, k in enumerate(order):
            buckets[assign_post_eligibility_split(n, rank)].append(local_dt[k])
        out[pid] = buckets
    return out


def _feat_index(table) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], float]]:
    out: dict[tuple[str, str], np.ndarray] = {}
    pids = table["plant_id"].to_pylist()
    dts = [_norm_dt(x) for x in table["datetime_raw"].to_pylist()]
    cols = {f: table[f].to_pylist() for f in FEATURES}
    y = table[TARGET].to_pylist()
    y_map: dict[tuple[str, str], float] = {}
    for i, pid in enumerate(pids):
        dt = dts[i]
        vals = [cols[f][i] for f in FEATURES]
        if any(v is None for v in vals):
            continue
        key = (str(pid), dt)
        out[key] = np.array(vals, dtype=float)
        if y[i] is not None:
            y_map[key] = float(y[i])
    return out, y_map


def compute_pairwise_contrast(root: Path, selected: list[dict[str, Any]], members_by_group: dict[str, list[str]]) -> dict[str, Any]:
    panel_path = root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"
    if not panel_path.is_file():
        panel_path = root / "artifacts/panel/plant_panel.parquet"
    mask_path = root / "artifacts/s05/S05_SUPPORT_MASKS.parquet"
    pred_path = root / "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet"
    idx_path = root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet"
    if not (panel_path.is_file() and mask_path.is_file() and pred_path.is_file() and idx_path.is_file()):
        return {"status": "STOP", "reason": "missing_upstream_s04_s05_s06_or_panel"}

    targets = [r["target_plant_id"] for r in selected]
    controls = sorted({r["control_plant_id"] for r in selected})
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
    for gid, src, tgt, rid, dt, ok, supported in zip(
        masks["group_id"].to_pylist(),
        masks["source_plant_id"].to_pylist(),
        masks["target_plant_id"].to_pylist(),
        masks["rule_id"].to_pylist(),
        masks["datetime_raw"].to_pylist(),
        masks["computable"].to_pylist(),
        masks["supported"].to_pylist(),
    ):
        del gid
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
        _params, _mae_val, est, _cand = _fit_select(FAM_SPLINE, SPLINE_GRID, np.vstack(Xtr), np.array(ytr), np.vstack(Xva), np.array(yva))
        models[pid] = est

    source_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for rec in selected:
        tgt = rec["target_plant_id"]
        ctrl = rec["control_plant_id"]
        twins_src = [p for p in members_by_group[rec["target_group_id"]] if p != tgt]
        test_dts = test_by.get(tgt) or []
        ctrl_sup = score_support(engines.get(ctrl) or {}, tgt, test_dts, feat_by)
        deltas = []
        for src in twins_src:
            inter = twin_sup.get((src, tgt), set()) & ctrl_sup
            if not inter or models.get(ctrl) is None:
                source_rows.append(
                    {
                        "target_plant_id": tgt,
                        "twin_source_plant_id": src,
                        "control_source_plant_id": ctrl,
                        "computable": False,
                        "reason": "empty_pairwise_intersection" if models.get(ctrl) is not None else "control_model_unfit",
                        "n_pairwise": len(inter),
                    }
                )
                continue
            y_true, y_twin, y_local, y_ctrl = [], [], [], []
            for dt in sorted(inter):
                trip = twin_pred.get((src, tgt, dt))
                vec = feat_by.get((tgt, dt))
                if trip is None or vec is None:
                    continue
                y_true.append(trip[0])
                y_twin.append(trip[1])
                y_local.append(trip[2])
                y_ctrl.append(float(models[ctrl].predict(vec.reshape(1, -1))[0]))
            if len(y_true) < 1:
                source_rows.append(
                    {
                        "target_plant_id": tgt,
                        "twin_source_plant_id": src,
                        "control_source_plant_id": ctrl,
                        "computable": False,
                        "reason": "missing_predictions_on_intersection",
                        "n_pairwise": len(inter),
                    }
                )
                continue
            yt = np.array(y_true)
            otg_twin = _mae(yt, np.array(y_twin)) - _mae(yt, np.array(y_local))
            otg_control = _mae(yt, np.array(y_ctrl)) - _mae(yt, np.array(y_local))
            delta = otg_control - otg_twin
            source_rows.append(
                {
                    "target_plant_id": tgt,
                    "twin_source_plant_id": src,
                    "control_source_plant_id": ctrl,
                    "computable": True,
                    "n_pairwise": len(y_true),
                    "otg_twin": otg_twin,
                    "otg_control": otg_control,
                    "delta": delta,
                    "support_alignment": "B_PAIRWISE_INTERSECTION",
                }
            )
            deltas.append(delta)
        if deltas:
            target_rows.append(
                {
                    "target_plant_id": tgt,
                    "group_id": rec["target_group_id"],
                    "n_sources": len(deltas),
                    "delta_j": float(sum(deltas) / len(deltas)),
                }
            )

    d_fleet = float(sum(float(r["delta_j"]) for r in target_rows) / len(target_rows)) if target_rows else None
    return {
        "status": "GO",
        "source_rows": source_rows,
        "target_rows": target_rows,
        "D_fleet_RQ3": d_fleet,
        "n_computable_targets": len(target_rows),
        "selected_controls": controls,
        "cs4": CS4,
    }


def write_rq3_artifacts(out: Path, selected: list[dict[str, Any]], excluded: list[dict[str, Any]], contrast: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "mapping.csv", selected, list(selected[0].keys()) if selected else ["target_plant_id"])
    write_csv(out / "excluded.csv", excluded, list(excluded[0].keys()) if excluded else ["target_plant_id", "reason"])
    source_rows = contrast.get("source_rows") or []
    target_rows = contrast.get("target_rows") or []
    if source_rows:
        write_csv(out / "source_contrasts.csv", source_rows, list(source_rows[0].keys()))
    if target_rows:
        write_csv(out / "target_contrasts.csv", target_rows, list(target_rows[0].keys()))
    dump_json(
        out / "summary.json",
        {
            "n_rq3_targets": len(selected),
            "n_excluded": len(excluded),
            "excluded": excluded,
            "selected_controls": contrast.get("selected_controls"),
            "D_fleet_RQ3": contrast.get("D_fleet_RQ3"),
            "n_computable_targets": contrast.get("n_computable_targets"),
            "support_alignment": "B_PAIRWISE_INTERSECTION",
            "contrast_orientation": "CONTROL_MINUS_TWIN",
            "target_aggregation": "M1_EQUAL_WEIGHT_ARITHMETIC_MEAN",
            "fleet_synthesis": "F1_TARGET_BALANCED_ARITHMETIC_MEAN",
            "cs4": contrast.get("cs4"),
        },
    )
