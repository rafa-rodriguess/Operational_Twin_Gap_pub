"""Deterministic S09 scientific robustness execution helpers.

This module is deliberately not an orchestrator stage and exposes no ``run``.
Stage 37 is the sole entry point.  The implementation consumes the frozen
Stage 33--36 design, keeps every arm one-factor-at-a-time, and writes all SoT
changes through the StageResult returned by its caller.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from arch.bootstrap import optimal_block_length

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import dump_json, parse_ts, sha256_file, write_csv
from src.lib.rq3_mbb import B_BOOT, _day, _norm_dt, empty_pairwise_support_in_draw, mbb_calendar_draw
from src.lib.robustness_struct import FEATURES, MAP, build_engine, feat_index, reconstruct_splits, row_eligible, score_support
from src.seeds import seed_uint32
from src.lib.models import (
    FAM_HGB,
    FAM_SPLINE,
    HGB_GRID,
    SPLINE_GRID,
    TieError,
    _fit_select,
    metrics,
)

OUT = "artifacts/s09_scientific"
REPORT = "reports/scientific/S09_SCIENTIFIC_ROBUSTNESS.md"
ANSWER = "prompts/prompts_answers/S09_SCIENTIFIC_ROBUSTNESS_EXECUTION - ANSWER.md"
JOB = "S09_SCIENTIFIC_ROBUSTNESS_EXECUTION"
MSG_GO = "GO — S09 SCIENTIFIC ROBUSTNESS COMPLETE"
MSG_STOP = "STOP — S09 SCIENTIFIC ROBUSTNESS EXECUTION INCOMPLETE"

CROSS_ARM_DESIGN = "ONE_FACTOR_AT_A_TIME"
CI_LEVEL = 0.95
R7_EXECUTION = "OMITTED_BY_PREOUTCOME_STRUCTURAL_AUDIT"
HIGH_OUTPUT_CUTOFF = "NONE"
AUXILIARY_STATUS = "FROZEN_AUXILIARY_NOT_MANDATORY_S09"
NONCOMPUTABLE = "NON_COMPUTABLE_NO_RESCUE"
PAIRWISE_ALIGNMENT = "M_pair_ijc = M_twin_i_to_j ∩ M_control_c_to_j"
CONTRAST_ORIENTATION = "CONTROL_MINUS_TWIN"
FINAL_FIT_PARTITION = "TRAIN_only"
BLOCK_SELECTOR = "arch.bootstrap.optimal_block_length_PPW2009_circular"
PERCENTILE_INTERVAL = "two_sided_percentile_95"
NO_BOOTSTRAP_REFIT = True

ORIGINS = {
    "O1": ("2025-04-11T00:00:00+00:00", "2025-04-28T00:00:00+00:00", "B3"),
    "O2": ("2025-04-28T00:00:00+00:00", "2025-05-14T00:00:00+00:00", "B4"),
    "O3": ("2025-05-14T00:00:00+00:00", "2025-05-31T00:00:00+00:00", "B5"),
}
R6_RQ1_COUNT = 77
R6_RQ2_COUNT = 30
R6_RQ3_COUNT = 13
R6_TARGETS = [
    "PS_002", "PS_003", "PS_035", "PS_039", "PS_042", "PS_043", "PS_044",
    "PS_045", "PS_047", "PS_048", "PS_049", "PS_050", "PS_051",
]

# Source strings are intentionally explicit: they are also the machine-auditable
# declaration of every frozen arm and its sole changed dimension.
ARM_SPECS: dict[str, dict[str, Any]] = {
    "R1_HGB": {
        "target": "y_dc_normalized", "coverage_col": "coverage_dc", "coverage": 0.80,
        "poa_strict_gt": 50.0, "interpolation": "exclude_TRUE", "family": FAM_HGB,
        "support": {"id": "CS4", "k": 5, "q": 0.95}, "rebuild_split": False,
        "changed_dimension": "model_family",
    },
    "R2_AC": {
        "target": "y_ac_normalized", "coverage_col": "coverage_ac", "coverage": 0.80,
        "poa_strict_gt": 50.0, "interpolation": "exclude_TRUE", "family": FAM_SPLINE,
        "support": {"id": "CS4", "k": 5, "q": 0.95}, "rebuild_split": True,
        "changed_dimension": "target_and_corresponding_coverage",
    },
    "R3_POA20": {
        "target": "y_dc_normalized", "coverage_col": "coverage_dc", "coverage": 0.80,
        "poa_strict_gt": 20.0, "interpolation": "exclude_TRUE", "family": FAM_SPLINE,
        "support": {"id": "CS4", "k": 5, "q": 0.95}, "rebuild_split": True,
        "changed_dimension": "poa_threshold",
    },
    "R3_POA100": {
        "target": "y_dc_normalized", "coverage_col": "coverage_dc", "coverage": 0.80,
        "poa_strict_gt": 100.0, "interpolation": "exclude_TRUE", "family": FAM_SPLINE,
        "support": {"id": "CS4", "k": 5, "q": 0.95}, "rebuild_split": True,
        "changed_dimension": "poa_threshold",
    },
    "R4_CS3": {
        "target": "y_dc_normalized", "coverage_col": "coverage_dc", "coverage": 0.80,
        "poa_strict_gt": 50.0, "interpolation": "exclude_TRUE", "family": FAM_SPLINE,
        "support": {"id": "CS3", "k": 5, "q": 0.90}, "rebuild_split": False,
        "changed_dimension": "common_support",
    },
    "R5_KEEP_ALL_VALID": {
        "target": "y_dc_normalized", "coverage_col": "coverage_dc", "coverage": 0.80,
        "poa_strict_gt": 50.0, "interpolation": "keep_all_valid", "family": FAM_SPLINE,
        "support": {"id": "CS4", "k": 5, "q": 0.95}, "rebuild_split": True,
        "changed_dimension": "interpolation_policy",
    },
}
MANDATORY_SCIENTIFIC_ARMS = tuple(ARM_SPECS) + ("R6_TEMPORAL",)
PRIMARY_BLOCK_FACTORS = (("BL_PRIMARY_0P5", 0.5), ("BL_PRIMARY_1P0", 1.0), ("BL_PRIMARY_2P0", 2.0))
R6_BLOCK_FACTORS = (("0P5", 0.5), ("1P0", 1.0), ("2P0", 2.0))
FORBIDDEN_FACTORIAL_ARMS = ("HGB_AC", "HGB_POA20", "AC_CS3", "POA100_KEEP_ALL_VALID", "R6_CS3")

PANEL_COLS = [
    "plant_id", "datetime_raw", "y_dc_normalized", "y_ac_normalized",
    "coverage_dc", "coverage_ac", *FEATURES,
    "dc_any_officially_interpolated",
    "interpolated_keys_poa_irradiance_wm2",
    "interpolated_keys_ghi_irradiance_wm2",
    "interpolated_keys_gri_irradiance_wm2",
    "interpolated_keys_panel_temperature_celsius",
    "interpolated_keys_ambient_temperature_celsius",
    "interpolated_keys_wind_speed_ms",
]
FROZEN_MANIFESTS = (
    "artifacts/s09_preoutcome_robustness_audit/S09_ROBUSTNESS_AUDIT_MANIFEST.json",
    "artifacts/s09_controller_freeze/S09_CONTROLLER_FREEZE_MANIFEST.json",
    "artifacts/s09_r6_freeze/S09_R6_FREEZE_MANIFEST.json",
    "artifacts/s09_r6_balanced_panel/S09_R6_BALANCED_PANEL_MANIFEST.json",
    "artifacts/s08_scientific/S08_MANIFEST.json",
)


class S09Stop(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def effective_block_length(base: int, factor: float) -> int:
    """Frozen whole-calendar-day rule: scale, ceil, then enforce minimum one."""
    return max(1, int(math.ceil(float(factor) * int(base))))


def otg_values(transfer_mae: float, local_mae: float) -> tuple[float, float | None]:
    """Absolute and secondary relative OTG, with no epsilon denominator."""
    absolute = float(transfer_mae - local_mae)
    relative = None if local_mae == 0 else float(absolute / local_mae)
    return absolute, relative


def asymmetry(otg_i_to_j: float, otg_j_to_i: float) -> float:
    return float(otg_i_to_j - otg_j_to_i)


def pairwise_intersection(twin: set[str], control: set[str]) -> set[str]:
    return twin & control


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _art(root: Path, path: Path, rows: int | None = None, fp: str | None = None) -> ArtifactRecord:
    payload = path.read_bytes()
    return ArtifactRecord(_rel(root, path), hashlib.sha256(payload).hexdigest(), len(payload), rows, fp)


def _tab(rec: ArtifactRecord) -> dict[str, Any]:
    out = {"path": rec.path, "sha256": rec.sha256, "bytes": rec.bytes}
    if rec.row_count is not None:
        out["row_count"] = rec.row_count
    if rec.schema_fingerprint is not None:
        out["schema_fingerprint"] = rec.schema_fingerprint
    return out


def _csv_fp(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    return hashlib.sha256("|".join(header).encode()).hexdigest()


def _pq_fp(path: Path) -> str:
    return hashlib.sha256("|".join(pq.read_schema(path).names).encode()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _flag(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _write_parquet(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        table = pa.table({field: pa.array([], type=pa.string()) for field in fields})
    else:
        table = pa.Table.from_pylist([{field: row.get(field) for field in fields} for row in rows])
    pq.write_table(table, path)


def _manifest_snapshot(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for rel in FROZEN_MANIFESTS:
        manifest_path = root / rel
        if not manifest_path.is_file():
            raise S09Stop("S09_STOP_MISSING_FROZEN_MANIFEST", f"missing frozen manifest {rel}")
        snapshot[rel] = sha256_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("artifacts") or []
        if isinstance(records, dict):
            records = list(records.values())
        for rec in records:
            artifact_rel = rec.get("path")
            if not artifact_rel:
                continue
            path = root / artifact_rel
            if not path.is_file():
                raise S09Stop("S09_STOP_MISSING_FROZEN_ARTIFACT", f"missing frozen artifact {artifact_rel}")
            digest = sha256_file(path)
            if rec.get("sha256") and digest != rec["sha256"]:
                raise S09Stop("S09_STOP_FROZEN_HASH_MISMATCH", f"hash mismatch {artifact_rel}")
            snapshot[artifact_rel] = digest
    return snapshot


def _validate_preconditions(ctx: StageContext, validations: list[ValidationRecord]) -> dict[str, str]:
    sot = ctx.sot
    stages = sot.get("stages") or {}
    freeze = sot.get("scientific_freeze") or {}
    if stages.get("S09") is not None:
        raise S09Stop("S09_STOP_ALREADY_EXISTS", "stages.S09 already exists")
    if (stages.get("S08") or {}).get("status") != "GO":
        raise S09Stop("S09_STOP_S08_NOT_GO", "stages.S08.status is not GO")
    if ((sot.get("scientific_analysis") or {}).get("s08") or {}).get("bookkeeping_reconciled") is not True:
        raise S09Stop("S09_STOP_STAGE32_BOOKKEEPING", "Stage 32 bookkeeping reconciliation is invalid")
    if (stages.get("S09_R6_BALANCED_PANEL_RECONCILIATION") or {}).get("status") != "GO":
        raise S09Stop("S09_STOP_STAGE36_NOT_GO", "Stage 36 balanced-panel reconciliation is not GO")
    panel = freeze.get("s09_r6_balanced_panel") or {}
    controller = freeze.get("s09_robustness_controller_resolution") or {}
    operation = freeze.get("s09_robustness_operationalization") or {}
    if panel.get("status") != "COMPLETE" or panel.get("unresolved_item_count") != 0 or panel.get("s09_scientific_execution_authorized") is not True:
        raise S09Stop("S09_STOP_R6_PANEL_NOT_AUTHORIZED", "R6 balanced-panel freeze is incomplete")
    if controller.get("status") != "COMPLETE" or controller.get("unresolved_item_count") != 0 or controller.get("s09_scientific_execution_authorized") is not True:
        raise S09Stop("S09_STOP_CONTROLLER_NOT_AUTHORIZED", "S09 controller resolution is incomplete")
    if operation.get("unresolved_item_count") != 0 or operation.get("s09_scientific_execution_authorized") is not True or str(operation.get("resolved_by_stage")) != "36":
        raise S09Stop("S09_STOP_OPERATIONALIZATION_NOT_AUTHORIZED", "S09 operationalization was not resolved by Stage 36")
    if [panel.get("rq1_balanced_count"), panel.get("rq2_balanced_count"), panel.get("rq3_balanced_count")] != [77, 30, 13]:
        raise S09Stop("S09_STOP_R6_BALANCED_COUNTS", "R6 balanced counts are not 77/30/13")
    validations.extend([
        ValidationRecord("stage32_bookkeeping", True, None),
        ValidationRecord("stage36_go", True, None),
        ValidationRecord("s09_authorized", True, None),
        ValidationRecord("stages_s09_absent", True, None),
        ValidationRecord("r6_balanced_77_30_13", True, None),
    ])
    snap = _manifest_snapshot(ctx.root)
    validations.append(ValidationRecord("stage33_36_and_s08_hashes", True, str(len(snap))))
    return snap


def _load_incidence(root: Path) -> list[dict[str, Any]]:
    rows = []
    path = root / "artifacts/s03_inference_dependency_audit/S03_RQ3_CONTRAST_INCIDENCE.csv"
    for row in _read_csv(path):
        rows.append({
            "target": row["target_plant_id"],
            "group": row["reference_group_id"],
            "twins": list(json.loads(row["twin_source_plant_ids"])),
            "control": row["control_source_plant_id"],
        })
    return sorted(rows, key=lambda r: r["target"])


def _load_panel(root: Path, sot: dict[str, Any], plants: list[str]) -> list[dict[str, Any]]:
    panel_rec = ((sot.get("scientific_freeze") or {}).get("s02") or {}).get("canonical_panel") or {}
    path = root / (panel_rec.get("path") or "artifacts/p02c/P02C_PLANT_PANEL.parquet")
    if panel_rec.get("sha256") and sha256_file(path) != panel_rec["sha256"]:
        raise S09Stop("S09_STOP_PANEL_HASH", "canonical panel hash mismatch")
    plist = ",".join("'" + p.replace("'", "''") + "'" for p in plants)
    cols = ", ".join(PANEL_COLS)
    con = duckdb.connect()
    records = con.execute(
        f"SELECT {cols} FROM read_parquet('{str(path).replace(chr(39), chr(39) * 2)}') WHERE plant_id IN ({plist})"
    ).fetchall()
    con.close()
    out = []
    for values in records:
        row = dict(zip(PANEL_COLS, values))
        row["datetime_raw"] = _norm_dt(row["datetime_raw"])
        out.append(row)
    return out


def _dataset(raw: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[str]]], dict[tuple[str, str], np.ndarray], dict[tuple[str, str], float]]:
    rows = [
        row for row in raw
        if row_eligible(
            row,
            target=spec["target"],
            coverage_col=spec["coverage_col"],
            coverage_thr=spec["coverage"],
            poa=spec["poa_strict_gt"],
            interp=spec["interpolation"],
        )
    ]
    splits = reconstruct_splits(rows)
    feats = feat_index(rows)
    y = {(str(r["plant_id"]), str(r["datetime_raw"])): float(r[spec["target"]]) for r in rows}
    return rows, splits, feats, y


def _r6_splits(rows: list[dict[str, Any]], origin_id: str) -> dict[str, dict[str, list[str]]]:
    origin_text, end_text, _ = ORIGINS[origin_id]
    origin, end = parse_ts(origin_text), parse_ts(end_text)
    by: dict[str, list[tuple[Any, str]]] = defaultdict(list)
    for row in rows:
        dt = str(row["datetime_raw"])
        by[str(row["plant_id"])].append((parse_ts(dt), dt))
    out: dict[str, dict[str, list[str]]] = {}
    for pid, items in by.items():
        ordered = sorted(items, key=lambda x: (x[0] or datetime.max.replace(tzinfo=timezone.utc), x[1]))
        pre = [dt for ts, dt in ordered if ts is not None and ts < origin]
        n_train = math.floor(0.75 * len(pre))
        test = [dt for ts, dt in ordered if ts is not None and origin <= ts < end]
        out[pid] = {"train": pre[:n_train], "validation": pre[n_train:], "test": test}
    return out


def _matrix(pid: str, dts: list[str], feats: dict[tuple[str, str], np.ndarray], y: dict[tuple[str, str], float]) -> tuple[np.ndarray, np.ndarray]:
    try:
        return np.vstack([feats[(pid, dt)] for dt in dts]), np.asarray([y[(pid, dt)] for dt in dts], dtype=float)
    except (KeyError, ValueError) as exc:
        raise S09Stop("S09_STOP_MATRIX_RECONCILIATION", f"matrix reconstruction failed {pid}: {exc}") from exc


def _fit_models(
    root: Path,
    arm_key: str,
    plants: list[str],
    target_plants: set[str],
    splits: dict[str, dict[str, list[str]]],
    feats: dict[tuple[str, str], np.ndarray],
    y: dict[tuple[str, str], float],
    scientific_family: str,
    selection_rows: list[dict[str, Any]],
    model_manifest: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], Any], dict[tuple[str, str], dict[str, list[float]]]]:
    models: dict[tuple[str, str], Any] = {}
    val_daily: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pid in plants:
        parts = splits.get(pid) or {}
        train, validation = parts.get("train") or [], parts.get("validation") or []
        if not train or not validation:
            raise S09Stop("S09_STOP_EMPTY_TRAIN_VALIDATION", f"{arm_key} empty TRAIN/VALIDATION for {pid}")
        Xtr, ytr = _matrix(pid, train, feats, y)
        Xva, yva = _matrix(pid, validation, feats, y)
        families = (FAM_SPLINE, FAM_HGB) if pid in target_plants else (scientific_family,)
        for family in dict.fromkeys(families):
            grid = SPLINE_GRID if family == FAM_SPLINE else HGB_GRID
            params, best_mae, estimator, candidates = _fit_select(family, grid, Xtr, ytr, Xva, yva)
            selected = next(r["candidate_id"] for r in candidates if r["fit_status"] == "ok" and r["validation_mae"] == best_mae)
            for candidate in candidates:
                selection_rows.append({
                    "arm_id": arm_key, "plant_id": pid, "model_family": family,
                    "candidate_id": candidate["candidate_id"],
                    "hyperparameters": json.dumps(candidate["params"], sort_keys=True),
                    "train_row_count": len(train), "validation_row_count": len(validation),
                    "validation_mae": candidate["validation_mae"],
                    "selected": candidate["candidate_id"] == selected,
                    "fit_status": candidate["fit_status"], "selection_partition": "VALIDATION",
                    "final_fit_partition": "TRAIN_only",
                })
            pred = np.asarray(estimator.predict(Xva), dtype=float)
            for dt, yt, yp in zip(validation, yva, pred):
                val_daily[(pid, family)][_day(dt)].append(float(abs(yp - yt)))
            models[(pid, family)] = estimator
            dest = root / OUT / "models" / arm_key / pid / family / "full.joblib"
            dest.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(estimator, dest)
            model_manifest.append({
                "arm_id": arm_key, "plant_id": pid, "model_family": family,
                "path": _rel(root, dest), "sha256": sha256_file(dest), "bytes": dest.stat().st_size,
                "selected_candidate_id": selected, "final_fit_partition": "TRAIN_only",
            })
    return models, val_daily


def _upstream_models(root: Path, family: str) -> dict[tuple[str, str], Any]:
    out: dict[tuple[str, str], Any] = {}
    for pid in MAP:
        path = root / f"artifacts/s04/models/{pid}/{family}/full.joblib"
        if not path.is_file():
            raise S09Stop("S09_STOP_MISSING_PRIMARY_MODEL", f"missing {path}")
        out[(pid, family)] = joblib.load(path)
    for pid in sorted(set(MAP.values())):
        if family == FAM_SPLINE:
            path = root / f"artifacts/s08_scientific/models/{pid}/{family}/full.joblib"
            if not path.is_file():
                raise S09Stop("S09_STOP_MISSING_PRIMARY_CONTROL_MODEL", f"missing {path}")
            out[(pid, family)] = joblib.load(path)
    return out


def _record_reused_models(
    root: Path,
    arm_id: str,
    family: str,
    model_manifest: list[dict[str, Any]],
) -> None:
    for pid in MAP:
        path = root / f"artifacts/s04/models/{pid}/{family}/full.joblib"
        model_manifest.append({
            "arm_id": arm_id, "plant_id": pid, "model_family": family,
            "path": _rel(root, path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "selected_candidate_id": "REUSED_S04_SELECTED",
            "final_fit_partition": "TRAIN_only", "reused": True,
        })
    if family == FAM_SPLINE:
        for pid in sorted(set(MAP.values())):
            path = root / f"artifacts/s08_scientific/models/{pid}/{family}/full.joblib"
            model_manifest.append({
                "arm_id": arm_id, "plant_id": pid, "model_family": family,
                "path": _rel(root, path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
                "selected_candidate_id": "REUSED_S08_SELECTED",
                "final_fit_partition": "TRAIN_only", "reused": True,
            })


def _upstream_prediction_cache(root: Path) -> dict[tuple[str, str, str, str], tuple[float, float, float]]:
    """Load reusable S04/S06/S08 predictions keyed by exact row identity."""
    cache: dict[tuple[str, str, str, str], tuple[float, float, float]] = {}
    transfer = pq.read_table(root / "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet")
    for row in transfer.to_pylist():
        family = str(row["model_family"])
        source = str(row["source_plant_id"])
        target = str(row["target_plant_id"])
        dt = _norm_dt(row["datetime_raw"])
        cache[(family, source, target, dt)] = (
            float(row["y_true"]), float(row["y_pred_transfer"]), float(row["y_pred_local"])
        )
    control = pq.read_table(root / "artifacts/s08_scientific/S08_CONTROL_TRANSFER_PREDICTIONS.parquet")
    for row in control.to_pylist():
        source = str(row["control_source_plant_id"])
        target = str(row["target_plant_id"])
        dt = _norm_dt(row["datetime_raw"])
        cache[(FAM_SPLINE, source, target, dt)] = (
            float(row["y_true"]), float(row["y_pred_control"]), float(row["y_pred_local"])
        )
    return cache


def _primary_support(root: Path, rule: str) -> dict[tuple[str, str], set[str]]:
    table = pq.read_table(root / "artifacts/s05/S05_SUPPORT_MASKS.parquet")
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for i in range(table.num_rows):
        if str(table["rule_id"][i].as_py()) == rule and bool(table["computable"][i].as_py()) and bool(table["supported"][i].as_py()):
            out[(str(table["source_plant_id"][i].as_py()), str(table["target_plant_id"][i].as_py()))].add(_norm_dt(table["datetime_raw"][i].as_py()))
    return out


def _support_sets(
    arm_key: str,
    incidence: list[dict[str, Any]],
    splits: dict[str, dict[str, list[str]]],
    feats: dict[tuple[str, str], np.ndarray],
    q: float,
    support_rows: list[dict[str, Any]],
    support_summary: list[dict[str, Any]],
    reused_twin: dict[tuple[str, str], set[str]] | None = None,
) -> dict[tuple[str, str], set[str]]:
    sources = sorted({src for inc in incidence for src in inc["twins"]} | {inc["control"] for inc in incidence})
    engines = {pid: build_engine(pid, (splits.get(pid) or {}).get("train") or [], feats, q) for pid in sources}
    result: dict[tuple[str, str], set[str]] = {}
    for inc in incidence:
        tgt = inc["target"]
        test = (splits.get(tgt) or {}).get("test") or []
        for source, role in [(s, "twin") for s in inc["twins"]] + [(inc["control"], "control")]:
            key = (source, tgt)
            supported = reused_twin.get(key, set()) if reused_twin is not None and role == "twin" else score_support(engines[source], tgt, test, feats)
            computable = bool(engines[source].get("computable")) and bool(supported)
            reason = "" if computable else (engines[source].get("reason") or "empty_support_no_rescue")
            result[key] = set(supported)
            denominator = len(test)
            support_summary.append({
                "arm_id": arm_key, "source_plant_id": source, "target_plant_id": tgt,
                "source_role": role, "support_rule": "CS3" if q == 0.90 else "CS4",
                "k": 5, "q": q, "n_target_test": denominator, "n_supported": len(supported),
                "support_retention": (len(supported) / denominator) if denominator else None,
                "computable": computable, "reason": reason,
            })
            for dt in test:
                support_rows.append({
                    "arm_id": arm_key, "source_plant_id": source, "target_plant_id": tgt,
                    "source_role": role, "datetime_raw": dt,
                    "support_rule": "CS3" if q == 0.90 else "CS4",
                    "supported": dt in supported, "computable": computable, "reason": reason,
                })
    return result


def _predict(model: Any, target: str, dts: list[str], feats: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
    if not dts:
        return np.asarray([], dtype=float)
    try:
        X = np.vstack([feats[(target, dt)] for dt in dts])
    except KeyError as exc:
        raise S09Stop("S09_STOP_MISSING_TEST_FEATURE", f"missing target feature {exc}") from exc
    pred = np.asarray(model.predict(X), dtype=float)
    if not np.all(np.isfinite(pred)):
        raise S09Stop("S09_STOP_NONFINITE_PREDICTION", f"nonfinite prediction for {target}")
    return pred


def _metric_row(
    arm_key: str,
    origin_id: str,
    source: str,
    target: str,
    family: str,
    role: str,
    dts: list[str],
    y_true: np.ndarray,
    y_transfer: np.ndarray,
    y_local: np.ndarray,
    support_rule: str,
) -> dict[str, Any]:
    transfer = metrics(y_true, y_transfer)
    local = metrics(y_true, y_local)
    absolute, relative = otg_values(transfer["mae"], local["mae"])
    return {
        "arm_id": arm_key, "origin_id": origin_id, "source_plant_id": source,
        "target_plant_id": target, "source_role": role, "model_family": family,
        "support_rule": support_rule, "n_supported": len(dts),
        "n_supported_days": len({_day(dt) for dt in dts}),
        "transfer_mae": transfer["mae"], "transfer_rmse": transfer["rmse"],
        "transfer_r2": transfer["r2"], "transfer_bias": transfer["bias"],
        "local_same_rows_mae": local["mae"], "local_same_rows_rmse": local["rmse"],
        "local_same_rows_r2": local["r2"], "local_same_rows_bias": local["bias"],
        "OTG_abs": absolute, "OTG_rel": relative,
    }


def _plant_balanced_directional(rows: list[dict[str, Any]]) -> float | None:
    by_target: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_target[row["target_plant_id"]].append(float(row["OTG_abs"]))
    plant_means = [float(np.mean(values)) for values in by_target.values() if values]
    return float(np.mean(plant_means)) if plant_means else None


def _plant_balanced_asymmetry(rows: list[dict[str, Any]]) -> float | None:
    by_plant: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row["abs_asymmetry"])
        by_plant[row["plant_a"]].append(value)
        by_plant[row["plant_b"]].append(value)
    plant_means = [float(np.mean(values)) for values in by_plant.values() if values]
    return float(np.mean(plant_means)) if plant_means else None


def _block_lengths(
    arm_key: str,
    targets: list[str],
    val_daily: dict[tuple[str, str], dict[str, list[float]]],
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for target in targets:
        family_lengths = []
        for family in (FAM_SPLINE, FAM_HGB):
            daily = val_daily[(target, family)]
            if len(daily) < 8:
                raise S09Stop("S09_STOP_BLOCK_SELECTOR_SERIES_SHORT", f"{arm_key} {target} {family} validation series has {len(daily)} days")
            series = np.asarray([float(np.mean(daily[d])) for d in sorted(daily)], dtype=float)
            selected = optimal_block_length(series)
            raw = float(selected["circular"].iloc[0])
            whole = max(1, int(math.ceil(raw)))
            family_lengths.append(whole)
            rows.append({
                "arm_id": arm_key, "origin_id": arm_key.split("__", 1)[1] if "__" in arm_key else "",
                "target_plant_id": target, "model_family": family,
                "n_validation_days": len(daily), "n_validation_rows": sum(map(len, daily.values())),
                "raw_selector_circular": raw, "raw_selector_stationary": float(selected["stationary"].iloc[0]),
                "whole_day_block_length": whole, "L_j": None, "selector": BLOCK_SELECTOR,
                "selector_partition": "VALIDATION_only",
            })
        out[target] = max(family_lengths)
        for row in rows:
            if row["arm_id"] == arm_key and row["target_plant_id"] == target:
                row["L_j"] = out[target]
    return out


def _primary_lengths(root: Path) -> dict[str, int]:
    rows = _read_csv(root / "artifacts/s08_scientific/S08_BLOCK_LENGTH_SELECTION.csv")
    out: dict[str, int] = {}
    for row in rows:
        out[row["target_plant_id"]] = int(float(row["L_j"]))
    return out


def _bootstrap_target(
    arm_id: str,
    origin_id: str,
    target: str,
    source_payloads: list[dict[str, Any]],
    base_l: int,
    factors: list[tuple[str, float]],
    draw_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
) -> None:
    all_days = sorted({day for payload in source_payloads for day in payload["days"]})
    if not all_days or not source_payloads:
        raise S09Stop("S09_STOP_RQ3_TARGET_EMPTY", f"{arm_id} {origin_id} target {target} has no pairwise payload")
    point = float(np.mean([payload["delta"] for payload in source_payloads]))
    for factor_id, factor in factors:
        effective = effective_block_length(base_l, factor)
        seed_arm = arm_id if factor_id == "BASE" else factor_id
        unit_id = f"{origin_id}:{target}" if origin_id else target
        seed = seed_uint32("RQ3", "ridge_spline", unit_id, seed_arm)
        draws: list[float] = []
        for draw_id in range(B_BOOT):
            rng = np.random.default_rng(np.random.SeedSequence([seed, draw_id]))
            sampled = mbb_calendar_draw(all_days, effective, rng)
            sampled_set = set(sampled)
            source_deltas = []
            for payload in source_payloads:
                if empty_pairwise_support_in_draw(sampled_set, payload["days"]):
                    raise S09Stop(
                        "S09_STOP_BOOTSTRAP_DRAW_EMPTY_PAIRWISE_SUPPORT",
                        f"S09_STOP_BOOTSTRAP_DRAW_EMPTY_PAIRWISE_SUPPORT arm={arm_id} origin={origin_id} target={target} draw={draw_id}",
                    )
                control_parts = [payload["control_by_day"][d] for d in sampled if d in payload["control_by_day"]]
                twin_parts = [payload["twin_by_day"][d] for d in sampled if d in payload["twin_by_day"]]
                if not control_parts or not twin_parts:
                    raise S09Stop(
                        "S09_STOP_BOOTSTRAP_DRAW_EMPTY_PAIRWISE_SUPPORT",
                        f"S09_STOP_BOOTSTRAP_DRAW_EMPTY_PAIRWISE_SUPPORT arm={arm_id} origin={origin_id} target={target} draw={draw_id}",
                    )
                source_deltas.append(float(np.concatenate(control_parts).mean() - np.concatenate(twin_parts).mean()))
            value = float(np.mean(source_deltas))
            draws.append(value)
            draw_rows.append({
                "arm_id": arm_id, "origin_id": origin_id, "block_arm_id": factor_id,
                "target_plant_id": target, "draw_id": draw_id, "seed": int(seed),
                "L_j": int(base_l), "L_effective": effective, "Delta_j_draw": value,
                "source_count_denominator": len(source_payloads), "empty_pairwise_support": False,
            })
        lo, hi = np.percentile(np.asarray(draws), [2.5, 97.5], method="linear")
        inference_rows.append({
            "arm_id": arm_id, "origin_id": origin_id, "block_arm_id": factor_id,
            "target_plant_id": target, "Delta_j": point, "source_count_denominator": len(source_payloads),
            "L_j": int(base_l), "L_effective": effective, "B_boot": B_BOOT,
            "ci_level": CI_LEVEL, "ci_type": "two_sided_percentile",
            "ci_lower": float(lo), "ci_upper": float(hi), "no_refit": True,
        })


def _execute_arm(
    root: Path,
    arm_id: str,
    origin_id: str,
    incidence: list[dict[str, Any]],
    splits: dict[str, dict[str, list[str]]],
    feats: dict[tuple[str, str], np.ndarray],
    y: dict[tuple[str, str], float],
    models: dict[tuple[str, str], Any],
    family: str,
    support_rule: str,
    supports: dict[tuple[str, str], set[str]],
    l_of: dict[str, int],
    rq1_allowed: set[tuple[str, str]] | None,
    rq2_allowed: set[tuple[str, str]] | None,
    rq3_targets: set[str] | None,
    r6_source_allowed: set[tuple[str, str, str]] | None,
    prediction_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    directional_rows: list[dict[str, Any]],
    asymmetry_rows: list[dict[str, Any]],
    rq1_summary: list[dict[str, Any]],
    rq2_summary: list[dict[str, Any]],
    pair_mask_rows: list[dict[str, Any]],
    source_contrasts: list[dict[str, Any]],
    target_contrasts: list[dict[str, Any]],
    draw_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    fleet_rows: list[dict[str, Any]],
    reuse_cache: dict[tuple[str, str, str, str], tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    local_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
    direction_index: dict[tuple[str, str], dict[str, Any]] = {}
    pair_payloads: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def predictions(source: str, target: str, dts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if reuse_cache is not None and all((family, source, target, dt) in reuse_cache for dt in dts):
            values = [reuse_cache[(family, source, target, dt)] for dt in dts]
            return (
                np.asarray([v[0] for v in values], dtype=float),
                np.asarray([v[1] for v in values], dtype=float),
                np.asarray([v[2] for v in values], dtype=float),
            )
        key = (target, tuple(dts))
        if key not in local_cache:
            local_cache[key] = _predict(models[(target, family)], target, dts, feats)
        true = np.asarray([y[(target, dt)] for dt in dts], dtype=float)
        return true, _predict(models[(source, family)], target, dts, feats), local_cache[key]

    for inc in incidence:
        target = inc["target"]
        for twin in inc["twins"]:
            if rq1_allowed is not None and (twin, target) not in rq1_allowed:
                continue
            dts = sorted(supports.get((twin, target), set()))
            if not dts:
                continue
            yt, yp, yl = predictions(twin, target, dts)
            row = _metric_row(arm_id, origin_id, twin, target, family, "twin", dts, yt, yp, yl, support_rule)
            row["reference_group_id"] = inc["group"]
            metric_rows.append(row)
            direction_index[(twin, target)] = row
            directional_rows.append(dict(row))
            for dt, a, b, c in zip(dts, yt, yp, yl):
                prediction_rows.append({
                    "arm_id": arm_id, "origin_id": origin_id, "reference_group_id": inc["group"],
                    "source_plant_id": twin, "target_plant_id": target, "source_role": "twin",
                    "model_family": family, "datetime_raw": dt, "y_true": float(a),
                    "y_pred_transfer": float(b), "y_pred_local": float(c),
                    "abs_error_transfer": float(abs(b - a)), "abs_error_local": float(abs(c - a)),
                    "support_rule": support_rule,
                })
    directional_current = list(direction_index.values())
    rq1_summary.append({
        "arm_id": arm_id, "origin_id": origin_id, "computable_directional_denominator": len(directional_current),
        "plant_balanced_mean_OTG": _plant_balanced_directional(directional_current),
        "balanced_panel_required": rq1_allowed is not None,
    })

    seen_pairs = sorted({tuple(sorted((src, tgt))) for src, tgt in direction_index})
    pairs = []
    for a, b in seen_pairs:
        if rq2_allowed is not None and (a, b) not in rq2_allowed:
            continue
        ab, ba = direction_index.get((a, b)), direction_index.get((b, a))
        if ab is None or ba is None:
            continue
        signed = asymmetry(float(ab["OTG_abs"]), float(ba["OTG_abs"]))
        row = {
            "arm_id": arm_id, "origin_id": origin_id, "plant_a": a, "plant_b": b,
            "OTG_a_to_b": ab["OTG_abs"], "OTG_b_to_a": ba["OTG_abs"],
            "signed_asymmetry": signed, "abs_asymmetry": abs(signed),
        }
        pairs.append(row)
        asymmetry_rows.append(row)
    rq2_summary.append({
        "arm_id": arm_id, "origin_id": origin_id, "computable_pair_denominator": len(pairs),
        "plant_balanced_mean_abs_asymmetry": _plant_balanced_asymmetry(pairs),
        "lexical_unordered_pairs": True, "balanced_panel_required": rq2_allowed is not None,
    })

    for inc in incidence:
        target, control = inc["target"], inc["control"]
        if rq3_targets is not None and target not in rq3_targets:
            continue
        for twin in inc["twins"]:
            if r6_source_allowed is not None and (origin_id, target, twin) not in r6_source_allowed:
                continue
            pair = pairwise_intersection(supports.get((twin, target), set()), supports.get((control, target), set()))
            if not pair:
                source_contrasts.append({
                    "arm_id": arm_id, "origin_id": origin_id, "reference_group_id": inc["group"],
                    "twin_source_plant_id": twin, "target_plant_id": target,
                    "control_source_plant_id": control, "computable": False,
                    "reason": "empty_pairwise_intersection_no_rescue", "orientation": CONTRAST_ORIENTATION,
                })
                continue
            dts = sorted(pair)
            yt, y_twin, yl = predictions(twin, target, dts)
            _, y_control, _ = predictions(control, target, dts)
            twin_metric = _metric_row(arm_id, origin_id, twin, target, family, "twin_pairwise", dts, yt, y_twin, yl, support_rule)
            control_metric = _metric_row(arm_id, origin_id, control, target, family, "control_pairwise", dts, yt, y_control, yl, support_rule)
            metric_rows.extend((twin_metric, control_metric))
            delta = float(control_metric["OTG_abs"] - twin_metric["OTG_abs"])
            source_row = {
                "arm_id": arm_id, "origin_id": origin_id, "reference_group_id": inc["group"],
                "twin_source_plant_id": twin, "target_plant_id": target,
                "control_source_plant_id": control, "n_pairwise_rows": len(dts),
                "n_pairwise_days": len({_day(dt) for dt in dts}),
                "twin_transfer_mae": twin_metric["transfer_mae"],
                "control_transfer_mae": control_metric["transfer_mae"],
                "local_same_rows_mae": twin_metric["local_same_rows_mae"],
                "twin_OTG_abs": twin_metric["OTG_abs"],
                "control_OTG_abs": control_metric["OTG_abs"],
                "delta_ij": delta, "computable": True, "reason": "",
                "orientation": CONTRAST_ORIENTATION, "equal_source_weight": True,
            }
            source_contrasts.append(source_row)
            control_by_day: dict[str, list[float]] = defaultdict(list)
            twin_by_day: dict[str, list[float]] = defaultdict(list)
            for dt, truth, pred_t, pred_c, pred_l in zip(dts, yt, y_twin, y_control, yl):
                pair_mask_rows.append({
                    "arm_id": arm_id, "origin_id": origin_id, "target_plant_id": target,
                    "twin_source_plant_id": twin, "control_source_plant_id": control,
                    "datetime_raw": dt, "pairwise_supported": True,
                    "alignment": "PAIRWISE_INTERSECTION", "support_rule": support_rule,
                })
                prediction_rows.append({
                    "arm_id": arm_id, "origin_id": origin_id, "reference_group_id": inc["group"],
                    "source_plant_id": control, "target_plant_id": target, "source_role": "control",
                    "model_family": family, "datetime_raw": dt, "y_true": float(truth),
                    "y_pred_transfer": float(pred_c), "y_pred_local": float(pred_l),
                    "abs_error_transfer": float(abs(pred_c - truth)),
                    "abs_error_local": float(abs(pred_l - truth)), "support_rule": support_rule,
                })
                control_by_day[_day(dt)].append(float(abs(pred_c - truth)))
                twin_by_day[_day(dt)].append(float(abs(pred_t - truth)))
            pair_payloads[target].append({
                "delta": delta, "days": set(control_by_day),
                "control_by_day": {k: np.asarray(v) for k, v in control_by_day.items()},
                "twin_by_day": {k: np.asarray(v) for k, v in twin_by_day.items()},
            })

    targets_expected = sorted(rq3_targets if rq3_targets is not None else {inc["target"] for inc in incidence})
    current_targets = []
    for target in targets_expected:
        payloads = pair_payloads.get(target) or []
        if not payloads:
            raise S09Stop("S09_STOP_REQUIRED_RQ3_TARGET", f"{arm_id} {origin_id} has no computable RQ3 source for {target}")
        value = float(np.mean([p["delta"] for p in payloads]))
        row = {
            "arm_id": arm_id, "origin_id": origin_id, "target_plant_id": target,
            "Delta_j": value, "source_count_denominator": len(payloads),
            "aggregation": "equal_weight_arithmetic_mean", "orientation": CONTRAST_ORIENTATION,
        }
        current_targets.append(row)
        target_contrasts.append(row)
        factors = [(f"R6_{origin_id}_{suffix}", factor) for suffix, factor in R6_BLOCK_FACTORS] if origin_id else [("BASE", 1.0)]
        _bootstrap_target(arm_id, origin_id, target, payloads, l_of[target], factors, draw_rows, inference_rows)
    fleet_value = float(np.mean([r["Delta_j"] for r in current_targets]))
    fleet_rows.append({
        "arm_id": arm_id, "origin_id": origin_id, "D_fleet_RQ3": fleet_value,
        "target_denominator": len(current_targets), "descriptive_only": True,
        "fleet_ci": None, "fleet_p_value": None, "cross_origin_pooled": False,
    })
    return {
        "rq1_directional_denominator": len(directional_current),
        "rq1_plant_balanced_mean_otg": rq1_summary[-1]["plant_balanced_mean_OTG"],
        "rq2_pair_denominator": len(pairs),
        "rq2_plant_balanced_mean_abs_asymmetry": rq2_summary[-1]["plant_balanced_mean_abs_asymmetry"],
        "rq3_target_denominator": len(current_targets),
        "rq3_descriptive_target_balanced_control_minus_twin": fleet_value,
    }


def _primary_block_sensitivity(
    root: Path,
    draw_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_rows = [r for r in _read_csv(root / "artifacts/s08_scientific/S08_SOURCE_LEVEL_MATCHED_CONTRASTS.csv") if _flag(r["computable"])]
    pred = pq.read_table(root / "artifacts/s08_scientific/S08_CONTROL_TRANSFER_PREDICTIONS.parquet").to_pylist()
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in source_rows:
        key = (row["twin_source_plant_id"], row["target_plant_id"])
        indexed[key] = {
            "delta": float(row["delta_ij"]), "days": set(),
            "control_by_day": defaultdict(list), "twin_by_day": defaultdict(list),
        }
    for row in pred:
        key = (str(row["twin_source_plant_id"]), str(row["target_plant_id"]))
        if key not in indexed:
            continue
        day = _day(_norm_dt(row["datetime_raw"]))
        indexed[key]["days"].add(day)
        indexed[key]["control_by_day"][day].append(abs(float(row["y_pred_control"]) - float(row["y_true"])))
        indexed[key]["twin_by_day"][day].append(abs(float(row["y_pred_twin"]) - float(row["y_true"])))
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (source, target), payload in indexed.items():
        del source
        payload["control_by_day"] = {k: np.asarray(v) for k, v in payload["control_by_day"].items()}
        payload["twin_by_day"] = {k: np.asarray(v) for k, v in payload["twin_by_day"].items()}
        by_target[target].append(payload)
    l_of = _primary_lengths(root)
    start = len(inference_rows)
    for target in sorted(by_target):
        _bootstrap_target(
            "PRIMARY_REF", "", target, by_target[target], l_of[target],
            list(PRIMARY_BLOCK_FACTORS), draw_rows, inference_rows,
        )
    return inference_rows[start:]


def _registry_rows() -> list[dict[str, Any]]:
    rows = [{
        "arm_id": "PRIMARY_REF", "origin_id": "", "row_type": "reference",
        "executed": False, "changed_dimension": "NONE", "factorial": False,
        "status": "REFERENCE_ONLY",
    }]
    for arm in ARM_SPECS:
        rows.append({
            "arm_id": arm, "origin_id": "", "row_type": "scientific_robustness",
            "executed": True, "changed_dimension": ARM_SPECS[arm]["changed_dimension"],
            "factorial": False, "status": "EXECUTED",
        })
    for origin in ORIGINS:
        rows.append({
            "arm_id": "R6_TEMPORAL", "origin_id": origin, "row_type": "scientific_robustness",
            "executed": True, "changed_dimension": "temporal_origin",
            "factorial": False, "status": "EXECUTED",
        })
    for arm, _ in PRIMARY_BLOCK_FACTORS:
        rows.append({
            "arm_id": arm, "origin_id": "", "row_type": "inference_sensitivity",
            "executed": True, "changed_dimension": "mbb_block_length_only",
            "factorial": False, "status": "EXECUTED_NO_REFIT",
        })
    rows.append({
        "arm_id": "R7_OMITTED", "origin_id": "", "row_type": "omitted",
        "executed": False, "changed_dimension": "high_output_cutoff",
        "factorial": False, "status": R7_EXECUTION,
    })
    return rows


def _summaries_for_sot(
    arm_summaries: dict[str, dict[str, Any]],
    support_summary: list[dict[str, Any]],
    inference_path: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm_id, values in arm_summaries.items():
        if arm_id.startswith("R6_TEMPORAL__"):
            continue
        srs = [r["support_retention"] for r in support_summary if r["arm_id"] == arm_id and r["support_retention"] is not None]
        result[arm_id] = {
            **values,
            "support_retention_min": min(srs) if srs else None,
            "support_retention_median": float(np.median(srs)) if srs else None,
            "support_retention_max": max(srs) if srs else None,
            "target_level_inference": inference_path,
        }
    origins: dict[str, Any] = {}
    for origin, (start, end, block) in ORIGINS.items():
        values = arm_summaries[f"R6_TEMPORAL__{origin}"]
        srs = [r["support_retention"] for r in support_summary if r["arm_id"] == f"R6_TEMPORAL__{origin}" and r["support_retention"] is not None]
        origins[origin] = {
            "origin": start, "test_half_open_end": end, "test_block": block,
            "fixed_rq1_denominator": R6_RQ1_COUNT,
            "fixed_rq2_denominator": R6_RQ2_COUNT,
            "fixed_rq3_denominator": R6_RQ3_COUNT,
            **values,
            "support_retention_min": min(srs) if srs else None,
            "support_retention_median": float(np.median(srs)) if srs else None,
            "support_retention_max": max(srs) if srs else None,
            "target_level_inference": inference_path,
        }
    result["R6_TEMPORAL"] = {"origins": origins, "cross_origin_pooled_estimate": None}
    return result


def execute_scientific(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    try:
        return _execute(ctx, now, validations)
    except TieError as exc:
        return _stop(now, validations, "S09_STOP_VALIDATION_MAE_TIE", str(exc))
    except S09Stop as exc:
        return _stop(now, validations, exc.reason, exc.detail)
    except Exception as exc:  # noqa: BLE001
        return _stop(now, validations, "S09_STOP_RUNTIME_ERROR", f"{type(exc).__name__}: {exc}")


def _stop(now: str, validations: list[ValidationRecord], reason: str, detail: str) -> StageResult:
    return StageResult(
        status="STOP",
        message=f"{MSG_STOP}: {detail}",
        sot_patch={
            "stages": {"S09": {
                "kind": "scientific_analysis", "status": "STOP", "reason": reason,
                "operational_stage_id": "37", "job": JOB, "finished_at_utc": now,
                "answer": ANSWER,
            }},
            "scientific_analysis": {"s09": {
                "status": "STOP", "operational_stage_id": "37", "job": JOB,
                "reason": reason, "unresolved_item_count": 1,
            }},
        },
        artifacts=[],
        validations=validations,
    )


def _execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    root = ctx.root
    frozen_before = _validate_preconditions(ctx, validations)
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)

    scope = json.loads((root / "artifacts/s03_finalization/S03_FINAL_SCOPE.json").read_text(encoding="utf-8"))
    incidence = _load_incidence(root)
    targets = sorted(scope["primary_plants"])
    controls = sorted(set(MAP.values()))
    required = sorted(set(targets) | set(controls))
    if {r["reference_plant_id"]: r["control_plant_id"] for r in scope["matched_control_mapping"]} != MAP:
        raise S09Stop("S09_STOP_FIXED_CONTROL_MAPPING", "fixed controls do not reconcile")
    raw = _load_panel(root, ctx.sot, required)
    upstream_predictions = _upstream_prediction_cache(root)

    analytic_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    support_summary: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    directional_rows: list[dict[str, Any]] = []
    asymmetry_rows: list[dict[str, Any]] = []
    rq1_summary: list[dict[str, Any]] = []
    rq2_summary: list[dict[str, Any]] = []
    pair_mask_rows: list[dict[str, Any]] = []
    source_contrasts: list[dict[str, Any]] = []
    target_contrasts: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    fleet_rows: list[dict[str, Any]] = []
    arm_summaries: dict[str, dict[str, Any]] = {}

    primary_spec = ARM_SPECS["R1_HGB"]
    primary_rows, primary_splits, primary_feats, primary_y = _dataset(raw, primary_spec)
    for pid, parts in primary_splits.items():
        for split, dts in parts.items():
            analytic_rows.extend({"arm_id": "PRIMARY_REF", "origin_id": "", "plant_id": pid, "datetime_raw": dt, "split": split} for dt in dts)
    primary_lengths = _primary_lengths(root)
    primary_cs4 = _primary_support(root, "CS4")
    primary_cs3 = _primary_support(root, "CS3")

    for arm_id, spec in ARM_SPECS.items():
        if spec["rebuild_split"]:
            rows, splits, feats, y = _dataset(raw, spec)
            for pid, parts in splits.items():
                for split, dts in parts.items():
                    analytic_rows.extend({"arm_id": arm_id, "origin_id": "", "plant_id": pid, "datetime_raw": dt, "split": split} for dt in dts)
            models, val_daily = _fit_models(root, arm_id, required, set(targets), splits, feats, y, spec["family"], selection_rows, model_records)
            l_of = _block_lengths(arm_id, targets, val_daily, block_rows)
            supports = _support_sets(arm_id, incidence, splits, feats, spec["support"]["q"], support_rows, support_summary)
        elif arm_id == "R1_HGB":
            rows, splits, feats, y = primary_rows, primary_splits, primary_feats, primary_y
            models = _upstream_models(root, FAM_HGB)
            _record_reused_models(root, arm_id, FAM_HGB, model_records)
            control_models, _ = _fit_models(root, arm_id, controls, set(), splits, feats, y, FAM_HGB, selection_rows, model_records)
            models.update(control_models)
            supports = _support_sets(arm_id, incidence, splits, feats, 0.95, support_rows, support_summary, reused_twin=primary_cs4)
            l_of = primary_lengths
        else:
            rows, splits, feats, y = primary_rows, primary_splits, primary_feats, primary_y
            models = _upstream_models(root, FAM_SPLINE)
            _record_reused_models(root, arm_id, FAM_SPLINE, model_records)
            supports = _support_sets(arm_id, incidence, splits, feats, 0.90, support_rows, support_summary, reused_twin=primary_cs3)
            l_of = primary_lengths
        del rows
        arm_summaries[arm_id] = _execute_arm(
            root, arm_id, "", incidence, splits, feats, y, models, spec["family"], spec["support"]["id"],
            supports, l_of, None, None, None, None,
            prediction_rows, metric_rows, directional_rows, asymmetry_rows, rq1_summary, rq2_summary,
            pair_mask_rows, source_contrasts, target_contrasts, draw_rows, inference_rows, fleet_rows,
            upstream_predictions if arm_id in {"R1_HGB", "R4_CS3"} else None,
        )

    r6_spec = {
        "target": "y_dc_normalized", "coverage_col": "coverage_dc", "coverage": 0.80,
        "poa_strict_gt": 50.0, "interpolation": "exclude_TRUE",
    }
    r6_rows, _, r6_feats, r6_y = _dataset(raw, r6_spec)
    rq1_allowed = {
        (r["source_plant_id"], r["target_plant_id"])
        for r in _read_csv(root / "artifacts/s09_r6_balanced_panel/S09_R6_RQ1_BALANCED_PANEL.csv")
        if _flag(r["in_balanced_panel"])
    }
    rq2_allowed = {
        tuple(sorted((r["plant_a"], r["plant_b"])))
        for r in _read_csv(root / "artifacts/s09_r6_balanced_panel/S09_R6_RQ2_BALANCED_PANEL.csv")
        if _flag(r["in_balanced_panel"])
    }
    r6_source_allowed = {
        (r["origin_id"], r["target_plant_id"], r["twin_source_plant_id"])
        for r in _read_csv(root / "artifacts/s09_r6_balanced_panel/S09_R6_RQ3_SOURCE_COUNTS_BY_ORIGIN.csv")
        if _flag(r["computable"])
    }
    if len(rq1_allowed) != 77 or len(rq2_allowed) != 30 or len(R6_TARGETS) != 13:
        raise S09Stop("S09_STOP_R6_BALANCED_COUNTS", "balanced-panel files do not yield 77/30/13")
    for origin_id in ORIGINS:
        arm_key = f"R6_TEMPORAL__{origin_id}"
        splits = _r6_splits(r6_rows, origin_id)
        for pid, parts in splits.items():
            for split, dts in parts.items():
                analytic_rows.extend({"arm_id": "R6_TEMPORAL", "origin_id": origin_id, "plant_id": pid, "datetime_raw": dt, "split": split} for dt in dts)
        models, val_daily = _fit_models(root, arm_key, required, set(R6_TARGETS), splits, r6_feats, r6_y, FAM_SPLINE, selection_rows, model_records)
        l_of = _block_lengths(arm_key, R6_TARGETS, val_daily, block_rows)
        supports = _support_sets(arm_key, incidence, splits, r6_feats, 0.95, support_rows, support_summary)
        arm_summaries[arm_key] = _execute_arm(
            root, "R6_TEMPORAL", origin_id, incidence, splits, r6_feats, r6_y, models, FAM_SPLINE,
            "CS4", supports, l_of, rq1_allowed, rq2_allowed, set(R6_TARGETS), r6_source_allowed,
            prediction_rows, metric_rows, directional_rows, asymmetry_rows, rq1_summary, rq2_summary,
            pair_mask_rows, source_contrasts, target_contrasts, draw_rows, inference_rows, fleet_rows,
            None,
        )
        if arm_summaries[arm_key]["rq1_directional_denominator"] != 77 or arm_summaries[arm_key]["rq2_pair_denominator"] != 30 or arm_summaries[arm_key]["rq3_target_denominator"] != 13:
            raise S09Stop("S09_STOP_R6_EXECUTED_DENOMINATORS", f"{origin_id} did not execute 77/30/13")

    primary_sensitivity = _primary_block_sensitivity(root, draw_rows, inference_rows)
    registry = _registry_rows()
    if any(row["factorial"] for row in registry) or any(row["arm_id"] in FORBIDDEN_FACTORIAL_ARMS for row in registry):
        raise S09Stop("S09_STOP_FACTORIAL_ARM", "factorial arm was created")

    protocol = {
        "cross_arm_design": CROSS_ARM_DESIGN,
        "arm_specs": ARM_SPECS,
        "mandatory_scientific_arms": list(MANDATORY_SCIENTIFIC_ARMS),
        "forbidden_factorial_arms": list(FORBIDDEN_FACTORIAL_ARMS),
        "features": list(FEATURES), "plant_id_feature": False, "state_feature": False,
        "tracker_albedo_index": "EXCLUDED", "imputation": False, "target_adaptation": False,
        "split": "eligibility_then_chronological_floor_60_20_20",
        "r6_split": "all_primary_eligible_pre_origin; N_train=floor(0.75*N_pre); remainder_VALIDATION; TEST_half_open",
        "final_fit_partition": FINAL_FIT_PARTITION,
        "spline_grid": SPLINE_GRID, "hgb_grid": HGB_GRID,
        "selection": "minimum local VALIDATION MAE; exact tie STOP",
        "support_scaling": "source TRAIN median/IQR only", "support_metric": "Euclidean",
        "rq1_formula": "OTG_abs = transfer_MAE - local_same_rows_MAE",
        "rq1_relative": "only_if_local_MAE_nonzero; no_epsilon",
        "rq2_formula": "A_ij = OTG_i_to_j - OTG_j_to_i",
        "rq3_alignment": PAIRWISE_ALIGNMENT, "rq3_orientation": CONTRAST_ORIENTATION,
        "rq3_target_aggregation": "equal_weight_mean_of_computable_source_deltas",
        "rq3_fleet": "descriptive_target_balanced_mean; no_fleet_CI_or_p_value",
        "B_boot": B_BOOT, "ci": PERCENTILE_INTERVAL, "bootstrap_refit": False,
        "block_selector": "arch.bootstrap.optimal_block_length circular column; Patton-Politis-White correction; VALIDATION only",
        "block_effective_rule": "max(1, ceil(a * L_j))",
        "seed_rule": "Operational_Twin_Gap|proposal_v1.3|{estimand}|{model_family}|{unit_id}|{arm_id}",
        "r6_seed_unit_includes_origin": True,
        "primary_block_factors": [0.5, 1.0, 2.0],
        "r1_r5_crossed_with_block_sensitivity": False,
        "r7_execution": R7_EXECUTION, "high_output_cutoff": HIGH_OUTPUT_CUTOFF,
        "auxiliary_feature_sensitivity_status": AUXILIARY_STATUS,
        "confirmatory_permutation": False, "cross_origin_inference": "NONE",
        "cross_origin_synthesis": "NO_POOLED_TEMPORAL_ESTIMAND",
        "secondary_metrics": ["RMSE_descriptive", "R2_descriptive", "mean_bias_descriptive"],
        "scientific_outcomes_control_gate": False,
        "python_version": sys.version, "numpy_version": np.__version__,
    }
    reconciliation = {
        "mandatory_scientific_arms_complete": True, "unresolved_item_count": 0,
        "ofat": True, "factorial_arms": False, "r7_omitted": True,
        "poa_ghi_promoted": False, "primary_analysis_denominators_unchanged": True,
        "r6_balanced_counts": {"RQ1": 77, "RQ2": 30, "RQ3": 13},
        "ps046_exclusion": "R6_RQ3_ONLY__NO_ELIGIBLE_O3_TEST_ROWS",
        "ps048_source_policy": NONCOMPUTABLE,
        "fixed_controls_unchanged": True, "target_adaptation": False,
        "fleet_ci": False, "fleet_p_value": False, "confirmatory_permutation": False,
        "cross_origin_inference": False, "empty_bootstrap_draws": 0,
        "historical_s08_and_stage33_36_unchanged": True,
    }
    robustness_rows = []
    for key, value in arm_summaries.items():
        robustness_rows.append({"arm_id": key, **value})
    for row in primary_sensitivity:
        robustness_rows.append({
            "arm_id": row["block_arm_id"], "rq1_directional_denominator": None,
            "rq1_plant_balanced_mean_otg": None, "rq2_pair_denominator": None,
            "rq2_plant_balanced_mean_abs_asymmetry": None,
            "rq3_target_denominator": 14, "rq3_descriptive_target_balanced_control_minus_twin": None,
        })

    paths: list[tuple[str, str, list[dict[str, Any]], list[str] | None]] = [
        ("csv", "S09_ARM_EXECUTION_REGISTRY.csv", registry, list(registry[0])),
        ("pq", "S09_ANALYTIC_INDEX.parquet", analytic_rows, ["arm_id", "origin_id", "plant_id", "datetime_raw", "split"]),
        ("csv", "S09_HYPERPARAMETER_SELECTION.csv", selection_rows, list(selection_rows[0])),
        ("pq", "S09_SUPPORT_MASKS.parquet", support_rows, list(support_rows[0])),
        ("csv", "S09_SUPPORT_SUMMARY.csv", support_summary, list(support_summary[0])),
        ("pq", "S09_TRANSFER_PREDICTIONS.parquet", prediction_rows, list(prediction_rows[0])),
        ("csv", "S09_TRANSFER_METRICS.csv", metric_rows, list(metric_rows[0])),
        ("csv", "S09_OTG_DIRECTIONAL.csv", directional_rows, list(directional_rows[0])),
        ("csv", "S09_ASYMMETRY.csv", asymmetry_rows, list(asymmetry_rows[0])),
        ("csv", "S09_RQ1_SUMMARY.csv", rq1_summary, list(rq1_summary[0])),
        ("csv", "S09_RQ2_SUMMARY.csv", rq2_summary, list(rq2_summary[0])),
        ("pq", "S09_RQ3_PAIRWISE_SUPPORT_MASKS.parquet", pair_mask_rows, list(pair_mask_rows[0])),
        ("csv", "S09_RQ3_SOURCE_LEVEL_CONTRASTS.csv", source_contrasts, sorted({k for r in source_contrasts for k in r})),
        ("csv", "S09_RQ3_TARGET_LEVEL_CONTRASTS.csv", target_contrasts, list(target_contrasts[0])),
        ("csv", "S09_RQ3_BLOCK_LENGTH_SELECTION.csv", block_rows, list(block_rows[0])),
        ("pq", "S09_RQ3_BOOTSTRAP_DRAWS.parquet", draw_rows, list(draw_rows[0])),
        ("csv", "S09_RQ3_TARGET_LEVEL_INFERENCE.csv", inference_rows, list(inference_rows[0])),
        ("csv", "S09_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv", fleet_rows, list(fleet_rows[0])),
        ("csv", "S09_PRIMARY_BLOCK_LENGTH_SENSITIVITY.csv", primary_sensitivity, list(primary_sensitivity[0])),
        ("csv", "S09_R6_ORIGIN_SUMMARY.csv", [r for r in robustness_rows if str(r["arm_id"]).startswith("R6_TEMPORAL")], list(robustness_rows[0])),
        ("csv", "S09_ROBUSTNESS_SUMMARY.csv", robustness_rows, list(robustness_rows[0])),
    ]
    dump_json(out / "S09_PROTOCOL.json", protocol)
    dump_json(out / "S09_MODEL_MANIFEST.json", {"models": model_records})
    dump_json(out / "S09_RECONCILIATION.json", reconciliation)
    for kind, filename, rows, fields in paths:
        path = out / filename
        if kind == "pq":
            _write_parquet(path, rows, fields or [])
        else:
            write_csv(path, rows, fields or [])

    report_path = root / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(robustness_rows, support_summary), encoding="utf-8")

    core_files = [
        out / "S09_PROTOCOL.json", out / "S09_MODEL_MANIFEST.json", out / "S09_RECONCILIATION.json",
        *[out / filename for _, filename, _, _ in paths], report_path,
    ]
    recs: list[ArtifactRecord] = []
    row_count = {filename: len(rows) for _, filename, rows, _ in paths}
    for path in core_files:
        rows = row_count.get(path.name)
        fp = _pq_fp(path) if path.suffix == ".parquet" else (_csv_fp(path) if path.suffix == ".csv" else None)
        recs.append(_art(root, path, rows, fp))
    model_recs = [_art(root, root / rec["path"]) for rec in model_records]
    dump_json(out / "S09_MANIFEST.json", {"job": JOB, "artifacts": [_tab(r) for r in recs + model_recs]})
    manifest_rec = _art(root, out / "S09_MANIFEST.json")
    recs.extend(model_recs)
    recs.append(manifest_rec)

    for rel, digest in frozen_before.items():
        if sha256_file(root / rel) != digest:
            raise S09Stop("S09_STOP_HISTORICAL_ARTIFACT_MUTATED", f"historical artifact mutated {rel}")
    validations.extend([
        ValidationRecord("one_factor_at_a_time", True, None),
        ValidationRecord("no_factorial_arms", True, None),
        ValidationRecord("r7_omitted", True, R7_EXECUTION),
        ValidationRecord("historical_s08_unchanged", True, None),
        ValidationRecord("historical_stage33_36_unchanged", True, None),
        ValidationRecord("mandatory_complete", True, "unresolved=0"),
    ])

    artifacts_map = {Path(r.path).name: _tab(r) for r in recs if r.path.startswith(OUT + "/") and "/models/" not in r.path}
    inference_rel = f"{OUT}/S09_RQ3_TARGET_LEVEL_INFERENCE.csv"
    sot_arms = _summaries_for_sot(arm_summaries, support_summary, inference_rel)
    patch = {
        "stages": {"S09": {
            "kind": "scientific_analysis", "status": "GO",
            "reason": "S09_SCIENTIFIC_ROBUSTNESS_COMPLETE",
            "operational_stage_id": "37", "job": JOB, "finished_at_utc": now,
            "answer": ANSWER, "report": _tab(next(r for r in recs if r.path == REPORT)),
            "manifest": _tab(manifest_rec),
        }},
        "scientific_analysis": {"s09": {
            "status": "GO", "operational_stage_id": "37", "job": JOB,
            "cross_arm_design": CROSS_ARM_DESIGN,
            "mandatory_scientific_arms_complete": True,
            "r7_execution": R7_EXECUTION,
            "auxiliary_feature_sensitivity_status": AUXILIARY_STATUS,
            "primary_analysis_denominators_unchanged": True,
            "results_do_not_control_gate": True,
            "arms": sot_arms,
            "primary_block_length_sensitivity": {
                "factors": [0.5, 1.0, 2.0], "no_refit": True,
                "artifact": artifacts_map["S09_PRIMARY_BLOCK_LENGTH_SENSITIVITY.csv"],
            },
            "artifacts": artifacts_map, "manifest": _tab(manifest_rec),
            "report": _tab(next(r for r in recs if r.path == REPORT)),
            "unresolved_item_count": 0,
        }},
    }
    return StageResult("GO", MSG_GO, patch, recs, validations)


def _report(rows: list[dict[str, Any]], support: list[dict[str, Any]]) -> str:
    lines = [
        "# S09 — Scientific robustness",
        "",
        "Pre-specified one-factor-at-a-time robustness execution. Values below are generated from the S09 artifacts. Results do not control the execution gate.",
        "",
        "| arm/origin | RQ1 n | RQ1 plant-balanced OTG | RQ2 n | RQ2 plant-balanced | RQ3 n | RQ3 control-minus-twin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("rq1_directional_denominator") is None:
            continue
        lines.append(
            f"| {row['arm_id']} | {row['rq1_directional_denominator']} | {row['rq1_plant_balanced_mean_otg']} | "
            f"{row['rq2_pair_denominator']} | {row['rq2_plant_balanced_mean_abs_asymmetry']} | "
            f"{row['rq3_target_denominator']} | {row['rq3_descriptive_target_balanced_control_minus_twin']} |"
        )
    srs = [r["support_retention"] for r in support if r["support_retention"] is not None]
    lines.extend([
        "",
        f"Support retention across computable source-target rows: min={min(srs)}, median={float(np.median(srs))}, max={max(srs)}.",
        "",
        "R6 is reported separately for O1/O2/O3 on fixed denominators 77/30/13. No origins are pooled. R7 is omitted by the pre-outcome structural audit. No POA_GHI arm is promoted.",
        "",
        "Target-level temporal MBB uses B=5000, two-sided 95% percentile intervals, validation-only PPW block selection, deterministic SHA-256 seeds, and no refit in draws. Fleet summaries are descriptive and have no CI or p-value.",
        "",
        f"**{MSG_GO}**",
        "",
    ])
    return "\n".join(lines)
