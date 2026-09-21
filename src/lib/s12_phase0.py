"""S12 Phase 0 schemas and inventory (no scientific recompute)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.io import sha256_file
from src.lib.split_support import parse_ts

ROW_KEY_FIELDS = ("plant_id", "datetime_raw")
PREDICTION_RECORD_FIELDS = (
    "row_key_plant_id",
    "row_key_datetime_raw",
    "source_plant_id",
    "target_plant_id",
    "model_family",
    "support_rule",
    "y_true",
    "y_pred_source_or_transfer",
    "y_pred_local",
    "split",
)
SUPPORT_MASK_FIELDS = (
    "source_plant_id",
    "target_plant_id",
    "rule_id",
    "datetime_raw",
    "supported",
    "computable",
    "kth_distance",
    "threshold",
)
ROADMAP_PHASES = (
    ("P0", "Pipeline audit and readiness"),
    ("P1", "Within-plant null calibration"),
    ("P2", "OTG × support retention"),
    ("P3", "Adaptive scaling sensitivity"),
    ("P4", "Bias/dispersion + recalibration ladder"),
    ("P5", "Concentration/stability diagnostics"),
    ("P6", "Matched training-history duration"),
    ("P7", "RQ3 control-dependence analysis"),
    ("P8", "Pooled fleet baseline"),
    ("P9", "Environmental-distribution distance"),
)
RESERVED_LABELS = {"R7": "P2 support-retention robustness", "R8": "S11 all structurally executable same-state controls (do not reuse)", "R9": "P3 MAD/IQR scaling sensitivity", "R10": "P6 matched TRAIN duration"}

AUDIT_PATHS = (
    "config/protocol.py",
    "src/lib/split_support.py",
    "src/lib/models.py",
    "src/lib/robustness_struct.py",
    "artifacts/p02c/P02C_PLANT_PANEL.parquet",
    "artifacts/twins/members.csv",
    "data/raw/br_pvgen/BR-PVGen_metadata.csv",
    "artifacts/s04/S04_ANALYTIC_INDEX.parquet",
    "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv",
    "artifacts/s04/S04_MODEL_PROTOCOL.json",
    "artifacts/s04/S04_MODEL_MANIFEST.json",
    "artifacts/s04/S04_VALIDATION_PREDICTIONS.parquet",
    "artifacts/s04/S04_LOCAL_TEST_PREDICTIONS.parquet",
    "artifacts/s05/S05_SUPPORT_MASKS.parquet",
    "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet",
    "artifacts/s07/S07_OTG_DIRECTIONAL.csv",
    "artifacts/rq3/mapping.csv",
    "artifacts/rq3/source_contrasts.csv",
    "artifacts/s11/S11_PHASE4_CONSOLIDATION.json",
    "paper/main.tex",
    "paper/main.pdf",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parquet_columns(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return list(pq.read_schema(path).names)


def unique_plant_datetime(path: Path) -> dict[str, Any]:
    table = pq.read_table(path, columns=["plant_id", "datetime_raw"])
    n = table.num_rows
    keys = list(zip(table["plant_id"].to_pylist(), table["datetime_raw"].to_pylist()))
    return {"n_rows": n, "n_unique_plant_datetime": len(set(keys)), "unique": n == len(set(keys))}


def train_history(index_path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(index_path)
    by: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "validation": 0, "test": 0})
    for pid, dt, split in zip(table["plant_id"].to_pylist(), table["datetime_raw"].to_pylist(), table["split"].to_pylist()):
        counts[str(pid)][str(split)] = counts[str(pid)].get(str(split), 0) + 1
        if str(split) == "train":
            by[str(pid)].append(str(dt))
    rows = []
    for pid in sorted(by):
        dts = sorted(by[pid], key=lambda x: parse_ts(x) or x)
        first, last = parse_ts(dts[0]), parse_ts(dts[-1])
        if first and first.tzinfo is None:
            first = first.replace(tzinfo=timezone.utc)
        if last and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        dur = (last - first).total_seconds() / 86400.0 if first and last else None
        rows.append(
            {
                "plant_id": pid,
                "train_first_ts": dts[0],
                "train_last_ts": dts[-1],
                "train_calendar_days": dur,
                "n_train_index_rows": len(dts),
                "n_validation_index_rows": counts[pid].get("validation", 0),
                "n_test_index_rows": counts[pid].get("test", 0),
            }
        )
    return rows


def s06_split_membership(root: Path) -> dict[str, int]:
    s06 = pq.read_table(root / "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet", columns=["target_plant_id", "datetime_raw"])
    idx = pq.read_table(root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet")
    split = {(str(p), str(d)): str(s) for p, d, s in zip(idx["plant_id"].to_pylist(), idx["datetime_raw"].to_pylist(), idx["split"].to_pylist())}
    out: dict[str, int] = defaultdict(int)
    for p, d in zip(s06["target_plant_id"].to_pylist(), s06["datetime_raw"].to_pylist()):
        out[split.get((str(p), str(d)), "MISSING")] += 1
    return dict(out)


def inspect_files(root: Path) -> dict[str, Any]:
    rec = {}
    for rel in AUDIT_PATHS:
        path = root / rel
        rec[rel] = {
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    return rec


def build_inventory(root: Path) -> dict[str, Any]:
    s06_cols = parquet_columns(root / "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet")
    s05_cols = parquet_columns(root / "artifacts/s05/S05_SUPPORT_MASKS.parquet")
    s04_cols = parquet_columns(root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet")
    panel_cols = parquet_columns(root / "artifacts/p02c/P02C_PLANT_PANEL.parquet")
    panel_uniq = unique_plant_datetime(root / "artifacts/p02c/P02C_PLANT_PANEL.parquet")
    idx_uniq = unique_plant_datetime(root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet")
    s06_splits = s06_split_membership(root)
    train = train_history(root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet")
    g3_dir = root / "artifacts/s11/S11_A1_G3_DIRECTIONAL.csv"
    rq3_header = (root / "artifacts/rq3/source_contrasts.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    models_dir = root / "artifacts/s04/models"
    n_model_plants = len([p for p in models_dir.iterdir() if p.is_dir()]) if models_dir.is_dir() else 0
    phase4 = _json(root / "artifacts/s11/S11_PHASE4_CONSOLIDATION.json")
    decisions = {k: (phase4.get("decisions") or {}).get(k, {}).get("decision") for k in ("A1", "A2", "A3", "A4")}
    pred_fields_ok = all(c in s06_cols for c in ("datetime_raw", "y_true", "y_pred_transfer", "y_pred_local", "source_plant_id", "target_plant_id"))
    mask_ok = all(c in s05_cols for c in ("datetime_raw", "supported", "source_plant_id", "target_plant_id", "rule_id"))
    return {
        "row_key_convention": {
            "fields": list(ROW_KEY_FIELDS),
            "panel_uniqueness": panel_uniq,
            "index_uniqueness": idx_uniq,
            "note": "Do not mutate the accepted panel in P0; plant_id+datetime_raw is unique in P02C and S04 index.",
        },
        "predictions": {
            "primary_rq1_rq2": {
                "status": "PERSISTED_READY" if pred_fields_ok else "BLOCKED_MISSING_INFORMATION",
                "path": "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet",
                "columns": s06_cols,
                "split_membership": s06_splits,
                "note": "S06 is TEST-only transfer/local predictions under CS4. Not aggregate MAE.",
            },
            "primary_local_validation": {
                "status": "PERSISTED_READY",
                "path": "artifacts/s04/S04_VALIDATION_PREDICTIONS.parquet",
            },
            "primary_rq3_pairwise_rows": {
                "status": "DETERMINISTIC_RECONSTRUCTION_READY",
                "path": "artifacts/rq3/source_contrasts.csv",
                "columns": rq3_header,
                "note": "Persists n_pairwise and OTG, not intersection row IDs. Reconstruct from S05 twin masks ∩ control CS4 (control engines require scoring persisted models or a light refit).",
            },
            "g3_sensitivity": {
                "status": "REQUIRES_REFIT",
                "path": "artifacts/s11/S11_A1_G3_DIRECTIONAL.csv" if g3_dir.is_file() else None,
                "note": "G3 directional OTG persisted; no G3 row-level predictions/support parquet (POA_GHI cohort was computed separately).",
            },
        },
        "support_masks": {
            "primary_cs4": {
                "status": "PERSISTED_READY" if mask_ok else "BLOCKED_MISSING_INFORMATION",
                "path": "artifacts/s05/S05_SUPPORT_MASKS.parquet",
                "columns": s05_cols,
                "note": "Boolean supported plus datetime_raw per source-target-rule. n_supported in S07 is not the mask.",
            },
            "rq3_intersection": {
                "status": "DETERMINISTIC_RECONSTRUCTION_READY",
                "note": "Pairwise common rows = S05 CS4 (twin source→target) ∩ control support on target TEST; control support not a stored mask.",
            },
        },
        "splits": {
            "status": "PERSISTED_READY",
            "artifact": "artifacts/s04/S04_ANALYTIC_INDEX.parquet",
            "columns": s04_cols,
            "code": "src/lib/split_support.py:assign_post_eligibility_split",
            "config": "config/protocol.py SPLIT eligibility_then_chronological_60_20_20",
            "row_key": list(ROW_KEY_FIELDS),
        },
        "train_histories": {
            "status": "PERSISTED_READY",
            "source": "artifacts/s04/S04_ANALYTIC_INDEX.parquet train rows",
            "plants": train,
        },
        "hyperparameters": {
            "status": "PERSISTED_READY",
            "selection_csv": "artifacts/s04/S04_HYPERPARAMETER_SELECTION.csv",
            "protocol": "artifacts/s04/S04_MODEL_PROTOCOL.json",
            "joblib_plants": n_model_plants,
            "note": "Selected SplineRidge/HGB hyperparameters and TRAIN-only joblib fits for 14 primary plants.",
        },
        "scalers_support_thresholds": {
            "status": "DETERMINISTIC_RECONSTRUCTION_READY",
            "persisted_in_masks": ["kth_distance", "threshold"] if "threshold" in s05_cols else [],
            "reconstruct_from": "source TRAIN features in panel + S04 index + CS4 k,q in config.protocol",
            "note": "Source medians/IQRs are not a standalone artifact; S05 stores per-row distance and tau.",
        },
        "canonical_inputs": {
            "panel": "artifacts/p02c/P02C_PLANT_PANEL.parquet",
            "panel_columns_include_target": "y_dc_normalized" in panel_cols,
            "members": "artifacts/twins/members.csv",
            "metadata": "data/raw/br_pvgen/BR-PVGen_metadata.csv",
        },
        "s11_inclusion": {
            "decisions": decisions,
            "all_include_core": all(v == "INCLUDE_CORE" for v in decisions.values()),
            "inserted_in_main_tex": False,
        },
        "reserved_labels": RESERVED_LABELS,
        "schemas": {
            "row_key": list(ROW_KEY_FIELDS),
            "future_prediction_record": list(PREDICTION_RECORD_FIELDS),
            "future_support_mask": list(SUPPORT_MASK_FIELDS),
        },
    }


def readiness_rows(inv: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"phase": "P1", "dependency": "canonical TRAIN membership + timestamps", "status": inv["splits"]["status"], "source": inv["splits"]["artifact"], "refit": False, "row_level": True, "action": "split TRAIN chronologically into A/B and 7-day interleaved blocks from S04 index", "cost": "low"},
        {"phase": "P1", "dependency": "half-history and placebo SplineRidge fits", "status": "REQUIRES_REFIT", "source": "artifacts/s04/models + panel", "refit": True, "row_level": True, "action": "new comparable fits; do not overwrite S04 joblib", "cost": "high"},
        {"phase": "P2", "dependency": "directional OTG and support_retention", "status": "PERSISTED_READY", "source": "artifacts/s07/S07_OTG_DIRECTIONAL.csv", "refit": False, "row_level": False, "action": "associate |OTG| and relative OTG with retention; R7 = retention>=0.5 subset", "cost": "low"},
        {"phase": "P2", "dependency": "row-level CS4 masks (optional strata)", "status": inv["support_masks"]["primary_cs4"]["status"], "source": "artifacts/s05/S05_SUPPORT_MASKS.parquet", "refit": False, "row_level": True, "action": "use masks if stratum needs row recovery; transfer-level retention already in S07", "cost": "low"},
        {"phase": "P3", "dependency": "source TRAIN features + CS4 engine", "status": "REQUIRES_REFIT", "source": "panel + protocol CS4; S05 threshold reconstructs current tau", "refit": True, "row_level": True, "action": "R9 sensitivity: IQR→1.4826*MAD→omit degenerate feature; do not promote to primary", "cost": "medium"},
        {"phase": "P4", "dependency": "TEST transfer/local predictions", "status": inv["predictions"]["primary_rq1_rq2"]["status"], "source": "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet", "refit": False, "row_level": True, "action": "MBE on persisted TEST rows", "cost": "low"},
        {"phase": "P4", "dependency": "transfer predictions on target VALIDATION", "status": "REQUIRES_INSTRUMENTATION", "source": "S04 joblib + VAL index; S06 is TEST-only", "refit": False, "row_level": True, "action": "score persisted models on VAL only; never estimate recalibration on TEST", "cost": "medium"},
        {"phase": "P5", "dependency": "directional OTG matrix", "status": "PERSISTED_READY", "source": "artifacts/s07/S07_OTG_DIRECTIONAL.csv", "refit": False, "row_level": False, "action": "leave-one-plant-out by dropping that plant as source and target; G2 vs non-G2 descriptive", "cost": "trivial"},
        {"phase": "P6", "dependency": "TRAIN calendar spans", "status": inv["train_histories"]["status"], "source": "S04 analytic index", "refit": False, "row_level": False, "action": "choose common recent TRAIN window from inventory", "cost": "low"},
        {"phase": "P6", "dependency": "refit on truncated TRAIN (R10)", "status": "REQUIRES_REFIT", "source": "panel + truncated index", "refit": True, "row_level": True, "action": "new fits; keep R8 name for S11 control-set arm", "cost": "high"},
        {"phase": "P7", "dependency": "RQ3 mapping and source contrasts", "status": "PERSISTED_READY", "source": "artifacts/rq3/mapping.csv + source_contrasts.csv", "refit": False, "row_level": False, "action": "group by three matched controls; leave-one-control-out descriptive; no cluster-bootstrap as main output", "cost": "trivial"},
        {"phase": "P8", "dependency": "pooled source models pool(-j)/pool(-G_j)", "status": "REQUIRES_REFIT", "source": "panel + S04 splits", "refit": True, "row_level": True, "action": "plant-balanced or equal-plant training weights; evaluate on twin-compatible S05/S06 TEST rows", "cost": "high"},
        {"phase": "P8", "dependency": "twin-compatible evaluation rows", "status": inv["support_masks"]["primary_cs4"]["status"], "source": "S05 CS4 ∩ S06 TEST keys", "refit": False, "row_level": True, "action": "restrict pooled predictions to existing supported TEST rows", "cost": "low"},
        {"phase": "P9", "dependency": "source TRAIN features and supported target features", "status": "DETERMINISTIC_RECONSTRUCTION_READY", "source": "panel + S04 train + S05 supported TEST", "refit": False, "row_level": True, "action": "energy distance/MMD jointly with retention; exploratory only", "cost": "medium"},
    ]


def phase_verdicts(rows: list[dict[str, Any]]) -> dict[str, str]:
    by: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by[r["phase"]].append(r["status"])
    rank = ["BLOCKED_MISSING_INFORMATION", "REQUIRES_REFIT", "REQUIRES_INSTRUMENTATION", "DETERMINISTIC_RECONSTRUCTION_READY", "PERSISTED_READY"]
    out = {}
    for phase, sts in by.items():
        out[phase] = next(s for s in rank if s in sts)
    return out


def build_roadmap(inv: dict[str, Any], verdicts: dict[str, str]) -> dict[str, Any]:
    phases = []
    for pid, title in ROADMAP_PHASES:
        phases.append({"id": pid, "title": title, "readiness": "COMPLETE_THIS_STAGE" if pid == "P0" else verdicts.get(pid)})
    return {
        "id": "S12_IMPROVEMENT_ROADMAP",
        "version": 1,
        "phases": phases,
        "reserved_labels": RESERVED_LABELS,
        "s11_a1_a4_main_text": inv["s11_inclusion"],
        "post_p9": [
            "integrate accepted S12 with A1–A4 in one manuscript-editing phase",
            "compile and assess page count",
            "only then consider layout compression",
        ],
        "authoritative_until_revised": True,
    }


def render_roadmap_md(roadmap: dict[str, Any], inv: dict[str, Any], verdicts: dict[str, str]) -> str:
    lines = [
        "# S12 improvement roadmap (Phase 0)",
        "",
        "Authoritative sequencing for S12 P0–P9 unless a later orchestrator request revises it. No scientific headline was computed in P0.",
        "",
        "## Phases",
        "",
    ]
    for p in roadmap["phases"]:
        lines.append(f"- **{p['id']}** — {p['title']}  \n  Readiness: `{p['readiness']}`")
    lines += [
        "",
        "## Reserved robustness labels",
        "",
        "- **R7**: P2 support-retention arm (retention ≥ 0.5).",
        "- **R8**: S11 all structurally executable same-state non-twin controls (already completed; do not reuse for TRAIN duration).",
        "- **R9**: P3 adaptive scaling sensitivity (not automatically primary).",
        "- **R10**: P6 matched TRAIN duration.",
        "",
        "## S11 A1–A4",
        "",
        f"Future main-text inclusion (not yet in `paper/main.tex`): {inv['s11_inclusion']['decisions']}.",
        "",
        "After P9: one consolidated manuscript edit of accepted S12 + A1–A4, then compile, then page-count work.",
        "",
        "## Inventory snapshot",
        "",
        f"- Predictions (primary TEST transfer/local): `{inv['predictions']['primary_rq1_rq2']['status']}`.",
        f"- Support masks (CS4 datetime+supported): `{inv['support_masks']['primary_cs4']['status']}`.",
        f"- Splits: `{inv['splits']['status']}`.",
        f"- TRAIN histories: `{inv['train_histories']['status']}`.",
        f"- Hyperparameters/models: `{inv['hyperparameters']['status']}`.",
        f"- Scalers/tau: `{inv['scalers_support_thresholds']['status']}`.",
        f"- Row key plant_id+datetime_raw unique on panel: `{inv['row_key_convention']['panel_uniqueness']['unique']}`.",
        "",
        "## P1–P9 verdicts",
        "",
    ]
    for pid, _title in ROADMAP_PHASES[1:]:
        lines.append(f"- {pid}: `{verdicts.get(pid)}`")
    return "\n".join(lines) + "\n"
