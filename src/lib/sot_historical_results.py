"""Recover historical table-02 / RQ3 C8 / primary retention into the SoT (no refit)."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from config.protocol import FAM_SPLINE, PRIMARY_GROUP_IDS

STATUS = "SOT_HISTORICAL_RESULTS_COMPLETED"
PAPER_TABLE02 = {
    "Primary": ("0.0097", "0.0266", "0.0262"),
    "R1_HGB": ("0.0079", "0.0257", "0.0675"),
    "R2_AC": ("0.0098", "0.0265", "0.0278"),
    "R3_POA20": ("0.0056", "0.0326", "0.0245"),
    "R3_POA100": ("0.0102", "0.0301", "0.0303"),
    "R4_stricter_support": ("0.0100", "0.0254", "0.0152"),
    "R5_keep_all_valid": ("0.0096", "0.0267", "0.0260"),
    "R6_O1": ("0.0152", "0.0174", "0.0028"),
    "R6_O2": ("0.0107", "0.0262", "0.0097"),
    "R6_O3": ("0.0021", "0.0420", "0.0126"),
}
DISPLAY = {
    "Primary": "Primary",
    "R1_HGB": "R1 HGB",
    "R2_AC": "R2 AC",
    "R3_POA20": "R3 POA20",
    "R3_POA100": "R3 POA100",
    "R4_stricter_support": "R4 stricter support",
    "R5_keep_all_valid": "R5 Keep valid",
    "R6_O1": "R6 O1",
    "R6_O2": "R6 O2",
    "R6_O3": "R6 O3",
}


class SourceMissing(RuntimeError):
    pass


class SnapshotHashMismatch(RuntimeError):
    pass


SNAPSHOT_DIR = "artifacts/sot_historical/sources"
REQUEST_INPUT_HEAD = "fad6059da6367f0711e18a9091ab370be3733ab7"
HISTORICAL_SNAPSHOTS = (
    ("S08_STATE_FLEET_C8_FIXED_SET_INTERVAL.json", "artifacts/s08_state_constrained_fleet_fixed_set_inference/S08_STATE_FLEET_C8_FIXED_SET_INTERVAL.json", "621edbe89434d66b454cc0a44548496b2e3857122c28b3fd13d15687cb180dd5"),
    ("S09_RQ1_SUMMARY.csv", "artifacts/s09_scientific/S09_RQ1_SUMMARY.csv", "d333c68858da69b054481ae5ec8b0af211739fde38bcd591d3bb294f37e31b4e"),
    ("S09_RQ2_SUMMARY.csv", "artifacts/s09_scientific/S09_RQ2_SUMMARY.csv", "451fe260a7081ec41c877294ab1ee3216e75fbf6df981baf21f570688db9694b"),
    ("S09_R6_ORIGIN_SUMMARY.csv", "artifacts/s09_scientific/S09_R6_ORIGIN_SUMMARY.csv", "2a45f18359fce1c32757a8b01d4d64630b08b27a8a6e09b065f3c293b223eae7"),
    ("S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv", "artifacts/s09_state_constrained_rq3_v2/S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv", "f59cce12aa67478757efc4b3222500d2eb1919bebe2ecc9ae53596f61ef76789"),
    ("S09_STATE_RQ3_R6_ORIGIN_SUMMARY.csv", "artifacts/s09_state_constrained_rq3_v2/S09_STATE_RQ3_R6_ORIGIN_SUMMARY.csv", "bccc9c7bb9c298e5a51ed3946203e182c28cd06e45313c5587118a90fac275d1"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_canonical(root: Path, rel: str) -> Path:
    candidates = [root / rel, root.parent / rel]
    for path in candidates:
        if path.is_file():
            return path
    raise SourceMissing(rel)


def _repo_rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def materialize_historical_snapshots(root: Path) -> dict[str, Path]:
    dest_dir = root / SNAPSHOT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, origin_rel, expected in HISTORICAL_SNAPSHOTS:
        dest = dest_dir / name
        if not dest.is_file():
            src = resolve_canonical(root, origin_rel)
            dest.write_bytes(src.read_bytes())
        dest.chmod(0o644)
        digest = _sha(dest)
        if digest != expected:
            raise SnapshotHashMismatch(f"{name} {digest} != {expected}")
        out[name] = dest
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _flag(val: object) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def _fmt(value: float, nd: int) -> str:
    return f"{float(value):.{nd}f}"


def _lookup(rows: list[dict[str, str]], arm: str, origin: str, field: str) -> float:
    hits = []
    for row in rows:
        if row.get("arm_id") != arm:
            continue
        got = (row.get("origin_id") or "").strip()
        if got != origin:
            continue
        hits.append(row)
    if len(hits) != 1:
        raise SourceMissing(f"{arm}/{origin or '-'} {field} n={len(hits)}")
    return float(hits[0][field])


def recover(root: Path, sot: dict[str, Any]) -> dict[str, Any]:
    tables = (sot.get("results") or {}).get("tables") or {}
    primary_rq1 = tables["rq1"]
    primary_rq2 = tables["rq2"]
    primary_rq3 = tables["rq3"]

    snaps = materialize_historical_snapshots(root)
    rq1_path = snaps["S09_RQ1_SUMMARY.csv"]
    rq2_path = snaps["S09_RQ2_SUMMARY.csv"]
    r6_path = snaps["S09_R6_ORIGIN_SUMMARY.csv"]
    rq3_path = snaps["S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv"]
    r6_rq3_path = snaps["S09_STATE_RQ3_R6_ORIGIN_SUMMARY.csv"]
    c8_path = snaps["S08_STATE_FLEET_C8_FIXED_SET_INTERVAL.json"]
    s07_path = root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"
    if not s07_path.is_file():
        raise SourceMissing("artifacts/s07/S07_OTG_DIRECTIONAL.csv")

    rq1_rows = _read_csv(rq1_path)
    rq2_rows = _read_csv(rq2_path)
    r6_rows = _read_csv(r6_path)
    rq3_rows = _read_csv(rq3_path)
    r6_rq3_rows = _read_csv(r6_rq3_path)
    c8 = json.loads(c8_path.read_text(encoding="utf-8"))
    s07 = _read_csv(s07_path)

    def r6_rq1(origin: str) -> float:
        rec = next((r for r in r6_rows if r["arm_id"] == f"R6_TEMPORAL__{origin}"), None)
        if rec is None:
            raise SourceMissing(f"R6 {origin} rq1")
        return float(rec["rq1_plant_balanced_mean_otg"])

    def r6_rq2(origin: str) -> float:
        rec = next((r for r in r6_rows if r["arm_id"] == f"R6_TEMPORAL__{origin}"), None)
        if rec is None:
            raise SourceMissing(f"R6 {origin} rq2")
        return float(rec["rq2_plant_balanced_mean_abs_asymmetry"])

    specs = [
        ("Primary", None, None, None, float(primary_rq1), float(primary_rq2), float(primary_rq3), "results.tables", ["artifacts/s07/S07_OTG_DIRECTIONAL.csv", "artifacts/s07/S07_ASYMMETRY.csv", "artifacts/rq3/summary.json"]),
        ("R1_HGB", "R1_HGB", "", None, None, None, None, "S09 snapshots", [f"{SNAPSHOT_DIR}/S09_RQ1_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_RQ2_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv"]),
        ("R2_AC", "R2_AC", "", None, None, None, None, "S09 snapshots", [f"{SNAPSHOT_DIR}/S09_RQ1_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_RQ2_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv"]),
        ("R3_POA20", "R3_POA20", "", None, None, None, None, "S09 snapshots", [f"{SNAPSHOT_DIR}/S09_RQ1_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_RQ2_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv"]),
        ("R3_POA100", "R3_POA100", "", None, None, None, None, "S09 snapshots", [f"{SNAPSHOT_DIR}/S09_RQ1_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_RQ2_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv"]),
        ("R4_stricter_support", "R4_CS3", "", None, None, None, None, "S09 snapshots (arm_id R4_CS3)", [f"{SNAPSHOT_DIR}/S09_RQ1_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_RQ2_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv"]),
        ("R5_keep_all_valid", "R5_KEEP_ALL_VALID", "", None, None, None, None, "S09 snapshots", [f"{SNAPSHOT_DIR}/S09_RQ1_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_RQ2_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv"]),
        ("R6_O1", None, None, "O1", None, None, None, "S09 R6 snapshots", [f"{SNAPSHOT_DIR}/S09_R6_ORIGIN_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_STATE_RQ3_R6_ORIGIN_SUMMARY.csv"]),
        ("R6_O2", None, None, "O2", None, None, None, "S09 R6 snapshots", [f"{SNAPSHOT_DIR}/S09_R6_ORIGIN_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_STATE_RQ3_R6_ORIGIN_SUMMARY.csv"]),
        ("R6_O3", None, None, "O3", None, None, None, "S09 R6 snapshots", [f"{SNAPSHOT_DIR}/S09_R6_ORIGIN_SUMMARY.csv", f"{SNAPSHOT_DIR}/S09_STATE_RQ3_R6_ORIGIN_SUMMARY.csv"]),
    ]
    table02: list[dict[str, Any]] = []
    for analysis, arm, origin, r6_origin, v1, v2, v3, source_note, source_paths in specs:
        if analysis == "Primary":
            rq1, rq2, rq3 = v1, v2, v3
        elif r6_origin:
            rq1, rq2 = r6_rq1(r6_origin), r6_rq2(r6_origin)
            r6_hit = [r for r in r6_rq3_rows if r.get("origin_id") == r6_origin]
            if len(r6_hit) != 1:
                raise SourceMissing(f"R6 {r6_origin} rq3 n={len(r6_hit)}")
            rq3 = float(r6_hit[0]["D_fleet_RQ3"])
        else:
            rq1 = _lookup(rq1_rows, arm, origin, "plant_balanced_mean_OTG")
            rq2 = _lookup(rq2_rows, arm, origin, "plant_balanced_mean_abs_asymmetry")
            rq3 = _lookup(rq3_rows, arm, origin, "D_fleet_RQ3")
        expect = PAPER_TABLE02[analysis]
        if (_fmt(rq1, 4), _fmt(rq2, 4), _fmt(rq3, 4)) != expect:
            raise SourceMissing(f"{analysis} rounded {(_fmt(rq1, 4), _fmt(rq2, 4), _fmt(rq3, 4))} != {expect}")
        table02.append({
            "analysis": analysis,
            "display_label": DISPLAY[analysis],
            "rq1_otg": rq1,
            "rq2_abs_asymmetry": rq2,
            "rq3_control_minus_twin": rq3,
            "source_note": source_note,
            "source_paths": source_paths,
        })

    ci_lo, ci_hi = float(c8["fleet_ci_lower"]), float(c8["fleet_ci_upper"])
    if _fmt(ci_lo, 4) != "0.0056" or _fmt(ci_hi, 4) != "0.0566":
        raise SourceMissing(f"C8 interval {_fmt(ci_lo, 4)} {_fmt(ci_hi, 4)}")

    primary_dirs = []
    for row in s07:
        if row.get("group_id") not in PRIMARY_GROUP_IDS:
            continue
        if row.get("model_family") != FAM_SPLINE or row.get("support_rule") != "CS4":
            continue
        if not _flag(row.get("computable")):
            continue
        primary_dirs.append(row)
    if len(primary_dirs) != 85:
        raise SourceMissing(f"primary directional n={len(primary_dirs)}")
    rets = [float(r["support_retention"]) for r in primary_dirs]
    median = float(statistics.median(rets))
    if _fmt(median, 3) != "0.541":
        raise SourceMissing(f"retention median {median}")
    defined = [r for r in primary_dirs if _flag(r.get("otg_rel_defined"))]
    extreme = max(defined, key=lambda r: float(r["otg_rel"]))
    ext_ret = float(extreme["support_retention"])
    if _fmt(ext_ret, 3) != "0.179":
        raise SourceMissing(f"low retention {ext_ret}")

    sources = {
        "S09_RQ1_SUMMARY.csv": rq1_path,
        "S09_RQ2_SUMMARY.csv": rq2_path,
        "S09_R6_ORIGIN_SUMMARY.csv": r6_path,
        "S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv": rq3_path,
        "S09_STATE_RQ3_R6_ORIGIN_SUMMARY.csv": r6_rq3_path,
        "S08_STATE_FLEET_C8_FIXED_SET_INTERVAL.json": c8_path,
        "S07_OTG_DIRECTIONAL.csv": s07_path,
    }
    source_meta = {}
    for name, path in sources.items():
        rel = _repo_rel(root, path)
        source_meta[name] = {
            "path": rel,
            "resolved_path": rel,
            "sha256": _sha(path),
            "in_study_tree": True,
            "materialized_snapshot": name != "S07_OTG_DIRECTIONAL.csv",
        }
    return {
        "status": STATUS,
        "table02": table02,
        "primary_uncertainty": {
            "rq3_fixed_set": {
                "estimand": "CONTROL_MINUS_TWIN",
                "point_estimate": float(primary_rq3),
                "ci_level": 0.95,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "method": c8.get("method_id"),
                "scope": c8.get("scope"),
                "applies_to": "RQ3 Primary only",
                "not_applied_to_robustness_arms": True,
                "artifact_path": f"{SNAPSHOT_DIR}/S08_STATE_FLEET_C8_FIXED_SET_INTERVAL.json",
                "artifact_sha256": _sha(c8_path),
                "artifact_point_estimate": c8.get("D_fleet_RQ3"),
            }
        },
        "primary_support_retention": {
            "n_directional_computable": 85,
            "median": median,
            "range_min": min(rets),
            "range_max": max(rets),
            "reported_low_retention_extreme": {
                "value": ext_ret,
                "source_plant_id": extreme["source_plant_id"],
                "target_plant_id": extreme["target_plant_id"],
                "group_id": extreme["group_id"],
                "otg_abs": float(extreme["otg_abs"]),
                "otg_rel": float(extreme["otg_rel"]),
                "local_same_rows_mae": float(extreme["local_same_rows_mae"]),
                "context": "maximum relative OTG among 85 primary CS4 SplineRidge computable directions",
            },
            "artifact_path": "artifacts/s07/S07_OTG_DIRECTIONAL.csv",
            "artifact_sha256": _sha(s07_path),
        },
        "sources": source_meta,
        "no_scientific_recomputation": True,
        "no_model_refit": True,
        "no_prediction_rescoring": True,
        "manuscript_unchanged": True,
    }
