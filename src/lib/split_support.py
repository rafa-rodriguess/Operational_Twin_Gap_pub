"""Blinded structural helpers for S03. No model fitting, MAE, OTG, or sealed-artifact I/O."""

from __future__ import annotations

import csv
import hashlib
import zipfile
import math
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.lib.matching import HIERARCHY, ranking_components, scientific_tuple

IRRADIANCE_WM2 = (20, 50, 100)
COVERAGE_DC = (0.80, 0.90, 1.00)
INTERP_POLICIES = ("keep_all_valid", "exclude_TRUE", "exclude_TRUE_and_UNKNOWN")
FEATURE_STRATEGIES = {
    "six_core": (
        "poa_irradiance_wm2",
        "ghi_irradiance_wm2",
        "gri_irradiance_wm2",
        "panel_temperature_celsius",
        "ambient_temperature_celsius",
        "wind_speed_ms",
    ),
    "POA_GHI": ("poa_irradiance_wm2", "ghi_irradiance_wm2"),
}
SUPPORT_RULES = ((1, 0.90), (1, 0.95), (5, 0.90), (5, 0.95))
PRIMARY_INTERP_POLICY = "exclude_TRUE"
R5_INTERP_POLICY = "keep_all_valid"
SPLIT_LATTICE_POLICIES = (PRIMARY_INTERP_POLICY, R5_INTERP_POLICY)
NN_ALGORITHM = "kd_tree"
QUANTILE_METHOD = "linear"
FEATURE_INTERP = {
    "poa_irradiance_wm2": "interpolated_keys_poa_irradiance_wm2",
    "ghi_irradiance_wm2": "interpolated_keys_ghi_irradiance_wm2",
    "gri_irradiance_wm2": "interpolated_keys_gri_irradiance_wm2",
    "panel_temperature_celsius": "interpolated_keys_panel_temperature_celsius",
    "ambient_temperature_celsius": "interpolated_keys_ambient_temperature_celsius",
    "wind_speed_ms": "interpolated_keys_wind_speed_ms",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: tuple(str(r.get(f) or "") for f in fields))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row[k] for k in fields})


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(obj))


def json_bytes(obj: Any) -> bytes:
    import json

    return json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")


def parse_ts(text: str) -> datetime | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def assign_post_eligibility_split(n: int, index0: int) -> str:
    """Chronological 60/20/20 after eligibility. train=floor(0.60*N), validation through floor(0.80*N)."""
    train_end = math.floor(0.60 * n)
    val_end = math.floor(0.80 * n)
    if index0 < train_end:
        return "train"
    if index0 < val_end:
        return "validation"
    return "test"


def order_eligible_local(ts_local: list, dt_raw_local: list[str], elig: np.ndarray) -> list[int]:
    positions = [int(j) for j in np.where(elig)[0]]
    sentinel = datetime.max.replace(tzinfo=timezone.utc)

    def key(j: int) -> tuple:
        t = ts_local[j]
        return (t if t is not None else sentinel, str(dt_raw_local[j] or ""))

    return sorted(positions, key=key)


def split_labels_for_eligible(n_plant: int, order: list[int]) -> np.ndarray:
    labels = np.full(n_plant, "", dtype=object)
    n = len(order)
    for rank, j in enumerate(order):
        labels[j] = assign_post_eligibility_split(n, rank)
    return labels


def partition_meta(labels: np.ndarray, ts_local: list, name: str) -> dict[str, Any]:
    idx = [j for j, lab in enumerate(labels) if lab == name]
    stamps = sorted(ts_local[j] for j in idx if ts_local[j] is not None)
    days = {ts_local[j].date() for j in idx if ts_local[j] is not None}
    duration_s = (stamps[-1] - stamps[0]).total_seconds() if len(stamps) >= 2 else 0.0
    return {
        f"n_{name}": len(idx),
        f"pct_{name}": (len(idx) / int(np.sum(labels != ""))) if np.any(labels != "") else None,
        f"n_{name}_days": len(days),
        f"first_{name}_ts": stamps[0].isoformat() if stamps else None,
        f"last_{name}_ts": stamps[-1].isoformat() if stamps else None,
        f"{name}_duration_seconds": duration_s,
        f"{name}_nonempty": bool(idx),
    }


def chronology_ok(labels: np.ndarray, ts_local: list) -> bool:
    def stamps(name: str) -> list:
        return [ts_local[j] for j, lab in enumerate(labels) if lab == name and ts_local[j] is not None]

    train, val, test = stamps("train"), stamps("validation"), stamps("test")
    if train and val and max(train) > min(val):
        return False
    if val and test and max(val) > min(test):
        return False
    if train and test and not val and max(train) > min(test):
        return False
    return True


def source_median_iqr(values: np.ndarray) -> tuple[float, float]:
    q1, med, q3 = np.quantile(values, [0.25, 0.50, 0.75], method=QUANTILE_METHOD)
    return float(med), float(q3 - q1)


def iqr_is_valid(iqr: float) -> bool:
    return bool(np.isfinite(iqr) and iqr > 0)


def standardize(values: np.ndarray, median: float, iqr: float) -> np.ndarray:
    return (values - median) / iqr


def loo_kth_distances(z: np.ndarray, k: int) -> np.ndarray:
    if z.shape[0] <= k:
        return np.full(z.shape[0], np.nan)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm=NN_ALGORITHM, metric="euclidean")
    nn.fit(z)
    dist, _ = nn.kneighbors(z, n_neighbors=k + 1, return_distance=True)
    return dist[:, k]


def q_linear(distances: np.ndarray, q: float) -> float:
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q, method=QUANTILE_METHOD))


def scale_matrix(feats: dict[str, np.ndarray], columns: tuple[str, ...], mask: np.ndarray) -> tuple[np.ndarray | None, dict[str, float], dict[str, float], str | None]:
    medians: dict[str, float] = {}
    iqrs: dict[str, float] = {}
    n = int(mask.sum())
    if n == 0:
        return None, medians, iqrs, "zero_complete_case"
    cols = []
    for name in columns:
        vals = feats[name][mask]
        if vals.size == 0 or not np.all(np.isfinite(vals)):
            return None, medians, iqrs, "nonfinite_feature"
        med, iqr = source_median_iqr(vals)
        medians[name] = med
        iqrs[name] = iqr
        if not iqr_is_valid(iqr):
            return None, medians, iqrs, f"degenerate_iqr:{name}"
        cols.append(standardize(vals, med, iqr))
    return np.column_stack(cols), medians, iqrs, None


def apply_scale(feats: dict[str, np.ndarray], columns: tuple[str, ...], mask: np.ndarray, medians: dict[str, float], iqrs: dict[str, float]) -> np.ndarray | None:
    if int(mask.sum()) == 0:
        return None
    cols = []
    for name in columns:
        vals = feats[name][mask]
        if not np.all(np.isfinite(vals)):
            return None
        cols.append(standardize(vals, medians[name], iqrs[name]))
    return np.column_stack(cols)


def complete_mask(feats: dict[str, np.ndarray], columns: tuple[str, ...]) -> np.ndarray:
    mask = np.ones(next(iter(feats.values())).shape[0], dtype=bool)
    for name in columns:
        vals = feats[name]
        mask &= np.isfinite(vals)
    return mask


def interp_any(flags: dict[str, np.ndarray], columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    n = next(iter(flags.values())).shape[0]
    any_true = np.zeros(n, dtype=bool)
    any_unknown = np.zeros(n, dtype=bool)
    for name in columns:
        col = flags[name]
        known = col == col  # False for object nan; for bool pandas might be different
        # numpy bool with None becomes object; we use float 1/0/nan
        arr = flags[name]
        any_true |= arr == 1
        any_unknown |= ~np.isfinite(arr)
    return any_true, any_unknown


def interp_policy_mask(any_true: np.ndarray, any_unknown: np.ndarray, policy: str) -> np.ndarray:
    if policy == "keep_all_valid":
        return np.ones(any_true.shape[0], dtype=bool)
    if policy == "exclude_TRUE":
        return ~any_true
    if policy == "exclude_TRUE_and_UNKNOWN":
        return (~any_true) & (~any_unknown)
    raise ValueError(policy)


def official_interpolation_semantics(readme: Path) -> dict[str, Any]:
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    return {
        "source": "data/raw/br_pvgen/Readme.txt" if readme.is_file() else None,
        "official_true_meaning": "true if the value at that timestamp was linearly interpolated",
        "official_false_meaning": "boolean false: value was not linearly interpolated",
        "empty_cell_policy": "UNKNOWN",
        "empty_never_mapped_to_false": True,
        "gap_rule": "short gaps (<= 15 minutes, bounded by valid neighboring values) in instantaneous variables",
        "readme_mentions_interpolation": "interpolated_keys_" in text and "linearly interpolated" in text,
        "p02c_interpolated_pct_semantics": "mean of TRUE among interpretable (TRUE/FALSE) rows only; NULL flags excluded from the denominator; 1.0 is not 'all records interpolated'",
        "status": "DETERMINED" if "linearly interpolated" in text else "BLOCKED",
    }


def sample_raw_flag_reconciliation(inverter_zip: Path, plant_id: str = "PS_001", limit: int = 40) -> dict[str, Any]:
    if not inverter_zip.is_file():
        return {"status": "BLOCKED", "reason": "inverter_zip_absent"}
    with zipfile.ZipFile(inverter_zip) as zf:
        names = [n for n in zf.namelist() if plant_id.lower() in n.lower() and n.endswith(".csv")]
        if not names:
            names = [n for n in zf.namelist() if n.endswith(".csv")][:1]
        if not names:
            return {"status": "BLOCKED", "reason": "no_csv_in_zip"}
        with zf.open(names[0]) as handle:
            raw = handle.read(250_000).decode("utf-8", errors="replace")
    lines = raw.splitlines()
    header = [c.strip() for c in lines[0].split(",")]
    dc_col = next((c for c in header if c.startswith("interpolated_keys_total_dc_power")), None)
    counts = {"true": 0, "false": 0, "empty": 0, "other": 0, "rows_read": 0}
    if dc_col is None:
        return {"status": "BLOCKED", "reason": "missing_dc_interp_column", "header_sample": header[:20], "zip_member": names[0]}
    idx = header.index(dc_col)
    for line in lines[1 : limit + 1]:
        if not line.strip():
            continue
        parts = line.split(",")
        if idx >= len(parts):
            continue
        token = parts[idx].strip().lower()
        counts["rows_read"] += 1
        if token == "":
            counts["empty"] += 1
        elif token in {"true", "1"}:
            counts["true"] += 1
        elif token in {"false", "0"}:
            counts["false"] += 1
        else:
            counts["other"] += 1
    return {
        "status": "RECONCILED",
        "zip_member": names[0],
        "column": dc_col,
        "sample_counts": counts,
        "parser_mapping": {"true": True, "false": False, "empty": None},
        "panel_aggregation": "dc_any_officially_interpolated is TRUE if any observed inverter flag is TRUE; FALSE only if all observed flags are FALSE; NULL if no interpretable flags",
    }


def three_time_blocks(timestamps: list[datetime]) -> list[tuple[datetime, datetime, int]]:
    if not timestamps:
        return []
    ordered = sorted(timestamps)
    t0, t1 = ordered[0], ordered[-1]
    span = (t1 - t0).total_seconds()
    if span <= 0:
        return [(t0, t1, len(ordered))]
    cuts = [t0, t0 + (t1 - t0) / 3, t0 + 2 * (t1 - t0) / 3, t1]
    out = []
    for i in range(3):
        a, b = cuts[i], cuts[i + 1]
        if i < 2:
            n = sum(1 for t in ordered if a <= t < b)
        else:
            n = sum(1 for t in ordered if a <= t <= b)
        out.append((a, b, n))
    return out


def rank_controls(
    reference: dict[str, str],
    candidates: list[dict[str, str]],
) -> list[dict[str, Any]]:
    ranked = []
    for cand in candidates:
        if cand.get("plant_id") == reference.get("plant_id"):
            continue
        if cand.get("group_id") and cand.get("group_id") == reference.get("group_id"):
            continue
        comp = ranking_components(reference, cand)
        if comp is None:
            continue
        ranked.append(
            {
                "control_plant_id": cand["plant_id"],
                "tuple": scientific_tuple(comp),
                "same_structure": comp["same_structure"],
                "abs_nominal_power_mw": str(comp["abs_nominal_power_mw"]),
                "abs_panel_efficiency_percentage": str(comp["abs_panel_efficiency_percentage"]),
                "abs_panel_bifaciality_coefficient": str(comp["abs_panel_bifaciality_coefficient"]),
            }
        )
    ranked.sort(key=lambda r: r["tuple"])
    return ranked
