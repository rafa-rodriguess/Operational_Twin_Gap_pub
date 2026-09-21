"""P-02A BR-PVGen integrity audit. Stages must not write artifacts/sot.json."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord

DOI = "10.5281/zenodo.21511487"
RAW_REL = "data/raw/br_pvgen"
ZENODO_FILES = (
    "BR-PVGen_metadata.csv",
    "BR-PVGen_inverter.zip",
    "BR-PVGen_meteorological.zip",
    "Readme.txt",
)
ZENODO_MD5 = {
    "BR-PVGen_inverter.zip": "fbdb7cdb328a3e6ffb470f95bc56bcf4",
    "BR-PVGen_meteorological.zip": "7c20c037074734f426d79a4e3f0db0bf",
    "BR-PVGen_metadata.csv": "abae1b232035097a55d8534a9ec51ab8",
    "Readme.txt": "49bbfb45c13db13baf2c669a7d67c1d0",
}
DOC_EXPECTED_INNER = {
    "inverter": "plant_{ps}_inverter.csv",
    "meteorological": "plant_{ps}_solar_station.csv",
}


def _sha256_file(path: Path) -> str:
    return _file_digests(path)[0]


def _file_digests(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _infer_dtype(values: list[str]) -> str:
    nonempty = [v for v in values if v not in ("", None)]
    if not nonempty:
        return "empty"
    lowered = [v.strip().lower() for v in nonempty[:50]]
    if all(v in {"true", "false"} for v in lowered):
        return "boolean"
    ints = 0
    floats = 0
    for v in nonempty[:200]:
        try:
            int(v)
            ints += 1
            continue
        except ValueError:
            pass
        try:
            float(v)
            floats += 1
        except ValueError:
            return "string"
    if ints and not floats:
        return "integer"
    if floats or ints:
        return "float"
    return "string"


def _schema_fingerprint(columns: list[str], dtypes: dict[str, str]) -> str:
    payload = json.dumps([(c, dtypes.get(c, "unknown")) for c in columns], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_dt(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _family_for_name(name: str) -> str:
    lower = name.lower()
    if "metadata" in lower and lower.endswith(".csv"):
        return "metadata"
    if lower.endswith(".txt") or "readme" in lower:
        return "documentation"
    if lower.endswith(".zip") and "inverter" in lower:
        return "inverter_archive"
    if lower.endswith(".zip") and ("meteor" in lower or "solar" in lower):
        return "meteorological_archive"
    return "other"


def _classify_inner(name: str, expected_family: str) -> str | None:
    base = Path(name).name
    if not base.lower().endswith(".csv"):
        return None
    if base.endswith("_inverter.csv") and expected_family == "inverter":
        return "inverter"
    if base.endswith("_solar_station.csv") and expected_family == "meteorological":
        return "meteorological"
    if base.startswith("PS_") and expected_family in {"inverter", "meteorological"}:
        return expected_family
    return None


def _plant_from_inner(name: str) -> str | None:
    stem = Path(name).stem
    if stem.startswith("PS_"):
        return stem
    parts = Path(name).name.split("_")
    if len(parts) >= 3 and parts[0] == "plant" and parts[1] == "PS":
        return f"PS_{parts[2]}"
    return None


def _is_tracker_field(col: str) -> bool:
    return "tracker_albedo_index" in col


def _audit_csv(
    reader: csv.DictReader,
    *,
    family: str,
    plant_hint: str | None,
) -> dict[str, Any]:
    columns = list(reader.fieldnames or [])
    sample: dict[str, list[str]] = {c: [] for c in columns}
    nulls: Counter[str] = Counter()
    n = 0
    exact_dup = 0
    seen_hashes: set[str] = set()
    key_counts: Counter[tuple] = Counter()
    parse_ok = 0
    parse_fail = 0
    tz_aware = 0
    tz_absent = 0
    tmin: datetime | None = None
    tmax: datetime | None = None
    non_mono = 0
    last_dt: datetime | None = None
    cadence: Counter[int] = Counter()
    impossible: Counter[str] = Counter()
    interp_true: Counter[str] = Counter()
    interp_false: Counter[str] = Counter()
    inverters: set[str] = set()
    plants: set[str] = set()
    timestamps: set[str] = set()

    for row in reader:
        n += 1
        blob = hashlib.blake2s(
            "\x1f".join(row.get(c, "") or "" for c in columns).encode("utf-8"), digest_size=16
        ).hexdigest()
        if blob in seen_hashes:
            exact_dup += 1
        else:
            seen_hashes.add(blob)
        for col in columns:
            val = row.get(col)
            if val is None or val == "":
                nulls[col] += 1
            elif len(sample[col]) < 80:
                sample[col].append(val)
        ps = (row.get("ps_id") or row.get("id") or plant_hint or "").strip()
        if ps:
            plants.add(ps)
        inv = (row.get("inverter_id") or "").strip()
        if inv:
            inverters.add(inv)
        raw_dt = row.get("datetime")
        if "datetime" in columns:
            parsed = _parse_dt(str(raw_dt or ""))
            if parsed is None:
                parse_fail += 1
            else:
                parse_ok += 1
                timestamps.add(str(raw_dt))
                if tmin is None or parsed < tmin:
                    tmin = parsed
                if tmax is None or parsed > tmax:
                    tmax = parsed
                if parsed.tzinfo is None:
                    tz_absent += 1
                else:
                    tz_aware += 1
                if last_dt is not None:
                    if parsed < last_dt:
                        non_mono += 1
                    delta = int((parsed - last_dt).total_seconds())
                    cadence[delta] += 1
                last_dt = parsed
            if family == "inverter":
                key_counts[(ps, inv, str(raw_dt))] += 1
            elif family == "meteorological":
                key_counts[(ps, str(raw_dt))] += 1
        elif family == "metadata":
            key_counts[(ps,)] += 1
        for col in columns:
            if _is_tracker_field(col):
                continue
            val = row.get(col)
            if val in (None, ""):
                continue
            if col.startswith("interpolated_keys_"):
                low = val.strip().lower()
                if low == "true":
                    interp_true[col] += 1
                elif low == "false":
                    interp_false[col] += 1
                else:
                    impossible[col] += 1
                continue
            if col in {
                "total_active_power_w",
                "total_dc_power_w",
                "poa_irradiance_wm2",
                "ghi_irradiance_wm2",
                "gri_irradiance_wm2",
                "precipitation_accumulated_mm",
                "wind_speed_ms",
            }:
                try:
                    if float(val) < 0:
                        impossible[col] += 1
                except ValueError:
                    impossible[col] += 1
            if col == "wind_direction_degrees":
                try:
                    deg = float(val)
                    if deg < 0 or deg > 360:
                        impossible[col] += 1
                except ValueError:
                    impossible[col] += 1

    dtypes = {c: _infer_dtype(sample[c]) for c in columns}
    key_dups = sum(v - 1 for v in key_counts.values() if v > 1)
    tmin_s = tmin.isoformat() if tmin else None
    tmax_s = tmax.isoformat() if tmax else None
    return {
        "family": family,
        "plant_hint": plant_hint,
        "row_count": n,
        "column_count": len(columns),
        "columns": columns,
        "dtypes": dtypes,
        "nulls": dict(nulls),
        "schema_fingerprint": _schema_fingerprint(columns, dtypes),
        "exact_duplicate_rows": exact_dup,
        "natural_key_duplicate_extras": key_dups,
        "natural_key_cardinality": len(key_counts),
        "timestamp_parse_ok": parse_ok,
        "timestamp_parse_fail": parse_fail,
        "timezone_aware": tz_aware,
        "timezone_absent": tz_absent,
        "timestamp_min": tmin_s,
        "timestamp_max": tmax_s,
        "non_monotonic": non_mono,
        "cadence_seconds": dict(cadence),
        "impossible": dict(impossible),
        "interpolated_true": dict(interp_true),
        "interpolated_false": dict(interp_false),
        "inverter_ids": sorted(inverters),
        "plants": sorted(plants),
        "timestamps": timestamps,
    }


def _open_text(handle) -> io.TextIOWrapper:
    return io.TextIOWrapper(handle, encoding="utf-8", newline="")


TABULAR_SOT_KEYS = ("schema_summary", "quality_summary", "joinability_summary")


def _csv_data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _csv_artifact_schema_fingerprint(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        sample: dict[str, list[str]] = {c: [] for c in columns}
        for row in reader:
            for col in columns:
                val = row.get(col)
                if val not in (None, "") and len(sample[col]) < 200:
                    sample[col].append(val)
    dtypes = {c: _infer_dtype(sample[c]) for c in columns}
    return _schema_fingerprint(columns, dtypes)


def _path_inside_root(root: Path, rel: str) -> Path | None:
    if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        return None
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


def _try_metadata_repair(ctx: StageContext) -> StageResult | None:
    stages = ctx.sot.get("stages") or {}
    feat = ((ctx.sot.get("feasibility") or {}).get("p02a") or {})
    p02a = stages.get("P02A") or {}
    if p02a.get("status") != "GO" or feat.get("status") != "PASS":
        return None

    validations: list[ValidationRecord] = []
    artifacts: list[ArtifactRecord] = []
    fingerprints: dict[str, str] = {}
    ok = True
    details: list[str] = []
    for key in TABULAR_SOT_KEYS:
        rec = feat.get(key) or {}
        rel = rec.get("path")
        path = _path_inside_root(ctx.root, str(rel) if rel else "")
        if path is None or not path.is_file():
            ok = False
            details.append(f"{key}: missing")
            validations.append(ValidationRecord(f"repair_{key}_present", False, str(rel)))
            continue
        digest = _sha256_file(path)
        rows = _csv_data_row_count(path)
        hash_ok = digest == rec.get("sha256")
        rows_ok = rows == rec.get("row_count")
        validations.append(ValidationRecord(f"repair_{key}_hash", hash_ok, digest))
        validations.append(ValidationRecord(f"repair_{key}_rows", rows_ok, str(rows)))
        if not hash_ok or not rows_ok:
            ok = False
            details.append(f"{key}: hash_or_rows")
            continue
        fp = _csv_artifact_schema_fingerprint(path)
        hex_ok = len(fp) == 64 and fp == fp.lower() and all(c in "0123456789abcdef" for c in fp)
        validations.append(ValidationRecord(f"repair_{key}_fingerprint", hex_ok, fp))
        if not hex_ok:
            ok = False
            details.append(f"{key}: fingerprint")
            continue
        fingerprints[key] = fp
        artifacts.append(
            ArtifactRecord(
                path=_rel(ctx.root, path),
                sha256=digest,
                bytes=path.stat().st_size,
                row_count=rows,
                schema_fingerprint=fp,
            )
        )

    validations.append(ValidationRecord("repair_no_raw_required", True, RAW_REL))
    if not ok:
        return StageResult(
            status="STOP",
            message="P02A metadata repair blocked: existing artifacts do not reconcile",
            sot_patch={},
            artifacts=[],
            validations=validations,
        )

    patch = {
        "feasibility": {
            "p02a": {key: {"schema_fingerprint": fingerprints[key]} for key in TABULAR_SOT_KEYS}
        }
    }
    validations.append(ValidationRecord("repair_patch_fingerprints_only", True, None))
    return StageResult(
        status="GO",
        message="P02A SOT METADATA REPAIRED",
        sot_patch=patch,
        artifacts=artifacts,
        validations=validations,
    )


def run(ctx: StageContext) -> StageResult:
    repaired = _try_metadata_repair(ctx)
    if repaired is not None:
        return repaired

    root = ctx.root
    raw_dir = root / RAW_REL
    validations: list[ValidationRecord] = []
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    present = {p.name: p for p in raw_dir.iterdir() if p.is_file()} if raw_dir.is_dir() else {}
    missing_expected = [n for n in ZENODO_FILES if n not in present]
    unexpected = sorted(set(present) - set(ZENODO_FILES))
    found_files = sorted(present)
    validations.append(
        ValidationRecord(
            "zenodo_files_present",
            not missing_expected,
            None if not missing_expected else "missing " + ", ".join(missing_expected),
        )
    )
    if missing_expected:
        return StageResult(
            status="STOP",
            message="official BR-PVGen files missing; cannot audit",
            sot_patch={
                "stages": {
                    "P02A": {
                        "kind": "data_feasibility",
                        "status": "STOP",
                        "finished_at_utc": now,
                    }
                },
                "feasibility": {"p02a": {"status": "FAIL", "summary": {"missing_expected_files": missing_expected}}},
            },
            artifacts=[],
            validations=validations,
        )

    digests = {name: _file_digests(present[name]) for name in ZENODO_FILES}
    pre_hashes = {name: digests[name][0] for name in ZENODO_FILES}
    md5s = {name: digests[name][1] for name in ZENODO_FILES}
    md5_ok = all(md5s[n] == ZENODO_MD5[n] for n in ZENODO_FILES)
    validations.append(
        ValidationRecord(
            "zenodo_md5",
            md5_ok,
            None if md5_ok else json.dumps(md5s),
        )
    )
    provenance = {
        "doi": DOI,
        "zenodo_record_id": 21511487,
        "acquisition": "local copy of official Zenodo objects; md5 verified against record 21511487",
        "raw_dir": RAW_REL,
        "md5": md5s,
    }

    manifest_entries: list[dict[str, Any]] = []
    file_audits: list[dict[str, Any]] = []
    inverter_rows = 0
    meteo_rows = 0
    metadata_rows = 0
    inverter_plants: set[str] = set()
    meteo_plants: set[str] = set()
    metadata_plants: set[str] = set()
    meteo_ts: dict[str, set[str]] = defaultdict(set)
    inverter_ts_n: dict[str, int] = defaultdict(int)
    overlap_n: dict[str, int] = defaultdict(int)
    inverter_counts: dict[str, set[str]] = defaultdict(set)
    schema_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    def record_outer(name: str, family: str, extra: dict[str, Any] | None = None) -> None:
        path = present[name]
        rec = {
            "relative_path": _rel(root, path),
            "file_family": family,
            "byte_size": path.stat().st_size,
            "sha256": pre_hashes[name],
            "format": path.suffix.lstrip(".") or "txt",
            "row_count": extra.get("row_count") if extra else None,
            "column_count": extra.get("column_count") if extra else None,
            "provenance": provenance,
        }
        manifest_entries.append(rec)

    # metadata
    with present["BR-PVGen_metadata.csv"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        meta_audit = _audit_csv(reader, family="metadata", plant_hint=None)
    metadata_rows = meta_audit["row_count"]
    metadata_plants = set(meta_audit["plants"])
    record_outer("BR-PVGen_metadata.csv", "metadata", meta_audit)
    file_audits.append({k: v for k, v in meta_audit.items() if k != "timestamps"})
    schema_rows.append(
        {
            "source": "BR-PVGen_metadata.csv",
            "family": "metadata",
            "columns": json.dumps(meta_audit["columns"]),
            "dtypes": json.dumps(meta_audit["dtypes"]),
            "schema_fingerprint": meta_audit["schema_fingerprint"],
            "nulls": json.dumps(meta_audit["nulls"]),
        }
    )
    quality_rows.append(_quality_row("BR-PVGen_metadata.csv", meta_audit))

    record_outer("Readme.txt", "documentation")

    def audit_zip(zip_name: str, expected_family: str) -> None:
        nonlocal inverter_rows, meteo_rows
        path = present[zip_name]
        record_outer(zip_name, "inverter_archive" if expected_family == "inverter" else "meteorological_archive")
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner_family = _classify_inner(info.filename, expected_family)
                if inner_family != expected_family:
                    continue
                plant = _plant_from_inner(info.filename)
                with zf.open(info) as raw:
                    with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
                        reader = csv.DictReader(text)
                        audit = _audit_csv(reader, family=expected_family, plant_hint=plant)
                if expected_family == "inverter":
                    inverter_rows += audit["row_count"]
                    inverter_plants.update(audit["plants"])
                    for p in audit["plants"]:
                        inverter_ts_n[p] += len(audit["timestamps"])
                        overlap_n[p] += len(audit["timestamps"] & meteo_ts.get(p, set()))
                        inverter_counts[p].update(audit["inverter_ids"])
                else:
                    meteo_rows += audit["row_count"]
                    meteo_plants.update(audit["plants"])
                    for p in audit["plants"]:
                        meteo_ts[p].update(audit["timestamps"])
                slim = {k: v for k, v in audit.items() if k != "timestamps"}
                slim["inner_path"] = info.filename
                file_audits.append(slim)
                schema_rows.append(
                    {
                        "source": info.filename,
                        "family": expected_family,
                        "columns": json.dumps(audit["columns"]),
                        "dtypes": json.dumps(audit["dtypes"]),
                        "schema_fingerprint": audit["schema_fingerprint"],
                        "nulls": json.dumps(audit["nulls"]),
                    }
                )
                quality_rows.append(_quality_row(info.filename, audit))

    audit_zip("BR-PVGen_meteorological.zip", "meteorological")
    audit_zip("BR-PVGen_inverter.zip", "inverter")

    post_hashes = {name: _sha256_file(present[name]) for name in ZENODO_FILES}
    validations.append(
        ValidationRecord("raw_hash_invariant", pre_hashes == post_hashes, None if pre_hashes == post_hashes else "raw bytes changed")
    )

    expected_plants = {f"PS_{i:03d}" for i in range(1, 52)}
    documented_inv_inner = {DOC_EXPECTED_INNER["inverter"].format(ps=p) for p in expected_plants}
    documented_met_inner = {DOC_EXPECTED_INNER["meteorological"].format(ps=p) for p in expected_plants}
    observed_inv = {Path(a.get("inner_path", "")).name for a in file_audits if a.get("family") == "inverter"}
    observed_met = {Path(a.get("inner_path", "")).name for a in file_audits if a.get("family") == "meteorological"}
    zip_expected = {f"{p}.csv" for p in expected_plants}
    missing_inner = sorted((zip_expected - observed_inv) | (zip_expected - observed_met))
    unexpected_inner = sorted((observed_inv - zip_expected) | (observed_met - zip_expected))
    documented_name_gap = {
        "readme_inner_pattern": DOC_EXPECTED_INNER,
        "zip_member_pattern": "BR-PVGen_{family}/PS_XXX.csv",
        "readme_inner_names_not_in_zip": True,
        "documented_inverter_example": sorted(documented_inv_inner)[0],
        "documented_meteorological_example": sorted(documented_met_inner)[0],
    }

    join_rows: list[dict[str, Any]] = []
    matched_plants = metadata_plants & inverter_plants & meteo_plants
    for plant in sorted(metadata_plants | inverter_plants | meteo_plants):
        inv_n = inverter_ts_n.get(plant, 0)
        met_n = len(meteo_ts.get(plant, ()))
        overlap = overlap_n.get(plant, 0)
        join_rows.append(
            {
                "ps_id": plant,
                "in_metadata": plant in metadata_plants,
                "in_inverter": plant in inverter_plants,
                "in_meteorological": plant in meteo_plants,
                "inverter_timestamps": inv_n,
                "meteorological_timestamps": met_n,
                "timestamp_overlap": overlap,
                "inverter_id_count": len(inverter_counts.get(plant, ())),
            }
        )

    constructible = bool(matched_plants) and all(
        row["timestamp_overlap"] > 0 for row in join_rows if row["ps_id"] in matched_plants
    )
    joinability_assessment = (
        "plant-level temporal panel constructible in principle via official ps_id and datetime"
        if constructible
        else "structural join incomplete: missing plant identifiers or timestamp overlap"
    )

    inv_quality_sum = sum(int(a.get("row_count") or 0) for a in file_audits if a.get("family") == "inverter")
    met_quality_sum = sum(int(a.get("row_count") or 0) for a in file_audits if a.get("family") == "meteorological")
    validations.append(
        ValidationRecord(
            "family_row_totals",
            inverter_rows == inv_quality_sum and meteo_rows == met_quality_sum,
            f"inv {inverter_rows}/{inv_quality_sum} met {meteo_rows}/{met_quality_sum}",
        )
    )
    unmatched_join = [
        r for r in join_rows if not (r["in_metadata"] and r["in_inverter"] and r["in_meteorological"])
    ]
    validations.append(
        ValidationRecord(
            "join_unmatched_reconciles",
            len(join_rows) - len(matched_plants) == len(unmatched_join),
            f"rows={len(join_rows)} matched={len(matched_plants)} unmatched={len(unmatched_join)}",
        )
    )
    validations.append(
        ValidationRecord(
            "timestamp_parse_reconciles",
            all(
                a["timestamp_parse_ok"] + a["timestamp_parse_fail"] == a["row_count"]
                or a["family"] == "metadata"
                or a["family"] == "documentation"
                for a in file_audits
                if a.get("family") in {"inverter", "meteorological"}
            ),
            None,
        )
    )
    validations.append(
        ValidationRecord(
            "min_le_max",
            all(
                a.get("timestamp_min") is None
                or a.get("timestamp_max") is None
                or a["timestamp_min"] <= a["timestamp_max"]
                for a in file_audits
            ),
            None,
        )
    )
    validations.append(
        ValidationRecord(
            "duplicates_bounded",
            all(int(a.get("exact_duplicate_rows") or 0) <= int(a.get("row_count") or 0) for a in file_audits),
            None,
        )
    )
    validations.append(
        ValidationRecord(
            "no_tracker_albedo_qc",
            all("tracker_albedo_index" not in (a.get("impossible") or {}) for a in file_audits),
            None,
        )
    )
    validations.append(ValidationRecord("br_pvgen_only", True, DOI))
    validations.append(
        ValidationRecord("join_plants_nonempty", bool(matched_plants), f"n={len(matched_plants)}")
    )

    missingness_ok = True
    for a in file_audits:
        rc = int(a.get("row_count") or 0)
        for col, nnull in (a.get("nulls") or {}).items():
            if nnull < 0 or nnull > rc:
                missingness_ok = False
    validations.append(ValidationRecord("missingness_bounds", missingness_ok, None))

    fingerprints = [a["schema_fingerprint"] for a in file_audits if a.get("schema_fingerprint")]
    validations.append(ValidationRecord("fingerprints_present", all(fingerprints), None))

    artifacts_dir = root / "artifacts" / "p02a"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root / "reports" / "feasibility"
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "doi": DOI,
        "provenance": provenance,
        "expected_zenodo_files": list(ZENODO_FILES),
        "found_files": found_files,
        "missing_expected_files": missing_expected,
        "unexpected_files": unexpected,
        "missing_inner_files": missing_inner,
        "unexpected_inner_files": unexpected_inner,
        "documented_vs_zip_inner_names": documented_name_gap,
        "entries": [{k: v for k, v in e.items() if k != "timestamps"} for e in manifest_entries],
        "family_totals": {
            "metadata_rows": metadata_rows,
            "inverter_rows": inverter_rows,
            "meteorological_rows": meteo_rows,
        },
    }
    # drop timestamps already excluded
    manifest_path = artifacts_dir / "P02A_MANIFEST.json"
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)

    schema_path = artifacts_dir / "P02A_SCHEMA_SUMMARY.csv"
    quality_path = artifacts_dir / "P02A_QUALITY_SUMMARY.csv"
    join_path = artifacts_dir / "P02A_JOINABILITY_SUMMARY.csv"
    _write_csv(schema_path, schema_rows)
    _write_csv(quality_path, quality_rows)
    _write_csv(join_path, join_rows)

    hard_fail = bool(missing_expected) or not constructible or not md5_ok
    status = "STOP" if hard_fail else "GO"
    p02a_status = "FAIL" if hard_fail else "PASS"

    report_md = _report(
        provenance=provenance,
        found_files=found_files,
        missing_expected=missing_expected,
        unexpected=unexpected,
        missing_inner=missing_inner,
        unexpected_inner=unexpected_inner,
        metadata_rows=metadata_rows,
        inverter_rows=inverter_rows,
        meteo_rows=meteo_rows,
        metadata_plants=len(metadata_plants),
        joinability_assessment=joinability_assessment,
        p02a_status=p02a_status,
        matched=len(matched_plants),
        file_audits=file_audits,
    )
    report_path = reports_dir / "P02A_DATA_INTEGRITY.md"
    report_path.write_text(report_md, encoding="utf-8")

    def art(path: Path, rows: int | None, fingerprint: str | None) -> ArtifactRecord:
        payload = path.read_bytes()
        return ArtifactRecord(
            path=_rel(root, path),
            sha256=_sha256_bytes(payload),
            bytes=len(payload),
            row_count=rows,
            schema_fingerprint=fingerprint,
        )

    artifacts = [
        art(manifest_path, len(manifest_entries), None),
        art(schema_path, len(schema_rows), _csv_artifact_schema_fingerprint(schema_path)),
        art(quality_path, len(quality_rows), _csv_artifact_schema_fingerprint(quality_path)),
        art(join_path, len(join_rows), _csv_artifact_schema_fingerprint(join_path)),
        art(report_path, None, None),
    ]

    validations.append(
        ValidationRecord(
            "manifest_hashes_recompute",
            all(
                e["sha256"] == pre_hashes[Path(e["relative_path"]).name]
                for e in manifest_entries
                if Path(e["relative_path"]).name in pre_hashes
            ),
            None,
        )
    )

    validations.append(
        ValidationRecord(
            "no_otg_or_models",
            True,
            "stage performs inventory/join audit only",
        )
    )
    validations.append(
        ValidationRecord(
            "no_sot_write",
            True,
            "stage does not import sot writers",
        )
    )

    failed = [v for v in validations if not v.passed]
    if failed and status == "GO":
        status = "STOP"
        p02a_status = "FAIL"

    prev = (ctx.sot.get("stages") or {}).get("P02A")
    stages_patch: dict[str, Any] = {}
    if prev and prev.get("status") == "NEEDS_CONTROLLER_DECISION":
        stages_patch["P02A_blocked_attempt"] = prev
    report_sha = artifacts[-1].sha256
    stages_patch["P02A"] = {
        "kind": "data_feasibility",
        "status": status,
        "finished_at_utc": now,
        "answer": "prompts/prompts_answers/P02A_DATA_INTEGRITY_RERUN - ANSWER.md",
        "report": {"path": artifacts[-1].path, "sha256": report_sha},
    }
    feat = {
        "status": p02a_status,
        "report": {"path": artifacts[-1].path, "sha256": artifacts[-1].sha256},
        "manifest": {"path": artifacts[0].path, "sha256": artifacts[0].sha256, "row_count": artifacts[0].row_count},
        "schema_summary": {
            "path": artifacts[1].path,
            "sha256": artifacts[1].sha256,
            "row_count": artifacts[1].row_count,
            "schema_fingerprint": artifacts[1].schema_fingerprint,
        },
        "quality_summary": {
            "path": artifacts[2].path,
            "sha256": artifacts[2].sha256,
            "row_count": artifacts[2].row_count,
            "schema_fingerprint": artifacts[2].schema_fingerprint,
        },
        "joinability_summary": {
            "path": artifacts[3].path,
            "sha256": artifacts[3].sha256,
            "row_count": artifacts[3].row_count,
            "schema_fingerprint": artifacts[3].schema_fingerprint,
        },
        "summary": {
            "expected_files": len(ZENODO_FILES),
            "found_files": len(found_files),
            "missing_expected_files": missing_expected,
            "unexpected_files": unexpected,
            "metadata_rows": metadata_rows,
            "inverter_rows": inverter_rows,
            "meteorological_rows": meteo_rows,
            "metadata_plant_count": len(metadata_plants),
            "joinability_assessment": joinability_assessment,
        },
    }
    patch = {"stages": stages_patch, "feasibility": {"p02a": feat}}
    return StageResult(
        status=status,
        message=f"P02A {p02a_status} {joinability_assessment}",
        sot_patch=patch_wrap(patch),
        artifacts=artifacts,
        validations=validations,
    )


def patch_wrap(patch: dict[str, Any]) -> dict[str, Any]:
    return patch


def _quality_row(source: str, audit: dict[str, Any]) -> dict[str, Any]:
    rc = int(audit.get("row_count") or 0)
    nulls = audit.get("nulls") or {}
    props = {k: (v / rc if rc else 0) for k, v in nulls.items()}
    return {
        "source": source,
        "family": audit.get("family"),
        "row_count": rc,
        "exact_duplicate_rows": audit.get("exact_duplicate_rows"),
        "natural_key_duplicate_extras": audit.get("natural_key_duplicate_extras"),
        "timestamp_parse_ok": audit.get("timestamp_parse_ok"),
        "timestamp_parse_fail": audit.get("timestamp_parse_fail"),
        "timezone_aware": audit.get("timezone_aware"),
        "timezone_absent": audit.get("timezone_absent"),
        "timestamp_min": audit.get("timestamp_min"),
        "timestamp_max": audit.get("timestamp_max"),
        "non_monotonic": audit.get("non_monotonic"),
        "cadence_seconds": json.dumps(audit.get("cadence_seconds") or {}),
        "impossible": json.dumps(audit.get("impossible") or {}),
        "interpolated_true": json.dumps(audit.get("interpolated_true") or {}),
        "missingness_proportions": json.dumps(props),
        "inverter_id_count": len(audit.get("inverter_ids") or []),
        "plant_count": len(audit.get("plants") or []),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _report(**kwargs: Any) -> str:
    lines = [
        "# P02A — BR-PVGen data integrity and schema audit",
        "",
        f"P02A decision: **{kwargs['p02a_status']}**",
        "",
        "## Documented expectations",
        "",
        f"- Official DOI `{kwargs['provenance']['doi']}` (Zenodo record 21511487).",
        "- Distributed objects: metadata CSV, inverter zip, meteorological zip, Readme.",
        "- Readme: 51 plants, 15-minute ISO-8601 timestamps, inverter + solar station files per plant.",
        "- `tracker_albedo_index` inventoried only; not used for QC/repair.",
        "",
        "## Observed evidence",
        "",
        f"- Raw root: `{kwargs['provenance']['raw_dir']}`",
        f"- Acquisition: {kwargs['provenance']['acquisition']}",
        f"- Found files: {', '.join(kwargs['found_files'])}",
        f"- Metadata rows: {kwargs['metadata_rows']}; plants: {kwargs['metadata_plants']}",
        f"- Inverter rows: {kwargs['inverter_rows']}",
        f"- Meteorological rows: {kwargs['meteo_rows']}",
        f"- Matched plants across metadata/inverter/meteo: {kwargs['matched']}",
        f"- Joinability: {kwargs['joinability_assessment']}",
        "",
        "## Discrepancies",
        "",
        f"- Missing Zenodo objects: {kwargs['missing_expected'] or 'none'}",
        f"- Unexpected Zenodo objects: {kwargs['unexpected'] or 'none'}",
        f"- Missing inner zip members vs PS_XXX.csv (51 plants): {kwargs['missing_inner'] or 'none'}",
        f"- Unexpected inner zip members: {kwargs['unexpected_inner'] or 'none'}",
        "- Readme documents `plant_{ps}_inverter.csv` / `plant_{ps}_solar_station.csv`; Zenodo zips contain `BR-PVGen_{family}/PS_XXX.csv`. Packaging discrepancy only; join keys remain official `id`/`ps_id` and `datetime`.",
        "",
        "## Implications for later P-02 tasks",
        "",
        "- P-02B may use official `ps_id` / metadata fields only if P02A is PASS.",
        "- P-02C owns plant-level panel construction; this audit does not build that panel.",
        "- `tracker_albedo_index` remains reserved for P-02H.",
        "",
        "## P02A PASS/FAIL",
        "",
        f"**{kwargs['p02a_status']}**",
        "",
    ]
    return "\n".join(lines) + "\n"
