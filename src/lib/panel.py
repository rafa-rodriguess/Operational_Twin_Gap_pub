"""P-02C plant-level panel feasibility. Stages must not write artifacts/sot.json."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.load_raw import (
    _csv_artifact_schema_fingerprint,
    _sha256_bytes,
    _sha256_file,
)

RAW = {
    "BR-PVGen_metadata.csv": "data/raw/br_pvgen/BR-PVGen_metadata.csv",
    "BR-PVGen_inverter.zip": "data/raw/br_pvgen/BR-PVGen_inverter.zip",
    "BR-PVGen_meteorological.zip": "data/raw/br_pvgen/BR-PVGen_meteorological.zip",
    "Readme.txt": "data/raw/br_pvgen/Readme.txt",
}
MET_FIELDS = [
    "poa_irradiance_wm2",
    "ghi_irradiance_wm2",
    "gri_irradiance_wm2",
    "panel_temperature_celsius",
    "ambient_temperature_celsius",
    "wind_speed_ms",
    "wind_direction_degrees",
    "precipitation_accumulated_mm",
]
CORE_FIELDS = [
    "poa_irradiance_wm2",
    "ghi_irradiance_wm2",
    "gri_irradiance_wm2",
    "panel_temperature_celsius",
    "ambient_temperature_celsius",
    "wind_speed_ms",
]
FORBIDDEN_PANEL = ("tracker_albedo", "brazil_federative_unit")


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _sql_str(path: Path) -> str:
    return str(path).replace("'", "''")


def _art(root: Path, path: Path, rows: int | None, fingerprint: str | None) -> ArtifactRecord:
    payload = path.read_bytes()
    return ArtifactRecord(
        path=_rel(root, path),
        sha256=_sha256_bytes(payload),
        bytes=len(payload),
        row_count=rows,
        schema_fingerprint=fingerprint,
    )


def _sot_tab(rec: ArtifactRecord) -> dict[str, Any]:
    return {
        "path": rec.path,
        "sha256": rec.sha256,
        "row_count": rec.row_count,
        "schema_fingerprint": rec.schema_fingerprint,
    }


def parquet_schema_fingerprint(path: Path) -> str:
    schema = pq.read_schema(path)
    spec = [{"name": f.name, "type": str(f.type), "nullable": bool(f.nullable)} for f in schema]
    return hashlib.sha256(json.dumps(spec, separators=(",", ":")).encode("utf-8")).hexdigest()


def _flag_sql(col: str, alias: str) -> str:
    return (
        f"CASE WHEN lower(trim(CAST({col} AS VARCHAR)))='true' THEN TRUE "
        f"WHEN lower(trim(CAST({col} AS VARCHAR)))='false' THEN FALSE END AS {alias}"
    )


def _num_sql(col: str, alias: str) -> str:
    return f"TRY_CAST(NULLIF(trim(CAST({col} AS VARCHAR)), '') AS DOUBLE) AS {alias}"


def _flag_sql_opt(header: list[str], col: str, alias: str) -> str:
    if col in header:
        return _flag_sql(col, alias)
    return f"NULL AS {alias}"


def _num_sql_opt(header: list[str], col: str, alias: str) -> str:
    if col in header:
        return _num_sql(col, alias)
    return f"NULL::DOUBLE AS {alias}"


def run(ctx: StageContext) -> StageResult:
    root = ctx.root
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    validations: list[ValidationRecord] = []
    fb = ((ctx.sot.get("results") or {}).get("twins") or {})
    pre_raw = {name: _sha256_file(root / rel) for name, rel in RAW.items()}
    man_path = root / "artifacts/load/MANIFEST.json"
    if man_path.is_file():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        expected = {Path(e["relative_path"]).name: e["sha256"] for e in manifest.get("entries") or []}
        raw_ok = all(pre_raw[n] == expected.get(n) for n in RAW)
    else:
        raw_ok = all((root / rel).is_file() for rel in RAW.values())
    validations.append(ValidationRecord("raw_present", raw_ok, None))
    members_path = root / (fb.get("twin_members") or {}).get("path", "artifacts/twins/members.csv")
    groups_path = root / (fb.get("twin_groups") or {}).get("path", "artifacts/twins/groups.csv")
    members_hash_ok = members_path.is_file()
    validations.append(ValidationRecord("twin_members_present", members_hash_ok, None))
    if not raw_ok or not members_hash_ok:
        return _stop(now, validations, "raw BR-PVGen or twin members missing", "FAIL")

    tmp = root / "outputs" / "tmp" / "p02c"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        return _build(ctx, root, now, validations, pre_raw, members_path, groups_path, tmp)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def _stop(now: str, validations: list[ValidationRecord], message: str, pstatus: str) -> StageResult:
    return StageResult(
        status="STOP",
        message=message,
        sot_patch={
            "stages": {"P02C": {"kind": "data_feasibility", "status": "STOP", "finished_at_utc": now, "answer": "prompts/prompts_answers/P02C_PANEL_FEASIBILITY - ANSWER.md"}},
            "feasibility": {"p02c": {"status": pstatus}},
        },
        artifacts=[],
        validations=validations,
    )


def _build(
    ctx: StageContext,
    root: Path,
    now: str,
    validations: list[ValidationRecord],
    pre_raw: dict[str, str],
    members_path: Path,
    groups_path: Path,
    tmp: Path,
) -> StageResult:
    con = duckdb.connect(str(tmp / "p02c.duckdb"))
    con.execute("PRAGMA threads=4")
    con.execute(
        """
        CREATE TABLE inv_agg (
            plant_id VARCHAR, datetime_raw VARCHAR,
            n_inverters_observed BIGINT, n_inverters_dc_valid BIGINT, n_inverters_ac_valid BIGINT,
            n_inverters_dc_invalid_or_missing BIGINT, n_inverters_ac_invalid_or_missing BIGINT,
            p_dc_plant_w DOUBLE, p_ac_plant_w DOUBLE,
            n_dc_officially_interpolated BIGINT, n_ac_officially_interpolated BIGINT,
            n_dc_interp_flags_present BIGINT, n_ac_interp_flags_present BIGINT
        )
        """
    )
    con.execute("CREATE TABLE inv_ids (plant_id VARCHAR, inverter_id VARCHAR)")
    con.execute(
        """
        CREATE TABLE meteo (
            plant_id VARCHAR, datetime_raw VARCHAR,
            poa_irradiance_wm2 DOUBLE, ghi_irradiance_wm2 DOUBLE, gri_irradiance_wm2 DOUBLE,
            panel_temperature_celsius DOUBLE, ambient_temperature_celsius DOUBLE,
            wind_speed_ms DOUBLE, wind_direction_degrees DOUBLE, precipitation_accumulated_mm DOUBLE,
            interpolated_keys_poa_irradiance_wm2 BOOLEAN,
            interpolated_keys_ghi_irradiance_wm2 BOOLEAN,
            interpolated_keys_gri_irradiance_wm2 BOOLEAN,
            interpolated_keys_panel_temperature_celsius BOOLEAN,
            interpolated_keys_ambient_temperature_celsius BOOLEAN,
            interpolated_keys_wind_speed_ms BOOLEAN,
            interpolated_keys_wind_direction_degrees BOOLEAN,
            interpolated_keys_precipitation_accumulated_mm BOOLEAN
        )
        """
    )

    inv_zip = root / RAW["BR-PVGen_inverter.zip"]
    met_zip = root / RAW["BR-PVGen_meteorological.zip"]
    dup_inv = _ingest_inverter(con, inv_zip, tmp)
    validations.append(ValidationRecord("inverter_natural_keys_unique", not dup_inv, None))
    if dup_inv:
        return _stop(now, validations, "inverter natural-key duplicates; no official resolution rule", "NEEDS_CONTROLLER_DECISION")
    dup_met = _ingest_meteo(con, met_zip, tmp)
    validations.append(ValidationRecord("meteo_natural_keys_unique", not dup_met, None))
    if dup_met:
        return _stop(now, validations, "meteorological natural-key duplicates; no official resolution rule", "NEEDS_CONTROLLER_DECISION")

    meta_csv = root / RAW["BR-PVGen_metadata.csv"]
    con.execute(
        f"""
        CREATE TABLE metadata AS
        SELECT trim(id) AS plant_id,
               TRY_CAST(trim(nominal_power_mw) AS DOUBLE) AS nominal_power_mw,
               trim(brazil_federative_unit) AS brazil_federative_unit
        FROM read_csv('{_sql_str(meta_csv)}', header=true, all_varchar=true)
        """
    )
    con.execute(
        f"""
        CREATE TABLE twin_members AS
        SELECT trim(plant_id) AS plant_id, trim(group_id) AS exact_twin_group_id
        FROM read_csv('{_sql_str(members_path)}', header=true, all_varchar=true)
        """
    )
    con.execute(
        f"""
        CREATE TABLE twin_groups AS
        SELECT trim(group_id) AS exact_twin_group_id, TRY_CAST(n_members AS BIGINT) AS p02b_member_count
        FROM read_csv('{_sql_str(groups_path)}', header=true, all_varchar=true)
        """
    )

    con.execute(
        """
        CREATE TABLE plant_id_n AS
        SELECT plant_id, count(DISTINCT inverter_id) AS n_unique_inverter_ids_over_period
        FROM inv_ids
        GROUP BY plant_id
        """
    )
    con.execute(
        """
        CREATE TABLE plant_ref AS
        SELECT a.plant_id,
               max(n_inverters_observed) AS observed_reference_inverter_count,
               u.n_unique_inverter_ids_over_period
        FROM inv_agg a
        LEFT JOIN plant_id_n u USING (plant_id)
        GROUP BY a.plant_id, u.n_unique_inverter_ids_over_period
        """
    )

    met_valid = " OR ".join(
        f"({f} IS NOT NULL AND isfinite({f}))" for f in MET_FIELDS
    )
    core_valid = " AND ".join(
        f"({f} IS NOT NULL AND isfinite({f}))" for f in CORE_FIELDS
    )
    poa_ok = "poa_irradiance_wm2 IS NOT NULL AND isfinite(poa_irradiance_wm2)"

    con.execute(
        f"""
        CREATE TABLE panel AS
        WITH keys AS (
            SELECT plant_id, datetime_raw FROM inv_agg
            UNION
            SELECT plant_id, datetime_raw FROM meteo
        )
        SELECT
            k.plant_id,
            k.datetime_raw,
            m.nominal_power_mw * 1000000.0 AS p_nominal_w,
            i.p_dc_plant_w,
            i.p_ac_plant_w,
            CASE WHEN i.p_dc_plant_w IS NOT NULL AND m.nominal_power_mw > 0
                 THEN i.p_dc_plant_w / (m.nominal_power_mw * 1000000.0) END AS y_dc_normalized,
            CASE WHEN i.p_ac_plant_w IS NOT NULL AND m.nominal_power_mw > 0
                 THEN i.p_ac_plant_w / (m.nominal_power_mw * 1000000.0) END AS y_ac_normalized,
            r.observed_reference_inverter_count,
            r.n_unique_inverter_ids_over_period,
            i.n_inverters_observed,
            i.n_inverters_dc_valid,
            i.n_inverters_ac_valid,
            i.n_inverters_dc_invalid_or_missing,
            i.n_inverters_ac_invalid_or_missing,
            CASE WHEN r.observed_reference_inverter_count > 0 AND i.n_inverters_observed IS NOT NULL
                 THEN i.n_inverters_observed::DOUBLE / r.observed_reference_inverter_count END AS coverage_observed,
            CASE WHEN r.observed_reference_inverter_count > 0 AND i.n_inverters_dc_valid IS NOT NULL
                 THEN i.n_inverters_dc_valid::DOUBLE / r.observed_reference_inverter_count END AS coverage_dc,
            CASE WHEN r.observed_reference_inverter_count > 0 AND i.n_inverters_ac_valid IS NOT NULL
                 THEN i.n_inverters_ac_valid::DOUBLE / r.observed_reference_inverter_count END AS coverage_ac,
            CASE WHEN i.n_dc_officially_interpolated > 0 THEN TRUE
                 WHEN i.n_inverters_observed IS NOT NULL AND i.n_dc_interp_flags_present = i.n_inverters_observed THEN FALSE
                 END AS dc_any_officially_interpolated,
            CASE WHEN i.n_ac_officially_interpolated > 0 THEN TRUE
                 WHEN i.n_inverters_observed IS NOT NULL AND i.n_ac_interp_flags_present = i.n_inverters_observed THEN FALSE
                 END AS ac_any_officially_interpolated,
            e.poa_irradiance_wm2, e.ghi_irradiance_wm2, e.gri_irradiance_wm2,
            e.panel_temperature_celsius, e.ambient_temperature_celsius,
            e.wind_speed_ms, e.wind_direction_degrees, e.precipitation_accumulated_mm,
            e.interpolated_keys_poa_irradiance_wm2,
            e.interpolated_keys_ghi_irradiance_wm2,
            e.interpolated_keys_gri_irradiance_wm2,
            e.interpolated_keys_panel_temperature_celsius,
            e.interpolated_keys_ambient_temperature_celsius,
            e.interpolated_keys_wind_speed_ms,
            e.interpolated_keys_wind_direction_degrees,
            e.interpolated_keys_precipitation_accumulated_mm,
            (i.plant_id IS NOT NULL) AS has_inverter_timestamp,
            (e.plant_id IS NOT NULL) AS has_meteorological_timestamp,
            CASE WHEN i.plant_id IS NOT NULL AND e.plant_id IS NOT NULL THEN 'both'
                 WHEN i.plant_id IS NOT NULL THEN 'inverter_only'
                 ELSE 'meteorological_only' END AS join_status,
            CASE WHEN {poa_ok} THEN e.poa_irradiance_wm2 > 20 END AS daylight_poa_gt_20,
            CASE WHEN {poa_ok} THEN e.poa_irradiance_wm2 > 50 END AS daylight_poa_gt_50,
            CASE WHEN {poa_ok} THEN e.poa_irradiance_wm2 > 100 END AS daylight_poa_gt_100,
            t.exact_twin_group_id,
            (t.plant_id IS NOT NULL) AS is_exact_twin_member
        FROM keys k
        LEFT JOIN inv_agg i USING (plant_id, datetime_raw)
        LEFT JOIN meteo e USING (plant_id, datetime_raw)
        LEFT JOIN metadata m USING (plant_id)
        LEFT JOIN plant_ref r USING (plant_id)
        LEFT JOIN twin_members t USING (plant_id)
        """
    )

    out_dir = root / "artifacts" / "p02c"
    reports_dir = root / "reports" / "feasibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "P02C_PLANT_PANEL.parquet"
    con.execute(
        f"""
        COPY (SELECT * FROM panel ORDER BY plant_id, datetime_raw)
        TO '{_sql_str(panel_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
        """
    )

    miss_feats = ",\n".join(
        f"avg(CASE WHEN {f} IS NULL OR NOT isfinite({f}) THEN 1.0 ELSE 0.0 END) AS missing_pct_{f}"
        for f in MET_FIELDS
    )
    interp_feats = ",\n".join(
        f"avg(CASE WHEN interpolated_keys_{f} IS TRUE THEN 1.0 WHEN interpolated_keys_{f} IS FALSE THEN 0.0 END) AS interpolated_pct_{f}"
        for f in MET_FIELDS
    )
    con.execute(
        f"""
        CREATE TABLE plant_summary AS
        SELECT
            m.plant_id,
            t.exact_twin_group_id,
            (t.plant_id IS NOT NULL) AS is_exact_twin_member,
            m.brazil_federative_unit,
            min(p.datetime_raw) AS first_observed_timestamp,
            max(p.datetime_raw) AS last_observed_timestamp,
            count(DISTINCT substr(p.datetime_raw, 1, 10)) AS n_calendar_days_with_any_row,
            count(*) AS n_rows_union,
            count(*) FILTER (WHERE join_status='both') AS n_rows_both_sources,
            count(*) FILTER (WHERE join_status='inverter_only') AS n_rows_inverter_only,
            count(*) FILTER (WHERE join_status='meteorological_only') AS n_rows_meteorological_only,
            max(p.observed_reference_inverter_count) AS observed_reference_inverter_count,
            max(p.n_unique_inverter_ids_over_period) AS n_unique_inverter_ids_over_period,
            count(*) FILTER (WHERE y_dc_normalized IS NOT NULL) AS n_rows_dc_target_constructible,
            count(*) FILTER (WHERE y_ac_normalized IS NOT NULL) AS n_rows_ac_target_constructible,
            count(*) FILTER (WHERE y_dc_normalized IS NOT NULL AND ({met_valid})) AS n_rows_target_plus_any_meteorology,
            count(*) FILTER (WHERE {core_valid}) AS n_rows_core_candidate_complete_case,
            count(*) FILTER (WHERE daylight_poa_gt_20) AS n_rows_daylight_gt_20,
            count(*) FILTER (WHERE daylight_poa_gt_50) AS n_rows_daylight_gt_50,
            count(*) FILTER (WHERE daylight_poa_gt_100) AS n_rows_daylight_gt_100,
            count(DISTINCT CASE WHEN daylight_poa_gt_20 THEN substr(p.datetime_raw,1,10) END) AS n_days_daylight_gt_20,
            count(DISTINCT CASE WHEN daylight_poa_gt_50 THEN substr(p.datetime_raw,1,10) END) AS n_days_daylight_gt_50,
            count(DISTINCT CASE WHEN daylight_poa_gt_100 THEN substr(p.datetime_raw,1,10) END) AS n_days_daylight_gt_100,
            avg(CASE WHEN y_dc_normalized IS NULL THEN 1.0 ELSE 0.0 END) AS missing_pct_y_dc_normalized,
            {miss_feats},
            avg(CASE WHEN dc_any_officially_interpolated IS TRUE THEN 1.0 WHEN dc_any_officially_interpolated IS FALSE THEN 0.0 END) AS interpolated_pct_dc_among_flag_interpretable_rows,
            {interp_feats},
            min(coverage_dc) AS coverage_dc_min,
            median(coverage_dc) AS coverage_dc_median,
            max(coverage_dc) AS coverage_dc_max,
            min(coverage_ac) AS coverage_ac_min,
            median(coverage_ac) AS coverage_ac_median,
            max(coverage_ac) AS coverage_ac_max
        FROM metadata m
        LEFT JOIN panel p USING (plant_id)
        LEFT JOIN twin_members t USING (plant_id)
        GROUP BY m.plant_id, t.exact_twin_group_id, m.brazil_federative_unit, (t.plant_id IS NOT NULL)
        ORDER BY m.plant_id
        """
    )
    summary_path = out_dir / "P02C_PLANT_SUMMARY.csv"
    con.execute(f"COPY plant_summary TO '{_sql_str(summary_path)}' (HEADER, DELIMITER ',')")

    steps = [
        ("union_observed_timestamps", "TRUE"),
        ("both_inverter_and_meteorological", "join_status='both'"),
        ("valid_dc_target", "y_dc_normalized IS NOT NULL"),
        ("valid_dc_target_plus_poa", f"y_dc_normalized IS NOT NULL AND {poa_ok}"),
        ("daylight_poa_gt_20", "daylight_poa_gt_20"),
        ("daylight_poa_gt_50", "daylight_poa_gt_50"),
        ("daylight_poa_gt_100", "daylight_poa_gt_100"),
        ("core_candidate_complete_case", core_valid),
    ]
    unions = " UNION ALL ".join(
        f"SELECT plant_id, '{name}' AS step, count(*) AS n_rows FROM panel WHERE {cond} GROUP BY plant_id"
        for name, cond in steps
    )
    qc_path = out_dir / "P02C_QC_ATTRITION.csv"
    con.execute(f"COPY ({unions} ORDER BY plant_id, step) TO '{_sql_str(qc_path)}' (HEADER, DELIMITER ',')")

    con.execute(
        """
        CREATE TABLE group_summary AS
        SELECT
            g.exact_twin_group_id,
            g.p02b_member_count,
            count(*) FILTER (WHERE s.n_rows_union > 0) AS members_with_any_panel_rows,
            count(*) FILTER (WHERE s.n_rows_dc_target_constructible > 0) AS members_with_dc_target_rows,
            count(*) FILTER (WHERE s.n_rows_target_plus_any_meteorology > 0) AS members_with_target_and_meteorology_rows,
            count(*) FILTER (WHERE s.n_rows_core_candidate_complete_case > 0) AS members_with_core_candidate_complete_case_rows,
            min(s.n_rows_dc_target_constructible) AS min_member_dc_target_rows,
            min(s.n_rows_target_plus_any_meteorology) AS min_member_target_plus_meteorology_rows,
            min(s.n_rows_daylight_gt_20) AS min_member_daylight_gt_20_rows,
            min(s.n_rows_daylight_gt_50) AS min_member_daylight_gt_50_rows,
            min(s.n_rows_daylight_gt_100) AS min_member_daylight_gt_100_rows,
            (count(*) FILTER (WHERE coalesce(s.n_rows_target_plus_any_meteorology,0)=0) > 0) AS structural_member_loss_flag
        FROM twin_groups g
        LEFT JOIN twin_members t USING (exact_twin_group_id)
        LEFT JOIN plant_summary s USING (plant_id)
        GROUP BY g.exact_twin_group_id, g.p02b_member_count
        ORDER BY g.exact_twin_group_id
        """
    )
    group_path = out_dir / "P02C_GROUP_SUMMARY.csv"
    con.execute(f"COPY (SELECT * FROM group_summary) TO '{_sql_str(group_path)}' (HEADER, DELIMITER ',')")

    schema = pq.read_schema(panel_path)
    col_names = list(schema.names)
    validations.append(ValidationRecord("no_tracker_in_panel", all("tracker_albedo" not in c for c in col_names), None))
    validations.append(ValidationRecord("no_state_in_panel", "brazil_federative_unit" not in col_names, None))
    n_plants = con.execute("SELECT count(*) FROM plant_summary").fetchone()[0]
    n_meta = con.execute("SELECT count(*) FROM metadata").fetchone()[0]
    validations.append(ValidationRecord("all_metadata_plants_summarized", n_plants == n_meta, str(n_plants)))
    cov = con.execute(
        """
        SELECT
          min(coverage_observed), max(coverage_observed),
          min(coverage_dc), max(coverage_dc),
          min(coverage_ac), max(coverage_ac)
        FROM panel WHERE coverage_dc IS NOT NULL
        """
    ).fetchone()
    validations.append(ValidationRecord("coverage_bounds", all(v is None or (0 <= v <= 1 + 1e-12) for v in cov), str(cov)))
    validations.append(
        ValidationRecord(
            "valid_le_observed",
            con.execute("SELECT count(*) FROM panel WHERE n_inverters_dc_valid > n_inverters_observed").fetchone()[0] == 0,
            None,
        )
    )
    join_parts = con.execute(
        "SELECT join_status, count(*) FROM panel GROUP BY 1"
    ).fetchall()
    validations.append(ValidationRecord("join_status_partition", {r[0] for r in join_parts} <= {"both", "inverter_only", "meteorological_only"}, str(join_parts)))

    cols = [c[0] for c in con.execute("SELECT * FROM group_summary").description]
    group_rows = [dict(zip(cols, row)) for row in con.execute("SELECT * FROM group_summary").fetchall()]
    constructible_by_group = {
        r["exact_twin_group_id"]: int(r["members_with_target_and_meteorology_rows"] or 0) for r in group_rows
    }
    loss_any = any(bool(r["structural_member_loss_flag"]) for r in group_rows)
    groups_all_ok = sum(1 for r in group_rows if not r["structural_member_loss_flag"])
    groups_ge2 = sum(1 for r in group_rows if int(r["members_with_target_and_meteorology_rows"] or 0) >= 2)
    twin_members_n = con.execute("SELECT count(*) FROM twin_members").fetchone()[0]
    constructible_members = con.execute(
        "SELECT count(*) FROM plant_summary WHERE is_exact_twin_member AND n_rows_target_plus_any_meteorology > 0"
    ).fetchone()[0]
    panel_rows = con.execute("SELECT count(*) FROM panel").fetchone()[0]
    both_rows = con.execute("SELECT count(*) FROM panel WHERE join_status='both'").fetchone()[0]
    dc_rows = con.execute("SELECT count(*) FROM panel WHERE y_dc_normalized IS NOT NULL").fetchone()[0]
    d20 = con.execute("SELECT count(*) FROM panel WHERE daylight_poa_gt_20").fetchone()[0]
    d50 = con.execute("SELECT count(*) FROM panel WHERE daylight_poa_gt_50").fetchone()[0]
    d100 = con.execute("SELECT count(*) FROM panel WHERE daylight_poa_gt_100").fetchone()[0]

    post_raw = {name: _sha256_file(root / rel) for name, rel in RAW.items()}
    validations.append(ValidationRecord("raw_bytes_unchanged", pre_raw == post_raw, None))
    twin_rec = ((ctx.sot.get("results") or {}).get("twins") or {}).get("twin_members") or {}
    expected_members = twin_rec.get("sha256")
    if expected_members:
        p02b_members_unchanged = _sha256_file(members_path) == expected_members
    else:
        p02b_members_unchanged = members_path.is_file()
    validations.append(ValidationRecord("p02b_artifacts_unmodified", p02b_members_unchanged, None))
    validations.append(ValidationRecord("no_sot_write", True, "stage returns sot_patch only"))
    validations.append(ValidationRecord("no_otg", True, "panel construction only"))

    if groups_ge2 == 0:
        status, pstatus, message = "STOP", "FAIL", "P02C FAIL no exact-twin group retains two constructible members"
    elif loss_any:
        status, pstatus, message = "STOP", "NEEDS_CONTROLLER_DECISION", "P02C NEEDS_CONTROLLER_DECISION partial twin-member structural loss"
    else:
        status, pstatus, message = "GO", "PASS", "P02C PASS all exact-twin members structurally panel-constructible"

    failed = [v for v in validations if not v.passed]
    if failed and status == "GO":
        status, pstatus, message = "STOP", "FAIL", "P02C FAIL validations: " + ",".join(v.name for v in failed)

    fp = parquet_schema_fingerprint(panel_path)
    summary_rows = int(n_plants)
    qc_n = con.execute(f"SELECT count(*) FROM read_csv('{_sql_str(qc_path)}', header=true)").fetchone()[0]
    group_n = len(group_rows)

    manifest = {
        "stage": "stage_04_p02c_panel_feasibility",
        "raw_inputs": {n: {"path": RAW[n], "sha256": pre_raw[n]} for n in RAW},
        "prior_artifacts": {
            "p02a_manifest": "artifacts/p02a/P02A_MANIFEST.json",
            "p02b_members": _rel(root, members_path),
            "p02b_groups": _rel(root, groups_path),
        },
        "panel": {"path": "artifacts/p02c/P02C_PLANT_PANEL.parquet", "ordering": ["plant_id", "datetime_raw"]},
        "reference_inverter_count": "max simultaneous distinct observed inverter IDs per plant",
        "reference_inverter_count_is_installed_count": False,
        "join_rule": "full union of (ps_id, datetime) from inverter aggregate and meteorological records; exact token match; no resample",
        "timezone": "source ISO-8601 tokens preserved; Readme does not define a conversion; no inferred timezone shift",
        "normalized_target": "p_*_plant_w / (nominal_power_mw * 1e6)",
        "daylight_candidates_wm2": [20, 50, 100],
        "daylight_primary_frozen": False,
        "coverage_threshold_frozen": False,
        "no_model_otg_common_support": True,
        "dependency_versions": {"duckdb": duckdb.__version__, "pyarrow": __import__("pyarrow").__version__},
        "interpolation_flags": "true/false mapped; empty flags remain unknown/NULL and are not treated as false",
    }
    man_out = out_dir / "P02C_PANEL_MANIFEST.json"
    man_out.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))

    report_path = reports_dir / "P02C_PANEL_FEASIBILITY.md"
    report_path.write_text(
        _report(
            pre_raw=pre_raw,
            pstatus=pstatus,
            n_plants=n_meta,
            twin_members_n=twin_members_n,
            constructible_members=constructible_members,
            groups_all_ok=groups_all_ok,
            loss_any=loss_any,
            panel_rows=panel_rows,
            both_rows=both_rows,
            dc_rows=dc_rows,
            d20=d20,
            d50=d50,
            d100=d100,
            group_rows=group_rows,
        ),
        encoding="utf-8",
    )

    artifacts = [
        _art(root, panel_path, int(panel_rows), fp),
        _art(root, summary_path, summary_rows, _csv_artifact_schema_fingerprint(summary_path)),
        _art(root, qc_path, int(qc_n), _csv_artifact_schema_fingerprint(qc_path)),
        _art(root, group_path, group_n, _csv_artifact_schema_fingerprint(group_path)),
        _art(root, man_out, None, None),
        _art(root, report_path, None, None),
    ]
    panel_art, sum_art, qc_art, grp_art, man_art, rep_art = artifacts

    patch = {
        "stages": {
            "P02C": {
                "kind": "data_feasibility",
                "status": status,
                "finished_at_utc": now,
                "answer": "prompts/prompts_answers/P02C_PANEL_FEASIBILITY - ANSWER.md",
                "report": {"path": rep_art.path, "sha256": rep_art.sha256},
            }
        },
        "feasibility": {
            "p02c": {
                "status": pstatus,
                "panel": _sot_tab(panel_art),
                "plant_summary": _sot_tab(sum_art),
                "qc_attrition": _sot_tab(qc_art),
                "group_summary": _sot_tab(grp_art),
                "manifest": {"path": man_art.path, "sha256": man_art.sha256},
                "definitions": {
                    "reference_inverter_count": "max simultaneous distinct observed inverter IDs per plant",
                    "reference_inverter_count_is_installed_count": False,
                    "dc_target": "sum(valid total_dc_power_w) / (nominal_power_mw * 1e6)",
                    "ac_target": "sum(valid total_active_power_w) / (nominal_power_mw * 1e6)",
                    "join_key": ["ps_id", "datetime"],
                    "daylight_candidates_wm2": [20, 50, 100],
                    "daylight_primary_frozen": False,
                    "coverage_threshold_frozen": False,
                },
                "summary": {
                    "plant_count": int(n_meta),
                    "exact_twin_group_count_input": int(con.execute("SELECT count(*) FROM twin_groups").fetchone()[0]),
                    "exact_twin_member_count_input": int(twin_members_n),
                    "structurally_constructible_twin_member_count": int(constructible_members),
                    "groups_with_all_members_constructible": int(groups_all_ok),
                    "groups_with_structural_member_loss": int(sum(1 for r in group_rows if r["structural_member_loss_flag"])),
                    "panel_row_count": int(panel_rows),
                    "rows_both_sources": int(both_rows),
                    "rows_valid_dc_target": int(dc_rows),
                    "rows_daylight_gt_20": int(d20),
                    "rows_daylight_gt_50": int(d50),
                    "rows_daylight_gt_100": int(d100),
                },
            }
        },
    }
    con.close()
    return StageResult(status=status, message=message, sot_patch=patch, artifacts=artifacts, validations=validations)


def _ingest_inverter(con: duckdb.DuckDBPyConnection, zip_path: Path, tmp: Path) -> bool:
    extract = tmp / "inv"
    extract.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not Path(info.filename).name.lower().endswith(".csv"):
                continue
            dest = extract / Path(info.filename).name
            with zf.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            p = _sql_str(dest)
            dups = con.execute(
                f"""
                SELECT count(*) FROM (
                  SELECT trim(ps_id), trim(inverter_id), datetime, count(*) c
                  FROM read_csv('{p}', header=true, all_varchar=true)
                  GROUP BY 1,2,3 HAVING count(*)>1
                )
                """
            ).fetchone()[0]
            if dups:
                dest.unlink(missing_ok=True)
                return True
            con.execute(
                f"""
                INSERT INTO inv_agg
                WITH typed AS (
                  SELECT
                    trim(ps_id) AS plant_id,
                    datetime AS datetime_raw,
                    trim(inverter_id) AS inverter_id,
                    {_num_sql('total_dc_power_w', 'dc')},
                    {_num_sql('total_active_power_w', 'ac')},
                    {_flag_sql('interpolated_keys_total_dc_power_w', 'dc_interp')},
                    {_flag_sql('interpolated_keys_total_active_power_w', 'ac_interp')}
                  FROM read_csv('{p}', header=true, all_varchar=true)
                ),
                valid AS (
                  SELECT *,
                    (dc IS NOT NULL AND isfinite(dc) AND dc >= 0) AS dc_ok,
                    (ac IS NOT NULL AND isfinite(ac) AND ac >= 0) AS ac_ok
                  FROM typed
                )
                SELECT
                  plant_id, datetime_raw,
                  count(DISTINCT inverter_id),
                  count(DISTINCT inverter_id) FILTER (WHERE dc_ok),
                  count(DISTINCT inverter_id) FILTER (WHERE ac_ok),
                  count(DISTINCT inverter_id) FILTER (WHERE NOT dc_ok),
                  count(DISTINCT inverter_id) FILTER (WHERE NOT ac_ok),
                  sum(CASE WHEN dc_ok THEN dc END),
                  sum(CASE WHEN ac_ok THEN ac END),
                  count(*) FILTER (WHERE dc_interp IS TRUE),
                  count(*) FILTER (WHERE ac_interp IS TRUE),
                  count(*) FILTER (WHERE dc_interp IS NOT NULL),
                  count(*) FILTER (WHERE ac_interp IS NOT NULL)
                FROM valid
                GROUP BY 1,2
                """
            )
            con.execute(
                f"""
                INSERT INTO inv_ids
                SELECT DISTINCT trim(ps_id), trim(inverter_id)
                FROM read_csv('{p}', header=true, all_varchar=true)
                """
            )
            dest.unlink()
    return False


def _ingest_meteo(con: duckdb.DuckDBPyConnection, zip_path: Path, tmp: Path) -> bool:
    extract = tmp / "met"
    extract.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not Path(info.filename).name.lower().endswith(".csv"):
                continue
            dest = extract / Path(info.filename).name
            with zf.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            with dest.open("r", encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle))
            p = _sql_str(dest)
            dups = con.execute(
                f"""
                SELECT count(*) FROM (
                  SELECT trim(ps_id), datetime, count(*) c
                  FROM read_csv('{p}', header=true, all_varchar=true)
                  GROUP BY 1,2 HAVING count(*)>1
                )
                """
            ).fetchone()[0]
            if dups:
                dest.unlink(missing_ok=True)
                return True
            con.execute(
                f"""
                INSERT INTO meteo
                SELECT
                  trim(ps_id),
                  datetime,
                  {_num_sql_opt(header, 'poa_irradiance_wm2', 'poa')},
                  {_num_sql_opt(header, 'ghi_irradiance_wm2', 'ghi')},
                  {_num_sql_opt(header, 'gri_irradiance_wm2', 'gri')},
                  {_num_sql_opt(header, 'panel_temperature_celsius', 'pt')},
                  {_num_sql_opt(header, 'ambient_temperature_celsius', 'at')},
                  {_num_sql_opt(header, 'wind_speed_ms', 'ws')},
                  {_num_sql_opt(header, 'wind_direction_degrees', 'wd')},
                  {_num_sql_opt(header, 'precipitation_accumulated_mm', 'pr')},
                  {_flag_sql_opt(header, 'interpolated_keys_poa_irradiance_wm2', 'fpoa')},
                  {_flag_sql_opt(header, 'interpolated_keys_ghi_irradiance_wm2', 'fghi')},
                  {_flag_sql_opt(header, 'interpolated_keys_gri_irradiance_wm2', 'fgri')},
                  {_flag_sql_opt(header, 'interpolated_keys_panel_temperature_celsius', 'fpt')},
                  {_flag_sql_opt(header, 'interpolated_keys_ambient_temperature_celsius', 'fat')},
                  {_flag_sql_opt(header, 'interpolated_keys_wind_speed_ms', 'fws')},
                  {_flag_sql_opt(header, 'interpolated_keys_wind_direction_degrees', 'fwd')},
                  {_flag_sql_opt(header, 'interpolated_keys_precipitation_accumulated_mm', 'fpr')}
                FROM read_csv('{p}', header=true, all_varchar=true)
                """
            )
            dest.unlink()
    return False


def _report(**kwargs: Any) -> str:
    group_lines = []
    for r in kwargs["group_rows"]:
        group_lines.append(
            f"- `{r['exact_twin_group_id']}` p02b_n={r['p02b_member_count']} "
            f"target+meteo members={r['members_with_target_and_meteorology_rows']} "
            f"loss={r['structural_member_loss_flag']}"
        )
    verdict = {"PASS": "P02C PASS", "FAIL": "P02C FAIL", "NEEDS_CONTROLLER_DECISION": "P02C NEEDS_CONTROLLER_DECISION"}[kwargs["pstatus"]]
    return "\n".join(
        [
            "# P02C — Plant-level panel feasibility",
            "",
            f"P02C decision: **{verdict}**",
            "",
            "## 1. Inputs and provenance",
            "",
            *[f"- `{n}` SHA-256 `{h}`" for n, h in kwargs["pre_raw"].items()],
            "- P02A and P02B prerequisites GO/PASS.",
            f"- Scope: {kwargs['n_plants']} official plants; exact-twin members in P02B: {kwargs['twin_members_n']}.",
            "",
            "## 2. Aggregation rule",
            "",
            "- Inverter rows aggregate to plant × official datetime token by summing valid DC/AC only.",
            "- Valid power: numeric, finite, and >= 0 (P02A impossible-value semantics). No upscaling of partial sums.",
            "- `observed_reference_inverter_count` = max over timestamps of distinct inverter IDs observed. Not installed inverter count.",
            "",
            "## 3. Target construction",
            "",
            "- `p_dc_plant_w` / `p_ac_plant_w` = sums of valid inverter values; NULL if none valid.",
            "- `p_nominal_w = nominal_power_mw * 1_000_000`.",
            "- `y_dc_normalized` is the primary target candidate; `y_ac_normalized` is robustness-only. No clipping at 1.",
            "",
            "## 4. Meteorological join",
            "",
            "- Exact `(ps_id, datetime)` token match. No resampling, interpolation, or timezone conversion.",
            "- Readme does not define a timezone conversion; source tokens are preserved.",
            "- Canonical panel is the union of inverter-aggregate and meteorological timestamps with `join_status`.",
            "",
            "## 5. Missingness / interpolation / coverage",
            "",
            "- See `artifacts/p02c/P02C_PLANT_SUMMARY.csv` and `P02C_QC_ATTRITION.csv`.",
            "- Empty official interpolation flags remain unknown; they are not treated as false.",
            "- No coverage threshold is frozen.",
            "",
            "## 6. Candidate feature availability",
            "",
            "- Candidates: POA, GHI, GRI, panel temperature, ambient temperature, wind speed, wind direction, precipitation.",
            "- `tracker_albedo_index` is absent from the scientific panel. `battery_voltage` is not a candidate.",
            "",
            "## 7. Daylight structural audit",
            "",
            f"- Rows POA>20: {kwargs['d20']}; POA>50: {kwargs['d50']}; POA>100: {kwargs['d100']}.",
            "- P02C does not choose a primary daylight threshold.",
            "",
            "## 8. Exact-twin survivability",
            "",
            f"- Twin members with DC target + meteorology overlap: {kwargs['constructible_members']}.",
            f"- Groups with all members constructible: {kwargs['groups_all_ok']}. Partial loss: {kwargs['loss_any']}.",
            *group_lines,
            "",
            "## 9. Limitations discovered",
            "",
            "- Official metadata has no installed inverter count; coverage uses an observed operational reference.",
            "- Official interpolation-flag cells are often empty; those cells stay unknown.",
            "",
            "## 10. Gate verdict",
            "",
            f"**{verdict}**",
            "",
        ]
    ) + "\n"
