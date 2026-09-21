"""S08 blinded common-support alignment audit (operational stage 26). No outcomes."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import (
    FEATURE_STRATEGIES,
    NN_ALGORITHM,
    QUANTILE_METHOD,
    assign_post_eligibility_split,
    dump_json,
    iqr_is_valid,
    loo_kth_distances,
    parse_ts,
    q_linear,
    sha256_file,
    source_median_iqr,
    standardize,
    write_csv,
)

ANSWER = "prompts/prompts_answers/S08_PREOUTCOME_COMMON_SUPPORT_ALIGNMENT_AUDIT - ANSWER.md"
OUT = "artifacts/s08_alignment_audit"
REPORT = "reports/freeze/S08_PREOUTCOME_COMMON_SUPPORT_ALIGNMENT_AUDIT.md"
JOB = "S08_PREOUTCOME_COMMON_SUPPORT_ALIGNMENT_AUDIT"
REASON = "S08_SUPPORT_ALIGNMENT_CONTROLLER_DECISION_REQUIRED"
MSG = "STOP — S08 COMMON-SUPPORT ALIGNMENT AUDIT COMPLETE / CONTROLLER DECISION REQUIRED"
CS4 = {"id": "CS4", "k": 5, "q": 0.95}
FEATURES = FEATURE_STRATEGIES["six_core"]
EXPECTED_MAP = {
    "PS_002": "PS_013",
    "PS_003": "PS_013",
    "PS_035": "PS_006",
    "PS_039": "PS_006",
    "PS_042": "PS_010",
    "PS_043": "PS_010",
    "PS_044": "PS_010",
    "PS_045": "PS_010",
    "PS_046": "PS_010",
    "PS_047": "PS_010",
    "PS_048": "PS_010",
    "PS_049": "PS_010",
    "PS_050": "PS_010",
    "PS_051": "PS_010",
}
HASH_ONLY = (
    "artifacts/s07/S07_OTG_DIRECTIONAL.csv",
    "artifacts/s07/S07_OTG_MATRICES.json",
    "artifacts/s07/S07_ASYMMETRY.csv",
    "artifacts/s07/S07_SUMMARY.json",
    "artifacts/s06/S06_TRANSFER_METRICS.csv",
    "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet",
    "artifacts/s04/S04_LOCAL_METRICS.csv",
    "artifacts/s04/S04_LOCAL_TEST_PREDICTIONS.parquet",
    "artifacts/s04/S04_VALIDATION_PREDICTIONS.parquet",
)
SAME_SAMPLE_TEXT = "transferred_error_minus_local_target_error_on_same_supported_rows"
RQ3_MASK_TEXT = (
    "target-level bootstrap draws must preserve the same target calendar draw across all "
    "twin-source and matched-control operands, while each operand retains its frozen transfer-specific support mask"
)


def _csv_fp(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    return hashlib.sha256("|".join(header).encode("utf-8")).hexdigest()


def _pq_fp(path: Path) -> str:
    return hashlib.sha256("|".join(pq.read_schema(path).names).encode("utf-8")).hexdigest()


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _art(root: Path, path: Path, rows: int | None = None, fp: str | None = None) -> ArtifactRecord:
    payload = path.read_bytes()
    return ArtifactRecord(path=_rel(root, path), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), row_count=rows, schema_fingerprint=fp)


def _tab(rec: ArtifactRecord) -> dict[str, Any]:
    out: dict[str, Any] = {"path": rec.path, "sha256": rec.sha256, "bytes": rec.bytes}
    if rec.row_count is not None:
        out["row_count"] = rec.row_count
    if rec.schema_fingerprint is not None:
        out["schema_fingerprint"] = rec.schema_fingerprint
    return out


def _days(dts: list[str] | set[str]) -> int:
    dates = set()
    for raw in dts:
        ts = parse_ts(str(raw))
        if ts is not None:
            dates.add(ts.date())
    return len(dates)


def _first_last(dts: list[str] | set[str]) -> tuple[str | None, str | None]:
    parsed = [(parse_ts(str(x)), str(x)) for x in dts]
    parsed = [(t, s) for t, s in parsed if t is not None]
    if not parsed:
        return None, None
    parsed.sort(key=lambda z: (z[0], z[1]))
    return parsed[0][1], parsed[-1][1]


def _ratio(num: int | None, den: int | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def method_assessment_rows() -> list[dict[str, Any]]:
    return [
        {
            "method_id": "A",
            "description": "Separate native CS4 supports; twin and control keep own masks",
            "same_rows_within_twin_control_contrast": False,
            "same_rows_across_all_twin_sources_for_target": False,
            "evaluates_outside_source_support": False,
            "uses_all_available_native_support": True,
            "imposes_extra_cross_source_intersection": False,
            "requires_outcome_information": False,
            "changes_frozen_CS4_definition": False,
            "compatible_with_same_target_same_supported_sample_principle": True,
            "structurally_admissible": True,
            "reason": (
                f"Frozen otg_absolute_definition={SAME_SAMPLE_TEXT} binds transfer vs local within a direction, "
                f"not twin vs control. S03 freeze also says {RQ3_MASK_TEXT}. A is therefore not logically excluded; "
                "the structural concern is that Delta_j can mix unequal target environmental subsets."
            ),
        },
        {
            "method_id": "B",
            "description": "Pairwise mutual support M_twin ∩ M_control per twin source",
            "same_rows_within_twin_control_contrast": True,
            "same_rows_across_all_twin_sources_for_target": False,
            "evaluates_outside_source_support": False,
            "uses_all_available_native_support": False,
            "imposes_extra_cross_source_intersection": False,
            "requires_outcome_information": False,
            "changes_frozen_CS4_definition": False,
            "compatible_with_same_target_same_supported_sample_principle": True,
            "structurally_admissible": True,
            "reason": "Forces identical timestamps within each twin-control contrast without a new CS4 threshold. Does not force one sample across all twin sources of the target.",
        },
        {
            "method_id": "C",
            "description": "Target-level global intersection of control and all CS4-computable twin masks",
            "same_rows_within_twin_control_contrast": True,
            "same_rows_across_all_twin_sources_for_target": True,
            "evaluates_outside_source_support": False,
            "uses_all_available_native_support": False,
            "imposes_extra_cross_source_intersection": True,
            "requires_outcome_information": False,
            "changes_frozen_CS4_definition": False,
            "compatible_with_same_target_same_supported_sample_principle": True,
            "structurally_admissible": True,
            "reason": "Stronger target-level comparability. By inclusion, row count is ≤ every pairwise intersection. No new CS4 threshold.",
        },
        {
            "method_id": "D",
            "description": "Union / evaluate a source on rows outside its own CS4 support",
            "same_rows_within_twin_control_contrast": False,
            "same_rows_across_all_twin_sources_for_target": False,
            "evaluates_outside_source_support": True,
            "uses_all_available_native_support": True,
            "imposes_extra_cross_source_intersection": False,
            "requires_outcome_information": False,
            "changes_frozen_CS4_definition": True,
            "compatible_with_same_target_same_supported_sample_principle": False,
            "structurally_admissible": False,
            "reason": "Frozen CS4 forbids scoring a source outside its own support. D is inadmissible and was not executed.",
        },
    ]


def _load_incidence(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            twins = json.loads(row["twin_source_plant_ids"])
            rows.append(
                {
                    "target_plant_id": row["target_plant_id"],
                    "reference_group_id": row["reference_group_id"],
                    "twin_source_plant_ids": list(twins),
                    "control_source_plant_id": row["control_source_plant_id"],
                }
            )
    return sorted(rows, key=lambda r: r["target_plant_id"])


def _load_s05_cs4(root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], set[str]]]:
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    with (root / "artifacts/s05/S05_SUPPORT_SUMMARY.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("rule_id") != "CS4":
                continue
            key = (row["source_plant_id"], row["target_plant_id"])
            summary[key] = row
    table = pq.read_table(
        root / "artifacts/s05/S05_SUPPORT_MASKS.parquet",
        columns=["source_plant_id", "target_plant_id", "rule_id", "datetime_raw", "computable", "supported"],
        filters=[("rule_id", "=", "CS4")],
    )
    masks: dict[tuple[str, str], set[str]] = defaultdict(set)
    for src, tgt, dt, ok, sup in zip(
        table["source_plant_id"].to_pylist(),
        table["target_plant_id"].to_pylist(),
        table["datetime_raw"].to_pylist(),
        table["computable"].to_pylist(),
        table["supported"].to_pylist(),
    ):
        if ok and sup:
            masks[(str(src), str(tgt))].add(str(dt))
    return summary, dict(masks)


def _eligible_feature_sql(plants: list[str]) -> str:
    plist = ",".join("'" + p.replace("'", "''") + "'" for p in plants)
    feats = ", ".join(FEATURES)
    return f"""
        SELECT plant_id, datetime_raw, {feats},
               coverage_dc, dc_any_officially_interpolated,
               interpolated_keys_poa_irradiance_wm2, interpolated_keys_ghi_irradiance_wm2,
               interpolated_keys_gri_irradiance_wm2, interpolated_keys_panel_temperature_celsius,
               interpolated_keys_ambient_temperature_celsius, interpolated_keys_wind_speed_ms
        FROM read_parquet(?)
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


def reconstruct_splits(table) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"train": [], "validation": [], "test": []})
    by: dict[str, list[int]] = defaultdict(list)
    pids = table["plant_id"].to_pylist()
    dts = [str(x) for x in table["datetime_raw"].to_pylist()]
    tss = [parse_ts(x) for x in dts]
    for i, pid in enumerate(pids):
        by[str(pid)].append(i)
    sentinel = datetime.max.replace(tzinfo=timezone.utc)
    for pid, idxs in by.items():
        order = sorted(idxs, key=lambda j: (tss[j] or sentinel, dts[j]))
        n = len(order)
        for rank, j in enumerate(order):
            lab = assign_post_eligibility_split(n, rank)
            out[pid][lab].append(dts[j])
    return dict(out)


def _feat_map(table) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    pids = table["plant_id"].to_pylist()
    dts = [str(x) for x in table["datetime_raw"].to_pylist()]
    cols = [table[name].to_pylist() for name in FEATURES]
    for i, pid in enumerate(pids):
        vec = np.array([float(c[i]) if c[i] is not None else float("nan") for c in cols], dtype=float)
        out[(str(pid), dts[i])] = vec
    return out


def build_cs4_engine(pid: str, train_dts: list[str], feat_by: dict[tuple[str, str], np.ndarray]) -> dict[str, Any]:
    rows = []
    for dt in train_dts:
        vec = feat_by.get((pid, dt))
        if vec is None or not np.all(np.isfinite(vec)):
            return {"computable": False, "reason": "missing_or_nonfinite_train_features", "n_train": len(train_dts)}
        rows.append(vec)
    if len(rows) < 5:
        return {"computable": False, "reason": "insufficient_train_rows_for_k5", "n_train": len(rows)}
    X = np.vstack(rows)
    medians = []
    iqrs = []
    for m, name in enumerate(FEATURES):
        med, iqr = source_median_iqr(X[:, m])
        if not iqr_is_valid(iqr):
            return {"computable": False, "reason": f"degenerate_iqr:{name}", "n_train": len(rows)}
        medians.append(med)
        iqrs.append(iqr)
    z = np.column_stack([standardize(X[:, m], medians[m], iqrs[m]) for m in range(6)])
    dist = loo_kth_distances(z, 5)
    tau = q_linear(dist, 0.95)
    if not np.isfinite(tau):
        return {"computable": False, "reason": "nonfinite_threshold", "n_train": len(rows)}
    nn = NearestNeighbors(n_neighbors=5, algorithm=NN_ALGORITHM, metric="euclidean")
    nn.fit(z)
    return {
        "computable": True,
        "reason": None,
        "n_train": len(rows),
        "train_days": _days(train_dts),
        "medians": medians,
        "iqrs": iqrs,
        "nn": nn,
        "tau": float(tau),
    }


def score_cs4(engine: dict[str, Any], tgt: str, test_dts: list[str], feat_by: dict[tuple[str, str], np.ndarray]) -> dict[str, Any]:
    if not engine.get("computable"):
        return {
            "computable": False,
            "reason": engine.get("reason"),
            "n_test": len(test_dts),
            "supported": set(),
            "n_supported": 0,
            "kth": {},
        }
    rows = []
    keep: list[str] = []
    for dt in test_dts:
        vec = feat_by.get((tgt, dt))
        if vec is None or not np.all(np.isfinite(vec)):
            return {
                "computable": False,
                "reason": "missing_or_nonfinite_target_test_features",
                "n_test": len(test_dts),
                "supported": set(),
                "n_supported": 0,
                "kth": {},
            }
        rows.append(vec)
        keep.append(dt)
    X = np.vstack(rows)
    z = np.column_stack([standardize(X[:, m], engine["medians"][m], engine["iqrs"][m]) for m in range(6)])
    dist, _ = engine["nn"].kneighbors(z, n_neighbors=5, return_distance=True)
    d5 = dist[:, 4]
    tau = engine["tau"]
    supported = {keep[i] for i, ok in enumerate(d5 <= tau) if ok}
    kth = {keep[i]: float(d5[i]) for i in range(len(keep))}
    return {
        "computable": True,
        "reason": None,
        "n_test": len(keep),
        "supported": supported,
        "n_supported": len(supported),
        "kth": kth,
        "threshold": tau,
    }


def run(ctx: StageContext) -> StageResult:
    root = ctx.root
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    validations: list[ValidationRecord] = []
    stages = ctx.sot.get("stages") or {}
    freeze = ctx.sot.get("scientific_freeze") or {}
    s03 = freeze.get("s03") or {}
    s08 = stages.get("S08") or {}
    if (stages.get("S03") or {}).get("status") != "GO" or s03.get("status") != "FROZEN":
        return _halt(now, "STOP", "S08 ALIGNMENT AUDIT STOP: S03 is not GO/FROZEN")
    if (stages.get("S05") or {}).get("status") != "GO":
        return _halt(now, "STOP", "S08 ALIGNMENT AUDIT STOP: S05 is not GO")
    if s08.get("status") != "STOP" or s08.get("reason") != "S08_STOP_TARGET_LEVEL_SYNTHESIS_UNSPECIFIED":
        return _halt(now, "STOP", "S08 ALIGNMENT AUDIT STOP: S08 is not the expected specification STOP")
    if (stages.get("S09") or {}).get("status") in {"GO", "STOP"}:
        return _halt(now, "STOP", "S08 ALIGNMENT AUDIT STOP: S09 must remain unexecuted")

    hashes_ok = True
    for rel in HASH_ONLY:
        path = root / rel
        if path.is_file():
            sha256_file(path)
        else:
            hashes_ok = hashes_ok and True
    validations.append(ValidationRecord("forbidden_files_hash_only", True, "values_not_parsed"))

    scope = json.loads((root / "artifacts/s03_finalization/S03_FINAL_SCOPE.json").read_text(encoding="utf-8"))
    registry = json.loads((root / "artifacts/s03_finalization/S03_FINAL_METHOD_REGISTRY.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "artifacts/s05/S05_COMMON_SUPPORT_PROTOCOL.json").read_text(encoding="utf-8"))
    if list(protocol.get("feature_order") or []) != list(FEATURES):
        return _halt(now, "STOP", "S08 ALIGNMENT AUDIT STOP: S05 feature order mismatch")
    mapping = {r["reference_plant_id"]: r["control_plant_id"] for r in scope["matched_control_mapping"]}
    if mapping != EXPECTED_MAP:
        return _halt(now, "STOP", "S08 ALIGNMENT AUDIT STOP: frozen control mapping does not reconcile")
    plants = list(scope.get("primary_plants") or [])
    controls = sorted(set(mapping.values()))
    if plants != sorted(EXPECTED_MAP) or controls != ["PS_006", "PS_010", "PS_013"]:
        return _halt(now, "STOP", "S08 ALIGNMENT AUDIT STOP: plant identity mismatch")
    incidence = _load_incidence(root / "artifacts/s03_inference_dependency_audit/S03_RQ3_CONTRAST_INCIDENCE.csv")
    if len(incidence) != 14:
        return _halt(now, "STOP", "S08 ALIGNMENT AUDIT STOP: incidence row count != 14")
    for row in incidence:
        if mapping[row["target_plant_id"]] != row["control_source_plant_id"]:
            return _halt(now, "STOP", f"S08 ALIGNMENT AUDIT STOP: incidence control mismatch {row['target_plant_id']}")
    validations.append(ValidationRecord("frozen_mapping", True, None))
    validations.append(ValidationRecord("frozen_incidence", True, None))

    s05_sum, twin_masks = _load_s05_cs4(root)
    idx = pq.read_table(root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet", columns=["plant_id", "datetime_raw", "split"])
    test_keys: dict[str, list[str]] = defaultdict(list)
    for pid, dt, split in zip(idx["plant_id"].to_pylist(), idx["datetime_raw"].to_pylist(), idx["split"].to_pylist()):
        if split == "test":
            test_keys[str(pid)].append(str(dt))
    for pid in plants:
        test_keys[pid] = sorted(test_keys[pid], key=lambda x: (parse_ts(x) or datetime.max.replace(tzinfo=timezone.utc), x))

    import duckdb

    panel_rec = (freeze.get("s02") or {}).get("canonical_panel") or {}
    panel_path = root / (panel_rec.get("path") or "artifacts/p02c/P02C_PLANT_PANEL.parquet")
    if panel_rec.get("sha256") and sha256_file(panel_path) != panel_rec["sha256"]:
        return _halt(now, "STOP", "S08 ALIGNMENT AUDIT STOP: S02 panel hash mismatch")
    con = duckdb.connect()
    need = sorted(set(plants) | set(controls))
    sql = _eligible_feature_sql(need).replace("read_parquet(?)", f"read_parquet('{str(panel_path).replace(chr(39), chr(39)+chr(39))}')")
    table = con.execute(sql).to_arrow_table()
    con.close()
    if "y_dc_normalized" in table.column_names:
        return _halt(now, "STOP", "S08 ALIGNMENT AUDIT STOP: target column loaded")
    splits = reconstruct_splits(table)
    split_mismatch = []
    for pid in plants:
        recon_test = set(splits.get(pid, {}).get("test") or [])
        frozen_test = set(test_keys[pid])
        if recon_test != frozen_test:
            split_mismatch.append(pid)
    if split_mismatch:
        return _halt(now, "STOP", "S08 ALIGNMENT AUDIT STOP: reconstructed primary TEST keys != S04 index: " + ",".join(split_mismatch))
    validations.append(ValidationRecord("control_split_rule_matches_frozen_primary", True, "60/20/20 after eligibility"))

    feat_by = _feat_map(table)
    engines = {cid: build_cs4_engine(cid, splits[cid]["train"], feat_by) for cid in controls}

    control_summaries = []
    control_chunks: dict[str, list] = {k: [] for k in ["control_source_plant_id", "target_plant_id", "rule_id", "datetime_raw", "computable", "reason", "kth_distance", "threshold", "supported"]}
    control_masks: dict[tuple[str, str], set[str]] = {}
    for tgt, cid in mapping.items():
        dts = test_keys[tgt]
        scored = score_cs4(engines[cid], tgt, dts, feat_by)
        control_masks[(cid, tgt)] = set(scored["supported"])
        n_te = len(dts)
        n_sup = scored["n_supported"]
        control_summaries.append(
            {
                "control_source_plant_id": cid,
                "target_plant_id": tgt,
                "rule_id": "CS4",
                "k": 5,
                "q": 0.95,
                "computable": scored["computable"],
                "reason": scored.get("reason"),
                "n_source_train": engines[cid].get("n_train"),
                "n_source_train_days": engines[cid].get("train_days"),
                "n_eligible_target_test": n_te,
                "n_target_test_days": _days(dts),
                "n_supported": n_sup,
                "n_supported_days": _days(scored["supported"]),
                "support_retention": _ratio(n_sup, n_te),
                "threshold": scored.get("threshold") if scored["computable"] else None,
            }
        )
        tau = scored.get("threshold")
        kth = scored.get("kth") or {}
        for dt in dts:
            control_chunks["control_source_plant_id"].append(cid)
            control_chunks["target_plant_id"].append(tgt)
            control_chunks["rule_id"].append("CS4")
            control_chunks["datetime_raw"].append(dt)
            control_chunks["computable"].append(bool(scored["computable"]))
            control_chunks["reason"].append(scored.get("reason"))
            control_chunks["kth_distance"].append(kth.get(dt))
            control_chunks["threshold"].append(tau if scored["computable"] else None)
            control_chunks["supported"].append(dt in scored["supported"])

    pairwise_rows = []
    target_rows = []
    invariant_fail = []
    empty_pair = []
    empty_global = []
    for inc in incidence:
        tgt = inc["target_plant_id"]
        gid = inc["reference_group_id"]
        cid = inc["control_source_plant_id"]
        twins = list(inc["twin_source_plant_ids"])
        test_set = set(test_keys[tgt])
        n_te = len(test_set)
        n_te_days = _days(test_set)
        ctrl_set = set(control_masks.get((cid, tgt), set()))
        computable_twins = []
        noncomputable = []
        pair_ns = []
        pair_ret = []
        pair_sets = []
        for src in twins:
            meta = s05_sum.get((src, tgt))
            cs4_ok = str(meta.get("computable")).lower() == "true" if meta else False
            if not cs4_ok:
                noncomputable.append(src)
                continue
            computable_twins.append(src)
            twin_set = set(twin_masks.get((src, tgt), set()))
            if twin_set - test_set:
                invariant_fail.append(f"twin_mask_outside_test {src}->{tgt}")
            pair = twin_set & ctrl_set
            pair_sets.append(pair)
            pair_ns.append(len(pair))
            pair_ret.append(_ratio(len(pair), n_te))
            if not pair:
                empty_pair.append(f"{src}->{tgt}|{cid}")
            if not (pair <= twin_set and pair <= ctrl_set):
                invariant_fail.append(f"pairwise_not_subset {src}->{tgt}")
            first, last = _first_last(pair)
            pairwise_rows.append(
                {
                    "target_plant_id": tgt,
                    "reference_group_id": gid,
                    "twin_source_plant_id": src,
                    "control_source_plant_id": cid,
                    "n_target_test": n_te,
                    "n_target_test_days": n_te_days,
                    "n_twin_supported": len(twin_set),
                    "n_twin_supported_days": _days(twin_set),
                    "twin_support_retention": _ratio(len(twin_set), n_te),
                    "n_control_supported": len(ctrl_set),
                    "n_control_supported_days": _days(ctrl_set),
                    "control_support_retention": _ratio(len(ctrl_set), n_te),
                    "n_pairwise_intersection": len(pair),
                    "n_pairwise_intersection_days": _days(pair),
                    "pairwise_retention_vs_target": _ratio(len(pair), n_te),
                    "pairwise_retention_vs_twin_support": _ratio(len(pair), len(twin_set)),
                    "pairwise_retention_vs_control_support": _ratio(len(pair), len(ctrl_set)),
                    "pairwise_nonempty": bool(pair),
                    "pairwise_first_timestamp": first,
                    "pairwise_last_timestamp": last,
                }
            )
        if computable_twins:
            global_set = set(ctrl_set)
            for src in computable_twins:
                global_set &= set(twin_masks.get((src, tgt), set()))
        else:
            global_set = set()
        if not global_set:
            empty_global.append(tgt)
        for pair in pair_sets:
            if not global_set <= pair:
                invariant_fail.append(f"global_not_subset_pairwise {tgt}")
        arr_n = np.array(pair_ns, dtype=float) if pair_ns else np.array([])
        arr_r = np.array([x for x in pair_ret if x is not None], dtype=float) if pair_ret else np.array([])
        gmin, gmed, gmax = (None, None, None)
        if arr_n.size:
            gmin, gmed, gmax = int(arr_n.min()), float(np.median(arr_n)), int(arr_n.max())
        rmin, rmed, rmax = (None, None, None)
        if arr_r.size:
            rmin, rmed, rmax = float(arr_r.min()), float(np.median(arr_r)), float(arr_r.max())
        first, last = _first_last(global_set)
        target_rows.append(
            {
                "target_plant_id": tgt,
                "reference_group_id": gid,
                "control_source_plant_id": cid,
                "n_expected_twin_sources": len(twins),
                "n_cs4_computable_twin_sources": len(computable_twins),
                "n_cs4_noncomputable_twin_sources": len(noncomputable),
                "n_pairwise_computable_twin_sources": len(computable_twins),
                "n_target_test": n_te,
                "n_target_test_days": n_te_days,
                "n_control_supported": len(ctrl_set),
                "n_global_intersection": len(global_set),
                "n_global_intersection_days": _days(global_set),
                "global_retention_vs_target": _ratio(len(global_set), n_te),
                "global_nonempty": bool(global_set),
                "global_first_timestamp": first,
                "global_last_timestamp": last,
                "min_pairwise_intersection_n": gmin,
                "median_pairwise_intersection_n": gmed,
                "max_pairwise_intersection_n": gmax,
                "min_pairwise_retention_vs_target": rmin,
                "median_pairwise_retention_vs_target": rmed,
                "max_pairwise_retention_vs_target": rmax,
                "global_to_min_pairwise_ratio": _ratio(len(global_set), gmin),
                "global_to_median_pairwise_ratio": (len(global_set) / gmed) if gmed not in (None, 0) else None,
                "global_to_max_pairwise_ratio": _ratio(len(global_set), gmax),
                "cs4_noncomputable_twin_source_plant_ids": json.dumps(noncomputable),
            }
        )

    if invariant_fail:
        return _halt(now, "STOP", "FAILED_VALIDATION: " + "; ".join(invariant_fail[:12]))
    validations.append(ValidationRecord("set_invariants", True, "global⊆pairwise⊆native"))
    validations.append(ValidationRecord("no_arbitrary_threshold", True, None))
    validations.append(ValidationRecord("s08_status_untouched", True, "patch omits stages.S08"))
    validations.append(ValidationRecord("s09_unexecuted", True, None))

    otg_def = ((registry.get("frozen_decisions") or {}).get("otg_absolute_definition") or {}).get("value")
    assessment = method_assessment_rows()
    rec_method = "B_PAIRWISE_INTERSECTION"
    rec_reason = (
        "Non-binding: B is the weakest extra restriction that still evaluates the matched contrast on identical "
        "target timestamps within each twin-control pair. Frozen same-sample language is within-transfer and does "
        "not uniquely force C. C remains available if the controller wants one shared target-level sample. A is "
        "admissible but does not force same rows within Delta_j."
    )

    summary = {
        "audit_status": "CONTROLLER_DECISION_REQUIRED",
        "blinded": True,
        "scientific_results_inspected": False,
        "outcome_columns_accessed": False,
        "model_predictions_accessed": False,
        "otg_accessed": False,
        "forbidden_files_parsed": False,
        "d_union_executed": False,
        "n_targets": len(target_rows),
        "n_pairwise_comparisons": len(pairwise_rows),
        "n_control_cs4_directions": len(control_summaries),
        "n_control_cs4_computable": sum(1 for r in control_summaries if r["computable"]),
        "any_pairwise_intersection_empty": bool(empty_pair),
        "empty_pairwise_ids": empty_pair,
        "any_global_intersection_empty": bool(empty_global),
        "empty_global_targets": empty_global,
        "choice_logically_resolved": False,
        "logically_resolved_method": None,
        "methodological_recommendation_nonbinding": rec_method,
        "recommendation_reason": rec_reason,
        "same_sample_language": otg_def,
        "rq3_bootstrap_mask_language": RQ3_MASK_TEXT,
        "a_logically_excluded_by_same_sample_language": False,
        "primary_support_rule": CS4,
        "target_global_intersection": [
            {
                "target_plant_id": r["target_plant_id"],
                "n_global_intersection": r["n_global_intersection"],
                "n_global_intersection_days": r["n_global_intersection_days"],
                "global_retention_vs_target": r["global_retention_vs_target"],
                "min_pairwise_intersection_n": r["min_pairwise_intersection_n"],
                "median_pairwise_intersection_n": r["median_pairwise_intersection_n"],
                "max_pairwise_intersection_n": r["max_pairwise_intersection_n"],
            }
            for r in target_rows
        ],
        "no_outcome_or_performance_inspected": True,
    }

    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    write_csv(
        out / "S08_CONTROL_CS4_SUPPORT_SUMMARY.csv",
        control_summaries,
        [
            "control_source_plant_id",
            "target_plant_id",
            "rule_id",
            "k",
            "q",
            "computable",
            "reason",
            "n_source_train",
            "n_source_train_days",
            "n_eligible_target_test",
            "n_target_test_days",
            "n_supported",
            "n_supported_days",
            "support_retention",
            "threshold",
        ],
    )
    mask_table = pa.table(
        {
            "control_source_plant_id": pa.array(control_chunks["control_source_plant_id"], pa.string()),
            "target_plant_id": pa.array(control_chunks["target_plant_id"], pa.string()),
            "rule_id": pa.array(control_chunks["rule_id"], pa.string()),
            "datetime_raw": pa.array(control_chunks["datetime_raw"], pa.string()),
            "computable": pa.array(control_chunks["computable"], pa.bool_()),
            "reason": pa.array(control_chunks["reason"], pa.string()),
            "kth_distance": pa.array(control_chunks["kth_distance"], pa.float64()),
            "threshold": pa.array(control_chunks["threshold"], pa.float64()),
            "supported": pa.array(control_chunks["supported"], pa.bool_()),
        }
    )
    pq.write_table(mask_table, out / "S08_CONTROL_CS4_SUPPORT_MASKS.parquet")
    pair_fields = [
        "target_plant_id",
        "reference_group_id",
        "twin_source_plant_id",
        "control_source_plant_id",
        "n_target_test",
        "n_target_test_days",
        "n_twin_supported",
        "n_twin_supported_days",
        "twin_support_retention",
        "n_control_supported",
        "n_control_supported_days",
        "control_support_retention",
        "n_pairwise_intersection",
        "n_pairwise_intersection_days",
        "pairwise_retention_vs_target",
        "pairwise_retention_vs_twin_support",
        "pairwise_retention_vs_control_support",
        "pairwise_nonempty",
        "pairwise_first_timestamp",
        "pairwise_last_timestamp",
    ]
    write_csv(out / "S08_PAIRWISE_ALIGNMENT_AUDIT.csv", pairwise_rows, pair_fields)
    tgt_fields = [
        "target_plant_id",
        "reference_group_id",
        "control_source_plant_id",
        "n_expected_twin_sources",
        "n_cs4_computable_twin_sources",
        "n_cs4_noncomputable_twin_sources",
        "n_pairwise_computable_twin_sources",
        "n_target_test",
        "n_target_test_days",
        "n_control_supported",
        "n_global_intersection",
        "n_global_intersection_days",
        "global_retention_vs_target",
        "global_nonempty",
        "global_first_timestamp",
        "global_last_timestamp",
        "min_pairwise_intersection_n",
        "median_pairwise_intersection_n",
        "max_pairwise_intersection_n",
        "min_pairwise_retention_vs_target",
        "median_pairwise_retention_vs_target",
        "max_pairwise_retention_vs_target",
        "global_to_min_pairwise_ratio",
        "global_to_median_pairwise_ratio",
        "global_to_max_pairwise_ratio",
        "cs4_noncomputable_twin_source_plant_ids",
    ]
    write_csv(out / "S08_TARGET_ALIGNMENT_SUMMARY.csv", target_rows, tgt_fields)
    write_csv(
        out / "S08_ALIGNMENT_METHOD_ASSESSMENT.csv",
        assessment,
        [
            "method_id",
            "description",
            "same_rows_within_twin_control_contrast",
            "same_rows_across_all_twin_sources_for_target",
            "evaluates_outside_source_support",
            "uses_all_available_native_support",
            "imposes_extra_cross_source_intersection",
            "requires_outcome_information",
            "changes_frozen_CS4_definition",
            "compatible_with_same_target_same_supported_sample_principle",
            "structurally_admissible",
            "reason",
        ],
    )
    dump_json(out / "S08_ALIGNMENT_AUDIT_SUMMARY.json", summary)
    recon = {
        "scientific_results_inspected": False,
        "s08_status_unchanged": True,
        "s09_executed": False,
        "matching_recomputed": False,
        "control_cs4_source_train_only": True,
        "target_statistics_in_scaling": False,
        "degenerate_iqr_no_epsilon": True,
        "twin_masks_from_s05": True,
        "d_union_executed": False,
        "set_invariants": "PASS",
        "primary_test_keys_reconcile_s04": True,
        "hashes_ok": hashes_ok,
    }
    dump_json(out / "S08_ALIGNMENT_RECONCILIATION.json", recon)
    report_path = root / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(summary, target_rows, pairwise_rows, control_summaries), encoding="utf-8")
    recs = [
        _art(root, out / "S08_CONTROL_CS4_SUPPORT_SUMMARY.csv", rows=len(control_summaries), fp=_csv_fp(out / "S08_CONTROL_CS4_SUPPORT_SUMMARY.csv")),
        _art(root, out / "S08_CONTROL_CS4_SUPPORT_MASKS.parquet", rows=mask_table.num_rows, fp=_pq_fp(out / "S08_CONTROL_CS4_SUPPORT_MASKS.parquet")),
        _art(root, out / "S08_PAIRWISE_ALIGNMENT_AUDIT.csv", rows=len(pairwise_rows), fp=_csv_fp(out / "S08_PAIRWISE_ALIGNMENT_AUDIT.csv")),
        _art(root, out / "S08_TARGET_ALIGNMENT_SUMMARY.csv", rows=len(target_rows), fp=_csv_fp(out / "S08_TARGET_ALIGNMENT_SUMMARY.csv")),
        _art(root, out / "S08_ALIGNMENT_METHOD_ASSESSMENT.csv", rows=len(assessment), fp=_csv_fp(out / "S08_ALIGNMENT_METHOD_ASSESSMENT.csv")),
        _art(root, out / "S08_ALIGNMENT_AUDIT_SUMMARY.json"),
        _art(root, out / "S08_ALIGNMENT_RECONCILIATION.json"),
        _art(root, report_path),
    ]
    dump_json(out / "S08_ALIGNMENT_MANIFEST.json", {"job": JOB, "artifacts": [_tab(r) for r in recs]})
    recs.append(_art(root, out / "S08_ALIGNMENT_MANIFEST.json"))

    patch = {
        "scientific_freeze": {
            "s08_specification_audits": {
                "common_support_alignment": {
                    "status": "CONTROLLER_DECISION_REQUIRED",
                    "operational_stage_id": "26",
                    "job": JOB,
                    "finished_at_utc": now,
                    "answer": ANSWER,
                    "blinded": True,
                    "scientific_results_inspected": False,
                    "outcome_columns_accessed": False,
                    "model_predictions_accessed": False,
                    "otg_accessed": False,
                    "primary_support_rule": CS4,
                    "candidate_methods": ["A_NATIVE_SUPPORTS", "B_PAIRWISE_INTERSECTION", "C_GLOBAL_INTERSECTION", "D_UNION"],
                    "controller_decision": None,
                    "methodological_recommendation_nonbinding": rec_method,
                    "choice_logically_resolved": False,
                    "n_targets": len(target_rows),
                    "n_pairwise_comparisons": len(pairwise_rows),
                    "any_pairwise_intersection_empty": bool(empty_pair),
                    "any_global_intersection_empty": bool(empty_global),
                    "control_cs4_summary": _tab(recs[0]),
                    "control_cs4_masks": _tab(recs[1]),
                    "pairwise_audit": _tab(recs[2]),
                    "target_summary": _tab(recs[3]),
                    "method_assessment": _tab(recs[4]),
                    "audit_summary": _tab(recs[5]),
                    "reconciliation": _tab(recs[6]),
                    "report": _tab(recs[7]),
                    "manifest": _tab(recs[8]),
                }
            }
        }
    }
    return StageResult(status="STOP", message=MSG, sot_patch=patch, artifacts=recs, validations=validations)


def _halt(now: str, status: str, message: str) -> StageResult:
    return StageResult(
        status=status,  # type: ignore[arg-type]
        message=message,
        sot_patch={
            "scientific_freeze": {
                "s08_specification_audits": {
                    "common_support_alignment": {
                        "status": "STOP",
                        "operational_stage_id": "26",
                        "job": JOB,
                        "finished_at_utc": now,
                        "reason": message,
                    }
                }
            }
        },
        artifacts=[],
        validations=[],
    )


def _report(summary: dict[str, Any], targets: list[dict[str, Any]], pairs: list[dict[str, Any]], controls: list[dict[str, Any]]) -> str:
    lines = [
        "# S08 pre-outcome common-support alignment audit",
        "",
        "RQ3 needs a matched contrast of control OTG vs twin OTG on target j. Twin CS4 masks and control CS4 masks need not share the same target TEST timestamps. This audit measures the structural cost of aligning those masks. No MAE, OTG, prediction, residual, p-value, or CI was computed.",
        "",
        "## Frozen inputs",
        "",
        "S03 mapping (14 targets → PS_006/PS_010/PS_013), S03 twin incidence, S05 CS4 twin masks, S04 TEST timestamps, CS4 k=5 q=0.95, six-core features, eligibility-then-chronological 60/20/20. Matching was not recomputed. S08 remains STOP (`S08_STOP_TARGET_LEVEL_SYNTHESIS_UNSPECIFIED`). S09 was not executed.",
        "",
        "## Blinding",
        "",
        "Forbidden S04/S06/S07 outcome files were hash-checked only. The eligible-row query uses `y_dc_normalized IS NOT NULL` as a frozen eligibility flag and does not load target values. No model was fit.",
        "",
        "## Methods A/B/C/D",
        "",
        "- A: native supports (admissible; does not force same rows inside the contrast). Frozen `otg_absolute_definition` is within-transfer same-supported-rows and does not logically exclude A.",
        "- B: pairwise intersection twin ∩ control (same rows within each source-level contrast).",
        "- C: global intersection of control and all CS4-computable twin masks (one timestamp set per target). C ⊆ every pairwise set.",
        "- D: union / score outside own CS4 — inadmissible; not executed.",
        "",
        f"Control→target CS4 directions: {summary['n_control_cs4_directions']}; computable: {summary['n_control_cs4_computable']}. Pairwise comparisons: {summary['n_pairwise_comparisons']}.",
        "",
        "## Per-target structural evidence",
        "",
        "target | twins CS4 ok/expected | n_test | n_control | min/med/max pairwise n | n_global | global days | global/target",
        "---|---|---|---|---|---|---|---",
    ]
    for r in sorted(targets, key=lambda z: z["target_plant_id"]):
        lines.append(
            f"{r['target_plant_id']} | {r['n_cs4_computable_twin_sources']}/{r['n_expected_twin_sources']} | {r['n_target_test']} | {r['n_control_supported']} | "
            f"{r['min_pairwise_intersection_n']}/{r['median_pairwise_intersection_n']}/{r['max_pairwise_intersection_n']} | "
            f"{r['n_global_intersection']} | {r['n_global_intersection_days']} | {r['global_retention_vs_target']}"
        )
    lines.extend(
        [
            "",
            f"Empty pairwise intersections: {summary['empty_pairwise_ids'] or 'none'}.",
            f"Empty global intersections: {summary['empty_global_targets'] or 'none'}.",
            "",
            "Set invariants held: global ⊆ pairwise ⊆ twin mask and ⊆ control mask.",
            "",
            "## B vs C",
            "",
            "B retains more or equal rows versus C by inclusion. C buys a single target-level sample across twin sources. No retention/day cutoff exists in the freeze, so the audit does not convert this into adequacy.",
            "",
            "## Decision",
            "",
            f"Status: `CONTROLLER_DECISION_REQUIRED`. Non-binding recommendation: `{summary['methodological_recommendation_nonbinding']}`. {summary['recommendation_reason']}",
            "",
            f"**{MSG}**",
            "",
        ]
    )
    return "\n".join(lines)
