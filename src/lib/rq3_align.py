"""Stage 44 helper: CS4 + B_PAIRWISE_INTERSECTION for state_constrained_support_v2."""

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

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import dump_json, parse_ts, sha256_file, write_csv
from src.lib.rq3_match import load_metadata, plant_state, snapshot_tree
from src.lib.load_raw import _csv_artifact_schema_fingerprint
from src.lib.support_align_helpers import (
    FEATURES,
    _days,
    _eligible_feature_sql,
    _feat_map,
    _first_last,
    _load_s05_cs4,
    _ratio,
    build_cs4_engine,
    reconstruct_splits,
    score_cs4,
)

JOB = "S08_STATE_CONSTRAINED_SUPPORT_ALIGNMENT"
REVISION = "state_constrained_support_v2"
MSG_GO = "GO — S08 REVISED SAME-STATE SUPPORT ALIGNMENT FROZEN"
MSG_STOP = "STOP — S08 STATE-CONSTRAINED SUPPORT ALIGNMENT INCOMPLETE"
OUT = "artifacts/s08_state_constrained_support_alignment"
REPORT = "reports/freeze/S08_STATE_CONSTRAINED_SUPPORT_ALIGNMENT.md"
ANSWER = None
CS4 = {"id": "CS4", "k": 5, "q": 0.95}
ALIGNMENT = "B_PAIRWISE_INTERSECTION"
FORMULA = "M_pair_ijc = M_CS4_i_to_j ∩ M_CS4_c(j)_to_j"
EMPTY_POLICY = "SOURCE_LEVEL_CONTRAST_NONCOMPUTABLE_NO_RESCUE"
PROTECTED = (
    "artifacts/s03_finalization",
    "artifacts/s03_inference_dependency_audit",
    "artifacts/s03_rq3_state_revision",
    "artifacts/s04",
    "artifacts/s05",
    "artifacts/s06",
    "artifacts/s07",
    "artifacts/s08",
    "artifacts/s08_alignment_audit",
    "artifacts/s08_scientific",
    "artifacts/s09_scientific",
    "artifacts/s10",
    "paper",
)
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
    "artifacts/s08_scientific/S08_SOURCE_LEVEL_MATCHED_CONTRASTS.csv",
    "artifacts/s08_scientific/S08_TARGET_LEVEL_MATCHED_CONTRASTS.csv",
    "artifacts/s08_scientific/S08_TARGET_LEVEL_INFERENCE.csv",
    "artifacts/s08_scientific/S08_FLEET_DESCRIPTIVE_SUMMARY.json",
)
PAIR_FIELDS = [
    "reference_group_id",
    "twin_source_plant_id",
    "target_plant_id",
    "control_source_plant_id",
    "n_target_test_rows",
    "n_twin_cs4_rows",
    "n_control_cs4_rows",
    "n_pairwise_intersection",
    "n_pairwise_days",
    "pairwise_retention_vs_target",
    "pairwise_retention_vs_twin",
    "pairwise_retention_vs_control",
    "computable",
    "reason",
]
TGT_FIELDS = [
    "target_plant_id",
    "reference_group_id",
    "control_source_plant_id",
    "n_expected_exact_twin_sources",
    "n_s05_cs4_computable_twin_sources",
    "n_pairwise_computable_sources",
    "n_target_test_rows",
    "n_control_cs4_rows",
    "min_pairwise_n",
    "median_pairwise_n",
    "max_pairwise_n",
    "min_pairwise_days",
    "median_pairwise_days",
    "max_pairwise_days",
    "included_source_ids",
    "excluded_source_ids",
    "excluded_reasons",
    "structurally_computable",
]
BA_FIELDS = [
    "target_plant_id",
    "reference_group_id",
    "old_control_source_plant_id",
    "new_control_source_plant_id",
    "old_control_same_state",
    "new_control_same_state",
    "old_control_cs4_rows",
    "new_control_cs4_rows",
    "old_pairwise_computable_source_count",
    "new_pairwise_computable_source_count",
    "old_min_pairwise_rows",
    "new_min_pairwise_rows",
    "old_median_pairwise_rows",
    "new_median_pairwise_rows",
    "old_max_pairwise_rows",
    "new_max_pairwise_rows",
    "old_target_structurally_computable",
    "new_target_structurally_computable",
    "ledger",
]
CTRL_SUM_FIELDS = [
    "control_source_plant_id",
    "target_plant_id",
    "target_state",
    "control_state",
    "same_state",
    "rule_id",
    "k",
    "q",
    "computable",
    "reason",
    "n_source_train",
    "n_source_train_days",
    "n_target_test_rows",
    "n_target_test_days",
    "n_supported",
    "n_supported_days",
    "support_retention",
    "threshold",
]


class S44Stop(Exception):
    def __init__(self, reason: str, details: str = ""):
        super().__init__(details or reason)
        self.reason = reason
        self.details = details or reason


def repo_relative(root: Path, path: Path) -> str:
    """Serialize a repository-backed path; reject traversal outside root."""
    root_r = root.resolve()
    path_r = path.resolve()
    try:
        rel = path_r.relative_to(root_r)
    except ValueError as exc:
        raise S44Stop("S44_STOP_PATH_OUTSIDE_ROOT", str(path)) from exc
    posix = rel.as_posix()
    if posix.startswith("..") or not path_r.is_relative_to(root_r):
        raise S44Stop("S44_STOP_PATH_OUTSIDE_ROOT", posix)
    return posix


def _nested(sot: dict[str, Any], *keys: str) -> Any:
    cur: Any = sot
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _art(root: Path, path: Path, rows: int | None = None, fp: str | None = None) -> ArtifactRecord:
    payload = path.read_bytes()
    return ArtifactRecord(path.relative_to(root).as_posix(), hashlib.sha256(payload).hexdigest(), len(payload), rows, fp)


def _tab(rec: ArtifactRecord) -> dict[str, Any]:
    out: dict[str, Any] = {"path": rec.path, "sha256": rec.sha256, "bytes": rec.bytes}
    if rec.row_count is not None:
        out["row_count"] = rec.row_count
    if rec.schema_fingerprint is not None:
        out["schema_fingerprint"] = rec.schema_fingerprint
    return out


def _pq_fp(path: Path) -> str:
    return hashlib.sha256("|".join(pq.read_schema(path).names).encode("utf-8")).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pairwise_intersection(twin: set[str], control: set[str]) -> set[str]:
    return set(twin) & set(control)


def target_is_structurally_computable(n_pairwise_computable: int) -> bool:
    return int(n_pairwise_computable) >= 1


def j_comp_structural(supported: list[str], computable_targets: list[str]) -> list[str]:
    allowed = set(supported)
    out = [pid for pid in computable_targets if pid in allowed]
    return sorted(set(out))


def _flag(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _stat(values: list[int]) -> tuple[int | None, float | None, int | None]:
    if not values:
        return None, None, None
    arr = np.array(values, dtype=float)
    return int(arr.min()), float(np.median(arr)), int(arr.max())


def validate_upstream(sot: dict[str, Any]) -> None:
    fin = _nested(sot, "stages", "S03_RQ3_STATE_REVISION_FINALIZATION") or {}
    if fin.get("status") != "GO":
        raise S44Stop("S44_STOP_STAGE43_NOT_GO", str(fin.get("status")))
    if _nested(sot, "scientific_freeze", "s03", "active_rq3_control_revision") != REVISION:
        raise S44Stop("S44_STOP_ACTIVE_REVISION")
    node = _nested(sot, "scientific_freeze", "s03", "rq3_revisions", REVISION) or {}
    if node.get("status") != "FROZEN":
        raise S44Stop("S44_STOP_REVISION_NOT_FROZEN", str(node.get("status")))
    if node.get("scientific_activation") is not True:
        raise S44Stop("S44_STOP_REVISION_NOT_ACTIVE")
    align = _nested(sot, "scientific_freeze", "s08_rq3_operationalization", "support_alignment") or {}
    if align.get("status") != "FROZEN":
        raise S44Stop("S44_STOP_S08_ALIGNMENT_NOT_FROZEN")
    if align.get("method") != ALIGNMENT:
        raise S44Stop("S44_STOP_ALIGNMENT_METHOD", str(align.get("method")))


def _hash_only(root: Path) -> None:
    for rel in HASH_ONLY:
        path = root / rel
        if path.is_file():
            sha256_file(path)


def _load_incidence(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(path):
        twins = json.loads(row["twin_source_plant_ids"])
        rows.append(
            {
                "target_plant_id": row["target_plant_id"],
                "reference_group_id": row["reference_group_id"],
                "control_source_plant_id": row["control_source_plant_id"],
                "twin_source_plant_ids": list(twins),
            }
        )
    return sorted(rows, key=lambda r: r["target_plant_id"])


def execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    try:
        return _execute(ctx, now, validations)
    except S44Stop as exc:
        validations.append(ValidationRecord(exc.reason, False, exc.details))
        return StageResult(
            status="STOP",
            message=MSG_STOP,
            sot_patch={
                "stages": {
                    "S08_STATE_CONSTRAINED_SUPPORT_ALIGNMENT": {
                        "kind": "scientific_support_realignment",
                        "status": "STOP",
                        "reason": exc.reason,
                        "operational_stage_id": "44",
                        "job": JOB,
                        "details": exc.details,
                        "finished_at_utc": now,
                        "answer": ANSWER,
                        "revision_id": REVISION,
                        "parent_rq3_revision": REVISION,
                    }
                }
            },
            artifacts=[],
            validations=validations,
        )


def _execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    root = ctx.root
    hist_before = snapshot_tree(root, PROTECTED)
    validate_upstream(ctx.sot)
    validations.append(ValidationRecord("upstream_revision", True, REVISION))
    _hash_only(root)
    validations.append(ValidationRecord("forbidden_files_hash_only", True, "values_not_parsed"))

    rev = _nested(ctx.sot, "scientific_freeze", "s03", "rq3_revisions", REVISION) or {}
    supported = list(rev.get("supported_targets") or [])
    unsupported = list(rev.get("unsupported_targets") or [])
    unsupported_reasons = dict(rev.get("unsupported_reasons") or {})
    mapping_rows = list(rev.get("selected_mapping") or [])
    mapping = {r["reference_plant_id"]: r["control_plant_id"] for r in mapping_rows}
    if sorted(mapping) != sorted(supported):
        raise S44Stop("S44_STOP_MAPPING_SUPPORT_MISMATCH")
    if len(mapping) != len(supported) or len(set(mapping)) != len(mapping):
        raise S44Stop("S44_STOP_DUPLICATE_TARGETS")
    if set(mapping) & set(unsupported):
        raise S44Stop("S44_STOP_UNSUPPORTED_IN_MAPPING")
    sel_rec = (rev.get("artifacts") or {}).get("selected_mapping") or {}
    rec_rel = str(sel_rec.get("path") or "artifacts/s03_rq3_state_revision/S03_STATE_SUPPORTED_MAPPING_SELECTED.csv")
    if Path(rec_rel).is_absolute():
        rec_rel = repo_relative(root, Path(rec_rel))
    sel_path = (root / rec_rel).resolve()
    sel_rel = repo_relative(root, sel_path)
    if not sel_path.is_file():
        raise S44Stop("S44_STOP_MAPPING_MISSING", sel_rel)
    if sel_rec.get("sha256") and sel_rec["sha256"] != sha256_file(sel_path):
        raise S44Stop("S44_STOP_MAPPING_HASH")
    file_map = {r["reference_plant_id"]: r["control_plant_id"] for r in _read_csv(sel_path)}
    if file_map != mapping:
        raise S44Stop("S44_STOP_MAPPING_RECONCILE")
    for tgt, cid in mapping.items():
        if not cid or cid == tgt:
            raise S44Stop("S44_STOP_BAD_CONTROL", tgt)

    protocol = json.loads((root / "artifacts/s05/S05_COMMON_SUPPORT_PROTOCOL.json").read_text(encoding="utf-8"))
    cs4 = protocol.get("cs4") or protocol.get("CS4") or {}
    if cs4.get("id") != "CS4" or int(cs4.get("k")) != 5 or float(cs4.get("q")) != 0.95:
        raise S44Stop("S44_STOP_CS4_PROTOCOL", str(cs4))
    if list(protocol.get("feature_order") or []) != list(FEATURES):
        raise S44Stop("S44_STOP_FEATURE_ORDER")
    if protocol.get("target_partition") != "test":
        raise S44Stop("S44_STOP_TARGET_PARTITION", str(protocol.get("target_partition")))
    validations.append(ValidationRecord("frozen_cs4", True, "k=5 q=0.95 six_core TEST"))

    meta_rec = _nested(ctx.sot, "scientific_freeze", "s01", "metadata") or {}
    meta_path = root / str(meta_rec.get("path") or "data/raw/br_pvgen/BR-PVGen_metadata.csv")
    if meta_rec.get("sha256") and sha256_file(meta_path) != meta_rec["sha256"]:
        raise S44Stop("S44_STOP_METADATA_HASH")
    meta = load_metadata(meta_path)
    for tgt, cid in mapping.items():
        tstate = plant_state(meta, tgt)
        cstate = plant_state(meta, cid)
        if not tstate or tstate != cstate:
            raise S44Stop("S44_STOP_CROSS_STATE", f"{tgt}:{cid}")

    inc_path = root / "artifacts/s03_rq3_state_revision/S03_STATE_SUPPORTED_CONTRAST_INCIDENCE.csv"
    incidence = [r for r in _load_incidence(inc_path) if r["target_plant_id"] in mapping]
    if sorted(r["target_plant_id"] for r in incidence) != sorted(mapping):
        raise S44Stop("S44_STOP_INCIDENCE_TARGETS")
    for row in incidence:
        if row["control_source_plant_id"] != mapping[row["target_plant_id"]]:
            raise S44Stop("S44_STOP_INCIDENCE_CONTROL", row["target_plant_id"])
        if row["target_plant_id"] in unsupported:
            raise S44Stop("S44_STOP_UNSUPPORTED_INCIDENCE", row["target_plant_id"])

    old_map = {
        r["reference_plant_id"]: r["control_plant_id"]
        for r in (_nested(ctx.sot, "scientific_freeze", "s03", "corrective_reaudit", "control_matching", "selected_mapping") or [])
    }
    hist_pair = _read_csv(root / "artifacts/s08_alignment_audit/S08_PAIRWISE_ALIGNMENT_AUDIT.csv")
    hist_tgt = _read_csv(root / "artifacts/s08_alignment_audit/S08_TARGET_ALIGNMENT_SUMMARY.csv")
    hist_ctrl = _read_csv(root / "artifacts/s08_alignment_audit/S08_CONTROL_CS4_SUPPORT_SUMMARY.csv")
    hist_pair_n: dict[str, list[int]] = defaultdict(list)
    hist_pair_comp: dict[str, int] = defaultdict(int)
    for row in hist_pair:
        tgt = row["target_plant_id"]
        n = int(float(row["n_pairwise_intersection"]))
        hist_pair_n[tgt].append(n)
        if _flag(row.get("pairwise_nonempty")):
            hist_pair_comp[tgt] += 1
    hist_ctrl_n = {(r["control_source_plant_id"], r["target_plant_id"]): int(float(r["n_supported"])) for r in hist_ctrl}
    hist_tgt_map = {r["target_plant_id"]: r for r in hist_tgt}

    s05_sum, twin_masks = _load_s05_cs4(root)
    idx = pq.read_table(root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet", columns=["plant_id", "datetime_raw", "split"])
    test_keys: dict[str, list[str]] = defaultdict(list)
    for pid, dt, split in zip(idx["plant_id"].to_pylist(), idx["datetime_raw"].to_pylist(), idx["split"].to_pylist()):
        if split == "test":
            test_keys[str(pid)].append(str(dt))
    for pid in mapping:
        test_keys[pid] = sorted(test_keys[pid], key=lambda x: (parse_ts(x) or datetime.max.replace(tzinfo=timezone.utc), x))

    import duckdb

    panel_rec = _nested(ctx.sot, "scientific_freeze", "s02", "canonical_panel") or {}
    panel_path = root / (panel_rec.get("path") or "artifacts/p02c/P02C_PLANT_PANEL.parquet")
    if panel_rec.get("sha256") and sha256_file(panel_path) != panel_rec["sha256"]:
        raise S44Stop("S44_STOP_PANEL_HASH")
    need = sorted(set(mapping) | set(mapping.values()))
    sql = _eligible_feature_sql(need).replace("read_parquet(?)", f"read_parquet('{str(panel_path).replace(chr(39), chr(39)+chr(39))}')")
    con = duckdb.connect()
    table = con.execute(sql).to_arrow_table()
    con.close()
    if "y_dc_normalized" in table.column_names:
        raise S44Stop("S44_STOP_OUTCOME_COLUMN_LOADED")
    splits = reconstruct_splits(table)
    mismatch = [pid for pid in mapping if set(splits.get(pid, {}).get("test") or []) != set(test_keys[pid])]
    if mismatch:
        raise S44Stop("S44_STOP_TEST_SPLIT_RECONCILE", ",".join(mismatch))
    validations.append(ValidationRecord("target_test_split", True, "S04 index"))
    feat_by = _feat_map(table)
    engines: dict[str, dict[str, Any]] = {}
    for cid in sorted(set(mapping.values())):
        engines[cid] = build_cs4_engine(cid, splits[cid]["train"], feat_by)

    scaler_rows = []
    thresh_rows = []
    for cid in sorted(engines):
        eng = engines[cid]
        for feat, med, iqr in zip(FEATURES, eng.get("medians") or [None] * 6, eng.get("iqrs") or [None] * 6):
            scaler_rows.append(
                {
                    "control_source_plant_id": cid,
                    "feature": feat,
                    "median": med if eng.get("computable") else None,
                    "iqr": iqr if eng.get("computable") else None,
                    "n_train": eng.get("n_train"),
                    "n_train_days": eng.get("train_days"),
                    "computable": eng.get("computable"),
                    "reason": eng.get("reason"),
                }
            )
        thresh_rows.append(
            {
                "control_source_plant_id": cid,
                "rule_id": "CS4",
                "k": 5,
                "q": 0.95,
                "threshold": eng.get("tau") if eng.get("computable") else None,
                "n_train": eng.get("n_train"),
                "n_train_days": eng.get("train_days"),
                "computable": eng.get("computable"),
                "reason": eng.get("reason"),
            }
        )

    control_summaries = []
    control_chunks: dict[str, list] = {k: [] for k in ["control_source_plant_id", "target_plant_id", "rule_id", "datetime_raw", "computable", "reason", "kth_distance", "threshold", "supported"]}
    control_masks: dict[tuple[str, str], set[str]] = {}
    for tgt in sorted(mapping):
        cid = mapping[tgt]
        dts = test_keys[tgt]
        scored = score_cs4(engines[cid], tgt, dts, feat_by)
        control_masks[(cid, tgt)] = set(scored["supported"])
        n_te = len(dts)
        control_summaries.append(
            {
                "control_source_plant_id": cid,
                "target_plant_id": tgt,
                "target_state": plant_state(meta, tgt),
                "control_state": plant_state(meta, cid),
                "same_state": True,
                "rule_id": "CS4",
                "k": 5,
                "q": 0.95,
                "computable": scored["computable"],
                "reason": scored.get("reason"),
                "n_source_train": engines[cid].get("n_train"),
                "n_source_train_days": engines[cid].get("train_days"),
                "n_target_test_rows": n_te,
                "n_target_test_days": _days(dts),
                "n_supported": scored["n_supported"],
                "n_supported_days": _days(scored["supported"]),
                "support_retention": _ratio(scored["n_supported"], n_te),
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
    pairwise_mask_rows: list[dict[str, Any]] = []
    target_rows = []
    source_noncomp = []
    for inc in incidence:
        tgt = inc["target_plant_id"]
        gid = inc["reference_group_id"]
        cid = inc["control_source_plant_id"]
        twins = list(inc["twin_source_plant_ids"])
        test_set = set(test_keys[tgt])
        n_te = len(test_set)
        ctrl_set = set(control_masks.get((cid, tgt), set()))
        if ctrl_set - test_set:
            raise S44Stop("S44_STOP_CONTROL_MASK_OUTSIDE_TEST", tgt)
        included = []
        excluded = []
        excl_reason = []
        pair_ns = []
        pair_days = []
        for src in twins:
            meta_s05 = s05_sum.get((src, tgt))
            cs4_ok = str(meta_s05.get("computable")).lower() == "true" if meta_s05 else False
            twin_set = set(twin_masks.get((src, tgt), set())) if cs4_ok else set()
            if twin_set - test_set:
                raise S44Stop("S44_STOP_TWIN_MASK_OUTSIDE_TEST", f"{src}->{tgt}")
            if not cs4_ok:
                excluded.append(src)
                excl_reason.append("TWIN_CS4_NONCOMPUTABLE")
                pairwise_rows.append(
                    {
                        "reference_group_id": gid,
                        "twin_source_plant_id": src,
                        "target_plant_id": tgt,
                        "control_source_plant_id": cid,
                        "n_target_test_rows": n_te,
                        "n_twin_cs4_rows": 0,
                        "n_control_cs4_rows": len(ctrl_set),
                        "n_pairwise_intersection": 0,
                        "n_pairwise_days": 0,
                        "pairwise_retention_vs_target": _ratio(0, n_te),
                        "pairwise_retention_vs_twin": None,
                        "pairwise_retention_vs_control": _ratio(0, len(ctrl_set)),
                        "computable": False,
                        "reason": "TWIN_CS4_NONCOMPUTABLE",
                    }
                )
                source_noncomp.append({"target": tgt, "twin": src, "reason": "TWIN_CS4_NONCOMPUTABLE"})
                continue
            pair = pairwise_intersection(twin_set, ctrl_set)
            if not (pair <= twin_set and pair <= ctrl_set):
                raise S44Stop("S44_STOP_PAIRWISE_NOT_SUBSET", f"{src}->{tgt}")
            ok = bool(pair)
            reason = None if ok else "EMPTY_PAIRWISE_INTERSECTION"
            if ok:
                included.append(src)
            else:
                excluded.append(src)
                excl_reason.append(reason)
                source_noncomp.append({"target": tgt, "twin": src, "reason": reason})
            pair_ns.append(len(pair))
            pair_days.append(_days(pair))
            pairwise_rows.append(
                {
                    "reference_group_id": gid,
                    "twin_source_plant_id": src,
                    "target_plant_id": tgt,
                    "control_source_plant_id": cid,
                    "n_target_test_rows": n_te,
                    "n_twin_cs4_rows": len(twin_set),
                    "n_control_cs4_rows": len(ctrl_set),
                    "n_pairwise_intersection": len(pair),
                    "n_pairwise_days": _days(pair),
                    "pairwise_retention_vs_target": _ratio(len(pair), n_te),
                    "pairwise_retention_vs_twin": _ratio(len(pair), len(twin_set)),
                    "pairwise_retention_vs_control": _ratio(len(pair), len(ctrl_set)),
                    "computable": ok,
                    "reason": reason or "",
                }
            )
            for dt in sorted(pair, key=lambda x: (parse_ts(x) or datetime.max.replace(tzinfo=timezone.utc), x)):
                pairwise_mask_rows.append(
                    {
                        "target_plant_id": tgt,
                        "twin_source_plant_id": src,
                        "control_source_plant_id": cid,
                        "datetime_raw": dt,
                        "pairwise_supported": True,
                        "support_rule": "CS4",
                        "alignment": ALIGNMENT,
                    }
                )
        n_comp = len(included)
        gmin, gmed, gmax = _stat(pair_ns)
        dmin, dmed, dmax = _stat(pair_days)
        target_rows.append(
            {
                "target_plant_id": tgt,
                "reference_group_id": gid,
                "control_source_plant_id": cid,
                "n_expected_exact_twin_sources": len(twins),
                "n_s05_cs4_computable_twin_sources": len(twins) - excl_reason.count("TWIN_CS4_NONCOMPUTABLE"),
                "n_pairwise_computable_sources": n_comp,
                "n_target_test_rows": n_te,
                "n_control_cs4_rows": len(ctrl_set),
                "min_pairwise_n": gmin,
                "median_pairwise_n": gmed,
                "max_pairwise_n": gmax,
                "min_pairwise_days": dmin,
                "median_pairwise_days": dmed,
                "max_pairwise_days": dmax,
                "included_source_ids": json.dumps(included),
                "excluded_source_ids": json.dumps(excluded),
                "excluded_reasons": json.dumps(excl_reason),
                "structurally_computable": target_is_structurally_computable(n_comp),
            }
        )

    pairwise_rows.sort(key=lambda r: (r["target_plant_id"], r["twin_source_plant_id"]))
    target_rows.sort(key=lambda r: r["target_plant_id"])
    pairwise_mask_rows.sort(key=lambda r: (r["target_plant_id"], r["twin_source_plant_id"], r["datetime_raw"]))
    j_comp = j_comp_structural(supported, [r["target_plant_id"] for r in target_rows if r["structurally_computable"]])
    if set(j_comp) - set(supported):
        raise S44Stop("S44_STOP_JCOMP_NOT_SUBSET")
    if set(j_comp) & set(unsupported):
        raise S44Stop("S44_STOP_UNSUPPORTED_IN_JCOMP")
    if not j_comp:
        raise S44Stop("S44_STOP_NO_STRUCTURAL_TARGET")

    ba_rows = []
    frozen_ids = sorted(set(supported) | set(unsupported))
    for tgt in frozen_ids:
        old_c = old_map.get(tgt, "")
        new_c = mapping.get(tgt, "")
        tstate = plant_state(meta, tgt)
        if tgt in unsupported:
            ba_rows.append(
                {
                    "target_plant_id": tgt,
                    "reference_group_id": next((r["reference_group_id"] for r in incidence), ""),
                    "old_control_source_plant_id": old_c,
                    "new_control_source_plant_id": "",
                    "old_control_same_state": bool(old_c and plant_state(meta, old_c) == tstate),
                    "new_control_same_state": False,
                    "old_control_cs4_rows": hist_ctrl_n.get((old_c, tgt)),
                    "new_control_cs4_rows": None,
                    "old_pairwise_computable_source_count": hist_pair_comp.get(tgt),
                    "new_pairwise_computable_source_count": None,
                    "old_min_pairwise_rows": _stat(hist_pair_n.get(tgt, []))[0],
                    "new_min_pairwise_rows": None,
                    "old_median_pairwise_rows": _stat(hist_pair_n.get(tgt, []))[1],
                    "new_median_pairwise_rows": None,
                    "old_max_pairwise_rows": _stat(hist_pair_n.get(tgt, []))[2],
                    "new_max_pairwise_rows": None,
                    "old_target_structurally_computable": int(float((hist_tgt_map.get(tgt) or {}).get("n_pairwise_computable_twin_sources") or 0)) >= 1,
                    "new_target_structurally_computable": False,
                    "ledger": "OUTSIDE_SAME_STATE_COMPARATOR_SUPPORT",
                }
            )
            continue
        nrow = next(r for r in target_rows if r["target_plant_id"] == tgt)
        ba_rows.append(
            {
                "target_plant_id": tgt,
                "reference_group_id": nrow["reference_group_id"],
                "old_control_source_plant_id": old_c,
                "new_control_source_plant_id": new_c,
                "old_control_same_state": bool(old_c and plant_state(meta, old_c) == tstate),
                "new_control_same_state": True,
                "old_control_cs4_rows": hist_ctrl_n.get((old_c, tgt)),
                "new_control_cs4_rows": nrow["n_control_cs4_rows"],
                "old_pairwise_computable_source_count": hist_pair_comp.get(tgt),
                "new_pairwise_computable_source_count": nrow["n_pairwise_computable_sources"],
                "old_min_pairwise_rows": _stat(hist_pair_n.get(tgt, []))[0],
                "new_min_pairwise_rows": nrow["min_pairwise_n"],
                "old_median_pairwise_rows": _stat(hist_pair_n.get(tgt, []))[1],
                "new_median_pairwise_rows": nrow["median_pairwise_n"],
                "old_max_pairwise_rows": _stat(hist_pair_n.get(tgt, []))[2],
                "new_max_pairwise_rows": nrow["max_pairwise_n"],
                "old_target_structurally_computable": int(float((hist_tgt_map.get(tgt) or {}).get("n_pairwise_computable_twin_sources") or 0)) >= 1,
                "new_target_structurally_computable": nrow["structurally_computable"],
                "ledger": "STAGE43_SUPPORTED",
            }
        )
    # fill group for unsupported from historical
    hist_inc = {r["target_plant_id"]: r["reference_group_id"] for r in hist_tgt}
    for row in ba_rows:
        if not row["reference_group_id"]:
            row["reference_group_id"] = hist_inc.get(row["target_plant_id"], "")
    ba_rows.sort(key=lambda r: r["target_plant_id"])

    noncomputable_supported = [r["target_plant_id"] for r in target_rows if not r["structurally_computable"]]
    flow = {
        "J_frozen": sorted(frozen_ids),
        "J_frozen_count": len(frozen_ids),
        "J_state_support": sorted(supported),
        "J_state_support_count": len(supported),
        "J_state_unsupported": [
            {"reference_plant_id": pid, "reason": unsupported_reasons.get(pid), "ledger": "OUTSIDE_SAME_STATE_COMPARATOR_SUPPORT"}
            for pid in sorted(unsupported)
        ],
        "excluded_frozen_to_support": [
            {"reference_plant_id": pid, "reason": unsupported_reasons.get(pid)} for pid in sorted(unsupported)
        ],
        "J_comp_structural": j_comp,
        "J_comp_structural_count": len(j_comp),
        "structurally_noncomputable_supported_targets": noncomputable_supported,
        "excluded_support_to_comp": [
            {"reference_plant_id": r["target_plant_id"], "reason": r["excluded_reasons"]}
            for r in target_rows
            if not r["structurally_computable"]
        ],
        "source_level_expected": sum(r["n_expected_exact_twin_sources"] for r in target_rows),
        "source_level_pairwise_computable": sum(r["n_pairwise_computable_sources"] for r in target_rows),
        "source_level_noncomputable": len(source_noncomp),
        "structurally_noncomputable_source_contrasts": source_noncomp,
    }

    protocol_out = {
        "revision_id": REVISION,
        "primary_support": CS4,
        "feature_strategy": "six_core",
        "feature_order": list(FEATURES),
        "target_partition": "test",
        "source_partition": "train",
        "support_alignment_method": ALIGNMENT,
        "support_formula": FORMULA,
        "empty_intersection_policy": EMPTY_POLICY,
        "global_intersection_required": False,
        "union_allowed": False,
        "cross_state_rescue": False,
        "new_retention_threshold": False,
        "new_day_threshold": False,
        "new_row_threshold": False,
        "minimum_twin_sources": None,
        "outcomes_accessed": False,
        "models_fit": False,
        "target_missingness_predicate_accessed": True,
        "s05_protocol_hash": sha256_file(root / "artifacts/s05/S05_COMMON_SUPPORT_PROTOCOL.json"),
    }

    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    scaler_path = out / "S08_STATE_CONTROL_SOURCE_SCALERS.csv"
    thresh_path = out / "S08_STATE_CONTROL_SOURCE_THRESHOLDS.csv"
    ctrl_sum_path = out / "S08_STATE_CONTROL_CS4_SUPPORT_SUMMARY.csv"
    pair_path = out / "S08_STATE_PAIRWISE_ALIGNMENT_AUDIT.csv"
    tgt_path = out / "S08_STATE_TARGET_ALIGNMENT_SUMMARY.csv"
    ba_path = out / "S08_STATE_SUPPORT_BEFORE_AFTER.csv"
    write_csv(scaler_path, scaler_rows, ["control_source_plant_id", "feature", "median", "iqr", "n_train", "n_train_days", "computable", "reason"])
    write_csv(thresh_path, thresh_rows, ["control_source_plant_id", "rule_id", "k", "q", "threshold", "n_train", "n_train_days", "computable", "reason"])
    write_csv(ctrl_sum_path, control_summaries, CTRL_SUM_FIELDS)
    write_csv(pair_path, pairwise_rows, PAIR_FIELDS)
    write_csv(tgt_path, target_rows, TGT_FIELDS)
    write_csv(ba_path, ba_rows, BA_FIELDS)

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
    ctrl_mask_path = out / "S08_STATE_CONTROL_CS4_SUPPORT_MASKS.parquet"
    pq.write_table(mask_table, ctrl_mask_path)
    pair_mask_table = pa.table(
        {
            "target_plant_id": pa.array([r["target_plant_id"] for r in pairwise_mask_rows], pa.string()),
            "twin_source_plant_id": pa.array([r["twin_source_plant_id"] for r in pairwise_mask_rows], pa.string()),
            "control_source_plant_id": pa.array([r["control_source_plant_id"] for r in pairwise_mask_rows], pa.string()),
            "datetime_raw": pa.array([r["datetime_raw"] for r in pairwise_mask_rows], pa.string()),
            "pairwise_supported": pa.array([r["pairwise_supported"] for r in pairwise_mask_rows], pa.bool_()),
            "support_rule": pa.array([r["support_rule"] for r in pairwise_mask_rows], pa.string()),
            "alignment": pa.array([r["alignment"] for r in pairwise_mask_rows], pa.string()),
        }
    )
    pair_mask_path = out / "S08_STATE_PAIRWISE_SUPPORT_MASKS.parquet"
    pq.write_table(pair_mask_table, pair_mask_path)

    recs = [
        _art(root, scaler_path, rows=len(scaler_rows), fp=_csv_artifact_schema_fingerprint(scaler_path)),
        _art(root, thresh_path, rows=len(thresh_rows), fp=_csv_artifact_schema_fingerprint(thresh_path)),
        _art(root, ctrl_mask_path, rows=len(control_chunks["datetime_raw"]), fp=_pq_fp(ctrl_mask_path)),
        _art(root, ctrl_sum_path, rows=len(control_summaries), fp=_csv_artifact_schema_fingerprint(ctrl_sum_path)),
        _art(root, pair_mask_path, rows=len(pairwise_mask_rows), fp=_pq_fp(pair_mask_path)),
        _art(root, pair_path, rows=len(pairwise_rows), fp=_csv_artifact_schema_fingerprint(pair_path)),
        _art(root, tgt_path, rows=len(target_rows), fp=_csv_artifact_schema_fingerprint(tgt_path)),
        _art(root, ba_path, rows=len(ba_rows), fp=_csv_artifact_schema_fingerprint(ba_path)),
    ]
    named = {
        "control_scalers": recs[0],
        "control_thresholds": recs[1],
        "control_cs4_masks": recs[2],
        "control_cs4_summary": recs[3],
        "pairwise_masks": recs[4],
        "pairwise_audit": recs[5],
        "target_alignment_summary": recs[6],
        "before_after": recs[7],
    }

    hist_after = snapshot_tree(root, PROTECTED)
    if hist_after != hist_before:
        raise S44Stop("S44_STOP_HISTORICAL_MUTATION", str(sorted(set(hist_after) ^ set(hist_before))[:12]))

    flow_path = out / "S08_STATE_SUPPORT_FLOW.json"
    dump_json(flow_path, flow)
    recs.append(_art(root, flow_path))
    named["support_flow"] = recs[-1]
    proto_path = out / "S08_STATE_SUPPORT_PROTOCOL.json"
    dump_json(proto_path, protocol_out)
    recs.append(_art(root, proto_path))
    named["protocol"] = recs[-1]
    recon = {
        "stage43_go": True,
        "active_revision": REVISION,
        "mapping_hash_ok": True,
        "same_state_all_supported": True,
        "unsupported_excluded": True,
        "cs4_k": 5,
        "cs4_q": 0.95,
        "alignment": ALIGNMENT,
        "union_used": False,
        "global_intersection_required": False,
        "j_comp_subset_supported": True,
        "outcomes_accessed": False,
        "models_fit": False,
        "historical_artifacts_unchanged": True,
        "historical_snapshot_sha256": hashlib.sha256(json.dumps(hist_before, sort_keys=True).encode()).hexdigest(),
        "scientific_results_computed": False,
        "s08_active_revision_written": False,
        "n_j_frozen": len(frozen_ids),
        "n_j_state_support": len(supported),
        "n_j_comp_structural": len(j_comp),
        "target_missingness_predicate_accessed": True,
    }
    recon_path = out / "S08_STATE_SUPPORT_RECONCILIATION.json"
    dump_json(recon_path, recon)
    recs.append(_art(root, recon_path))
    named["reconciliation"] = recs[-1]
    manifest = {
        "job": JOB,
        "revision_id": REVISION,
        "run_id": ctx.run_id,
        "finished_at_utc": now,
        "artifacts": {k: _tab(v) for k, v in named.items()},
    }
    man_path = out / "S08_STATE_SUPPORT_MANIFEST.json"
    dump_json(man_path, manifest)
    recs.append(_art(root, man_path))
    named["manifest"] = recs[-1]

    report_path = root / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ctrl_lines = [
        f"| {r['target_plant_id']} | {r['control_source_plant_id']} | {r['target_state']} | {r['n_target_test_rows']} | {r['n_supported']} | {r['support_retention']} | {r['n_supported_days']} |"
        for r in control_summaries
    ]
    tgt_lines = [
        f"| {r['target_plant_id']} | {r['n_expected_exact_twin_sources']} | {r['n_pairwise_computable_sources']} | {r['structurally_computable']} | {r['min_pairwise_n']} | {r['median_pairwise_n']} | {r['max_pairwise_n']} |"
        for r in target_rows
    ]
    report_path.write_text(
        "# S08 state-constrained support alignment\n\n"
        "## A. Inputs\n\n"
        f"- Stage 43 active revision: `{REVISION}`\n"
        f"- support protocol: CS4 k=5 q=0.95 six_core target TEST\n"
        f"- pairwise alignment: `{ALIGNMENT}` `{FORMULA}`\n"
        f"- Stage 43 mapping: `{sel_rel}` sha256 `{sha256_file(sel_path)}`\n"
        f"- S05 protocol sha256 `{protocol_out['s05_protocol_hash']}`\n\n"
        "## B. Structural flow\n\n"
        "```text\nJ_frozen -> J_state_support -> J_comp_structural\n```\n\n"
        f"- J_frozen count = {len(frozen_ids)}\n"
        f"- excluded to J_state_support: {json.dumps(flow['excluded_frozen_to_support'])}\n"
        f"- J_state_support count = {len(supported)}\n"
        f"- J_comp_structural count = {len(j_comp)}\n"
        f"- structurally noncomputable supported targets: {json.dumps(noncomputable_supported)}\n"
        "- Stage 43 unsupported targets are outside revised RQ3 comparator support, not Stage 44 support failures.\n\n"
        "## C. Revised control support\n\n"
        "| target | control | state | n_test | n_cs4 | retention | days |\n"
        "|---|---|---|---|---|---|---|\n"
        + "\n".join(ctrl_lines)
        + "\n\nEvery revised mapping row is same-state.\n\n"
        "## D. Pairwise support\n\n"
        "| target | expected twins | pairwise-computable | structurally computable | min n | median n | max n |\n"
        "|---|---|---|---|---|---|---|\n"
        + "\n".join(tgt_lines)
        + "\n\n"
        f"Source-level expected={flow['source_level_expected']}; pairwise-computable={flow['source_level_pairwise_computable']}; "
        f"noncomputable={flow['source_level_noncomputable']}. Empty intersection ⇒ `EMPTY_PAIRWISE_INTERSECTION`, no rescue. "
        "Global target-level intersection is not required.\n\n"
        "## E. Historical vs revised structural comparison\n\n"
        "Stage 26 vs Stage 44 geometry only is in `S08_STATE_SUPPORT_BEFORE_AFTER.csv`. "
        "No OTG, MAE, Delta, or other outcome columns are present. Historical technical-only mapping remains sensitivity/provenance.\n\n"
        "## F. Scientific meaning\n\n"
        "Stage 43 ensured the revised non-twin comparator is drawn from the same state as the target. Stage 44 now ensures the nominal twin and the revised non-twin control will be evaluated on the same target timestamps within each source-level RQ3 contrast. This removes the systematic cross-state comparator imbalance at the coarse state level and preserves environmental common-support comparability within each contrast, without claiming complete control of geography or climate.\n\n"
        "- same-state does not equal same climate;\n"
        "- support alignment does not prove causal isolation of nominal equivalence;\n"
        "- it removes two specific comparator disadvantages: (1) systematic cross-state control mismatch; (2) unequal scoring timestamps inside a matched contrast.\n\n"
        "## G. Non-actions\n\n"
        "No model fit, predictions, MAE, OTG, Delta, bootstrap, CI/p-value, or Stage 45. "
        "Historical S04–S07, S08 alignment/scientific, S09, S10, and paper trees are unchanged.\n",
        encoding="utf-8",
    )
    recs.append(_art(root, report_path))
    named["report"] = recs[-1]

    arts_meta = {k: _tab(v) for k, v in named.items()}
    patch = {
        "stages": {
            "S08_STATE_CONSTRAINED_SUPPORT_ALIGNMENT": {
                "kind": "scientific_support_realignment",
                "status": "GO",
                "reason": "S44_REVISED_SUPPORT_ALIGNMENT_FROZEN",
                "operational_stage_id": "44",
                "job": JOB,
                "run_id": ctx.run_id,
                "finished_at_utc": now,
                "answer": ANSWER,
                "report": REPORT,
                "revision_id": REVISION,
                "parent_rq3_revision": REVISION,
                "artifacts": arts_meta,
            }
        },
        "scientific_analysis": {
            "s08_revisions": {
                REVISION: {
                    "support_alignment": {
                        "status": "GO",
                        "reason": "S44_REVISED_SUPPORT_ALIGNMENT_FROZEN",
                        "revision_id": REVISION,
                        "scientific_results_computed": False,
                        "outcomes_accessed": False,
                        "target_missingness_predicate_accessed": True,
                        "primary_support": CS4,
                        "support_alignment_method": ALIGNMENT,
                        "support_formula": FORMULA,
                        "empty_intersection_policy": EMPTY_POLICY,
                        "global_intersection_required": False,
                        "union_allowed": False,
                        "cross_state_rescue": False,
                        "j_frozen_reference": {"ids": sorted(frozen_ids), "count": len(frozen_ids)},
                        "j_state_support": {"ids": sorted(supported), "count": len(supported)},
                        "j_state_unsupported": flow["J_state_unsupported"],
                        "j_comp_structural": {"ids": j_comp, "count": len(j_comp)},
                        "structurally_noncomputable_supported_targets": noncomputable_supported,
                        "structurally_noncomputable_source_contrasts": source_noncomp,
                        "selected_mapping_reference": {"path": sel_rel, "sha256": sha256_file(sel_path)},
                        "control_support_summary": _tab(named["control_cs4_summary"]),
                        "pairwise_alignment_summary": _tab(named["pairwise_audit"]),
                        "target_alignment_summary": _tab(named["target_alignment_summary"]),
                        "support_flow": _tab(named["support_flow"]),
                        "before_after": _tab(named["before_after"]),
                        "protocol": _tab(named["protocol"]),
                        "reconciliation": _tab(named["reconciliation"]),
                        "manifest": _tab(named["manifest"]),
                        "report": REPORT,
                        "next_stage": "45",
                    }
                }
            }
        },
    }
    validations.append(ValidationRecord("no_sot_write", True, "sot_patch only"))
    validations.append(ValidationRecord("no_stage45", True, None))
    validations.append(ValidationRecord("historical_immutable", True, None))
    return StageResult(status="GO", message=MSG_GO, sot_patch=patch, artifacts=recs, validations=validations)
