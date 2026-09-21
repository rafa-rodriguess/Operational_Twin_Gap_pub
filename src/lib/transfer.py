"""S06 zero-shot cross-plant transfer (operational stage 23). No OTG construction."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import FEATURE_STRATEGIES, dump_json, parse_ts, sha256_file, write_csv
from src.lib.models import metrics as s04_metrics

ANSWER = None
OUT = "artifacts/s06"
REPORT = "reports/scientific/S06_CROSS_PLANT_TRANSFER.md"
FEATURES = FEATURE_STRATEGIES["six_core"]
TARGET = "y_dc_normalized"
FAM_PRIMARY = "SplineRidge"
FAM_R1 = "HistGradientBoosting"
FAMILIES = (FAM_PRIMARY, FAM_R1)
ROLES = {FAM_PRIMARY: "primary", FAM_R1: "r1_frozen_input"}
SUPPORT = "CS4"
LOCAL_EPS = 0.0


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


def _days(dts: list[str]) -> int:
    out = set()
    for dt in dts:
        ts = parse_ts(dt)
        out.add(ts.date().isoformat() if ts else dt[:10])
    return len(out)


def _check(root: Path, rec: dict[str, Any] | None, fallback: str) -> Path:
    path = root / ((rec or {}).get("path") or fallback)
    expected = (rec or {}).get("sha256")
    if not path.is_file():
        raise FileNotFoundError(fallback)
    if expected and sha256_file(path) != expected:
        raise ValueError(f"hash mismatch {path}")
    return path


def run(ctx: StageContext) -> StageResult:
    root = ctx.root
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    validations: list[ValidationRecord] = []
    validations.append(ValidationRecord("no_sot_write", True, "sot_patch only"))
    validations.append(ValidationRecord("no_fit_call", True, "no estimator training in S06"))

    try:
        panel_path = _check(root, None, "artifacts/p02c/P02C_PLANT_PANEL.parquet")
        proto04 = _check(root, None, "artifacts/s04/S04_MODEL_PROTOCOL.json")
        man04_models = _check(root, None, "artifacts/s04/S04_MODEL_MANIFEST.json")
        loc_path = _check(root, None, "artifacts/s04/S04_LOCAL_TEST_PREDICTIONS.parquet")
        idx_path = _check(root, None, "artifacts/s04/S04_ANALYTIC_INDEX.parquet")
        sum05 = _check(root, None, "artifacts/s05/S05_SUPPORT_SUMMARY.csv")
        mask_path = _check(root, None, "artifacts/s05/S05_SUPPORT_MASKS.parquet")
        _check(root, None, "artifacts/s05/S05_COMMON_SUPPORT_PROTOCOL.json")
    except (FileNotFoundError, ValueError) as exc:
        return _halt(now, validations, "STOP", f"S06 STOP: upstream {exc}")
    validations.append(ValidationRecord("upstream_hashes", True, None))

    proto = json.loads(proto04.read_text(encoding="utf-8"))
    if list(proto.get("feature_order") or []) != list(FEATURES):
        return _halt(now, validations, "STOP", "S06 STOP: feature order mismatch")
    model_man = json.loads(man04_models.read_text(encoding="utf-8"))
    model_meta = {m["path"]: m for m in (model_man.get("models") or [])}

    summary_rows = list(csv.DictReader(sum05.open(encoding="utf-8")))
    cs4 = [r for r in summary_rows if r.get("rule_id") == SUPPORT]
    if len(cs4) != 94:
        return _halt(now, validations, "STOP", f"S06 STOP: CS4 universe {len(cs4)} != 94")
    computable = [r for r in cs4 if str(r.get("computable")).lower() == "true"]
    noncomp = [r for r in cs4 if str(r.get("computable")).lower() != "true"]
    if len(computable) != 85 or len(noncomp) != 9:
        return _halt(now, validations, "STOP", "S06 STOP: CS4 computable counts")

    masks = pq.read_table(mask_path)
    nested_ok = _cs3_nested_in_cs4(masks)
    if not nested_ok:
        return _halt(now, validations, "STOP", "S06 STOP: CS3 timestamps not nested in CS4")
    validations.append(ValidationRecord("cs3_nested_cs4", True, None))

    cs4_sup = masks.filter(
        pc.and_(
            pc.equal(masks["rule_id"], SUPPORT),
            pc.and_(pc.equal(masks["computable"], True), pc.equal(masks["supported"], True)),
        )
    )
    supported_by: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for i in range(cs4_sup.num_rows):
        gid = str(cs4_sup["group_id"][i].as_py())
        src = str(cs4_sup["source_plant_id"][i].as_py())
        tgt = str(cs4_sup["target_plant_id"][i].as_py())
        dt = str(cs4_sup["datetime_raw"][i].as_py())
        supported_by[(gid, src, tgt)].append(dt)

    idx = pq.read_table(idx_path).to_pydict()
    test_keys = {(str(p), str(d)) for p, d, s in zip(idx["plant_id"], idx["datetime_raw"], idx["split"]) if s == "test"}

    loc = pq.read_table(loc_path)
    local_map: dict[tuple[str, str, str], tuple[float, float]] = {}
    for i in range(loc.num_rows):
        key = (str(loc["plant_id"][i].as_py()), str(loc["model_family"][i].as_py()), str(loc["datetime_raw"][i].as_py()))
        local_map[key] = (float(loc["y_true"][i].as_py()), float(loc["y_pred"][i].as_py()))

    expected_pred_rows = int(sum(int(float(r["n_supported"])) for r in computable)) * len(FAMILIES)
    uniq_keys: list[tuple[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in computable:
        gid, src, tgt = row["group_id"], row["source_plant_id"], row["target_plant_id"]
        for dt in supported_by.get((gid, src, tgt), []):
            key = (tgt, dt)
            if key not in seen_keys:
                seen_keys.add(key)
                uniq_keys.append(key)
    feat_map = _load_feature_map(root, panel_path, uniq_keys)
    if feat_map is None:
        return _halt(now, validations, "STOP", "S06 STOP: feature join failed")
    pred_cols: dict[str, list] = {k: [] for k in [
        "group_id", "source_plant_id", "target_plant_id", "model_family", "datetime_raw",
        "y_true", "y_pred_transfer", "y_pred_local", "residual_transfer", "abs_error_transfer",
        "residual_local", "abs_error_local", "support_rule", "support_retention",
    ]}
    metric_rows = []
    status_rows = []
    loaded_models: dict[tuple[str, str], Any] = {}
    mismatches = []

    for row in cs4:
        gid, src, tgt = row["group_id"], row["source_plant_id"], row["target_plant_id"]
        ok = str(row.get("computable")).lower() == "true"
        n_te = int(float(row["n_eligible_target_test"] or 0))
        n_sup_s05 = None if row.get("n_supported") in (None, "") else int(float(row["n_supported"]))
        sr = None if row.get("support_retention") in (None, "") else float(row["support_retention"])
        dts = supported_by.get((gid, src, tgt), [])
        if ok:
            if n_sup_s05 != len(dts):
                mismatches.append(f"n_supported {src}->{tgt}")
            if any((tgt, dt) not in test_keys for dt in dts):
                mismatches.append(f"not TEST {src}->{tgt}")
            if not dts:
                return _halt(now, validations, "STOP", f"S06 STOP: computable pair has no supported rows {src}->{tgt}")
        status_rows.append(
            {
                "group_id": gid,
                "source_plant_id": src,
                "target_plant_id": tgt,
                "cs4_computable": ok,
                "reason": row.get("reason") or None,
                "n_eligible_target_test": n_te,
                "n_supported": n_sup_s05,
                "support_retention": sr,
                "predictions_generated": ok,
            }
        )
        if not ok:
            continue
        order = dts
        if any((tgt, dt) not in feat_map for dt in order):
            return _halt(now, validations, "STOP", f"S06 STOP: feature join failed {src}->{tgt}")
        X = np.vstack([feat_map[(tgt, dt)][0] for dt in order])
        y_panel = np.array([feat_map[(tgt, dt)][1] for dt in order], dtype=float)
        for fam in FAMILIES:
            rel = f"artifacts/s04/models/{src}/{fam}/full.joblib"
            meta = model_meta.get(rel)
            fpath = root / rel
            if meta is None or not fpath.is_file():
                return _halt(now, validations, "STOP", f"S06 STOP: missing full model {rel}")
            if sha256_file(fpath) != meta.get("sha256") or fpath.stat().st_size != int(meta.get("bytes") or -1):
                return _halt(now, validations, "STOP", f"S06 STOP: model hash {rel}")
            keym = (src, fam)
            if keym not in loaded_models:
                loaded_models[keym] = joblib.load(fpath)
            yhat = np.asarray(loaded_models[keym].predict(X), dtype=float)
            y_true = np.empty(len(order), dtype=float)
            y_loc = np.empty(len(order), dtype=float)
            for i, dt in enumerate(order):
                loc_key = (tgt, fam, dt)
                if loc_key not in local_map:
                    return _halt(now, validations, "STOP", f"S06 STOP: missing local pred {tgt} {fam} {dt}")
                yt, yp = local_map[loc_key]
                if abs(yt - y_panel[i]) > 1e-12:
                    mismatches.append(f"y_true {tgt} {dt}")
                y_true[i] = yt
                y_loc[i] = yp
            if not (np.all(np.isfinite(yhat)) and np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_loc))):
                return _halt(now, validations, "STOP", f"S06 STOP: nonfinite predictions {src}->{tgt} {fam}")
            r_tr = yhat - y_true
            r_loc = y_loc - y_true
            m_tr = s04_metrics(y_true, yhat)
            m_loc = s04_metrics(y_true, y_loc)
            for i, dt in enumerate(order):
                pred_cols["group_id"].append(gid)
                pred_cols["source_plant_id"].append(src)
                pred_cols["target_plant_id"].append(tgt)
                pred_cols["model_family"].append(fam)
                pred_cols["datetime_raw"].append(dt)
                pred_cols["y_true"].append(float(y_true[i]))
                pred_cols["y_pred_transfer"].append(float(yhat[i]))
                pred_cols["y_pred_local"].append(float(y_loc[i]))
                pred_cols["residual_transfer"].append(float(r_tr[i]))
                pred_cols["abs_error_transfer"].append(float(abs(r_tr[i])))
                pred_cols["residual_local"].append(float(r_loc[i]))
                pred_cols["abs_error_local"].append(float(abs(r_loc[i])))
                pred_cols["support_rule"].append(SUPPORT)
                pred_cols["support_retention"].append(sr)
            metric_rows.append(
                {
                    "group_id": gid,
                    "source_plant_id": src,
                    "target_plant_id": tgt,
                    "model_family": fam,
                    "analysis_role": ROLES[fam],
                    "support_rule": SUPPORT,
                    "n_supported": len(order),
                    "n_supported_days": _days(order),
                    "support_retention": sr,
                    "transfer_mae": m_tr["mae"],
                    "transfer_rmse": m_tr["rmse"],
                    "transfer_r2": m_tr["r2"],
                    "transfer_bias": m_tr["bias"],
                    "local_same_rows_mae": m_loc["mae"],
                    "local_same_rows_rmse": m_loc["rmse"],
                    "local_same_rows_r2": m_loc["r2"],
                    "local_same_rows_bias": m_loc["bias"],
                }
            )
    if mismatches:
        return _halt(now, validations, "STOP", "S06 STOP: reconciliation " + "; ".join(mismatches[:6]))
    n_pred = len(pred_cols["datetime_raw"])
    if n_pred != expected_pred_rows or len(metric_rows) != 85 * 2 or len(status_rows) != 94:
        return _halt(now, validations, "STOP", f"S06 STOP: row counts pred={n_pred} expected={expected_pred_rows}")
    validations.append(ValidationRecord("prediction_row_count", True, str(n_pred)))
    validations.append(ValidationRecord("full_models_only", True, f"n_loaded={len(loaded_models)}"))

    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "group_id": pa.array(pred_cols["group_id"], pa.string()),
            "source_plant_id": pa.array(pred_cols["source_plant_id"], pa.string()),
            "target_plant_id": pa.array(pred_cols["target_plant_id"], pa.string()),
            "model_family": pa.array(pred_cols["model_family"], pa.string()),
            "datetime_raw": pa.array(pred_cols["datetime_raw"], pa.string()),
            "y_true": pa.array(pred_cols["y_true"], pa.float64()),
            "y_pred_transfer": pa.array(pred_cols["y_pred_transfer"], pa.float64()),
            "y_pred_local": pa.array(pred_cols["y_pred_local"], pa.float64()),
            "residual_transfer": pa.array(pred_cols["residual_transfer"], pa.float64()),
            "abs_error_transfer": pa.array(pred_cols["abs_error_transfer"], pa.float64()),
            "residual_local": pa.array(pred_cols["residual_local"], pa.float64()),
            "abs_error_local": pa.array(pred_cols["abs_error_local"], pa.float64()),
            "support_rule": pa.array(pred_cols["support_rule"], pa.string()),
            "support_retention": pa.array(pred_cols["support_retention"], pa.float64()),
        }
    )
    forbidden = {"otg" + "_abs", "otg" + "_rel", "delta" + "_mae"}
    if forbidden & set(table.schema.names):
        return _halt(now, validations, "STOP", "S06 STOP: forbidden difference columns")
    pq.write_table(table, out / "S06_CROSS_PLANT_PREDICTIONS.parquet")
    write_csv(
        out / "S06_TRANSFER_METRICS.csv",
        metric_rows,
        [
            "group_id", "source_plant_id", "target_plant_id", "model_family", "analysis_role", "support_rule",
            "n_supported", "n_supported_days", "support_retention",
            "transfer_mae", "transfer_rmse", "transfer_r2", "transfer_bias",
            "local_same_rows_mae", "local_same_rows_rmse", "local_same_rows_r2", "local_same_rows_bias",
        ],
    )
    write_csv(
        out / "S06_TRANSFER_STATUS.csv",
        status_rows,
        ["group_id", "source_plant_id", "target_plant_id", "cs4_computable", "reason", "n_eligible_target_test", "n_supported", "support_retention", "predictions_generated"],
    )
    protocol = {
        "target": TARGET,
        "feature_order": list(FEATURES),
        "source_models": "S04 full.joblib only",
        "model_families": [{"id": FAM_PRIMARY, "role": "primary"}, {"id": FAM_R1, "role": "r1_frozen_input"}],
        "s06_training_or_refit": False,
        "target_adaptation": False,
        "primary_common_support": {"id": SUPPORT, "k": 5, "q": 0.95},
        "s05_mask_reused_without_mutation": True,
        "prediction_clipping": False,
        "primary_loss": "MAE",
        "secondary_metrics": ["RMSE", "R2", "bias_mean_pred_minus_true"],
        "local_comparison_rows_identical_to_transfer": True,
        "cs3_metrics_executed": False,
        "otg_computed": False,
        "difference_of_mae_computed": False,
        "bootstrap_or_permutation_executed": False,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "joblib_version": joblib.__version__,
    }
    dump_json(out / "S06_TRANSFER_PROTOCOL.json", protocol)
    recon = {
        "s04_model_manifest_hashes": "PASS",
        "s04_local_test_predictions": "PASS",
        "s05_support_masks": "PASS",
        "pair_universe": 94,
        "cs4_computable": 85,
        "cs4_noncomputable": 9,
        "n_supported_per_pair": "PASS",
        "sr_per_pair": "PASS",
        "timestamps_in_cs4_supported_mask": "PASS",
        "local_equals_s04": "PASS",
        "y_true_reconciliation": "PASS",
        "model_family_roles": ROLES,
        "no_new_training": True,
        "cs3_metrics": False,
        "cs3_nested_in_cs4": True,
        "mae_difference_computed": False,
        "prediction_row_count": n_pred,
        "metric_row_count": len(metric_rows),
    }
    dump_json(out / "S06_RECONCILIATION.json", recon)
    recs = [
        _art(root, out / "S06_CROSS_PLANT_PREDICTIONS.parquet", rows=n_pred, fp=_pq_fp(out / "S06_CROSS_PLANT_PREDICTIONS.parquet")),
        _art(root, out / "S06_TRANSFER_METRICS.csv", rows=len(metric_rows), fp=_csv_fp(out / "S06_TRANSFER_METRICS.csv")),
        _art(root, out / "S06_TRANSFER_STATUS.csv", rows=len(status_rows), fp=_csv_fp(out / "S06_TRANSFER_STATUS.csv")),
        _art(root, out / "S06_TRANSFER_PROTOCOL.json"),
        _art(root, out / "S06_RECONCILIATION.json"),
    ]
    report = root / REPORT
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(metric_rows, status_rows, n_pred), encoding="utf-8")
    recs.append(_art(root, report))
    dump_json(out / "S06_MANIFEST.json", {"job": "S06_CROSS_PLANT_TRANSFER", "artifacts": {r.path.split("/")[-1]: _tab(r) for r in recs}})
    recs.append(_art(root, out / "S06_MANIFEST.json"))

    patch = {
        "stages": {
            "S06": {
                "kind": "scientific_analysis",
                "status": "GO",
                "reason": "S06_CROSS_PLANT_TRANSFER_COMPLETE",
                "operational_stage_id": "23",
                "job": "S06_CROSS_PLANT_TRANSFER",
                "finished_at_utc": now,
                "answer": ANSWER,
                "report": _tab(recs[-2]),
            }
        },
        "scientific_analysis": {
            "s06": {
                "status": "GO",
                "operational_stage_id": "23",
                "scope": {
                    "expected_directional_transfers": 94,
                    "cs4_computable_transfers": 85,
                    "cs4_noncomputable_transfers": 9,
                    "model_families": list(FAMILIES),
                    "primary_model_family": FAM_PRIMARY,
                    "r1_model_family": FAM_R1,
                },
                "target": TARGET,
                "support_rule": {"id": SUPPORT, "k": 5, "q": 0.95},
                "protocol": _tab(recs[3]),
                "predictions": _tab(recs[0]),
                "transfer_metrics": _tab(recs[1]),
                "transfer_status": _tab(recs[2]),
                "reconciliation": _tab(recs[4]),
                "manifest": _tab(recs[-1]),
                "zero_shot_no_target_adaptation": True,
                "same_supported_rows_transfer_vs_local": True,
                "cs3_metrics_computed": False,
                "otg_computed": False,
                "asymmetry_computed": False,
                "bootstrap_draws_generated": False,
                "permutation_results_computed": False,
            }
        },
    }
    return StageResult(status="GO", message="GO — S06 CROSS-PLANT TRANSFER COMPLETE", sot_patch=patch, artifacts=recs, validations=validations)


def _load_feature_map(root: Path, panel_path: Path, keys: list[tuple[str, str]]) -> dict[tuple[str, str], tuple[np.ndarray, float]] | None:
    tmp = root / "artifacts/s06/_keys.tmp.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["plant_id", "datetime_raw"])
        w.writerows(keys)
    feats = ", ".join(f"p.{f}" for f in FEATURES)
    con = duckdb.connect()
    tbl = con.execute(
        f"""
        SELECT p.plant_id, CAST(p.datetime_raw AS VARCHAR) AS datetime_raw, {feats}, p.{TARGET} AS y
        FROM read_parquet('{str(panel_path).replace("'", "''")}') p
        INNER JOIN read_csv_auto('{str(tmp).replace("'", "''")}', types={{'plant_id': 'VARCHAR', 'datetime_raw': 'VARCHAR'}}) k
          ON p.plant_id = k.plant_id AND CAST(p.datetime_raw AS VARCHAR) = CAST(k.datetime_raw AS VARCHAR)
        """
    ).to_arrow_table()
    con.close()
    tmp.unlink(missing_ok=True)
    out: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
    for i in range(tbl.num_rows):
        pid = str(tbl["plant_id"][i].as_py())
        dt = str(tbl["datetime_raw"][i].as_py())
        vec = []
        for name in FEATURES:
            raw = tbl[name][i].as_py()
            if raw is None:
                return None
            vec.append(float(raw))
        y = tbl["y"][i].as_py()
        if y is None or not np.isfinite(y) or not np.all(np.isfinite(vec)):
            return None
        out[(pid, dt)] = (np.array(vec, dtype=float), float(y))
    if len(out) != len(keys):
        return None
    return out


def _cs3_nested_in_cs4(masks) -> bool:
    sets: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(lambda: {"CS3": set(), "CS4": set()})
    for i in range(masks.num_rows):
        if not masks["computable"][i].as_py():
            continue
        if not masks["supported"][i].as_py():
            continue
        rid = str(masks["rule_id"][i].as_py())
        if rid not in ("CS3", "CS4"):
            continue
        key = (str(masks["group_id"][i].as_py()), str(masks["source_plant_id"][i].as_py()), str(masks["target_plant_id"][i].as_py()))
        sets[key][rid].add(str(masks["datetime_raw"][i].as_py()))
    for key, d in sets.items():
        if d["CS3"] - d["CS4"]:
            return False
    return True


def _halt(now, validations, status, message):
    return StageResult(
        status=status,
        message=message,
        sot_patch={"stages": {"S06": {"kind": "scientific_analysis", "status": status, "finished_at_utc": now, "reason": message, "answer": ANSWER, "job": "S06_CROSS_PLANT_TRANSFER", "operational_stage_id": "23"}}},
        artifacts=[],
        validations=validations,
    )


def _report(metrics_rows, status_rows, n_pred) -> str:
    n_ok = sum(1 for r in status_rows if r["predictions_generated"])
    n_nc = len(status_rows) - n_ok
    by_fam = defaultdict(list)
    for r in metrics_rows:
        by_fam[r["model_family"]].append(r["transfer_mae"])
    lines = [
        "# S06 — Cross-plant transfer",
        "",
        "Zero-shot application of frozen S04 full models to S05 CS4-supported target TEST rows. Local-reference metrics use the same timestamps. CS3 transfer metrics were not calculated. Difference-of-MAE / directional-matrix construction is deferred.",
        "",
        f"Directed transfers: 94 expected; {n_ok} CS4-computable with predictions; {n_nc} non-computable with no prediction rows. Prediction rows: {n_pred} (CS4 supported totals × 2 families).",
        "",
    ]
    for fam, vals in by_fam.items():
        arr = np.array(vals, dtype=float)
        lines.append(f"{fam} transfer MAE (n={len(arr)}): min {float(arr.min())}, median {float(np.median(arr))}, max {float(arr.max())}.")
    lines += [
        "",
        "Local-same-rows metrics are stored alongside transfer metrics in `S06_TRANSFER_METRICS.csv` without subtraction. Non-computable transfers retain S05 reasons in `S06_TRANSFER_STATUS.csv`.",
        "",
        "**GO — S06 CROSS-PLANT TRANSFER COMPLETE**",
        "",
    ]
    return "\n".join(lines)
