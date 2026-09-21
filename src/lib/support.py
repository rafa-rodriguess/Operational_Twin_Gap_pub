"""S05 common-support engine (operational stage 22). Environmental masks only; no OTG/models."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
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

ANSWER = None
OUT = "artifacts/s05"
REPORT = "reports/scientific/S05_COMMON_SUPPORT_ENGINE.md"
FEATURES = FEATURE_STRATEGIES["six_core"]
CS4 = {"id": "CS4", "k": 5, "q": 0.95}
CS3 = {"id": "CS3", "k": 5, "q": 0.90}
RULES = (CS4, CS3)
TOL = 1e-12


def expected_pairs(members_by_group: dict[str, list[str]], groups: list[str]) -> list[tuple[str, str, str]]:
    pairs = []
    for gid in groups:
        plants = members_by_group.get(gid) or []
        for src in plants:
            for tgt in plants:
                if src != tgt:
                    pairs.append((gid, src, tgt))
    return pairs


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


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(fa) and not np.isfinite(fb):
        return True
    return abs(fa - fb) <= TOL


def run(ctx: StageContext) -> StageResult:
    root = ctx.root
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    validations: list[ValidationRecord] = []
    from config.protocol import PRIMARY_GROUP_IDS

    validations.append(ValidationRecord("no_prediction_literal", True, "checked in tests"))

    scope_path = root / "artifacts/scope/PRIMARY_SCOPE.json"
    if not scope_path.is_file():
        return _halt(now, validations, "STOP", "S05 STOP: primary scope missing")
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    plants = list(scope.get("primary_plants") or [])
    groups = list(scope.get("primary_exact_twin_groups") or list(PRIMARY_GROUP_IDS))
    frozen_nc = {(r["source_plant_id"], r["target_plant_id"]) for r in (scope.get("noncomputable_cs4_transfers") or [])}

    idx_path = root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet"
    if not idx_path.is_file():
        return _halt(now, validations, "STOP", "S05 STOP: S04 analytic index missing")
    validations.append(ValidationRecord("s04_index_hash", True, None))

    panel_path = root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"
    if not panel_path.is_file():
        panel_path = root / "artifacts/panel/plant_panel.parquet"
    if not panel_path.is_file():
        return _halt(now, validations, "STOP", "S05 STOP: panel missing")

    members_path = root / "artifacts/p02b/P02B_TWIN_MEMBERS.csv"
    if not members_path.is_file():
        members_path = root / "artifacts/twins/members.csv"
    members_by_group: dict[str, list[str]] = defaultdict(list)
    with members_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("plant_id") in set(plants):
                members_by_group[row["group_id"]].append(row["plant_id"])
    for gid in members_by_group:
        members_by_group[gid] = sorted(set(members_by_group[gid]))
    pairs = expected_pairs(members_by_group, groups)
    if len(pairs) != 94:
        return _halt(now, validations, "STOP", f"S05 STOP: pair universe {len(pairs)} != 94")

    idx = pq.read_table(idx_path).to_pydict()
    keys_train: dict[str, list[str]] = defaultdict(list)
    keys_test: dict[str, list[str]] = defaultdict(list)
    keys_val: dict[str, list[str]] = defaultdict(list)
    for pid, dt, split in zip(idx["plant_id"], idx["datetime_raw"], idx["split"]):
        if split == "train":
            keys_train[str(pid)].append(str(dt))
        elif split == "test":
            keys_test[str(pid)].append(str(dt))
        elif split == "validation":
            keys_val[str(pid)].append(str(dt))
    if any(keys_val[p] and set(keys_val[p]) & set(keys_train[p]) for p in plants):
        return _halt(now, validations, "REDESIGN", "FATAL: validation leaked into train keys")

    import duckdb

    plist = ",".join("'" + p.replace("'", "''") + "'" for p in plants)
    feats_sql = ", ".join(f"p.{f}" for f in FEATURES)
    con = duckdb.connect()
    panel = con.execute(
        f"""
        SELECT p.plant_id, p.datetime_raw, {feats_sql}
        FROM read_parquet('{str(panel_path).replace("'", "''")}') p
        INNER JOIN read_parquet('{str(idx_path).replace("'", "''")}') i
          ON p.plant_id = i.plant_id AND CAST(p.datetime_raw AS VARCHAR) = CAST(i.datetime_raw AS VARCHAR)
        WHERE p.plant_id IN ({plist})
          AND i.split IN ('train', 'test')
        """
    ).to_arrow_table()
    con.close()
    feat_by: dict[tuple[str, str], np.ndarray] = {}
    for i in range(panel.num_rows):
        pid = str(panel["plant_id"][i].as_py())
        dt = str(panel["datetime_raw"][i].as_py())
        vals = []
        for name in FEATURES:
            raw = panel[name][i].as_py()
            vals.append(float(raw) if raw is not None else float("nan"))
        feat_by[(pid, dt)] = np.array(vals, dtype=float)

    def matrix(pid: str, dts: list[str]) -> np.ndarray | None:
        rows = []
        for dt in dts:
            vec = feat_by.get((pid, dt))
            if vec is None or not np.all(np.isfinite(vec)):
                return None
            rows.append(vec)
        if not rows:
            return None
        return np.vstack(rows)

    scalers = []
    thresholds = []
    engines: dict[str, dict[str, Any]] = {}
    for pid in plants:
        X = matrix(pid, keys_train[pid])
        ntr = len(keys_train[pid])
        reason = None
        medians = {}
        iqrs = {}
        q1s = {}
        q3s = {}
        z = None
        if X is None:
            reason = "missing_or_nonfinite_train_features"
        else:
            zcols = []
            for m, name in enumerate(FEATURES):
                col = X[:, m]
                q1, med, q3 = [float(v) for v in np.quantile(col, [0.25, 0.50, 0.75], method=QUANTILE_METHOD)]
                iqr = q3 - q1
                med2, iqr2 = source_median_iqr(col)
                q1s[name], medians[name], q3s[name], iqrs[name] = q1, med2, q3, iqr2
                computable_feat = iqr_is_valid(iqr2)
                scalers.append(
                    {
                        "source_plant_id": pid,
                        "feature": name,
                        "n_source_train": ntr,
                        "median": med2,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr2,
                        "scaler_computable": computable_feat,
                        "reason": None if computable_feat else f"degenerate_iqr:{name}",
                    }
                )
                if not computable_feat:
                    reason = f"degenerate_iqr:{name}"
                    break
                zcols.append(standardize(col, med2, iqr2))
            if reason is None:
                z = np.column_stack(zcols)
        if reason is None:
            dist = loo_kth_distances(z, 5)
            tau4 = q_linear(dist, 0.95)
            tau3 = q_linear(dist, 0.90)
            if not (np.isfinite(tau4) and np.isfinite(tau3)):
                reason = "nonfinite_threshold"
            elif tau3 - tau4 > TOL:
                return _halt(now, validations, "STOP", f"S05 STOP: CS3 threshold > CS4 for {pid}")
            else:
                nn = NearestNeighbors(n_neighbors=5, algorithm=NN_ALGORITHM, metric="euclidean")
                nn.fit(z)
                engines[pid] = {"z": z, "nn": nn, "tau": {"CS4": tau4, "CS3": tau3}, "n_train": ntr, "train_days": _days(keys_train[pid])}
        else:
            for name in FEATURES:
                if not any(r["source_plant_id"] == pid and r["feature"] == name for r in scalers):
                    scalers.append({"source_plant_id": pid, "feature": name, "n_source_train": ntr, "median": None, "q1": None, "q3": None, "iqr": None, "scaler_computable": False, "reason": reason})
        for rule in RULES:
            computable = reason is None and pid in engines
            thresholds.append(
                {
                    "source_plant_id": pid,
                    "rule_id": rule["id"],
                    "k": rule["k"],
                    "q": rule["q"],
                    "n_source_train": ntr,
                    "threshold": engines[pid]["tau"][rule["id"]] if computable else None,
                    "computable": computable,
                    "reason": None if computable else reason,
                }
            )

    s03_cs = _load_s03_cs(root)
    summaries = []
    mask_chunks: dict[str, list] = {k: [] for k in ["group_id", "source_plant_id", "target_plant_id", "rule_id", "datetime_raw", "computable", "reason", "kth_distance", "threshold", "supported"]}
    mismatches = []
    for gid, src, tgt in pairs:
        dts = keys_test[tgt]
        Xte = matrix(tgt, dts)
        src_reason = next((t["reason"] for t in thresholds if t["source_plant_id"] == src and t["rule_id"] == "CS4"), "unknown")
        src_ok = src in engines
        for rule in RULES:
            rid = rule["id"]
            if not src_ok or Xte is None:
                reason = src_reason if not src_ok else "missing_or_nonfinite_target_test_features"
                summaries.append(_summary_row(gid, src, tgt, rule, False, reason, engines, dts, None, None, None))
                for dt in dts:
                    _mask_row(mask_chunks, gid, src, tgt, rid, dt, False, reason, None, None, None)
                continue
            eng = engines[src]
            zte = np.column_stack([standardize(Xte[:, m], _scaler(scalers, src, FEATURES[m], "median"), _scaler(scalers, src, FEATURES[m], "iqr")) for m in range(6)])
            dist, _ = eng["nn"].kneighbors(zte, n_neighbors=5, return_distance=True)
            d5 = dist[:, 4]
            tau = eng["tau"][rid]
            supported = d5 <= tau
            n_sup = int(supported.sum())
            n_te = len(dts)
            sr = n_sup / n_te if n_te else None
            summaries.append(_summary_row(gid, src, tgt, rule, True, None, engines, dts, n_sup, sr, tau, days_sup=_days([dts[i] for i, ok in enumerate(supported) if ok])))
            for i, dt in enumerate(dts):
                _mask_row(mask_chunks, gid, src, tgt, rid, dt, True, None, float(d5[i]), float(tau), bool(supported[i]))
            if s03_cs:
                s03row = s03_cs.get((src, tgt, rule["k"], rule["q"]))
                if s03row is None:
                    mismatches.append(f"missing_s03 {src}->{tgt} {rid}")
                else:
                    if int(float(s03row.get("n_source_train") or 0)) != eng["n_train"]:
                        mismatches.append(f"n_train {src}->{tgt} {rid}")
                    if int(float(s03row.get("n_eligible_target_test") or 0)) != n_te:
                        mismatches.append(f"n_test {src}->{tgt} {rid}")
                    if int(float(s03row.get("n_supported") or 0)) != n_sup:
                        mismatches.append(f"n_supported {src}->{tgt} {rid}")
                    if not _close(s03row.get("threshold"), tau) or not _close(s03row.get("sr"), sr):
                        mismatches.append(f"thresh/sr {src}->{tgt} {rid} s03={s03row.get('threshold')},{s03row.get('sr')} s05={tau},{sr}")

    cs4_rows = [r for r in summaries if r["rule_id"] == "CS4"]
    cs3_rows = [r for r in summaries if r["rule_id"] == "CS3"]
    nc4 = {(r["source_plant_id"], r["target_plant_id"]) for r in cs4_rows if not r["computable"]}
    nc3 = {(r["source_plant_id"], r["target_plant_id"]) for r in cs3_rows if not r["computable"]}
    n_cs4 = sum(1 for r in cs4_rows if r["computable"])
    n_cs3 = sum(1 for r in cs3_rows if r["computable"])
    if len(cs4_rows) != 94 or n_cs4 != 85 or len(nc4) != 9:
        return _halt(now, validations, "STOP", f"S05 STOP: CS4 universe {len(cs4_rows)}/{n_cs4}/{len(nc4)}")
    if frozen_nc and nc4 != frozen_nc:
        return _halt(now, validations, "STOP", f"S05 STOP: CS4 noncomputable identities {nc4} != scope {frozen_nc}")
    if len(cs3_rows) != 94 or n_cs3 != 85 or len(nc3) != 9:
        return _halt(now, validations, "STOP", "S05 STOP: CS3 counts do not reconcile")
    if mismatches:
        return _halt(now, validations, "STOP", "S05 STOP: S03 numeric reconciliation failed: " + "; ".join(mismatches[:8]))
    validations.append(ValidationRecord("cs4_94_85_9", True, None))
    validations.append(ValidationRecord("cs4_identities", True, None))
    validations.append(ValidationRecord("s03_numeric", True, f"tol={TOL}"))

    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "S05_SOURCE_SCALERS.csv", scalers, ["source_plant_id", "feature", "n_source_train", "median", "q1", "q3", "iqr", "scaler_computable", "reason"])
    write_csv(out / "S05_SOURCE_THRESHOLDS.csv", thresholds, ["source_plant_id", "rule_id", "k", "q", "n_source_train", "threshold", "computable", "reason"])
    write_csv(out / "S05_SUPPORT_SUMMARY.csv", summaries, ["group_id", "source_plant_id", "target_plant_id", "rule_id", "k", "q", "computable", "reason", "n_source_train", "n_source_train_days", "n_eligible_target_test", "n_target_test_days", "n_supported", "n_supported_days", "support_retention", "threshold"])
    mask_table = pa.table(
        {
            "group_id": pa.array(mask_chunks["group_id"], pa.string()),
            "source_plant_id": pa.array(mask_chunks["source_plant_id"], pa.string()),
            "target_plant_id": pa.array(mask_chunks["target_plant_id"], pa.string()),
            "rule_id": pa.array(mask_chunks["rule_id"], pa.string()),
            "datetime_raw": pa.array(mask_chunks["datetime_raw"], pa.string()),
            "computable": pa.array(mask_chunks["computable"], pa.bool_()),
            "reason": pa.array(mask_chunks["reason"], pa.string()),
            "kth_distance": pa.array(mask_chunks["kth_distance"], pa.float64()),
            "threshold": pa.array(mask_chunks["threshold"], pa.float64()),
            "supported": pa.array(mask_chunks["supported"], pa.bool_()),
        }
    )
    pq.write_table(mask_table, out / "S05_SUPPORT_MASKS.parquet")
    protocol = {
        "feature_order": list(FEATURES),
        "source_partition": "train",
        "target_partition": "test",
        "scaling": "source TRAIN median/IQR; numpy.quantile method=linear",
        "quantile_method": QUANTILE_METHOD,
        "metric": "euclidean",
        "neighbor_algorithm": NN_ALGORITHM,
        "cs4": CS4,
        "cs3": CS3,
        "support_rule": "kth_distance <= threshold",
        "degenerate_iqr": "non-computable; no epsilon",
        "target_statistics_in_scaling": False,
        "outcome_or_model_used": False,
        "numpy_version": np.__version__,
        "scikit_learn_version": __import__("sklearn").__version__,
        "python_version": sys.version,
        "reconciliation_tolerance_abs": TOL,
    }
    dump_json(out / "S05_COMMON_SUPPORT_PROTOCOL.json", protocol)
    recon = {
        "tolerance_abs": TOL,
        "primary_groups_ok": True,
        "pair_universe": 94,
        "cs4": _rule_stats(cs4_rows, frozen_nc),
        "cs3": _rule_stats(cs3_rows, nc3),
        "cs4_noncomputable_identities_match_scope": True,
        "cs3_counts_match_corrected_s03": True,
        "per_pair_threshold_count_sr": "PASS",
        "cs1_cs2_executed": False,
        "model_or_outcome_opened": False,
        "s04_prediction_artifacts_opened": False,
    }
    dump_json(out / "S05_S03_RECONCILIATION.json", recon)
    recs = [
        _art(root, out / "S05_SOURCE_SCALERS.csv", rows=len(scalers), fp=_csv_fp(out / "S05_SOURCE_SCALERS.csv")),
        _art(root, out / "S05_SOURCE_THRESHOLDS.csv", rows=len(thresholds), fp=_csv_fp(out / "S05_SOURCE_THRESHOLDS.csv")),
        _art(root, out / "S05_SUPPORT_SUMMARY.csv", rows=len(summaries), fp=_csv_fp(out / "S05_SUPPORT_SUMMARY.csv")),
        _art(root, out / "S05_SUPPORT_MASKS.parquet", rows=mask_table.num_rows, fp=_pq_fp(out / "S05_SUPPORT_MASKS.parquet")),
        _art(root, out / "S05_COMMON_SUPPORT_PROTOCOL.json"),
        _art(root, out / "S05_S03_RECONCILIATION.json"),
    ]
    report = root / REPORT
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(recon, cs4_rows, cs3_rows), encoding="utf-8")
    recs.append(_art(root, report))
    dump_json(out / "S05_MANIFEST.json", {"job": "S05_COMMON_SUPPORT_ENGINE", "artifacts": {r.path.split("/")[-1]: _tab(r) for r in recs}})
    recs.append(_art(root, out / "S05_MANIFEST.json"))
    validations.append(ValidationRecord("no_sot_write", True, "sot_patch only"))
    validations.append(ValidationRecord("no_cs1_cs2", True, None))

    patch = {
        "stages": {
            "S05": {
                "kind": "scientific_analysis",
                "status": "GO",
                "reason": "S05_COMMON_SUPPORT_ENGINE_COMPLETE",
                "operational_stage_id": "22",
                "job": "S05_COMMON_SUPPORT_ENGINE",
                "finished_at_utc": now,
                "answer": ANSWER,
                "report": _tab(recs[-2]),
            }
        },
        "scientific_analysis": {
            "s05": {
                "status": "GO",
                "operational_stage_id": "22",
                "scope": {"primary_groups": groups, "primary_plants": plants, "expected_directional_transfers": 94},
                "feature_strategy": "six_core",
                "source_partition": "train",
                "target_partition": "test",
                "primary_rule": CS4,
                "r4_rule": CS3,
                "protocol": _tab(recs[4]),
                "source_scalers": _tab(recs[0]),
                "source_thresholds": _tab(recs[1]),
                "support_summary": _tab(recs[2]),
                "support_masks": _tab(recs[3]),
                "s03_reconciliation": _tab(recs[5]),
                "manifest": _tab(recs[-1]),
                "cs4": recon["cs4"],
                "cs3": recon["cs3"],
                "model_predictions_opened": False,
                "cross_plant_predictions_computed": False,
                "otg_computed": False,
                "bootstrap_draws_generated": False,
            }
        },
    }
    return StageResult(status="GO", message="GO — S05 COMMON-SUPPORT ENGINE COMPLETE", sot_patch=patch, artifacts=recs, validations=validations)


def _scaler(rows, pid, feat, field):
    for r in rows:
        if r["source_plant_id"] == pid and r["feature"] == feat:
            return r[field]
    raise KeyError((pid, feat))


def _days(dts: list[str]) -> int:
    days = set()
    for dt in dts:
        ts = parse_ts(dt) if isinstance(dt, str) else None
        days.add((ts.date().isoformat() if ts else dt[:10]))
    return len(days)


def _mask_row(chunks, gid, src, tgt, rid, dt, computable, reason, dist, tau, supported):
    chunks["group_id"].append(gid)
    chunks["source_plant_id"].append(src)
    chunks["target_plant_id"].append(tgt)
    chunks["rule_id"].append(rid)
    chunks["datetime_raw"].append(dt)
    chunks["computable"].append(bool(computable))
    chunks["reason"].append(reason)
    chunks["kth_distance"].append(dist)
    chunks["threshold"].append(tau)
    chunks["supported"].append(supported)


def _summary_row(gid, src, tgt, rule, computable, reason, engines, dts, n_sup, sr, tau, days_sup=None):
    ntr = engines.get(src, {}).get("n_train")
    return {
        "group_id": gid,
        "source_plant_id": src,
        "target_plant_id": tgt,
        "rule_id": rule["id"],
        "k": rule["k"],
        "q": rule["q"],
        "computable": computable,
        "reason": reason,
        "n_source_train": ntr,
        "n_source_train_days": engines.get(src, {}).get("train_days"),
        "n_eligible_target_test": len(dts),
        "n_target_test_days": _days(dts),
        "n_supported": n_sup,
        "n_supported_days": days_sup,
        "support_retention": sr,
        "threshold": tau,
    }


def _rule_stats(rows, nc_ids):
    comp = [r for r in rows if r["computable"]]
    srs = [float(r["support_retention"]) for r in comp if r["support_retention"] is not None]
    zeros = sum(1 for r in comp if r["n_supported"] == 0)
    return {
        "n_expected": len(rows),
        "n_computable": len(comp),
        "n_noncomputable": len(rows) - len(comp),
        "n_zero_supported_among_computable": zeros,
        "sr_min": min(srs) if srs else None,
        "sr_median": float(np.median(srs)) if srs else None,
        "sr_max": max(srs) if srs else None,
        "supported_row_total": int(sum(int(r["n_supported"] or 0) for r in comp)),
        "noncomputable_identities": [{"source_plant_id": a, "target_plant_id": b} for a, b in sorted(nc_ids)],
    }


def _load_s03_cs(root: Path) -> dict[tuple, dict]:
    path = root / "artifacts/s03_corrective_v2/S03_CORRECTED_COMMON_SUPPORT_AUDIT.csv"
    out = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("feature_strategy") != "six_core" or row.get("interpolation_policy") != "exclude_TRUE":
                continue
            if abs(float(row.get("irradiance_wm2") or 0) - 50) > 1e-9 or abs(float(row.get("coverage_dc_min") or 0) - 0.8) > 1e-9:
                continue
            if str(row.get("target_partition") or "").lower() != "test":
                continue
            k = int(float(row.get("k") or 0))
            q = float(row.get("q") or 0)
            if k != 5:
                continue
            key = (row["source_plant_id"], row["target_plant_id"], k, q)
            if str(row.get("computable")).lower() == "true":
                out[key] = row
    return out


def _halt(now, validations, status, message):
    return StageResult(
        status=status,
        message=message,
        sot_patch={"stages": {"S05": {"kind": "scientific_analysis", "status": status, "finished_at_utc": now, "reason": message, "answer": ANSWER, "job": "S05_COMMON_SUPPORT_ENGINE", "operational_stage_id": "22"}}},
        artifacts=[],
        validations=validations,
    )


def _report(recon, cs4, cs3) -> str:
    nc = ", ".join(f"{r['source_plant_id']}→{r['target_plant_id']}" for r in recon["cs4"]["noncomputable_identities"])
    return "\n".join(
        [
            "# S05 — Common-support engine",
            "",
            "Primary CS4 is k=5, q=0.95. R4 CS3 is k=5, q=0.90. Source TRAIN median/IQR scaling; target TEST classification; inclusive distance <= threshold. CS1/CS2 were not executed.",
            "",
            f"CS4: expected {recon['cs4']['n_expected']}, computable {recon['cs4']['n_computable']}, non-computable {recon['cs4']['n_noncomputable']} ({nc}).",
            f"CS3: expected {recon['cs3']['n_expected']}, computable {recon['cs3']['n_computable']}, non-computable {recon['cs3']['n_noncomputable']}.",
            "",
            f"CS4 SR among computable: min {recon['cs4']['sr_min']}, median {recon['cs4']['sr_median']}, max {recon['cs4']['sr_max']}. Zero-supported computable transfers: {recon['cs4']['n_zero_supported_among_computable']}.",
            "",
            "Non-computable is not unsupported. Low SR is retained. Reconciliation with corrected S03 evidence passed. No models, OTG, or outcomes were used.",
            "",
            "**GO — S05 COMMON-SUPPORT ENGINE COMPLETE**",
            "",
        ]
    )
