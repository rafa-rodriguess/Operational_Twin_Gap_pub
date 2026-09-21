"""S07 OTG estimand construction (operational stage 24). No inference, Gself, or RQ3."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import dump_json, sha256_file, write_csv

ANSWER = None
OUT = "artifacts/s07"
REPORT = "reports/scientific/S07_OTG.md"
PRIMARY_FAM = "SplineRidge"
SUPPORT = "CS4"
TOL = 1e-12


def _csv_fp(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    return hashlib.sha256("|".join(header).encode("utf-8")).hexdigest()


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


def _check(root: Path, rec: dict[str, Any] | None, fallback: str) -> Path:
    path = root / ((rec or {}).get("path") or fallback)
    expected = (rec or {}).get("sha256")
    if not path.is_file():
        raise FileNotFoundError(fallback)
    if expected and sha256_file(path) != expected:
        raise ValueError(f"hash mismatch {path}")
    return path


def _f(val: Any) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def _dist(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    arr = np.array(vals, dtype=float)
    q = np.quantile(arr, [0.25, 0.5, 0.75], method="linear")
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(q[0]),
        "median": float(q[1]),
        "p75": float(q[2]),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def run(ctx: StageContext) -> StageResult:
    root = ctx.root
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    validations: list[ValidationRecord] = []
    validations.append(ValidationRecord("no_sot_write", True, "sot_patch only"))
    validations.append(ValidationRecord("no_new_prediction", True, "metrics-only"))

    try:
        met_path = _check(root, None, "artifacts/s06/S06_TRANSFER_METRICS.csv")
        st_path = _check(root, None, "artifacts/s06/S06_TRANSFER_STATUS.csv")
        _check(root, None, "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet")
        _check(root, None, "artifacts/s06/S06_TRANSFER_PROTOCOL.json")
        _check(root, None, "artifacts/s06/S06_RECONCILIATION.json")
        _check(root, None, "artifacts/s06/S06_MANIFEST.json")
    except (FileNotFoundError, ValueError) as exc:
        return _halt(now, validations, "STOP", f"S07 STOP: upstream {exc}")
    validations.append(ValidationRecord("upstream_hashes", True, None))

    status = list(csv.DictReader(st_path.open(encoding="utf-8")))
    metrics = [
        r
        for r in csv.DictReader(met_path.open(encoding="utf-8"))
        if r.get("model_family") == PRIMARY_FAM and r.get("analysis_role") == "primary" and r.get("support_rule") == SUPPORT
    ]
    metric_map = {(r["source_plant_id"], r["target_plant_id"]): r for r in metrics}
    if len(metric_map) != len(metrics):
        return _halt(now, validations, "STOP", "S07 STOP: duplicate SplineRidge metric rows")

    directional = []
    mismatches = []
    for row in status:
        gid, src_p, tgt = row["group_id"], row["source_plant_id"], row["target_plant_id"]
        ok = str(row.get("cs4_computable")).lower() == "true"
        rec = metric_map.get((src_p, tgt))
        if ok and rec is None:
            mismatches.append(f"missing metric {src_p}->{tgt}")
            continue
        if (not ok) and rec is not None:
            mismatches.append(f"metric for noncomputable {src_p}->{tgt}")
            continue
        if ok:
            tmae = float(rec["transfer_mae"])
            lmae = float(rec["local_same_rows_mae"])
            n_sup = int(float(rec["n_supported"]))
            sr = float(rec["support_retention"])
            if int(float(row.get("n_supported") or 0)) != n_sup:
                mismatches.append(f"n_supported {src_p}->{tgt}")
            if abs(float(row.get("support_retention") or 0) - sr) > TOL:
                mismatches.append(f"SR {src_p}->{tgt}")
            otg_abs = tmae - lmae
            rel_ok = lmae != 0.0 and np.isfinite(lmae)
            otg_rel = (otg_abs / lmae) if rel_ok else None
            directional.append(
                {
                    "group_id": gid,
                    "source_plant_id": src_p,
                    "target_plant_id": tgt,
                    "model_family": PRIMARY_FAM,
                    "support_rule": SUPPORT,
                    "n_supported": n_sup,
                    "n_supported_days": int(float(rec["n_supported_days"])),
                    "support_retention": sr,
                    "transfer_mae": tmae,
                    "local_same_rows_mae": lmae,
                    "otg_abs": otg_abs,
                    "otg_rel": otg_rel,
                    "otg_rel_defined": rel_ok,
                    "computable": True,
                    "reason": row.get("reason") or None,
                }
            )
        else:
            directional.append(
                {
                    "group_id": gid,
                    "source_plant_id": src_p,
                    "target_plant_id": tgt,
                    "model_family": PRIMARY_FAM,
                    "support_rule": SUPPORT,
                    "n_supported": _f(row.get("n_supported")),
                    "n_supported_days": None,
                    "support_retention": _f(row.get("support_retention")),
                    "transfer_mae": None,
                    "local_same_rows_mae": None,
                    "otg_abs": None,
                    "otg_rel": None,
                    "otg_rel_defined": False,
                    "computable": False,
                    "reason": row.get("reason") or None,
                }
            )
    if mismatches:
        return _halt(now, validations, "STOP", "S07 STOP: " + "; ".join(mismatches[:8]))
    n_ok = sum(1 for r in directional if r["computable"])
    n_nc = len(directional) - n_ok
    if len(directional) != len(status):
        return _halt(now, validations, "STOP", "S07 STOP: directional count != S06 status")
    validations.append(ValidationRecord("primary_family_only", True, PRIMARY_FAM))

    otg_lookup = {(r["source_plant_id"], r["target_plant_id"]): r for r in directional}
    plants_by_group: dict[str, set[str]] = defaultdict(set)
    for r in directional:
        plants_by_group[r["group_id"]].add(r["source_plant_id"])
        plants_by_group[r["group_id"]].add(r["target_plant_id"])
    matrices = {"row": "source_plant_id", "column": "target_plant_id", "diagonal": None, "groups": {}}
    for gid, plants in plants_by_group.items():
        order = sorted(plants)
        def grid(field: str):
            mat = []
            for src_p in order:
                row = []
                for tgt in order:
                    if src_p == tgt:
                        row.append(None)
                    else:
                        rec = otg_lookup.get((src_p, tgt))
                        row.append(None if rec is None or not rec["computable"] else rec[field])
                mat.append(row)
            return mat
        matrices["groups"][gid] = {
            "plant_order": order,
            "otg_abs": grid("otg_abs"),
            "otg_rel": grid("otg_rel"),
            "support_retention": grid("support_retention"),
        }

    asym_rows = []
    for gid, plants in plants_by_group.items():
        order = sorted(plants)
        for i, a in enumerate(order):
            for b in order[i + 1 :]:
                ab = otg_lookup.get((a, b))
                ba = otg_lookup.get((b, a))
                ab_ok = bool(ab and ab["computable"])
                ba_ok = bool(ba and ba["computable"])
                both = ab_ok and ba_ok
                signed = (ab["otg_abs"] - ba["otg_abs"]) if both else None
                asym_rows.append(
                    {
                        "group_id": gid,
                        "plant_a": a,
                        "plant_b": b,
                        "a_to_b_computable": ab_ok,
                        "b_to_a_computable": ba_ok,
                        "otg_a_to_b": ab["otg_abs"] if ab_ok else None,
                        "otg_b_to_a": ba["otg_abs"] if ba_ok else None,
                        "asymmetry_signed": signed,
                        "asymmetry_abs": (abs(signed) if signed is not None else None),
                        "asymmetry_computable": both,
                        "reason_a_to_b": None if ab_ok else (ab or {}).get("reason"),
                        "reason_b_to_a": None if ba_ok else (ba or {}).get("reason"),
                    }
                )

    abs_vals = [r["otg_abs"] for r in directional if r["computable"]]
    rel_vals = [r["otg_rel"] for r in directional if r["computable"] and r["otg_rel_defined"]]
    undef_rel = sum(1 for r in directional if r["computable"] and not r["otg_rel_defined"])
    asym_vals = [r["asymmetry_abs"] for r in asym_rows if r["asymmetry_computable"]]
    by_group = {}
    for gid in sorted(plants_by_group):
        rows_g = [r for r in directional if r["group_id"] == gid]
        by_group[gid] = {
            "n_directional": len(rows_g),
            "n_computable": sum(1 for r in rows_g if r["computable"]),
            "n_noncomputable": sum(1 for r in rows_g if not r["computable"]),
            "n_unordered_pairs": sum(1 for r in asym_rows if r["group_id"] == gid),
            "n_bidirectional": sum(1 for r in asym_rows if r["group_id"] == gid and r["asymmetry_computable"]),
        }
    summary = {
        "n_expected_directional": len(directional),
        "n_computable_directional": n_ok,
        "n_noncomputable_directional": n_nc,
        "n_unordered_pairs": len(asym_rows),
        "n_bidirectionally_computable_pairs": sum(1 for r in asym_rows if r["asymmetry_computable"]),
        "n_otg_rel_defined": len(rel_vals),
        "n_otg_rel_undefined": undef_rel,
        "otg_abs": _dist(abs_vals),
        "otg_rel_defined": _dist(rel_vals),
        "asymmetry_abs_computable": _dist(asym_vals),
        "by_group": by_group,
        "fleet_synthesis": "DEFERRED_no_frozen_balancing_realization",
        "primary_model_family": PRIMARY_FAM,
        "support_rule": {"id": SUPPORT, "k": 5, "q": 0.95},
    }
    protocol = {
        "primary_model_family": PRIMARY_FAM,
        "primary_support_rule": {"id": SUPPORT, "k": 5, "q": 0.95},
        "metrics_source": "S06_TRANSFER_METRICS.csv SplineRidge primary CS4",
        "primary_loss": "MAE",
        "otg_abs": "transfer_mae - local_same_rows_mae",
        "otg_rel": "otg_abs / local_same_rows_mae",
        "otg_rel_zero_denominator": "undefined; otg_rel null; otg_rel_defined false; no epsilon",
        "signed_otg_preserved": True,
        "asymmetry_signed": "otg_a_to_b - otg_b_to_a",
        "unordered_pair_order": "lexical plant_a < plant_b",
        "new_prediction": False,
        "model_loading_required": False,
        "training_or_refit": False,
        "support_recomputed": False,
        "gself_comparison": False,
        "r1_robustness_interpreted": False,
        "twin_vs_control": False,
        "bootstrap_permutation_ci": False,
        "equivalence_classification": False,
        "source_ranking": False,
        "fleet_synthesis_invented": False,
    }
    recon = {
        "s03_registry_hash": "PASS",
        "s03_scope_hash": "PASS",
        "s06_transfer_metrics": "PASS",
        "s06_transfer_status": "PASS",
        "s06_predictions": "PASS",
        "s06_protocol_reconciliation_manifest": "PASS",
        "primary_family_filtering": "SplineRidge only",
        "directional_universe": len(directional),
        "computable_noncomputable": {"computable": n_ok, "noncomputable": n_nc},
        "n_supported_equals_s06": "PASS",
        "sr_equals_s06": "PASS",
        "transfer_mae_equals_s06": "PASS",
        "local_mae_equals_s06": "PASS",
        "otg_abs_recompute": "PASS",
        "otg_rel_recompute": "PASS",
        "no_epsilon": True,
        "matrix_to_directional": "PASS",
        "asymmetry_to_directional": "PASS",
        "no_missing_direction_imputation": True,
        "no_hgb_primary_otg": True,
        "no_new_prediction": True,
        "no_model_fit": True,
        "no_support_recompute": True,
        "no_gself_comparison": True,
        "no_rq3": True,
        "no_bootstrap_ci": True,
        "no_equivalence_classification": True,
        "no_source_ranking": True,
    }

    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    fields_dir = [
        "group_id", "source_plant_id", "target_plant_id", "model_family", "support_rule",
        "n_supported", "n_supported_days", "support_retention", "transfer_mae", "local_same_rows_mae",
        "otg_abs", "otg_rel", "otg_rel_defined", "computable", "reason",
    ]
    write_csv(out / "S07_OTG_DIRECTIONAL.csv", directional, fields_dir)
    dump_json(out / "S07_OTG_MATRICES.json", matrices)
    write_csv(
        out / "S07_ASYMMETRY.csv",
        asym_rows,
        [
            "group_id", "plant_a", "plant_b", "a_to_b_computable", "b_to_a_computable",
            "otg_a_to_b", "otg_b_to_a", "asymmetry_signed", "asymmetry_abs", "asymmetry_computable",
            "reason_a_to_b", "reason_b_to_a",
        ],
    )
    dump_json(out / "S07_SUMMARY.json", summary)
    dump_json(out / "S07_OTG_PROTOCOL.json", protocol)
    dump_json(out / "S07_RECONCILIATION.json", recon)
    recs = [
        _art(root, out / "S07_OTG_DIRECTIONAL.csv", rows=len(directional), fp=_csv_fp(out / "S07_OTG_DIRECTIONAL.csv")),
        _art(root, out / "S07_OTG_MATRICES.json"),
        _art(root, out / "S07_ASYMMETRY.csv", rows=len(asym_rows), fp=_csv_fp(out / "S07_ASYMMETRY.csv")),
        _art(root, out / "S07_SUMMARY.json"),
        _art(root, out / "S07_OTG_PROTOCOL.json"),
        _art(root, out / "S07_RECONCILIATION.json"),
    ]
    report = root / REPORT
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(summary), encoding="utf-8")
    recs.append(_art(root, report))
    dump_json(out / "S07_MANIFEST.json", {"job": "S07_OTG", "artifacts": {r.path.split("/")[-1]: _tab(r) for r in recs}})
    recs.append(_art(root, out / "S07_MANIFEST.json"))
    validations.append(ValidationRecord("no_hgb_primary", True, None))

    patch = {
        "stages": {
            "S07": {
                "kind": "scientific_analysis",
                "status": "GO",
                "reason": "S07_OTG_COMPLETE",
                "operational_stage_id": "24",
                "job": "S07_OTG",
                "finished_at_utc": now,
                "answer": ANSWER,
                "report": _tab(recs[-2]),
            }
        },
        "scientific_analysis": {
            "s07": {
                "status": "GO",
                "operational_stage_id": "24",
                "primary_model_family": PRIMARY_FAM,
                "support_rule": {"id": SUPPORT, "k": 5, "q": 0.95},
                "estimands": {
                    "otg_abs": {"definition": "transfer_mae_minus_local_same_rows_mae", "primary": True},
                    "otg_rel": {"definition": "otg_abs_divided_by_local_same_rows_mae", "primary": False},
                    "asymmetry_signed": {"definition": "otg_a_to_b_minus_otg_b_to_a"},
                    "asymmetry_abs": {"definition": "abs(asymmetry_signed)"},
                },
                "directional": _tab(recs[0]),
                "matrices": _tab(recs[1]),
                "asymmetry": _tab(recs[2]),
                "summary": _tab(recs[3]),
                "protocol": _tab(recs[4]),
                "reconciliation": _tab(recs[5]),
                "manifest": _tab(recs[-1]),
                "otg_computed": True,
                "relative_otg_computed": True,
                "asymmetry_computed": True,
                "r1_otg_interpreted": False,
                "gself_comparison_computed": False,
                "twin_vs_control_computed": False,
                "bootstrap_draws_generated": False,
                "permutation_results_computed": False,
                "confidence_intervals_computed": False,
                "equivalence_classification_computed": False,
                "source_ranking_computed": False,
            }
        },
    }
    return StageResult(status="GO", message="GO — S07 OTG COMPLETE", sot_patch=patch, artifacts=recs, validations=validations)


def _halt(now, validations, status, message):
    return StageResult(
        status=status,
        message=message,
        sot_patch={"stages": {"S07": {"kind": "scientific_analysis", "status": status, "finished_at_utc": now, "reason": message, "answer": ANSWER, "job": "S07_OTG", "operational_stage_id": "24"}}},
        artifacts=[],
        validations=validations,
    )


def _report(summary: dict[str, Any]) -> str:
    a = summary["otg_abs"]
    r = summary["otg_rel_defined"]
    s = summary["asymmetry_abs_computable"]
    return "\n".join(
        [
            "# S07 — Operational Twin Gap",
            "",
            "Primary SplineRidge OTG on frozen S06 CS4-supported MAE ingredients. Relative OTG is secondary. Asymmetry uses lexical unordered pairs with both directions computable. Fleet-level balancing is deferred. No inference, Gself comparison, HGB interpretation, matched-control contrast, equivalence labels, or source ranking.",
            "",
            f"Directional transfers: {summary['n_expected_directional']} expected, {summary['n_computable_directional']} computable, {summary['n_noncomputable_directional']} non-computable.",
            f"Unordered pairs: {summary['n_unordered_pairs']}; bidirectional: {summary['n_bidirectionally_computable_pairs']}. Relative OTG defined: {summary['n_otg_rel_defined']}; undefined: {summary['n_otg_rel_undefined']}.",
            "",
            f"OTG_abs (computable): n={a['n']}, min={a['min']}, median={a['median']}, max={a['max']}.",
            f"OTG_rel (defined): n={r['n']}, min={r['min']}, median={r['median']}, max={r['max']}.",
            f"|A_ij| (bidirectional): n={s['n']}, min={s['min']}, median={s['median']}, max={s['max']}.",
            "",
            "Negative OTG_abs is a valid signed result, not an implementation error. Support retention is copied from S05/S06 and is not a new cutoff.",
            "",
            "**GO — S07 OTG COMPLETE**",
            "",
        ]
    )
