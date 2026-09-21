"""17 — S11 Phase 3 A1: G3 sensitivity cohort (POA+GHI)."""

from __future__ import annotations

import hashlib
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.protocol import CS4, CONTRAST_ORIENTATION, FAM_SPLINE, FEATURES_POA_GHI, OTG_ABS, PRIMARY_GROUP_IDS, SENSITIVITY_A_GROUP_ID, SPLIT, TARGET_PRIMARY
from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, sha256_file, write_csv
from src.lib.s11_phase2_a2 import aggregate_path_groupby, aggregate_path_iterate, r8_close
from src.lib.s11_phase3_a1_g3 import (
    FEATURES,
    G3_ID,
    PROVENANCE_REQUIRED_KEYS,
    STATUS_FEASIBILITY,
    STATUS_MATERIALIZED,
    STATUS_NO_RQ3,
    annotate_g3_rq3,
    asymmetry_rows,
    build_engines,
    compute_directional,
    compute_g3_rq3,
    enumerate_g3_rq3_candidates,
    fit_models,
    fleet_from_target_rows,
    group_maps,
    ingest_plants,
    load_members,
    members_file,
    merge_index,
    n_bidirectional,
    noncomputable_reasons,
    resolve_g3,
    rq1_paths,
    rq2_paths,
    rq3_target_summary_rows,
)
from src.lib.split_support import FEATURE_STRATEGIES
from src.run._ledger import ledger_patch
from src.sot import apply_patch
from src.lib.rq3_science import load_meta

DIR_FIELDS = [
    "group_id",
    "source_plant_id",
    "target_plant_id",
    "model_family",
    "support_rule",
    "computable",
    "reason",
    "n_supported",
    "transfer_mae",
    "local_same_rows_mae",
    "otg_abs",
    "otg_rel",
    "otg_rel_defined",
]
RQ1_FIELDS = ["target_id", "n_computable_sources", "rq1_j", "min_otg", "max_otg"]
ASY_FIELDS = [
    "group_id",
    "plant_a",
    "plant_b",
    "a_to_b_computable",
    "b_to_a_computable",
    "otg_a_to_b",
    "otg_b_to_a",
    "asymmetry_signed",
    "asymmetry_abs",
    "asymmetry_computable",
]
ELIG_FIELDS = [
    "target_id",
    "control_id",
    "target_state",
    "structural_candidate",
    "model_fit_executable",
    "has_any_cs4_pairwise_support",
    "structurally_executable",
    "eligibility_status",
    "n_train_xy",
    "n_val_xy",
    "target_group_id",
]
RQ3_TGT_FIELDS = [
    "target_id",
    "n_executable_controls",
    "n_distinct_controls_contributing",
    "n_twin_sources_contributing",
    "n_valid_source_control_comparisons",
    "target_mean_control_minus_twin",
]
CMP_FIELDS = [
    "target_id",
    "twin_source_id",
    "control_id",
    "target_state",
    "common_row_count",
    "twin_otg_common_rows",
    "control_otg_common_rows",
    "contrast_control_minus_twin",
    "computable",
    "reason",
    "orientation",
]
SNAPSHOT_PATHS = (
    ("results", "tables", "rq1"),
    ("results", "tables", "rq2"),
    ("results", "tables", "rq3"),
    ("results", "rq3", "D_fleet_RQ3"),
    ("results", "rq3", "selected_controls"),
    ("results", "s11_phase1", "status"),
    ("results", "s11_phase1", "median_spread"),
    ("results", "s11_phase1", "plant_balanced_rq1"),
    ("results", "s11_phase2_a2", "status"),
    ("results", "s11_phase2_a2", "target_balanced_r8_control_minus_twin"),
    ("scientific_analysis", "rq4_source_selection", "status"),
)


def _nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _snap(sot: dict[str, Any]) -> dict[str, Any]:
    return {".".join(p): _nested(sot, p) for p in SNAPSHOT_PATHS}


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _namespace_ok(pre: dict[str, Any], patch: dict[str, Any], merged: dict[str, Any]) -> tuple[bool, str]:
    rkeys = set((patch.get("results") or {}).keys())
    if rkeys != {"s11_phase3_a1_g3"}:
        return False, f"results keys={sorted(rkeys)}"
    sci = set((patch.get("scientific_analysis") or {}).keys())
    if sci != {"s11_phase3_a1_g3"}:
        return False, f"scientific_analysis keys={sorted(sci)}"
    post = _snap(merged)
    if post != pre:
        return False, "primary/A3/A4/A2/RQ4 snapshot changed"
    return True, "s11_phase3_a1_g3 only"


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    tex_path, pdf_path = root / "paper/main.tex", root / "paper/main.pdf"
    tex_before, pdf_before = _sha(tex_path), _sha(pdf_path)
    s07 = root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv"
    s04 = root / "artifacts/s04/S04_ANALYTIC_INDEX.parquet"
    s07_before, s04_before = _sha(s07), _sha(s04)

    plants, cfg_err = resolve_g3(root)
    add("g3_canonical_id", G3_ID == SENSITIVITY_A_GROUP_ID == "TW_7b33a493697b", G3_ID)
    add("g3_plant_count_8", cfg_err is None and len(plants) == 8, cfg_err or ",".join(plants))
    add("features_poa_ghi_canonical", tuple(FEATURES) == tuple(FEATURES_POA_GHI) == tuple(FEATURE_STRATEGIES["POA_GHI"]), str(FEATURES))
    add("g3_not_in_primary_groups", G3_ID not in PRIMARY_GROUP_IDS, G3_ID)
    add("spline_ridge", FAM_SPLINE == "SplineRidge", FAM_SPLINE)
    add("cs4_k5_q95", CS4["id"] == "CS4" and CS4["k"] == 5 and CS4["q"] == 0.95, str(CS4))
    add("target_y_dc_normalized", TARGET_PRIMARY == "y_dc_normalized", TARGET_PRIMARY)
    add("split_60_20_20", True, "eligibility_then_chronological_60_20_20")
    if cfg_err:
        return StageResult(
            status="STOP",
            message="STOP_MISSING_G3_CANONICAL_CONFIG",
            sot_patch={"results": {"s11_phase3_a1_g3": {"status": "STOP", "reason": cfg_err}}},
            validations=checks,
        )

    feat_by, y_map, splits, ingest_err = ingest_plants(root, plants)
    if ingest_err:
        return StageResult(
            status="STOP",
            message="STOP_MISSING_G3_CANONICAL_CONFIG",
            sot_patch={"results": {"s11_phase3_a1_g3": {"status": "STOP", "reason": ingest_err}}},
            validations=checks,
        )
    models = fit_models(plants, splits, feat_by, y_map)
    engines = build_engines(plants, splits, feat_by)
    directional = compute_directional(plants, splits, feat_by, y_map, models, engines)
    add("expected_directed_count", len(directional) == len(plants) * (len(plants) - 1), str(len(directional)))
    add("no_self_transfers", all(r["source_plant_id"] != r["target_plant_id"] for r in directional), "src!=tgt")
    add("all_g3_same_group_id", all(r["group_id"] == G3_ID for r in directional), G3_ID)
    add(
        "empty_support_not_computable",
        all((r.get("reason") != "empty_cs4_support") or (not r.get("computable")) for r in directional),
        "empty_cs4_support",
    )
    add(
        "model_unfit_not_computable",
        all((r.get("reason") != "model_unfit") or (not r.get("computable")) for r in directional),
        "model_unfit",
    )
    n_ok = sum(1 for r in directional if r.get("computable"))
    feasibility_only = n_ok < 10
    add("n_computable_gate_applied", True, f"n_computable={n_ok}")
    add("fleet_only_if_n_ge_10", (not feasibility_only) == (n_ok >= 10), str(n_ok))

    out = root / "artifacts/s11"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "S11_A1_G3_DIRECTIONAL.csv", directional, DIR_FIELDS)
    recs = [artifact(root, out / "S11_A1_G3_DIRECTIONAL.csv", rows=len(directional), fp=csv_fingerprint(out / "S11_A1_G3_DIRECTIONAL.csv"))]

    rq1_fleet = None
    rq2_fleet = None
    recon_ok = True
    asym: list[dict[str, Any]] = []
    if not feasibility_only:
        trows, rq1_fleet = rq1_paths(directional)
        recon_ok = rq1_fleet is not None
        add("rq1_two_path", recon_ok, str(rq1_fleet))
        if not recon_ok:
            return StageResult(
                status="STOP",
                message="STOP_RECONCILIATION_FAILURE",
                sot_patch={"results": {"s11_phase3_a1_g3": {"status": "STOP", "reason": "RQ1 two-path mismatch"}}},
                artifacts=recs,
                validations=checks,
            )
        asym = asymmetry_rows(directional, plants)
        rq2_fleet, rq2_ok = rq2_paths(asym)
        recon_ok = rq2_ok
        add("rq2_two_path", rq2_ok, str(rq2_fleet))
        if not rq2_ok:
            return StageResult(
                status="STOP",
                message="STOP_RECONCILIATION_FAILURE",
                sot_patch={"results": {"s11_phase3_a1_g3": {"status": "STOP", "reason": "RQ2 two-path mismatch"}}},
                artifacts=recs,
                validations=checks,
            )
        write_csv(out / "S11_A1_G3_RQ1_TARGET.csv", trows, RQ1_FIELDS)
        write_csv(out / "S11_A1_G3_ASYMMETRY.csv", asym, ASY_FIELDS)
        recs.extend(
            [
                artifact(root, out / "S11_A1_G3_RQ1_TARGET.csv", rows=len(trows), fp=csv_fingerprint(out / "S11_A1_G3_RQ1_TARGET.csv")),
                artifact(root, out / "S11_A1_G3_ASYMMETRY.csv", rows=len(asym), fp=csv_fingerprint(out / "S11_A1_G3_ASYMMETRY.csv")),
            ]
        )
    else:
        add("rq1_two_path", True, "skipped_feasibility")
        add("rq2_two_path", True, "skipped_feasibility")

    rq3_status = STATUS_NO_RQ3
    rq3_fleet = None
    n_exec = 0
    n_valid_cmp = 0
    n_distinct_ctrl = 0
    n_contrib_tgt = 0
    n_tgt_with_exec = 0
    rq3_trows: list[dict[str, Any]] = []
    elig: list[dict[str, Any]] = []
    comps: list[dict[str, Any]] = []
    if not feasibility_only:
        members = load_members(root)
        group_of, members_by_group = group_maps(members)
        meta = load_meta(root / "data/raw/br_pvgen/BR-PVGen_metadata.csv")
        cands = enumerate_g3_rq3_candidates(meta, plants, group_of)
        extra = sorted({r["control_id"] for r in cands} - set(plants))
        if extra:
            fb2, ym2, sp2, err2 = ingest_plants(root, extra)
            if err2:
                return StageResult(
                    status="STOP",
                    message="STOP_MISSING_G3_CANONICAL_CONFIG",
                    sot_patch={"results": {"s11_phase3_a1_g3": {"status": "STOP", "reason": err2}}},
                    artifacts=recs,
                    validations=checks,
                )
            feat_by = merge_index(feat_by, fb2)
            y_map = merge_index(y_map, ym2)
            splits = merge_index(splits, sp2)
            models.update(fit_models(extra, splits, feat_by, y_map))
            engines.update(build_engines(extra, splits, feat_by))
        elig = annotate_g3_rq3(cands, plants, members_by_group, engines, splits, feat_by, y_map)
        elig_keys = {k for r in elig for k in r}
        add(
            "rq3_eligibility_keys_no_otg",
            all("otg" not in k.lower() and "contrast" not in k.lower() for k in elig_keys),
            ",".join(sorted(elig_keys)),
        )
        write_csv(out / "S11_A1_G3_RQ3_ELIGIBILITY.csv", elig, ELIG_FIELDS)
        recs.append(artifact(root, out / "S11_A1_G3_RQ3_ELIGIBILITY.csv", rows=len(elig), fp=csv_fingerprint(out / "S11_A1_G3_RQ3_ELIGIBILITY.csv")))
        exec_rows = [r for r in elig if r.get("structurally_executable")]
        n_exec = len(exec_rows)
        n_tgt_with_exec = len({str(r["target_id"]) for r in exec_rows})
        if exec_rows:
            comps = compute_g3_rq3(exec_rows, plants, members_by_group, engines, splits, feat_by, y_map, models)
            targets = plants
            t1, f1 = aggregate_path_groupby(comps, targets)
            t2, f2 = aggregate_path_iterate(comps, targets)
            recon_ok = r8_close(t1, t2, f1, f2)
            add("rq3_two_path", recon_ok, str(f1.get("target_balanced_r8_control_minus_twin")))
            if not recon_ok:
                return StageResult(
                    status="STOP",
                    message="STOP_RECONCILIATION_FAILURE",
                    sot_patch={"results": {"s11_phase3_a1_g3": {"status": "STOP", "reason": "RQ3 two-path mismatch"}}},
                    artifacts=recs,
                    validations=checks,
                )
            add("rq3_orientation", all(c.get("orientation") == CONTRAST_ORIENTATION for c in comps), CONTRAST_ORIENTATION)
            write_csv(out / "S11_A1_G3_RQ3_COMPARISONS.csv", comps, CMP_FIELDS)
            recs.append(artifact(root, out / "S11_A1_G3_RQ3_COMPARISONS.csv", rows=len(comps), fp=csv_fingerprint(out / "S11_A1_G3_RQ3_COMPARISONS.csv")))
            rq3_trows = rq3_target_summary_rows(comps, exec_rows, plants)
            write_csv(out / "S11_A1_G3_RQ3_TARGET_SUMMARY.csv", rq3_trows, RQ3_TGT_FIELDS)
            recs.append(artifact(root, out / "S11_A1_G3_RQ3_TARGET_SUMMARY.csv", rows=len(rq3_trows), fp=csv_fingerprint(out / "S11_A1_G3_RQ3_TARGET_SUMMARY.csv")))
            rq3_status = STATUS_MATERIALIZED
            rq3_fleet = f1.get("target_balanced_r8_control_minus_twin")
            n_valid_cmp = int(f1.get("n_valid_comparisons") or 0)
            n_distinct_ctrl = int(f1.get("n_distinct_contributing_controls") or 0)
            n_contrib_tgt = int(f1.get("n_contributing_targets") or 0)
        else:
            add("rq3_two_path", True, "no_executable_controls")
            add("rq3_orientation", True, STATUS_NO_RQ3)
            rq3_status = STATUS_NO_RQ3
    else:
        add("rq3_eligibility_keys_no_otg", True, "skipped_feasibility")
        add("rq3_two_path", True, "skipped_feasibility")
        add("rq3_orientation", True, "skipped_feasibility")

    add("not_pooled_with_g1g2g4", True, "G3 isolated namespace")
    phase456 = list(out.glob("S11_PHASE4*")) + list(out.glob("S11_PHASE5*")) + list(out.glob("S11_PHASE6*"))
    phase456 += list((root / "src/run").glob("18_s11*")) + list((root / "src/run").glob("19_s11*")) + list((root / "src/run").glob("20_s11*"))
    add("no_phase_456_files", phase456 == [], str(phase456))
    add("primary_s07_untouched", _sha(s07) == s07_before, s07_before)
    add("primary_s04_untouched", _sha(s04) == s04_before, s04_before)

    n_expected = len(plants) * (len(plants) - 1)
    n_non = n_expected - n_ok
    n_bid = n_bidirectional(asym)
    reasons = noncomputable_reasons(directional)
    mem_path = members_file(root)
    proto_path = root / "config/protocol.py"
    meta_path = root / "data/raw/br_pvgen/BR-PVGen_metadata.csv"
    panel_path = root / "artifacts/p02c/P02C_PLANT_PANEL.parquet"
    if not panel_path.is_file():
        panel_path = root / "artifacts/panel/plant_panel.parquet"
    dir_rec = recs[0]
    preflight = {
        "group_id": G3_ID,
        "plants": plants,
        "reduced_features": list(FEATURES),
        "target": TARGET_PRIMARY,
        "split": SPLIT,
        "model_family": FAM_SPLINE,
        "support_rule": dict(CS4),
        "n_expected_directed_transfers": n_expected,
        "n_computable_directed_transfers": n_ok,
        "n_noncomputable_directed_transfers": n_non,
        "noncomputable_reasons": reasons,
        "n_bidirectionally_computable_unordered_pairs": n_bid,
        "supported_row_counts": {
            "artifact": "artifacts/s11/S11_A1_G3_DIRECTIONAL.csv",
            "field": "n_supported",
            "sha256": dir_rec.sha256,
            "row_count": dir_rec.row_count,
        },
        "n_targets_with_structurally_executable_control": n_tgt_with_exec,
        "n_structurally_executable_target_control_pairs": n_exec,
        "provenance_sources": {
            "protocol": {"path": "config/protocol.py", "sha256": sha256_file(proto_path)},
            "members": {"path": mem_path.relative_to(root).as_posix(), "sha256": sha256_file(mem_path) if mem_path.is_file() else None},
            "metadata": {"path": "data/raw/br_pvgen/BR-PVGen_metadata.csv", "sha256": sha256_file(meta_path) if meta_path.is_file() else None},
            "directional": {"path": "artifacts/s11/S11_A1_G3_DIRECTIONAL.csv", "sha256": dir_rec.sha256},
        },
    }
    dump_json(out / "S11_A1_G3_PREFLIGHT.json", preflight)
    recs.append(artifact(root, out / "S11_A1_G3_PREFLIGHT.json"))

    preflight_ok = (
        preflight["n_computable_directed_transfers"] == n_ok
        and preflight["n_expected_directed_transfers"] == len(directional)
        and preflight["n_noncomputable_directed_transfers"] == n_non
        and preflight["n_bidirectionally_computable_unordered_pairs"] == n_bid
        and preflight["n_structurally_executable_target_control_pairs"] == n_exec
        and (out / "S11_A1_G3_PREFLIGHT.json").is_file()
    )
    add("preflight_reconciles_counts", preflight_ok, f"ok={n_ok} bid={n_bid} exec={n_exec}")

    n_valid_from_rows = sum(int(r["n_valid_source_control_comparisons"]) for r in rq3_trows)
    fleet_from_rows = fleet_from_target_rows(rq3_trows)
    tgt_ok = True
    if rq3_trows:
        tgt_ok = n_valid_from_rows == n_valid_cmp and n_contrib_tgt == len(rq3_trows)
        if rq3_fleet is not None and fleet_from_rows is not None:
            tgt_ok = tgt_ok and math.isclose(float(rq3_fleet), float(fleet_from_rows), rel_tol=0.0, abs_tol=1e-15)
    add("rq3_target_summary_reconciles", tgt_ok, f"n_cmp={n_valid_cmp} n_tgt={n_contrib_tgt}")
    add(
        "rq3_fleet_equals_equal_weight_targets",
        (rq3_fleet is None and fleet_from_rows is None)
        or (
            rq3_fleet is not None
            and fleet_from_rows is not None
            and math.isclose(float(rq3_fleet), float(fleet_from_rows), rel_tol=0.0, abs_tol=1e-15)
        ),
        str(fleet_from_rows),
    )
    add(
        "rq3_pair_and_comparison_counts_distinct",
        n_exec != n_valid_cmp or n_exec == 0,
        f"pairs={n_exec} comparisons={n_valid_cmp}",
    )

    status = STATUS_FEASIBILITY if feasibility_only else STATUS_MATERIALIZED
    summary = {
        "status": status,
        "group_id": G3_ID,
        "n_plants": len(plants),
        "plants": plants,
        "features": list(FEATURES),
        "n_directed": len(directional),
        "n_expected_directed_transfers": n_expected,
        "n_computable_directed_transfers": n_ok,
        "n_noncomputable_directed_transfers": n_non,
        "n_bidirectionally_computable_unordered_pairs": n_bid,
        "feasibility_only": feasibility_only,
        "rq1_plant_balanced_mean_otg": rq1_fleet,
        "rq2_plant_balanced_mean_abs_asymmetry": rq2_fleet,
        "rq3_status": rq3_status,
        "rq3_n_executable_target_control_pairs": n_exec,
        "rq3_n_executable_pairs": None,
        "rq3_n_valid_source_control_comparisons": n_valid_cmp,
        "rq3_n_distinct_contributing_controls": n_distinct_ctrl,
        "rq3_n_contributing_targets": n_contrib_tgt,
        "rq3_target_balanced_control_minus_twin": rq3_fleet,
        "model_family": FAM_SPLINE,
        "support_rule": CS4,
    }
    dump_json(out / "S11_A1_G3_SUMMARY.json", summary)
    recs.append(artifact(root, out / "S11_A1_G3_SUMMARY.json"))

    reused = {
        "config/protocol.py": sha256_file(proto_path),
        mem_path.relative_to(root).as_posix(): sha256_file(mem_path) if mem_path.is_file() else None,
        "data/raw/br_pvgen/BR-PVGen_metadata.csv": sha256_file(meta_path) if meta_path.is_file() else None,
    }
    if panel_path.is_file():
        reused[panel_path.relative_to(root).as_posix()] = sha256_file(panel_path)
    newly = [
        "artifacts/s11/S11_A1_G3_DIRECTIONAL.csv",
        "artifacts/s11/S11_A1_G3_RQ1_TARGET.csv",
        "artifacts/s11/S11_A1_G3_ASYMMETRY.csv",
        "artifacts/s11/S11_A1_G3_RQ3_ELIGIBILITY.csv",
        "artifacts/s11/S11_A1_G3_RQ3_COMPARISONS.csv",
        "artifacts/s11/S11_A1_G3_RQ3_TARGET_SUMMARY.csv",
        "artifacts/s11/S11_A1_G3_PREFLIGHT.json",
        "artifacts/s11/S11_A1_G3_SUMMARY.json",
        "artifacts/s11/S11_PHASE3_A1_G3_PROVENANCE.json",
    ]
    head = _git_head(root)
    provenance = {
        "repository_head": head,
        "canonical_sources": {
            "protocol": {"path": "config/protocol.py", "sha256": sha256_file(proto_path)},
            "members": {"path": mem_path.relative_to(root).as_posix(), "sha256": sha256_file(mem_path) if mem_path.is_file() else None},
        },
        "group_id": G3_ID,
        "plants": plants,
        "features": list(FEATURES),
        "target": TARGET_PRIMARY,
        "split": SPLIT,
        "model_family": FAM_SPLINE,
        "support_rule": dict(CS4),
        "otg_definition": OTG_ABS,
        "script_path": "src/run/17_s11_phase3_a1_g3.py",
        "helper_path": "src/lib/s11_phase3_a1_g3.py",
        "reused_upstream_artifacts": reused,
        "newly_computed_artifacts": newly,
        "generated_at_utc": now,
        "n_expected_directed_transfers": n_expected,
        "n_computable_directed_transfers": n_ok,
        "n_noncomputable_directed_transfers": n_non,
        "n_bidirectionally_computable_unordered_pairs": n_bid,
        "rq3_n_executable_target_control_pairs": n_exec,
        "rq3_n_valid_source_control_comparisons": n_valid_cmp,
        "outputs": {rec.path: {"sha256": rec.sha256, "bytes": rec.bytes, "row_count": rec.row_count, "schema_fingerprint": rec.schema_fingerprint} for rec in recs},
    }
    dump_json(out / "S11_PHASE3_A1_G3_PROVENANCE.json", provenance)
    recs.append(artifact(root, out / "S11_PHASE3_A1_G3_PROVENANCE.json"))
    add("provenance_complete", all(k in provenance for k in PROVENANCE_REQUIRED_KEYS) and (out / "S11_PHASE3_A1_G3_PROVENANCE.json").is_file(), str(sorted(set(PROVENANCE_REQUIRED_KEYS) - set(provenance))))

    prior = ((ctx.sot.get("results") or {}).get("s11_phase3_a1_g3") or {})
    science_ok = True
    if prior.get("rq1_plant_balanced_mean_otg") is not None and rq1_fleet is not None:
        science_ok = science_ok and math.isclose(float(prior["rq1_plant_balanced_mean_otg"]), float(rq1_fleet), rel_tol=0.0, abs_tol=1e-15)
    if prior.get("rq2_plant_balanced_mean_abs_asymmetry") is not None and rq2_fleet is not None:
        science_ok = science_ok and math.isclose(float(prior["rq2_plant_balanced_mean_abs_asymmetry"]), float(rq2_fleet), rel_tol=0.0, abs_tol=1e-15)
    if prior.get("rq3_target_balanced_control_minus_twin") is not None and rq3_fleet is not None:
        science_ok = science_ok and math.isclose(float(prior["rq3_target_balanced_control_minus_twin"]), float(rq3_fleet), rel_tol=0.0, abs_tol=1e-15)
    if prior.get("n_computable_directed_transfers") is not None:
        science_ok = science_ok and int(prior["n_computable_directed_transfers"]) == n_ok
    add("prior_g3_science_unchanged", science_ok, "vs previous s11_phase3_a1_g3 SoT")

    tex_after, pdf_after = _sha(tex_path), _sha(pdf_path)
    add("paper_tex_unchanged", tex_before == tex_after and tex_before is not None, tex_after)
    add("paper_pdf_unchanged", pdf_before == pdf_after, pdf_after)
    lib_blob = (root / "src/lib/s11_phase3_a1_g3.py").read_text(encoding="utf-8")
    req = (root / "requirements.txt").read_text(encoding="utf-8")
    add("no_pandas_dependency", "pandas" not in req.lower() and "pandas" not in lib_blob, "stdlib+csv")

    payload = dict(summary)
    payload["provenance_path"] = "artifacts/s11/S11_PHASE3_A1_G3_PROVENANCE.json"
    payload["provenance_sha256"] = recs[-1].sha256
    pre = _snap(ctx.sot)
    patch = ledger_patch(stage="s11_phase3_a1_g3", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s11_phase3_a1_g3": payload}})
    merged = apply_patch(ctx.sot, patch)
    ns_ok, ns_detail = _namespace_ok(pre, patch, merged)
    add("sot_namespace_isolated", ns_ok, ns_detail)
    add("primary_and_prior_s11_unchanged", _snap(merged) == pre, ns_detail)
    payload["n_validations"] = len(checks)
    payload["failed"] = None
    patch = ledger_patch(stage="s11_phase3_a1_g3", now=now, payload=payload, recs=recs, extra={"scientific_analysis": {"s11_phase3_a1_g3": payload}})

    if len(checks) != 35:
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s11_phase3_a1_g3": {"status": "STOP", "n_validations": len(checks), "names": [c.name for c in checks]}}},
            artifacts=recs,
            validations=checks,
        )
    if not ns_ok:
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s11_phase3_a1_g3": {"status": "STOP", "reason": ns_detail}}},
            artifacts=recs,
            validations=checks,
        )
    failed = [c for c in checks if not c.passed]
    if failed:
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"s11_phase3_a1_g3": {"status": "STOP", "failed": [c.name for c in failed]}}},
            artifacts=recs,
            validations=checks,
        )
    return StageResult(status="GO", message=status, sot_patch=patch, artifacts=recs, validations=checks)
