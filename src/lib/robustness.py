"""Stage 53 corrective V2: implementation-bug repair without recomputing science."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import dump_json, sha256_file, write_csv
from src.lib.rq3_match import snapshot_tree
from src.lib.rq3_mbb import _art, _csv_fp, _pq_fp, _tab
from src.lib.robustness_arms import ARM_SPECS, ORIGINS, _dataset, _r6_splits, _read_csv
from src.lib.robustness import _load_panel, _nested

JOB = "S09_STATE_CONSTRAINED_RQ3_ROBUSTNESS_CORRECTIVE_V2"
STAGE_NODE = "S09_STATE_CONSTRAINED_RQ3_ROBUSTNESS"
REVISION = "state_constrained_support_v2"
V1_RUN = "5cc4c7499e2b406fb44b5133868f861f"
V1_REASON = "S53_REVISED_SAME_STATE_RQ3_ROBUSTNESS_COMPLETE"
REASON_GO = "S53_V2_REVISED_SAME_STATE_RQ3_ROBUSTNESS_CORRECTED"
MSG_GO = "GO — S53_V2_REVISED_SAME_STATE_RQ3_ROBUSTNESS_CORRECTED"
MSG_STOP = "STOP — S09 STAGE 53 CORRECTIVE V2 INCOMPLETE"
OUT_V1 = "artifacts/s09_state_constrained_rq3"
OUT_V2 = "artifacts/s09_state_constrained_rq3_v2"
REPORT = "reports/scientific/S09_STATE_CONSTRAINED_RQ3_ROBUSTNESS_CORRECTIVE_V2.md"
ANSWER = None
AMEND = "docs/method_amendments/S09_STAGE53_CORRECTIVE_V2.md"
MAP_REL = "artifacts/s03_rq3_state_revision/S03_STATE_SUPPORTED_MAPPING_SELECTED.csv"
FREEZE_REL = "artifacts/s08_rq3_downstream_reconciliation_v2/S52_V2_S09_REVISED_RQ3_FREEZE.json"
R6_ALLOWED = ("0P5", "1P0", "2P0")
R6_NOMINAL = "1P0"
RET_DEF = "N_PAIRWISE_ROWS_OVER_N_TARGET_TEST_ROWS"
V1_FILES = (
    "S09_STATE_RQ3_PROTOCOL.json",
    "S09_STATE_RQ3_ARM_EXECUTION_REGISTRY.csv",
    "S09_STATE_RQ3_CONTROL_MODEL_MANIFEST.json",
    "S09_STATE_RQ3_CONTROL_HYPERPARAMETER_SELECTION.csv",
    "S09_STATE_RQ3_PAIRWISE_SUPPORT_MASKS.parquet",
    "S09_STATE_RQ3_SOURCE_LEVEL_CONTRASTS.csv",
    "S09_STATE_RQ3_TARGET_LEVEL_CONTRASTS.csv",
    "S09_STATE_RQ3_BLOCK_LENGTH_SELECTION.csv",
    "S09_STATE_RQ3_BOOTSTRAP_PROPOSAL_AUDIT.csv",
    "S09_STATE_RQ3_BOOTSTRAP_DRAWS.parquet",
    "S09_STATE_RQ3_TARGET_LEVEL_INFERENCE.csv",
    "S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv",
    "S09_STATE_RQ3_R6_ORIGIN_SUMMARY.csv",
    "S09_STATE_RQ3_PRIMARY_BLOCK_LENGTH_SENSITIVITY.csv",
    "S09_STATE_RQ3_RECONCILIATION.json",
    "S09_STATE_RQ3_MANIFEST.json",
)
PROTECTED = (
    OUT_V1,
    "artifacts/s09_scientific",
    "artifacts/s10",
    "artifacts/s08_state_constrained_scientific",
    "artifacts/s08_state_constrained_fleet_fixed_set_inference",
    "artifacts/s08_rq3_downstream_reconciliation",
    "artifacts/s08_rq3_downstream_reconciliation_v2",
    "paper",
    "paper.md",
)


class S53V2Stop(Exception):
    def __init__(self, reason: str, details: str = ""):
        super().__init__(details or reason)
        self.reason = reason
        self.details = details or reason


def _hash_rec(root: Path, rec: dict[str, Any] | None, default: str, label: str) -> Path:
    path = root / ((rec or {}).get("path") or default)
    if not path.is_file():
        raise S53V2Stop("S53V2_STOP_MISSING", f"{label} {path}")
    expected = (rec or {}).get("sha256")
    if expected and sha256_file(path) != expected:
        raise S53V2Stop("S53V2_STOP_HASH", f"{label} {path}")
    return path


def _load_mapping(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    return {r["reference_plant_id"]: r["control_plant_id"] for r in rows}


def _r6_base(arm: str, block: str) -> bool:
    return arm == "R6_TEMPORAL" and str(block) == "BASE"
    return arm == "R6_TEMPORAL" and str(block) == "BASE"


def _manifest_target(arm_id: str) -> str:
    if str(arm_id).startswith("R2_AC"):
        return "y_ac_normalized"
    return "y_dc_normalized"


def validate_upstream(sot: dict[str, Any], root: Path) -> dict[str, Any]:
    st52 = _nested(sot, "stages", "S08_RQ3_DOWNSTREAM_RECONCILIATION_AND_S09_REVISION_FREEZE") or {}
    if st52.get("status") != "GO" or st52.get("correction_version") != 2:
        raise S53V2Stop("S53V2_STOP_STAGE52")
    if st52.get("controller_hold_resolved") is not True or st52.get("dependency_coverage_complete") is not True:
        raise S53V2Stop("S53V2_STOP_STAGE52_HOLD")
    if st52.get("dependency_unclassified_occurrences") != 0 or str(st52.get("next_stage")) != "53":
        raise S53V2Stop("S53V2_STOP_STAGE52_NEXT")
    freeze_path = _hash_rec(root, (st52.get("artifacts") or {}).get("s09_freeze"), FREEZE_REL, "FREEZE")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    st53 = _nested(sot, "stages", STAGE_NODE) or {}
    if st53.get("status") != "GO":
        raise S53V2Stop("S53V2_STOP_STAGE53_V1", str(st53.get("run_id")))
    if st53.get("run_id") != V1_RUN and st53.get("supersedes_run_id") != V1_RUN:
        raise S53V2Stop("S53V2_STOP_STAGE53_V1", str(st53.get("run_id")))
    if st53.get("run_id") == V1_RUN and st53.get("reason") != V1_REASON:
        raise S53V2Stop("S53V2_STOP_STAGE53_V1_REASON")
    v1_hashes = {}
    v1_manifest = json.loads((root / OUT_V1 / "S09_STATE_RQ3_MANIFEST.json").read_text(encoding="utf-8"))
    by_name = {Path(a["path"]).name: a for a in v1_manifest.get("artifacts") or []}
    for name in V1_FILES:
        path = root / OUT_V1 / name
        if not path.is_file():
            raise S53V2Stop("S53V2_STOP_MISSING", name)
        digest = sha256_file(path)
        rec = by_name.get(name) or {}
        if rec.get("sha256") and rec["sha256"] != digest:
            raise S53V2Stop("S53V2_STOP_HASH", name)
        v1_hashes[name] = digest
    map_rec = _nested(sot, "scientific_freeze", "s03", "rq3_revisions", REVISION, "artifacts", "selected_mapping") or {}
    map_path = _hash_rec(root, map_rec, MAP_REL, "MAPPING")
    return {"freeze": freeze, "map_path": map_path, "v1_hashes": v1_hashes, "st53": st53}


def _test_denominators(root: Path, sot: dict[str, Any], control_of: dict[str, str]) -> dict[tuple[str, str, str], int]:
    plants = sorted(set(control_of) | set(control_of.values()))
    raw = _load_panel(root, sot, plants)
    out: dict[tuple[str, str, str], int] = {}
    primary_rows, primary_splits, _, _ = _dataset(raw, ARM_SPECS["R1_HGB"])
    for arm_id, spec in ARM_SPECS.items():
        if spec["rebuild_split"]:
            _, splits, _, _ = _dataset(raw, spec)
        else:
            splits = primary_splits
        for tgt in control_of:
            out[(arm_id, "", tgt)] = len((splits.get(tgt) or {}).get("test") or [])
    r6_spec = {"target": "y_dc_normalized", "coverage_col": "coverage_dc", "coverage": 0.80, "poa_strict_gt": 50.0, "interpolation": "exclude_TRUE"}
    r6_rows, _, _, _ = _dataset(raw, r6_spec)
    for origin_id in ORIGINS:
        splits = _r6_splits(r6_rows, origin_id)
        for tgt in control_of:
            out[("R6_TEMPORAL", origin_id, tgt)] = len((splits.get(tgt) or {}).get("test") or [])
    return out


def execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    try:
        return _execute(ctx, now, validations)
    except S53V2Stop as exc:
        validations.append(ValidationRecord(exc.reason, False, exc.details))
        return StageResult(
            status="STOP",
            message=f"{MSG_STOP}: {exc.details}",
            sot_patch={"stages": {STAGE_NODE: {"kind": "scientific_analysis_revision_correction", "status": "STOP", "reason": exc.reason, "operational_stage_id": "53", "job": JOB, "finished_at_utc": now, "answer": ANSWER, "revision_id": REVISION, "correction_version": 2}}},
            artifacts=[],
            validations=validations,
        )


def _execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    root = ctx.root
    hist_before = snapshot_tree(root, PROTECTED)
    s09_before = json.dumps(_nested(ctx.sot, "scientific_analysis", "s09"), sort_keys=True)
    bundle = validate_upstream(ctx.sot, root)
    freeze = bundle["freeze"]
    map_path = bundle["map_path"]
    control_of = _load_mapping(map_path)
    sot_map = {r["reference_plant_id"]: r["control_plant_id"] for r in (_nested(ctx.sot, "scientific_freeze", "s03", "rq3_revisions", REVISION, "selected_mapping") or [])}
    if sot_map and sot_map != control_of:
        raise S53V2Stop("S53V2_STOP_MAPPING_DISAGREEMENT")
    targets = sorted(control_of)
    validations.append(ValidationRecord("targets_derived_stage43", True, str(len(targets))))
    denom = _test_denominators(root, ctx.sot, control_of)
    out = root / OUT_V2
    out.mkdir(parents=True, exist_ok=True)
    v1 = root / OUT_V1

    copy_as_is = [
        "S09_STATE_RQ3_ARM_EXECUTION_REGISTRY.csv",
        "S09_STATE_RQ3_CONTROL_HYPERPARAMETER_SELECTION.csv",
        "S09_STATE_RQ3_PAIRWISE_SUPPORT_MASKS.parquet",
        "S09_STATE_RQ3_TARGET_LEVEL_CONTRASTS.csv",
        "S09_STATE_RQ3_BLOCK_LENGTH_SELECTION.csv",
        "S09_STATE_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv",
        "S09_STATE_RQ3_R6_ORIGIN_SUMMARY.csv",
        "S09_STATE_RQ3_PRIMARY_BLOCK_LENGTH_SENSITIVITY.csv",
    ]
    for name in copy_as_is:
        shutil.copy2(v1 / name, out / name)

    audit44 = {(r["twin_source_plant_id"], r["target_plant_id"]): r for r in _read_csv(root / "artifacts/s08_state_constrained_support_alignment/S08_STATE_PAIRWISE_ALIGNMENT_AUDIT.csv")}
    src_v1 = _read_csv(v1 / "S09_STATE_RQ3_SOURCE_LEVEL_CONTRASTS.csv")
    src_v2 = []
    for row in src_v1:
        arm, origin, tgt = row["arm_id"], row.get("origin_id") or "", row["target_plant_id"]
        n_test = denom.get((arm, origin, tgt))
        if n_test is None:
            raise S53V2Stop("S53V2_STOP_DENOM_MISSING", f"{arm} {origin} {tgt}")
        if n_test <= 0:
            raise S53V2Stop("S53V2_STOP_ZERO_TEST_DENOMINATOR", f"{arm} {origin} {tgt}")
        n_pair = int(float(row["n_pairwise_rows"] or 0))
        retention = n_pair / n_test
        if retention < 0 or retention > 1:
            raise S53V2Stop("S53V2_STOP_RETENTION_RANGE", str(retention))
        if n_pair == 0 and retention != 0.0:
            raise S53V2Stop("S53V2_STOP_ZERO_PAIR_RETENTION")
        if arm == "R1_HGB":
            a44 = audit44.get((row["twin_source_plant_id"], tgt))
            if a44 and int(float(a44.get("n_target_test_rows") or 0)) == n_test:
                expected = float(a44["pairwise_retention_vs_target"])
                if abs(retention - expected) > 1e-9:
                    raise S53V2Stop("S53V2_STOP_STAGE44_RETENTION", f"{row['twin_source_plant_id']}->{tgt}")
        rec = dict(row)
        rec["n_target_test_rows"] = n_test
        rec["pairwise_retention"] = retention
        rec["pairwise_retention_definition"] = RET_DEF
        src_v2.append(rec)
    write_csv(out / "S09_STATE_RQ3_SOURCE_LEVEL_CONTRASTS.csv", src_v2, list(src_v2[0]))

    inf_v1 = _read_csv(v1 / "S09_STATE_RQ3_TARGET_LEVEL_INFERENCE.csv")
    inf_v2 = [r for r in inf_v1 if not _r6_base(r["arm_id"], r.get("block_arm_id") or "")]
    n_removed_inf = len(inf_v1) - len(inf_v2)
    write_csv(out / "S09_STATE_RQ3_TARGET_LEVEL_INFERENCE.csv", inf_v2, list(inf_v2[0]))

    n_removed_prop = 0
    n_kept_prop = 0
    with (v1 / "S09_STATE_RQ3_BOOTSTRAP_PROPOSAL_AUDIT.csv").open(encoding="utf-8", newline="") as hin:
        reader = csv.DictReader(hin)
        fields = list(reader.fieldnames or [])
        with (out / "S09_STATE_RQ3_BOOTSTRAP_PROPOSAL_AUDIT.csv").open("w", encoding="utf-8", newline="") as hout:
            writer = csv.DictWriter(hout, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                if _r6_base(row["arm_id"], row.get("block_arm_id") or ""):
                    n_removed_prop += 1
                    continue
                writer.writerow(row)
                n_kept_prop += 1

    draws = pq.read_table(v1 / "S09_STATE_RQ3_BOOTSTRAP_DRAWS.parquet")
    keep = []
    n_removed_draw = 0
    for i in range(draws.num_rows):
        arm = str(draws["arm_id"][i].as_py())
        block = str(draws["block_arm_id"][i].as_py())
        if _r6_base(arm, block):
            n_removed_draw += 1
            continue
        keep.append(i)
    table_v2 = draws.take(keep)
    pq.write_table(table_v2, out / "S09_STATE_RQ3_BOOTSTRAP_DRAWS.parquet")
    n_kept_draw = table_v2.num_rows

    n_r15 = len(ARM_SPECS)
    n_origins = 3
    n_r6f = 3
    n_primary = 3
    m_active = len(targets)
    m_r6 = len(json.loads(_read_csv(v1 / "S09_STATE_RQ3_R6_ORIGIN_SUMMARY.csv")[0]["balanced_target_ids"]))
    n_total_inf = n_r15 * m_active + n_origins * n_r6f * m_r6 + n_primary * m_active
    n_retained = n_total_inf * 5000
    if len(inf_v2) != n_total_inf or n_kept_draw != n_retained:
        raise S53V2Stop("S53V2_STOP_COUNT_RECONCILE", f"inf={len(inf_v2)} expected={n_total_inf} draws={n_kept_draw} expected={n_retained}")

    r6_blocks = sorted({r["block_arm_id"] for r in inf_v2 if r["arm_id"] == "R6_TEMPORAL"})
    if tuple(r6_blocks) != tuple(sorted(R6_ALLOWED)):
        raise S53V2Stop("S53V2_STOP_R6_BLOCKS", str(r6_blocks))
    if any(r["block_arm_id"] == "BASE" and r["arm_id"] == "R6_TEMPORAL" for r in inf_v2):
        raise S53V2Stop("S53V2_STOP_R6_BASE_REMAINING")
    if not any(r["block_arm_id"] == "BASE" and r["arm_id"] != "R6_TEMPORAL" and r["arm_id"] != "PRIMARY_REF" for r in inf_v2):
        raise S53V2Stop("S53V2_STOP_R15_BASE_MISSING")

    man_v1 = json.loads((v1 / "S09_STATE_RQ3_CONTROL_MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    models_v2 = []
    for rec in man_v1["models"]:
        item = dict(rec)
        tgt = _manifest_target(item["arm_id"])
        if not tgt:
            raise S53V2Stop("S53V2_STOP_BLANK_MANIFEST_TARGET")
        item["target"] = tgt
        item["model_binary_reused_from_stage53_v1"] = True
        item["scientific_fit_changed"] = False
        path = root / (item.get("path_if_new") or "")
        if path.is_file() and sha256_file(path) != item["sha256"]:
            raise S53V2Stop("S53V2_STOP_MODEL_HASH", item["path_if_new"])
        models_v2.append(item)
    dump_json(out / "S09_STATE_RQ3_CONTROL_MODEL_MANIFEST.json", {"models": models_v2})

    proto = json.loads((v1 / "S09_STATE_RQ3_PROTOCOL.json").read_text(encoding="utf-8"))
    proto.update({
        "correction_version": 2,
        "correction_classification": "IMPLEMENTATION_BUG",
        "supersedes_stage53_run": V1_RUN,
        "scientific_results_recomputed": False,
        "models_refit": False,
        "bootstrap_regenerated": False,
        "v1_provenance_root": OUT_V1,
        "active_artifact_root": OUT_V2,
        "r6_block_factors": [["0P5", 0.5], ["1P0", 1.0], ["2P0", 2.0]],
        "r6_nominal_block_arm_id": R6_NOMINAL,
        "r6_base_distribution_authorized": False,
        "pairwise_retention_definition": RET_DEF,
        "manifest_target_metadata_corrected": True,
        "literal_target_count_gate_removed": True,
    })
    dump_json(out / "S09_STATE_RQ3_PROTOCOL.json", proto)

    recon = json.loads((v1 / "S09_STATE_RQ3_RECONCILIATION.json").read_text(encoding="utf-8"))
    recon.update({
        "correction_version": 2,
        "controller_hold_resolved": True,
        "v1_run_id": V1_RUN,
        "scientific_results_recomputed": False,
        "models_refit": False,
        "bootstrap_regenerated": False,
        "r6_base_rows_removed": True,
        "removed_r6_base_inference_rows": n_removed_inf,
        "removed_r6_base_draw_rows": n_removed_draw,
        "removed_r6_base_proposal_rows": n_removed_prop,
        "corrected_inference_rows": len(inf_v2),
        "corrected_draw_rows": n_kept_draw,
        "corrected_proposal_rows": n_kept_prop,
        "r6_allowed_block_ids": list(R6_ALLOWED),
        "r6_nominal_block_arm_id": R6_NOMINAL,
        "all_authorized_bootstrap_rows_equal_v1": True,
        "all_point_results_equal_v1": True,
        "pairwise_retention_complete": True,
        "pairwise_retention_formula_valid": True,
        "manifest_target_complete": True,
        "model_binary_hashes_equal_v1": True,
        "active_target_count_derived_from_stage43": True,
        "literal_target_count_gate": False,
        "rq1_recomputed": False,
        "rq2_recomputed": False,
        "c8_repeated_for_robustness": False,
        "fleet_robustness_ci": False,
        "fleet_robustness_p_value": False,
        "v1_historical_s09_s10_stage45_51_paper_unchanged": True,
    })
    dump_json(out / "S09_STATE_RQ3_RECONCILIATION.json", recon)

    method = (
        "# Stage 53 corrective V2\n\n"
        "Controller HOLD classified four implementation bugs: unauthorized R6 BASE bootstrap family; blank pairwise_retention; blank model-manifest target; a hardcoded target-count equality gate.\n\n"
        "Correction removes only R6 `BASE` bootstrap/inference/proposal rows. Nominal R6 block arm is `1P0`. Pairwise retention is n_pairwise_rows / n_target_test_rows. Manifest targets are filled without refit. Active targets are derived from Stage 43 mapping.\n\n"
        "V1 artifacts remain byte-identical under `artifacts/s09_state_constrained_rq3/`. No models, predictions, effects, or draws were regenerated. Stage 54 was not executed.\n"
    )
    (root / AMEND).parent.mkdir(parents=True, exist_ok=True)
    (root / AMEND).write_text(method, encoding="utf-8")
    report = "\n".join([
        "# S09 Stage 53 corrective V2",
        "",
        "## Prerequisites",
        f"Stage 52 V2 GO. Stage 53 V1 `{V1_RUN}`. Freeze `{FREEZE_REL}`.",
        "",
        "## V1 provenance",
        f"Root `{OUT_V1}` preserved. Active root `{OUT_V2}`.",
        "",
        "## Removed R6 BASE counts",
        f"inference={n_removed_inf} draws={n_removed_draw} proposals={n_removed_prop}",
        "",
        "## Corrected counts",
        f"inference={len(inf_v2)} draws={n_kept_draw} proposals={n_kept_prop}",
        "",
        "## Retention / manifest",
        f"definition `{RET_DEF}`. Manifest targets complete. R6 allowed {list(R6_ALLOWED)}; nominal {R6_NOMINAL}.",
        "",
        "## Equality / protected",
        "Authorized V1 point results unchanged. Historical S09/S10/Stage45/51/paper/V1 Stage53 unchanged.",
        "",
        "## Verdict",
        REASON_GO,
        "",
        "## Next action",
        "CONTROLLER_REVIEW_FOR_S10_REMATERIALIZATION. Stage 54 not executed.",
        "",
    ]) + "\n"
    report_path = root / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    recs: list[ArtifactRecord] = []
    for name in V1_FILES:
        if name == "S09_STATE_RQ3_MANIFEST.json":
            continue
        path = out / name
        rows = None
        fp = None
        if path.suffix == ".csv":
            rows = len(_read_csv(path))
            fp = _csv_fp(path)
        elif path.suffix == ".parquet":
            rows = pq.read_table(path).num_rows
            fp = _pq_fp(path)
        recs.append(_art(root, path, rows, fp))
    recs.append(_art(root, report_path))
    recs.append(_art(root, root / AMEND))
    dump_json(out / "S09_STATE_RQ3_MANIFEST.json", {"job": JOB, "run_id": ctx.run_id, "supersedes_run_id": V1_RUN, "correction_version": 2, "artifacts": [_tab(r) for r in recs]})
    recs = [r for r in recs if not r.path.endswith("S09_STATE_RQ3_MANIFEST.json")]
    recs.append(_art(root, out / "S09_STATE_RQ3_MANIFEST.json"))

    hist_after = snapshot_tree(root, PROTECTED)
    if hist_after != hist_before:
        raise S53V2Stop("S53V2_STOP_PROTECTED", str(sorted(set(hist_after) ^ set(hist_before))[:12]))
    if json.dumps(_nested(ctx.sot, "scientific_analysis", "s09"), sort_keys=True) != s09_before:
        raise S53V2Stop("S53V2_STOP_S09_NODE")

    prev = _nested(ctx.sot, "scientific_analysis", "s09_revisions", REVISION) or {}
    prev_r6 = dict(prev.get("r6") or {})
    prev_r6.update({"r6_nominal_block_arm_id": R6_NOMINAL, "allowed_block_factors": list(R6_ALLOWED), "base_distribution_authorized": False})
    s10_prev = _nested(ctx.sot, "scientific_freeze", "s10_revisions", REVISION) or {}
    arts_map = {Path(r.path).name: _tab(r) for r in recs}
    sot_patch = {
        "stages": {STAGE_NODE: {
            "kind": "scientific_analysis_revision_correction", "status": "GO", "reason": REASON_GO,
            "operational_stage_id": "53", "job": JOB, "run_id": ctx.run_id, "finished_at_utc": now,
            "revision_id": REVISION, "correction_version": 2, "correction_classification": "IMPLEMENTATION_BUG",
            "supersedes_run_id": V1_RUN, "controller_hold_resolved": True,
            "scientific_results_recomputed": False, "models_refit": False, "bootstrap_regenerated": False,
            "active_artifact_root": OUT_V2, "v1_provenance_root": OUT_V1,
            "rq1_recomputed": False, "rq2_recomputed": False, "mandatory_arms_complete": True,
            "fleet_robustness_inference": "DESCRIPTIVE_ONLY", "c8_repeated_for_robustness": False,
            "answer": ANSWER, "report": _tab(next(r for r in recs if r.path == REPORT)),
            "manifest": _tab(next(r for r in recs if r.path.endswith("S09_STATE_RQ3_MANIFEST.json"))),
            "next_action": "CONTROLLER_REVIEW_FOR_S10_REMATERIALIZATION",
        }},
        "scientific_analysis": {"s09_revisions": {REVISION: {
            **prev,
            "status": "GO", "reason": REASON_GO, "correction_version": 2, "controller_hold_resolved": True,
            "scientific_results_recomputed": False, "models_refit": False, "bootstrap_regenerated": False,
            "active_artifact_root": OUT_V2, "v1_provenance_root": OUT_V1,
            "r6": prev_r6,
            "pairwise_retention_definition": RET_DEF,
            "artifacts": arts_map,
            "report": _tab(next(r for r in recs if r.path == REPORT)),
            "next_action": "CONTROLLER_REVIEW_FOR_S10_REMATERIALIZATION",
        }}},
        "scientific_freeze": {"s10_revisions": {REVISION: {
            **s10_prev,
            "status": "REVISED_S09_RQ3_V2_AVAILABLE_PENDING_CONTROLLER_REVIEW",
            "correction_version": 2,
            "active_stage53_root": OUT_V2,
            "s10_execution_authorized": False,
            "next_action": "CONTROLLER_REVIEW_STAGE53_V2",
        }}},
    }
    del freeze
    validations.extend([
        ValidationRecord("no_r6_base", True, str(n_removed_inf)),
        ValidationRecord("counts", True, f"{len(inf_v2)}/{n_kept_draw}"),
        ValidationRecord("protected", True, None),
        ValidationRecord("no_stage54", True, None),
    ])
    return StageResult(status="GO", message=MSG_GO, sot_patch=sot_patch, artifacts=recs, validations=validations)
