"""30 — Presentation-only polish for manuscript Figures 2 and 3."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, dump_json, sha256_file
from src.lib.figure_polish import render_consistent_na_map, render_relative_and_placebo_ecdf
from src.run._ledger import ledger_patch

STATUS = "FIGURE_LABEL_AND_PLACEBO_ECDF_POLISH_MATERIALIZED"
SOURCE_RELS = (
    "artifacts/s07/S07_OTG_DIRECTIONAL.csv",
    "artifacts/s04/S04_GSELF_SUMMARY.csv",
    "artifacts/s12/p1/S12_P1_OTG_HALF_DIRECTIONAL.csv",
    "artifacts/s12/p1/S12_P1_PLACEBO_PLANT_SUMMARY.csv",
)


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    root = ctx.root
    checks: list[ValidationRecord] = []

    def add(name: str, passed: bool, details: str | None = None) -> None:
        checks.append(ValidationRecord(name, bool(passed), details))

    missing = [rel for rel in SOURCE_RELS if not (root / rel).is_file()]
    if missing:
        return StageResult(
            status="STOP",
            message="STOP_SOURCE_VALUE_NOT_FOUND",
            sot_patch={"results": {"figure_polish": {"status": "STOP", "missing": missing}}},
            validations=[ValidationRecord("sources_present", False, str(missing))],
        )

    source_before = {rel: sha256_file(root / rel) for rel in SOURCE_RELS}
    tex = root / "paper/main.tex"
    tex_before = _sha(tex)
    ecdf_path = root / "paper/figure_01_relative_otg_ecdf.pdf"
    map_path = root / "paper/figure_02_operational_twin_map.pdf"

    ecdf_meta = render_relative_and_placebo_ecdf(root, ecdf_path)
    map_meta = render_consistent_na_map(root, map_path)

    out = root / "artifacts/figure_polish"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": STATUS,
        "tier": 3,
        "generated_at_utc": now,
        "changes": {
            "manuscript_figure_2": "added a separate matched-scale ECDF panel overlaying half-history twin transfer and two within-plant placebo distributions",
            "manuscript_figure_3": "standardized diagonal and non-computable cell labels to NA",
        },
        "ecdf": ecdf_meta,
        "operational_map": map_meta,
        "source_artifacts": {rel: source_before[rel] for rel in SOURCE_RELS},
        "no_refit": True,
        "no_rescoring": True,
        "no_new_estimator": True,
        "no_scientific_artifact_rewrite": True,
    }
    dump_json(out / "FIGURE_POLISH_SUMMARY.json", summary)

    recs = [
        artifact(root, ecdf_path),
        artifact(root, map_path),
        artifact(root, out / "FIGURE_POLISH_SUMMARY.json"),
    ]
    add("source_artifacts_unchanged", all(sha256_file(root / rel) == digest for rel, digest in source_before.items()), "four frozen sources")
    add("paper_tex_unchanged_by_stage", tex_before == _sha(tex), tex_before)
    add("ecdf_denominators", ecdf_meta["primary_relative_n"] == 85 and ecdf_meta["half_history_n"] == 85 and ecdf_meta["chrono_placebo_n"] == 13 and ecdf_meta["interleaved_placebo_n"] == 13, str(ecdf_meta))
    add("matched_scale_only", ecdf_meta["primary_not_compared_directly_with_placebo"], "placebo panel uses absolute OTG throughout")
    add("na_consistent", map_meta["missing_label"] == map_meta["diagonal_label"] == "NA", "NA")
    add("pdf_outputs_nonempty", ecdf_path.stat().st_size > 1000 and map_path.stat().st_size > 1000, f"{ecdf_path.stat().st_size},{map_path.stat().st_size}")

    payload = {
        "status": STATUS,
        "tier": 3,
        "ecdf_path": "paper/figure_01_relative_otg_ecdf.pdf",
        "ecdf_sha256": recs[0].sha256,
        "map_path": "paper/figure_02_operational_twin_map.pdf",
        "map_sha256": recs[1].sha256,
        "summary_path": "artifacts/figure_polish/FIGURE_POLISH_SUMMARY.json",
        "summary_sha256": recs[2].sha256,
        "no_refit": True,
        "no_rescoring": True,
        "no_new_estimator": True,
    }
    patch = ledger_patch(
        stage="figure_polish",
        now=now,
        payload=payload,
        recs=recs,
        extra={
            "figures": {
                "figure_01_relative_otg_ecdf": {
                    "path": payload["ecdf_path"],
                    "sha256": payload["ecdf_sha256"],
                    "source_artifact_paths": [SOURCE_RELS[0], SOURCE_RELS[2], SOURCE_RELS[3]],
                    "status": "MATERIALIZED",
                },
                "figure_02_operational_twin_map": {
                    "path": payload["map_path"],
                    "sha256": payload["map_sha256"],
                    "source_artifact_paths": [SOURCE_RELS[0], SOURCE_RELS[1]],
                    "missing_label": "NA",
                    "status": "MATERIALIZED",
                },
            }
        },
    )
    failed = [record for record in checks if not record.passed]
    if failed:
        return StageResult(
            status="STOP",
            message="STOP_RECONCILIATION_FAILURE",
            sot_patch={"results": {"figure_polish": {"status": "STOP", "failed": [record.name for record in failed]}}},
            artifacts=recs,
            validations=checks,
        )
    return StageResult(status="GO", message=STATUS, sot_patch=patch, artifacts=recs, validations=checks)
