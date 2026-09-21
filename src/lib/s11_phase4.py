"""S11 Phase 4: decision synthesis from accepted Phase 1–3 artifacts. No recompute."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from src.io import sha256_file

PHASE13_SOURCES = (
    "artifacts/s11/S11_A1_G3_PREFLIGHT.json",
    "artifacts/s11/S11_A1_G3_DIRECTIONAL.csv",
    "artifacts/s11/S11_A1_G3_RQ1_TARGET.csv",
    "artifacts/s11/S11_A1_G3_ASYMMETRY.csv",
    "artifacts/s11/S11_A1_G3_RQ3_ELIGIBILITY.csv",
    "artifacts/s11/S11_A1_G3_RQ3_COMPARISONS.csv",
    "artifacts/s11/S11_A1_G3_RQ3_TARGET_SUMMARY.csv",
    "artifacts/s11/S11_A1_G3_SUMMARY.json",
    "artifacts/s11/S11_PHASE3_A1_G3_PROVENANCE.json",
    "artifacts/s11/S11_A2_R8_ELIGIBILITY.csv",
    "artifacts/s11/S11_A2_R8_ELIGIBILITY_SUMMARY.json",
    "artifacts/s11/S11_A2_R8_COMPARISONS.csv",
    "artifacts/s11/S11_A2_R8_TARGET_SUMMARY.csv",
    "artifacts/s11/S11_A2_R8_SUMMARY.json",
    "artifacts/s11/S11_PHASE2_A2_PROVENANCE.json",
    "artifacts/s11/S11_A3_TARGET_SPREAD.csv",
    "artifacts/s11/S11_A3_SUMMARY.json",
    "artifacts/s11/S11_PHASE1_PROVENANCE.json",
    "artifacts/s11/S11_A4_RQ1_BY_GROUP.csv",
    "artifacts/s11/S11_A4_TARGET_MEANS.csv",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_inventory(root: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for rel in PHASE13_SOURCES:
        path = root / rel
        out[rel] = {
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    return out


def pdf_page_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        text = subprocess.check_output(["pdfinfo", str(path)], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in text.splitlines():
        if line.lower().startswith("pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def collect_headlines(root: Path, sot: dict[str, Any]) -> dict[str, Any]:
    g3 = _load_json(root / "artifacts/s11/S11_A1_G3_SUMMARY.json")
    a2 = _load_json(root / "artifacts/s11/S11_A2_R8_SUMMARY.json")
    a2e = _load_json(root / "artifacts/s11/S11_A2_R8_ELIGIBILITY_SUMMARY.json")
    a3 = _load_json(root / "artifacts/s11/S11_A3_SUMMARY.json")
    a4 = _load_csv(root / "artifacts/s11/S11_A4_RQ1_BY_GROUP.csv")
    res = sot.get("results") or {}
    primary = (res.get("tables") or {})
    rq3 = res.get("rq3") or {}
    a4_map = {r["twin_group"]: float(r["plant_balanced_rq1"]) for r in a4}
    return {
        "g3": {
            "group_id": g3["group_id"],
            "plants": g3["plants"],
            "n_plants": g3["n_plants"],
            "features": g3["features"],
            "n_computable_directed_transfers": g3["n_computable_directed_transfers"],
            "n_expected_directed_transfers": g3["n_expected_directed_transfers"],
            "n_bidirectionally_computable_unordered_pairs": g3["n_bidirectionally_computable_unordered_pairs"],
            "rq1": g3["rq1_plant_balanced_mean_otg"],
            "rq2": g3["rq2_plant_balanced_mean_abs_asymmetry"],
            "rq3": g3["rq3_target_balanced_control_minus_twin"],
            "rq3_n_executable_target_control_pairs": g3["rq3_n_executable_target_control_pairs"],
            "rq3_n_valid_source_control_comparisons": g3["rq3_n_valid_source_control_comparisons"],
            "rq3_n_distinct_contributing_controls": g3["rq3_n_distinct_contributing_controls"],
        },
        "a2": {
            "n_primary_rq3_targets": a2e["n_primary_rq3_targets"],
            "per_target_control_counts": a2e["per_target_control_counts"],
            "median_eligible_controls_per_target": a2e["median_eligible_controls_per_target"],
            "distinct_eligible_control_count": a2e["distinct_eligible_control_count"],
            "n_valid_comparisons": a2["n_valid_comparisons"],
            "target_balanced_r8": a2["target_balanced_r8_control_minus_twin"],
            "orientation": a2["orientation"],
        },
        "a3": {
            "eligible_target_count": a3["eligible_target_count"],
            "n_eligible_by_twin_group": a3["n_eligible_by_twin_group"],
            "median_spread": a3["median_spread"],
            "max_spread": a3["max_spread"],
            "max_target": a3["max_target"],
        },
        "a4": {"by_group": a4_map, "rows": a4},
        "primary": {
            "rq1": primary.get("rq1"),
            "rq2": primary.get("rq2"),
            "rq3": primary.get("rq3"),
            "D_fleet_RQ3": rq3.get("D_fleet_RQ3"),
            "selected_controls": rq3.get("selected_controls"),
        },
    }


def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return a == b


def inspect_manuscript(root: Path) -> dict[str, Any]:
    tex = (root / "paper/main.tex").read_text(encoding="utf-8")
    pdf = root / "paper/main.pdf"
    return {
        "tex_sha256": sha256_file(root / "paper/main.tex"),
        "pdf_sha256": sha256_file(pdf) if pdf.is_file() else None,
        "pdf_pages": pdf_page_count(pdf),
        "mentions_g3_excluded": "Excluded (reduced features)" in tex,
        "has_g2_heatmap": "figure_02_operational_twin_map" in tex,
        "has_rq4_source_selection": "sec:source-selection" in tex,
        "has_table_ii": "tab:primary-robustness" in tex,
        "rq3_three_controls_limitation": "only three distinct controls" in tex,
        "fig1_concept": "figure_01_experimental_concept.pdf" in tex,
        "gself_paragraph": "G^{self}" in tex or r"G^{self}" in tex,
    }


def build_decisions(head: dict[str, Any], ms: dict[str, Any]) -> dict[str, Any]:
    """Editorial inclusion of A1–A4 in the main text; page optimization is deferred."""
    pages = ms.get("pdf_pages")
    a3_only_g2 = set((head["a3"]["n_eligible_by_twin_group"] or {}).keys()) == {"G2"}
    a4_has_negative = any(float(v) < 0 for v in head["a4"]["by_group"].values())
    a2_r8 = float(head["a2"]["target_balanced_r8"])
    primary_rq3 = float(head["primary"]["D_fleet_RQ3"])
    items = {
        "A1": {
            "decision": "INCLUDE_CORE",
            "scientific_rationale": "Table I already flags G3 as excluded for reduced features; a short sensitivity result answers whether residual OTG is an artifact of the six-feature primary cohort.",
            "claim_allowed": "Under POA+GHI only, the eight-plant G3 cohort still exhibits residual plant-balanced OTG, directional asymmetry, and a positive same-state control-minus-twin contrast; G3 is not pooled with primary RQ1–RQ3.",
            "claim_prohibited": "Do not treat G3 headlines as interchangeable with primary estimates or as a six-feature replication.",
            "estimated_footprint": "Methods 1 sentence + Results 1 short paragraph (~8–12 lines). No new figure.",
            "insertion_location": "Methods surrogate paragraph (feature-set note) and a new Results sensitivity paragraph after RQ2; Table I status can stay 'sensitivity, reduced features'.",
            "selection_basis": "reviewer-likely gap in Table I, not proximity of G3 RQ1 to primary RQ1",
        },
        "A2": {
            "decision": "INCLUDE_CORE",
            "scientific_rationale": "The manuscript already discloses that revised RQ3 uses only three distinct controls; R8 tests whether the twin-vs-control signal survives the full structurally executable same-state control set.",
            "claim_allowed": "Broadening from one matched control per target to all structurally executable same-state non-twins (median 2 per target; 12 distinct controls; 118 comparisons) yields a target-balanced CONTROL_MINUS_TWIN contrast of the same sign as primary RQ3.",
            "claim_prohibited": "Do not present R8 as a replacement primary estimand, a p-value, or evidence that more controls make twins universally better.",
            "estimated_footprint": "Results 1–2 sentences and optional Table II row (~4–6 lines). No new figure.",
            "insertion_location": "RQ3 results after the three-control sentence; optional robustness table row labeled R8 descriptive.",
            "selection_basis": "addresses an existing limitation, not that R8 is larger than primary RQ3",
            "r8_vs_primary_note": {
                "r8": a2_r8,
                "primary_rq3": primary_rq3,
                "r8_larger": a2_r8 > primary_rq3,
                "used_for_selection": False,
            },
        },
        "A3": {
            "decision": "INCLUDE_CORE",
            "scientific_rationale": "User editorial decision after the Phase 4 synthesis: place the G2 source-choice spread statistic in the main text next to Fig. 2, with the G2-only scope preserved.",
            "claim_allowed": "Within G2, source choice was consequential: among its ten targets with at least two computable twin sources, the median gap between the best and worst source was 0.1057 normalized MAE (maximum 0.1315).",
            "claim_prohibited": "Do not generalize source-choice spread to G1/G4 or to the secondary RQ4 deployment analysis.",
            "estimated_footprint": "One Results sentence (~2–3 lines) near the G2 operational-twin map. No separate figure.",
            "insertion_location": "RQ1 paragraph on the G2 map (Fig. 2).",
            "selection_basis": "user editorial decision after scientific synthesis; not result sign/magnitude and not page budget",
            "scope_g2_only": a3_only_g2,
        },
        "A4": {
            "decision": "INCLUDE_CORE",
            "scientific_rationale": "User editorial decision after the Phase 4 synthesis: state between-group plant-balanced OTG heterogeneity in the main RQ1 results.",
            "claim_allowed": "Group-level plant-balanced OTG varied from −0.0089 (G4) to 0.0152 (G2), with G1 near zero (0.0011).",
            "claim_prohibited": "Do not read A4 as within-group source-choice variation or as evidence that G4 twins are interchangeable.",
            "estimated_footprint": "One Results sentence (~2 lines) adjacent to the RQ1 headline or G2-map discussion. No separate figure.",
            "insertion_location": "RQ1 after the plant-balanced headline, or adjacent to the G2-map discussion.",
            "selection_basis": "user editorial decision after scientific synthesis; includes the negative G4 mean rather than selecting on sign",
            "includes_negative_group_mean": a4_has_negative,
        },
    }
    core = ["A1", "A2", "A3", "A4"]
    if_space: list[str] = []
    omit_or_supp: list[str] = []
    space_plan = {
        "current_pdf_pages": pages,
        "indin_limit": 6,
        "already_over_limit": pages is not None and pages >= 6,
        "sequencing": "insert_all_first_then_assess_page_count",
        "page_count_does_not_constrain_inclusion": True,
        "deferred_until_after_insertion": [
            "compress G2 heatmap / Fig. 2",
            "condense Gself paragraph",
            "remove Fig. 1",
            "omit any of A1–A4",
        ],
        "steps": [
            {
                "order": 1,
                "action": "Insert A1, A2, A3, and A4 into the main manuscript (next phase). Do not compress or omit in Phase 4.",
                "expected_recovery": "not applicable; insertion precedes assessment",
            },
            {
                "order": 2,
                "action": "Compile and measure page count against the INDIN 6-page target.",
                "expected_recovery": "assessment only",
            },
            {
                "order": 3,
                "action": "Only after the expanded compile, consider heatmap compression, Gself condensation, or Fig. 1 removal.",
                "expected_recovery": "deferred",
            },
        ],
        "fit_assessment": "deferred until after insertion and compile",
    }
    payload = {
        "A1": {
            "methods": True,
            "results": True,
            "table_ii_row": False,
            "abstract_clause": False,
            "discussion": True,
            "limitation": True,
            "conclusion": False,
        },
        "A2": {
            "methods": True,
            "results": True,
            "table_ii_row": True,
            "abstract_clause": False,
            "discussion": True,
            "limitation": True,
            "conclusion": False,
        },
        "A4": {
            "methods": False,
            "results": True,
            "table_ii_row": False,
            "abstract_clause": False,
            "discussion": False,
            "limitation": False,
            "conclusion": False,
        },
        "A3": {
            "methods": False,
            "results": True,
            "table_ii_row": False,
            "abstract_clause": False,
            "discussion": False,
            "limitation": True,
            "conclusion": False,
            "supplement": False,
        },
        "do_not_change": ["title", "abstract_core_RQ1_RQ2_RQ3", "original_contributions_list"],
        "candidate_wording": {
            "A3": "Within G2, source choice was consequential: among its ten targets with at least two computable twin sources, the median gap between the best and worst source was 0.1057 normalized MAE (maximum 0.1315).",
            "A4": "Group-level plant-balanced OTG varied from −0.0089 (G4) to 0.0152 (G2), with G1 near zero (0.0011).",
        },
    }
    synthesis = {
        "beyond_paper_a": {
            "A1_generality": "Same qualitative OTG/asymmetry/control-minus-twin pattern appears in a reduced-feature G3 cohort that the paper currently only excludes.",
            "A2_control_robustness": "Twin-vs-control is not an artifact of one matched control; executable-control expansion keeps a positive target-balanced contrast.",
            "A3_source_identity": "Inside G2, best-vs-worst twin source OTG spread is large; this is already qualitatively visible in Fig. 2.",
            "A4_group_heterogeneity": "Primary group means span negative to positive OTG, so the plant-balanced headline is not a homogeneous twin-group property.",
        },
        "would_materially_change": {
            "abstract": False,
            "conclusion": False,
            "limitations": True,
            "table_ii": "optional R8 descriptive row only",
            "figure_strategy": "no new figures; do not compress or remove figures in Phase 4; page optimization deferred until after insertion",
        },
    }
    return {
        "items": items,
        "recommended_inclusion_set": core,
        "recommended_if_space_set": if_space,
        "recommended_supplement_or_omit_set": omit_or_supp,
        "selection_rule": "user editorial decision after scientific synthesis; estimand sign is not a selection input; page count does not constrain inclusion",
        "space_plan": space_plan,
        "phase6_payload": payload,
        "synthesis": synthesis,
    }


def render_markdown(head: dict[str, Any], ms: dict[str, Any], decisions: dict[str, Any]) -> str:
    items = decisions["items"]
    lines = [
        "# S11 Phase 4 inclusion decision",
        "",
        "Decision synthesis only. No estimand was recomputed. Manuscript files were not edited.",
        "",
        "All four additions (A1, A2, A3, A4) are selected for the **main manuscript** by user editorial decision after the scientific synthesis. Page-count optimization is deferred until after insertion.",
        "",
        f"Current compiled manuscript (context only, not a constraint): {ms.get('pdf_pages')} pages (INDIN limit 6).",
        "",
        "## Headlines used",
        "",
        f"- A1/G3 `{head['g3']['group_id']}` n={head['g3']['n_plants']} POA+GHI; directed {head['g3']['n_computable_directed_transfers']}/{head['g3']['n_expected_directed_transfers']}; bidir {head['g3']['n_bidirectionally_computable_unordered_pairs']}; RQ1 {head['g3']['rq1']}; RQ2 {head['g3']['rq2']}; RQ3 {head['g3']['rq3']}; executable pairs {head['g3']['rq3_n_executable_target_control_pairs']}; comparisons {head['g3']['rq3_n_valid_source_control_comparisons']}; distinct controls {head['g3']['rq3_n_distinct_contributing_controls']}.",
        f"- A2/R8 targets {head['a2']['n_primary_rq3_targets']}; median executable controls {head['a2']['median_eligible_controls_per_target']}; distinct {head['a2']['distinct_eligible_control_count']}; comparisons {head['a2']['n_valid_comparisons']}; R8 {head['a2']['target_balanced_r8']}; primary D_fleet {head['primary']['D_fleet_RQ3']}.",
        f"- A3 G2-only n={head['a3']['eligible_target_count']}; median spread {head['a3']['median_spread']}; max {head['a3']['max_spread']} at {head['a3']['max_target']}.",
        f"- A4 group RQ1 {head['a4']['by_group']}; primary RQ1 {head['primary']['rq1']}.",
        "",
        "## Decisions",
        "",
    ]
    for key in ("A1", "A2", "A3", "A4"):
        it = items[key]
        lines += [
            f"### {key}: `{it['decision']}`",
            "",
            f"- Rationale: {it['scientific_rationale']}",
            f"- Allowed: {it['claim_allowed']}",
            f"- Prohibited: {it['claim_prohibited']}",
            f"- Footprint: {it['estimated_footprint']}",
            f"- Insertion: {it['insertion_location']}",
            "",
        ]
    lines += [
        f"**Include in main:** {', '.join(decisions['recommended_inclusion_set']) or 'none'}.",
        f"**If space:** {', '.join(decisions['recommended_if_space_set']) or 'none'}.",
        f"**Supplement/omit:** {', '.join(decisions['recommended_supplement_or_omit_set']) or 'none'}.",
        "",
        "## What Phases 1–3 add",
        "",
        "- Generality (A1): residual OTG exists in a reduced-feature cohort; do not pool with primary.",
        "- Control robustness (A2): the twin-vs-control contrast is not an artifact of a single matched control.",
        "- Source identity (A3): G2 source choice can change OTG by ~0.11 median; already visible in Fig. 2.",
        "- Group heterogeneity (A4): primary group means are not homogeneous (G4 negative).",
        "",
        "## Phase 5 space plan",
        "",
    ]
    for step in decisions["space_plan"]["steps"]:
        lines.append(f"{step['order']}. {step['action']} ({step.get('expected_recovery', '')}).")
    lines += [
        "",
        "Sequencing: **insert all four first, then assess/optimize page count.** Do not compress figures or omit A3/A4 in this phase.",
        "",
        "## Phase 6 payload",
        "",
        "All four items go to the main text. A1: methods + results + discussion + limitation. A2: methods (minimal) + results + preferred Table II row + discussion/limitation. A3: one G2-only results sentence near Fig. 2. A4: one between-group results sentence at the RQ1 headline. No new figures. Abstract/conclusion clauses only if later integration shows they are non-redundant.",
        "",
    ]
    return "\n".join(lines) + "\n"
