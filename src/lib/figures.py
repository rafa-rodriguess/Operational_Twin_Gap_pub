"""Stage 38 helpers: materialize frozen S10 tables/figures. No new science."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.split_support import dump_json, sha256_file

JOB = "S10_FIGURES_TABLES_MATERIALIZATION"
MSG_GO = "GO — S10 FIGURES/TABLES COMPLETE"
MSG_STOP = "STOP — S10 MATERIALIZATION INCOMPLETE"
MSG_REDESIGN = "REDESIGN — S10 EXPOSES FATAL EVIDENCE/SPECIFICATION CONTRADICTION"
OUT = "artifacts/s10"
REPORT = "reports/scientific/S10_FIGURES_TABLES.md"
ANSWER = None
PAPER_FILES = (
    "table_01_nominal_twin_groups.csv",
    "table_02_primary_robustness_summary.csv",
    "figure_01_experimental_concept.pdf",
    "figure_02_operational_twin_map.pdf",
    "figure_03_twin_vs_matched_nontwin.pdf",
)
NEED_MAP = {
    "[NEED TABLE 01]": "[TABLE table_01_nominal_twin_groups.csv]",
    "[NEED TABLE 02]": "[TABLE table_02_primary_robustness_summary.csv]",
    "[NEED PLOT 01]": "[PLOT figure_01_experimental_concept.pdf]",
    "[NEED PLOT 02]": "[PLOT figure_02_operational_twin_map.pdf]",
    "[NEED PLOT 03]": "[PLOT figure_03_twin_vs_matched_nontwin.pdf]",
}
SOURCE_RELS = (
    "artifacts/s01/S01_TWIN_GROUPS_FROZEN.csv",
    "artifacts/s04/S04_GSELF_SUMMARY.csv",
    "artifacts/s07/S07_OTG_DIRECTIONAL.csv",
    "artifacts/s07/S07_ASYMMETRY.csv",
    "artifacts/s07/S07_SUMMARY.json",
    "artifacts/s08_scientific/S08_TARGET_LEVEL_MATCHED_CONTRASTS.csv",
    "artifacts/s08_scientific/S08_TARGET_LEVEL_INFERENCE.csv",
    "artifacts/s08_scientific/S08_FLEET_DESCRIPTIVE_SUMMARY.json",
    "artifacts/s09_scientific/S09_RQ1_SUMMARY.csv",
    "artifacts/s09_scientific/S09_RQ2_SUMMARY.csv",
    "artifacts/s09_scientific/S09_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv",
    "artifacts/s09_scientific/S09_R6_ORIGIN_SUMMARY.csv",
)
TABLE02_ROWS = (
    ("Primary", None, None),
    ("R1 HGB", "R1_HGB", ""),
    ("R2 AC", "R2_AC", ""),
    ("R3 POA20", "R3_POA20", ""),
    ("R3 POA100", "R3_POA100", ""),
    ("R4 CS3", "R4_CS3", ""),
    ("R5 Keep all valid", "R5_KEEP_ALL_VALID", ""),
    ("R6 O1", "R6_TEMPORAL", "O1"),
    ("R6 O2", "R6_TEMPORAL", "O2"),
    ("R6 O3", "R6_TEMPORAL", "O3"),
)
FIG2_GROUP = "TW_4467a039e00b"
FAM_SPLINE = "SplineRidge"
CS4 = "CS4"


class S10Stop(Exception):
    def __init__(self, reason: str, details: str = ""):
        super().__init__(details or reason)
        self.reason = reason
        self.details = details or reason


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _art(root: Path, path: Path, rows: int | None = None, fp: str | None = None) -> ArtifactRecord:
    payload = path.read_bytes()
    return ArtifactRecord(_rel(root, path), hashlib.sha256(payload).hexdigest(), len(payload), rows, fp)


def _tab(rec: ArtifactRecord) -> dict[str, Any]:
    out = {"path": rec.path, "sha256": rec.sha256, "bytes": rec.bytes}
    if rec.row_count is not None:
        out["row_count"] = rec.row_count
    if rec.schema_fingerprint is not None:
        out["schema_fingerprint"] = rec.schema_fingerprint
    return out


def _csv_fp(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    return hashlib.sha256("|".join(header).encode()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _flag(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _fmt4(value: float) -> str:
    return f"{float(value):.4f}"


def _walk_sot_files(node: Any, found: dict[str, dict[str, Any]]) -> None:
    if isinstance(node, dict):
        path = node.get("path")
        if isinstance(path, str) and path.startswith("artifacts/"):
            found.setdefault(path, node)
        for value in node.values():
            _walk_sot_files(value, found)
    elif isinstance(node, list):
        for value in node:
            _walk_sot_files(value, found)


def _nested(sot: dict[str, Any], *keys: str) -> Any:
    cur: Any = sot
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def plant_balanced_mean_otg(rows: list[dict[str, str]], primary_groups: set[str]) -> float:
    by_target: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("group_id") not in primary_groups:
            continue
        if row.get("model_family") != FAM_SPLINE or row.get("support_rule") != CS4:
            continue
        if not _flag(row.get("computable")):
            continue
        by_target[str(row["target_plant_id"])].append(float(row["otg_abs"]))
    if not by_target:
        raise S10Stop("S10_STOP_PRIMARY_RQ1_EMPTY", "no computable primary OTG directions")
    per_target = [sum(vals) / len(vals) for vals in by_target.values()]
    return float(sum(per_target) / len(per_target))


def plant_balanced_mean_abs_asymmetry(rows: list[dict[str, str]], primary_groups: set[str]) -> float:
    by_plant: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("group_id") not in primary_groups:
            continue
        if not _flag(row.get("asymmetry_computable")):
            continue
        value = float(row["asymmetry_abs"])
        by_plant[str(row["plant_a"])].append(value)
        by_plant[str(row["plant_b"])].append(value)
    if not by_plant:
        raise S10Stop("S10_STOP_PRIMARY_RQ2_EMPTY", "no bidirectionally computable primary pairs")
    per_plant = [sum(vals) / len(vals) for vals in by_plant.values()]
    return float(sum(per_plant) / len(per_plant))


def _lookup_s09(rows: list[dict[str, str]], arm_id: str, origin_id: str, field: str) -> float:
    matches = [
        r for r in rows
        if r.get("arm_id") == arm_id and str(r.get("origin_id") or "") == origin_id
    ]
    if len(matches) != 1:
        raise S10Stop("S10_STOP_S09_ROW", f"expected one {arm_id}/{origin_id} row, got {len(matches)}")
    return float(matches[0][field])


def _visual_grammar() -> dict[str, Any]:
    return {
        "id": "S10_VISUAL_GRAMMAR_V1",
        "typography": "compact_serif_ieee_two_column",
        "preferred_serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "target_text_pt_at_placement": [7, 8],
        "line_pt": [0.6, 1.0],
        "background": "#FFFFFF",
        "spine": "#333333",
        "grid": "#DDDDDD",
        "chartjunk": False,
        "internal_figure_title": False,
        "output": "vector_pdf",
        "embed_text": True,
        "colors": {
            "nominal_twin_source": "#00629B",
            "matched_nontwin_control": "#E69F00",
            "local_target": "#4D4D4D",
            "common_support": "#008C95",
            "self_transfer": "#777777",
            "diverging_neg": "#0072B2",
            "diverging_zero": "#FFFFFF",
            "diverging_pos": "#D55E00",
            "zero_reference": "#4D4D4D",
        },
        "significance_encoded_by_color": False,
        "legend_location": "below_outside_or_direct_labels",
    }


def _apply_rc(grammar: dict[str, Any]) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": grammar["preferred_serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "lines.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    })


def _assert_paper_dir(root: Path) -> Path:
    paper = (root / "paper").resolve()
    try:
        paper.relative_to(root.resolve())
    except ValueError as exc:
        raise S10Stop("S10_STOP_UNSAFE_PAPER_PATH", str(paper)) from exc
    if paper.name != "paper":
        raise S10Stop("S10_STOP_UNSAFE_PAPER_PATH", str(paper))
    return paper


def _wipe_paper(paper: Path) -> list[str]:
    removed: list[str] = []
    if paper.exists():
        for child in sorted(paper.iterdir(), key=lambda p: p.as_posix()):
            removed.append(child.relative_to(paper).as_posix() + ("/" if child.is_dir() else ""))
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    paper.mkdir(parents=True, exist_ok=True)
    return removed


def _write_csv_ordered(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def _parse_states(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise S10Stop("S10_STOP_TABLE01_STATES", "empty states")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [p.strip() for p in text.strip("[]").replace('"', "").split(",") if p.strip()]
    if not isinstance(parsed, list) or not parsed:
        raise S10Stop("S10_STOP_TABLE01_STATES", raw)
    return "/".join(str(s) for s in parsed)


def _status_for(group_id: str, primary: set[str], sensitivity: set[str], noneval: set[str]) -> str:
    hits = []
    if group_id in primary:
        hits.append("Primary")
    if group_id in sensitivity:
        hits.append("Sensitivity only")
    if group_id in noneval:
        hits.append("Non-evaluable")
    if len(hits) != 1:
        raise S10Stop("S10_STOP_TABLE01_STATUS", f"{group_id} mapped to {hits}")
    return hits[0]


def build_table_01(groups: list[dict[str, str]], sot: dict[str, Any]) -> list[dict[str, Any]]:
    primary = set(_nested(sot, "scientific_freeze", "s03", "corrective_reaudit", "control_matching", "primary_groups") or [])
    sensitivity = set(_nested(sot, "feasibility", "p02f", "sensitivity_a_extension", "group_ids") or [])
    noneval = set(_nested(sot, "feasibility", "p02f", "out_of_scope_non_evaluable_groups") or [])
    ordered = sorted(groups, key=lambda r: r["group_id"])
    if len(ordered) != 5:
        raise S10Stop("S10_STOP_TABLE01_COUNT", f"expected 5 groups, got {len(ordered)}")
    rows = []
    for rec in ordered:
        gid = rec["group_id"]
        rows.append({
            "Nominal twin group": gid,
            "Nominal power (MW)": rec["key_nominal_power_mw"],
            "Structure": rec["key_structure_type"],
            "Plants": int(rec["n_members"]),
            "States": _parse_states(rec["states"]),
            "Analytical status": _status_for(gid, primary, sensitivity, noneval),
        })
    return rows


def build_table_02(root: Path, sot: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_groups = set(_nested(sot, "scientific_freeze", "s03", "corrective_reaudit", "control_matching", "primary_groups") or [])
    otg_rows = _read_csv(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv")
    asy_rows = _read_csv(root / "artifacts/s07/S07_ASYMMETRY.csv")
    rq1_s09 = _read_csv(root / "artifacts/s09_scientific/S09_RQ1_SUMMARY.csv")
    rq2_s09 = _read_csv(root / "artifacts/s09_scientific/S09_RQ2_SUMMARY.csv")
    rq3_s09 = _read_csv(root / "artifacts/s09_scientific/S09_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv")
    r6 = _read_csv(root / "artifacts/s09_scientific/S09_R6_ORIGIN_SUMMARY.csv")
    fleet = json.loads((root / "artifacts/s08_scientific/S08_FLEET_DESCRIPTIVE_SUMMARY.json").read_text(encoding="utf-8"))
    primary_rq1 = plant_balanced_mean_otg(otg_rows, primary_groups)
    primary_rq2 = plant_balanced_mean_abs_asymmetry(asy_rows, primary_groups)
    primary_rq3 = float(fleet["D_fleet_RQ3"])
    sot_rq3 = _nested(sot, "scientific_analysis", "s08", "D_fleet_RQ3")
    if sot_rq3 is not None and not math.isclose(primary_rq3, float(sot_rq3), rel_tol=0, abs_tol=1e-18):
        raise S10Stop("S10_STOP_PRIMARY_RQ3", "S08 fleet scalar does not match SoT")
    values: dict[str, tuple[float, float, float]] = {"Primary": (primary_rq1, primary_rq2, primary_rq3)}
    for label, arm, origin in TABLE02_ROWS[1:7]:
        values[label] = (
            _lookup_s09(rq1_s09, arm, origin, "plant_balanced_mean_OTG"),
            _lookup_s09(rq2_s09, arm, origin, "plant_balanced_mean_abs_asymmetry"),
            _lookup_s09(rq3_s09, arm, origin, "D_fleet_RQ3"),
        )
    r6_by = {r["arm_id"]: r for r in r6}
    for label, _, origin in TABLE02_ROWS[7:]:
        rec = r6_by.get(f"R6_TEMPORAL__{origin}")
        if rec is None:
            raise S10Stop("S10_STOP_R6_ORIGIN", origin)
        values[label] = (
            float(rec["rq1_plant_balanced_mean_otg"]),
            float(rec["rq2_plant_balanced_mean_abs_asymmetry"]),
            float(rec["rq3_descriptive_target_balanced_control_minus_twin"]),
        )
    table = []
    for label, _, _ in TABLE02_ROWS:
        rq1, rq2, rq3 = values[label]
        table.append({
            "Analysis": label,
            "RQ1: plant-balanced OTG": _fmt4(rq1),
            "RQ2: plant-balanced |A|": _fmt4(rq2),
            "RQ3: control-twin": _fmt4(rq3),
        })
    recon = {
        "primary_rq1_exact": primary_rq1,
        "primary_rq2_exact": primary_rq2,
        "primary_rq3_exact": primary_rq3,
        "primary_rq1_aggregation": "plant_balanced_mean_OTG",
        "primary_rq2_aggregation": "plant_balanced_mean_absolute_asymmetry",
        "rq3_orientation": "CONTROL_MINUS_TWIN",
        "r6_pooled": False,
        "exact_values": {k: list(v) for k, v in values.items()},
    }
    return table, recon


def _draw_box(ax, xy, w, h, text, facecolor, edgecolor="#333333"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5, color="#111111", wrap=True)


def render_figure_01(path: Path, grammar: dict[str, Any]) -> dict[str, Any]:
    colors = grammar["colors"]
    fig, ax = plt.subplots(figsize=(7.16, 2.55), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("")
    boxes = [
        ((0.01, 0.38), 0.11, 0.28, "Nominal\nmetadata", "#E8F1F8", colors["nominal_twin_source"]),
        ((0.15, 0.38), 0.12, 0.28, "Exact pair\n(i, j)", "#E8F1F8", colors["nominal_twin_source"]),
        ((0.30, 0.62), 0.14, 0.26, "Surrogate f_i\n(source)", "#E8F1F8", colors["nominal_twin_source"]),
        ((0.30, 0.12), 0.14, 0.26, "Surrogate f_j\n(target)", "#EEEEEE", colors["local_target"]),
        ((0.48, 0.38), 0.14, 0.28, "Common\nenvironmental\nsupport", "#E6F4F5", colors["common_support"]),
        ((0.66, 0.38), 0.14, 0.28, "Same supported\ntarget rows\nD_test(j|i)", "#E6F4F5", colors["common_support"]),
        ((0.84, 0.62), 0.14, 0.26, "E_i→j\ntransferred", "#E8F1F8", colors["nominal_twin_source"]),
        ((0.84, 0.12), 0.14, 0.26, "E_j→j^(i)\nlocal same-row", "#EEEEEE", colors["local_target"]),
    ]
    for args in boxes:
        _draw_box(ax, *args[:3], args[3], args[4], args[5])
    arrows = [
        ((0.12, 0.52), (0.15, 0.52)),
        ((0.27, 0.52), (0.30, 0.75)),
        ((0.27, 0.52), (0.30, 0.25)),
        ((0.44, 0.75), (0.48, 0.58)),
        ((0.44, 0.25), (0.48, 0.46)),
        ((0.62, 0.52), (0.66, 0.52)),
        ((0.80, 0.58), (0.84, 0.75)),
        ((0.80, 0.46), (0.84, 0.25)),
    ]
    for p0, p1 in arrows:
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#333333"))
    ax.annotate("OTG_i→j", xy=(0.91, 0.50), xytext=(0.70, 0.04),
                fontsize=7, ha="center", color=colors["nominal_twin_source"],
                arrowprops=dict(arrowstyle="-", color=colors["nominal_twin_source"], lw=0.7))
    ax.text(0.91, 0.50, "−", ha="center", va="center", fontsize=9, color=colors["zero_reference"])
    ax.add_patch(FancyArrowPatch((0.91, 0.38), (0.91, 0.12), arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color=colors["zero_reference"]))
    ax.text(0.52, 0.90, "independently fit; no adaptation / recalibration / fine-tuning", ha="center", fontsize=6.5, color=colors["local_target"])
    ax.text(0.52, 0.02, "G_j^self = interpretive within-asset reference after OTG (not a threshold; not in the subtraction)",
            ha="center", fontsize=6.5, color=colors["self_transfer"])
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return {"width_in": 7.16, "height_in": 2.55, "placement": "two_column_for_legibility", "scientific_values": False}


def render_figure_02(path: Path, grammar: dict[str, Any], plants: list[str], matrix: list[list[float | None]],
                     diagonal: list[list[bool]], gself: list[float]) -> dict[str, Any]:
    colors = grammar["colors"]
    finite = [v for row in matrix for v in row if v is not None] + list(gself)
    if not finite:
        raise S10Stop("S10_STOP_FIG2_EMPTY", "no finite OTG/Gself values")
    m = max(abs(v) for v in finite)
    if m <= 0:
        raise S10Stop("S10_STOP_FIG2_SCALE", "non-positive scale extent")
    cmap = LinearSegmentedColormap.from_list(
        "otg_div", [colors["diverging_neg"], colors["diverging_zero"], colors["diverging_pos"]], N=256
    )
    norm = TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
    n = len(plants)
    fig = plt.figure(figsize=(7.16, 6.4), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.06], width_ratios=[1.0, 0.12], hspace=0.42, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    axg = fig.add_subplot(gs[0, 1], sharey=ax)
    cax = fig.add_subplot(gs[1, :])
    import numpy as np
    grid = np.full((n, n), np.nan, dtype=float)
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val is not None:
                grid[i, j] = val
    mesh = ax.pcolormesh(np.arange(n + 1) - 0.5, np.arange(n + 1) - 0.5, grid, cmap=cmap, norm=norm, shading="flat")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=5.5, color="#555555")
            elif matrix[i][j] is None:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=5.5, color="#555555")
            else:
                ax.text(j, i, f"{matrix[i][j]:.3f}", ha="center", va="center", fontsize=5.2, color="#111111")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(plants, rotation=90)
    ax.set_yticklabels(plants)
    ax.set_xlabel("Target plant")
    ax.set_ylabel("Source plant")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    axg.pcolormesh(np.array([0, 1]), np.arange(n + 1) - 0.5, np.array(gself).reshape(n, 1), cmap=cmap, norm=norm, shading="flat")
    for i, val in enumerate(gself):
        axg.text(0.5, i, f"{val:.3f}", ha="center", va="center", fontsize=5.2, color="#111111")
    axg.set_xticks([0.5])
    axg.set_xticklabels([r"$G^{self}$"], fontsize=7)
    axg.tick_params(axis="y", labelleft=False)
    axg.invert_yaxis()
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label("OTG / $G^{self}$ (normalized MAE)")
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return {"width_in": 7.16, "height_in": 6.4, "placement": "two_column", "M": m, "n_plants": n, "center": 0.0}


def render_figure_03(path: Path, grammar: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    colors = grammar["colors"]
    ordered = sorted(rows, key=lambda r: r["target_plant_id"])
    if len(ordered) != 14:
        raise S10Stop("S10_STOP_FIG3_COUNT", f"expected 14 targets, got {len(ordered)}")
    y = list(range(len(ordered)))
    x = [float(r["Delta_j"]) for r in ordered]
    lo = [float(r["ci_lower"]) for r in ordered]
    hi = [float(r["ci_upper"]) for r in ordered]
    labels = [r["target_plant_id"] for r in ordered]
    fig, ax = plt.subplots(figsize=(3.5, 5.6), dpi=300)
    ax.axvline(0.0, color=colors["zero_reference"], linewidth=0.8, zorder=0)
    ax.errorbar(x, y, xerr=[ [a - b for a, b in zip(x, lo)], [c - a for a, c in zip(x, hi)] ],
                fmt="o", color=colors["nominal_twin_source"], ecolor=colors["local_target"],
                elinewidth=0.8, markersize=3.5, capsize=1.5, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\Delta_j$ (control $-$ twin)")
    ax.set_ylabel("Target plant")
    ax.grid(axis="x", color=grammar["grid"], linewidth=0.4)
    ax.tick_params(length=2)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return {"width_in": 3.5, "height_in": 5.6, "placement": "one_column", "n_targets": 14, "order": labels}


def _fig2_payload(root: Path) -> tuple[list[str], list[list[float | None]], list[list[bool]], list[float], dict[str, Any]]:
    members = [r for r in _read_csv(root / "artifacts/s01/S01_TWIN_MEMBERS_FROZEN.csv") if r["group_id"] == FIG2_GROUP]
    plants = sorted({r["plant_id"] for r in members})
    if len(plants) != 10:
        raise S10Stop("S10_STOP_FIG2_PLANTS", f"expected 10 plants, got {plants}")
    otg = _read_csv(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv")
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in otg:
        if row.get("group_id") != FIG2_GROUP:
            continue
        if row.get("model_family") != FAM_SPLINE or row.get("support_rule") != CS4:
            continue
        lookup[(row["source_plant_id"], row["target_plant_id"])] = row
    gself_rows = {
        r["plant_id"]: float(r["gself_mean_signed"])
        for r in _read_csv(root / "artifacts/s04/S04_GSELF_SUMMARY.csv")
        if r.get("model_family") == FAM_SPLINE
    }
    matrix: list[list[float | None]] = []
    diag: list[list[bool]] = []
    n_comp = 0
    n_na = 0
    for src in plants:
        row_vals: list[float | None] = []
        row_diag: list[bool] = []
        for tgt in plants:
            if src == tgt:
                row_vals.append(None)
                row_diag.append(True)
                continue
            rec = lookup.get((src, tgt))
            if rec is None:
                raise S10Stop("S10_STOP_FIG2_MISSING_CELL", f"{src}->{tgt}")
            row_diag.append(False)
            if _flag(rec.get("computable")):
                row_vals.append(float(rec["otg_abs"]))
                n_comp += 1
            else:
                row_vals.append(None)
                n_na += 1
        matrix.append(row_vals)
        diag.append(row_diag)
    gself = []
    for pid in plants:
        if pid not in gself_rows:
            raise S10Stop("S10_STOP_FIG2_GSELF", pid)
        gself.append(gself_rows[pid])
    recon = {"group": FIG2_GROUP, "plants": plants, "computable_cells": n_comp, "na_offdiag": n_na, "diagonal": "NA"}
    return plants, matrix, diag, gself, recon


def _pdf_ok(path: Path) -> None:
    raw = path.read_bytes()
    if len(raw) < 8 or not raw.startswith(b"%PDF"):
        raise S10Stop("S10_STOP_PDF", path.as_posix())


def _update_paper_md(root: Path, hashes: dict[str, str]) -> tuple[str, int, int]:
    path = root / "paper.md"
    text = path.read_text(encoding="utf-8")
    ref_before = text.count("[NEED REFERENCE")
    for needle, token in NEED_MAP.items():
        if needle not in text:
            raise S10Stop("S10_STOP_MARKER_MISSING", needle)
        text = text.replace(needle, token)
    meta = {
        "[TABLE table_01_nominal_twin_groups.csv]": ("table_01_nominal_twin_groups.csv", hashes["paper/table_01_nominal_twin_groups.csv"]),
        "[TABLE table_02_primary_robustness_summary.csv]": ("table_02_primary_robustness_summary.csv", hashes["paper/table_02_primary_robustness_summary.csv"]),
        "[PLOT figure_01_experimental_concept.pdf]": ("figure_01_experimental_concept.pdf", hashes["paper/figure_01_experimental_concept.pdf"]),
        "[PLOT figure_02_operational_twin_map.pdf]": ("figure_02_operational_twin_map.pdf", hashes["paper/figure_02_operational_twin_map.pdf"]),
        "[PLOT figure_03_twin_vs_matched_nontwin.pdf]": ("figure_03_twin_vs_matched_nontwin.pdf", hashes["paper/figure_03_twin_vs_matched_nontwin.pdf"]),
    }
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    seen_heading = set()
    for line in lines:
        out.append(line)
        stripped = line.strip()
        if stripped.startswith("### ") and any(tok in stripped for tok in meta):
            for tok, (fname, digest) in meta.items():
                if tok in stripped and tok not in seen_heading:
                    seen_heading.add(tok)
                    out.append("\n")
                    out.append("- **Materialization status:** MATERIALIZED\n")
                    out.append(f"- **Materialized file:** paper/{fname}\n")
                    out.append(f"- **SHA-256:** `{digest}`\n")
                    break
    text = "".join(out)
    if "[NEED TABLE" in text or "[NEED PLOT" in text:
        raise S10Stop("S10_STOP_MARKERS_REMAIN", "unresolved NEED TABLE/PLOT")
    if text.count("[NEED REFERENCE") != ref_before:
        raise S10Stop("S10_STOP_REFERENCE_MARKERS", "NEED REFERENCE markers changed")
    path.write_text(text, encoding="utf-8")
    return sha256_file(path), text.count("[NEED TABLE"), text.count("[NEED PLOT")


def _validate_preconditions(ctx: StageContext, validations: list[ValidationRecord]) -> dict[str, str]:
    sot = ctx.sot
    if _nested(sot, "stages", "S09", "status") != "GO":
        raise S10Stop("S10_STOP_S09_NOT_GO", "stages.S09.status is not GO")
    if str(_nested(sot, "stages", "S09", "operational_stage_id")) != "37":
        raise S10Stop("S10_STOP_STAGE37", "S09 operational stage is not 37")
    paper_md = ctx.root / "paper.md"
    writing = ctx.root / "writing_method.md"
    if not paper_md.is_file():
        raise S10Stop("S10_STOP_PAPER_MD", "missing paper.md")
    if not writing.is_file():
        raise S10Stop("S10_STOP_WRITING_METHOD", "missing writing_method.md")
    existing = _nested(sot, "stages", "S10")
    if existing is not None and existing.get("status") == "GO":
        paper = ctx.root / "paper"
        present = sorted(p.name for p in paper.iterdir()) if paper.is_dir() else []
        raise S10Stop(
            "S10_STOP_PREEXISTING_TRACKED_ASSETS",
            "S10 already GO; paper/ already contains materialized assets "
            f"{present}; NEED TABLE/PLOT tokens are already replaced. "
            "IEEE rematerialization would silently delete tracked paper files.",
        )
    text = paper_md.read_text(encoding="utf-8")
    for marker in NEED_MAP:
        if text.count(marker) != 2:
            raise S10Stop("S10_STOP_MARKER_COUNT", f"{marker} count={text.count(marker)}")
    if "# INTERNAL — Evidence and Asset Specification Registry" not in text:
        raise S10Stop("S10_STOP_REGISTRY", "missing registry heading")
    existing = _nested(sot, "stages", "S10")
    if existing is not None and existing.get("status") == "GO":
        raise S10Stop("S10_STOP_ALREADY_EXISTS", "stages.S10 already exists")
    registered: dict[str, dict[str, Any]] = {}
    _walk_sot_files(sot, registered)
    snapshot: dict[str, str] = {}
    for rel in SOURCE_RELS:
        path = ctx.root / rel
        if not path.is_file():
            raise S10Stop("S10_STOP_MISSING_SOURCE", rel)
        digest = sha256_file(path)
        rec = registered.get(rel)
        if rec is None:
            raise S10Stop("S10_STOP_SOURCE_UNREGISTERED", rel)
        if rec.get("sha256") and rec["sha256"] != digest:
            raise S10Stop("S10_STOP_SOURCE_HASH", f"{rel} SoT {rec['sha256']} file {digest}")
        snapshot[rel] = digest
    validations.append(ValidationRecord("preconditions", True, "S09 GO and SoT-authorized sources"))
    return snapshot


def execute_s10(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    try:
        return _execute(ctx, now, validations)
    except S10Stop as exc:
        validations.append(ValidationRecord(exc.reason, False, exc.details))
        existing = _nested(ctx.sot, "stages", "S10") or {}
        stage_key = "S10_IEEE" if existing.get("status") == "GO" else "S10"
        return StageResult(
            status="STOP",
            message=MSG_STOP,
            sot_patch={
                "stages": {
                    stage_key: {
                        "kind": "scientific_output_materialization",
                        "status": "STOP",
                        "reason": exc.reason,
                        "operational_stage_id": "38",
                        "job": "S10_FIGURES_TABLES_MATERIALIZATION_IEEE",
                        "details": exc.details,
                        "finished_at_utc": now,
                        "prior_s10_status": existing.get("status"),
                    }
                }
            },
            artifacts=[],
            validations=validations,
        )


def _execute(ctx: StageContext, now: str, validations: list[ValidationRecord]) -> StageResult:
    root = ctx.root
    source_hashes = _validate_preconditions(ctx, validations)
    grammar = _visual_grammar()
    _apply_rc(grammar)
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    grammar_path = out / "S10_VISUAL_GRAMMAR.json"
    dump_json(grammar_path, grammar)
    paper = _assert_paper_dir(root)
    removed = _wipe_paper(paper)
    groups = _read_csv(root / "artifacts/s01/S01_TWIN_GROUPS_FROZEN.csv")
    t01 = build_table_01(groups, ctx.sot)
    t01_path = paper / PAPER_FILES[0]
    _write_csv_ordered(t01_path, t01, list(t01[0].keys()))
    t02, t02_recon = build_table_02(root, ctx.sot)
    t02_path = paper / PAPER_FILES[1]
    _write_csv_ordered(t02_path, t02, list(t02[0].keys()))
    fig1_path = paper / PAPER_FILES[2]
    fig1_meta = render_figure_01(fig1_path, grammar)
    plants, matrix, diag, gself, fig2_recon = _fig2_payload(root)
    fig2_path = paper / PAPER_FILES[3]
    fig2_meta = render_figure_02(fig2_path, grammar, plants, matrix, diag, gself)
    inf = _read_csv(root / "artifacts/s08_scientific/S08_TARGET_LEVEL_INFERENCE.csv")
    fig3_path = paper / PAPER_FILES[4]
    fig3_meta = render_figure_03(fig3_path, grammar, inf)
    for pdf in (fig1_path, fig2_path, fig3_path):
        _pdf_ok(pdf)
    leftover = sorted(p.name for p in paper.iterdir() if p.is_file())
    if leftover != sorted(PAPER_FILES):
        raise S10Stop("S10_STOP_PAPER_CONTENTS", str(leftover))
    if any(p.is_dir() for p in paper.iterdir()):
        raise S10Stop("S10_STOP_PAPER_SUBDIR", "subdirectory remains")
    hashes = {f"paper/{name}": sha256_file(paper / name) for name in PAPER_FILES}
    paper_md_sha, need_table, need_plot = _update_paper_md(root, hashes)
    recs = [
        _art(root, t01_path, rows=len(t01), fp=_csv_fp(t01_path)),
        _art(root, t02_path, rows=len(t02), fp=_csv_fp(t02_path)),
        _art(root, fig1_path),
        _art(root, fig2_path),
        _art(root, fig3_path),
        _art(root, grammar_path),
    ]
    recon = {
        "no_science_change": True,
        "source_hashes": source_hashes,
        "removed_from_paper": removed,
        "paper_files": list(PAPER_FILES),
        "table_01": {"n_rows": 5, "groups": [r["Nominal twin group"] for r in t01], "status": [r["Analytical status"] for r in t01]},
        "table_02": t02_recon,
        "figure_01": fig1_meta,
        "figure_02": {**fig2_recon, **fig2_meta},
        "figure_03": fig3_meta,
        "paper_md_sha256": paper_md_sha,
        "need_table_after": need_table,
        "need_plot_after": need_plot,
        "asset_hashes": hashes,
        "visual_grammar_sha256": sha256_file(grammar_path),
    }
    recon_path = out / "S10_ASSET_RECONCILIATION.json"
    dump_json(recon_path, recon)
    recs.append(_art(root, recon_path))
    report_path = root / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# S10 — Figures and tables\n\n"
        "Presentation-only materialization from frozen S01/S04/S07/S08/S09 artifacts. "
        "No new estimand, arm, threshold, or inference.\n\n"
        f"- Table 01 rows: {len(t01)}\n"
        f"- Table 02 rows: {len(t02)}\n"
        f"- Primary RQ1 plant-balanced OTG (exact): `{t02_recon['primary_rq1_exact']}`\n"
        f"- Primary RQ2 plant-balanced |A| (exact): `{t02_recon['primary_rq2_exact']}`\n"
        f"- Primary RQ3 D_fleet CONTROL_MINUS_TWIN (exact): `{t02_recon['primary_rq3_exact']}`\n"
        f"- Figure 02 group `{FIG2_GROUP}` plants `{','.join(plants)}`; computable `{fig2_recon['computable_cells']}`; NA `{fig2_recon['na_offdiag']}`; M=`{fig2_meta['M']}`\n"
        f"- Figure 03 targets: 14, ascending `target_plant_id`\n"
        f"- paper.md SHA-256 `{paper_md_sha}`\n"
        f"- unresolved NEED TABLE/PLOT: `{need_table}` / `{need_plot}`\n\n"
        f"**{MSG_GO}**\n",
        encoding="utf-8",
    )
    recs.append(_art(root, report_path))
    recs.append(_art(root, root / "paper.md"))
    manifest = {"job": JOB, "artifacts": [_tab(r) for r in recs]}
    manifest_path = out / "S10_MANIFEST.json"
    dump_json(manifest_path, manifest)
    recs.append(_art(root, manifest_path))
    validations.append(ValidationRecord("assets", True, "five flat paper files"))
    validations.append(ValidationRecord("markers", True, "NEED TABLE/PLOT replaced; NEED REFERENCE untouched"))
    t01_tab = _tab(recs[0])
    t02_tab = _tab(recs[1])
    patch = {
        "stages": {
            "S10": {
                "kind": "scientific_output_materialization",
                "status": "GO",
                "reason": "S10_FIGURES_TABLES_COMPLETE",
                "operational_stage_id": "38",
                "job": JOB,
                "run_id": ctx.run_id,
                "finished_at_utc": now,
                "answer": ANSWER,
                "report": _tab(_art(root, report_path)),
                "manifest": _tab(_art(root, manifest_path)),
            }
        },
        "scientific_analysis": {
            "s10": {
                "status": "GO",
                "source_stage_basis": ["S01", "S04", "S07", "S08", "S09"],
                "asset_count": 5,
                "flat_paper_directory": True,
                "unresolved_table_markers": 0,
                "unresolved_plot_markers": 0,
                "visual_grammar": _tab(_art(root, grammar_path)),
                "reconciliation": _tab(_art(root, recon_path)),
                "paper_md_sha256": paper_md_sha,
                "results_do_not_control_science": True,
            }
        },
        "tables": {
            "table_01_nominal_twin_groups": {
                **t01_tab,
                "source_artifact_paths": ["artifacts/s01/S01_TWIN_GROUPS_FROZEN.csv"],
                "status": "MATERIALIZED",
            },
            "table_02_primary_robustness_summary": {
                **t02_tab,
                "source_artifact_paths": [
                    "artifacts/s07/S07_OTG_DIRECTIONAL.csv",
                    "artifacts/s07/S07_ASYMMETRY.csv",
                    "artifacts/s08_scientific/S08_FLEET_DESCRIPTIVE_SUMMARY.json",
                    "artifacts/s09_scientific/S09_RQ1_SUMMARY.csv",
                    "artifacts/s09_scientific/S09_RQ2_SUMMARY.csv",
                    "artifacts/s09_scientific/S09_RQ3_FLEET_DESCRIPTIVE_SUMMARY.csv",
                    "artifacts/s09_scientific/S09_R6_ORIGIN_SUMMARY.csv",
                ],
                "primary_rq1_aggregation": "plant_balanced_mean_OTG",
                "primary_rq2_aggregation": "plant_balanced_mean_absolute_asymmetry",
                "rq3_orientation": "CONTROL_MINUS_TWIN",
                "r6_pooled": False,
                "status": "MATERIALIZED",
            },
        },
        "figures": {
            "figure_01_experimental_concept": {
                **_tab(recs[2]),
                "format": "PDF",
                "vector_pdf": True,
                "source_method_basis": "frozen_OTG_same_row_protocol",
                "status": "MATERIALIZED",
                "visual_grammar_id": grammar["id"],
                "visual_grammar_sha256": sha256_file(grammar_path),
            },
            "figure_02_operational_twin_map": {
                **_tab(recs[3]),
                "format": "PDF",
                "vector_pdf": True,
                "source_artifact_paths": [
                    "artifacts/s07/S07_OTG_DIRECTIONAL.csv",
                    "artifacts/s04/S04_GSELF_SUMMARY.csv",
                ],
                "status": "MATERIALIZED",
                "visual_grammar_id": grammar["id"],
                "visual_grammar_sha256": sha256_file(grammar_path),
            },
            "figure_03_twin_vs_matched_nontwin": {
                **_tab(recs[4]),
                "format": "PDF",
                "vector_pdf": True,
                "source_artifact_paths": ["artifacts/s08_scientific/S08_TARGET_LEVEL_INFERENCE.csv"],
                "status": "MATERIALIZED",
                "visual_grammar_id": grammar["id"],
                "visual_grammar_sha256": sha256_file(grammar_path),
            },
        },
    }
    return StageResult(status="GO", message=MSG_GO, sot_patch=patch, artifacts=recs, validations=validations)
