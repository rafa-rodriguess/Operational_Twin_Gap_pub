"""Presentation-only refinements for manuscript Figures 2 and 3."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config.protocol import FAM_SPLINE, PRIMARY_GROUP_IDS
from src.lib.figures import (
    CS4,
    FIG2_GROUP,
    _apply_rc,
    _visual_grammar,
    render_figure_02,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _flag(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def render_relative_and_placebo_ecdf(root: Path, path: Path) -> dict[str, Any]:
    """Preserve the primary relative-OTG ECDF and add a matched-scale placebo panel."""

    grammar = _visual_grammar()
    _apply_rc(grammar)
    colors = grammar["colors"]

    primary = [
        row
        for row in _rows(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv")
        if row.get("group_id") in PRIMARY_GROUP_IDS
        and row.get("model_family") == FAM_SPLINE
        and row.get("support_rule") == CS4
        and _flag(row.get("computable"))
        and _flag(row.get("otg_rel_defined"))
    ]
    half = [
        float(row["otg_abs"])
        for row in _rows(root / "artifacts/s12/p1/S12_P1_OTG_HALF_DIRECTIONAL.csv")
        if _flag(row.get("computable"))
    ]
    placebo_rows = _rows(root / "artifacts/s12/p1/S12_P1_PLACEBO_PLANT_SUMMARY.csv")
    chrono = [
        float(row["otg_placebo_chrono"])
        for row in placebo_rows
        if _flag(row.get("computable_chrono")) and row.get("otg_placebo_chrono") not in {"", None}
    ]
    interleaved = [
        float(row["otg_placebo_interleaved"])
        for row in placebo_rows
        if _flag(row.get("computable_interleaved")) and row.get("otg_placebo_interleaved") not in {"", None}
    ]
    if (len(primary), len(half), len(chrono), len(interleaved)) != (85, 85, 13, 13):
        raise ValueError(
            "unexpected ECDF denominators: "
            f"primary={len(primary)}, half={len(half)}, chrono={len(chrono)}, interleaved={len(interleaved)}"
        )

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.0), dpi=300)
    ax_rel, ax_null = axes

    rel = [float(row["otg_rel"]) for row in primary]
    x, y = _ecdf(rel)
    ax_rel.step(x, y, where="post", color=colors["nominal_twin_source"], linewidth=1.1)
    p75 = float(np.quantile(rel, 0.75, method="linear"))
    ax_rel.axvline(p75, color=colors["zero_reference"], linestyle="--", linewidth=0.8)
    group_colors = {
        "TW_04b9f5d95694": "#009E73",
        "TW_4467a039e00b": "#0072B2",
        "TW_88a108921af7": "#D55E00",
    }
    group_labels = {
        "TW_04b9f5d95694": "G1",
        "TW_4467a039e00b": "G2",
        "TW_88a108921af7": "G4",
    }
    group_y = {
        "TW_04b9f5d95694": 0.025,
        "TW_4467a039e00b": 0.045,
        "TW_88a108921af7": 0.065,
    }
    for group_id in sorted(PRIMARY_GROUP_IDS):
        values = [float(row["otg_rel"]) for row in primary if row["group_id"] == group_id]
        ax_rel.plot(
            values,
            [group_y[group_id]] * len(values),
            "|",
            color=group_colors[group_id],
            markersize=3.2,
            markeredgewidth=0.6,
            label=group_labels[group_id],
        )
    ax_rel.set_xlabel("Relative OTG")
    ax_rel.set_ylabel("ECDF")
    ax_rel.set_ylim(0, 1.02)
    ax_rel.grid(axis="both", color=grammar["grid"], linewidth=0.4)
    ax_rel.text(0.01, 0.97, "(a)", transform=ax_rel.transAxes, ha="left", va="top", fontweight="bold")
    ax_rel.legend(loc="lower right", frameon=False, ncol=1, fontsize=5.8, handlelength=1.0)

    curves = (
        (half, "Half-history twin transfer", colors["nominal_twin_source"], "-", 1.1),
        (chrono, "Chronological placebo", colors["matched_nontwin_control"], "--", 0.9),
        (interleaved, "Interleaved placebo", colors["common_support"], ":", 1.0),
    )
    for values, label, color, style, width in curves:
        x, y = _ecdf(values)
        ax_null.step(x, y, where="post", label=f"{label} (n={len(values)})", color=color, linestyle=style, linewidth=width)
    ax_null.axvline(0.0, color=colors["zero_reference"], linewidth=0.7)
    ax_null.set_xlabel("Absolute OTG (normalized MAE)")
    ax_null.set_ylabel("ECDF")
    ax_null.set_ylim(0, 1.02)
    ax_null.grid(axis="both", color=grammar["grid"], linewidth=0.4)
    ax_null.text(0.01, 0.97, "(b)", transform=ax_null.transAxes, ha="left", va="top", fontweight="bold")
    ax_null.legend(loc="lower right", frameon=False, fontsize=6.0)

    fig.tight_layout(h_pad=1.0)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return {
        "primary_relative_n": len(primary),
        "primary_relative_p75": p75,
        "half_history_n": len(half),
        "chrono_placebo_n": len(chrono),
        "interleaved_placebo_n": len(interleaved),
        "matched_scale_panel": True,
        "primary_not_compared_directly_with_placebo": True,
    }


def render_consistent_na_map(root: Path, path: Path) -> dict[str, Any]:
    """Regenerate the G2 map with one missing-value label: ``NA``."""

    grammar = _visual_grammar()
    _apply_rc(grammar)
    otg = [
        row
        for row in _rows(root / "artifacts/s07/S07_OTG_DIRECTIONAL.csv")
        if row.get("group_id") == FIG2_GROUP
        and row.get("model_family") == FAM_SPLINE
        and row.get("support_rule") == CS4
    ]
    plants = sorted({row["source_plant_id"] for row in otg} | {row["target_plant_id"] for row in otg})
    if len(plants) != 10:
        raise ValueError(f"expected 10 G2 plants, got {plants}")
    lookup = {(row["source_plant_id"], row["target_plant_id"]): row for row in otg}
    matrix: list[list[float | None]] = []
    diagonal: list[list[bool]] = []
    for source in plants:
        values: list[float | None] = []
        diagonal_row: list[bool] = []
        for target in plants:
            is_diagonal = source == target
            diagonal_row.append(is_diagonal)
            if is_diagonal:
                values.append(None)
                continue
            row = lookup.get((source, target))
            if row is None:
                raise ValueError(f"missing G2 direction {source}->{target}")
            values.append(float(row["otg_abs"]) if _flag(row.get("computable")) else None)
        matrix.append(values)
        diagonal.append(diagonal_row)

    gself_lookup = {
        row["plant_id"]: float(row["gself_mean_signed"])
        for row in _rows(root / "artifacts/s04/S04_GSELF_SUMMARY.csv")
        if row.get("model_family") == FAM_SPLINE
    }
    gself = [gself_lookup[plant] for plant in plants]
    meta = render_figure_02(path, grammar, plants, matrix, diagonal, gself)
    return {
        **meta,
        "group_id": FIG2_GROUP,
        "missing_label": "NA",
        "diagonal_label": "NA",
        "computable_cells": sum(value is not None for row in matrix for value in row),
    }
