"""Frozen RQ4 constants. Do not change from confirmatory outcomes."""

from __future__ import annotations

from itertools import product

SPEC_PATH = "docs/method_amendments/RQ4_SOURCE_SELECTION_FROZEN_SPEC.md"
SPEC_SHA256 = "75ecc2d242a576655cf7e17580f737f570723e9ceb39a203d9a576cbf8304e66"
PARTITION_PATH = "artifacts/scope/PRIMARY_SCOPE.json"
SUMMARY_PATH = "artifacts/p02c/P02C_PLANT_SUMMARY.csv"
PANEL_PATH = "artifacts/p02c/P02C_PLANT_PANEL.parquet"
META_PATH = "data/raw/br_pvgen/BR-PVGen_metadata.csv"
MEMBERS_PATH = "artifacts/twins/members.csv"
OUT_DIR = "artifacts/rq4"

FEATURES = (
    "poa_irradiance_wm2",
    "ghi_irradiance_wm2",
    "gri_irradiance_wm2",
    "panel_temperature_celsius",
    "ambient_temperature_celsius",
    "wind_speed_ms",
)
TARGET = "y_dc_normalized"
PROHIBITED = "tracker_albedo_index"
META_FIELDS = (
    "is_panel_bifacial",
    "nominal_power_mw",
    "number_of_panels",
    "panel_area_mm2",
    "panel_bifaciality_coefficient",
    "panel_efficiency_percentage",
    "panel_temperature_coefficient",
    "structure_type",
)
META_NUMERIC = (
    "nominal_power_mw",
    "number_of_panels",
    "panel_area_mm2",
    "panel_bifaciality_coefficient",
    "panel_efficiency_percentage",
    "panel_temperature_coefficient",
)
META_BOOL = ("is_panel_bifacial",)
META_CAT = ("structure_type",)

L_DAYS = 30
H_DAYS = 45
K_MAX = 45
CUT_SPACING = 45
K_GRID = (0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45)
M_SHORTLIST = 5
RHO = 0.30
CS4_K = 5
CS4_Q = 0.95
TAU_PRIMARY = 0.05
TAU_STRICT = 0.02
BOOT_B = 5000
BOOT_SEED = 42
TRAIN_FRAC = 0.80

SPLINE_GRID = [{"n_knots": k, "alpha": a} for k, a in product((4, 6, 8), (0.1, 1, 10))]
HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaf, "l2_regularization": l2}
    for lr, leaf, l2 in product((0.05, 0.10), (15, 31), (0, 1))
]
