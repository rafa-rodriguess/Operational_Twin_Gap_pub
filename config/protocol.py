"""Scientific protocol as code. Changing a value requires a protocol_id bump."""

from __future__ import annotations

PROTOCOL_ID = "otg_protocol_v1"
SEED_PREFIX = "Operational_Twin_Gap|proposal_v1.3"

DOI = "10.5281/zenodo.21511487"
RAW_DIR = "data/raw/br_pvgen"
RAW_FILES = (
    "BR-PVGen_metadata.csv",
    "BR-PVGen_inverter.zip",
    "BR-PVGen_meteorological.zip",
    "Readme.txt",
)
ZENODO_MD5 = {
    "BR-PVGen_inverter.zip": "fbdb7cdb328a3e6ffb470f95bc56bcf4",
    "BR-PVGen_meteorological.zip": "7c20c037074734f426d79a4e3f0db0bf",
    "BR-PVGen_metadata.csv": "abae1b232035097a55d8534a9ec51ab8",
    "Readme.txt": "49bbfb45c13db13baf2c669a7d67c1d0",
}

TWIN_KEY = (
    "is_panel_bifacial",
    "nominal_power_mw",
    "number_of_panels",
    "panel_area_mm2",
    "panel_bifaciality_coefficient",
    "panel_efficiency_percentage",
    "panel_temperature_coefficient",
    "structure_type",
)
STATE_FIELD = "brazil_federative_unit"
EXCLUDE_FROM_TWIN_KEY = ("id", STATE_FIELD)
TRACKER_ALBEDO = "EXCLUDE"

TARGET_PRIMARY = "y_dc_normalized"
TARGET_ROBUSTNESS_R2 = "y_ac_normalized"
FEATURES_SIX_CORE = (
    "poa_irradiance_wm2",
    "ghi_irradiance_wm2",
    "gri_irradiance_wm2",
    "panel_temperature_celsius",
    "ambient_temperature_celsius",
    "wind_speed_ms",
)
FEATURES_POA_GHI = ("poa_irradiance_wm2", "ghi_irradiance_wm2")

POA_PRIMARY = 50.0
POA_R3 = (20.0, 100.0)
COVERAGE_DC_PRIMARY = 0.80
INTERP_PRIMARY = "exclude_TRUE"
INTERP_R5 = "keep_all_valid"
NO_FEATURE_IMPUTATION = True
SPLIT = "eligibility_then_chronological_60_20_20"
TRAIN_FRAC = 0.60
VAL_FRAC = 0.80

# Expected group hashes under TWIN_KEY (recomputed from metadata; used for reconciliation).
PRIMARY_GROUP_IDS = (
    "TW_04b9f5d95694",  # 1 MW FIXED, n=2, RJ
    "TW_4467a039e00b",  # 2.5 MW TRACKER, n=10, BA/GO/MS
    "TW_88a108921af7",  # 4 MW TRACKER, n=2, SP
)
SENSITIVITY_A_GROUP_ID = "TW_7b33a493697b"  # 2 MW FIXED, reduced features
NON_EVALUABLE_GROUP_ID = "TW_f8e8375afe38"  # 3 MW FIXED

CS4 = {"id": "CS4", "k": 5, "q": 0.95}
CS3 = {"id": "CS3", "k": 5, "q": 0.90}
NN_ALGORITHM = "kd_tree"
QUANTILE_METHOD = "linear"
DEGENERATE_IQR = "not_computable"  # no epsilon rescue

PRIMARY_LOSS = "MAE_on_normalized_power"
OTG_ABS = "transferred_error_minus_local_target_error_on_same_supported_rows"
OTG_REL = "secondary_only"
RQ1_SYNTHESIS = "plant_balanced_mean_OTG"
RQ2_SYNTHESIS = "plant_balanced_mean_absolute_asymmetry"
EQUIVALENCE_MARGIN = None
GSELF_BLOCKS = 3

FAM_SPLINE = "SplineRidge"
FAM_HGB = "HistGradientBoosting"
SPLINE_DEGREE = 3
SPLINE_N_KNOTS = (4, 6, 8)
SPLINE_ALPHA = (0.1, 1.0, 10.0)
HGB_LEARNING_RATE = (0.05, 0.10)
HGB_MAX_LEAF_NODES = (15, 31)
HGB_L2 = (0, 1)
HGB_MAX_ITER = 300
HGB_EARLY_STOPPING = False
HGB_LOSS = "absolute_error"
MLP_IN_SCOPE = False
HYPERPARAM_SELECTION = "local_chronological_validation_MAE_only"
FINAL_FIT = "train_only_after_validation_selection"

MATCH_HIERARCHY = (
    "structure_type",
    "nominal_power_mw",
    "panel_efficiency_percentage",
    "panel_bifaciality_coefficient",
    "is_panel_bifacial",
    "number_of_panels",
    "panel_area_mm2",
    "panel_temperature_coefficient",
)
MATCH_REPLACEMENT = "WITH_REPLACEMENT"
MATCH_REUSE = "ALLOWED_AND_RECORDED"
MATCH_CALIPER = None
RQ3_SAME_STATE = True
RQ3_CROSS_STATE_FALLBACK = False
RQ3_POPULATION = "targets_with_unique_same_state_control"
SUPPORT_ALIGNMENT = "B_PAIRWISE_INTERSECTION"
CONTRAST_ORIENTATION = "CONTROL_MINUS_TWIN"
TARGET_AGGREGATION = "M1_EQUAL_WEIGHT_ARITHMETIC_MEAN"
FLEET_SYNTHESIS = "F1_TARGET_BALANCED_ARITHMETIC_MEAN"

BOOTSTRAP = "moving_block_bootstrap_calendar_time"
B_BOOT = 5000
CI_LEVEL = 0.95
CI_TYPE = "percentile"
NO_REFIT_IN_DRAWS = True
EMPTY_DRAW_POLICY = "P1_CONDITIONAL_VALID_DRAW_RESAMPLING"
C8_METHOD = "C8_BONFERRONI_SIMULTANEOUS_TARGET_MBB_PROJECTION"
C8_SCOPE = "FIXED_SET_CONDITIONAL_FLEET_UNCERTAINTY"
C8_ALPHA_FAMILY = 0.05
C8_ON_ROBUSTNESS = False

ROBUSTNESS_DESIGN = "OFAT"
R7_OMITTED = True
R6_ORIGINS = ("O1", "O2", "O3")
R6_BALANCED_PANEL = True
BLOCK_LENGTH_ARMS = ("0.5*L_j", "L_j", "2*L_j")
