# S12 P8 — Leave-target-out pooled fleet baseline

Secondary benchmark only. Accepted primary RQ1/RQ2/RQ3, S11, source-selection, and S12 P0–P7 are unchanged. Headline comparison uses the 85 accepted primary CS4 SplineRidge directions and the exact S06/S07 TEST support rows. No pooled support mask is defined. RQ2 is not redefined. RQ3 is out of scope.

Status: `S12_P8_POOLED_FLEET_BASELINE_MATERIALIZED`.

pooled_outperforms_twin: plant-balanced GAIN is positive for both variants and both bootstrap CIs lie above zero. Nominal twin equivalence remains scientifically informative but is not the strongest zero-shot deployment strategy among those tested. row- and plant-balanced pooling select different configurations on 9/14 targets; their PB OTGs differ by 0.0025179709995671647 and PB GAINs differ by 0.002517970999567163; therefore weighting choice affects the pooled baseline to that observed degree. G2 versus non-G2 plant-balanced GAIN is reported descriptively; non-G2 contains few plants.

## Setup

- 14 primary plants; 6 unique S04 SplineRidge full selected tuples; leave-target-out donors = 13.
- Features: poa_irradiance_wm2, ghi_irradiance_wm2, gri_irradiance_wm2, panel_temperature_celsius, ambient_temperature_celsius, wind_speed_ms (no plant/state/group IDs).
- Selection: equal mean of donor-plant VALIDATION MAE; lexicographic hyperparameter tie-break; final fit on donor TRAIN only.
- P8-B weights `w_ij = 1/n_j`. SplineTransformer uses accepted uniform knots (no sample-weight fit). The same weight vector is passed as `scaler__sample_weight` and `ridge__sample_weight`, so each donor has equal total row weight before preprocessing and Ridge.
- Row-pooled algorithm unchanged; frozen recon vs first P8 run: `True`.

## Fleet RQ1-style OTG (equal-target)

- Twin PB OTG = 0.009723604857785641 (accepted 0.009723604857785641; n_dir=85, n_targets=14).
- Row-pooled PB OTG = -0.00497426794148001.
- Plant-balanced pooled PB OTG = -0.0024562969419128456.
- PB GAIN row vs twin = 0.014697872799265651.
- PB GAIN balanced vs twin = 0.012179901799698488.

## Paired GAIN vs twin (85 directions)

- Row-pooled: mean=0.017406247146281317, median=0.006947090364652159, P25=0.001852391143331203, P75=0.015702165741794942, P95=0.10432546181543012, min=-0.010738767061942434, max=0.1326187256761859; n>0=70, n<0=15, n=0=0; PB=0.014697872799265653.
- Plant-balanced: mean=0.014719912012137178, median=0.004649214859298839, P25=-0.0005597039920530039, P75=0.014773624998860951, P95=0.10051337341011637, min=-0.01473636611237511, max=0.12427533805293056; n>0=60, n<0=25, n=0=0; PB=0.012179901799698486.

## Target-cluster bootstrap (5000, seed 20260921)

- Row-pooled PB GAIN 95% percentile CI = [0.010391494156256828, 0.0183922243087188].
- Plant-balanced PB GAIN 95% percentile CI = [0.00801334799751679, 0.015614029868533122].

## Groups (descriptive)

- G2: n_dir=81, PB GAIN row=0.01801389240530327, bal=0.015242383236250764.
- non-G2 (G1+G4, small): n_dir=4, PB GAIN row=0.006407823784171606, bal=0.004523698208317793.
- G1: n_dir=2; G4: n_dir=2.

## Selection

- Candidate set size = 6.
- Targets with different row vs balanced selected tuples = 9.
- Frequency row-pooled: {"{\"alpha\": 0.1, \"n_knots\": 4}": 5, "{\"alpha\": 0.1, \"n_knots\": 8}": 9}.
- Frequency plant-balanced: {"{\"alpha\": 0.1, \"n_knots\": 4}": 1, "{\"alpha\": 0.1, \"n_knots\": 6}": 9, "{\"alpha\": 0.1, \"n_knots\": 8}": 4}.

Optional common-row best-twin diagnostic: n_targets=10 (not headline).

RQ2 policy: `not_redefined_pooled_is_target_specific_not_source_asymmetry`. RQ3: `P8_RQ3_NOT_RUN_POOLED_BASELINE_SCOPE`.

Independent reconciliation max |Δ| = 1.734723475976807e-18. Models persisted: 28.
