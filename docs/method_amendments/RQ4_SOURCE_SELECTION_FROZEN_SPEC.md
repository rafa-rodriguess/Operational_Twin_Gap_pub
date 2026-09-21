# RQ4 source-selection protocol — frozen specification

**Job:** `FREEZE_RQ4_SOURCE_SELECTION_PROTOCOL`  
**Canonical path:** `docs/method_amendments/RQ4_SOURCE_SELECTION_FROZEN_SPEC.md`  
**Archived prompt:** `docs/prompts/FREEZE_RQ4_SOURCE_SELECTION_PROTOCOL.md`  
**SoT freeze namespace:** `scientific_freeze.rq4_source_selection`  
**SoT results namespace (later only):** `scientific_analysis.rq4_source_selection`  
**This document computes no RQ4 outcomes.**

---

## 1. Status

CANONICAL SCIENTIFIC SPECIFICATION. After this file is committed, none of its scientific parameters may be changed based on observed RQ4 outcomes unless a formally documented FATAL methodological problem is identified. Amendments must be created **before** using affected outcomes.

## 2. Scientific motivation

Nominal technical similarity is not necessarily sufficient for operational model interchangeability across plants. RQ4 is the operational follow-on: given previously trained source models and a new target plant, how much target-side information is required to select a source whose future performance is close to the best source that could have been selected retrospectively. The scientific object is the **value of target-side information for source-model selection under temporal uncertainty**. Distinguish: (1) metadata-only source choice; (2) limited target-side verification; (3) the future oracle source; (4) intrinsic temporal instability of the best-source identity/ranking.

## 3. Exact RQ4

RQ4 — How much target-side verification data are needed to select a source model whose future transfer error is close to the best available source, and when does the remaining selection regret become no larger than the temporal instability of the future source ranking itself?

Do not materially rewrite this question during future execution.

## 4. Scientific scope

**In scope:** source-model selection; metadata-only source choice; target-side verification; future source-selection regret; temporal stability of source rankings.

**Out of scope (not primary):** model fine-tuning; target-side retraining; target adaptation; forecasting architecture comparison; hyperparameter benchmark; transfer-learning architecture benchmark; generic similarity-metric benchmark.

Target outcomes choose among **existing** source models. They do not train or adapt them. No local target model is required for primary RQ4.

## 5. Dataset and cohort

- Dataset: canonical BR-PVGen analytical data already in the project Source of Truth. No external dataset.
- Confirmatory **target cohort:** the protected target cohort already frozen in the project partition artifact. Reconstruct programmatically. Verify exact target IDs, artifact hash, target count. No outcome-based membership changes.
- Confirmatory **source cohort:** freeze before opening confirmatory target outcomes. Construct from the previously authorized development/source population. Confirmatory target plants must not be used as outcome-trained sources for other confirmatory targets. Persist exact source IDs and hash before confirmatory scoring.
- Primary source-pool policy: **P_ALL** (all structurally and analytically eligible plants except the target itself). Do not restrict the primary pool to exact nominal twins, same structure type, or same state (those may be reported secondarily).

## 6. Target and features

- Target: `y_dc_normalized`.
- Prohibited field: `tracker_albedo_index` — never feature, target, metadata-distance input, support/stratification/matching variable, proxy, QC proxy, or explanatory variable.
- Frozen six-core features **in this order:**
  1. `poa_irradiance_wm2`
  2. `ghi_irradiance_wm2`
  3. `gri_irradiance_wm2`
  4. `panel_temperature_celsius`
  5. `ambient_temperature_celsius`
  6. `wind_speed_ms`
- No additional primary features after outcomes are opened.

## 7. Eligibility

A row is eligible only when all hold:

- `poa_irradiance_wm2 > 50`
- `coverage_dc >= 0.80`
- `y_dc_normalized` is finite
- all six active features are finite
- rows explicitly marked TRUE for interpolation in DC or an active feature are excluded

UNKNOWN interpolation state must **not** be recoded as FALSE. Reuse canonical project eligibility semantics.

## 8. Decision instance

Fundamental unit: `(target plant j, decision cut c)`.

At cut `c`: candidate source models already exist; sources may use only information **strictly before** `c`; target-side information becomes available after `c`; the selected source is evaluated on a future period **not** used for source selection.

Cuts are deterministic. Primary spacing: **45 calendar days**. For target-level primary summaries, successive cuts’ future reference windows must not overlap. Cut eligibility depends only on timestamps, required pre-cut source history, required post-cut target horizon, feature/data availability, and predefined structural rules. Never select cuts by predictive performance.

## 9. Source training

- Lookback **L = 30 calendar days**. `SOURCE_WINDOW = [c - 30 days, c)`.
- No observation at or after `c` in source fitting, preprocessing, hyperparameter selection, support construction, or metadata-distance estimation.
- Within a valid 30-day window: first **80%** chronological = TRAIN; final **20%** = VALIDATION. No random split. Hyperparameters from source VALIDATION only.

## 10. Primary model

Family: **SplineRidge**. Pipeline: `SplineTransformer` → `StandardScaler` → `Ridge`.

- `degree = 3`
- `n_knots ∈ {4, 6, 8}`
- `alpha ∈ {0.1, 1, 10}`
- Criterion: minimum finite source VALIDATION MAE
- Exact ties: (1) smaller `n_knots`; (2) smaller `alpha`
- Record selected parameters for every source/cut
- No target observation in hyperparameter selection

## 11. Sensitivity model

`HistGradientBoostingRegressor` only as predefined sensitivity.

- `learning_rate ∈ {0.05, 0.10}`
- `max_leaf_nodes ∈ {15, 31}`
- `l2_regularization ∈ {0, 1}`
- Fixed: `loss = "absolute_error"`, `max_iter = 300`, `early_stopping = False`, `min_samples_leaf = 20`

HGB must never replace SplineRidge as primary based on observed RQ4 results. Sensitivity does not reselect L, H, K_max, k, M, rho, metadata distance, candidate pool, or estimands.

## 12. Candidate source universe

Primary: **P_ALL** as in §5. Target ≠ source. After filtering (§15–§18), a decision instance needs **≥ 2** candidate sources; otherwise structurally non-informative for regret: record, exclude from primary regret estimation, and report candidate-count distribution (p25, median, p75) and number of lost instances.

## 13. Metadata distance

Primary: **Gower distance** on frozen technical metadata only:

- `is_panel_bifacial`
- `nominal_power_mw`
- `number_of_panels`
- `panel_area_mm2`
- `panel_bifaciality_coefficient`
- `panel_efficiency_percentage`
- `panel_temperature_coefficient`
- `structure_type`

Numeric: standard Gower range-normalized absolute difference. Boolean/categorical: match = 0, mismatch = 1. Aggregate the mean over fields valid for that pair. If a field is missing on either plant, drop that field for that pair. If no valid fields remain, distance is undefined and the candidate is excluded. Identical missing-value treatment for all targets. No outcome-informed weights. No operational target outcome or transfer-error in metadata distance.

## 14. Shortlist

**M = 5.** For each target: Gower to all valid candidates; sort ascending; keep five nearest (or all if fewer than five). Record whether the shortlist contains an exact nominal twin. Do not force twins into the shortlist or give them extra priority.

**Ties (k = 0 scientific estimand):** minimum-distance ties → **expected regret under uniform selection** among tied minimum-distance candidates. Do not simulate the random draw. Display-only identity: lexicographically smallest plant ID (must not replace the expected-regret estimand).

## 15. Common support

Canonical **CS4**, source-defined, TRAIN only:

1. source TRAIN rows only  
2. feature medians  
3. feature IQRs  
4. robust-standardize the six-core features  
5. Euclidean nearest-neighbour support, **kNN = 5**  
6. leave-one-out 5th-neighbour distances  
7. threshold = **95th percentile** of those distances  

Target rows use the frozen source TRAIN transform and threshold. No target outcome in CS4.

**Degeneracy:** invalid/degenerate TRAIN IQR on any required feature → source/cut support engine invalid; record structurally unavailable; do not replace IQR ad hoc after results.

**Coverage `rho = 0.30`.** Domain `D_cov` = eligible target rows with finite six-core features in `[c, c + K_max)` (feature-only; `y_dc_normalized` not required for the coverage ratio). Numerator = count of `D_cov` rows inside CS4 of source `s`. Denominator = `|D_cov|`. Require `coverage = numerator / denominator >= 0.30` (undefined if denominator = 0 → source invalid). After filtering, for each `W_k` and for `R`, compare candidates on the **intersection** of supported eligible rows across retained candidates. Pairwise candidate-specific target samples are prohibited for ranking.

## 16. Verification windows

`K_max = 45` calendar days. Exact grid:

`k ∈ {0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45}`

No additional `k` after outcomes. For `k > 0`: `W_k = [c, c + k)`. For `k = 0`: no target outcome; metadata-only selection.

## 17. Future reference window

`H = 45` calendar days. `R = [c + 45 days, c + 90 days)`. Mandatory: `W_k ∩ R = ∅` for every `k`. **The same `R` for every `k` within a decision instance.**

## 18. Source-selection rule

- **k = 0:** metadata only; `Reg_metadata` = expected future regret under uniform choice among all minimum-Gower ties (§14).
- **k > 0:** `MAE_s(W_k)` on common supported target rows; `s_hat(k) = argmin_s MAE_s(W_k)`. No refitting, fine-tuning, recalibration, or adaptation.
- **Verification MAE ties:** (1) lower Gower distance; (2) lexicographically smaller source plant ID. No stochastic ties.

## 19. Baselines

- **Random:** `L_random(R) = mean_s L_s(R)` over the candidate set; `Reg_random = L_random(R) - L*(R)`. Analytic; do not simulate draws.
- **Metadata:** `Reg_metadata` from the k = 0 policy (primary operational baseline).
- **Oracle:** regret 0 by definition; never an implementable operational strategy.

## 20. Absolute regret

`L_s(R) = MAE_s(R)` on common R rows. `s* = argmin_s L_s(R)`, `L*(R) = min_s L_s(R)`. Oracle used only after selection for evaluation.

`Reg_abs(k) = L_{s_hat(k)}(R) - L*(R)`  
= extra future normalized MAE because the selected source was not the retrospective best.

For k = 0 with metadata ties of size T: `Reg_metadata = (mean_{s in T} L_s(R)) - L*(R)`.

## 21. Relative regret

Primary scale: `Reg_rel(k) = Reg_abs(k) / L*(R)` only when `L*(R)` is finite and `L*(R) > 0`. **No epsilon.** Invalid relative-regret instances remain missing and must be counted. Absolute regret is mandatory secondary. Both always materialized.

## 22. Temporal noise floor

Within every future `R`, partition calendar time into alternating seven-day blocks A, B, A, B, … Compute both directions: best source on A → regret on B; best source on B → regret on A. `NoiseFloor_abs = mean(Reg_A_to_B, Reg_B_to_A)` and, where valid, `NoiseFloor_rel`. Empirical temporal-stability reference — **not** a theoretical irreducible lower bound.

Secondary chronological-half diagnostic: first-half selection → second-half regret and the reverse.

`ExcessNoise(k) = Reg_abs(k) - NoiseFloor_abs`. Positive: selection regret exceeds empirical within-future temporal instability.

## 23. T1

Per instance: `Delta_meta = Reg_random - Reg_metadata` (positive favours metadata). Primary estimand: target-balanced **mean** `Delta_meta`. Also report target-balanced median. Inference: 95% target-cluster bootstrap CI. Claim improvement over random only if the CI is fully above zero.

## 24. T2

Primary verification endpoint **k = 45** (do not replace with a more favourable observed k). `Delta_verify = Reg_metadata - Reg_45` (positive favours verification). Primary: target-balanced mean; also median. 95% target-cluster bootstrap CI.

## 25. T3

Primary `tau = 0.05`; strict sensitivity `tau = 0.02`. For every k: target-balanced **median** `Reg_rel(k)` and 95% bootstrap CI. `k_tau` = earliest k such that `upper95CI[median Reg_rel(k)] <= tau` **and** this remains true for every later k on the frozen grid. Persistence mandatory. If none: report that no window up to 45 days demonstrated the prespecified regret tolerance. Do not extend `K_max`.

## 26. T4

For every k: target-balanced median `ExcessNoise(k)` and 95% bootstrap CI. `k_noise` = earliest k with `upper95CI[median ExcessNoise(k)] <= 0` and persistence for every later k. If none: report that selection regret remains distinguishable from the empirical temporal-instability reference within the evaluated horizon.

**T3/T4 cases (no redesign):**

1. both exist → finite horizon reaches operational tolerance and temporal-resolution reference  
2. `k_tau` only → tolerance reached; residual error still above temporal reference  
3. `k_noise` only → temporal-resolution limit reached but exceeds operational tolerance  
4. neither → evaluated horizon insufficient under both criteria  

## 27. Aggregation

Primary: (1) summary within each target; (2) equal weight per target at fleet level. Targets with more valid cuts must not dominate. Inference unit: **target plant**. Cuts within a target, candidate-source rows, and timestamps are not independent fleet replicates.

## 28. Bootstrap inference

`B = 5000`. Cluster bootstrap by target plant (all valid instances travel with the target). Deterministic seed frozen in the SoT **before** confirmatory scoring. This freeze sets `bootstrap_seed = 42` (study `config/project.yaml` seed). 95% percentile interval. Do not change B after confirmatory results.

## 29. Multiplicity discipline

No independent significance testing at every k. Primary inferential objects: (1) T1 `Delta_meta`; (2) T2 `Delta_verify` at k=45; (3) persistent `k_tau`; (4) persistent `k_noise`. The full k curve is one structured trajectory.

## 30. Secondary analyses

Exactly: p75/p90 relative regret; absolute regret; top-1 future oracle recovery `I[s_hat(k)==s*]` (intuitive only; not primary success); Spearman rank correlation of W_k vs R rankings when ≥ 3 candidates; exact-twin present vs absent; same-state descriptive; structure-type descriptive; seasonal descriptive; chronological temporal-drift diagnostic; HGB sensitivity. Do not promote secondaries to new primaries after outcomes.

## 31. Leakage invariants

1. no source data at or after `c` in training  
2. no target outcomes in source training  
3. no target outcome in metadata distance  
4. no R outcome in W_k selection  
5. no R outcome in shortlist construction  
6. no R outcome in support construction  
7. no R outcome in hyperparameter selection  
8. no target-local model for primary RQ4  
9. no fine-tuning  
10. no adaptation  
11. candidate pool frozen before source-selection scoring  
12. R identical across k  
13. supported R rows identical across candidate sources  

## 32. Validation tests

Future implementation must test at least items 1–51 listed in the archived prompt (`docs/prompts/FREEZE_RQ4_SOURCE_SELECTION_PROTOCOL.md`, section REQUIRED VALIDATION TESTS), including: deterministic target/source cohort reconstruction and hashes; target ≠ source; TRAIN/VALIDATION strictly before `c`; 80/20 chronological split; frozen feature order and eligibility; `tracker_albedo_index` absent; SplineRidge primary and exact grids; L=30, H=45, K_max=45, exact k; deterministic non-overlapping primary R; W_k ∩ R = ∅; same R for all k; P_ALL; Gower; M=5; rho=0.30; CS4 TRAIN, kNN=5, q=0.95; common target rows; ≥2 candidates; analytic random and metadata-tie baselines; deterministic verification ties; oracle on same future rows; relative regret without epsilon; 7-day A/B noise both directions; chronological-half sensitivity; target-balanced aggregation; target-cluster bootstrap B=5000 with SoT seed; persistent k_tau and k_noise; no per-k fishing; HGB does not change primary; deterministic rerun; manuscript values only from SoT/artifacts.

## 33. Source of Truth contract

Only scientific SoT: `artifacts/sot.json`. Freeze namespace `scientific_freeze.rq4_source_selection` with at least:

`status`, `rq4_wording`, `dataset`, `target`, `features`, `eligibility`, `target_cohort`, `source_cohort`, `source_lookback_days`, `train_validation_split`, `primary_model`, `primary_model_grid`, `sensitivity_model`, `sensitivity_model_grid`, `k_grid`, `K_max`, `H`, `cut_spacing_days`, `candidate_pool`, `metadata_distance`, `metadata_fields`, `shortlist_M`, `rho`, `support_definition`, `regret_absolute_definition`, `regret_relative_definition`, `random_baseline_definition`, `metadata_baseline_definition`, `oracle_definition`, `noise_floor_definition`, `tau_primary`, `tau_sensitivity`, `T1`, `T2`, `T3`, `T4`, `aggregation`, `bootstrap_B`, `bootstrap_seed`, `secondary_analyses`, `claim_guardrails`, `validation_invariants`, `specification_path`, `specification_sha256`.

Empirical results later only under `scientific_analysis.rq4_source_selection`. Do not mix design parameters and observed results.

## 34. Artifact contract

Predictions, support masks, decision-instance results, and bootstrap draws may live in Parquet/Arrow outside the SoT. For each large artifact the SoT records path, SHA-256, row count, schema fingerprint, and summaries used for manuscript reporting. No manuscript value from an unregistered artifact.

## 35. Claim rules

Claims must stay conditional on the observed fleet, protected target cohort, frozen source pool, evaluated horizon, and common-support domain.

Do **not** claim: a universal number of target days for PV plants; causal source superiority; causal interpretation of metadata distance; universal superiority of Gower; performance outside BR-PVGen; theoretical irreducibility of the empirical noise floor; elimination of future adaptation or recalibration.

## 36. Null-result interpretations

All primary patterns are scientifically valid (metadata±verification help; neither helps; residual at/above noise floor). No outcome triggers automatic redesign.

## 37. Figure specification

X: verification days k. Y: target-balanced relative regret. Display metadata-only at k=0; verification curve; 95% uncertainty; tau=5%; tau=2%; empirical temporal noise-floor reference. Secondary/supplement: absolute regret. The figure must show how quickly target-side information improves selection and when extra verification ceases to be practically distinguishable.

## 38. Table specification

Rows: random baseline; metadata k=0; k=1,2,3,5,7,10,14,21,30,45; temporal noise reference; oracle.  
Columns: valid targets; valid decision instances; median candidate count; median relative regret; p75 relative regret; p90 relative regret; median absolute regret; top-1 oracle recovery; rank Spearman; relevant 95% CI.  
No scientific number typed by hand.

## 39. Future orchestration contract

`orchestrator.py` is the only official executor:

`orchestrator.py` → imports `src/run/stage_XX_*.py` → `run(ctx)` → `StageResult` → validates tests/artifacts/`sot_patch` → atomically updates `artifacts/sot.json`.

A stage exposes `run(ctx: StageContext) -> StageResult`; never runs another stage; never writes `artifacts/sot.json` directly; returns `sot_patch` and hashed artifacts. Never instruct direct execution of a stage module. **This freeze job does not implement RQ4 scoring stages.**

---

## Frozen parameter summary

| Parameter | Frozen value |
| --- | --- |
| L | 30 calendar days |
| TRAIN/VAL | 80/20 chronological |
| Primary | SplineRidge degree 3; n_knots {4,6,8}; alpha {0.1,1,10} |
| HGB sensitivity | lr {0.05,0.10}; max_leaf_nodes {15,31}; l2 {0,1}; loss absolute_error; max_iter 300; early_stopping False; min_samples_leaf 20 |
| k | {0,1,2,3,5,7,10,14,21,30,45} |
| K_max, H, cut spacing | 45 calendar days |
| R | [c+45d, c+90d) |
| Pool | P_ALL |
| Distance | Gower, listed metadata fields |
| M | 5 |
| rho | 0.30 |
| CS4 | TRAIN, kNN=5, q=0.95 |
| tau | 0.05 (primary), 0.02 (strict) |
| B | 5000 |
| bootstrap_seed | 42 |
| T2 k | 45 |
