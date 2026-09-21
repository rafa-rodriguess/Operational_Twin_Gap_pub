# JOB NAME

FREEZE_RQ4_SOURCE_SELECTION_PROTOCOL

# STATUS

CANONICAL SCIENTIFIC SPECIFICATION.

This job creates the authoritative methodological specification for RQ4.

The resulting specification defines the scientific question, estimands, data windows, models, candidate-source selection, common support, baselines, inference, validation invariants, Source of Truth contract, and claim rules.

After this specification is committed, none of its scientific parameters may be changed based on observed RQ4 outcomes unless a formally documented FATAL methodological problem is identified.

---

# IMPLEMENTATION OUTPUT

Create exactly:

docs/method_amendments/RQ4_SOURCE_SELECTION_FROZEN_SPEC.md

This document becomes the authoritative scientific specification for RQ4.

---

# ROLE

You are the scientific-methodology executor responsible for materializing the frozen RQ4 protocol.

Your task is NOT to execute RQ4 outcomes.

Your task is to create the complete specification required for a later implementation.

The specification must be sufficiently precise that a separate executor can implement RQ4 without making additional methodological decisions.

---

# SCIENTIFIC CONTEXT

Previous analysis establishes that nominal technical similarity alone is not necessarily sufficient to guarantee operational model interchangeability across plants.

RQ4 addresses the operational decision that follows:

Given several previously trained source models and a new target plant, how much target-side information is required to select a source whose future performance is close to the best source that could have been selected retrospectively?

The scientific object is:

the value of target-side information for source-model selection under temporal uncertainty.

The study must distinguish:

1. source selection using metadata alone;
2. source selection using limited target-side verification;
3. the future oracle source;
4. intrinsic temporal instability in the identity/ranking of the best source.

---

# RQ4

Freeze the research question exactly as:

RQ4 — How much target-side verification data are needed to select a source model whose future transfer error is close to the best available source, and when does the remaining selection regret become no larger than the temporal instability of the future source ranking itself?

Do not materially rewrite this question during future execution.

---

# SCIENTIFIC SCOPE

RQ4 studies:

- source-model selection;
- metadata-only source choice;
- target-side verification;
- future source-selection regret;
- temporal stability of source rankings.

RQ4 is NOT primarily:

- model fine-tuning;
- target-side retraining;
- target adaptation;
- forecasting architecture comparison;
- hyperparameter benchmark;
- transfer-learning architecture benchmark;
- generic similarity-metric benchmark.

Target outcomes are used to choose among existing source models, not to train or adapt them.

---

# DATASET

Use the canonical BR-PVGen analytical data already represented in the project Source of Truth.

No external dataset may be introduced into the empirical analysis.

Target:

y_dc_normalized

Prohibited field:

tracker_albedo_index

The prohibited field must not be used as:

- feature;
- target;
- metadata-distance input;
- support variable;
- stratification variable;
- matching variable;
- proxy;
- QC proxy;
- explanatory variable.

---

# FEATURE SET

Use exactly the frozen six-core feature set in this order:

1. poa_irradiance_wm2
2. ghi_irradiance_wm2
3. gri_irradiance_wm2
4. panel_temperature_celsius
5. ambient_temperature_celsius
6. wind_speed_ms

No additional primary features may be added after outcomes are opened.

---

# PRIMARY ROW ELIGIBILITY

A row is eligible only when:

- poa_irradiance_wm2 > 50;
- coverage_dc >= 0.80;
- y_dc_normalized is finite;
- all six active features are finite;
- rows explicitly marked TRUE for interpolation in DC or an active feature are excluded.

UNKNOWN interpolation state must not silently be recoded as FALSE.

The future implementation must reuse the canonical eligibility semantics already present in the project.

---

# FUNDAMENTAL DECISION UNIT

The fundamental source-selection decision instance is:

(target plant j, decision cut c)

At each cut c:

- candidate source models already exist;
- source models may use only information strictly before c;
- target-side information becomes available after c;
- the selected source is evaluated on a future period not used for source selection.

No local target model is required for RQ4.

---

# SOURCE TRAINING LOOKBACK

Freeze:

L = 30 calendar days

For source plant s at cut c:

SOURCE_WINDOW = [c - 30 days, c)

No observation at or after c may enter:

- source fitting;
- source preprocessing;
- hyperparameter selection;
- support construction;
- metadata-distance estimation.

---

# SOURCE TRAIN / VALIDATION SPLIT

Within each valid 30-day source window:

- first 80% chronologically = TRAIN;
- final 20% chronologically = VALIDATION.

No random train/validation split.

Hyperparameters are selected exclusively from source VALIDATION performance.

---

# PRIMARY MODEL

Primary model family:

SplineRidge

Pipeline:

1. SplineTransformer
2. StandardScaler
3. Ridge

Freeze:

degree = 3

Hyperparameter grid:

n_knots ∈ {4, 6, 8}

alpha ∈ {0.1, 1, 10}

Selection criterion:

minimum finite source VALIDATION MAE.

No target observation may influence hyperparameter selection.

Exact ties must be deterministic.

Exact tie hierarchy:

1. smaller n_knots;
2. smaller alpha.

Record the selected parameters for every source/cut.

---

# MODEL SENSITIVITY

Use:

HistGradientBoostingRegressor

as a predefined sensitivity analysis only.

Grid:

learning_rate ∈ {0.05, 0.10}

max_leaf_nodes ∈ {15, 31}

l2_regularization ∈ {0, 1}

Fixed:

- loss = "absolute_error"
- max_iter = 300
- early_stopping = False
- min_samples_leaf = 20

HGB must never replace SplineRidge as primary based on observed RQ4 results.

---

# VERIFICATION HORIZON

Freeze maximum target-side verification:

K_max = 45 calendar days

Use exactly:

k ∈ {0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45}

No additional k may be introduced after outcomes are observed.

---

# VERIFICATION WINDOW

For every k > 0:

W_k = [c, c + k)

For:

k = 0

no target outcome is available.

Therefore k=0 represents metadata-only source selection.

---

# FUTURE REFERENCE WINDOW

Freeze:

H = 45 calendar days

Define:

R = [c + 45 days, c + 90 days)

Therefore:

W_k ∩ R = ∅

for every k.

The future window R must be identical for every k within the same decision instance.

This is a mandatory invariant.

---

# DECISION-CUT CONSTRUCTION

Decision cuts must be deterministic.

Primary cut spacing:

45 calendar days

For target-level primary summaries, future reference windows associated with successive cuts must not overlap.

Cut eligibility must depend only on:

- timestamps;
- required pre-cut source history;
- required post-cut target horizon;
- feature/data availability;
- predefined structural rules.

Cuts must never be selected based on predictive performance.

---

# SOURCE CANDIDATE UNIVERSE

Primary source-pool policy:

P_ALL

All structurally and analytically eligible plants may initially act as source candidates except the target plant itself.

Do not restrict the primary pool to:

- exact nominal twins;
- same structure type;
- same state.

These characteristics may be reported secondarily.

---

# CONFIRMATORY TARGET COHORT

Use the protected target cohort already frozen in the project partition artifact.

The target cohort must be reconstructed programmatically from the canonical artifact.

The implementation must verify:

- exact target IDs;
- artifact hash;
- target count;
- no outcome-based membership changes.

No target may enter or leave the confirmatory cohort based on RQ4 outcomes.

---

# SOURCE COHORT FOR CONFIRMATORY EXECUTION

Before opening confirmatory target outcomes, freeze the source population.

Confirmatory target plants must not be used as outcome-trained source plants for other confirmatory targets.

The source pool must therefore be constructed from the previously authorized development/source population.

Persist the exact source IDs and hash before confirmatory scoring begins.

---

# TECHNICAL METADATA DISTANCE

Primary metadata distance:

Gower distance

Use only frozen technical metadata.

Metadata family:

- is_panel_bifacial
- nominal_power_mw
- number_of_panels
- panel_area_mm2
- panel_bifaciality_coefficient
- panel_efficiency_percentage
- panel_temperature_coefficient
- structure_type

No operational target outcome or transfer-error information may enter metadata distance.

---

# GOWER DISTANCE

For numeric metadata:

use standard Gower range-normalized absolute difference.

For boolean/categorical metadata:

- match = 0;
- mismatch = 1.

Aggregate over valid metadata fields.

The exact missing-value treatment must be deterministic and implemented identically for all targets.

No outcome-informed feature weighting is permitted.

---

# SOURCE SHORTLIST

Freeze:

M = 5

For each target:

1. compute Gower distance to all valid source candidates;
2. sort ascending;
3. retain the five nearest candidates.

If fewer than five valid sources exist:

retain all valid sources.

Record whether the shortlist contains an exact nominal twin.

---

# METADATA DISTANCE TIES

For scientific evaluation of metadata-only selection:

minimum-distance ties must be handled as expected regret under uniform selection among tied minimum-distance candidates.

Do not simulate the random tie.

For display-only source identity:

use lexicographically smallest plant ID.

The display source must not replace the expected-regret estimand.

---

# COMMON SUPPORT

Use canonical CS4 source-defined common support.

For every source:

1. use source TRAIN rows only;
2. calculate feature medians;
3. calculate feature IQRs;
4. robust-standardize the six-core features;
5. construct Euclidean nearest-neighbor support;
6. use kNN = 5;
7. calculate leave-one-out 5th-neighbor distances;
8. define the support threshold as their 95th percentile.

Target rows are evaluated using the frozen source TRAIN transformation and threshold.

No target outcome enters CS4.

---

# CS4 DEGENERACY

If any required TRAIN feature has invalid/degenerate IQR:

the source/cut support engine is invalid.

Do not replace IQR ad hoc after results are observed.

Record the source/cut as structurally unavailable.

---

# SUPPORT COVERAGE

Freeze:

rho = 0.30

A source may remain in the candidate set only when its support coverage satisfies:

coverage >= 0.30

Coverage must be computed using the predefined feature-only evaluation domain.

The exact numerator and denominator must be documented in the implementation.

---

# COMMON TARGET ROWS

Candidate sources must always be compared on identical target observations.

After candidate filtering:

for each W_k use the intersection of supported rows across retained candidates.

For R use the intersection of supported rows across retained candidates.

Pairwise candidate-specific target samples are prohibited for comparative source ranking.

---

# MINIMUM CANDIDATES

A source-selection decision instance requires at least:

2 candidate sources

Instances with fewer than two candidates are structurally non-informative for regret.

Record but exclude them from primary regret estimation.

Report:

- candidate-count distribution;
- p25;
- median;
- p75;
- number of lost instances.

---

# K = 0 POLICY

At k = 0:

source selection uses technical metadata only.

The scientific metadata baseline is the expected result under uniform choice among all minimum-Gower-distance ties.

Call its future regret:

Reg_metadata

---

# K > 0 POLICY

For each candidate source s and verification window W_k, calculate:

MAE_s(W_k)

using the common supported target rows.

Choose:

s_hat(k) = argmin_s MAE_s(W_k)

No source refitting.

No target fine-tuning.

No recalibration.

No adaptation.

Target-side observations are used only for candidate evaluation and source selection.

---

# VERIFICATION TIE BREAK

Exact verification-MAE ties must follow:

1. lower Gower metadata distance;
2. lexicographically smaller source plant ID.

No stochastic tie breaking.

---

# FUTURE LOSS

For each eligible source candidate:

L_s(R) = MAE_s(R)

computed on the common R rows.

Define the retrospective oracle:

s* = argmin_s L_s(R)

and:

L*(R) = min_s L_s(R)

Oracle information is used only after source selection for evaluation.

---

# ABSOLUTE REGRET

Define:

Reg_abs(k) = L_s_hat(k)(R) - L*(R)

Interpretation:

the additional future normalized MAE incurred because the selected source was not the retrospectively best candidate.

---

# RELATIVE REGRET

Primary regret scale:

Reg_rel(k) = Reg_abs(k) / L*(R)

Compute only when:

- L*(R) is finite;
- L*(R) > 0.

Do not add epsilon.

Invalid relative-regret instances remain missing and must be counted.

---

# PRIMARY OUTCOME SCALE

Primary:

relative regret

Mandatory secondary:

absolute regret

Both must always be materialized.

---

# RANDOM SOURCE BASELINE

Define uniform-random expected future source loss analytically.

For candidate set S:

L_random(R) = mean_s L_s(R)

and:

Reg_random = L_random(R) - L*(R)

Do not simulate random source draws.

---

# METADATA BASELINE

The k=0 policy produces:

Reg_metadata

This is the primary operational baseline against which target verification is compared.

---

# ORACLE BASELINE

Oracle regret:

0

by definition.

The oracle is never described as an implementable operational strategy.

---

# REFERENCE TEMPORAL NOISE FLOOR

The purpose of the reference-noise calculation is to distinguish:

- inadequate verification data;
- intrinsic temporal variation in which source appears best.

Within every future R:

partition calendar time into alternating seven-day blocks:

A, B, A, B, ...

Compute both directions:

1. select best source on A and measure regret on B;
2. select best source on B and measure regret on A.

Define:

NoiseFloor_abs = mean(Reg_A_to_B, Reg_B_to_A)

and, where valid:

NoiseFloor_rel

This quantity is an empirical temporal-stability reference.

Never describe it as a theoretical irreducible lower bound.

---

# TEMPORAL-DRIFT SENSITIVITY

Additionally split R chronologically into:

- first half;
- second half.

Evaluate:

- first-half selection → second-half regret;
- second-half selection → first-half regret.

This is a secondary temporal-drift diagnostic.

---

# EXCESS REGRET ABOVE TEMPORAL NOISE

Define:

ExcessNoise(k) = Reg_abs(k) - NoiseFloor_abs

Positive values indicate source-selection regret exceeding the empirical within-future temporal instability reference.

---

# T1 — VALUE OF METADATA OVER RANDOM SELECTION

Per decision instance:

Delta_meta = Reg_random - Reg_metadata

Positive values favor technical metadata.

Primary estimand:

target-balanced mean Delta_meta.

Also report:

target-balanced median Delta_meta.

Primary inference:

95% target-cluster bootstrap CI.

Interpretation rule:

- CI fully above zero → evidence that metadata improves source selection over random choice;
- otherwise → do not claim improvement over random choice.

---

# T2 — VALUE OF TARGET VERIFICATION OVER METADATA

Primary verification endpoint:

k = 45

Define:

Delta_verify = Reg_metadata - Reg_45

Positive values favor target verification.

Primary estimand:

target-balanced mean Delta_verify.

Also report median.

Inference:

95% target-cluster bootstrap CI.

Do not replace k=45 with a more favorable observed k.

---

# T3 — OPERATIONAL SUFFICIENCY

Primary tolerance:

tau = 0.05

Strict sensitivity:

tau = 0.02

For every k estimate:

target-balanced median Reg_rel(k)

and its 95% bootstrap CI.

Define:

k_tau

as the earliest k for which:

upper95CI[median Reg_rel(k)] <= tau

AND this criterion remains satisfied for every later k in the frozen k grid.

Persistence is mandatory.

If no k qualifies:

report that no verification window up to 45 days demonstrated the prespecified regret tolerance.

Do not extend K_max based on this result.

---

# T4 — TEMPORAL-RESOLUTION SUFFICIENCY

For every k estimate:

target-balanced median ExcessNoise(k)

and its 95% bootstrap CI.

Define:

k_noise

as the earliest k for which:

upper95CI[median ExcessNoise(k)] <= 0

AND the criterion remains true for every later k.

If no k qualifies:

report that source-selection regret remains distinguishable from the empirical temporal-instability reference within the evaluated horizon.

---

# INTERPRETATION OF T3 AND T4

Case 1:

k_tau exists AND k_noise exists.

Conclusion:

a finite target-verification horizon reaches both the operational tolerance and the empirical temporal-resolution reference.

Case 2:

k_tau exists AND k_noise does not exist.

Conclusion:

the operational tolerance is reached, but residual selection error remains greater than the empirical temporal-stability reference.

Case 3:

k_tau does not exist AND k_noise exists.

Conclusion:

verification reaches the empirical temporal-resolution limit, but that limit itself exceeds the prespecified operational tolerance.

Case 4:

Neither exists.

Conclusion:

the evaluated verification horizon is insufficient under both criteria.

No design change may be made in response to which case is observed.

---

# TARGET-BALANCED AGGREGATION

Targets may contribute different numbers of valid cuts.

Therefore primary aggregation must be:

1. calculate the relevant summary within each target;
2. give every target equal weight in the fleet-level estimand.

Do not allow targets with more valid cuts to dominate fleet estimates.

---

# INFERENCE UNIT

Primary independent resampling unit:

target plant

Decision cuts within the same target are not independent replicates.

Candidate-source rows are not independent replicates.

Individual timestamps are not independent replicates for fleet inference.

---

# BOOTSTRAP

Freeze:

B = 5000

Use cluster bootstrap by target plant.

When a target is sampled:

all of its valid decision instances travel with it.

Use a deterministic seed.

The seed must be frozen in the Source of Truth before confirmatory scoring begins.

Bootstrap interval:

95% percentile interval.

No change to B based on observed confirmatory results.

---

# MULTIPLICITY DISCIPLINE

Do not perform independent significance testing at every k.

Primary inferential objects are exactly:

1. T1 Delta_meta;
2. T2 Delta_verify at k=45;
3. persistent k_tau;
4. persistent k_noise.

The full k curve is reported as one structured trajectory.

---

# SECONDARY OUTPUTS

Predefine exactly:

- p75 relative regret;
- p90 relative regret;
- absolute regret;
- top-1 future oracle recovery;
- source-ranking Spearman correlation;
- exact-twin candidate present vs absent;
- same-state descriptive comparison;
- structure-type descriptive comparison;
- seasonal descriptive comparison;
- chronological temporal-drift diagnostic;
- HGB sensitivity.

Do not promote secondary results to new primary hypotheses after outcomes are observed.

---

# TOP-1 RECOVERY

Compute:

I[s_hat(k) == s*]

and summarize across targets.

Use only as an intuitive secondary metric.

Do not treat exact source identity recovery as the primary success criterion.

Two candidate sources may have materially indistinguishable future error.

---

# RANK CONCORDANCE

Where at least 3 source candidates exist:

compute Spearman correlation between:

- candidate ranking in W_k;
- candidate ranking in R.

Use this as a source-ranking stability descriptor only.

---

# EXACT-TWIN CONNECTION

For each decision instance record:

exact_twin_candidate_present = TRUE/FALSE

Report regret curves stratified descriptively by this indicator.

Do not force exact twins into the shortlist.

Do not give exact twins priority beyond the frozen metadata-distance algorithm.

---

# MODEL-FAMILY SENSITIVITY

Run the identical RQ4 design using HGB source models.

Do not reselect:

- L;
- H;
- K_max;
- k;
- M;
- rho;
- metadata distance;
- candidate pool;
- estimands.

Sensitivity asks only whether substantive source-selection conclusions depend strongly on model family.

---

# CLAIM RULES

Allowed claims must remain conditional on:

- observed fleet;
- protected target cohort;
- frozen candidate source pool;
- evaluated time horizon;
- common-support domain.

Do NOT claim:

- universal number of target days required for PV plants;
- causal source superiority;
- causal interpretation of metadata distance;
- universal superiority of Gower distance;
- universal source-selection performance outside BR-PVGen;
- theoretical irreducibility of the empirical noise floor;
- elimination of the need for future adaptation or recalibration.

---

# NULL-RESULT VALIDITY

All primary result patterns are scientifically valid.

Metadata helps and verification helps:

Target-side observations add source-selection information beyond technical metadata.

Metadata helps and verification does not:

Technical metadata contains useful information, but the evaluated target-verification horizon adds no reliable incremental benefit.

Metadata does not help and verification helps:

Static technical metadata is insufficient, but observed target behavior provides useful information for source selection.

Neither helps:

Neither available metadata nor limited target-side verification reliably identifies better future sources.

Verification reaches noise floor:

Residual source-selection error becomes comparable to temporal instability in source ranking.

Verification remains above noise floor:

Residual source-selection error remains larger than the empirical temporal-stability reference.

No outcome triggers automatic redesign.

---

# PRIMARY FIGURE SPECIFICATION

The central RQ4 figure must show:

X-axis:

verification days k

Y-axis:

target-balanced relative regret

Display:

- metadata-only point at k=0;
- verification regret curve;
- 95% uncertainty;
- tau=5%;
- tau=2%;
- empirical temporal noise-floor reference.

Secondary panel or supplement:

absolute regret.

The figure must allow a reader to answer:

how quickly does target-side information improve source selection, and when does additional verification cease to produce practically distinguishable benefit?

---

# PRIMARY TABLE SPECIFICATION

Rows:

- random baseline;
- metadata k=0;
- k=1;
- k=2;
- k=3;
- k=5;
- k=7;
- k=10;
- k=14;
- k=21;
- k=30;
- k=45;
- temporal noise reference;
- oracle.

Columns:

- valid targets;
- valid decision instances;
- median candidate count;
- median relative regret;
- p75 relative regret;
- p90 relative regret;
- median absolute regret;
- top-1 oracle recovery;
- rank Spearman;
- relevant 95% CI.

No scientific number may be typed manually into the table.

---

# LEAKAGE INVARIANTS

Future implementation must verify:

1. no source data at or after c enter training;
2. no target outcomes enter source training;
3. no target outcome enters metadata distance;
4. no R outcome enters W_k selection;
5. no R outcome enters shortlist construction;
6. no R outcome enters support construction;
7. no R outcome enters hyperparameter selection;
8. no target-local model is fitted for primary RQ4;
9. no fine-tuning occurs;
10. no adaptation occurs;
11. candidate pool is frozen before source-selection scoring;
12. R is identical across k;
13. supported R rows are identical across candidate sources.

---

# REQUIRED VALIDATION TESTS

The eventual implementation must test at minimum:

1. deterministic target-cohort reconstruction;
2. target cohort hash matches SoT;
3. deterministic source-cohort reconstruction;
4. source cohort hash matches SoT;
5. target cannot equal source;
6. source TRAIN strictly precedes c;
7. source VALIDATION strictly precedes c;
8. 80/20 chronological source split;
9. no target outcomes in source model selection;
10. exact frozen feature order;
11. exact eligibility rule;
12. tracker_albedo_index absent;
13. SplineRidge primary;
14. exact spline hyperparameter grid;
15. exact HGB sensitivity grid;
16. L exactly 30;
17. H exactly 45;
18. K_max exactly 45;
19. exact k grid;
20. deterministic cuts;
21. non-overlapping primary R windows within target;
22. W_k and R disjoint;
23. same R for all k;
24. P_ALL candidate policy;
25. Gower distance only for primary metadata rule;
26. M exactly 5;
27. rho exactly 0.30;
28. CS4 source TRAIN only;
29. kNN support k=5;
30. CS4 threshold quantile=0.95;
31. same target rows across source candidates;
32. candidate count >=2;
33. random baseline analytic;
34. metadata-tie expectation analytic;
35. verification tie rule deterministic;
36. oracle computed on same future rows;
37. relative regret has no epsilon;
38. invalid relative denominators remain missing;
39. alternating 7-day noise blocks deterministic;
40. both A→B and B→A evaluated;
41. chronological-half sensitivity deterministic;
42. target-balanced aggregation before fleet estimate;
43. target-cluster bootstrap;
44. B=5000;
45. deterministic seed from SoT;
46. persistent k_tau rule exact;
47. persistent k_noise rule exact;
48. no per-k significance fishing;
49. HGB does not change primary model;
50. deterministic rerun outputs;
51. manuscript values derived only from SoT/artifacts.

---

# SOURCE OF TRUTH CONTRACT

The only scientific Source of Truth is:

artifacts/sot.json

The specification must define the canonical freeze namespace:

scientific_freeze.rq4_source_selection

It must contain at minimum:

- status
- rq4_wording
- dataset
- target
- features
- eligibility
- target_cohort
- source_cohort
- source_lookback_days
- train_validation_split
- primary_model
- primary_model_grid
- sensitivity_model
- sensitivity_model_grid
- k_grid
- K_max
- H
- cut_spacing_days
- candidate_pool
- metadata_distance
- metadata_fields
- shortlist_M
- rho
- support_definition
- regret_absolute_definition
- regret_relative_definition
- random_baseline_definition
- metadata_baseline_definition
- oracle_definition
- noise_floor_definition
- tau_primary
- tau_sensitivity
- T1
- T2
- T3
- T4
- aggregation
- bootstrap_B
- bootstrap_seed
- secondary_analyses
- claim_guardrails
- validation_invariants
- specification_path
- specification_sha256

Future empirical results must be written only under:

scientific_analysis.rq4_source_selection

Do not mix design parameters and observed results.

---

# LARGE ARTIFACT CONTRACT

Predictions, support masks, decision-instance results and bootstrap draws may be stored outside the SoT in Parquet/Arrow.

For every large artifact the SoT must record:

- path;
- SHA-256;
- row count;
- schema fingerprint;
- summaries used for manuscript reporting.

No manuscript value may be reconstructed manually from an unregistered artifact.

---

# ORCHESTRATION CONTRACT FOR FUTURE IMPLEMENTATION

orchestrator.py is the only official executor.

Future implementation must follow:

orchestrator.py
→ imports src/stages/stage_XX_*.py
→ calls run(ctx)
→ receives StageResult
→ validates tests/artifacts/sot_patch
→ atomically updates artifacts/sot.json
→ records state.

A stage:

- must expose run(ctx: StageContext) -> StageResult;
- must never directly execute another stage;
- must never write directly to artifacts/sot.json;
- must return changes through sot_patch;
- must return artifacts with hashes.

Never instruct direct execution of a stage module.

---

# OUTPUT SPECIFICATION CONTENT

Create:

docs/method_amendments/RQ4_SOURCE_SELECTION_FROZEN_SPEC.md

The document must contain the following sections:

1. Status
2. Scientific motivation
3. Exact RQ4
4. Scientific scope
5. Dataset and cohort
6. Target and features
7. Eligibility
8. Decision instance
9. Source training
10. Primary model
11. Sensitivity model
12. Candidate source universe
13. Metadata distance
14. Shortlist
15. Common support
16. Verification windows
17. Future reference window
18. Source-selection rule
19. Baselines
20. Absolute regret
21. Relative regret
22. Temporal noise floor
23. T1
24. T2
25. T3
26. T4
27. Aggregation
28. Bootstrap inference
29. Multiplicity discipline
30. Secondary analyses
31. Leakage invariants
32. Validation tests
33. Source of Truth contract
34. Artifact contract
35. Claim rules
36. Null-result interpretations
37. Figure specification
38. Table specification
39. Future orchestration contract

The specification must contain no unresolved methodological choices.

---

# FREEZE RULE

After materialization:

RQ4_SOURCE_SELECTION_FROZEN_SPEC.md

is scientifically frozen.

Future outcome results must not be used to alter:

- L;
- H;
- K_max;
- k grid;
- candidate pool;
- M;
- Gower distance;
- metadata fields;
- rho;
- CS4;
- model family;
- hyperparameter grid;
- target aggregation;
- bootstrap B;
- T1–T4;
- tau values;
- noise-floor definition.

Changes require a formal amendment created BEFORE using the affected outcomes, except for a documented FATAL correctness problem.

---

# WRITE TO SoT

When this specification is formally activated, write through sot_patch only:

scientific_freeze.rq4_source_selection.status = "FROZEN"

and populate all fields defined in the Source of Truth contract above.

Do not write observed RQ4 results in this job.

---

# VALIDATION OF THIS SPECIFICATION JOB

Before completion verify:

- exact RQ4 present;
- no methodological placeholder remains;
- all numerical parameters above are explicit;
- all estimands are mathematically defined;
- all four primary tests are defined;
- all tie rules are defined;
- all leakage rules are defined;
- all inference rules are defined;
- SoT paths are explicit;
- future result namespace is separate from freeze namespace;
- no RQ4 outcomes were computed by this job.

---

# PROMPT ARCHIVAL REQUIREMENT

Save this exact prompt as:

docs/prompts/FREEZE_RQ4_SOURCE_SELECTION_PROTOCOL.md

The archived prompt must preserve all scientific instructions in this prompt.

---

# VERSION CONTROL REQUIREMENT

After generating the frozen RQ4 specification:

1. inspect the git diff;
2. verify only intended specification/prompt files and authorized SoT freeze fields changed;
3. commit;
4. push to the configured repository.

Suggested commit message:

Freeze canonical RQ4 source-selection protocol

---

# DELIVERABLES

Return:

- specification path;
- specification SHA-256;
- archived prompt path;
- exact RQ4 wording;
- frozen parameter summary;
- SoT namespace;
- validation results;
- confirmation that no outcome analysis was executed;
- final commit SHA;
- push status.

---

# SUCCESS MESSAGE

Return exactly:

RQ4_SOURCE_SELECTION_PROTOCOL_FROZEN

Do not execute RQ4 scoring in this job.

Do not generate the next implementation stage unless explicitly requested.
