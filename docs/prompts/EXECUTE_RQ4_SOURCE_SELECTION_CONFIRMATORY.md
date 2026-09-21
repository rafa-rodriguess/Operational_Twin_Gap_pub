# JOB NAME

EXECUTE_RQ4_SOURCE_SELECTION_CONFIRMATORY

# STATUS

CONFIRMATORY EXECUTION OF A FROZEN SCIENTIFIC PROTOCOL.

This job implements and executes RQ4 exactly as frozen in:

docs/method_amendments/RQ4_SOURCE_SELECTION_FROZEN_SPEC.md

The authoritative frozen specification SHA-256 is:

75ecc2d242a576655cf7e17580f737f570723e9ceb39a203d9a576cbf8304e66

The authoritative Source of Truth is:

artifacts/sot.json

The frozen namespace is:

scientific_freeze.rq4_source_selection

Observed RQ4 results must be written only under:

scientific_analysis.rq4_source_selection

No scientific design choice may be altered in this job.

---

# PURPOSE

Execute the frozen RQ4 source-selection study and materialize all confirmatory results, validation evidence, inferential summaries, registered artifacts, table-ready values, and figure-ready values in the project Source of Truth.

This job MUST produce the empirical answer to RQ4.

This job MUST NOT revise RQ4 methodology in response to observed results.

---

# AUTHORITATIVE SCIENTIFIC CONTRACT

Before writing implementation code or reading confirmatory target outcomes:

1. load `artifacts/sot.json`;
2. verify `scientific_freeze.rq4_source_selection.status == "FROZEN"`;
3. verify `scientific_freeze.rq4_source_selection.specification_sha256 == "75ecc2d242a576655cf7e17580f737f570723e9ceb39a203d9a576cbf8304e66"`;
4. hash `docs/method_amendments/RQ4_SOURCE_SELECTION_FROZEN_SPEC.md` and verify the same SHA-256;
5. treat that frozen specification as authoritative over any implementation convenience or older code;
6. do not modify frozen scientific parameters.

If any of these checks fail:

STOP before confirmatory scoring.

Do not infer a replacement value.

Do not repair the design silently.

Return the inconsistency as a blocking methodological error.

---

# EXACT RQ4

The research question remains exactly:

RQ4 — How much target-side verification data are needed to select a source model whose future transfer error is close to the best available source, and when does the remaining selection regret become no larger than the temporal instability of the future source ranking itself?

Do not rewrite it.

---

# FROZEN SANITY CHECKS

Before execution, verify at minimum that the frozen specification still contains and the SoT agrees with:

- dataset: canonical BR-PVGen analytical data;
- target: `y_dc_normalized`;
- prohibited field: `tracker_albedo_index`;
- features in exact order:
  1. `poa_irradiance_wm2`
  2. `ghi_irradiance_wm2`
  3. `gri_irradiance_wm2`
  4. `panel_temperature_celsius`
  5. `ambient_temperature_celsius`
  6. `wind_speed_ms`;
- source lookback L = 30 calendar days;
- source split = chronological 80/20;
- primary model = SplineRidge;
- degree = 3;
- n_knots = {4, 6, 8};
- alpha = {0.1, 1, 10};
- sensitivity model = HistGradientBoostingRegressor;
- k grid = {0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45};
- K_max = 45;
- H = 45;
- cut spacing = 45 calendar days;
- candidate pool = P_ALL;
- metadata distance = Gower;
- shortlist M = 5;
- rho = 0.30;
- common support = CS4, source TRAIN only, kNN = 5, threshold quantile = 0.95;
- minimum candidates = 2;
- primary regret = relative regret;
- mandatory secondary regret = absolute regret;
- bootstrap B = 5000;
- deterministic bootstrap seed = the frozen SoT seed;
- tau primary = 0.05;
- tau sensitivity = 0.02;
- T1, T2, T3 and T4 exactly as frozen.

Any mismatch is blocking.

---

# NO EXTERNAL DATA

Use only project-authorized canonical BR-PVGen data and metadata already represented by the project and its Source of Truth.

Do not download, merge, enrich, or infer from external datasets.

---

# EXECUTION ARCHITECTURE

`orchestrator.py` is the only official executor.

Implement exactly these new stage modules unless an existing stage with the same scientific responsibility already exists and is demonstrably equivalent:

1. `src/run/12_rq4_preflight.py`
2. `src/run/13_rq4_execute.py`
3. `src/run/14_rq4_finalize.py`

Each stage must expose:

`run(ctx: StageContext) -> StageResult`

A stage must:

- never execute another stage directly;
- never write directly to `artifacts/sot.json`;
- return all SoT changes via `sot_patch`;
- return artifact records with hashes;
- return validation records;
- be deterministic for identical inputs.

Execute through:

`python orchestrator.py run --from-stage 12 --to-stage 14`

Never execute a stage module directly.

---

# STAGE 12 — PRE-SCORING FREEZE AND PREFLIGHT

Stage 12 exists to complete all structural checks that MUST occur before confirmatory target outcomes are used for source selection or future evaluation.

It must NOT compute RQ4 outcome metrics.

## 12.1 Target cohort reconstruction

Reconstruct the protected confirmatory target cohort programmatically from the canonical project partition artifact already authorized by prior stages.

Verify and persist:

- exact target IDs;
- target count;
- canonical partition artifact path;
- canonical partition artifact SHA-256;
- deterministic cohort hash computed from the ordered target IDs;
- no outcome-based membership changes.

Do not derive cohort membership from RQ4 performance.

If the canonical target cohort cannot be reconstructed unambiguously:

STOP.

## 12.2 Source cohort reconstruction and freeze

Construct the confirmatory source population only from the previously authorized development/source population.

Confirmatory target plants must not be outcome-trained sources for other confirmatory targets.

Verify and persist BEFORE scoring:

- exact source IDs;
- source count;
- source-population provenance;
- canonical source-population artifact/path;
- source-population artifact SHA-256 when available;
- deterministic cohort hash from the ordered source IDs;
- target/source disjointness.

Do not choose sources using RQ4 outcomes.

If the authorized development/source population cannot be identified unambiguously:

STOP.

## 12.3 Cohort-resolution persistence

Do not overwrite or reinterpret frozen design choices.

If exact cohort IDs/hashes were not previously materialized, add an immutable structural-resolution object under:

`scientific_freeze.rq4_source_selection.cohort_resolution`

containing at minimum:

- resolved_before_scoring = true;
- target_ids;
- target_count;
- target_ids_sha256;
- target_partition_artifact_path;
- target_partition_artifact_sha256;
- source_ids;
- source_count;
- source_ids_sha256;
- source_population_provenance;
- source_population_artifact_path;
- source_population_artifact_sha256 where available;
- target_source_disjoint = true;
- bootstrap_seed;
- resolution_stage = "12";
- no_outcomes_used = true.

This is structural materialization of the already-frozen cohort contract, not a new scientific choice.

If exact cohort resolution would require an outcome-informed decision:

STOP.

## 12.4 Eligibility and schema preflight

Verify without redesign:

- exact six-feature order;
- target availability;
- `tracker_albedo_index` excluded from every prohibited role;
- canonical eligibility semantics available;
- interpolation TRUE excluded;
- interpolation UNKNOWN not silently FALSE;
- required timestamp semantics available;
- required technical metadata fields available for Gower;
- model libraries support the frozen primary and sensitivity estimators.

## 12.5 Preflight artifact

Create:

`artifacts/rq4/RQ4_PREFLIGHT.json`

It must contain all gates, exact cohort IDs/hashes, frozen-spec hash, source/target counts, feature-order verification, prohibited-field verification, bootstrap seed, and a single final preflight decision:

`GO_FOR_CONFIRMATORY_SCORING`

or

`STOP_BEFORE_SCORING`

Stage 13 may proceed only when Stage 12 returns GO.

---

# STAGE 13 — CONFIRMATORY RQ4 EXECUTION

Implement the full frozen RQ4 design.

Do not change any scientific parameter.

## 13.1 Decision instances

Fundamental unit:

(target plant j, decision cut c)

Cuts must be deterministic and satisfy the frozen 45-day spacing and non-overlapping primary future windows within target.

Cut eligibility must use only frozen structural rules.

Do not select cuts using predictive performance.

## 13.2 Source model training

For every source s and cut c:

`SOURCE_WINDOW = [c - 30 days, c)`

Within it:

- first 80% chronological = TRAIN;
- final 20% = VALIDATION.

No source observation at or after c may enter:

- model fitting;
- preprocessing;
- hyperparameter selection;
- CS4 construction;
- any source-derived scaling.

Primary model:

SplineTransformer → StandardScaler → Ridge

with frozen grid and deterministic tie rule.

Record selected hyperparameters for every source/cut.

## 13.3 HGB sensitivity

Run HistGradientBoostingRegressor only as the predefined sensitivity analysis.

Use the exact frozen grid and fixed values.

HGB must not replace SplineRidge as primary regardless of results.

## 13.4 Metadata distance and shortlist

Use only the frozen Gower metadata rule and exact metadata fields.

Use the deterministic missing-field treatment already specified in the frozen document.

For every target:

- compute Gower distance to all valid source candidates;
- sort ascending;
- retain M = 5, or all if fewer than 5;
- record exact-twin presence;
- preserve analytic expected regret for minimum-distance ties at k=0;
- use lexicographically smallest source ID only for display identity.

No outcome may enter metadata distance or shortlist construction.

## 13.5 CS4

For every source/cut, build CS4 from source TRAIN only:

- medians;
- IQRs;
- robust standardization;
- Euclidean nearest-neighbor support;
- kNN = 5;
- leave-one-out fifth-neighbor distances;
- threshold = 95th percentile.

Invalid/degenerate IQR:

source/cut unavailable.

No ad hoc replacement.

## 13.6 Coverage

Use rho = 0.30 exactly.

Use the feature-only coverage domain defined by the frozen specification.

Persist numerator, denominator and coverage for audit.

## 13.7 Common rows

For every retained candidate set:

- W_k comparisons must use the intersection of supported target rows across candidates;
- R comparisons must use the intersection of supported target rows across candidates.

Pairwise candidate-specific target samples are prohibited for source ranking.

Persist row-set fingerprints so identical-row invariants can be validated.

## 13.8 Verification windows

Use exactly:

k = 0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45

For k > 0:

`W_k = [c, c+k)`

For k = 0:

metadata only.

No new k values.

## 13.9 Future window

Use exactly:

`R = [c+45, c+90)`

R must be identical across all k for a decision instance.

No R outcome may enter source selection.

## 13.10 Candidate minimum

At least two candidate sources are required for regret estimation.

Instances with fewer than two candidates:

- retain in structural accounting;
- mark non-informative;
- exclude from primary regret estimation;
- report missing reason.

## 13.11 Selection and future loss

For k > 0:

select by minimum MAE on common W_k rows.

Exact verification ties:

1. lower Gower distance;
2. lexicographically smaller source ID.

No refitting, fine-tuning, recalibration, or adaptation.

Future losses:

`L_s(R) = MAE_s(R)`

Oracle:

`s* = argmin_s L_s(R)`

`L*(R) = min_s L_s(R)`

## 13.12 Regret

Always materialize:

- Reg_abs(k);
- Reg_rel(k), only when L*(R) is finite and >0;
- invalid relative-regret denominator count;
- Reg_random;
- Reg_metadata;
- oracle regret = 0;
- ExcessNoise(k).

No epsilon.

## 13.13 Temporal noise floor

Within each R:

alternate seven-day blocks A, B, A, B, ...

Compute both:

- A selection → B regret;
- B selection → A regret.

Then:

`NoiseFloor_abs = mean(Reg_A_to_B, Reg_B_to_A)`

and valid relative counterpart.

Also perform the frozen chronological-half sensitivity.

Do not call the noise floor theoretically irreducible.

## 13.14 Required instance-level artifacts

Create deterministic, machine-readable artifacts under `artifacts/rq4/`.

At minimum:

- `RQ4_TARGET_COHORT.json`
- `RQ4_SOURCE_COHORT.json`
- `RQ4_SOURCE_MODELS.parquet`
- `RQ4_DECISION_INSTANCES.parquet`
- `RQ4_CANDIDATE_LOSSES.parquet`
- `RQ4_SUPPORT_COVERAGE.parquet`
- `RQ4_REGRET_BY_INSTANCE.parquet`
- `RQ4_TEMPORAL_NOISE.parquet`
- `RQ4_HGB_SENSITIVITY.parquet`

If predictions or row-level support masks are necessary for reproducibility, persist them as registered Parquet/Arrow artifacts rather than embedding them in the SoT.

---

# STAGE 14 — AGGREGATION, INFERENCE, RESULTS AND SoT

Stage 14 consumes only registered Stage-13 artifacts and the frozen protocol.

It must not make new scientific choices.

## 14.1 Target-balanced aggregation

For every primary estimand:

1. summarize within target;
2. give every target equal fleet-level weight.

Do not let targets with more valid cuts dominate.

## 14.2 Bootstrap

Use:

B = 5000

Resampling unit:

target plant.

When a target is sampled, all of its valid decision instances travel with it.

Use the frozen deterministic seed.

Use 95% percentile intervals.

Do not change B or seed based on results.

Persist bootstrap draws or an equivalently complete deterministic registered artifact sufficient to reproduce every reported CI.

Recommended path:

`artifacts/rq4/RQ4_BOOTSTRAP_DRAWS.parquet`

## 14.3 T1

Compute exactly:

`Delta_meta = Reg_random - Reg_metadata`

Report:

- target-balanced mean;
- target-balanced median;
- 95% target-cluster bootstrap CI;
- frozen interpretation rule.

Do not claim metadata improvement unless the CI for the primary mean is fully above zero.

## 14.4 T2

At k = 45 only:

`Delta_verify = Reg_metadata - Reg_45`

Report:

- target-balanced mean;
- median;
- 95% target-cluster bootstrap CI;
- frozen interpretation rule.

Do not substitute another k.

## 14.5 T3

For every frozen k:

estimate target-balanced median Reg_rel(k) and 95% CI.

Compute persistent `k_tau` exactly:

earliest k with upper95CI <= 0.05 and the condition remaining true at every later k.

Also calculate the frozen tau = 0.02 sensitivity.

If no k qualifies, report that explicitly.

Do not extend K_max.

## 14.6 T4

For every frozen k:

estimate target-balanced median ExcessNoise(k) and 95% CI.

Compute persistent `k_noise` exactly:

earliest k with upper95CI <= 0 and persistence at every later k.

If none qualifies, report that explicitly.

## 14.7 T3/T4 joint interpretation

Apply only the four frozen cases from the specification.

Do not create a fifth case.

Do not redesign after seeing the result.

## 14.8 Mandatory secondary outputs

Materialize exactly the frozen secondary outputs:

- p75 relative regret;
- p90 relative regret;
- absolute regret;
- top-1 future oracle recovery;
- source-ranking Spearman correlation where >=3 candidates;
- exact-twin candidate present vs absent;
- same-state descriptive comparison;
- structure-type descriptive comparison;
- seasonal descriptive comparison;
- chronological temporal-drift diagnostic;
- HGB sensitivity.

Do not promote any secondary output into a new primary hypothesis.

## 14.9 Primary result artifacts

Create at minimum:

- `artifacts/rq4/RQ4_PRIMARY_CURVE.parquet`
- `artifacts/rq4/RQ4_PRIMARY_TABLE.csv`
- `artifacts/rq4/RQ4_SECONDARY_RESULTS.parquet`
- `artifacts/rq4/RQ4_SUMMARY.json`
- `artifacts/rq4/RQ4_VALIDATION.json`
- `artifacts/rq4/RQ4_MANIFEST.json`

Generate the central RQ4 figure from registered result artifacts, not manually typed values.

Create:

`artifacts/rq4/RQ4_PRIMARY_FIGURE.png`

The figure must follow the frozen figure specification.

No scientific number in the table or figure may be typed manually.

---

# SoT RESULT CONTRACT

Write empirical results only under:

`scientific_analysis.rq4_source_selection`

Do not put empirical outcomes inside `scientific_freeze.rq4_source_selection`.

The analysis namespace must contain, at minimum:

- status;
- protocol_specification_path;
- protocol_specification_sha256;
- execution_commit;
- target_cohort;
- source_cohort;
- target_count;
- source_count;
- decision_instance_total;
- decision_instance_informative;
- decision_instance_noninformative;
- noninformative_reason_counts;
- candidate_count_distribution;
- invalid_relative_regret_count;
- primary_model;
- sensitivity_model;
- k_grid;
- primary_curve;
- random_baseline;
- metadata_baseline;
- oracle_baseline;
- temporal_noise_reference;
- T1;
- T2;
- T3;
- T4;
- T3_T4_joint_case;
- tau_primary;
- tau_sensitivity;
- k_tau;
- k_tau_strict;
- k_noise;
- secondary_summary;
- HGB_sensitivity_summary;
- validation_summary;
- artifact_manifest_path;
- artifact_manifest_sha256;
- no_design_change_after_outcomes;
- completed_at_utc.

`primary_curve` must contain one structured entry for every frozen k with at least:

- k;
- valid_targets;
- valid_decision_instances;
- median_candidate_count;
- median_relative_regret;
- relative_regret_ci95_low;
- relative_regret_ci95_high;
- p75_relative_regret;
- p90_relative_regret;
- median_absolute_regret;
- top1_oracle_recovery;
- rank_spearman;
- invalid_relative_regret_count.

If a value is undefined under the frozen rules, store null plus a machine-readable missing reason.

Do not fabricate values to complete the schema.

---

# LARGE ARTIFACT REGISTRATION

For every Parquet/Arrow/CSV/JSON/figure artifact used to support manuscript results, register in the SoT:

- path;
- SHA-256;
- row count when tabular;
- schema fingerprint when tabular;
- byte size;
- deterministic role/purpose;
- summaries used for manuscript reporting when applicable.

No manuscript value may depend on an unregistered artifact.

No manual reconstruction of a reported number is allowed.

---

# VALIDATION INVARIANTS

Implement and execute all 51 validation tests frozen in the RQ4 specification.

Do not merely copy their names.

Each validation must have:

- name;
- passed boolean;
- observed evidence;
- artifact/path or computed value supporting the check.

At minimum explicitly verify:

1. target cohort reconstructed deterministically;
2. target cohort hash;
3. source cohort reconstructed deterministically;
4. source cohort hash;
5. target != source;
6. source TRAIN strictly before c;
7. source VALIDATION strictly before c;
8. chronological 80/20;
9. no target outcome in source model selection;
10. feature order exact;
11. eligibility exact;
12. tracker_albedo_index absent;
13. SplineRidge primary;
14. spline grid exact;
15. HGB grid exact;
16. L=30;
17. H=45;
18. K_max=45;
19. k grid exact;
20. cuts deterministic;
21. non-overlapping primary R within target;
22. W_k and R disjoint;
23. same R across k;
24. P_ALL;
25. Gower primary;
26. M=5;
27. rho=0.30;
28. CS4 source TRAIN only;
29. kNN=5;
30. threshold quantile=0.95;
31. same target rows across candidates;
32. candidate count >=2 for regret;
33. random baseline analytic;
34. metadata-tie expectation analytic;
35. verification tie deterministic;
36. oracle on same future rows;
37. no epsilon;
38. invalid denominators missing;
39. alternating 7-day blocks deterministic;
40. A→B and B→A;
41. chronological-half sensitivity deterministic;
42. target-balanced aggregation before fleet estimate;
43. target-cluster bootstrap;
44. B=5000;
45. deterministic frozen seed;
46. persistent k_tau exact;
47. persistent k_noise exact;
48. no per-k significance fishing;
49. HGB does not replace primary;
50. deterministic rerun outputs;
51. manuscript values derived only from SoT/registered artifacts.

A GO result with any failed required validation is prohibited.

---

# DETERMINISTIC RERUN CHECK

Scientific result artifacts must be deterministic for identical inputs and frozen seed.

After producing the final registered scientific artifacts, rerun the deterministic result-generation path or an equivalent reproducibility check and verify that scientific artifact hashes and primary SoT summaries are identical.

Operational timestamps and run IDs may differ and must not be included in scientific-content hashes.

Record this evidence in `RQ4_VALIDATION.json`.

---

# ZERO-INFORMATION / NULL RESULTS

Do not redesign if the study produces weak, null, adverse, or structurally limited results.

If some targets/cuts are non-informative:

record and exclude only according to frozen rules.

If no valid decision instance survives:

do not alter M, rho, k, support, candidate pool, models, windows, or cohorts.

Materialize a completed structural result documenting zero informative instances and why the frozen protocol could not estimate primary regret.

This is not permission to invent estimands.

---

# CLAIM DISCIPLINE

Interpret results only within the frozen claim guardrails.

Do not claim:

- a universal number of target days for PV plants;
- causal source superiority;
- causal meaning of Gower distance;
- universal superiority of Gower;
- universal performance outside BR-PVGen;
- theoretical irreducibility of the empirical noise floor;
- elimination of future adaptation/recalibration needs.

Use the exact frozen null-result interpretations.

---

# PROMPT ARCHIVAL

Save this exact prompt as:

`docs/prompts/EXECUTE_RQ4_SOURCE_SELECTION_CONFIRMATORY.md`

The archived prompt must preserve all instructions in this task.

Register its SHA-256 in the SoT artifact registry.

---

# VERSION CONTROL

Before commit:

1. inspect git diff;
2. confirm no frozen scientific parameter was outcome-modified;
3. confirm empirical results appear only under `scientific_analysis.rq4_source_selection`;
4. confirm any addition under `scientific_freeze.rq4_source_selection.cohort_resolution` was created before scoring and contains only structural cohort resolution;
5. confirm all result artifacts are registered;
6. confirm no external data were added;
7. confirm no unrequested next research question or redesign was introduced.

Commit and push to the configured repository.

Suggested commit message:

Execute frozen RQ4 source-selection analysis

---

# DELIVERABLES

Return:

- preflight status;
- frozen spec path and verified SHA-256;
- target cohort count and hash;
- source cohort count and hash;
- stage paths created;
- exact execution command;
- number of valid targets;
- total decision instances;
- informative decision instances;
- non-informative decision instances;
- candidate-count p25/median/p75;
- T1 estimate and 95% CI;
- T2 estimate and 95% CI;
- k_tau at tau=0.05;
- k_tau at tau=0.02;
- k_noise;
- T3/T4 joint case;
- temporal noise-floor summary;
- primary result artifact paths/hashes;
- validation status for all 51 invariants;
- confirmation that no frozen design parameter changed after outcomes;
- SoT namespace written;
- final commit SHA;
- push status.

---

# SUCCESS MESSAGE

If and only if all required execution stages complete and all required validations pass, return exactly:

RQ4_SOURCE_SELECTION_RESULTS_MATERIALIZED

If execution is structurally complete but primary estimands are undefined because zero informative instances survive the frozen design, return exactly:

RQ4_SOURCE_SELECTION_RESULTS_STRUCTURALLY_NONINFORMATIVE

If a required pre-scoring or methodological invariant fails, do not score past the failure and return a blocking result explaining the exact invariant.

Do not modify the frozen protocol to rescue the result.

Do not start any subsequent research question or redesign in this job.