# P02C — Plant-level panel feasibility

P02C decision: **P02C PASS**

## 1. Inputs and provenance

- `BR-PVGen_metadata.csv` SHA-256 `ba45f4dfb557bbfca37c3ee16c8b7d2a5f0aec9fad4e6b6774be760c4852237c`
- `BR-PVGen_inverter.zip` SHA-256 `58ec09c4da8e8d61659bda5dd9db5fe27ebaea7bc547117464e910b7aaf902c6`
- `BR-PVGen_meteorological.zip` SHA-256 `a4b23123c16bf6a97a8d282e6ad18c2e697ac990d0b96602c68c6b2abfce240f`
- `Readme.txt` SHA-256 `4e8ec256c926b32cea8e8e1f2b9fb334f75cedd391eca570cdd229006add8a20`
- P02A and P02B prerequisites GO/PASS.
- Scope: 51 official plants; exact-twin members in P02B: 30.

## 2. Aggregation rule

- Inverter rows aggregate to plant × official datetime token by summing valid DC/AC only.
- Valid power: numeric, finite, and >= 0 (P02A impossible-value semantics). No upscaling of partial sums.
- `observed_reference_inverter_count` = max over timestamps of distinct inverter IDs observed. Not installed inverter count.

## 3. Target construction

- `p_dc_plant_w` / `p_ac_plant_w` = sums of valid inverter values; NULL if none valid.
- `p_nominal_w = nominal_power_mw * 1_000_000`.
- `y_dc_normalized` is the primary target candidate; `y_ac_normalized` is robustness-only. No clipping at 1.

## 4. Meteorological join

- Exact `(ps_id, datetime)` token match. No resampling, interpolation, or timezone conversion.
- Readme does not define a timezone conversion; source tokens are preserved.
- Canonical panel is the union of inverter-aggregate and meteorological timestamps with `join_status`.

## 5. Missingness / interpolation / coverage

- See `artifacts/p02c/P02C_PLANT_SUMMARY.csv` and `P02C_QC_ATTRITION.csv`.
- Empty official interpolation flags remain unknown; they are not treated as false.
- No coverage threshold is frozen.

## 6. Candidate feature availability

- Candidates: POA, GHI, GRI, panel temperature, ambient temperature, wind speed, wind direction, precipitation.
- `tracker_albedo_index` is absent from the scientific panel. `battery_voltage` is not a candidate.

## 7. Daylight structural audit

- Rows POA>20: 487977; POA>50: 453614; POA>100: 414718.
- P02C does not choose a primary daylight threshold.

## 8. Exact-twin survivability

- Twin members with DC target + meteorology overlap: 30.
- Groups with all members constructible: 5. Partial loss: False.
- `TW_04b9f5d95694` p02b_n=2 target+meteo members=2 loss=False
- `TW_4467a039e00b` p02b_n=10 target+meteo members=10 loss=False
- `TW_7b33a493697b` p02b_n=8 target+meteo members=8 loss=False
- `TW_88a108921af7` p02b_n=2 target+meteo members=2 loss=False
- `TW_f8e8375afe38` p02b_n=8 target+meteo members=8 loss=False

## 9. Limitations discovered

- Official metadata has no installed inverter count; coverage uses an observed operational reference.
- Official interpolation-flag cells are often empty; those cells stay unknown.

## 10. Gate verdict

**P02C PASS**

