# S12 P1 — Within-plant null calibration

Secondary calibration only. Accepted primary OTG / RQ1–RQ3 / S11 / source-selection are unchanged and are not training-size matched to these half-history placebos.

Status: `S12_P1_NULL_CALIBRATION_MATERIALIZED`. Interpretation: **substantial_overlap**.

OTG_half sits above placebo medians but does not exceed both P95 thresholds; residual-penalty interpretation should be qualified.

## Placebo (one value per computable plant)

- Chronological: n=13/14, median=0.00229094872427809, P75=0.007514574189506282, P95=0.01988578615716107, min=-0.015897712552686724, max=0.032311976466878375.
- Interleaved 7-day: n=13/14, median=-0.0012576642802873683, P75=0.0022758132155462282, P95=0.012730800624211034, min=-0.022947509377177516, max=0.02330629525170346.
- Paired chrono−interleaved: n=13, median=0.003548613004565458.

## OTG_half (training-size-matched real transfer)

- Expected directional=94, computable=85, noncomputable=9.
- Directional mean=0.013318089969005041, median=0.006416486637765409, plant-balanced mean=0.01088498434655154 (n_targets=14).
- P75=0.017220796436985618, P95=0.1000888936708523, min=-0.03295536758253015, max=0.1216721297838952.
- Support retention among computable: median=0.5277777777777778.

## Null exceedance (OTG_half vs placebo P95, not full-TRAIN primary OTG)

- P95 chrono=0.01988578615716107; 18 / 85 = 0.21176470588235294.
- P95 interleaved=0.012730800624211034; 26 / 85 = 0.3058823529411765.

## Target-balanced sensitivity

{
  "directional_unweighted_mean": 0.013318089969005041,
  "mean_per_target_exceedance_frac_p95_chrono": 0.22222222222222224,
  "mean_per_target_exceedance_frac_p95_interleaved": 0.2926587301587302,
  "n_targets": 14,
  "note": "Plant-balanced = equal weight per target (RQ1 synthesis). Unweighted directional mean can be G2-heavy.",
  "plant_balanced_G2_only": 0.013719831673044542,
  "plant_balanced_all_computable_targets": 0.01088498434655154,
  "plant_balanced_iterate_path": 0.01088498434655154,
  "plant_balanced_non_G2": 0.003797866030319036
}

Independent reconciliation max |Δ| = 0.0.
