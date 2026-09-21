# S12 P5 — plant-delete influence and G2/non-G2 stability

Persisted primary SplineRidge+CS4 artifacts only. No refit, no new scoring, no R7/R9/P4/G3. Descriptive network influence, not an iid jackknife CI.

Status: `S12_P5_PLANT_DELETE_STABILITY_MATERIALIZED`. Full primary RQ1 0.009723604857785641 (n=85); RQ2 PB |A| 0.026622430133826737 (n_pairs=38).

14-plant RQ1 LOPO: min 0.003239152507605244, max 0.012830978674477312, median 0.010674131827326395, range 0.009591826166872068; max |shift| 0.006484452350180397 (PS_047); same-sign 14/14.
14-plant RQ2 LOPO: min 0.011603159904507117, max 0.0314187209210933, median 0.027886593115242782, range 0.019815561016586186; max |shift| 0.01501927022931962 (PS_047).

G2-only RQ1 0.015181496979951397 (n_dir 81); non-G2 pooled -0.003921125447628748; G1 0.0010783871471068823; G4 -0.008920638042364377.
G2-only RQ2 0.03647491048721349; non-G2 RQ2 0.004454349338706552.
G2-delete max |RQ1 shift| 0.008759998714464378 (PS_047).

RQ3: `P5_RQ3_NOT_RUN_STABILITY_SCOPE`.
Interpretation: Leave-one-plant-out RQ1 over 14 deletes ranges [0.003239152507605244, 0.012830978674477312] (median 0.010674131827326395; range 0.009591826166872068) versus full primary 0.009723604857785641. Largest absolute shift is 0.006484452350180397 when deleting PS_047; max positive shift 0.003107373816691671 (PS_035), max negative shift -0.006484452350180397 (PS_047). 14/14 replicates keep the sign of full RQ1. RQ2 PB |A| ranges [0.011603159904507117, 0.0314187209210933] (median 0.027886593115242782) versus full 0.026622430133826737; largest absolute shift 0.01501927022931962 (PS_047). G2-only RQ1 0.015181496979951397 (n_dir 81) versus non-G2 pooled -0.003921125447628748 (G1 0.0010783871471068823, G4 -0.008920638042364377). These are descriptive influence diagnostics on a transfer network, not iid jackknife intervals.
Descriptors: leave-one-plant-out RQ1/RQ2 remain defined for all 14 primary deletions; influence is ranked by observed absolute shift, without evaluative plant scores; G2-only versus G1/G4 baselines are subgroup contrasts, not delete-one replicates; non-G2 is a small descriptive cohort; pair estimability can fail when a two-plant group is broken.
threshold_rules_used=False. Reconciliation max |Δ|=0.0 ok=True.
