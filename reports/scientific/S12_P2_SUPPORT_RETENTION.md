# S12 P2 — OTG × support retention (R7)

Persisted-data diagnostic only. No refit and no new prediction scoring. Accepted primary RQ1–RQ3 / S11 / S12 P0–P1 unchanged.

Status: `S12_P2_SUPPORT_RETENTION_MATERIALIZED`.

## Association (n=85 computable primary directional transfers)

- Absolute OTG vs retention Spearman rho=-0.2737736955247215; descriptive p=0.011234004038787652; target-cluster bootstrap 95% percentile CI [-0.5063571487826426, -0.03486494194987956] (valid 5000/5000, seed 20260921).
- Relative OTG vs retention rho=-0.30046902481923; CI [-0.5259737461222844, -0.05081407171361724] (valid 5000/5000). Relative OTG is interpreted separately.

Interpretation: **negative_otg_retention_association**. Lower support retention tends to accompany larger absolute transfer penalties (bootstrap CI excludes 0).

## Retention terciles

Cutpoints (numpy quantile method=linear): 1/3=0.3507362784471218, 2/3=0.6816901408450704.

- LOW [min, 0.3507362784471218): n=28, targets=10, median OTG_abs=0.009356163792156636, P75 abs=0.015906247527336704, median rel=0.22115568768182808, plant-balanced abs=0.013460255204958254, groups={'TW_4467a039e00b': 28}.
- MID [0.3507362784471218, 0.6816901408450704): n=28, targets=12, median OTG_abs=0.006219077795974835, P75 abs=0.019829021077206452, median rel=0.09075277838185522, plant-balanced abs=0.011969924429543031, groups={'TW_4467a039e00b': 26, 'TW_88a108921af7': 2}.
- HIGH [0.6816901408450704, max]: n=29, targets=11, median OTG_abs=0.003812474935823658, P75 abs=0.006793773823267998, median rel=0.06064903023201766, plant-balanced abs=0.009614343325281341, groups={'TW_04b9f5d95694': 2, 'TW_4467a039e00b': 27}.

## R7 support_retention >= 0.5

- RQ1: n_dir=47, targets=12, sources=12; plant-balanced OTG=0.008740101787088115 vs primary 0.009723604857785641 (difference -0.0009835030706975261).
- RQ2: n_pairs=18; plant-balanced |asymmetry|=0.03501702437885093 vs primary 0.026622430133826737 (difference 0.008394594245024194).
- RQ3: `R7_RQ3_NOT_COMPUTABLE_FROM_PERSISTED_ARTIFACTS`. RQ3 uses pairwise TEST intersection of twin CS4 and control CS4 (n_pairwise in source_contrasts.csv) without persisted control-side or intersection retention; S07 support_retention is twin-transfer only. Filtering only the twin leg would be a one-sided proxy. No refit/new scoring in P2.

R7 RQ1 plant-balanced 0.008740101787088115 vs primary 0.009723604857785641 (signed difference -0.0009835030706975261). Support coverage is a modifier to the extent this numerical shift matters; R7 is not labeled better or worse.

Reconciliation max |Δ|=0.0.
