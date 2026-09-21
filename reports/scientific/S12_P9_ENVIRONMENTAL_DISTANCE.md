# S12 P9 — Environmental-distribution distance

Exploratory explanatory sensitivity only. Primary OTG remains accepted S07 zero-shot CS4 SplineRidge. No refit, no scoring, no support rebuild. P8 pooled outputs are not P9 outcomes. RQ2 environmental results are descriptive (38 pairs, shared plants).

Status: `S12_P9_ENVIRONMENTAL_DISTANCE_MATERIALIZED`.

environmental_mismatch_tracks_support_loss: ED_full is negatively associated with support retention and the target-cluster CI excludes zero. residual_mismatch vs OTG is weak or uncertain at the target-cluster bootstrap. distance_mostly_acts_through_support_loss_or_is_uncertain: partial Spearman CI includes zero or is undefined. RQ2 descriptive: Spearman A_ED vs A_OTG rho=-0.2846044424991793; sign concordance {'n_nonzero': 38, 'n_concordant': 18, 'fraction': 0.47368421052631576}. No iid CI.

## Setup

- 14 plants, 85 directions, 38 bidirectional pairs; features: poa_irradiance_wm2, ghi_irradiance_wm2, gri_irradiance_wm2, panel_temperature_celsius, ambient_temperature_celsius, wind_speed_ms.
- Source TRAIN median/IQR scaling; distances in source-standardized space.
- Headline cap 2048; sensitivity cap 1024; seed 20260921.
- Energy Distance headline; RBF MMD² corroborative.

## Associations (85 directions)

- ED_full vs retention: rho=-0.8744576900527653, CI=[-0.9205475560236458, -0.7953464547840055].
- MMD2_full vs retention: rho=-0.898045729919875, CI=[-0.9331039677418183, -0.8344504497720814].
- ED_removed vs retention: rho=0.37402775063513777.
- ED_supported vs |OTG|: rho=0.08960328317373462, p=0.4147792409661798, CI=[-0.1941528082715405, 0.36752990765185795].
- ED_supported vs rel OTG: rho=0.12468243111197967, CI=[-0.17572338279369396, 0.4109453575505093].
- MMD2_supported vs |OTG|: rho=0.10850107484854407, CI=[-0.16580216435921605, 0.371917462082444].
- Partial Spearman ED vs OTG | retention: rho=-0.29246274566286823, CI=[-0.5440692693310749, 0.021856325547176248].
- Partial Spearman MMD vs OTG | retention: rho=-0.26221709946391486, CI=[-0.5154716954656756, 0.057810537570841945].

## Sampling stability (2048 vs 1024)

- Spearman ED 0.9967559116669924; MMD 0.9975962478014462; median |ΔED|=0.015999918521719003; max |ΔED|=0.05588776045970345.

## Association stability (2048 vs 1024 caps)

Under the 1024 cap, ED_full vs retention remains strongly negative (rho_2048=-0.8744576900527653, rho_1024=-0.8607777994918898); ED_supported vs |OTG| remains small (rho_2048=0.08960328317373462, rho_1024=0.0813171780340043); partial Spearman remains modestly negative (rho_2048=-0.29246274566286823, rho_1024=-0.30366685086475015). The qualitative P9 conclusions — mismatch tracks support loss; residual distance does not clearly add beyond retention — are stable under the smaller cap. No magnitude cutoff was applied.

[
  {
    "absolute_difference": 0.013679890560875485,
    "kind": "spearman",
    "n_1024": 85,
    "n_2048": 85,
    "name": "ed_full_vs_support_retention",
    "reason_1024": null,
    "reason_2048": null,
    "rho_1024": -0.8607777994918898,
    "rho_2048": -0.8744576900527653,
    "signed_difference": 0.013679890560875485
  },
  {
    "absolute_difference": 0.012429157709595517,
    "kind": "spearman",
    "n_1024": 85,
    "n_2048": 85,
    "name": "mmd2_full_vs_support_retention",
    "reason_1024": null,
    "reason_2048": null,
    "rho_1024": -0.8856165722102795,
    "rho_2048": -0.898045729919875,
    "signed_difference": 0.012429157709595517
  },
  {
    "absolute_difference": 0.008286105139730313,
    "kind": "spearman",
    "n_1024": 85,
    "n_2048": 85,
    "name": "ed_supported_vs_otg_abs",
    "reason_1024": null,
    "reason_2048": null,
    "rho_1024": 0.0813171780340043,
    "rho_2048": 0.08960328317373462,
    "signed_difference": -0.008286105139730313
  },
  {
    "absolute_difference": 0.007641196013289039,
    "kind": "spearman",
    "n_1024": 85,
    "n_2048": 85,
    "name": "mmd2_supported_vs_otg_abs",
    "reason_1024": null,
    "reason_2048": null,
    "rho_1024": 0.10085987883525503,
    "rho_2048": 0.10850107484854407,
    "signed_difference": -0.007641196013289039
  },
  {
    "absolute_difference": 0.01120410520188192,
    "kind": "partial",
    "n_1024": 85,
    "n_2048": 85,
    "name": "partial_ed_supported_otg_abs_given_retention",
    "reason_1024": null,
    "reason_2048": null,
    "rho_1024": -0.30366685086475015,
    "rho_2048": -0.29246274566286823,
    "signed_difference": -0.01120410520188192
  },
  {
    "absolute_difference": 0.01874477724620277,
    "kind": "partial",
    "n_1024": 85,
    "n_2048": 85,
    "name": "partial_mmd2_supported_otg_abs_given_retention",
    "reason_1024": null,
    "reason_2048": null,
    "rho_1024": -0.28096187671011763,
    "rho_2048": -0.26221709946391486,
    "signed_difference": -0.01874477724620277
  }
]

## Target-balanced (n=14, caution)

{'caution': 'n=14 weighting sensitivity, not replacement for 85-direction analysis', 'n_targets': 14, 'spearman_ed_otg': {'n': 14, 'pvalue': 0.828626422928612, 'reason': None, 'rho': 0.06373626373626373, 'x': 'mean_ed_supported', 'y': 'mean_otg_abs'}, 'spearman_ed_retention': {'n': 14, 'pvalue': 0.05615392734370259, 'reason': None, 'rho': -0.5208791208791209, 'x': 'mean_ed_supported', 'y': 'mean_support_retention'}, 'spearman_mmd_otg': {'n': 14, 'pvalue': 0.899093096973967, 'reason': None, 'rho': 0.03736263736263736, 'x': 'mean_mmd2_supported', 'y': 'mean_otg_abs'}, 'spearman_mmd_retention': {'n': 14, 'pvalue': 0.04282204937819734, 'reason': None, 'rho': -0.5472527472527473, 'x': 'mean_mmd2_supported', 'y': 'mean_support_retention'}}

## Groups

- G2 n=81; non-G2 n=4 (small).

## RQ2 descriptive (38 pairs)

- Spearman A_ED vs A_OTG: {'n': 38, 'pvalue': 0.08330371635617728, 'reason': None, 'rho': -0.2846044424991793, 'x': 'a_ed', 'y': 'a_otg'}.
- Sign concordance ED: {'fraction': 0.47368421052631576, 'n_concordant': 18, 'n_nonzero': 38}.

## Scientific consequence

1. Environmental mismatch is associated with support loss and/or residual OTG only to the extent of the reported Spearmans; it does not by itself fully explain nonzero twin OTG.
2. Partial Spearman CI includes zero or is weak: residual environmental distance does not clearly add information beyond support retention.
3. RQ2 signed environmental asymmetry vs signed OTG is descriptive only (n=38 pairs, shared plants): rho=-0.2846044424991793, concordance={'n_nonzero': 38, 'n_concordant': 18, 'fraction': 0.47368421052631576}.
4. Environmental distance is not sufficient as a standalone source-selection rule on this evidence; it may complement retention diagnostics but is not a deployment criterion by itself.

P2 recon max |Δ| retention=0.0, OTG=0.0. Independent reconciliation max |Δ|=0.0.
