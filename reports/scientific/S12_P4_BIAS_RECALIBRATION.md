# S12 P4 — bias/dispersion and VALIDATION-only recalibration

Secondary analysis. Accepted primary OTG/RQ1–RQ3/S11/P1–P3 are not replaced. Residual sign is `prediction - y_true`. Centered MAE/RMSE are not an additive MAE decomposition.

Status: `S12_P4_RECALIBRATION_MATERIALIZED`. Universe: 85 primary computable directions (planned 85). Offset-eligible 85; affine-eligible 85; affine ineligible 0.

Headline plant-balanced OTG: k=0 0.009723604857785641; k=1 0.0041222054933101855; k=2 -0.000946480273195094. PB Delta_OTG offset 0.005601399364475456 (53/85 improve, 32/85 worsen); PB Delta_OTG affine 0.010670085130980736 (61/85 improve, 24/85 worsen).
The ratio 1 - PB_OTG_k2/PB_OTG_k0 = 1.0973384137917999 overshoots 1 because PB_OTG_k2 is negative; it is not a literal share of penalty removed. The k=1 ratio 0.5760620105814402 is optional fleet-level context only.

Median |delta| 0.027402493349015244; median |b-1| 0.061955366855692695; extreme finite slopes 0.
Spearman |MBE_transfer| vs OTG_0 rho=0.41229235880398674 CI [0.2182650297827801, 0.6331270574983107].
RQ2 plant-balanced |A|: k0 0.026622430133826737 (n=38); k1 0.015496536535324687 (n=38); k2 0.01175731889409727 (n=38).
RQ3: `P4_RQ3_NOT_RUN_RECALIBRATION_SCOPE`.

Interpretation: Offset correction reduces plant-balanced OTG from 0.009723604857785641 to 0.0041222054933101855 (PB Delta_OTG 0.005601399364475456; 53/85 directions improve, 32/85 worsen). Affine correction reduces it further to -0.000946480273195094 (PB Delta_OTG 0.010670085130980736; 61/85 improve, 24/85 worsen), moving fleet-level plant-balanced OTG slightly below zero. RQ2 plant-balanced |A| is 0.026622430133826737 (k=0), 0.015496536535324687 (k=1), 0.01175731889409727 (k=2) on 38 same-level pairs. Median |delta| 0.027402493349015244; median |b-1| 0.061955366855692695. The ratio 1 - PB_OTG_k2/PB_OTG_k0 = 1.0973384137917999 is not a share of penalty removed: PB_OTG_k2 is negative, so the ratio overshoots 1. Headline comparisons use absolute plant-balanced OTG and Delta_OTG, not that ratio.
Descriptors: offset correction removes a substantial portion of the fleet-level residual penalty; affine correction adds substantial reduction beyond offset and moves plant-balanced OTG slightly below zero; response remains heterogeneous because nontrivial sets of directions worsen under both corrections.
No arbitrary numeric interpretation gates (threshold_rules_used=False).
Model actions: {'PS_002': 'reused_p3', 'PS_003': 'reused_p3', 'PS_035': 'reused_p3', 'PS_039': 'reused_p3', 'PS_042': 'reused_p3', 'PS_043': 'reused_p3', 'PS_044': 'reused_p3', 'PS_045': 'reused_p3', 'PS_046': 'reused_p3', 'PS_047': 'reused_p3', 'PS_048': 'reused_p3', 'PS_049': 'reused_p3', 'PS_050': 'reused_p3', 'PS_051': 'reused_p3'}. Equivalence max |Δpred|=4.393263530744207e-12, max |ΔOTG|=7.931155732165962e-14.
Reconciliation max |Δ|=7.931155732165962e-14 ok=True.
