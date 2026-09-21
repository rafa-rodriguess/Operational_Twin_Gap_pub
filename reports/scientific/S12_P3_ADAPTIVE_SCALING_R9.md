# S12 P3 — R9 adaptive scaling sensitivity

Sensitivity arm only. IQR → 1.4826×MAD → omit feature from the support metric. Predictive SplineRidge features are unchanged. Accepted primary protocol is not replaced.

Status: `S12_P3_R9_ADAPTIVE_SCALING_MATERIALIZED`.

Computability: primary 85/94; R9 94/94; recovered 9; lost 0; PS_048 recovered 9/9.

RQ1 plant-balanced: primary 0.009723604857785641 vs R9 0.007921538431514311 (Δ -0.0018020664262713305); overlap-only n=85 plant-balanced 0.009723604857782689 (Δ -2.9524993561125257e-15).
RQ2 plant-balanced |A|: primary 0.026622430133826737 vs R9 0.02605554770989024 (Δ -0.0005668824239364982); overlap pairs 38 → 0.02662243013382678.
RQ3: `R9_RQ3_DEFERRED_NOT_REQUIRED_FOR_SCALER_SENSITIVITY`.

Support change on 85 primary identities: identical 85, R9 superset 0, subset 0, partial 0; median Jaccard 1.0, min 1.0.

R9 recovers all 9 PS_048 source directions. RQ1 plant-balanced primary 0.009723604857785641 vs R9 0.007921538431514311 (difference -0.0018020664262713305). R9 remains a sensitivity arm, not a new primary protocol.

Reconciliation max |Δ|=0.0.
