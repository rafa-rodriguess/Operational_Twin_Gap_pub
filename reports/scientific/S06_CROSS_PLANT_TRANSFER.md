# S06 — Cross-plant transfer

Zero-shot application of frozen S04 full models to S05 CS4-supported target TEST rows. Local-reference metrics use the same timestamps. CS3 transfer metrics were not calculated. Difference-of-MAE / directional-matrix construction is deferred.

Directed transfers: 94 expected; 85 CS4-computable with predictions; 9 non-computable with no prediction rows. Prediction rows: 74330 (CS4 supported totals × 2 families).

SplineRidge transfer MAE (n=85): min 0.0316825504910523, median 0.0684385128181217, max 0.19683393639506486.
HistGradientBoosting transfer MAE (n=85): min 0.034886454518341256, median 0.06446816851543244, max 0.18675034660649797.

Local-same-rows metrics are stored alongside transfer metrics in `S06_TRANSFER_METRICS.csv` without subtraction. Non-computable transfers retain S05 reasons in `S06_TRANSFER_STATUS.csv`.

**GO — S06 CROSS-PLANT TRANSFER COMPLETE**
