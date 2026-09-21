# S12 improvement roadmap (Phase 0)

Authoritative sequencing for S12 P0–P9 unless a later orchestrator request revises it. No scientific headline was computed in P0.

## Phases

- **P0** — Pipeline audit and readiness  
  Readiness: `COMPLETE_THIS_STAGE`
- **P1** — Within-plant null calibration  
  Readiness: `REQUIRES_REFIT`
- **P2** — OTG × support retention  
  Readiness: `PERSISTED_READY`
- **P3** — Adaptive scaling sensitivity  
  Readiness: `REQUIRES_REFIT`
- **P4** — Bias/dispersion + recalibration ladder  
  Readiness: `REQUIRES_INSTRUMENTATION`
- **P5** — Concentration/stability diagnostics  
  Readiness: `PERSISTED_READY`
- **P6** — Matched training-history duration  
  Readiness: `REQUIRES_REFIT`
- **P7** — RQ3 control-dependence analysis  
  Readiness: `PERSISTED_READY`
- **P8** — Pooled fleet baseline  
  Readiness: `REQUIRES_REFIT`
- **P9** — Environmental-distribution distance  
  Readiness: `DETERMINISTIC_RECONSTRUCTION_READY`

## Reserved robustness labels

- **R7**: P2 support-retention arm (retention ≥ 0.5).
- **R8**: S11 all structurally executable same-state non-twin controls (already completed; do not reuse for TRAIN duration).
- **R9**: P3 adaptive scaling sensitivity (not automatically primary).
- **R10**: P6 matched TRAIN duration.

## S11 A1–A4

Future main-text inclusion (not yet in `paper/main.tex`): {'A1': 'INCLUDE_CORE', 'A2': 'INCLUDE_CORE', 'A3': 'INCLUDE_CORE', 'A4': 'INCLUDE_CORE'}.

After P9: one consolidated manuscript edit of accepted S12 + A1–A4, then compile, then page-count work.

## Inventory snapshot

- Predictions (primary TEST transfer/local): `PERSISTED_READY`.
- Support masks (CS4 datetime+supported): `PERSISTED_READY`.
- Splits: `PERSISTED_READY`.
- TRAIN histories: `PERSISTED_READY`.
- Hyperparameters/models: `PERSISTED_READY`.
- Scalers/tau: `DETERMINISTIC_RECONSTRUCTION_READY`.
- Row key plant_id+datetime_raw unique on panel: `True`.

## P1–P9 verdicts

- P1: `REQUIRES_REFIT`
- P2: `PERSISTED_READY`
- P3: `REQUIRES_REFIT`
- P4: `REQUIRES_INSTRUMENTATION`
- P5: `PERSISTED_READY`
- P6: `REQUIRES_REFIT`
- P7: `PERSISTED_READY`
- P8: `REQUIRES_REFIT`
- P9: `DETERMINISTIC_RECONSTRUCTION_READY`
