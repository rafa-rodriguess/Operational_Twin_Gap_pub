"""04 — local SplineRidge / HGB models and Gself."""

from __future__ import annotations

from datetime import datetime, timezone

from src.contracts import StageContext, StageResult
from src.lib import models
from src.run._ledger import adopt


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    result = models.run(ctx)
    payload = {
        "status": result.status,
        "message": result.message,
        "model_protocol": "artifacts/s04/S04_MODEL_PROTOCOL.json",
        "local_test_predictions": "artifacts/s04/S04_LOCAL_TEST_PREDICTIONS.parquet",
    }
    return adopt(result, "models", now, payload)
