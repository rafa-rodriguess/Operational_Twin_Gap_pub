"""06 — zero-shot cross-plant transfer on supported target TEST rows."""

from __future__ import annotations

from datetime import datetime, timezone

from src.contracts import StageContext, StageResult
from src.lib import transfer
from src.run._ledger import adopt


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    result = transfer.run(ctx)
    payload = {
        "status": result.status,
        "message": result.message,
        "transfer_metrics": "artifacts/s06/S06_TRANSFER_METRICS.csv",
        "predictions": "artifacts/s06/S06_CROSS_PLANT_PREDICTIONS.parquet",
    }
    return adopt(result, "transfer", now, payload)
