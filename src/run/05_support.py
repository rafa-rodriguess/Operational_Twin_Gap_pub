"""05 — source-TRAIN common support (CS4 primary, CS3 robustness)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.contracts import StageContext, StageResult
from src.lib import support
from src.run._ledger import adopt


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    result = support.run(ctx)
    payload = {
        "status": result.status,
        "message": result.message,
        "support_summary": "artifacts/s05/S05_SUPPORT_SUMMARY.csv",
        "support_masks": "artifacts/s05/S05_SUPPORT_MASKS.parquet",
        "protocol": "artifacts/s05/S05_COMMON_SUPPORT_PROTOCOL.json",
    }
    return adopt(result, "support", now, payload)
