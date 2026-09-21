"""07 — Operational Twin Gap (RQ1) and directional asymmetry (RQ2)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.contracts import StageContext, StageResult
from src.lib import otg
from src.run._ledger import adopt


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    result = otg.run(ctx)
    summary_path = ctx.root / "artifacts/s07/S07_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    payload = {
        "status": result.status,
        "message": result.message,
        "summary": summary,
        "directional": "artifacts/s07/S07_OTG_DIRECTIONAL.csv",
        "asymmetry": "artifacts/s07/S07_ASYMMETRY.csv",
    }
    return adopt(result, "rq1_rq2", now, payload)
