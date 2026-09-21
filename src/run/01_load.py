"""01 — verify official BR-PVGen files and write a load manifest."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.contracts import StageContext, StageResult
from src.lib import load_raw
from src.run._ledger import adopt


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    result = load_raw.run(ctx)
    man_src = ctx.root / "artifacts/p02a/P02A_MANIFEST.json"
    if man_src.is_file():
        dest = ctx.root / "artifacts/load/MANIFEST.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(man_src, dest)
    payload = {"status": result.status, "message": result.message}
    if man_src.is_file():
        payload["manifest"] = "artifacts/load/MANIFEST.json"
        try:
            man = json.loads(man_src.read_text(encoding="utf-8"))
            payload["summary"] = {"metadata_rows": (man.get("family_totals") or {}).get("metadata_rows")}
        except json.JSONDecodeError:
            pass
    return adopt(result, "load", now, payload)
