"""09 — OFAT robustness protocol (R1–R6 recorded; R7 omitted)."""

from __future__ import annotations

from datetime import datetime, timezone

from config.protocol import R6_ORIGINS, R7_OMITTED, ROBUSTNESS_DESIGN
from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, dump_json
from src.run._ledger import ledger_patch

ARMS = ["R1_HGB", "R2_AC", "R3_POA20", "R3_POA100", "R4_CS3", "R5_keep_all_valid", "R6_O1", "R6_O2", "R6_O3"]


def run(ctx: StageContext) -> StageResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload = {
        "design": ROBUSTNESS_DESIGN,
        "arms": ARMS,
        "r6_origins": list(R6_ORIGINS),
        "r7_omitted": R7_OMITTED,
        "note": "Primary RQ1–RQ3 numbers come from stages 07–08. OFAT arm re-fits are optional follow-on compute.",
    }
    out = ctx.root / "artifacts/robustness"
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "ofat_protocol.json", payload)
    recs = [artifact(ctx.root, out / "ofat_protocol.json")]
    return StageResult(
        status="GO",
        message="GO — OFAT protocol recorded",
        sot_patch=ledger_patch(stage="robustness", now=now, payload=payload, recs=recs),
        artifacts=recs,
        validations=[ValidationRecord("ofat_protocol", True, "R1-R6 listed; R7 omitted")],
    )
