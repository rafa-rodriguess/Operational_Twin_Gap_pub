#!/usr/bin/env python3
"""Compare numeric leaves of the old freeze SoT with the new ledger.

Usage (from new_repo, or pass paths):

    python scripts/compare_sot.py \\
        --old ../artifacts/sot.json \\
        --new artifacts/sot.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SKIP_KEY_PARTS = (
    "frozen",
    "freeze",
    "controller",
    "pending",
    "sha256",
    "finished_at",
    "answer",
    "job",
    "kind",
    "operational_stage",
    "last_job",
    "blinding",
    "unfrozen",
    "manifest",
    "path",
    "bytes",
    "schema_fingerprint",
    "report",
    "revision_id",
    "reason",
)


HEADLINE = {
    "old": {
        "rq3_D_fleet": [
            "scientific_analysis",
            "s08_revisions",
            "state_constrained_support_v2",
            "scientific",
            "D_fleet_RQ3",
        ],
    },
    "new": {
        "rq1_otg": ["results", "tables", "rq1"],
        "rq2_abs": ["results", "tables", "rq2"],
        "rq3_D_fleet": ["results", "rq3", "D_fleet_RQ3"],
        "n_rq3_targets": ["results", "rq3", "n_rq3_targets"],
    },
}


def nested(data: dict[str, Any], path: list[str]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def collect_numbers(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        low = prefix.lower()
        if any(part in low for part in SKIP_KEY_PARTS):
            return out
        for key, val in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(collect_numbers(val, path))
    elif isinstance(obj, bool):
        return out
    elif isinstance(obj, (int, float)) and math.isfinite(float(obj)):
        out[prefix] = float(obj)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()
    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))

    print("headline")
    pairs = [
        ("rq1 OTG", None, nested(new, HEADLINE["new"]["rq1_otg"])),
        ("rq2 |A|", None, nested(new, HEADLINE["new"]["rq2_abs"])),
        ("rq3 D_fleet", nested(old, HEADLINE["old"]["rq3_D_fleet"]), nested(new, HEADLINE["new"]["rq3_D_fleet"])),
        ("rq3 n_targets", None, nested(new, HEADLINE["new"]["n_rq3_targets"])),
    ]
    for name, a, b in pairs:
        if a is None:
            print(f"  {name}: new={b}")
        else:
            ok = a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=args.rtol, abs_tol=args.atol)
            print(f"  {name}: old={a} new={b} {'MATCH' if ok else 'DIFF'}")

    old_s07 = args.old.parent / "s07" / "S07_SUMMARY.json"
    new_s07 = args.new.parent / "s07" / "S07_SUMMARY.json"
    if old_s07.is_file() and new_s07.is_file():
        o7 = json.loads(old_s07.read_text(encoding="utf-8"))
        n7 = json.loads(new_s07.read_text(encoding="utf-8"))
        print("s07 summary")
        for key in ("n_expected_directional", "n_computable_directional"):
            print(f"  {key}: old={o7.get(key)} new={n7.get(key)}")
        for dist, stat in (("otg_rel_defined", "median"), ("otg_rel_defined", "p75"), ("otg_abs", "mean")):
            oa = ((o7.get(dist) or {}) or {}).get(stat)
            nb = ((n7.get(dist) or {}) or {}).get(stat)
            ok = oa is not None and nb is not None and math.isclose(float(oa), float(nb), rel_tol=args.rtol, abs_tol=args.atol)
            print(f"  {dist}.{stat}: old={oa} new={nb} {'MATCH' if ok else 'DIFF'}")

    old_nums = collect_numbers(old.get("scientific_analysis") or {})
    new_nums = collect_numbers(new.get("results") or {})
    print(f"numeric leaves scientific_analysis={len(old_nums)} results={len(new_nums)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
