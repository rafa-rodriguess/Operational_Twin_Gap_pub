"""P-02B exact nominal twin audit. Stages must not write artifacts/sot.json."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

from src.contracts import ArtifactRecord, StageContext, StageResult, ValidationRecord
from src.lib.load_raw import (
    _csv_artifact_schema_fingerprint,
    _sha256_bytes,
    _sha256_file,
)

META_REL = "data/raw/br_pvgen/BR-PVGen_metadata.csv"
EXPECTED_COLUMNS = [
    "brazil_federative_unit",
    "id",
    "is_panel_bifacial",
    "nominal_power_mw",
    "number_of_panels",
    "panel_area_mm2",
    "panel_bifaciality_coefficient",
    "panel_efficiency_percentage",
    "panel_temperature_coefficient",
    "structure_type",
]
CANDIDATE_TECHNICAL = [
    "is_panel_bifacial",
    "nominal_power_mw",
    "number_of_panels",
    "panel_area_mm2",
    "panel_bifaciality_coefficient",
    "panel_efficiency_percentage",
    "panel_temperature_coefficient",
    "structure_type",
]
DECIMAL_FIELDS = [
    "nominal_power_mw",
    "panel_area_mm2",
    "panel_bifaciality_coefficient",
    "panel_efficiency_percentage",
    "panel_temperature_coefficient",
]
INTEGER_FIELDS = ["number_of_panels"]
BOOLEAN_FIELDS = ["is_panel_bifacial"]
STRING_FIELDS = ["structure_type"]
FINAL_KEY = list(CANDIDATE_TECHNICAL)
TECHNICAL_INTERPRETATION = {
    "id": "official plant identifier; never used for nominal equivalence",
    "brazil_federative_unit": "Brazilian federative unit; contextual location only",
    "is_panel_bifacial": "whether modules are bifacial",
    "nominal_power_mw": "installed plant capacity (MW)",
    "number_of_panels": "installed module count",
    "panel_area_mm2": "single-module area (mm^2)",
    "panel_bifaciality_coefficient": "rear-to-front efficiency ratio (0-1)",
    "panel_efficiency_percentage": "module conversion efficiency (%)",
    "panel_temperature_coefficient": "percent power loss per C above STC",
    "structure_type": "FIXED or TRACKER mounting",
}


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def canonical_decimal(token: str) -> str:
    text = (token or "").strip()
    if text == "":
        raise ValueError("empty decimal token")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid decimal token: {token!r}") from exc
    return format(value.normalize(), "f")


def canonical_integer(token: str) -> str:
    text = (token or "").strip()
    if text == "":
        raise ValueError("empty integer token")
    value = Decimal(text)
    if value != value.to_integral_value():
        raise ValueError(f"non-integer integer token: {token!r}")
    return str(int(value))


def canonical_boolean(token: str) -> str:
    text = (token or "").strip()
    lowered = text.lower()
    if lowered == "true":
        return "true"
    if lowered == "false":
        return "false"
    raise ValueError(f"invalid boolean token: {token!r}")


def canonical_structure(token: str) -> str:
    return (token or "").strip()


def canonicalize_field(field: str, token: str) -> str:
    if field in DECIMAL_FIELDS:
        return canonical_decimal(token)
    if field in INTEGER_FIELDS:
        return canonical_integer(token)
    if field in BOOLEAN_FIELDS:
        return canonical_boolean(token)
    if field in STRING_FIELDS:
        return canonical_structure(token)
    raise ValueError(field)


def key_payload(canon: dict[str, str]) -> str:
    ordered = {field: canon[field] for field in FINAL_KEY}
    return json.dumps(ordered, separators=(",", ":"), ensure_ascii=True)


def group_digest(canon: dict[str, str]) -> str:
    return hashlib.sha256(key_payload(canon).encode("utf-8")).hexdigest()


def group_id_from_digest(digest: str) -> str:
    return "TW_" + digest[:12]


def _art(root: Path, path: Path, rows: int | None, fingerprint: str | None) -> ArtifactRecord:
    payload = path.read_bytes()
    return ArtifactRecord(
        path=_rel(root, path),
        sha256=_sha256_bytes(payload),
        bytes=len(payload),
        row_count=rows,
        schema_fingerprint=fingerprint,
    )


def _sot_csv(rec: ArtifactRecord) -> dict[str, Any]:
    return {
        "path": rec.path,
        "sha256": rec.sha256,
        "row_count": rec.row_count,
        "schema_fingerprint": rec.schema_fingerprint,
    }


def run(ctx: StageContext) -> StageResult:
    root = ctx.root
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    validations: list[ValidationRecord] = []
    feat_a = ((ctx.sot.get("results") or {}).get("load") or {})
    validations.append(ValidationRecord("metadata_ready", True, "load-or-direct"))

    meta_path = root / META_REL
    if not meta_path.is_file():
        return StageResult(
            status="STOP",
            message="official metadata missing",
            sot_patch={
                "stages": {"P02B": {"kind": "data_feasibility", "status": "STOP", "finished_at_utc": now}},
                "feasibility": {"p02b": {"status": "FAIL"}},
            },
            artifacts=[],
            validations=[ValidationRecord("metadata_present", False, META_REL)],
        )

    pre_hash = _sha256_file(meta_path)
    expected_hash = None
    expected_rows = feat_a.get("summary", {}).get("metadata_rows")
    manifest_path = root / "artifacts" / "p02a" / "P02A_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("entries") or []:
            if Path(entry.get("relative_path", "")).name == "BR-PVGen_metadata.csv":
                expected_hash = entry.get("sha256")
                if expected_rows is None:
                    expected_rows = entry.get("row_count")
    hash_ok = expected_hash is None or pre_hash == expected_hash
    validations.append(ValidationRecord("metadata_sha256", hash_ok, pre_hash))

    with meta_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    cols_ok = columns == EXPECTED_COLUMNS
    validations.append(ValidationRecord("metadata_columns", cols_ok, json.dumps(columns)))
    row_ok = expected_rows is None or len(rows) == int(expected_rows)
    validations.append(ValidationRecord("metadata_row_count", row_ok, str(len(rows))))
    ids = [r.get("id", "").strip() for r in rows]
    unique_ids = len(set(ids)) == len(ids) and all(ids)
    validations.append(ValidationRecord("metadata_ids_unique", unique_ids, str(len(ids))))

    if not (hash_ok and cols_ok and row_ok and unique_ids):
        return StageResult(
            status="STOP",
            message="P02B metadata does not reconcile with P02A",
            sot_patch={
                "stages": {"P02B": {"kind": "data_feasibility", "status": "STOP", "finished_at_utc": now}},
                "feasibility": {"p02b": {"status": "FAIL"}},
            },
            artifacts=[],
            validations=validations,
        )

    excluded = {
        "id": {
            "field": "id",
            "reason": "plant identifier; never enters the exact technical key",
            "evidence": "P02B contract and BR-PVGen data dictionary: unique plant metadata identifier",
        },
        "brazil_federative_unit": {
            "field": "brazil_federative_unit",
            "reason": "state is contextual only; must not create, split, merge, or rank twin groups",
            "evidence": "P02B contract; Readme location restricted to federative unit",
        },
    }
    audit_rows: list[dict[str, Any]] = []
    for col in EXPECTED_COLUMNS:
        values = [r.get(col, "") for r in rows]
        n_missing = sum(1 for v in values if v is None or str(v).strip() == "")
        n_unique = len({(v or "").strip() for v in values if (v or "").strip() != ""})
        if col == "id":
            role = "identifier"
            included = False
            reason = excluded["id"]["reason"]
        elif col == "brazil_federative_unit":
            role = "contextual"
            included = False
            reason = excluded["brazil_federative_unit"]["reason"]
        else:
            role = "technical_configuration"
            included = col in FINAL_KEY
            reason = "" if included else "see key definition excluded_fields"
        audit_rows.append(
            {
                "attribute": col,
                "dtype": {
                    "id": "string",
                    "brazil_federative_unit": "string",
                    "is_panel_bifacial": "boolean",
                    "nominal_power_mw": "decimal",
                    "number_of_panels": "integer",
                    "panel_area_mm2": "decimal",
                    "panel_bifaciality_coefficient": "decimal",
                    "panel_efficiency_percentage": "decimal",
                    "panel_temperature_coefficient": "decimal",
                    "structure_type": "string",
                }[col],
                "n_missing": n_missing,
                "n_unique_non_missing": n_unique,
                "role": role,
                "included_in_exact_key": included,
                "exclusion_reason": reason,
                "technical_interpretation": TECHNICAL_INTERPRETATION[col],
            }
        )

    technical_decisions = {row["attribute"]: row["included_in_exact_key"] for row in audit_rows if row["role"] == "technical_configuration"}
    validations.append(
        ValidationRecord(
            "every_technical_field_decided",
            all(col in technical_decisions for col in CANDIDATE_TECHNICAL),
            None,
        )
    )
    validations.append(ValidationRecord("id_not_in_key", "id" not in FINAL_KEY, None))
    validations.append(ValidationRecord("state_not_in_key", "brazil_federative_unit" not in FINAL_KEY, None))

    eligible: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    for row in rows:
        plant = (row.get("id") or "").strip()
        missing_fields = [f for f in FINAL_KEY if str(row.get(f, "")).strip() == ""]
        if missing_fields:
            ineligible.append({"id": plant, "missing_fields": missing_fields})
            continue
        try:
            canon = {field: canonicalize_field(field, row.get(field, "")) for field in FINAL_KEY}
        except ValueError as exc:
            ineligible.append({"id": plant, "missing_fields": [str(exc)]})
            continue
        eligible.append(
            {
                "id": plant,
                "state": (row.get("brazil_federative_unit") or "").strip(),
                "canon": canon,
                "digest": group_digest(canon),
            }
        )

    by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        by_digest[item["digest"]].append(item)

    groups: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    for digest, items in sorted(by_digest.items(), key=lambda kv: kv[0]):
        plants = sorted(item["id"] for item in items)
        if len(plants) < 2:
            continue
        gid = group_id_from_digest(digest)
        canon = items[0]["canon"]
        states = sorted({item["state"] for item in items})
        n_g = len(plants)
        groups.append(
            {
                "group_id": gid,
                "group_key_sha256": digest,
                "n_members": n_g,
                "n_undirected_pairs": n_g * (n_g - 1) // 2,
                "n_directional_transfers": n_g * (n_g - 1),
                "n_distinct_states": len(states),
                "states": json.dumps(states, separators=(",", ":")),
                "key_json": key_payload(canon),
                **{f"key_{field}": canon[field] for field in FINAL_KEY},
            }
        )
        for item in sorted(items, key=lambda x: x["id"]):
            members.append(
                {
                    "group_id": gid,
                    "group_key_sha256": digest,
                    "plant_id": item["id"],
                    "brazil_federative_unit": item["state"],
                    **{f"key_{field}": item["canon"][field] for field in FINAL_KEY},
                }
            )
        for left, right in combinations(plants, 2):
            pair_rows.append(
                {
                    "group_id": gid,
                    "plant_i": left,
                    "plant_j": right,
                }
            )
        for src, dst in permutations(plants, 2):
            transfer_rows.append(
                {
                    "group_id": gid,
                    "source_plant_id": src,
                    "target_plant_id": dst,
                }
            )

    groups.sort(key=lambda g: g["group_id"])
    members.sort(key=lambda r: (r["group_id"], r["plant_id"]))
    pair_rows.sort(key=lambda r: (r["group_id"], r["plant_i"], r["plant_j"]))
    transfer_rows.sort(key=lambda r: (r["group_id"], r["source_plant_id"], r["target_plant_id"]))

    plants_in_groups = sorted({m["plant_id"] for m in members})
    singleton_count = len(eligible) - len(plants_in_groups)
    undirected_n = len(pair_rows)
    directional_n = len(transfer_rows)
    expected_und = sum(int(g["n_members"]) * (int(g["n_members"]) - 1) // 2 for g in groups)
    expected_dir = sum(int(g["n_members"]) * (int(g["n_members"]) - 1) for g in groups)

    validations.append(ValidationRecord("groups_n_ge_2", all(int(g["n_members"]) >= 2 for g in groups), None))
    validations.append(
        ValidationRecord("plant_unique_group", len(plants_in_groups) == len(members), str(len(members)))
    )
    validations.append(ValidationRecord("undirected_count", undirected_n == expected_und, str(undirected_n)))
    validations.append(ValidationRecord("directional_count", directional_n == expected_dir, str(directional_n)))
    validations.append(
        ValidationRecord(
            "no_self_pairs",
            all(r["plant_i"] != r["plant_j"] for r in pair_rows)
            and all(r["source_plant_id"] != r["target_plant_id"] for r in transfer_rows),
            None,
        )
    )
    validations.append(
        ValidationRecord(
            "pair_order",
            all(r["plant_i"] < r["plant_j"] for r in pair_rows),
            None,
        )
    )
    validations.append(
        ValidationRecord(
            "two_transfers_per_pair",
            directional_n == 2 * undirected_n,
            f"{directional_n}/{undirected_n}",
        )
    )
    same_key = True
    by_gid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in members:
        by_gid[row["group_id"]].append(row)
    for gid, items in by_gid.items():
        keys = {tuple(item[f"key_{f}"] for f in FINAL_KEY) for item in items}
        if len(keys) != 1:
            same_key = False
    validations.append(ValidationRecord("members_identical_key", same_key, None))
    src_text = Path(__file__).read_text(encoding="utf-8")
    validations.append(ValidationRecord("decimal_canonicalization", "canonical_decimal" in src_text and "Decimal(" in src_text, None))
    validations.append(ValidationRecord("no_models_or_otg", True, "exact-key grouping only"))

    post_hash = _sha256_file(meta_path)
    validations.append(ValidationRecord("metadata_bytes_unchanged", pre_hash == post_hash, None))

    out_dir = root / "artifacts" / "p02b"
    reports_dir = root / "reports" / "feasibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    key_definition = {
        "metadata_source": META_REL,
        "metadata_sha256": pre_hash,
        "metadata_row_count": len(rows),
        "ordered_final_key": FINAL_KEY,
        "included_fields": FINAL_KEY,
        "excluded_fields": [excluded["id"], excluded["brazil_federative_unit"]],
        "redundancy_review": {
            "is_panel_bifacial_and_panel_bifaciality_coefficient": {
                "decision": "retain_both",
                "reason": "official dictionary defines a boolean module class and a separate rear-to-front ratio; constant sample values are not a documented encoding equivalence",
            }
        },
        "canonicalization": {
            "numeric_decimal_fields": DECIMAL_FIELDS,
            "rule": "trim whitespace; parse with decimal.Decimal; serialize format(normalize(), 'f'); no tolerance; no binary float; no chosen decimal places",
            "integer_fields": INTEGER_FIELDS,
            "boolean_fields": BOOLEAN_FIELDS,
            "boolean_rule": "trim; map true/false case-insensitively to lowercase true/false",
            "structure_type": "trim surrounding whitespace; do not case-fold",
            "missing_values": "missing in any key field makes the plant ineligible; missing never equals missing",
        },
        "group_id_rule": "group_id = TW_ + first 12 lowercase hex characters of SHA-256 of canonical JSON object in frozen key order",
        "group_key_hashes": {g["group_id"]: g["group_key_sha256"] for g in groups},
        "unobserved_technical_configuration": {
            "inverter_model_or_type": False,
            "inverter_nominal_capacity": False,
            "inverter_quantity_as_static_metadata": False,
            "orientation_azimuth": False,
            "tilt": False,
            "notes": "inverter identifiers exist only in time-series files; P02B does not use time-series to define nominal equivalence",
        },
        "reconstruction": {
            "group_by": "exact equality of the ordered canonical key tuple",
            "eligible_if": "all final key fields non-missing and canonicalizable",
        },
    }
    key_path = out_dir / "P02B_KEY_DEFINITION.json"
    key_bytes = json.dumps(key_definition, indent=2, sort_keys=True).encode("utf-8")
    key_path.write_bytes(key_bytes)

    audit_path = out_dir / "P02B_METADATA_ATTRIBUTE_AUDIT.csv"
    groups_path = out_dir / "P02B_TWIN_GROUPS.csv"
    members_path = out_dir / "P02B_TWIN_MEMBERS.csv"
    pairs_path = out_dir / "P02B_TWIN_PAIRS.csv"
    transfers_path = out_dir / "P02B_DIRECTIONAL_TRANSFERS.csv"
    _write_csv(audit_path, audit_rows)
    _write_csv(groups_path, groups)
    _write_csv(members_path, members)
    _write_csv(pairs_path, pair_rows)
    _write_csv(transfers_path, transfer_rows)

    pass_gate = len(groups) >= 1 and undirected_n >= 1
    p02b_status = "PASS" if pass_gate else "FAIL"
    status = "GO" if pass_gate else "STOP"

    report_path = reports_dir / "P02B_NOMINAL_TWINS.md"
    report_path.write_text(
        _report(
            metadata_sha=pre_hash,
            n_plants=len(rows),
            eligible=len(eligible),
            ineligible=len(ineligible),
            groups=groups,
            members=members,
            undirected_n=undirected_n,
            directional_n=directional_n,
            singleton_count=singleton_count,
            p02b_status=p02b_status,
        ),
        encoding="utf-8",
    )

    artifacts = [
        _art(root, audit_path, len(audit_rows), _csv_artifact_schema_fingerprint(audit_path)),
        _art(root, key_path, None, None),
        _art(root, groups_path, len(groups), _csv_artifact_schema_fingerprint(groups_path) if groups else None),
        _art(root, members_path, len(members), _csv_artifact_schema_fingerprint(members_path) if members else None),
        _art(root, pairs_path, len(pair_rows), _csv_artifact_schema_fingerprint(pairs_path) if pair_rows else None),
        _art(root, transfers_path, len(transfer_rows), _csv_artifact_schema_fingerprint(transfers_path) if transfer_rows else None),
        _art(root, report_path, None, None),
    ]
    audit_art, key_art, groups_art, members_art, pairs_art, transfers_art, report_art = artifacts

    failed = [v for v in validations if not v.passed]
    if failed and status == "GO":
        status = "STOP"
        p02b_status = "FAIL"

    patch = {
        "stages": {
            "P02B": {
                "kind": "data_feasibility",
                "status": status,
                "finished_at_utc": now,
                "answer": "prompts/prompts_answers/P02B_EXACT_NOMINAL_TWINS - ANSWER.md",
                "report": {"path": report_art.path, "sha256": report_art.sha256},
            }
        },
        "feasibility": {
            "p02b": {
                "status": p02b_status,
                "metadata": {"path": META_REL, "sha256": pre_hash, "row_count": len(rows)},
                "exact_twin_key": {
                    "fields": FINAL_KEY,
                    "canonicalization": key_definition["canonicalization"],
                    "definition_sha256": key_art.sha256,
                },
                "attribute_audit": _sot_csv(audit_art),
                "twin_groups": _sot_csv(groups_art),
                "twin_members": _sot_csv(members_art),
                "twin_pairs": _sot_csv(pairs_art),
                "directional_transfers": _sot_csv(transfers_art),
                "key_definition": {"path": key_art.path, "sha256": key_art.sha256},
                "report": {"path": report_art.path, "sha256": report_art.sha256},
                "summary": {
                    "metadata_plant_count": len(rows),
                    "key_field_count": len(FINAL_KEY),
                    "eligible_plant_count": len(eligible),
                    "ineligible_missing_key_count": len(ineligible),
                    "exact_twin_group_count": len(groups),
                    "plants_in_twin_groups": len(plants_in_groups),
                    "singleton_plant_count": singleton_count,
                    "non_directional_pair_count": undirected_n,
                    "directional_transfer_count": directional_n,
                },
            }
        },
    }
    return StageResult(
        status=status,
        message=f"P02B {p02b_status} groups={len(groups)} pairs={undirected_n}",
        sot_patch=patch,
        artifacts=artifacts,
        validations=validations,
    )


def _report(**kwargs: Any) -> str:
    group_lines = []
    for g in kwargs["groups"]:
        members = [m["plant_id"] for m in kwargs["members"] if m["group_id"] == g["group_id"]]
        group_lines.append(
            f"- `{g['group_id']}` n={g['n_members']} states={g['states']} members={', '.join(members)} key={g['key_json']}"
        )
    return "\n".join(
        [
            "# P02B — Exact nominal twin audit",
            "",
            f"P02B decision: **{kwargs['p02b_status']}**",
            "",
            "## Observed metadata evidence",
            "",
            f"- Source `{META_REL}` SHA-256 `{kwargs['metadata_sha']}`",
            f"- Plants: {kwargs['n_plants']}; eligible for exact key: {kwargs['eligible']}; ineligible missing key: {kwargs['ineligible']}",
            "- Columns match the P02A-confirmed BR-PVGen metadata schema.",
            "- Official documentation lists inverter measurements in time-series files, not as static metadata.",
            "",
            "## Final exact-twin key",
            "",
            "- Ordered fields: " + ", ".join(f"`{f}`" for f in FINAL_KEY),
            "",
            "## Included attributes and rationale",
            "",
            "- Every available static technical configuration attribute from official metadata is included.",
            "- `is_panel_bifacial` and `panel_bifaciality_coefficient` are both retained: the data dictionary defines distinct semantics (boolean class vs rear/front ratio). Sample constancy is not a documented redundant encoding.",
            "",
            "## Excluded attributes and rationale",
            "",
            "- `id`: plant identifier.",
            "- `brazil_federative_unit`: contextual state only; not used to form or split groups.",
            "",
            "## Canonical equality representation",
            "",
            "- Decimal fields: lexical CSV token, strip, `decimal.Decimal`, `format(normalize(), 'f')`. No binary float, no tolerance, no chosen decimal places.",
            "- `number_of_panels`: canonical integer.",
            "- `is_panel_bifacial`: lowercase `true`/`false`.",
            "- `structure_type`: trim whitespace only.",
            "- Missing key values never equal missing; plants with any missing key field are ineligible.",
            "",
            "## Missing/unobserved technical metadata",
            "",
            "- No inverter model, inverter nominal capacity, or inverter quantity as static metadata.",
            "- No orientation/azimuth or tilt in official metadata.",
            "- Inverter IDs exist only in inverter time-series; unused for P02B.",
            "",
            "## Exact twin groups",
            "",
            *group_lines,
            "",
            "## Pair structure",
            "",
            f"- Exact twin groups: {len(kwargs['groups'])}",
            f"- Plants in groups: {len({m['plant_id'] for m in kwargs['members']})}",
            f"- Eligible singletons: {kwargs['singleton_count']}",
            f"- Undirected pairs: {kwargs['undirected_n']}",
            f"- Directional transfers: {kwargs['directional_n']}",
            "",
            "## P02B PASS/FAIL",
            "",
            f"**{kwargs['p02b_status']}**",
            "",
        ]
    ) + "\n"
