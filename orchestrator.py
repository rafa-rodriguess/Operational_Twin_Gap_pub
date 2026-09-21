#!/usr/bin/env python3
"""Official executor. Stages return patches; this process validates and commits."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sot import apply_patch, atomic_write, load as load_sot, sha256_file, sha256_sot, snapshot
from src.contracts import ALLOWED_STATUS, ArtifactRecord, StageContext, StageResult, ValidationRecord

STATUS_PATH = Path("outputs") / "logs" / "orchestrator_status.json"


class OrchestratorError(RuntimeError):
    pass


def project_root() -> Path:
    env = os.environ.get("STUDY_ROOT")
    if env:
        return Path(env).resolve()
    return ROOT


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_stage_id(raw: str) -> str:
    text = str(raw).strip()
    if text.lower().startswith("s") and text[1:].isdigit():
        text = text[1:]
    if text.isdigit():
        return f"{int(text):02d}"
    raise OrchestratorError(f"invalid stage id: {raw}")


def discover_stage_files(root: Path, stage_id: str) -> list[Path]:
    directory = root / "src" / "run"
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob(f"{stage_id}_*.py") if p.is_file())


def discover_all_stage_ids(root: Path) -> list[str]:
    directory = root / "src" / "run"
    if not directory.is_dir():
        return []
    found: set[str] = set()
    for path in directory.glob("[0-9][0-9]_*.py"):
        prefix = path.name.split("_", 1)[0]
        if prefix.isdigit():
            found.add(f"{int(prefix):02d}")
    return sorted(found, key=int)


def resolve_stage_path(root: Path, stage_id: str) -> Path:
    matches = discover_stage_files(root, stage_id)
    if not matches:
        raise OrchestratorError(f"no stage module for {stage_id} under src/run/{stage_id}_*.py")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise OrchestratorError(f"ambiguous stage {stage_id}: {names}")
    return matches[0]


def load_stage_module(root: Path, stage_id: str):
    path = resolve_stage_path(root, stage_id)
    spec = importlib.util.spec_from_file_location(f"otg_run_{stage_id}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise OrchestratorError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _inside_root(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_result(result: Any) -> StageResult:
    if not isinstance(result, StageResult):
        raise OrchestratorError(f"stage returned {type(result)!r}, expected StageResult")
    if result.status not in ALLOWED_STATUS:
        raise OrchestratorError(f"invalid status: {result.status}")
    json.dumps(result.sot_patch)
    return result


def _validate_validations(result: StageResult) -> None:
    failed = [v for v in result.validations if not v.passed]
    if failed and result.status == "GO":
        names = ", ".join(v.name for v in failed)
        raise OrchestratorError(f"GO with failed validations: {names}")


def _validate_artifacts(root: Path, artifacts: list[ArtifactRecord]) -> None:
    for rec in artifacts:
        if not isinstance(rec, ArtifactRecord):
            raise OrchestratorError("artifact is not ArtifactRecord")
        rel = rec.path
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise OrchestratorError(f"illegal artifact path: {rel}")
        path = (root / rel).resolve()
        if not _inside_root(root, path):
            raise OrchestratorError(f"artifact outside repository: {rel}")
        if not path.is_file():
            raise OrchestratorError(f"artifact missing: {rel}")
        size = path.stat().st_size
        if size != rec.bytes:
            raise OrchestratorError(f"artifact byte mismatch: {rel}")
        digest = sha256_file(path)
        if digest != rec.sha256:
            raise OrchestratorError(f"artifact hash mismatch: {rel}")


def _stage_id_sort_key(stage_id: str) -> tuple:
    text = str(stage_id)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def derive_operational_bookkeeping(runs: list[dict[str, Any]], last: dict[str, Any] | None) -> dict[str, Any]:
    latest: dict[str, str] = {}
    for run in runs:
        sid = str(run.get("stage_id") or "")
        if not sid:
            continue
        latest[sid] = str(run.get("status") or "")
    completed = sorted((sid for sid, status in latest.items() if status == "GO"), key=_stage_id_sort_key)
    failed = None
    if last and last.get("status") in {"STOP", "REDESIGN", "FAILED_RUNTIME"}:
        failed = {"stage_id": last.get("stage_id"), "status": last.get("status"), "run_id": last.get("run_id")}
    return {"completed": completed, "failed": failed}


def _record_status(root: Path, payload: dict[str, Any]) -> None:
    path = root / STATUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {"runs": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            existing = loaded if isinstance(loaded, dict) else {"runs": []}
        except json.JSONDecodeError:
            existing = {"runs": []}
    runs = list(existing.get("runs") or [])
    runs.append(payload)
    existing["runs"] = runs
    existing["last"] = payload
    derived = derive_operational_bookkeeping(runs, payload)
    existing["completed"] = derived["completed"]
    existing["failed"] = derived["failed"]
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def run_stage(stage_id: str) -> StageResult:
    root = project_root()
    stage_id = normalize_stage_id(stage_id)
    run_id = uuid.uuid4().hex
    started = _now()
    sot_path = root / "artifacts" / "sot.json"
    pre_hash = sha256_sot(sot_path)
    data = snapshot(load_sot(sot_path), sot_path)
    ctx = StageContext(root=root, stage_id=stage_id, run_id=run_id, sot=data)
    module, stage_path = load_stage_module(root, stage_id)
    if not hasattr(module, "run"):
        raise OrchestratorError(f"{stage_path.name} has no run(ctx)")
    try:
        raw = module.run(ctx)
    except OrchestratorError:
        raise
    except Exception as exc:
        _record_status(
            root,
            {
                "run_id": run_id,
                "stage_id": stage_id,
                "started_at_utc": started,
                "finished_at_utc": _now(),
                "status": "FAILED_RUNTIME",
                "error": f"{type(exc).__name__}: {exc}",
                "pre_sot_sha256": pre_hash,
                "post_sot_sha256": None,
                "artifact_paths": [],
            },
        )
        raise OrchestratorError(f"runtime error in stage {stage_id}: {exc}") from exc

    result = _validate_result(raw)
    mid_hash = sha256_sot(sot_path)
    if mid_hash != pre_hash:
        raise OrchestratorError("stage mutated artifacts/sot.json directly")
    _validate_validations(result)
    _validate_artifacts(root, result.artifacts)
    merged = apply_patch(data, result.sot_patch)
    post_hash = pre_hash
    if result.status in {"GO", "STOP", "REDESIGN"}:
        atomic_write(merged, sot_path)
        post_hash = sha256_sot(sot_path)
    _record_status(
        root,
        {
            "run_id": run_id,
            "stage_id": stage_id,
            "started_at_utc": started,
            "finished_at_utc": _now(),
            "status": result.status,
            "error": None if result.status == "GO" else result.message,
            "pre_sot_sha256": pre_hash,
            "post_sot_sha256": post_hash,
            "artifact_paths": [{"path": a.path, "sha256": a.sha256} for a in result.artifacts],
        },
    )
    if result.status in {"STOP", "REDESIGN"}:
        raise OrchestratorError(f"stage {stage_id} halted with {result.status}: {result.message}")
    return result


def run_range(from_id: str, to_id: str) -> list[StageResult]:
    root = project_root()
    start = normalize_stage_id(from_id)
    end = normalize_stage_id(to_id)
    if int(start) > int(end):
        raise OrchestratorError("from-stage must be <= to-stage")
    existing = [sid for sid in discover_all_stage_ids(root) if int(start) <= int(sid) <= int(end)]
    results: list[StageResult] = []
    for sid in existing:
        results.append(run_stage(sid))
    return results


def cmd_list() -> int:
    root = project_root()
    ids = discover_all_stage_ids(root)
    if not ids:
        print("no stages under src/run/")
        return 0
    for sid in ids:
        files = discover_stage_files(root, sid)
        print(f"{sid}  {files[0].name if files else '?'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OTG scientific orchestrator")
    sub = parser.add_subparsers(dest="command")
    run_p = sub.add_parser("run")
    run_p.add_argument("--stage")
    run_p.add_argument("--from-stage")
    run_p.add_argument("--to-stage")
    sub.add_parser("list")
    args = parser.parse_args(argv)

    if args.command == "list" or args.command is None:
        if args.command == "list":
            return cmd_list()
        print("commands: python orchestrator.py list | run --stage XX | run --from-stage XX --to-stage YY")
        return 2

    if args.command == "run":
        try:
            if args.stage and (args.from_stage or args.to_stage):
                print("use either --stage or --from-stage/--to-stage", file=sys.stderr)
                return 2
            if args.stage:
                result = run_stage(args.stage)
                print(f"{result.status} stage={normalize_stage_id(args.stage)} {result.message}")
                return 0
            if args.from_stage or args.to_stage:
                if not args.from_stage or not args.to_stage:
                    print("--from-stage and --to-stage are both required", file=sys.stderr)
                    return 2
                results = run_range(args.from_stage, args.to_stage)
                print(f"GO range {len(results)} stage(s)")
                return 0
            print("run requires --stage or --from-stage/--to-stage", file=sys.stderr)
            return 2
        except OrchestratorError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except Exception:
            traceback.print_exc()
            return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
