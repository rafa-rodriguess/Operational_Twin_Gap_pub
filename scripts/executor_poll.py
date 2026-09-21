#!/usr/bin/env python3
"""One executor tick: pull, inspect the FIFO queue, print JSON.

Does not run scientific stages. With ``--claim``, moves the FIFO head into
``prompts/prompts_running/`` and commits (optionally pushes) so the controller
LLM can see that execution has started.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"
RUNNING = PROMPTS / "prompts_running"
EXECUTED = PROMPTS / "prompts_executed"
ANSWERS = PROMPTS / "prompts_answers"
SUPERSEDED = PROMPTS / "prompts_superseded"

QUEUE_NAME_RE = re.compile(r"^Q\d{8}T\d{6}Z__.+\.md$")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _md_files(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    names = [
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.suffix == ".md" and p.name != ".gitkeep"
    ]
    names.sort()
    return names


def parse_queue_filename(name: str) -> tuple[str, str] | None:
    """Return (queue_id, logical_stem) or None if not a valid queued job name."""
    if not QUEUE_NAME_RE.match(name):
        return None
    queue_id, rest = name.split("__", 1)
    logical_stem = rest[: -len(".md")]
    return queue_id, logical_stem


def logical_prompt_filename(queued_name: str) -> str:
    parsed = parse_queue_filename(queued_name)
    if parsed is None:
        raise ValueError(f"not a valid queue filename: {queued_name}")
    return f"{parsed[1]}.md"


def inspect_queue(root: Path) -> dict:
    prompts = root / "prompts"
    running_dir = prompts / "prompts_running"
    payload: dict = {
        "at_utc": _now(),
        "conflict_reason": None,
        "git_error": None,
        "invalid_jobs": [],
        "job": None,
        "jobs": [],
        "pending_count": 0,
        "running": [],
        "status": "IDLE",
    }

    running = _md_files(running_dir)
    payload["running"] = running
    if len(running) > 1:
        payload["status"] = "QUEUE_CONFLICT"
        payload["conflict_reason"] = "multiple_running_jobs"
        return payload
    if len(running) == 1:
        payload["status"] = "RECOVERY_REQUIRED"
        payload["job"] = running[0]
        return payload

    direct = _md_files(prompts)
    valid: list[str] = []
    invalid: list[str] = []
    for name in direct:
        if parse_queue_filename(name) is None:
            invalid.append(name)
        else:
            valid.append(name)
    valid.sort()
    payload["jobs"] = valid
    payload["pending_count"] = len(valid)
    payload["invalid_jobs"] = invalid

    if invalid:
        payload["status"] = "QUEUE_INVALID"
        return payload

    ids = [parse_queue_filename(name)[0] for name in valid]
    if len(ids) != len(set(ids)):
        payload["status"] = "QUEUE_CONFLICT"
        payload["conflict_reason"] = "duplicate_queue_id"
        return payload

    if not valid:
        payload["status"] = "IDLE"
        return payload

    payload["status"] = "JOB"
    payload["job"] = valid[0]
    return payload


class ClaimError(RuntimeError):
    pass


def claim_queued_job(root: Path, queued_name: str) -> str:
    """Move a valid queued file into prompts_running/. Returns the logical filename.

    Does not commit. Call ``commit_claim`` immediately after so the controller sees RUNNING.
    """
    parsed = parse_queue_filename(queued_name)
    if parsed is None:
        raise ClaimError(f"not a valid queue filename: {queued_name}")
    logical = logical_prompt_filename(queued_name)
    src = root / "prompts" / queued_name
    dest = root / "prompts" / "prompts_running" / logical
    if not src.is_file():
        raise ClaimError(f"queued file missing: {queued_name}")
    if dest.exists():
        raise ClaimError(f"QUEUE_CONFLICT: {logical} already in prompts_running/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    rel_src = src.relative_to(root).as_posix()
    rel_dest = dest.relative_to(root).as_posix()
    moved = _run(["git", "mv", rel_src, rel_dest], root)
    if moved.returncode != 0:
        raise ClaimError((moved.stderr or moved.stdout or "git mv failed").strip())
    return logical


def commit_claim(root: Path, logical_filename: str) -> str:
    """Commit the claim so the other LLM sees an in-progress job. Returns commit SHA."""
    stem = logical_filename[: -len(".md")] if logical_filename.endswith(".md") else logical_filename
    message = f"Claim {stem} prompt as RUNNING.\n"
    committed = _run(
        ["git", "commit", "-m", message, "--", f"prompts/prompts_running/{logical_filename}"],
        root,
    )
    if committed.returncode != 0:
        raise ClaimError((committed.stderr or committed.stdout or "git commit failed").strip())
    sha = _run(["git", "rev-parse", "HEAD"], root)
    if sha.returncode != 0:
        raise ClaimError((sha.stderr or "missing claim SHA").strip())
    return (sha.stdout or "").strip()


def push_claim(root: Path) -> None:
    branch = _load_branch(root)
    pushed = _run(["git", "push", "origin", "HEAD"], root)
    if pushed.returncode != 0:
        raise ClaimError((pushed.stderr or pushed.stdout or "git push failed").strip())


def _run(args: list[str], cwd: Path, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout=stdout,
            stderr=(stderr + f"\ntimeout after {timeout}s: {' '.join(args)}").strip(),
        )


def _load_branch(root: Path) -> str:
    env = root / ".env"
    branch = "main"
    if env.exists():
        for raw in env.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line.startswith("GIT_PRV_BRANCH="):
                branch = line.split("=", 1)[1].strip().strip('"').strip("'") or branch
    return branch


def git_pull(root: Path) -> subprocess.CompletedProcess[str]:
    pull_script = root / "git_pull_prv.sh"
    if pull_script.is_file():
        return _run(["bash", str(pull_script)], root, timeout=90)
    branch = _load_branch(root)
    fetched = _run(["git", "fetch", "--prune", "origin"], root)
    if fetched.returncode != 0:
        return fetched
    return _run(["git", "pull", "--ff-only", "origin", branch], root)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    skip_pull = "--no-pull" in args or os.environ.get("EXECUTOR_POLL_SKIP_PULL") == "1"
    do_claim = "--claim" in args
    do_push = "--push" in args
    root = ROOT
    for path in (PROMPTS, RUNNING, EXECUTED, ANSWERS, SUPERSEDED):
        path.mkdir(parents=True, exist_ok=True)

    if not skip_pull:
        pull = git_pull(root)
        if pull.returncode != 0:
            payload = {
                "at_utc": _now(),
                "conflict_reason": None,
                "git_error": (pull.stderr or pull.stdout or "git pull failed").strip(),
                "invalid_jobs": [],
                "job": None,
                "jobs": [],
                "pending_count": 0,
                "running": [],
                "status": "GIT_ERROR",
            }
            print(json.dumps(payload, ensure_ascii=True), flush=True)
            return 1

    payload = inspect_queue(root)
    if do_claim and payload.get("status") == "JOB" and payload.get("job"):
        try:
            logical = claim_queued_job(root, payload["job"])
            sha = commit_claim(root, logical)
            payload = inspect_queue(root)
            payload["claim_sha"] = sha
            payload["claimed"] = logical
            if do_push:
                push_claim(root)
                payload["claim_pushed"] = True
        except ClaimError as exc:
            payload = inspect_queue(root)
            payload["status"] = "GIT_ERROR"
            payload["git_error"] = str(exc)
            print(json.dumps(payload, ensure_ascii=True), flush=True)
            return 1

    print(json.dumps(payload, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
