"""Stage contracts. Stages return patches; only orchestrator.py writes the SoT."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ALLOWED_STATUS = frozenset({"GO", "STOP", "REDESIGN"})


@dataclass(frozen=True)
class StageContext:
    root: Path
    stage_id: str
    run_id: str
    sot: dict[str, Any]


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    sha256: str
    bytes: int
    row_count: int | None = None
    schema_fingerprint: str | None = None


@dataclass(frozen=True)
class ValidationRecord:
    name: str
    passed: bool
    details: str | None = None


@dataclass
class StageResult:
    status: Literal["GO", "STOP", "REDESIGN"]
    message: str
    sot_patch: dict[str, Any]
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    validations: list[ValidationRecord] = field(default_factory=list)
