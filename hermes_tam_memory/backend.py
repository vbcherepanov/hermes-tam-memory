"""TAM wire contracts: the local server (``tam``) and the team server (``tam-team``) differ slightly.

Local ``memory_recall`` groups hits by type (``{"results": {"fact": [...]}}``); the team server
returns a ranked list of ``{"scope", "record"}`` items and rejects unknown arguments. Saves differ
the same way. Everything else in the plugin works on the neutral types below.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

RECORD_TYPES = ("fact", "decision", "solution", "lesson", "convention")
IMPORTANCE_LEVELS = ("low", "medium", "high", "critical")
REQUIRED_TOOLS = frozenset({"memory_save", "memory_recall"})
TEAM_MARKER_TOOL = "memory_scopes"


class Backend(StrEnum):
    LOCAL = "local"
    TEAM = "team"


class ContractError(Exception):
    """TAM answered with a shape this plugin does not understand."""


@dataclass(frozen=True)
class SaveRequest:
    content: str
    type: str = "fact"
    project: str = "general"
    tags: Sequence[str] = ()
    context: str = ""
    importance: str = "medium"
    source_format: str = "auto"
    request_id: str = ""


@dataclass(frozen=True)
class SaveOutcome:
    saved: bool
    record_id: int | None
    rejected_by_quality_gate: bool
    deduplicated: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "saved": self.saved,
            "id": self.record_id,
            "deduplicated": self.deduplicated,
            "rejected_by_quality_gate": self.rejected_by_quality_gate,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Memory:
    record_id: int | None
    content: str
    type: str
    project: str
    created_at: str
    score: float
    context: str
    tags: Sequence[str]
    author: str

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.record_id,
            "type": self.type,
            "project": self.project,
            "created_at": self.created_at,
            "score": round(self.score, 4),
            "content": self.content,
        }
        if self.author:
            data["author"] = self.author
        return data


def detect_backend(tools: frozenset[str]) -> Backend:
    missing = REQUIRED_TOOLS - tools
    if missing:
        raise ContractError(f"server does not expose required tools: {', '.join(sorted(missing))}")
    return Backend.TEAM if TEAM_MARKER_TOOL in tools else Backend.LOCAL


def save_arguments(backend: Backend, request: SaveRequest) -> dict[str, Any]:
    args: dict[str, Any] = {
        "content": request.content,
        "type": request.type,
        "project": request.project,
        "tags": list(request.tags),
        "context": request.context,
        "importance": request.importance,
        "source_format": request.source_format,
    }
    if backend is Backend.TEAM and request.request_id:
        args["request_id"] = request.request_id
    return args


def recall_arguments(backend: Backend, query: str, limit: int, project: str | None) -> dict[str, Any]:
    args: dict[str, Any] = {"query": query, "limit": limit}
    if project:
        args["project"] = project
    if backend is Backend.LOCAL:
        args["detail"] = "full"
    return args


def parse_save(backend: Backend, payload: Any) -> SaveOutcome:
    if not isinstance(payload, dict):
        raise ContractError(f"memory_save returned {type(payload).__name__}")
    data = _as_dict(payload.get("data")) if backend is Backend.TEAM else payload
    saved = bool(data.get("saved"))
    record_id = data.get("id") if isinstance(data.get("id"), int) else None
    if saved:
        return SaveOutcome(True, record_id, False, bool(data.get("deduplicated")), "")
    quality = _as_dict(data.get("quality")) or data
    rejected = backend is Backend.TEAM or bool(data.get("rejected_by_quality_gate"))
    reason = str(quality.get("reason") or "rejected by TAM")
    return SaveOutcome(False, None, rejected, False, reason)


def parse_recall(backend: Backend, payload: Any) -> list[Memory]:
    if not isinstance(payload, dict):
        raise ContractError(f"memory_recall returned {type(payload).__name__}")
    results = payload.get("results")
    if backend is Backend.TEAM:
        if not isinstance(results, list):
            raise ContractError("team memory_recall has no results list")
        records = [(item.get("record"), _team_author(item), "fact") for item in results if isinstance(item, dict)]
    else:
        if not isinstance(results, dict):
            raise ContractError("memory_recall has no grouped results")
        records = [
            (item, "", str(kind)) for kind, group in results.items() if isinstance(group, list) for item in group
        ]
    memories = [_memory(record, author, kind) for record, author, kind in records if isinstance(record, dict)]
    memories = [m for m in memories if m.content]
    memories.sort(key=lambda m: m.score, reverse=True)
    return memories


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _team_author(item: Mapping[str, Any]) -> str:
    creator = _as_dict(_as_dict(item.get("record")).get("created_by"))
    return str(creator.get("display_name") or creator.get("user_id") or "")


def _memory(record: Mapping[str, Any], author: str, default_type: str) -> Memory:
    tags = record.get("tags", [])
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except json.JSONDecodeError:
            tags = []
    try:
        score = float(record.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return Memory(
        record_id=record.get("id") if isinstance(record.get("id"), int) else None,
        content=str(record.get("content") or "").strip(),
        type=str(record.get("type") or default_type),
        project=str(record.get("project") or ""),
        created_at=str(record.get("created_at") or ""),
        score=score,
        context=str(record.get("context") or ""),
        tags=[str(t) for t in tags] if isinstance(tags, list) else [],
        author=author,
    )
