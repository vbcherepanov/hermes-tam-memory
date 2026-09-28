"""Turn cleaning, the capture noise filter, and recall formatting under a token budget."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .backend import Memory

CHARS_PER_TOKEN = 4
RECALL_ITEM_MAX_CHARS = 700
RECALL_HEADER = "## Recalled from total-agent-memory (TAM)"
SESSION_CONTEXT_PREFIX = "hermes session "
TRUNCATION_MARK = " […]"

_INJECTED_BLOCK_RE = re.compile(r"<memory-context>[\s\S]*?</memory-context>", re.IGNORECASE)
_DATA_URI_RE = re.compile(r"data:[^;,\s]+;base64,[A-Za-z0-9+/=]+")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def clean_text(text: str, max_chars: int) -> str:
    """Drop injected recall blocks and inline binary, collapse blank runs, cap the length."""
    cleaned = _INJECTED_BLOCK_RE.sub("", text or "")
    cleaned = _DATA_URI_RE.sub("[inline data]", cleaned)
    cleaned = _BLANK_RUN_RE.sub("\n\n", cleaned).strip()
    return truncate(cleaned, max_chars)


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - len(TRUNCATION_MARK))].rstrip() + TRUNCATION_MARK


@dataclass(frozen=True)
class TurnVerdict:
    keep: bool
    reason: str


def judge_turn(user: str, assistant: str, *, min_chars: int, is_trivial: Callable[[str], bool]) -> TurnVerdict:
    """Client-side noise filter; TAM's quality gate (when configured) scores whatever passes."""
    if not assistant:
        return TurnVerdict(False, "empty_assistant")
    if is_trivial(user):
        return TurnVerdict(False, "trivial_prompt")
    if len(user) + len(assistant) < min_chars:
        return TurnVerdict(False, "too_short")
    return TurnVerdict(True, "")


def format_turn(user: str, assistant: str) -> str:
    return f"User: {user}\nAssistant: {assistant}"


def session_context(session_id: str) -> str:
    return f"{SESSION_CONTEXT_PREFIX}{session_id}" if session_id else ""


@dataclass(frozen=True)
class RecallBlock:
    text: str
    count: int


def format_recall(
    memories: Sequence[Memory], *, budget_tokens: int, max_items: int, exclude_session: str = ""
) -> RecallBlock:
    """Render memories best-first until the token budget is spent; skip this session's own turns."""
    budget_chars = budget_tokens * CHARS_PER_TOKEN - len(RECALL_HEADER) - 1
    marker = session_context(exclude_session)
    lines: list[str] = []
    seen: set[str] = set()
    for memory in memories:
        if len(lines) >= max_items:
            break
        if marker and memory.context == marker:
            continue
        body = " ".join(memory.content.split())
        if body in seen:
            continue
        line = _render(memory, truncate(body, RECALL_ITEM_MAX_CHARS))
        if len(line) + 1 > budget_chars:
            # Keep packing smaller, lower-ranked items; only the first item is cut to fit.
            if lines:
                continue
            line = truncate(line, budget_chars - 1)
            if len(line) <= len(TRUNCATION_MARK):
                break
        seen.add(body)
        lines.append(line)
        budget_chars -= len(line) + 1
    if not lines:
        return RecallBlock("", 0)
    return RecallBlock("\n".join([RECALL_HEADER, *lines]), len(lines))


def _render(memory: Memory, body: str) -> str:
    meta = [memory.type]
    if memory.project:
        meta.append(memory.project)
    if memory.created_at:
        meta.append(memory.created_at[:10])
    if memory.author:
        meta.append(f"by {memory.author}")
    return f"- [{' · '.join(meta)}] {body}"
