"""Hermes ``MemoryProvider`` backed by total-agent-memory (TAM) over MCP."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt, spawn_context_thread

from . import __version__
from .backend import (
    IMPORTANCE_LEVELS,
    RECORD_TYPES,
    ContractError,
    SaveOutcome,
    SaveRequest,
    parse_recall,
    parse_save,
    recall_arguments,
    save_arguments,
)
from .config import TOKEN_ENV, TamConfig, config_path, load_config, read_token
from .config import save_config as write_config
from .connection import TamConnection, TamUnavailable
from .mcp_client import HttpTransport, McpClient, StdioTransport, ToolCallError, Transport, TransportClosed
from .text import clean_text, format_recall, format_turn, judge_turn, session_context, truncate

logger = logging.getLogger(__name__)

PROVIDER_NAME = "tam"
PROVIDER_LABEL = "TAM"
TOOL_RECALL = "tam_recall"
TOOL_SAVE = "tam_save"
NON_PRIMARY_CONTEXTS = frozenset({"cron", "subagent", "flush"})
TURN_TAGS = ("hermes", "conversation-turn")
BUILTIN_MEMORY_TAGS = ("hermes", "builtin-memory")
TOOL_RECALL_DEFAULT_LIMIT = 8
TOOL_RECALL_MAX_LIMIT = 20
QUERY_MAX_CHARS = 2000
WRITER_RETRY_S = 2.0
WRITER_JOIN_S = 2.0
TOOL_CONTENT_MAX_CHARS = 20000
# Environment passed to a spawned TAM: no Hermes secrets, only what TAM itself reads.
SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "SHELL",
        "TMPDIR",
        "SYSTEMROOT",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "TEMP",
        "TMP",
        "PATHEXT",
    }
)
SAFE_ENV_PREFIXES = ("XDG_", "TAM_", "MEMORY_", "CLAUDE_MEMORY_", "HF_", "FASTEMBED_", "OLLAMA_")

RECALL_SCHEMA = {
    "name": TOOL_RECALL,
    "description": (
        "Search long-term memory in total-agent-memory (TAM): facts, decisions, solutions and "
        "past conversations saved by Hermes and by other agents that share this TAM store. "
        "Relevant memories are already injected each turn; call this for a targeted lookup."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in natural language."},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": TOOL_RECALL_MAX_LIMIT,
                "default": TOOL_RECALL_DEFAULT_LIMIT,
            },
            "project": {"type": "string", "description": "Restrict to one TAM project (omit to search all)."},
        },
        "required": ["query"],
    },
}
SAVE_SCHEMA = {
    "name": TOOL_SAVE,
    "description": (
        "Save a durable memory to total-agent-memory (TAM): a fact, decision (include the WHY in "
        "context), solution, lesson or convention worth recalling in future sessions. TAM may "
        "reject low-value records through its quality gate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The self-contained knowledge to remember."},
            "type": {"type": "string", "enum": list(RECORD_TYPES), "default": "fact"},
            "importance": {"type": "string", "enum": list(IMPORTANCE_LEVELS), "default": "medium"},
            "context": {"type": "string", "description": "Why it matters or where it came from."},
        },
        "required": ["content"],
    },
}


@dataclass(frozen=True)
class _WriteJob:
    request: SaveRequest
    origin: str


def child_environment(config: TamConfig, environ: Mapping[str, str]) -> dict[str, str]:
    env = {
        key: value
        for key, value in environ.items()
        if (key in SAFE_ENV_KEYS or key.startswith(SAFE_ENV_PREFIXES)) and key != TOKEN_ENV
    }
    if config.memory_dir:
        env["TAM_MEMORY_DIR"] = os.path.expanduser(config.memory_dir)
    env.update(config.env)
    return env


def resolve_command(config: TamConfig) -> str | None:
    return shutil.which(os.path.expanduser(config.command)) if config.command else None


def _hermes_home() -> str:
    from hermes_constants import get_hermes_home

    return str(get_hermes_home())


def _tool_error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


class TamMemoryProvider(MemoryProvider):
    """Saves completed turns to TAM and injects relevant TAM memories before each turn."""

    def __init__(self) -> None:
        self._config = TamConfig()
        self._connection: TamConnection | None = None
        self._session_id = ""
        self._write_enabled = False
        self._tags: tuple[str, ...] = TURN_TAGS
        self._last_recall: RecallStatus | None = None
        self._jobs: deque[_WriteJob] = deque()
        self._jobs_cond = threading.Condition()
        self._in_flight = 0
        self._accepting = False
        self._stop = False
        self._writer: threading.Thread | None = None
        self._outcomes: Counter[str] = Counter()

    # -- identity & availability ------------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        config = load_config(_hermes_home())
        if config.mode == "remote":
            return bool(config.url)
        return resolve_command(config) is not None

    def unavailable_reason(self) -> str:
        config = load_config(_hermes_home())
        if config.mode == "remote":
            return f"TAM remote mode needs a url in {config_path(_hermes_home())}; run `hermes memory setup tam`."
        return (
            f"TAM command {config.command!r} was not found on PATH. Install TAM "
            "(`pipx install total-agent-memory` or `uv tool install total-agent-memory`) "
            "or set `command` via `hermes memory setup tam`."
        )

    # -- lifecycle ----------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        if self._connection is not None:
            # Re-initialization must not orphan the previous TAM process or writer thread.
            self.shutdown()
        hermes_home = str(kwargs.get("hermes_home") or _hermes_home())
        self._config = load_config(hermes_home)
        self._session_id = session_id
        agent_context = str(kwargs.get("agent_context") or "primary")
        self._write_enabled = self._config.auto_capture and agent_context not in NON_PRIMARY_CONTEXTS
        identity = str(kwargs.get("agent_identity") or "")
        self._tags = TURN_TAGS + ((f"hermes-profile:{identity}",) if identity and identity != "default" else ())
        self._connection = TamConnection(
            self._client_factory(self._config),
            startup_timeout=self._config.startup_timeout,
            spawn_thread=spawn_context_thread,
        )
        with self._jobs_cond:
            self._accepting, self._stop = True, False
        self._writer = spawn_context_thread(self._write_loop, name="tam-writer")
        self._writer.start()
        self._connection.warm_up()
        logger.info(
            "tam.initialized mode=%s project=%s capture=%s recall=%s agent_context=%s",
            self._config.mode,
            self._config.project,
            self._write_enabled,
            self._config.auto_recall,
            agent_context,
        )

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        self._session_id = new_session_id

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        if not self.flush(self._config.shutdown_timeout):
            logger.warning("tam.session_end_flush_incomplete pending=%d", self.pending_writes)

    def shutdown(self) -> None:
        with self._jobs_cond:
            self._accepting = False
        drained = self.flush(self._config.shutdown_timeout)
        with self._jobs_cond:
            self._stop = True
            if self._jobs:
                self._count("abandoned", len(self._jobs))
                self._jobs.clear()
            self._jobs_cond.notify_all()
        metrics: dict[str, Any] = {}
        if self._connection is not None:
            # Closing first wakes a writer blocked on connect; its in-flight job is counted as abandoned.
            self._connection.close()
            metrics = self._connection.metrics.snapshot()
        if self._writer is not None:
            self._writer.join(WRITER_JOIN_S)
        logger.info(
            "tam.shutdown drained=%s outcomes=%s metrics=%s",
            drained,
            json.dumps(self.outcomes, sort_keys=True),
            json.dumps(metrics, sort_keys=True),
        )

    # -- prompt & recall -------------------------------------------------------

    def system_prompt_block(self) -> str:
        if self._connection is None:
            return ""
        return (
            "# TAM long-term memory\n"
            f"total-agent-memory is active (project: {self._config.project}). Relevant memories are recalled "
            f"automatically. Use {TOOL_RECALL} for a targeted search and {TOOL_SAVE} to keep a durable fact, "
            "decision or solution."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_recall = None
        connection = self._connection
        if connection is None or not self._config.auto_recall or is_trivial_prompt(query):
            return ""
        text = clean_text(query, QUERY_MAX_CHARS)
        if not text:
            return ""
        project = self._config.project if self._config.recall_scope == "project" else None
        # Over-fetch: this session's own turns are dropped client-side (they are in the transcript already).
        fetch = min(TOOL_RECALL_MAX_LIMIT, self._config.recall_limit + 4)
        try:
            payload, backend = connection.call(
                "memory_recall",
                lambda b: recall_arguments(b, text, fetch, project),
                timeout=self._config.recall_timeout,
                wait=self._config.recall_timeout,
                shared_deadline=True,
            )
            memories = parse_recall(backend, payload)
        except (TamUnavailable, ToolCallError, ContractError) as exc:
            logger.debug("tam.prefetch_skipped error=%s", exc)
            return ""
        block = format_recall(
            memories,
            budget_tokens=self._config.recall_budget_tokens,
            max_items=self._config.recall_limit,
            exclude_session=session_id or self._session_id,
        )
        if block.count:
            self._last_recall = RecallStatus(provider_label=PROVIDER_LABEL, count=block.count)
        logger.debug("tam.prefetch hits=%d injected=%d chars=%d", len(memories), block.count, len(block.text))
        return block.text

    def recall_status(self) -> RecallStatus | None:
        return self._last_recall

    # -- capture ------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._write_enabled:
            return
        limit = self._config.max_turn_chars // 2
        user, assistant = clean_text(user_content, limit), clean_text(assistant_content, limit)
        verdict = judge_turn(user, assistant, min_chars=self._config.min_turn_chars, is_trivial=is_trivial_prompt)
        if not verdict.keep:
            self._count(f"skipped_{verdict.reason}")
            return
        self._enqueue(
            _WriteJob(
                SaveRequest(
                    content=format_turn(user, assistant),
                    type="fact",
                    project=self._config.project,
                    tags=self._tags,
                    context=session_context(session_id or self._session_id),
                    source_format="conversation",
                    request_id=str(uuid.uuid4()),
                ),
                origin="turn",
            )
        )

    def on_memory_write(self, action: str, target: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        # replace/remove carry no stable TAM record identity; mirroring them would duplicate or guess.
        if action != "add" or not self._write_enabled:
            return
        text = clean_text(content, self._config.max_turn_chars)
        if not text:
            return
        self._enqueue(
            _WriteJob(
                SaveRequest(
                    content=text,
                    type="fact",
                    project=self._config.project,
                    tags=(*BUILTIN_MEMORY_TAGS, f"hermes-{target}"),
                    context=session_context(self._session_id),
                    request_id=str(uuid.uuid4()),
                ),
                origin="builtin_memory",
            )
        )

    @property
    def outcomes(self) -> dict[str, int]:
        with self._jobs_cond:
            return dict(self._outcomes)

    def _count(self, outcome: str, n: int = 1) -> None:
        with self._jobs_cond:
            self._outcomes[outcome] += n

    @property
    def pending_writes(self) -> int:
        with self._jobs_cond:
            return len(self._jobs) + self._in_flight

    def flush(self, timeout: float) -> bool:
        """Wait until queued writes reach TAM; False if some remain after ``timeout``."""
        with self._jobs_cond:
            self._jobs_cond.notify_all()
            return self._jobs_cond.wait_for(lambda: not self._jobs and not self._in_flight, timeout)

    def _enqueue(self, job: _WriteJob) -> None:
        with self._jobs_cond:
            if not self._accepting:
                self._count("rejected_after_shutdown")
                return
            if len(self._jobs) >= self._config.pending_limit:
                self._jobs.popleft()
                self._count("dropped_queue_full")
                logger.warning("tam.write_queue_full limit=%d dropped_oldest=1", self._config.pending_limit)
            self._jobs.append(job)
            self._jobs_cond.notify_all()

    def _write_loop(self) -> None:
        while True:
            with self._jobs_cond:
                self._jobs_cond.wait_for(lambda: self._stop or bool(self._jobs))
                if self._stop:
                    return
                job = self._jobs.popleft()
                self._in_flight += 1
            retry = False
            try:
                retry = not self._write(job)
            finally:
                with self._jobs_cond:
                    self._in_flight -= 1
                    if retry and self._stop:
                        self._count("abandoned")
                    elif retry:
                        self._jobs.appendleft(job)
                    self._jobs_cond.notify_all()
            if retry:
                with self._jobs_cond:
                    self._jobs_cond.wait_for(lambda: self._stop, WRITER_RETRY_S)

    def _write(self, job: _WriteJob) -> bool:
        """Persist one job. False means TAM is unreachable and the job should be retried."""
        connection = self._connection
        if connection is None:
            return False
        try:
            payload, backend = connection.call(
                "memory_save",
                lambda b: save_arguments(b, job.request),
                timeout=self._config.request_timeout,
                wait=self._config.startup_timeout,
            )
            outcome = parse_save(backend, payload)
        except TamUnavailable as exc:
            logger.debug("tam.write_deferred origin=%s error=%s", job.origin, exc)
            return False
        except (ToolCallError, ContractError) as exc:
            self._count("failed")
            logger.warning("tam.write_failed origin=%s error=%s", job.origin, exc)
            return True
        self._record(job, outcome)
        return True

    def _record(self, job: _WriteJob, outcome: SaveOutcome) -> None:
        if outcome.saved:
            self._count("deduplicated" if outcome.deduplicated else "saved")
            logger.debug(
                "tam.write_saved origin=%s id=%s deduplicated=%s", job.origin, outcome.record_id, outcome.deduplicated
            )
        else:
            self._count("rejected_quality_gate" if outcome.rejected_by_quality_gate else "rejected")
            logger.info("tam.write_rejected origin=%s reason=%s", job.origin, outcome.reason)

    # -- tools ---------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [RECALL_SCHEMA, SAVE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if tool_name == TOOL_RECALL:
            return self._tool_recall(args)
        if tool_name == TOOL_SAVE:
            return self._tool_save(args)
        return _tool_error(f"unknown TAM tool: {tool_name}")

    def _tool_recall(self, args: Mapping[str, Any]) -> str:
        query = clean_text(str(args.get("query") or ""), QUERY_MAX_CHARS)
        if not query:
            return _tool_error("query is required")
        try:
            limit = max(1, min(TOOL_RECALL_MAX_LIMIT, int(args.get("limit") or TOOL_RECALL_DEFAULT_LIMIT)))
        except (TypeError, ValueError):
            return _tool_error("limit must be an integer")
        project = str(args.get("project") or "").strip() or None
        try:
            payload, backend = self._require_connection().call(
                "memory_recall",
                lambda b: recall_arguments(b, query, limit, project),
                timeout=self._config.request_timeout,
                wait=self._config.request_timeout,
            )
            memories = parse_recall(backend, payload)
        except (TamUnavailable, ToolCallError, ContractError) as exc:
            return _tool_error(f"TAM recall failed: {exc}")
        return json.dumps(
            {"count": len(memories), "results": [m.to_dict() for m in memories[:limit]]}, ensure_ascii=False
        )

    def _tool_save(self, args: Mapping[str, Any]) -> str:
        content = truncate(str(args.get("content") or "").strip(), TOOL_CONTENT_MAX_CHARS)
        if not content:
            return _tool_error("content is required")
        record_type = str(args.get("type") or "fact")
        importance = str(args.get("importance") or "medium")
        if record_type not in RECORD_TYPES:
            return _tool_error(f"type must be one of {', '.join(RECORD_TYPES)}")
        if importance not in IMPORTANCE_LEVELS:
            return _tool_error(f"importance must be one of {', '.join(IMPORTANCE_LEVELS)}")
        request = SaveRequest(
            content=content,
            type=record_type,
            project=self._config.project,
            tags=("hermes", "explicit"),
            context=str(args.get("context") or "").strip(),
            importance=importance,
            request_id=str(uuid.uuid4()),
        )
        try:
            payload, backend = self._require_connection().call(
                "memory_save",
                lambda b: save_arguments(b, request),
                timeout=self._config.request_timeout,
                wait=self._config.request_timeout,
            )
            outcome = parse_save(backend, payload)
        except (TamUnavailable, ToolCallError, ContractError) as exc:
            return _tool_error(f"TAM save failed: {exc}")
        self._record(_WriteJob(request, "tool"), outcome)
        return json.dumps(outcome.to_dict(), ensure_ascii=False)

    def _require_connection(self) -> TamConnection:
        if self._connection is None:
            raise TamUnavailable("TAM provider is not initialized")
        return self._connection

    # -- setup ---------------------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "mode",
                "description": "How Hermes reaches TAM",
                "default": "local",
                "choices": ["local", "remote"],
            },
            {"key": "command", "description": "TAM command (local mode)", "default": "tam", "when": {"mode": "local"}},
            {
                "key": "memory_dir",
                "description": "TAM data directory, blank for TAM's default (local mode)",
                "when": {"mode": "local"},
            },
            {
                "key": "url",
                "description": "TAM MCP endpoint, e.g. http://127.0.0.1:3737/mcp/ (remote mode)",
                "when": {"mode": "remote"},
            },
            {
                "key": "token",
                "description": "Team server token, blank for an unauthenticated server (remote mode)",
                "secret": True,
                "env_var": TOKEN_ENV,
                "when": {"mode": "remote"},
            },
            {"key": "project", "description": "TAM project for saved turns", "default": "hermes"},
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        write_config(values, hermes_home)

    def get_status_config(self, provider_config: dict[str, Any]) -> dict[str, Any]:
        hermes_home = _hermes_home()
        config = load_config(hermes_home)
        status: dict[str, Any] = {
            "config_file": str(config_path(hermes_home)),
            "mode": config.mode,
            "project": config.project,
            "auto_capture": config.auto_capture,
            "auto_recall": config.auto_recall,
            "recall_scope": config.recall_scope,
        }
        if config.mode == "remote":
            status["url"] = config.url or "(not set)"
            status["token"] = "set" if read_token() else "not set"
        else:
            status["command"] = resolve_command(config) or f"{config.command} (NOT FOUND on PATH)"
            status["memory_dir"] = config.memory_dir or "(TAM default)"
        return status

    def post_setup(self, hermes_home: str, config: dict[str, Any]) -> None:
        from .setup_flow import run_setup

        run_setup(self, hermes_home, config)

    def _client_factory(self, config: TamConfig):
        def build() -> McpClient:
            transport: Transport
            if config.mode == "remote":
                transport = HttpTransport(config.url, token=read_token())
            else:
                command = resolve_command(config)
                if command is None:
                    raise TransportClosed(f"TAM command {config.command!r} not found on PATH")
                transport = StdioTransport(
                    [command, *config.args], child_environment(config, os.environ), spawn_thread=spawn_context_thread
                )
            return McpClient(transport, client_version=__version__)

        return build
