"""One lazily-built, self-healing MCP connection to TAM with exponential backoff and call metrics."""

from __future__ import annotations

import bisect
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from .backend import Backend, ContractError, detect_backend
from .mcp_client import McpClient, McpError, McpTimeout, RpcError, ToolCallError

logger = logging.getLogger(__name__)

BACKOFF_INITIAL_S = 2.0
BACKOFF_MAX_S = 120.0
CLOSE_TIMEOUT_S = 3.0
LATENCY_BUCKETS_S = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class TamUnavailable(Exception):
    """TAM cannot be reached right now; callers degrade instead of failing the agent."""


class CallMetrics:
    """Counter per (tool, outcome) plus a latency histogram per tool."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: Counter[tuple[str, str]] = Counter()
        self.latency: Counter[tuple[str, str]] = Counter()

    def observe(self, tool: str, outcome: str, seconds: float) -> None:
        index = bisect.bisect_left(LATENCY_BUCKETS_S, seconds)
        bucket = str(LATENCY_BUCKETS_S[index]) if index < len(LATENCY_BUCKETS_S) else "+Inf"
        with self._lock:
            self.calls[(tool, outcome)] += 1
            self.latency[(tool, bucket)] += 1

    def count(self, tool: str, outcome: str) -> int:
        with self._lock:
            return self.calls[(tool, outcome)]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": {f"{tool}:{outcome}": n for (tool, outcome), n in sorted(self.calls.items())},
                "latency_seconds": {f"{tool}:le={bucket}": n for (tool, bucket), n in sorted(self.latency.items())},
            }


class TamConnection:
    """Thread-safe owner of the MCP client. ``get(wait)`` never blocks longer than ``wait``."""

    def __init__(
        self,
        factory: Callable[[], McpClient],
        *,
        startup_timeout: float,
        spawn_thread: Callable[..., threading.Thread],
        clock: Callable[[], float] = time.monotonic,
        backoff_initial: float = BACKOFF_INITIAL_S,
        backoff_max: float = BACKOFF_MAX_S,
    ) -> None:
        self._factory = factory
        self._startup_timeout = startup_timeout
        self._spawn_thread = spawn_thread
        self._clock = clock
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._cond = threading.Condition()
        self._client: McpClient | None = None
        self._backend: Backend | None = None
        self._connecting = False
        self._closed = False
        self._failures = 0
        self._next_attempt = 0.0
        self.last_error = ""
        self.metrics = CallMetrics()

    @property
    def backend(self) -> Backend | None:
        with self._cond:
            return self._backend

    @property
    def connected(self) -> bool:
        with self._cond:
            return self._client is not None and self._client.alive

    def warm_up(self) -> None:
        """Start connecting in the background without waiting."""
        self.get(0.0)

    def get(self, wait: float) -> tuple[McpClient, Backend] | None:
        deadline = self._clock() + wait
        with self._cond:
            while True:
                if self._closed:
                    return None
                client = self._client
                if client is not None and client.alive and self._backend is not None:
                    return client, self._backend
                if client is not None:
                    self._drop_locked(client, "TAM process exited")
                if not self._connecting and self._clock() >= self._next_attempt:
                    self._connecting = True
                    self._spawn_thread(self._connect, name="tam-connect").start()
                remaining = deadline - self._clock()
                if remaining <= 0 or not self._connecting:
                    return None
                self._cond.wait(remaining)

    def call(
        self,
        tool: str,
        build_arguments: Callable[[Backend], Mapping[str, Any]],
        *,
        timeout: float,
        wait: float,
        shared_deadline: bool = False,
    ) -> tuple[Any, Backend]:
        """Call ``tool`` with arguments shaped for the connected backend.

        ``wait`` bounds waiting for a (re)connect; with ``shared_deadline`` the connect wait and the
        call together stay within ``timeout``.
        """
        waited_from = self._clock()
        pair = self.get(min(wait, timeout) if shared_deadline else wait)
        if pair is None:
            raise TamUnavailable(self.last_error or "TAM is not connected yet")
        client, backend = pair
        started = self._clock()
        if shared_deadline:
            timeout -= started - waited_from
            if timeout <= 0:
                raise TamUnavailable(f"{tool} deadline spent waiting for TAM to start")
        try:
            result = client.call_tool(tool, build_arguments(backend), timeout)
        except ToolCallError:
            self.metrics.observe(tool, "tool_error", self._clock() - started)
            raise
        except (McpTimeout, RpcError) as exc:
            # A timed-out stdio request leaves the server usable; the next call proceeds normally.
            self.metrics.observe(
                tool, "timeout" if isinstance(exc, McpTimeout) else "rpc_error", self._clock() - started
            )
            raise TamUnavailable(str(exc)) from exc
        except McpError as exc:
            self.metrics.observe(tool, "transport_error", self._clock() - started)
            with self._cond:
                self._drop_locked(client, str(exc))
            raise TamUnavailable(str(exc)) from exc
        self.metrics.observe(tool, "ok", self._clock() - started)
        return result, backend

    def close(self) -> None:
        with self._cond:
            self._closed = True
            client, self._client = self._client, None
            self._cond.notify_all()
        if client is not None:
            client.close(CLOSE_TIMEOUT_S)

    def _connect(self) -> None:
        client: McpClient | None = None
        error = ""
        backend: Backend | None = None
        started = self._clock()
        try:
            client = self._factory()
            info = client.connect(self._startup_timeout)
            backend = detect_backend(info.tools)
        except (McpError, ContractError) as exc:
            error = str(exc)
        if error and client is not None:
            client.close(CLOSE_TIMEOUT_S)
        with self._cond:
            self._connecting = False
            if self._closed:
                if client is not None and not error:
                    client.close(CLOSE_TIMEOUT_S)
            elif error or client is None or backend is None:
                self._failures += 1
                delay = min(self._backoff_max, self._backoff_initial * 2 ** (self._failures - 1))
                self._next_attempt = self._clock() + delay
                self.last_error = error
                logger.warning("tam.connect_failed attempt=%d retry_in_s=%.0f error=%s", self._failures, delay, error)
            else:
                self._client, self._backend = client, backend
                self._failures = 0
                self.last_error = ""
                server = client.server
                logger.info(
                    "tam.connected backend=%s server=%s version=%s protocol=%s seconds=%.2f",
                    backend.value,
                    server.name if server else "",
                    server.version if server else "",
                    server.protocol_version if server else "",
                    self._clock() - started,
                )
            self._cond.notify_all()

    def _drop_locked(self, client: McpClient, reason: str) -> None:
        if self._client is not client:
            return
        self._client = None
        self._failures += 1
        delay = min(self._backoff_max, self._backoff_initial * 2 ** (self._failures - 1))
        self._next_attempt = self._clock() + delay
        self.last_error = reason
        logger.warning("tam.connection_lost retry_in_s=%.0f reason=%s", delay, reason)
        self._spawn_thread(lambda: client.close(CLOSE_TIMEOUT_S), name="tam-close").start()
