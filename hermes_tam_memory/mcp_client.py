"""Minimal MCP client (JSON-RPC 2.0) over stdio or streamable HTTP, stdlib only.

TAM is reached through the same MCP surface every other agent uses, so the plugin adds
no Python dependencies to the Hermes environment and works with any TAM install
(pipx, uv tool, Docker, team server) without importing TAM's heavy runtime.
"""

from __future__ import annotations

import itertools
import json
import logging
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = frozenset({"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"})
CLIENT_NAME = "hermes-tam-memory"
JSONRPC_METHOD_NOT_FOUND = -32601
STDERR_LOG_LINE_CHARS = 500
HTTP_MAX_REDIRECTS = 2
HTTP_REDIRECT_CODES = frozenset({301, 302, 307, 308})
SESSION_HEADER = "Mcp-Session-Id"
PROTOCOL_HEADER = "MCP-Protocol-Version"


class McpError(Exception):
    """Base error for MCP client failures."""


class TransportClosed(McpError):
    """The server went away; the connection must be rebuilt."""


class McpTimeout(McpError):
    """The server did not answer within the deadline."""


class RpcError(McpError):
    """The server answered with a JSON-RPC error object."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code = code


class ToolCallError(McpError):
    """The tool ran and reported ``isError``."""


class Transport(Protocol):
    def start(self) -> None: ...

    def request(self, method: str, params: Mapping[str, Any], timeout: float) -> dict[str, Any]: ...

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None: ...

    def set_protocol_version(self, version: str) -> None: ...

    @property
    def alive(self) -> bool: ...

    def close(self, timeout: float) -> None: ...


class _Pending:
    __slots__ = ("event", "message")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.message: dict[str, Any] | None = None


class StdioTransport:
    """Newline-delimited JSON-RPC to a spawned MCP server process."""

    def __init__(
        self,
        argv: Sequence[str],
        env: Mapping[str, str],
        *,
        spawn_thread: Callable[..., threading.Thread],
        cwd: str | None = None,
    ) -> None:
        self._argv = list(argv)
        self._env = dict(env)
        self._cwd = cwd
        self._spawn_thread = spawn_thread
        self._proc: subprocess.Popen[str] | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._closed = threading.Event()

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                self._argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._env,
                cwd=self._cwd,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise TransportClosed(f"cannot start {self._argv[0]!r}: {exc}") from exc
        self._spawn_thread(self._read_stdout, name="tam-mcp-stdout").start()
        self._spawn_thread(self._read_stderr, name="tam-mcp-stderr").start()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None and not self._closed.is_set()

    def set_protocol_version(self, version: str) -> None:
        """Stdio carries no per-request version header."""

    def request(self, method: str, params: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        request_id = next(self._ids)
        slot = _Pending()
        with self._pending_lock:
            if self._closed.is_set():
                raise TransportClosed("stdio transport is closed")
            self._pending[request_id] = slot
        try:
            self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)})
            if not slot.event.wait(timeout):
                raise McpTimeout(f"{method} timed out after {timeout:.1f}s")
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)
        if slot.message is None:
            raise TransportClosed(f"server exited during {method}")
        return _unwrap(slot.message)

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = dict(params)
        self._write(message)

    def close(self, timeout: float) -> None:
        proc = self._proc
        self._mark_closed()
        if proc is None or proc.poll() is not None:
            return
        # Closing stdin is the MCP stdio shutdown signal; escalate only if the server ignores it.
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError as exc:
            logger.debug("tam.stdio_close_stdin_failed error=%s", exc)
        for stop in (None, proc.terminate, proc.kill):
            if stop is not None:
                stop()
            try:
                proc.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                continue
        logger.warning("tam.stdio_process_not_reaped pid=%s", proc.pid)

    def _write(self, message: Mapping[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or not self.alive:
            raise TransportClosed("stdio transport is not running")
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            with self._write_lock:
                proc.stdin.write(line)
                proc.stdin.flush()
        except (OSError, ValueError) as exc:
            self._mark_closed()
            raise TransportClosed(f"write to TAM failed: {exc}") from exc

    def _read_stdout(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for line in proc.stdout:
                line = line.strip()
                if line:
                    self._dispatch(line)
        except (OSError, ValueError) as exc:
            logger.debug("tam.stdio_read_failed error=%s", exc)
        finally:
            self._mark_closed()

    def _read_stderr(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        try:
            for line in proc.stderr:
                if line.strip():
                    logger.debug("tam.server_stderr line=%s", line.rstrip()[:STDERR_LOG_LINE_CHARS])
        except (OSError, ValueError) as exc:
            logger.debug("tam.stdio_stderr_failed error=%s", exc)

    def _dispatch(self, line: str) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("tam.stdio_non_json_line line=%s", line[:STDERR_LOG_LINE_CHARS])
            return
        if not isinstance(message, dict):
            return
        if "method" in message:
            self._answer_server_request(message)
            return
        request_id = message.get("id")
        with self._pending_lock:
            slot = self._pending.get(request_id) if isinstance(request_id, int) else None
        if slot is not None:
            slot.message = message
            slot.event.set()

    def _answer_server_request(self, message: Mapping[str, Any]) -> None:
        if "id" not in message:
            return
        if message.get("method") == "ping":
            reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"], "result": {}}
        else:
            reply = {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": JSONRPC_METHOD_NOT_FOUND, "message": "method not supported by client"},
            }
        try:
            self._write(reply)
        except TransportClosed as exc:
            logger.debug("tam.stdio_reply_failed method=%s error=%s", message.get("method"), exc)

    def _mark_closed(self) -> None:
        with self._pending_lock:
            self._closed.set()
            waiting = list(self._pending.values())
        for slot in waiting:
            slot.event.set()


class HttpTransport:
    """MCP streamable HTTP: one POST per message, JSON or single-response SSE replies."""

    def __init__(self, url: str, *, token: str = "", opener: urllib.request.OpenerDirector | None = None) -> None:
        self._url = url
        self._token = token
        self._session_id = ""
        self._protocol_version = ""
        self._ids = itertools.count(1)
        self._closed = False
        self._opener = opener or urllib.request.build_opener(_NoRedirect())

    def start(self) -> None:
        if not self._url.startswith(("http://", "https://")):
            raise TransportClosed(f"TAM url must be http(s): {self._url!r}")

    @property
    def alive(self) -> bool:
        return not self._closed

    def set_protocol_version(self, version: str) -> None:
        self._protocol_version = version

    def request(self, method: str, params: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        request_id = next(self._ids)
        body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)}
        status, content_type, payload = self._post(body, timeout)
        if status == 202 or not payload:
            raise RpcError(-32603, f"empty reply to {method} (HTTP {status})")
        return _unwrap(_select_reply(content_type, payload, request_id))

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = dict(params)
        self._post(message, timeout=10.0)

    def close(self, timeout: float) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._session_id:
            return
        request = urllib.request.Request(self._url, method="DELETE", headers=self._headers())
        try:
            with self._opener.open(request, timeout=timeout):
                pass
        except (urllib.error.URLError, OSError) as exc:
            logger.debug("tam.http_session_delete_failed error=%s", exc)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if self._session_id:
            headers[SESSION_HEADER] = self._session_id
        if self._protocol_version:
            headers[PROTOCOL_HEADER] = self._protocol_version
        return headers

    def _post(self, message: Mapping[str, Any], timeout: float) -> tuple[int, str, str]:
        if self._closed:
            raise TransportClosed("http transport is closed")
        data = json.dumps(message, ensure_ascii=False).encode("utf-8")
        url = self._url
        for _ in range(HTTP_MAX_REDIRECTS + 1):
            request = urllib.request.Request(url, data=data, method="POST", headers=self._headers())
            try:
                with self._opener.open(request, timeout=timeout) as response:
                    session_id = response.headers.get(SESSION_HEADER)
                    if session_id:
                        self._session_id = session_id
                    payload = response.read().decode("utf-8", errors="replace")
                    return response.status, response.headers.get("Content-Type", ""), payload
            except urllib.error.HTTPError as exc:
                location = exc.headers.get("Location") if exc.headers else None
                if exc.code in HTTP_REDIRECT_CODES and location:
                    url = urllib.parse.urljoin(url, location)
                    continue
                detail = exc.read().decode("utf-8", errors="replace")[:STDERR_LOG_LINE_CHARS]
                if exc.code == 404 and self._session_id:
                    self._closed = True
                    raise TransportClosed("TAM HTTP session expired") from exc
                if exc.code in (401, 403):
                    raise RpcError(exc.code, f"TAM rejected the token: {detail}") from exc
                raise TransportClosed(f"TAM HTTP {exc.code}: {detail}") from exc
            except TimeoutError as exc:
                raise McpTimeout(f"TAM HTTP request timed out after {timeout:.1f}s") from exc
            except (urllib.error.URLError, OSError) as exc:
                raise TransportClosed(f"TAM HTTP unreachable: {exc}") from exc
        raise TransportClosed(f"too many redirects from {self._url}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface redirects so POST bodies are re-sent explicitly (urllib drops them on 307/308)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _select_reply(content_type: str, payload: str, request_id: int) -> dict[str, Any]:
    if "text/event-stream" in content_type:
        for event in _sse_messages(payload):
            if event.get("id") == request_id:
                return event
        raise RpcError(-32603, "SSE stream ended without a reply")
    try:
        decoded: Any = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RpcError(-32700, f"invalid JSON from TAM: {exc}") from exc
    if isinstance(decoded, list):
        decoded = next((m for m in decoded if isinstance(m, dict) and m.get("id") == request_id), None)
    if not isinstance(decoded, dict):
        raise RpcError(-32603, "reply is not a JSON-RPC object")
    return decoded


def _sse_messages(payload: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    data_lines: list[str] = []
    for raw in [*payload.splitlines(), ""]:
        if raw.startswith("data:"):
            data_lines.append(raw[5:].lstrip(" "))
        elif raw == "" and data_lines:
            try:
                parsed = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                logger.debug("tam.sse_bad_event data=%s", "\n".join(data_lines)[:STDERR_LOG_LINE_CHARS])
            else:
                if isinstance(parsed, dict):
                    messages.append(parsed)
            data_lines = []
    return messages


def _unwrap(message: Mapping[str, Any]) -> dict[str, Any]:
    error = message.get("error")
    if isinstance(error, dict):
        raise RpcError(int(error.get("code", -32603)), str(error.get("message", "unknown error")))
    result = message.get("result")
    if not isinstance(result, dict):
        raise RpcError(-32603, "reply has no result object")
    return result


@dataclass(frozen=True)
class ServerInfo:
    name: str
    version: str
    protocol_version: str
    tools: frozenset[str]


class McpClient:
    """Handshake plus ``tools/call`` on top of a transport."""

    def __init__(self, transport: Transport, *, client_version: str) -> None:
        self._transport = transport
        self._client_version = client_version
        self.server: ServerInfo | None = None

    @property
    def alive(self) -> bool:
        return self._transport.alive

    def connect(self, timeout: float) -> ServerInfo:
        self._transport.start()
        result = self._transport.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": self._client_version},
            },
            timeout,
        )
        version = str(result.get("protocolVersion", ""))
        if version not in SUPPORTED_PROTOCOL_VERSIONS:
            raise McpError(f"unsupported MCP protocol version from server: {version!r}")
        self._transport.set_protocol_version(version)
        self._transport.notify("notifications/initialized")
        tools = self._transport.request("tools/list", {}, timeout).get("tools", [])
        server_info = result.get("serverInfo")
        info: dict[str, Any] = server_info if isinstance(server_info, dict) else {}
        self.server = ServerInfo(
            name=str(info.get("name", "")),
            version=str(info.get("version", "")),
            protocol_version=version,
            tools=frozenset(t["name"] for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str)),
        )
        return self.server

    def call_tool(self, name: str, arguments: Mapping[str, Any], timeout: float) -> Any:
        result = self._transport.request("tools/call", {"name": name, "arguments": dict(arguments)}, timeout)
        text = "".join(
            part.get("text", "")
            for part in result.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if result.get("isError"):
            raise ToolCallError(text or f"{name} failed")
        structured = result.get("structuredContent")
        if structured is not None and not text:
            return structured
        try:
            return json.loads(text) if text else structured
        except json.JSONDecodeError:
            return text

    def close(self, timeout: float) -> None:
        self._transport.close(timeout)
