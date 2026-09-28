from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from support import FAKE_SERVER

from hermes_tam_memory.mcp_client import (
    HttpTransport,
    McpClient,
    McpTimeout,
    RpcError,
    StdioTransport,
    ToolCallError,
    TransportClosed,
)


def _thread(target, *, name: str, daemon: bool = True, args: tuple = (), kwargs: dict[str, Any] | None = None):
    return threading.Thread(target=target, name=name, daemon=daemon, args=args, kwargs=kwargs)


def _stdio_client(tmp_path: Path, **env: str) -> McpClient:
    environment = {"FAKE_TAM_STORE": str(tmp_path / "store.json"), **env}
    return McpClient(
        StdioTransport([sys.executable, str(FAKE_SERVER)], environment, spawn_thread=_thread), client_version="test"
    )


class TestStdio:
    def test_handshake_save_and_recall(self, tmp_path):
        client = _stdio_client(tmp_path)
        info = client.connect(10)
        assert info.name == "total-agent-memory" and "memory_save" in info.tools
        assert client.call_tool("memory_save", {"content": "billing uses postgres"}, 5)["saved"] is True
        result = client.call_tool("memory_recall", {"query": "billing postgres"}, 5)
        assert result["results"]["fact"][0]["content"] == "billing uses postgres"
        client.close(3)
        assert not client.alive

    def test_tool_error_is_raised(self, tmp_path):
        client = _stdio_client(tmp_path)
        client.connect(10)
        with pytest.raises(ToolCallError):
            client.call_tool("memory_stats", {}, 5)
        client.close(3)

    def test_server_exit_fails_pending_and_later_calls(self, tmp_path):
        client = _stdio_client(tmp_path, FAKE_TAM_EXIT_AFTER="1")
        client.connect(10)
        client.call_tool("memory_save", {"content": "one"}, 5)
        with pytest.raises(TransportClosed):
            client.call_tool("memory_save", {"content": "two"}, 5)
        assert not client.alive

    def test_timeout(self, tmp_path):
        client = _stdio_client(tmp_path, FAKE_TAM_RECALL_DELAY="2")
        client.connect(10)
        with pytest.raises(McpTimeout):
            client.call_tool("memory_recall", {"query": "x"}, 0.2)
        client.close(3)

    def test_missing_command(self):
        transport = StdioTransport(["/nonexistent/tam-binary"], {}, spawn_thread=_thread)
        with pytest.raises(TransportClosed):
            McpClient(transport, client_version="t").connect(1)


class _McpHttpHandler(BaseHTTPRequestHandler):
    server: _McpHttpServer

    def log_message(self, *_: Any) -> None:
        return

    def do_DELETE(self) -> None:
        self.server.deleted = True
        self.send_response(200)
        self.end_headers()

    def do_POST(self) -> None:
        state = self.server
        if self.path == "/mcp":
            self.send_response(307)
            self.send_header("Location", "/mcp/")
            self.end_headers()
            return
        if state.token and self.headers.get("Authorization") != f"Bearer {state.token}":
            self._reply(401, "application/json", b'{"error":"Bearer token required"}')
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        state.requests.append(
            {
                "body": body,
                "session": self.headers.get("Mcp-Session-Id"),
                "protocol": self.headers.get("MCP-Protocol-Version"),
            }
        )
        if state.expire_session and body.get("method") == "tools/call":
            self._reply(404, "application/json", b"{}")
            return
        if "id" not in body:
            self._reply(202, "application/json", b"")
            return
        result: dict[str, Any]
        if body["method"] == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": "total-agent-memory", "version": "h"},
                "capabilities": {},
            }
        elif body["method"] == "tools/list":
            result = {"tools": [{"name": "memory_save"}, {"name": "memory_recall"}, {"name": "memory_scopes"}]}
        else:
            result = {"content": [{"type": "text", "text": json.dumps({"echo": body["params"]["arguments"]})}]}
        message = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode()
        headers = {"Mcp-Session-Id": "sess-1"} if body["method"] == "initialize" else {}
        if state.sse:
            ping = b'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
            self._reply(200, "text/event-stream", ping + b"event: message\ndata: " + message + b"\n\n", headers)
        else:
            self._reply(200, "application/json", message, headers)

    def _reply(self, status: int, content_type: str, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)


class _McpHttpServer(ThreadingHTTPServer):
    def __init__(self, *, sse: bool = False, token: str = "", expire_session: bool = False) -> None:
        super().__init__(("127.0.0.1", 0), _McpHttpHandler)
        self.sse, self.token, self.expire_session = sse, token, expire_session
        self.requests: list[dict[str, Any]] = []
        self.deleted = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/mcp"


@pytest.fixture
def http_server(request) -> Iterator[_McpHttpServer]:
    server = _McpHttpServer(**getattr(request, "param", {}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


class TestHttp:
    @pytest.mark.parametrize("http_server", [{"sse": False}, {"sse": True}], indirect=True)
    def test_json_and_sse_replies_follow_redirect_and_keep_session(self, http_server):
        client = McpClient(HttpTransport(http_server.url), client_version="t")
        info = client.connect(5)
        assert "memory_scopes" in info.tools
        assert client.call_tool("memory_recall", {"query": "q"}, 5) == {"echo": {"query": "q"}}
        last = http_server.requests[-1]
        assert last["session"] == "sess-1" and last["protocol"] == "2025-06-18"
        client.close(2)
        assert http_server.deleted

    @pytest.mark.parametrize("http_server", [{"token": "s3cret"}], indirect=True)
    def test_bearer_token(self, http_server):
        with pytest.raises(RpcError) as info:
            McpClient(HttpTransport(http_server.url + "/"), client_version="t").connect(5)
        assert info.value.code == 401
        client = McpClient(HttpTransport(http_server.url + "/", token="s3cret"), client_version="t")
        assert client.connect(5).name == "total-agent-memory"

    @pytest.mark.parametrize("http_server", [{"expire_session": True}], indirect=True)
    def test_expired_session_closes_transport(self, http_server):
        client = McpClient(HttpTransport(http_server.url + "/"), client_version="t")
        client.connect(5)
        with pytest.raises(TransportClosed):
            client.call_tool("memory_recall", {"query": "q"}, 5)
        assert not client.alive

    def test_unreachable_and_bad_scheme(self):
        with pytest.raises(TransportClosed):
            McpClient(HttpTransport("http://127.0.0.1:9/mcp/"), client_version="t").connect(2)
        with pytest.raises(TransportClosed):
            McpClient(HttpTransport("ftp://example/mcp"), client_version="t").connect(2)
