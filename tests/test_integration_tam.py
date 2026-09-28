"""End-to-end against a real TAM install, isolated in temp directories.

Opt-in: set TAM_INTEGRATION_COMMAND to a ``tam`` executable (and TAM_INTEGRATION_TEAM_COMMAND to
``tam-team`` for the team-server test). Nothing here touches the caller's own TAM data: HOME and
TAM_MEMORY_DIR point into pytest's tmp_path, and provider API keys are not passed to TAM.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import urllib.request
from pathlib import Path

import pytest
from agent.memory_manager import MemoryManager
from support import wait_until

from hermes_tam_memory.provider import TOOL_RECALL, TOOL_SAVE, TamMemoryProvider

TAM_COMMAND = os.environ.get("TAM_INTEGRATION_COMMAND", "")
TEAM_COMMAND = os.environ.get("TAM_INTEGRATION_TEAM_COMMAND", "")
FIRST_Q = "What did we decide about the message broker for the order pipeline?"
FIRST_A = "We chose RabbitMQ 4 with quorum queues for the order pipeline; Kafka was rejected as too heavy."

pytestmark = pytest.mark.integration


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    env = {"HOME": str(tmp_path / "tam-home"), "MEMORY_QUALITY_GATE_ENABLED": "auto"}
    if os.environ.get("TAM_MODEL_CACHE"):
        env["TAM_MODEL_CACHE"] = os.environ["TAM_MODEL_CACHE"]
    return env


def _run_session(hermes_home: Path, session_id: str, turns: list[tuple[str, str]], query: str) -> str:
    manager = MemoryManager()
    manager.add_provider(TamMemoryProvider())
    manager.initialize_all(session_id=session_id, platform="cli", hermes_home=str(hermes_home))
    provider = manager.get_provider("tam")
    wait_until(lambda: provider._connection.connected, timeout=120)
    context = manager.prefetch_all(query) if query else ""
    for user, assistant in turns:
        manager.sync_all(user, assistant, session_id=session_id)
    assert manager.flush_pending(30)
    assert provider.flush(60)
    manager.on_session_end([])
    manager.shutdown_all()
    return context


@pytest.mark.skipif(not TAM_COMMAND, reason="TAM_INTEGRATION_COMMAND not set")
def test_local_tam_turn_survives_into_next_session(hermes_home, tmp_path):
    config = {
        "mode": "local",
        "command": TAM_COMMAND,
        "memory_dir": str(tmp_path / "tam-data"),
        "env": _isolated_env(tmp_path),
        "project": "orders",
        "recall_timeout": 7.5,
    }
    (hermes_home / "tam.json").write_text(json.dumps(config))

    _run_session(hermes_home, "it-1", [(FIRST_Q, FIRST_A)], query="")
    context = _run_session(hermes_home, "it-2", [], query="which broker do we use for orders?")
    assert "RabbitMQ 4" in context

    provider = TamMemoryProvider()
    provider.initialize("it-3", hermes_home=str(hermes_home))
    try:
        saved = json.loads(
            provider.handle_tool_call(
                TOOL_SAVE,
                {
                    "content": "Order pipeline retries use exponential backoff capped at 5 minutes.",
                    "type": "convention",
                },
            )
        )
        assert saved["saved"] is True
        found = json.loads(provider.handle_tool_call(TOOL_RECALL, {"query": "order pipeline retries backoff"}))
        assert any("exponential backoff" in r["content"] for r in found["results"])
    finally:
        provider.shutdown()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.skipif(not TEAM_COMMAND, reason="TAM_INTEGRATION_TEAM_COMMAND not set")
def test_team_server_with_bearer_token(hermes_home, tmp_path):
    # No provider keys reach the server: TAM must not make paid LLM calls from a test.
    inherited = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_TOKEN", "_SECRET"))}
    env = {**inherited, **_isolated_env(tmp_path)}
    root = tmp_path / "team"
    subprocess.run([TEAM_COMMAND, "--root", str(root), "user-add", "alice", "Alice"], env=env, check=True)
    token_file = tmp_path / "token"
    subprocess.run(
        [TEAM_COMMAND, "--root", str(root), "token-create", "--client", "hermes", "--out", str(token_file), "alice"],
        env=env,
        check=True,
    )
    port = _free_port()
    log = (tmp_path / "team.log").open("wb")
    server = subprocess.Popen(
        [TEAM_COMMAND, "--root", str(root), "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_until(lambda: _healthy(port), timeout=120, interval=0.5)
        config = {"mode": "remote", "url": f"http://127.0.0.1:{port}/mcp", "project": "orders", "recall_timeout": 7.5}
        (hermes_home / "tam.json").write_text(json.dumps(config))
        os.environ["TAM_API_TOKEN"] = token_file.read_text().strip()
        _run_session(hermes_home, "team-1", [(FIRST_Q, FIRST_A)], query="")
        context = _run_session(hermes_home, "team-2", [], query="which broker do we use for orders?")
        assert "RabbitMQ 4" in context and "by Alice" in context
    finally:
        os.environ.pop("TAM_API_TOKEN", None)
        server.terminate()
        server.wait(timeout=30)
        log.close()


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


@pytest.mark.skipif(not TAM_COMMAND, reason="TAM_INTEGRATION_COMMAND not set")
def test_local_tam_over_http_transport(hermes_home, tmp_path):
    port = _free_port()
    inherited = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_TOKEN", "_SECRET"))}
    env = {
        **inherited,
        **_isolated_env(tmp_path),
        "TAM_MEMORY_DIR": str(tmp_path / "tam-http-data"),
        "MCP_TRANSPORT": "http",
        "MCP_HTTP_HOST": "127.0.0.1",
        "MCP_HTTP_PORT": str(port),
    }
    log = (tmp_path / "tam-http.log").open("wb")
    server = subprocess.Popen([TAM_COMMAND], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_until(lambda: _port_open(port), timeout=120, interval=0.5)
        config = {"mode": "remote", "url": f"http://127.0.0.1:{port}/mcp", "project": "orders", "recall_timeout": 7.5}
        (hermes_home / "tam.json").write_text(json.dumps(config))
        _run_session(hermes_home, "http-1", [(FIRST_Q, FIRST_A)], query="")
        context = _run_session(hermes_home, "http-2", [], query="which broker do we use for orders?")
        assert "RabbitMQ 4" in context
    finally:
        server.terminate()
        server.wait(timeout=30)
        log.close()


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0
