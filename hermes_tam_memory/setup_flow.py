"""``hermes memory setup tam``: plain line prompts (scriptable), config write, live connection check."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

from agent.memory_provider import spawn_context_thread

from . import __version__
from .backend import ContractError, detect_backend
from .config import MODES, TOKEN_ENV, TamConfig, config_path, load_config, read_token, save_config
from .connection import CLOSE_TIMEOUT_S
from .mcp_client import HttpTransport, McpClient, McpError, StdioTransport, Transport
from .provider import TamMemoryProvider, child_environment, resolve_command

MAX_PROMPT_ATTEMPTS = 3
CLEAR_VALUE = "-"
DEFAULT_REMOTE_URL_EXAMPLE = "http://127.0.0.1:3737/mcp/"


def _ask(
    label: str,
    default: str = "",
    choices: Sequence[str] | None = None,
    read_line: Callable[[], str] = lambda: sys.stdin.readline(),
) -> str:
    suffix = f" [{default}]" if default else ""
    for _ in range(MAX_PROMPT_ATTEMPTS):
        sys.stdout.write(f"  {label}{suffix}: ")
        sys.stdout.flush()
        value = read_line().strip() or default
        if choices is None or value in choices:
            return value
        print(f"  Please answer one of: {', '.join(choices)}")
    print(f"  Keeping {default!r}.")
    return default


def _ask_secret(label: str) -> str:
    from hermes_cli.secret_prompt import masked_secret_prompt

    return masked_secret_prompt(f"  {label}: ").strip()


def _write_token(token: str, hermes_home: str) -> None:
    from hermes_cli.config import save_env_value
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    marker = set_hermes_home_override(hermes_home)
    try:
        save_env_value(TOKEN_ENV, token)
    finally:
        reset_hermes_home_override(marker)


def _activate(hermes_config: dict[str, Any], provider_name: str) -> None:
    from hermes_cli.config import save_config as save_hermes_config

    memory = hermes_config.get("memory")
    if not isinstance(memory, dict):
        memory = hermes_config["memory"] = {}
    memory["provider"] = provider_name
    save_hermes_config(hermes_config)


def collect_values(current: TamConfig) -> dict[str, Any]:
    """Prompt for the settings a user must choose; everything else keeps its value in tam.json."""
    print("\n  Configuring total-agent-memory (TAM):\n")
    print("  local  = Hermes starts your installed `tam` command (data stays on this machine)")
    print("  remote = Hermes connects to a TAM HTTP endpoint or a team server (URL + token)\n")
    values: dict[str, Any] = {"mode": _ask("Connection mode (local/remote)", current.mode, MODES)}
    if values["mode"] == "local":
        values["command"] = _ask("TAM command or absolute path", current.command)
        memory_dir = _ask("TAM data directory (blank keeps the current value, '-' = TAM default)", current.memory_dir)
        values["memory_dir"] = "" if memory_dir == CLEAR_VALUE else memory_dir
    else:
        values["url"] = _ask(f"TAM MCP URL (e.g. {DEFAULT_REMOTE_URL_EXAMPLE})", current.url)
    values["project"] = _ask("TAM project name for Hermes memories", current.project)
    return values


def run_setup(provider: TamMemoryProvider, hermes_home: str, hermes_config: dict[str, Any]) -> None:
    current = load_config(hermes_home)
    values = collect_values(current)
    token = ""
    if values["mode"] == "remote":
        token = _ask_secret("Team server token (blank to keep the current one / none)")
    config = save_config(values, hermes_home)
    _activate(hermes_config, provider.name)
    if token:
        _write_token(token, hermes_home)
    print(f"\n  Saved {config_path(hermes_home)}")
    print(f"  Memory provider: {provider.name} (activation saved to config.yaml)")
    if token:
        print(f"  {TOKEN_ENV} saved to .env")
    print(f"\n  {check_connection(config, token)}")
    print("\n  Start a new session to activate.\n")


def check_connection(config: TamConfig, token: str = "") -> str:
    """Connect once with the new settings and report the result in one line."""
    transport: Transport
    if config.mode == "remote":
        if not config.url:
            return "Not checked: no URL configured."
        transport = HttpTransport(config.url, token=token or read_token())
    else:
        command = resolve_command(config)
        if command is None:
            return (
                f"TAM command {config.command!r} not found. Install it with "
                "`pipx install total-agent-memory` (or `uv tool install total-agent-memory`)."
            )
        transport = StdioTransport(
            [command, *config.args], child_environment(config, os.environ), spawn_thread=spawn_context_thread
        )
    client = McpClient(transport, client_version=__version__)
    try:
        info = client.connect(config.startup_timeout)
        backend = detect_backend(info.tools)
    except (McpError, ContractError) as exc:
        return f"Could not reach TAM: {exc}"
    finally:
        client.close(CLOSE_TIMEOUT_S)
    return f"Connected: {info.name} {info.version} ({backend.value} server, MCP {info.protocol_version})."
