"""Shared test helpers: the fake TAM configuration and polling."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_SERVER = Path(__file__).with_name("fake_tam_server.py")


def wait_until(predicate: Callable[[], Any], timeout: float = 10.0, interval: float = 0.02) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


@dataclass
class FakeTam:
    hermes_home: Path
    store: Path
    log: Path

    def configure(self, *, env: dict[str, str] | None = None, **values: Any) -> dict[str, Any]:
        config = {
            "mode": "local",
            "command": sys.executable,
            "args": [str(FAKE_SERVER)],
            "env": {"FAKE_TAM_STORE": str(self.store), "FAKE_TAM_LOG": str(self.log), **(env or {})},
            "startup_timeout": 10,
            "request_timeout": 5,
            "shutdown_timeout": 5,
            **values,
        }
        (self.hermes_home / "tam.json").write_text(json.dumps(config), encoding="utf-8")
        return config

    def calls(self, name: str | None = None) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        entries = [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]
        return [e for e in entries if name is None or e["name"] == name]

    def records(self) -> list[dict[str, Any]]:
        return json.loads(self.store.read_text()) if self.store.exists() else []
