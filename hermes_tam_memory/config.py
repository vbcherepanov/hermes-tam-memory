"""Plugin configuration: ``$HERMES_HOME/tam.json`` (non-secret) plus the ``TAM_API_TOKEN`` secret."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "tam.json"
TOKEN_ENV = "TAM_API_TOKEN"
DEFAULT_COMMAND = "tam"
DEFAULT_PROJECT = "hermes"
MODES = ("local", "remote")
RECALL_SCOPES = ("all", "project")

Mode = Literal["local", "remote"]
RecallScope = Literal["all", "project"]

# (minimum, maximum) for every numeric option; out-of-range values are clamped, not rejected.
_BOUNDS: dict[str, tuple[float, float]] = {
    "recall_limit": (1, 20),
    "recall_budget_tokens": (100, 8000),
    "min_turn_chars": (0, 2000),
    "max_turn_chars": (500, 50000),
    "request_timeout": (1.0, 120.0),
    "startup_timeout": (5.0, 600.0),
    "recall_timeout": (0.5, 7.5),
    "shutdown_timeout": (0.5, 30.0),
    "pending_limit": (10, 5000),
}


@dataclass(frozen=True)
class TamConfig:
    mode: Mode = "local"
    command: str = DEFAULT_COMMAND
    args: tuple[str, ...] = ()
    memory_dir: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    url: str = ""
    project: str = DEFAULT_PROJECT
    auto_capture: bool = True
    auto_recall: bool = True
    recall_scope: RecallScope = "all"
    recall_limit: int = 6
    recall_budget_tokens: int = 800
    min_turn_chars: int = 80
    max_turn_chars: int = 6000
    request_timeout: float = 15.0
    startup_timeout: float = 90.0
    # Must stay under Hermes' 8 s external prefetch deadline so a slow recall degrades to "" here.
    recall_timeout: float = 5.0
    shutdown_timeout: float = 5.0
    pending_limit: int = 200

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["args"] = list(self.args)
        data["env"] = dict(self.env)
        return data


def config_path(hermes_home: str | os.PathLike[str]) -> Path:
    return Path(hermes_home) / CONFIG_FILENAME


def parse_config(raw: Mapping[str, Any]) -> TamConfig:
    """Build a config from untrusted JSON: unknown keys and bad values fall back to defaults with a warning."""
    defaults = TamConfig()
    known = {f.name for f in fields(TamConfig)}
    for key in sorted(set(raw) - known):
        logger.warning("tam.config_unknown_key key=%s", key)
    values: dict[str, Any] = {}
    for f in fields(TamConfig):
        if f.name not in raw or raw[f.name] is None:
            continue
        default = getattr(defaults, f.name)
        parsed = _coerce(f.name, raw[f.name], default)
        if parsed is None:
            logger.warning("tam.config_invalid_value key=%s value=%r using_default=%r", f.name, raw[f.name], default)
            continue
        values[f.name] = parsed
    return TamConfig(**values)


def load_config(hermes_home: str | os.PathLike[str]) -> TamConfig:
    from utils import read_json_or_empty

    return parse_config(read_json_or_empty(config_path(hermes_home)))


def save_config(values: Mapping[str, Any], hermes_home: str | os.PathLike[str]) -> TamConfig:
    """Merge ``values`` into ``tam.json`` (validated round-trip) and return the effective config."""
    from utils import atomic_json_write, read_json_or_empty

    path = config_path(hermes_home)
    merged = {**read_json_or_empty(path), **{k: v for k, v in values.items() if k != "token"}}
    config = parse_config(merged)
    atomic_json_write(path, config.to_json(), mode=0o600, sort_keys=True)
    return config


def read_token() -> str:
    from agent.secret_scope import get_secret

    return (get_secret(TOKEN_ENV, "") or "").strip()


def _coerce(name: str, value: Any, default: Any) -> Any | None:
    if isinstance(default, bool):
        return _as_bool(value)
    if isinstance(default, (int, float)):
        return _as_number(name, value, type(default))
    if name == "mode":
        return value if value in MODES else None
    if name == "recall_scope":
        return value if value in RECALL_SCOPES else None
    if name == "args":
        return tuple(value) if isinstance(value, list) and all(isinstance(v, str) for v in value) else None
    if name == "env":
        ok = isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())
        return dict(value) if ok else None
    if isinstance(default, str):
        return value.strip() if isinstance(value, str) else None
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return None


def _as_number(name: str, value: Any, kind: type) -> float | int | None:
    if isinstance(value, bool):
        return None
    try:
        number = kind(value)
    except (TypeError, ValueError):
        return None
    low, high = _BOUNDS[name]
    return kind(min(max(number, low), high))
