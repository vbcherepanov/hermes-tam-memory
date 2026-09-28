from __future__ import annotations

from pathlib import Path

import pytest
from support import FakeTam


@pytest.fixture
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.delenv("TAM_API_TOKEN", raising=False)
    monkeypatch.delenv("TAM_MEMORY_DIR", raising=False)
    return hermes


@pytest.fixture
def fake_tam(hermes_home: Path, tmp_path: Path) -> FakeTam:
    fake = FakeTam(hermes_home=hermes_home, store=tmp_path / "store.json", log=tmp_path / "calls.jsonl")
    fake.configure()
    return fake
