from __future__ import annotations

import importlib.util
import io
import json
import sys

import pytest
from support import REPO_ROOT

from hermes_tam_memory.config import TamConfig
from hermes_tam_memory.provider import TamMemoryProvider
from hermes_tam_memory.setup_flow import check_connection

PLUGIN_DIR = REPO_ROOT / "hermes_tam_memory"


class TestSetupFlow:
    def test_scripted_local_setup_writes_config_and_activates(self, fake_tam, hermes_home, monkeypatch, capsys):
        # mode, command, memory dir, project; blank answers keep the pre-seeded fake command.
        monkeypatch.setattr(sys, "stdin", io.StringIO("local\n\n-\nbilling\n"))
        from hermes_cli.config import load_config

        TamMemoryProvider().post_setup(str(hermes_home), load_config())
        out = capsys.readouterr().out
        assert "Connected: total-agent-memory fake (local server" in out
        stored = json.loads((hermes_home / "tam.json").read_text())
        assert stored["project"] == "billing" and stored["memory_dir"] == "" and stored["command"] == sys.executable
        assert load_config()["memory"]["provider"] == "tam"

    def test_invalid_mode_is_reprompted(self, fake_tam, hermes_home, monkeypatch, capsys):
        monkeypatch.setattr(sys, "stdin", io.StringIO("cloud\nlocal\n\n\n\n"))
        from hermes_cli.config import load_config

        TamMemoryProvider().post_setup(str(hermes_home), load_config())
        assert "Please answer one of: local, remote" in capsys.readouterr().out

    def test_check_connection_reports_missing_command_and_url(self):
        assert "not found" in check_connection(TamConfig(command="definitely-not-a-tam-binary"))
        assert "no URL" in check_connection(TamConfig(mode="remote"))
        assert "Could not reach TAM" in check_connection(
            TamConfig(mode="remote", url="http://127.0.0.1:9/mcp/", startup_timeout=5)
        )

    def test_status_config_masks_token(self, fake_tam, hermes_home, monkeypatch):
        fake_tam.configure(mode="remote", url="http://127.0.0.1:3737/mcp/")
        monkeypatch.setenv("TAM_API_TOKEN", "tok-9f3a1c")
        status = TamMemoryProvider().get_status_config({})
        assert status["token"] == "set" and "tok-9f3a1c" not in json.dumps(status)


class TestPluginLayout:
    def test_directory_plugin_registers_provider_like_hermes_loader(self):
        spec = importlib.util.spec_from_file_location(
            "tam_dir_plugin_probe", PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        registered = []

        class Ctx:
            def register_memory_provider(self, provider):
                registered.append(provider)

        module.register(Ctx())
        assert [p.name for p in registered] == ["tam"]
        assert "MemoryProvider" in (PLUGIN_DIR / "__init__.py").read_text()[:8192]

    def test_versions_agree(self):
        import tomllib

        from hermes_tam_memory import __version__

        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        manifest = (PLUGIN_DIR / "plugin.yaml").read_text()
        assert pyproject["project"]["version"] == __version__
        assert f"version: {__version__}" in manifest
        assert "name: tam\n" in manifest

    def test_hermes_admission_validator_passes(self):
        validate = pytest.importorskip("hermes_cli.plugin_validate")
        report = validate.validate_plugin_dir(PLUGIN_DIR)
        assert report.ok, report.to_dict()
