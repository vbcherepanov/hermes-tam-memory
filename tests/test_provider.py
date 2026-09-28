from __future__ import annotations

import json
import logging

import pytest
from agent.memory_manager import MemoryManager
from support import wait_until

from hermes_tam_memory.config import TOKEN_ENV, TamConfig
from hermes_tam_memory.provider import (
    TOOL_RECALL,
    TOOL_SAVE,
    TamMemoryProvider,
    child_environment,
)
from hermes_tam_memory.text import RECALL_HEADER, session_context

BILLING_Q = "Which database does the billing service use in production?"
BILLING_A = "The billing service uses PostgreSQL 18 with logical replication to the analytics cluster."


def _started(provider: TamMemoryProvider, session_id: str = "s1", **kwargs) -> TamMemoryProvider:
    provider.initialize(session_id, **kwargs)
    wait_until(lambda: provider._connection is not None and provider._connection.connected)
    return provider


@pytest.fixture
def provider(fake_tam, hermes_home):
    instance = TamMemoryProvider()
    yield instance
    instance.shutdown()


class TestAvailability:
    def test_local_needs_resolvable_command(self, fake_tam, hermes_home):
        assert TamMemoryProvider().is_available()
        fake_tam.configure(command="definitely-not-a-tam-binary")
        provider = TamMemoryProvider()
        assert not provider.is_available()
        assert "not found on PATH" in provider.unavailable_reason()

    def test_remote_needs_url(self, fake_tam):
        fake_tam.configure(mode="remote", url="")
        assert not TamMemoryProvider().is_available()
        fake_tam.configure(mode="remote", url="http://127.0.0.1:3737/mcp/")
        assert TamMemoryProvider().is_available()

    def test_child_environment_excludes_hermes_secrets(self):
        environ = {
            "PATH": "/bin",
            "OPENAI_API_KEY": "sk-x",
            TOKEN_ENV: "t",
            "TAM_MEMORY_DIR": "/a",
            "MEMORY_MODE": "fast",
            "GITHUB_TOKEN": "g",
        }
        env = child_environment(TamConfig(memory_dir="/data", env={"EXTRA": "1"}), environ)
        assert env == {"PATH": "/bin", "TAM_MEMORY_DIR": "/data", "MEMORY_MODE": "fast", "EXTRA": "1"}


class TestCaptureAndRecall:
    def test_turn_is_saved_and_recalled_in_next_session(self, fake_tam, provider):
        _started(provider, "s1")
        provider.sync_turn(BILLING_Q, BILLING_A, session_id="s1")
        assert provider.flush(5)
        saves = fake_tam.calls("memory_save")
        assert len(saves) == 1
        args = saves[0]["arguments"]
        assert args["content"] == f"User: {BILLING_Q}\nAssistant: {BILLING_A}"
        assert args["source_format"] == "conversation" and args["project"] == "hermes"
        assert args["context"] == session_context("s1") and "request_id" not in args
        # Same session: its own turn is already in the transcript, so it is not injected again.
        assert provider.prefetch("billing database", session_id="s1") == ""
        provider.shutdown()

        later = _started(TamMemoryProvider(), "s2")
        try:
            block = later.prefetch("what database does billing use?", session_id="s2")
            assert block.startswith(RECALL_HEADER) and "PostgreSQL 18" in block
            assert later.recall_status().count == 1
        finally:
            later.shutdown()

    def test_noise_and_non_primary_contexts_are_not_saved(self, fake_tam, provider):
        _started(provider, "s1")
        provider.sync_turn("thanks!", "You're welcome — happy to help with anything else later on.")
        provider.sync_turn("2+2?", "4")
        assert provider.flush(5)
        assert fake_tam.calls("memory_save") == []
        assert provider.outcomes == {"skipped_trivial_prompt": 1, "skipped_too_short": 1}

        cron = _started(TamMemoryProvider(), "c1", agent_context="cron")
        cron.sync_turn(BILLING_Q, BILLING_A)
        assert cron.flush(5)
        cron.shutdown()
        assert fake_tam.calls("memory_save") == []

    def test_injected_recall_is_stripped_before_saving(self, fake_tam, provider):
        _started(provider)
        provider.sync_turn(f"<memory-context>old stuff</memory-context>{BILLING_Q}", BILLING_A)
        assert provider.flush(5)
        assert "old stuff" not in fake_tam.calls("memory_save")[0]["arguments"]["content"]

    def test_quality_gate_rejection_is_counted_not_raised(self, fake_tam, provider, caplog):
        fake_tam.configure(
            env={"FAKE_TAM_STORE": str(fake_tam.store), "FAKE_TAM_LOG": str(fake_tam.log), "FAKE_TAM_REJECT": "billing"}
        )
        _started(provider)
        with caplog.at_level(logging.INFO, logger="hermes_tam_memory"):
            provider.sync_turn(BILLING_Q, BILLING_A)
            assert provider.flush(5)
        assert provider.outcomes == {"rejected_quality_gate": 1}
        assert "tam.write_rejected" in caplog.text

    def test_builtin_memory_add_is_mirrored(self, fake_tam, provider):
        _started(provider)
        provider.on_memory_write("add", "user", "Prefers concise answers in Russian.")
        provider.on_memory_write("remove", "user", "Prefers concise answers in Russian.")
        assert provider.flush(5)
        saves = fake_tam.calls("memory_save")
        assert len(saves) == 1 and "hermes-user" in saves[0]["arguments"]["tags"]

    def test_recall_scope_project_filters(self, fake_tam, provider):
        fake_tam.configure(
            env={"FAKE_TAM_STORE": str(fake_tam.store), "FAKE_TAM_LOG": str(fake_tam.log)},
            recall_scope="project",
            project="billing",
        )
        _started(provider)
        provider.prefetch("anything about billing")
        assert fake_tam.calls("memory_recall")[0]["arguments"]["project"] == "billing"

    def test_trivial_prompt_skips_recall(self, fake_tam, provider):
        _started(provider)
        assert provider.prefetch("ok") == ""
        assert fake_tam.calls("memory_recall") == []


class TestDegradation:
    def test_unavailable_tam_never_raises_and_keeps_writes_queued(self, fake_tam, provider):
        fake_tam.configure(command="definitely-not-a-tam-binary")
        provider.initialize("s1")
        provider.sync_turn(BILLING_Q, BILLING_A)
        assert provider.prefetch("billing database") == ""
        assert provider.recall_status() is None
        result = json.loads(provider.handle_tool_call(TOOL_RECALL, {"query": "billing"}))
        assert "TAM recall failed" in result["error"]
        assert not provider.flush(0.5)
        assert provider.pending_writes == 1

    def test_shutdown_with_unreachable_tam_is_bounded_and_counts_abandoned(self, fake_tam, hermes_home):
        fake_tam.configure(command="definitely-not-a-tam-binary", shutdown_timeout=0.5)
        provider = TamMemoryProvider()
        provider.initialize("s1")
        provider.sync_turn(BILLING_Q, BILLING_A)
        provider.shutdown()
        assert provider.outcomes.get("abandoned") == 1
        provider.sync_turn(BILLING_Q, BILLING_A)
        assert provider.outcomes.get("rejected_after_shutdown") == 1

    def test_reconnects_after_tam_crash(self, fake_tam, provider):
        fake_tam.configure(
            env={"FAKE_TAM_STORE": str(fake_tam.store), "FAKE_TAM_LOG": str(fake_tam.log), "FAKE_TAM_EXIT_AFTER": "1"}
        )
        _started(provider)
        provider._connection._backoff_initial = 0.05
        provider.sync_turn(BILLING_Q, BILLING_A)
        provider.sync_turn(
            "Where do the analytics replicas live?",
            "In the eu-west analytics cluster, three replicas behind pgbouncer.",
        )
        assert provider.flush(10)
        assert len(fake_tam.records()) == 2

    def test_slow_recall_times_out_to_empty(self, fake_tam, provider):
        fake_tam.configure(
            env={"FAKE_TAM_STORE": str(fake_tam.store), "FAKE_TAM_RECALL_DELAY": "3"}, recall_timeout=0.5
        )
        _started(provider)
        assert provider.prefetch("billing database") == ""


class TestTools:
    def test_save_and_recall_tools(self, fake_tam, provider):
        _started(provider)
        saved = json.loads(
            provider.handle_tool_call(
                TOOL_SAVE, {"content": "Deploys freeze on Fridays.", "type": "convention", "importance": "high"}
            )
        )
        assert saved["saved"] is True
        found = json.loads(provider.handle_tool_call(TOOL_RECALL, {"query": "deploys fridays", "limit": 3}))
        assert found["count"] == 1 and found["results"][0]["type"] == "convention"

    def test_tool_argument_validation(self, fake_tam, provider):
        _started(provider)
        assert "error" in json.loads(provider.handle_tool_call(TOOL_SAVE, {"content": ""}))
        assert "error" in json.loads(provider.handle_tool_call(TOOL_SAVE, {"content": "x", "type": "gossip"}))
        assert "error" in json.loads(provider.handle_tool_call(TOOL_RECALL, {"query": "x", "limit": "many"}))
        assert "error" in json.loads(provider.handle_tool_call("tam_other", {}))

    def test_schemas_do_not_shadow_core_tools(self, provider):
        from toolsets import _HERMES_CORE_TOOLS

        names = {schema["name"] for schema in provider.get_tool_schemas()}
        assert names == {TOOL_RECALL, TOOL_SAVE} and not names & set(_HERMES_CORE_TOOLS)


class TestTeamBackend:
    def test_team_shapes_and_request_id(self, fake_tam, provider):
        fake_tam.configure(
            env={"FAKE_TAM_STORE": str(fake_tam.store), "FAKE_TAM_LOG": str(fake_tam.log), "FAKE_TAM_BACKEND": "team"}
        )
        _started(provider, "s1")
        provider.sync_turn(BILLING_Q, BILLING_A)
        assert provider.flush(5)
        assert fake_tam.calls("memory_save")[0]["arguments"]["request_id"]
        block = provider.prefetch("billing database postgresql", session_id="other")
        assert "by Alice" in block
        assert "detail" not in fake_tam.calls("memory_recall")[0]["arguments"]


class TestThroughMemoryManager:
    def test_full_lifecycle_via_hermes_manager(self, fake_tam, hermes_home):
        manager = MemoryManager()
        manager.add_provider(TamMemoryProvider())
        manager.initialize_all(session_id="m1", platform="cli", hermes_home=str(hermes_home))
        provider = manager.get_provider("tam")
        wait_until(lambda: provider._connection.connected)
        assert "TAM long-term memory" in manager.build_system_prompt()
        manager.sync_all(
            BILLING_Q, BILLING_A, session_id="m1", messages=[{"role": "tool", "content": "secret tool output"}]
        )
        assert manager.flush_pending(5)
        assert provider.flush(5)
        manager.on_session_end([])
        manager.shutdown_all()
        assert "secret tool output" not in json.dumps(fake_tam.calls())

        second = MemoryManager()
        second.add_provider(TamMemoryProvider())
        second.initialize_all(session_id="m2", platform="cli", hermes_home=str(hermes_home))
        wait_until(lambda: second.get_provider("tam")._connection.connected)
        context = second.prefetch_all("remind me which database billing uses")
        assert "PostgreSQL 18" in context
        assert "TAM" in second.describe_recall()
        second.shutdown_all()


class TestReinitialize:
    def test_second_initialize_replaces_connection_and_writer(self, fake_tam, provider):
        _started(provider, "s1")
        first_connection, first_writer = provider._connection, provider._writer
        _started(provider, "s2")
        assert provider._connection is not first_connection and not first_connection.connected
        wait_until(lambda: not first_writer.is_alive())
        provider.sync_turn(BILLING_Q, BILLING_A)
        assert provider.flush(5)
        assert fake_tam.calls("memory_save")[0]["arguments"]["context"] == session_context("s2")
