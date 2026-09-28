from __future__ import annotations

import json
import os
import stat

import pytest

from hermes_tam_memory.backend import (
    Backend,
    ContractError,
    Memory,
    SaveRequest,
    detect_backend,
    parse_recall,
    parse_save,
    recall_arguments,
    save_arguments,
)
from hermes_tam_memory.config import TamConfig, load_config, parse_config, save_config
from hermes_tam_memory.text import (
    RECALL_HEADER,
    clean_text,
    format_recall,
    format_turn,
    judge_turn,
    session_context,
)


def _memory(content: str, score: float = 1.0, context: str = "", **extra: str) -> Memory:
    return Memory(
        record_id=1,
        content=content,
        type=extra.get("type", "fact"),
        project=extra.get("project", "p"),
        created_at="2026-09-20T10:00:00Z",
        score=score,
        context=context,
        tags=(),
        author=extra.get("author", ""),
    )


class TestConfig:
    def test_defaults_when_file_missing(self, hermes_home):
        assert load_config(hermes_home) == TamConfig()

    def test_invalid_values_fall_back_and_numbers_clamp(self):
        config = parse_config(
            {
                "mode": "cloud",
                "recall_limit": 999,
                "recall_timeout": 60,
                "auto_recall": "no",
                "args": "not-a-list",
                "env": {"A": 1},
                "surprise": True,
            }
        )
        assert config.mode == "local"
        assert config.recall_limit == 20
        assert config.recall_timeout == 7.5
        assert config.auto_recall is False
        assert config.args == ()
        assert config.env == {}

    def test_save_merges_drops_token_and_is_private(self, hermes_home):
        (hermes_home / "tam.json").write_text(json.dumps({"recall_limit": 3}))
        config = save_config({"mode": "remote", "url": " http://x/mcp/ ", "token": "secret"}, hermes_home)
        stored = json.loads((hermes_home / "tam.json").read_text())
        assert config.url == "http://x/mcp/" and config.recall_limit == 3
        assert "token" not in stored and stored["mode"] == "remote"
        if os.name == "posix":
            assert stat.S_IMODE((hermes_home / "tam.json").stat().st_mode) == 0o600


class TestText:
    def test_clean_text_strips_injected_context_and_data_uris(self):
        raw = "<memory-context>old recall</memory-context>look at data:image/png;base64,AAAA==\n\n\n\nnow"
        assert clean_text(raw, 1000) == "look at [inline data]\n\nnow"

    def test_clean_text_truncates(self):
        assert clean_text("x" * 50, 20).endswith("[…]")
        assert len(clean_text("x" * 50, 20)) == 20

    @pytest.mark.parametrize(
        ("user", "assistant", "reason"),
        [
            ("thanks!", "You're welcome, anything else I can help with today?", "trivial_prompt"),
            ("/model", "switched model to something with a long description here", "trivial_prompt"),
            ("what is 2+2", "4", "too_short"),
            ("explain the deploy pipeline in detail please", "", "empty_assistant"),
        ],
    )
    def test_noise_is_skipped(self, user, assistant, reason):
        from agent.memory_provider import is_trivial_prompt

        verdict = judge_turn(user, assistant, min_chars=80, is_trivial=is_trivial_prompt)
        assert (verdict.keep, verdict.reason) == (False, reason)

    def test_substantive_turn_is_kept(self):
        from agent.memory_provider import is_trivial_prompt

        verdict = judge_turn(
            "Which database does billing use?",
            "Billing runs on PostgreSQL 18 with logical replication to the analytics cluster.",
            min_chars=80,
            is_trivial=is_trivial_prompt,
        )
        assert verdict.keep

    def test_format_recall_respects_budget_order_and_session_exclusion(self):
        memories = [
            _memory("current session turn", 9.0, context=session_context("s1")),
            _memory("PostgreSQL 18 for billing", 5.0, project="billing"),
            _memory("PostgreSQL 18 for billing", 4.0),
            _memory("y" * 3000, 3.0),
            _memory("short one fits", 1.0),
        ]
        block = format_recall(memories, budget_tokens=150, max_items=5, exclude_session="s1")
        assert block.text.startswith(RECALL_HEADER)
        assert "current session turn" not in block.text
        assert block.text.count("PostgreSQL 18 for billing") == 1
        assert "[fact · billing · 2026-09-20] PostgreSQL 18 for billing" in block.text
        assert "yyyy" not in block.text and "short one fits" in block.text
        assert block.count == 2
        assert len(block.text) <= 150 * 4

    def test_format_recall_caps_items_and_handles_empty(self):
        memories = [_memory(f"fact number {i}", 10 - i) for i in range(6)]
        assert format_recall(memories, budget_tokens=2000, max_items=3, exclude_session="").count == 3
        assert format_recall([], budget_tokens=2000, max_items=3).text == ""

    def test_format_turn(self):
        assert format_turn("q", "a") == "User: q\nAssistant: a"


class TestBackend:
    def test_detect_backend(self):
        assert detect_backend(frozenset({"memory_save", "memory_recall"})) is Backend.LOCAL
        assert detect_backend(frozenset({"memory_save", "memory_recall", "memory_scopes"})) is Backend.TEAM
        with pytest.raises(ContractError):
            detect_backend(frozenset({"memory_save"}))

    def test_arguments_match_each_server_contract(self):
        request = SaveRequest(content="c", project="p", tags=("t",), request_id="rid")
        assert "request_id" not in save_arguments(Backend.LOCAL, request)
        assert save_arguments(Backend.TEAM, request)["request_id"] == "rid"
        assert recall_arguments(Backend.LOCAL, "q", 3, None) == {"query": "q", "limit": 3, "detail": "full"}
        assert recall_arguments(Backend.TEAM, "q", 3, "p") == {"query": "q", "limit": 3, "project": "p"}

    def test_parse_save_variants(self):
        assert parse_save(Backend.LOCAL, {"saved": True, "id": 7, "deduplicated": True}).deduplicated
        rejected = parse_save(Backend.LOCAL, {"saved": False, "rejected_by_quality_gate": True, "reason": "vague"})
        assert rejected.rejected_by_quality_gate and rejected.reason == "vague"
        team = parse_save(Backend.TEAM, {"data": {"saved": True, "id": 3, "deduplicated": False}})
        assert team.saved and team.record_id == 3
        assert parse_save(Backend.TEAM, {"data": {"saved": False, "quality": {"reason": "x"}}}).rejected_by_quality_gate
        with pytest.raises(ContractError):
            parse_save(Backend.LOCAL, "nope")

    def test_parse_recall_local_and_team(self):
        local = parse_recall(
            Backend.LOCAL,
            {
                "results": {
                    "decision": [{"id": 1, "content": "use pg", "score": 0.5}],
                    "fact": [{"id": 2, "content": "billing", "score": 0.9, "tags": '["a"]'}],
                }
            },
        )
        assert [m.record_id for m in local] == [2, 1]
        assert local[1].type == "decision" and local[0].tags == ["a"]
        team = parse_recall(
            Backend.TEAM,
            {"results": [{"record": {"id": 5, "content": "x", "score": 1, "created_by": {"display_name": "Alice"}}}]},
        )
        assert team[0].author == "Alice"
        with pytest.raises(ContractError):
            parse_recall(Backend.TEAM, {"results": {}})
