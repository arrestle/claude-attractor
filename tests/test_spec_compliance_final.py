"""Tests for spec compliance gaps — Wave Final.

Groups:
  TestMaxTurnsDefaults        — Task 1
  TestShellProcessCallback    — Task 2
  TestParallelToolCalls       — Task 3
  TestSessionEndEvent         — Task 4
  TestMiddlewareChain         — Task 5
  TestHttpServer              — Task 6
  TestInterviewerAnswer       — Task 7
  TestR13Validation           — Task 8
  TestAnthropicDescriptions   — Task 9
  TestApplyPatchV4a           — Task 10
"""

from __future__ import annotations

import pytest

from attractor_agent.session import SessionConfig
from attractor_agent.subagent import spawn_subagent


class TestMaxTurnsDefaults:
    """Task 1 — §9 SessionConfig defaults."""

    def test_session_config_max_turns_defaults_to_zero(self):
        """SessionConfig() with no args must have max_turns=0 (unlimited)."""
        config = SessionConfig()
        assert config.max_turns == 0, (
            f"Expected max_turns=0 (unlimited per spec §9), got {config.max_turns}"
        )

    def test_session_config_max_tool_rounds_defaults_to_zero(self):
        """SessionConfig() with no args must have max_tool_rounds_per_turn=0."""
        config = SessionConfig()
        assert config.max_tool_rounds_per_turn == 0, (
            f"Expected max_tool_rounds_per_turn=0, got {config.max_tool_rounds_per_turn}"
        )

    def test_spawn_subagent_max_turns_defaults_to_zero(self):
        """spawn_subagent() max_turns default must be 0 per spec §9."""
        import inspect

        sig = inspect.signature(spawn_subagent)
        assert sig.parameters["max_turns"].default == 0, (
            f"Expected spawn_subagent max_turns default=0, "
            f"got {sig.parameters['max_turns'].default}"
        )

    def test_spawn_subagent_max_tool_rounds_defaults_to_zero(self):
        """spawn_subagent() max_tool_rounds default must be 0."""
        import inspect

        sig = inspect.signature(spawn_subagent)
        assert sig.parameters["max_tool_rounds"].default == 0

    @pytest.mark.asyncio
    async def test_session_zero_max_turns_does_not_limit(self):
        """With max_turns=0, a session must NOT hit the turn limit on turn 1."""
        from unittest.mock import AsyncMock, MagicMock

        from attractor_agent.session import Session, SessionConfig
        from attractor_llm.types import Message, Response, Usage

        config = SessionConfig(max_turns=0, max_tool_rounds_per_turn=0)
        mock_client = MagicMock()
        mock_response = Response(
            id="resp-1",
            model="test-model",
            content=[],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
            provider="test",
        )
        mock_response.message = Message.assistant("Done.")
        mock_client.complete = AsyncMock(return_value=mock_response)

        session = Session(client=mock_client, config=config)
        result = await session.submit("Hello")
        assert "[Turn limit reached]" not in result, (
            "max_turns=0 should mean unlimited, not zero turns allowed"
        )
