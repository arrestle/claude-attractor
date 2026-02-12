"""Tests for polish features: prompt layering, variable expansion, OpenAI-compat adapter.

Covers:
- Prompt layering: precedence rules, user override, goal injection, context filtering
- Variable expansion: $var, ${var}, escaped dollar signs, undefined handling, non-scalar skip
- OpenAI-compat adapter: request building, response parsing, tool calls, error handling
"""

from __future__ import annotations

import httpx
import pytest
import respx

from attractor_agent.prompt_layer import build_system_prompt, layer_prompt_for_node
from attractor_llm.adapters.base import ProviderConfig
from attractor_llm.adapters.openai_compat import OpenAICompatAdapter
from attractor_llm.errors import AuthenticationError, ServerError
from attractor_llm.types import (
    FinishReason,
    Message,
    Request,
)
from attractor_pipeline.variable_expansion import expand_node_prompt, expand_variables

# ================================================================== #
# Prompt Layering
# ================================================================== #


class TestPromptLayering:
    def test_profile_only(self):
        result = build_system_prompt(profile_prompt="You are an expert.")
        assert result == "You are an expert."

    def test_profile_plus_goal(self):
        result = build_system_prompt(
            profile_prompt="You are an expert.",
            pipeline_goal="Build a REST API",
        )
        assert "You are an expert." in result
        assert "[GOAL] Build a REST API" in result

    def test_profile_plus_goal_plus_node(self):
        result = build_system_prompt(
            profile_prompt="Base prompt.",
            pipeline_goal="Build X",
            node_instruction="Focus on error handling",
        )
        assert "Base prompt." in result
        assert "[GOAL] Build X" in result
        assert "[INSTRUCTION] Focus on error handling" in result

    def test_user_override_replaces_everything(self):
        result = build_system_prompt(
            profile_prompt="This should be gone.",
            pipeline_goal="This too.",
            node_instruction="And this.",
            user_override="My custom prompt only.",
        )
        assert result == "My custom prompt only."
        assert "gone" not in result

    def test_empty_user_override_doesnt_replace(self):
        result = build_system_prompt(
            profile_prompt="Keep me.",
            user_override="",
        )
        assert result == "Keep me."

    def test_none_user_override_doesnt_replace(self):
        result = build_system_prompt(
            profile_prompt="Keep me.",
            user_override=None,
        )
        assert result == "Keep me."

    def test_context_filtering_skips_internal_keys(self):
        result = build_system_prompt(
            profile_prompt="Base.",
            pipeline_context={
                "language": "Python",
                "_internal": "hidden",
                "parallel.fork.results": "hidden",
                "manager.mgr.iterations": "hidden",
                "codergen.plan.output": "hidden",
                "visible_key": "visible_value",
            },
        )
        assert "language: Python" in result
        assert "visible_key: visible_value" in result
        assert "_internal" not in result
        assert "parallel" not in result
        assert "manager" not in result
        assert "codergen" not in result

    def test_context_non_scalar_skipped(self):
        result = build_system_prompt(
            profile_prompt="Base.",
            pipeline_context={
                "name": "test",
                "nested": {"a": 1},
                "items": [1, 2, 3],
            },
        )
        assert "name: test" in result
        assert "nested" not in result
        assert "items" not in result

    def test_layer_prompt_for_node_convenience(self):
        result = layer_prompt_for_node(
            profile_prompt="Profile base.",
            goal="Build widget",
            node_system_prompt="Be concise.",
        )
        assert "Profile base." in result
        assert "[GOAL] Build widget" in result
        assert "[INSTRUCTION] Be concise." in result

    def test_layer_prompt_user_override_via_convenience(self):
        result = layer_prompt_for_node(
            profile_prompt="Ignored.",
            goal="Ignored.",
            user_system_prompt="User wins.",
        )
        assert result == "User wins."


# ================================================================== #
# Variable Expansion
# ================================================================== #


class TestVariableExpansion:
    def test_simple_variable(self):
        assert expand_variables("Hello $name", {"name": "World"}) == "Hello World"

    def test_braced_variable(self):
        assert expand_variables("Hello ${name}", {"name": "World"}) == "Hello World"

    def test_goal_variable(self):
        result = expand_node_prompt("Build $goal now", {"goal": "a widget"})
        assert result == "Build a widget now"

    def test_multiple_variables(self):
        result = expand_variables(
            "$greeting $name, welcome to $place",
            {"greeting": "Hello", "name": "Alice", "place": "Wonderland"},
        )
        assert result == "Hello Alice, welcome to Wonderland"

    def test_undefined_keep(self):
        result = expand_variables("$known and $unknown", {"known": "yes"})
        assert result == "yes and $unknown"

    def test_undefined_empty(self):
        result = expand_variables("$known and $unknown", {"known": "yes"}, undefined="empty")
        assert result == "yes and "

    def test_undefined_error(self):
        with pytest.raises(KeyError, match="unknown"):
            expand_variables("$unknown", {}, undefined="error")

    def test_escaped_dollar(self):
        result = expand_variables("Price is \\$100", {"100": "nope"})
        assert result == "Price is $100"

    def test_non_scalar_skipped(self):
        result = expand_variables(
            "$name and $data",
            {"name": "Alice", "data": {"nested": True}},
        )
        assert result == "Alice and $data"

    def test_integer_value(self):
        result = expand_variables("Port $port", {"port": 8080})
        assert result == "Port 8080"

    def test_boolean_value(self):
        result = expand_variables("Debug: $debug", {"debug": True})
        assert result == "Debug: True"

    def test_dotted_variable_name(self):
        result = expand_variables(
            "Output: $codergen.plan.output",
            {"codergen.plan.output": "the plan"},
        )
        assert result == "Output: the plan"

    def test_empty_template(self):
        assert expand_variables("", {"x": "y"}) == ""

    def test_no_variables(self):
        assert expand_variables("plain text", {"x": "y"}) == "plain text"

    def test_adjacent_variables(self):
        result = expand_variables("$a$b$c", {"a": "1", "b": "2", "c": "3"})
        assert result == "123"


# ================================================================== #
# OpenAI-Compatible Adapter
# ================================================================== #


class TestOpenAICompatAdapter:
    @pytest.mark.asyncio
    @respx.mock
    async def test_simple_completion(self):
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "model": "llama3",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )
        )

        adapter = OpenAICompatAdapter(
            ProviderConfig(
                base_url="http://localhost:11434/v1",
                api_key="test",
            )
        )
        req = Request(model="llama3", messages=[Message.user("Hi")])

        resp = await adapter.complete(req)
        assert resp.text == "Hello!"
        assert resp.provider == "openai-compat"
        assert resp.finish_reason == FinishReason.STOP
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5

    @pytest.mark.asyncio
    @respx.mock
    async def test_tool_call_response(self):
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-456",
                    "model": "llama3",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_abc",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path": "/tmp/test.py"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 15},
                },
            )
        )

        adapter = OpenAICompatAdapter(
            ProviderConfig(
                base_url="http://localhost:11434/v1",
                api_key="test",
            )
        )
        req = Request(model="llama3", messages=[Message.user("Read file")])

        resp = await adapter.complete(req)
        assert resp.finish_reason == FinishReason.TOOL_CALLS
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "read_file"
        assert resp.tool_calls[0].tool_call_id == "call_abc"

    @pytest.mark.asyncio
    @respx.mock
    async def test_401_raises_auth_error(self):
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                401,
                json={"error": {"message": "invalid key"}},
            )
        )

        adapter = OpenAICompatAdapter(
            ProviderConfig(
                base_url="http://localhost:11434/v1",
                api_key="bad",
            )
        )
        req = Request(model="llama3", messages=[Message.user("Hi")])

        with pytest.raises(AuthenticationError):
            await adapter.complete(req)

    @pytest.mark.asyncio
    @respx.mock
    async def test_500_raises_server_error(self):
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                500,
                json={"error": {"message": "internal error"}},
            )
        )

        adapter = OpenAICompatAdapter(
            ProviderConfig(
                base_url="http://localhost:11434/v1",
                api_key="test",
            )
        )
        req = Request(model="llama3", messages=[Message.user("Hi")])

        with pytest.raises(ServerError):
            await adapter.complete(req)

    @pytest.mark.asyncio
    @respx.mock
    async def test_system_prompt_included(self):
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "llama3",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )
        )

        adapter = OpenAICompatAdapter(
            ProviderConfig(
                base_url="http://localhost:11434/v1",
                api_key="test",
            )
        )
        req = Request(
            model="llama3",
            system="You are helpful.",
            messages=[Message.user("Hi")],
        )

        await adapter.complete(req)

        # Verify the request body included the system message
        call = respx.calls[0]
        body = call.request.content
        import json as _json

        request_data = _json.loads(body)
        messages = request_data["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."

    @pytest.mark.asyncio
    @respx.mock
    async def test_tools_included_in_request(self):
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "llama3",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )
        )

        from attractor_llm.types import Tool

        tool = Tool(
            name="my_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        )

        adapter = OpenAICompatAdapter(
            ProviderConfig(
                base_url="http://localhost:11434/v1",
                api_key="test",
            )
        )
        req = Request(
            model="llama3",
            messages=[Message.user("Use the tool")],
            tools=[tool],
        )

        await adapter.complete(req)

        import json as _json

        request_data = _json.loads(respx.calls[0].request.content)
        assert "tools" in request_data
        assert request_data["tools"][0]["function"]["name"] == "my_tool"

    @pytest.mark.asyncio
    @respx.mock
    async def test_temperature_passed(self):
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "llama3",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )
        )

        adapter = OpenAICompatAdapter(
            ProviderConfig(
                base_url="http://localhost:11434/v1",
                api_key="test",
            )
        )
        req = Request(
            model="llama3",
            messages=[Message.user("Hi")],
            temperature=0.7,
        )

        await adapter.complete(req)

        import json as _json

        request_data = _json.loads(respx.calls[0].request.content)
        assert request_data["temperature"] == 0.7

    @pytest.mark.asyncio
    @respx.mock
    async def test_max_tokens_finish_reason(self):
        respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "llama3",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "truncat"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 100},
                },
            )
        )

        adapter = OpenAICompatAdapter(
            ProviderConfig(
                base_url="http://localhost:11434/v1",
                api_key="test",
            )
        )
        req = Request(model="llama3", messages=[Message.user("Hi")])

        resp = await adapter.complete(req)
        assert resp.finish_reason == FinishReason.MAX_TOKENS

    @pytest.mark.asyncio
    @respx.mock
    async def test_custom_base_url(self):
        respx.post("http://my-server:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "x",
                    "model": "custom",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                },
            )
        )

        adapter = OpenAICompatAdapter(
            ProviderConfig(
                base_url="http://my-server:8000/v1",
                api_key="token",
            )
        )
        req = Request(model="custom", messages=[Message.user("Hi")])

        resp = await adapter.complete(req)
        assert resp.text == "ok"
