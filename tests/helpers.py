"""Shared test fixtures: MockAdapter, mock LLM responses, test helpers.

The MockAdapter is a programmable LLM adapter that returns scripted
responses. It allows testing the full agentic loop (LLM -> tool calls
-> tool results -> text response) without real API calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from attractor_llm.types import (
    ContentPart,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventKind,
    Usage,
)


class MockAdapter:
    """Programmable mock adapter for testing.

    Takes a list of Response objects (or callables). Each call to
    complete() pops the next response. Raises if the script runs out.

    Usage::

        adapter = MockAdapter(responses=[
            # Turn 1: model wants to call a tool
            Response(
                message=Message(role=Role.ASSISTANT, content=[
                    ContentPart.tool_call_part("tc-1", "shell", '{"command": "echo hi"}'),
                ]),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
            # Turn 2: model produces text after seeing tool result
            Response(
                message=Message.assistant("Done! Output was: hi"),
                finish_reason=FinishReason.STOP,
            ),
        ])
        client = Client()
        client.register_adapter("mock", adapter)
    """

    def __init__(
        self,
        responses: Sequence[Response | Exception] | None = None,
    ) -> None:
        self._responses: list[Response | Exception] = list(responses or [])
        self._call_count = 0
        self._requests: list[Request] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def requests(self) -> list[Request]:
        """All requests received (for inspection in tests)."""
        return self._requests

    def add_response(self, response: Response | Exception) -> None:
        """Add a response to the script."""
        self._responses.append(response)

    async def complete(self, request: Request) -> Response:
        self._call_count += 1
        self._requests.append(request)

        if not self._responses:
            raise RuntimeError(
                f"MockAdapter exhausted: {self._call_count} calls but no more scripted responses"
            )

        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        resp = await self.complete(request)
        yield StreamEvent(
            kind=StreamEventKind.START,
            model=request.model,
            provider="mock",
        )
        if resp.text:
            yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=resp.text)
        yield StreamEvent(
            kind=StreamEventKind.FINISH,
            finish_reason=resp.finish_reason,
        )

    async def close(self) -> None:
        pass


# ------------------------------------------------------------------ #
# Response builders (convenience for tests)
# ------------------------------------------------------------------ #


def make_text_response(text: str, model: str = "mock-model") -> Response:
    """Create a simple text response."""
    return Response(
        id="mock-resp",
        model=model,
        provider="mock",
        message=Message.assistant(text),
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def make_tool_call_response(
    tool_name: str,
    arguments: dict[str, Any] | str,
    tool_call_id: str = "tc-1",
    model: str = "mock-model",
) -> Response:
    """Create a response with a single tool call."""
    return Response(
        id="mock-resp",
        model=model,
        provider="mock",
        message=Message(
            role=Role.ASSISTANT,
            content=[
                ContentPart.tool_call_part(tool_call_id, tool_name, arguments),
            ],
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(input_tokens=10, output_tokens=15),
    )


def make_multi_tool_response(
    calls: list[tuple[str, str, dict[str, Any] | str]],
    model: str = "mock-model",
) -> Response:
    """Create a response with multiple tool calls.

    calls: list of (tool_call_id, tool_name, arguments) tuples
    """
    parts = [ContentPart.tool_call_part(tc_id, name, args) for tc_id, name, args in calls]
    return Response(
        id="mock-resp",
        model=model,
        provider="mock",
        message=Message(role=Role.ASSISTANT, content=parts),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(input_tokens=10, output_tokens=20),
    )
