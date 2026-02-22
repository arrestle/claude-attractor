"""CodergenBackend implementations that bridge the pipeline to LLM providers.

This module contains the concrete backends that connect the Attractor
pipeline engine to the Coding Agent Loop and the Unified LLM Client.

AgentLoopBackend: Wraps a coding agent Session as a CodergenBackend.
    Pipeline node -> Agent Session -> LLM Client -> Provider API

DirectLLMBackend: Calls the LLM Client directly (no agent loop).
    Pipeline node -> LLM Client -> Provider API
    Simpler, no tools, good for simple prompt-response nodes.

ClaudeCodeBackend: Shells out to the claude-code CLI.
    Pipeline node -> claude-code --print -> Vertex AI / Anthropic API
    Uses whatever auth claude-code is configured with (Vertex AI, API key, etc.).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

import anyio

from attractor_agent.abort import AbortSignal
from attractor_agent.profiles import get_profile
from attractor_agent.prompt_layer import layer_prompt_for_node
from attractor_agent.session import Session, SessionConfig
from attractor_agent.tools.core import ALL_CORE_TOOLS
from attractor_llm.client import Client
from attractor_llm.types import Message, Request
from attractor_pipeline.engine.runner import HandlerResult, Outcome
from attractor_pipeline.graph import Node


class AgentLoopBackend:
    """Bridges the Coding Agent Loop to the pipeline's CodergenBackend interface.

    Wraps a Client in an agent Session per call, giving the LLM access to
    developer tools (read_file, write_file, edit_file, shell, grep, glob).

    Usage::

        client = Client()
        client.register_adapter("anthropic", AnthropicAdapter(config))
        backend = AgentLoopBackend(client)

        # Register with pipeline handlers:
        registry.register("codergen", CodergenHandler(backend=backend))
    """

    def __init__(
        self,
        client: Client,
        *,
        default_model: str = "claude-sonnet-4-5",
        default_provider: str | None = None,
        system_prompt: str = "",
        include_tools: bool = True,
    ) -> None:
        self._client = client
        self._default_model = default_model
        self._default_provider = default_provider
        self._system_prompt = system_prompt
        self._include_tools = include_tools

    async def run(
        self,
        node: Node,
        prompt: str,
        context: dict[str, Any],
        abort_signal: AbortSignal | None = None,
    ) -> str | HandlerResult:
        """Execute an LLM call through the agent loop.

        Creates a fresh Session for each node execution, configured
        with the node's LLM settings (model, provider, reasoning_effort).
        """
        # Resolve provider first (needed for profile lookup)
        provider = node.llm_provider or self._default_provider

        # Load provider profile for provider-specific defaults
        profile = get_profile(provider or "")

        # Resolve model: node attr > backend default > profile default
        model = node.llm_model or self._default_model or profile.default_model

        # Build session config from node attributes
        config = SessionConfig(
            model=model,
            provider=provider,
            system_prompt=layer_prompt_for_node(
                profile_prompt=profile.system_prompt,
                goal=context.get("goal", ""),
                context=context,
                node_system_prompt=node.attrs.get("system_prompt", ""),
                user_system_prompt=self._system_prompt,
            ),
            max_turns=10,  # Allow tool-call rounds within a single pipeline node
            max_tool_rounds_per_turn=15,
            reasoning_effort=node.reasoning_effort or None,
        )

        # Apply profile defaults (only fills in unset values)
        config = profile.apply_to_config(config)

        # Create tools list with profile-customized descriptions
        if self._include_tools:
            tools = profile.get_tools(list(ALL_CORE_TOOLS))
        else:
            tools: list[Any] = []

        # Run session with error handling
        try:
            async with Session(
                client=self._client,
                config=config,
                tools=tools,
                abort_signal=abort_signal,
            ) as session:
                result = await session.submit(prompt)
        except Exception as exc:  # noqa: BLE001
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        # Check for session-level error responses
        if result.startswith("[Error:") or result.startswith("[Session aborted]"):
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=result,
                output=result,
            )

        return result


class DirectLLMBackend:
    """Calls the LLM Client directly without the agent loop.

    Simpler than AgentLoopBackend -- no tools, no multi-turn, just a
    single prompt-response call. Good for simple LLM tasks like
    summarization, classification, or text generation.
    """

    def __init__(
        self,
        client: Client,
        *,
        default_model: str = "claude-sonnet-4-5",
        default_provider: str | None = None,
    ) -> None:
        self._client = client
        self._default_model = default_model
        self._default_provider = default_provider

    async def run(
        self,
        node: Node,
        prompt: str,
        context: dict[str, Any],
        abort_signal: AbortSignal | None = None,
    ) -> str | HandlerResult:
        """Execute a single LLM call (no tools, no agent loop)."""
        model = node.llm_model or self._default_model
        provider = node.llm_provider or self._default_provider

        request = Request(
            model=model,
            provider=provider,
            messages=[Message.user(prompt)],
            system=f"You are working on: {context.get('goal', '')}",
            reasoning_effort=node.reasoning_effort or None,
        )

        try:
            response = await self._client.complete(request)
        except Exception as exc:  # noqa: BLE001
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        text = response.text or ""
        if not text:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason="Empty response from LLM",
            )

        return text


class ClaudeCodeBackend:
    """Shells out to the claude-code CLI as a CodergenBackend.

    Spawns ``claude-code --print <prompt>`` for each codergen node.
    Uses whatever authentication claude-code is configured with:
    - Google Vertex AI (via ~/.claude/settings.json + gcloud auth)
    - Direct Anthropic API (via ANTHROPIC_API_KEY)

    This avoids re-implementing provider auth and lets users leverage
    claude-code's built-in file tools, agentic loop, and context window.

    Usage::

        backend = ClaudeCodeBackend(working_dir="/path/to/sos-reports")
        registry = HandlerRegistry()
        register_default_handlers(registry, codergen_backend=backend)
    """

    def __init__(
        self,
        *,
        claude_bin: str | None = None,
        working_dir: str | None = None,
        max_turns: int = 10,
        model: str | None = None,
    ) -> None:
        self._claude_bin = claude_bin or self._find_claude_binary()
        self._working_dir = working_dir
        self._max_turns = max_turns
        self._model = model

    @staticmethod
    def _find_claude_binary() -> str:
        for name in ("claude-code", "claude"):
            path = shutil.which(name)
            if path:
                return path
        raise FileNotFoundError(
            "claude-code CLI not found. Install with: "
            "npm install -g @anthropic-ai/claude-code"
        )

    async def run(
        self,
        node: Node,
        prompt: str,
        context: dict[str, Any],
        abort_signal: AbortSignal | None = None,
    ) -> str | HandlerResult:
        """Execute an LLM call by spawning claude-code --print."""
        enriched = self._build_prompt(node, prompt, context)
        timeout = self._parse_timeout(node.timeout) or 900.0

        cmd = [self._claude_bin, "--print"]
        if self._max_turns:
            cmd.extend(["--max-turns", str(self._max_turns)])
        model = node.llm_model or self._model
        if model:
            cmd.extend(["--model", model])
        cmd.append(enriched)

        def _run_subprocess() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self._working_dir,
            )

        try:
            result = await anyio.to_thread.run_sync(_run_subprocess)
        except subprocess.TimeoutExpired:
            return HandlerResult(
                status=Outcome.RETRY,
                failure_reason=f"claude-code timed out after {timeout}s",
            )

        if abort_signal and abort_signal.is_set:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason="Aborted",
            )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "Credit balance is too low" in stderr:
                return HandlerResult(
                    status=Outcome.FAIL,
                    failure_reason="LLM credit balance too low",
                )
            if "Could not load the default credentials" in stderr:
                return HandlerResult(
                    status=Outcome.FAIL,
                    failure_reason=(
                        "Google Cloud credentials not configured. "
                        "Run: gcloud auth application-default login"
                    ),
                )
            return HandlerResult(
                status=Outcome.RETRY,
                failure_reason=f"claude-code exited {result.returncode}: {stderr[:500]}",
            )

        output = result.stdout.strip()
        if not output:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason="Empty response from claude-code",
            )

        return output

    def _build_prompt(self, node: Node, prompt: str, context: dict[str, Any]) -> str:
        parts: list[str] = []

        goal = context.get("goal", "")
        if goal:
            parts.append(f"## Pipeline Goal\n{goal}")

        relevant = {
            k: v for k, v in context.items()
            if not k.startswith(("_", "human."))
            and k not in ("goal", "outcome", "preferred_label")
        }
        if relevant:
            parts.append(
                "## Context from Previous Stages\n"
                + json.dumps(relevant, indent=2, default=str)
            )

        parts.append(f"## Task: {node.label or node.id}\n{prompt}")

        parts.append(
            "## Instructions\n"
            "Respond with your full analysis or output. "
            "Do not ask clarifying questions — work with what you have."
        )

        return "\n\n".join(parts)

    @staticmethod
    def _parse_timeout(timeout_str: str) -> float | None:
        if not timeout_str:
            return None
        s = timeout_str.strip()
        if s.endswith("ms"):
            return float(s[:-2]) / 1000
        if s.endswith("s"):
            return float(s[:-1])
        if s.endswith("m"):
            return float(s[:-1]) * 60
        if s.endswith("h"):
            return float(s[:-1]) * 3600
        try:
            return float(s)
        except ValueError:
            return None
