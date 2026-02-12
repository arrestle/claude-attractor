"""System prompt layering -- compose prompts from multiple sources.

Builds the final system prompt from up to 4 layers with clear precedence:

1. **Profile layer** (lowest priority): Provider-specific base prompt
   (e.g., Claude Code style, codex-rs style)
2. **Pipeline layer**: Goal and context from the DOT graph
3. **Node layer**: Per-node instructions from prompt/system_prompt attrs
4. **User layer** (highest priority): Explicit user override

Each layer can append to or replace the previous layers.

Usage::

    from attractor_agent.prompt_layer import build_system_prompt, PromptLayer

    prompt = build_system_prompt(
        profile_prompt="You are an expert software engineer...",
        pipeline_goal="Build a REST API",
        node_instruction="Focus on error handling",
        user_override=None,  # None = don't override
    )

Spec reference: coding-agent-loop-spec S6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PromptLayer:
    """A single layer in the system prompt stack."""

    source: str  # "profile", "pipeline", "node", "user"
    content: str
    mode: str = "append"  # "append" or "replace"


def build_system_prompt(
    *,
    profile_prompt: str = "",
    pipeline_goal: str = "",
    pipeline_context: dict[str, Any] | None = None,
    node_instruction: str = "",
    user_override: str | None = None,
) -> str:
    """Build the final system prompt from layered sources.

    Precedence (highest wins):
    - user_override: If set, replaces everything
    - node_instruction: Appended after profile + pipeline
    - pipeline_goal: Appended as context section
    - profile_prompt: Base layer

    Args:
        profile_prompt: Provider profile's system prompt.
        pipeline_goal: The DOT graph's goal attribute.
        pipeline_context: Additional context variables from the pipeline.
        node_instruction: Per-node system_prompt or instruction attribute.
        user_override: If set, replaces the entire prompt.

    Returns:
        The composed system prompt string.
    """
    # User override replaces everything
    if user_override is not None and user_override.strip():
        return user_override.strip()

    parts: list[str] = []

    # Layer 1: Profile base prompt
    if profile_prompt:
        parts.append(profile_prompt.strip())

    # Layer 2: Pipeline goal and context
    if pipeline_goal:
        goal_section = f"\n\n[GOAL] {pipeline_goal}"
        parts.append(goal_section)

    if pipeline_context:
        ctx_items = [
            f"  {k}: {v}"
            for k, v in pipeline_context.items()
            if isinstance(v, (str, int, float, bool))
            and not k.startswith("_")
            and not k.startswith("parallel.")
            and not k.startswith("manager.")
            and not k.startswith("codergen.")
        ]
        if ctx_items:
            ctx_section = "\n[CONTEXT]\n" + "\n".join(ctx_items)
            parts.append(ctx_section)

    # Layer 3: Node-specific instruction
    if node_instruction:
        node_section = f"\n\n[INSTRUCTION] {node_instruction.strip()}"
        parts.append(node_section)

    return "\n".join(parts).strip()


def layer_prompt_for_node(
    *,
    profile_prompt: str = "",
    goal: str = "",
    context: dict[str, Any] | None = None,
    node_system_prompt: str = "",
    user_system_prompt: str = "",
) -> str:
    """Convenience wrapper for building a node's system prompt.

    Called by the pipeline backend when preparing a session for
    a codergen node.
    """
    return build_system_prompt(
        profile_prompt=profile_prompt,
        pipeline_goal=goal,
        pipeline_context=context,
        node_instruction=node_system_prompt,
        user_override=user_system_prompt if user_system_prompt else None,
    )
