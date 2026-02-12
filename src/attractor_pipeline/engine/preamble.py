"""Fidelity resume preamble generator.

When a pipeline is resumed from a checkpoint, the LLM at the resumed
node has no context about what happened in prior nodes. The preamble
generator builds a summary of completed work so the LLM can continue
with full fidelity -- as if it had been running the whole time.

The preamble includes:
- Pipeline goal
- Completed nodes with their outputs (truncated)
- Current position in the graph
- Retry/redirect counts if any
- Reason for checkpoint (interruption, error, etc.)

Usage::

    from attractor_pipeline.engine.preamble import generate_resume_preamble

    preamble = generate_resume_preamble(
        graph=graph,
        checkpoint=checkpoint,
        max_output_chars=500,
    )
    # Inject preamble into the context for the LLM
    context["_resume_preamble"] = preamble

Spec reference: attractor-spec S5.4.
"""

from __future__ import annotations

from typing import Any

from attractor_pipeline.engine.runner import Checkpoint
from attractor_pipeline.graph import Graph


def generate_resume_preamble(
    graph: Graph,
    checkpoint: Checkpoint,
    *,
    max_output_chars: int = 500,
    max_total_chars: int = 4000,
    include_context: bool = True,
) -> str:
    """Generate a fidelity preamble for resuming a checkpointed pipeline.

    Args:
        graph: The pipeline graph being resumed.
        checkpoint: The saved checkpoint state.
        max_output_chars: Max chars per node output in the summary.
            Set to 0 to exclude outputs entirely.
        include_context: Whether to include relevant context variables.

    Returns:
        A formatted preamble string suitable for injection into
        the LLM's system prompt or as a user message prefix.
    """
    parts: list[str] = []

    # Header
    parts.append("[RESUME] This pipeline is being resumed from a checkpoint.")
    parts.append(f"Pipeline: {graph.name}")
    if graph.goal:
        parts.append(f"Goal: {graph.goal}")

    # Completed work summary
    completed = checkpoint.completed_nodes
    if completed:
        parts.append("")
        parts.append(f"Completed nodes ({len(completed)}):")
        for node_info in completed:
            node_id = (
                node_info.get("node_id", "?") if isinstance(node_info, dict) else str(node_info)
            )
            node = graph.get_node(node_id)
            node_label = node.label or node_id if node else node_id
            node_shape = node.shape if node else "?"

            line = f"  - {node_label} ({node_shape})"

            # Include truncated output if available
            if max_output_chars > 0:
                output_key = f"codergen.{node_id}.output"
                output = checkpoint.context_values.get(output_key, "")
                if output and isinstance(output, str):
                    truncated = output[:max_output_chars]
                    if len(output) > max_output_chars:
                        truncated += f"... [{len(output)} chars total]"
                    line += f"\n    Output: {truncated}"

            parts.append(line)

    # Current position
    parts.append("")
    current = checkpoint.current_node_id
    current_node = graph.get_node(current) if current else None
    if current_node:
        parts.append(f"Resuming at: {current_node.label or current_node.id} ({current_node.shape})")
        if current_node.prompt:
            parts.append(f"Node prompt: {current_node.prompt[:200]}")
    else:
        parts.append(f"Resuming at node: {current}")

    # Retry/redirect state
    if checkpoint.node_retry_counts:
        retries = ", ".join(f"{k}: {v}" for k, v in checkpoint.node_retry_counts.items())
        parts.append(f"Retry counts: {retries}")

    if checkpoint.goal_gate_redirect_count > 0:
        parts.append(f"Goal gate redirects: {checkpoint.goal_gate_redirect_count}")

    # Relevant context variables (filtered)
    if include_context:
        relevant = _extract_relevant_context(checkpoint.context_values)
        if relevant:
            parts.append("")
            parts.append("Context from prior stages:")
            for key, value in relevant.items():
                val_str = str(value)
                if len(val_str) > max_output_chars:
                    val_str = val_str[:max_output_chars] + "..."
                parts.append(f"  {key}: {val_str}")

    parts.append("")
    parts.append("Continue from the current node. Use the context above to maintain continuity.")

    result = "\n".join(parts)

    # Enforce total size cap to prevent blowing LLM context windows
    if max_total_chars > 0 and len(result) > max_total_chars:
        truncated = result[:max_total_chars]
        # Try to cut at a newline boundary for cleanliness
        last_nl = truncated.rfind("\n")
        if last_nl > max_total_chars // 2:
            truncated = truncated[:last_nl]
        result = truncated + f"\n... [preamble truncated at {max_total_chars} chars]"

    return result


def _extract_relevant_context(
    context: dict[str, Any],
) -> dict[str, Any]:
    """Extract context variables relevant for the preamble.

    Filters out internal keys and large binary data.
    """
    relevant: dict[str, Any] = {}
    skip_prefixes = ("_", "parallel.", "manager.", "codergen.")

    for key, value in context.items():
        # Skip internal keys
        if any(key.startswith(p) for p in skip_prefixes):
            continue
        # Skip non-scalar, non-string values
        if not isinstance(value, (str, int, float, bool)):
            continue
        # Skip very long values (they'll be in the node outputs)
        if isinstance(value, str) and len(value) > 2000:
            continue
        relevant[key] = value

    return relevant
