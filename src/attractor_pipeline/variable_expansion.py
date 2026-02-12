"""Variable expansion for DOT pipeline prompts.

Expands $variable references in node prompts using values from
the pipeline context. This is an AST-level transform that runs
before each node is executed.

Supports:
- $goal -- the pipeline's goal attribute
- $variable -- any string value from the context dict
- ${variable} -- braced form for disambiguation
- Escaped \\$literal -- produces literal $ (no expansion)

Security: values are NOT sanitized here (this is for prompts sent
to the LLM, not DOT structure). DOT-structural expansion is handled
separately in manager.py with sanitization.

Spec reference: attractor-spec S9.1-9.3.
"""

from __future__ import annotations

import re
from typing import Any

# Matches $name or ${name}, but not \$name (escaped)
_VAR_PATTERN = re.compile(
    r"(?<!\\)"  # negative lookbehind: not preceded by backslash
    r"\$"  # literal dollar sign
    r"(?:"
    r"\{([^}]+)\}"  # ${braced_name} -> group 1
    r"|"
    r"([a-zA-Z_][a-zA-Z0-9_.]*)"  # $bare_name -> group 2
    r")"
)


def expand_variables(
    template: str,
    context: dict[str, Any],
    *,
    undefined: str = "keep",
) -> str:
    """Expand $variable references in a template string.

    Args:
        template: String containing $variable references.
        context: Dict of variable values. Only str/int/float/bool
            values are expanded; others are skipped.
        undefined: What to do with undefined variables:
            "keep" -- leave $var as-is (default)
            "empty" -- replace with empty string
            "error" -- raise KeyError

    Returns:
        The expanded string.

    Raises:
        KeyError: If undefined="error" and a variable is not found.
    """

    def replacer(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name in context:
            value = context[name]
            if isinstance(value, (str, int, float, bool)):
                return str(value)
            # Non-scalar values: keep the reference
            return match.group(0)
        # Undefined variable
        if undefined == "empty":
            return ""
        if undefined == "error":
            raise KeyError(f"Undefined variable: ${name}")
        return match.group(0)  # keep as-is

    result = _VAR_PATTERN.sub(replacer, template)
    # Unescape \$ -> $
    result = result.replace("\\$", "$")
    return result


def expand_node_prompt(
    prompt: str,
    context: dict[str, Any],
) -> str:
    """Expand variables in a node's prompt attribute.

    This is the main entry point used by the pipeline engine
    before passing the prompt to the codergen handler.

    Undefined variables are kept as-is (safe default -- no data loss).
    """
    return expand_variables(prompt, context, undefined="keep")
