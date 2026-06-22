"""Tool Implementor agent: generates Python function bodies for atomic tool nodes."""
from __future__ import annotations

import ast
import json
import logging
from typing import Any

from src.agents.base_agent import BaseAgent, _find_text_block
from src.orchestrator.state import ExecType, SkillNode

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Tool Implementor Agent in a recursive skill-tree generation system.

You receive a list of ATOMIC tool nodes, each with a description, exec_type, input_schema, and output_schema.
Your job is to generate the Python function body for each node.

Rules:
- Return ONLY the function body — the lines that go INSIDE the function, indented with 4 spaces.
  Do NOT include the `def` line, decorators, or any outer scope code.
- Every single line of the function body — INCLUDING import statements — must begin with
  exactly 4 spaces of indentation. There must be no lines at column 0.
- The function receives its arguments as plain Python values matching the input_schema properties.
- The function must return a dict matching the output_schema properties.
- For DETERMINISTIC_CODE:
  - Implement the full logic using only Python stdlib (no third-party imports).
  - The body must be correct and self-contained. Use helper functions defined inline if needed,
    but define them as nested functions inside the body (they will be indented under the body).
  - All stdlib imports must be inside the function body, each indented with 4 spaces
    (e.g., `    import re`, `    import json`, `    import datetime`).
- For EXTERNAL_API:
  - Use `httpx` (already available) for HTTP calls.
  - Mark all configurable values (base URL, API key, endpoint path) with a
    `# CONFIGURE: <what to set>` comment on the same line.
  - Import httpx inside the function body, indented with 4 spaces: `    import httpx`.
  - Include basic error handling: raise a RuntimeError with a descriptive message on non-2xx status.
- Every branch must end with a return statement that produces a dict matching the output_schema.
- Do not add any explanation or markdown — return raw Python code only.

You MUST respond with a JSON array parallel to the input array — one entry per input node:
[
  {
    "implementation": "<function body as a single string, lines separated by \\n, each line indented 4 spaces>"
  },
  ...
]
"""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "implementation": {"type": "string"},
        },
        "required": ["implementation"],
        "additionalProperties": False,
    },
}

# One node per call: Sonnet with thinking burns tokens fast; batching causes truncation
_BATCH_SIZE = 1
_MAX_SYNTAX_RETRIES = 2


class ToolImplementorAgent(BaseAgent):
    MODEL = "claude-sonnet-4-6"
    MAX_TOKENS = 16000  # Sonnet + thinking needs headroom; base default of 8192 truncates

    def __init__(self) -> None:
        super().__init__(system_prompt=_SYSTEM_PROMPT)

    async def implement(self, nodes: list[SkillNode]) -> list[SkillNode]:
        """Generate and assign function bodies for tool atomic nodes (in-place)."""
        if not nodes:
            return nodes

        # Process in batches to avoid overly long prompts
        for batch_start in range(0, len(nodes), _BATCH_SIZE):
            batch = nodes[batch_start : batch_start + _BATCH_SIZE]
            await self._implement_batch(batch)

        self.log_usage()
        return nodes

    async def _implement_batch(self, nodes: list[SkillNode]) -> None:
        node_dicts = [
            {
                "name": n.name,
                "description": n.description,
                "exec_type": n.exec_type.value if n.exec_type else None,
                "input_schema": n.input_schema,
                "output_schema": n.output_schema,
            }
            for n in nodes
        ]

        user_content = (
            "Generate implementations for these atomic tool nodes:\n"
            + json.dumps(node_dicts, indent=2)
        )

        message = await self._call(
            messages=[{"role": "user", "content": user_content}],
            output_schema=_OUTPUT_SCHEMA,
        )

        text_block = _find_text_block(message)
        try:
            results: list[dict[str, Any]] = json.loads(text_block.text)
        except json.JSONDecodeError as exc:
            stop_reason = getattr(message, "stop_reason", "unknown")
            raise ValueError(
                f"ToolImplementor response truncated or malformed "
                f"(stop_reason={stop_reason!r}): {exc}"
            ) from exc

        if len(results) != len(nodes):
            raise ValueError(
                f"ToolImplementor returned {len(results)} results for {len(nodes)} nodes"
            )

        for node, result in zip(nodes, results):
            raw = result.get("implementation", "")
            normalised = _normalise_indent(raw)
            syntax_error = _check_syntax(normalised)
            if syntax_error:
                normalised = await self._retry_until_valid(node, normalised, syntax_error)
            node.implementation = normalised
            logger.debug("Implementation generated for: %s", node.name)


    async def _retry_until_valid(
        self, node: SkillNode, bad_body: str, error: str
    ) -> str:
        """Re-prompt the LLM with the syntax error until the body parses, up to _MAX_SYNTAX_RETRIES."""
        for attempt in range(1, _MAX_SYNTAX_RETRIES + 1):
            logger.warning(
                "Syntax error in '%s' (attempt %d/%d): %s",
                node.name, attempt, _MAX_SYNTAX_RETRIES, error,
            )
            user_content = (
                f"The implementation you generated for '{node.name}' has a Python syntax error:\n\n"
                f"  {error}\n\n"
                f"Broken code:\n```python\n{bad_body}\n```\n\n"
                "Fix the syntax error and return the corrected function body as a JSON array "
                "with a single entry: [{\"implementation\": \"<fixed body>\"}]\n"
                "Rules: every line must be indented 4 spaces; no def line; no markdown."
            )
            message = await self._call(
                messages=[{"role": "user", "content": user_content}],
                output_schema=_OUTPUT_SCHEMA,
            )
            text_block = _find_text_block(message)
            try:
                results = json.loads(text_block.text)
                fixed = _normalise_indent(results[0].get("implementation", ""))
            except Exception:
                continue
            new_error = _check_syntax(fixed)
            if not new_error:
                logger.info("Syntax fixed for '%s' on retry %d.", node.name, attempt)
                return fixed
            bad_body, error = fixed, new_error

        logger.error(
            "Could not fix syntax for '%s' after %d retries; last error: %s",
            node.name, _MAX_SYNTAX_RETRIES, error,
        )
        return bad_body


def _check_syntax(body: str) -> str | None:
    """Return a syntax error message if body is not valid Python, else None."""
    # Wrap in a dummy function so the indented body is a valid parse unit
    wrapped = "def _f():\n" + "\n".join("    " + line if line.strip() else line for line in body.splitlines())
    try:
        ast.parse(wrapped)
        return None
    except SyntaxError as exc:
        return f"{exc.msg} (line {exc.lineno})"


def _normalise_indent(body: str) -> str:
    """Normalise function body to use exactly 4-space base indentation.

    Anchors on the first non-empty line's indent level (the LLM's actual base
    indent) rather than the minimum-common-whitespace baseline, which breaks
    when import statements sit at column-0 while body lines are at column-4.
    """
    lines = body.splitlines()
    # Strip empty leading/trailing lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return "    raise NotImplementedError()"
    # Determine the base indent from the first non-empty line
    first_non_empty = next((l for l in lines if l.strip()), lines[0])
    base_indent = len(first_non_empty) - len(first_non_empty.lstrip())
    result = []
    for line in lines:
        if not line.strip():
            result.append("")
            continue
        line_indent = len(line) - len(line.lstrip())
        # Map base_indent → 4, base_indent+4 → 8, etc.
        relative = line_indent - base_indent
        result.append("    " + " " * max(relative, 0) + line.lstrip())
    return "\n".join(result)
