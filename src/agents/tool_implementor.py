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
- For OPENSOURCE_LIBRARY:
  - Implement using well-known open-source libraries (pandas, numpy, yfinance, scikit-learn, etc.).
  - Import the library inside the function body, indented with 4 spaces.
  - The body must be correct and self-contained.
- For EXTERNAL_API:
  - Prefer a well-known Python client library for the target service when one exists
    (e.g. `yfinance` for Yahoo Finance, `stripe` for Stripe, `twilio` for Twilio,
    `boto3` for AWS, `newsapi-python` for NewsAPI, `requests` for generic REST).
    Import it inside the function body.
  - Only fall back to `httpx` when no stable client library exists. When you do,
    use only documented, stable API endpoints — never undocumented internal paths
    (paths containing version slugs like `/v7/`, `/v8/` with no public spec).
  - For any API key, token, or credential: use `os.environ["KEY_NAME"]` (add
    `    import os` inside the function body). Never write a literal placeholder
    string — a missing env var will raise a clear `KeyError` on first call rather
    than silently sending wrong credentials.
  - Include basic error handling: raise a RuntimeError with a descriptive message on non-2xx status.
- Every branch must end with a return statement that produces a dict matching the output_schema.
- Do not add any explanation or markdown — return raw Python code only.

## Media artifacts (only for nodes flagged with `emits_media`)
Some nodes must ALSO emit a binary artifact (an image or a downloadable file) so the frontend
can display or download it. For a node flagged `emits_media` with `media_types` listed:
- The function is ASYNC. Its body runs inside `async def`, so you MAY use `await`.
- The function receives an extra parameter `tool_context` (already added to the signature).
- Build the binary payload as `bytes`, then persist it as an ADK artifact:
      from google.genai import types
      _part = types.Part.from_bytes(data=<the bytes>, mime_type="<one of the media_types>")
      await tool_context.save_artifact(filename="<name>.<ext>", artifact=_part)
  Every one of these lines must be indented 4 spaces (inside the function body).
- Include the artifact filename in the returned dict under the key `artifact_filename` (add it to
  the return alongside the output_schema fields).
- To generate an image without heavy deps, prefer Pillow (`from PIL import Image`) or matplotlib
  (`import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt`) writing to an
  `io.BytesIO`. For a text/CSV/PDF file, build the bytes directly.
- Still return a dict matching the output_schema (plus `artifact_filename`).

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

    async def implement(self, nodes: list[SkillNode], feedback: str | None = None) -> list[SkillNode]:
        """Generate and assign function bodies for tool atomic nodes (in-place).

        Args:
            nodes: Nodes to implement.
            feedback: Optional human feedback for per-skill retry (HITL-2). When provided,
                      it is appended to the user prompt so the LLM can address the concern.
        """
        if not nodes:
            return nodes

        # Process in batches to avoid overly long prompts
        for batch_start in range(0, len(nodes), _BATCH_SIZE):
            batch = nodes[batch_start : batch_start + _BATCH_SIZE]
            await self._implement_batch(batch, feedback=feedback)

        self.log_usage()
        return nodes

    async def _implement_batch(self, nodes: list[SkillNode], feedback: str | None = None) -> None:
        node_dicts = [
            {
                "name": n.name,
                "description": n.description,
                "exec_type": n.exec_type.value if n.exec_type else None,
                "input_schema": n.input_schema,
                "output_schema": n.output_schema,
                "emits_media": bool(n.output_media_types),
                "media_types": n.output_media_types,
            }
            for n in nodes
        ]

        user_content = (
            "Generate implementations for these atomic tool nodes:\n"
            + json.dumps(node_dicts, indent=2)
        )
        if feedback:
            user_content += f"\n\nUser feedback (address this in your implementation): {feedback}"

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
            is_async = bool(node.output_media_types)
            normalised = _normalise_indent(raw)
            syntax_error = _check_syntax(normalised, is_async=is_async)
            if syntax_error:
                normalised = await self._retry_until_valid(
                    node, normalised, syntax_error, is_async=is_async
                )
            node.implementation = normalised
            logger.debug("Implementation generated for: %s", node.name)


    async def _retry_until_valid(
        self, node: SkillNode, bad_body: str, error: str, *, is_async: bool = False
    ) -> str:
        """Re-prompt the LLM with the syntax error until the body parses, up to _MAX_SYNTAX_RETRIES."""
        for attempt in range(1, _MAX_SYNTAX_RETRIES + 1):
            logger.warning(
                "Syntax error in '%s' (attempt %d/%d): %s",
                node.name, attempt, _MAX_SYNTAX_RETRIES, error,
            )
            async_note = (
                " This is an ASYNC function body — `await` is allowed at the top level."
                if is_async else ""
            )
            user_content = (
                f"The implementation you generated for '{node.name}' has a Python syntax error:\n\n"
                f"  {error}\n\n"
                f"Broken code:\n```python\n{bad_body}\n```\n\n"
                "Fix the syntax error and return the corrected function body as a JSON array "
                "with a single entry: [{\"implementation\": \"<fixed body>\"}]\n"
                f"Rules: every line must be indented 4 spaces; no def line; no markdown.{async_note}"
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
            new_error = _check_syntax(fixed, is_async=is_async)
            if not new_error:
                logger.info("Syntax fixed for '%s' on retry %d.", node.name, attempt)
                return fixed
            bad_body, error = fixed, new_error

        logger.error(
            "Could not fix syntax for '%s' after %d retries; last error: %s",
            node.name, _MAX_SYNTAX_RETRIES, error,
        )
        return bad_body


def _check_syntax(body: str, *, is_async: bool = False) -> str | None:
    """Return a syntax error message if body is not valid Python, else None.

    Media nodes compile to `async def`, so their body may use top-level `await`;
    wrap in `async def` for the check or the await would be flagged as a syntax error.
    """
    # Body is already 4-space indented; just wrap in a dummy function.
    prefix = "async def _f():\n" if is_async else "def _f():\n"
    wrapped = prefix + body
    try:
        ast.parse(wrapped)
        return None
    except SyntaxError as exc:
        return f"{exc.msg} (line {exc.lineno})"


def _normalise_indent(body: str) -> str:
    """Normalise function body to exactly 4-space base indentation.

    Handles all common LLM indentation mistakes:
    - All lines at 0-indent → shift to 4.
    - All lines at N-indent → dedent to 0, re-add 4 (preserving relative nesting).
    - Mixed 0-and-positive indent (e.g. 0-indent imports + 4-indent body) →
      promote 0-indent lines to the minimum positive indent, then dedent+re-add 4.
    """
    import textwrap
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return "    raise NotImplementedError()"

    non_empty_indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    min_indent = min(non_empty_indents)
    positive_indents = [i for i in non_empty_indents if i > 0]

    if min_indent == 0 and positive_indents:
        # Some lines forgot indentation. Promote them to the minimum positive
        # indent so relative nesting is preserved after dedent.
        base = min(positive_indents)
        lines = [
            " " * base + line if (line.strip() and len(line) - len(line.lstrip()) == 0) else line
            for line in lines
        ]

    # textwrap.dedent strips the common leading whitespace, preserving nesting.
    dedented = textwrap.dedent("\n".join(lines))

    result = []
    for line in dedented.splitlines():
        result.append(("    " + line) if line.strip() else "")
    return "\n".join(result)
