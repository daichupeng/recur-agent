"""Prompt Engineer agent: generates instruction strings and session-state wiring for LLM atomic nodes."""
from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base_agent import BaseAgent, _find_text_block
from src.orchestrator.state import SkillNode

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Prompt Engineer Agent in a recursive skill-tree generation system.

You receive a list of LLM_PROMPT atomic nodes that have already been assigned input/output schemas.
Each node will be compiled into a Google ADK LlmAgent. Your job is to:

1. Write an `instruction` — the full system prompt the LlmAgent will use at runtime.
   The instruction must:
   - State the node's objective precisely.
   - Tell the agent exactly which ADK session state keys to READ at the start of each turn
     (use `tool_context.state["key"]` in prose, e.g. "Read the user profile from session state key `user_profile`").
   - Tell the agent exactly which ADK session state keys to WRITE before returning
     (e.g. "Write your result to session state key `coverage_recommendation`").
   - Be self-contained: the agent has no other context beyond this instruction and the session state.

2. Identify `state_reads` — the list of session state key names the agent reads.
3. Identify `state_writes` — the list of session state key names the agent writes.
4. Identify `state_scopes` — a mapping of state key → "PERSISTENT" for any key that must
   survive across separate invocations. Default every state key to EPHEMERAL. Only mark a
   key PERSISTENT if EITHER:
   - the node description explicitly implies durability (words like 'remember', 'history',
     'next time', 'over time', 'previously seen'), OR
   - the key appears in the node's `persistent_keys_required` list (keys that an ancestor
     composite explicitly declared PERSISTENT — you MUST honour these).
   Do not mark PERSISTENT speculatively for keys not in either category. List ONLY the
   PERSISTENT keys (omit the rest — they default to EPHEMERAL).

For state key naming: use snake_case, derive from the node's name and schema properties.
The output_schema properties should map to state_writes keys.
The input_schema properties should map to state_reads keys (or be passed directly if this is
the first node in a sequential chain).

You MUST respond with a JSON array parallel to the input array — one entry per input node:
[
  {
    "instruction": "<full system prompt string>",
    "state_reads": ["key1", "key2"],
    "state_writes": ["key3"],
    "state_scopes": {"key3": "PERSISTENT"}
  },
  ...
]
"""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "instruction": {"type": "string"},
            "state_reads": {"type": "array", "items": {"type": "string"}},
            "state_writes": {"type": "array", "items": {"type": "string"}},
            "state_scopes": {
                "type": "object",
                "additionalProperties": {"type": "string", "enum": ["EPHEMERAL", "PERSISTENT"]},
            },
        },
        "required": ["instruction", "state_reads", "state_writes"],
        "additionalProperties": False,
    },
}


class PromptEngineerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(system_prompt=_SYSTEM_PROMPT)

    async def engineer(
        self,
        nodes: list[SkillNode],
        feedback: str | None = None,
        persistent_keys_hint: dict[str, set[str]] | None = None,
    ) -> list[SkillNode]:
        """Generate instruction and session-state wiring for LLM atomic nodes (in-place).

        Args:
            nodes: Nodes to engineer prompts for.
            feedback: Optional human feedback for per-skill retry (HITL-2). When provided,
                      it is appended to the user prompt so the LLM can address the concern.
            persistent_keys_hint: Maps node_id → set of state-key names that ancestor
                composites declared PERSISTENT. The PromptEngineer must honour these by
                including them in state_scopes when the node writes them.
        """
        if not nodes:
            return nodes

        node_dicts = []
        for n in nodes:
            d: dict[str, Any] = {
                "name": n.name,
                "description": n.description,
                "input_schema": n.input_schema,
                "output_schema": n.output_schema,
            }
            required = sorted((persistent_keys_hint or {}).get(n.id, set()))
            if required:
                d["persistent_keys_required"] = required
            node_dicts.append(d)

        user_content = (
            "Generate instructions and session-state wiring for these LLM atomic nodes:\n"
            + json.dumps(node_dicts, indent=2)
        )
        if feedback:
            user_content += f"\n\nUser feedback (address this in the instruction): {feedback}"

        message = await self._call(
            messages=[{"role": "user", "content": user_content}],
            output_schema=_OUTPUT_SCHEMA,
        )

        text_block = _find_text_block(message)
        try:
            results: list[dict[str, Any]] = json.loads(text_block.text)
        except json.JSONDecodeError as e:
            # Log the problematic JSON for debugging
            logger.error("Failed to parse PromptEngineer output: %s", e)
            logger.error("Text block (first 500 chars): %s", text_block.text[:500])
            raise ValueError(
                f"PromptEngineer returned invalid JSON: {e}. "
                f"See logs for output sample."
            ) from e

        if len(results) != len(nodes):
            raise ValueError(
                f"PromptEngineer returned {len(results)} results for {len(nodes)} nodes"
            )

        for node, result in zip(nodes, results):
            node.instruction = result["instruction"]
            node.state_reads = result.get("state_reads", [])
            node.state_writes = result.get("state_writes", [])
            # Keep only explicitly PERSISTENT scopes; EPHEMERAL is the implied default.
            node.state_scopes = {
                str(k): str(v)
                for k, v in (result.get("state_scopes") or {}).items()
                if str(v) == "PERSISTENT"
            }
            logger.debug("Instruction engineered for: %s", node.name)

        self.log_usage()
        return nodes
