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

For state key naming: use snake_case, derive from the node's name and schema properties.
The output_schema properties should map to state_writes keys.
The input_schema properties should map to state_reads keys (or be passed directly if this is
the first node in a sequential chain).

You MUST respond with a JSON array parallel to the input array — one entry per input node:
[
  {
    "instruction": "<full system prompt string>",
    "state_reads": ["key1", "key2"],
    "state_writes": ["key3"]
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
        },
        "required": ["instruction", "state_reads", "state_writes"],
        "additionalProperties": False,
    },
}


class PromptEngineerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(system_prompt=_SYSTEM_PROMPT)

    async def engineer(self, nodes: list[SkillNode], feedback: str | None = None) -> list[SkillNode]:
        """Generate instruction and session-state wiring for LLM atomic nodes (in-place).

        Args:
            nodes: Nodes to engineer prompts for.
            feedback: Optional human feedback for per-skill retry (HITL-2). When provided,
                      it is appended to the user prompt so the LLM can address the concern.
        """
        if not nodes:
            return nodes

        node_dicts = [
            {
                "name": n.name,
                "description": n.description,
                "input_schema": n.input_schema,
                "output_schema": n.output_schema,
            }
            for n in nodes
        ]
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
        results: list[dict[str, Any]] = json.loads(text_block.text)

        if len(results) != len(nodes):
            raise ValueError(
                f"PromptEngineer returned {len(results)} results for {len(nodes)} nodes"
            )

        for node, result in zip(nodes, results):
            node.instruction = result["instruction"]
            node.state_reads = result.get("state_reads", [])
            node.state_writes = result.get("state_writes", [])
            logger.debug("Instruction engineered for: %s", node.name)

        self.log_usage()
        return nodes
