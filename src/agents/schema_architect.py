"""Schema Architect agent: hydrates I/O signatures on approved atomic nodes."""
from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base_agent import BaseAgent, _find_text_block
from src.orchestrator.state import SkillNode

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Schema Architect Agent in a recursive skill-tree generation system.

You receive a list of ATOMIC skill nodes. Each atomic node represents a single, indivisible
operation (a deterministic function, a single API call, or a single-turn LLM prompt).

For each node you must define its complete static interface:
  - input_schema: the exact JSON Schema object describing the function's inputs
  - output_schema: the exact JSON Schema object describing the function's outputs

Rules:
- All schemas must be valid JSON Schema (draft-07 compatible)
- Input/output types must be strict data structures — no ambiguous "Any" or state objects
- The output schema must directly resolve the node's stated objective
- Use "additionalProperties": false on all objects
- Prefer flat schemas; avoid deeply nested structures

You MUST respond with a JSON array parallel to the input array — one entry per input node:
[
  {
    "input_schema": { ... },
    "output_schema": { ... }
  },
  ...
]
"""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        },
        "required": ["input_schema", "output_schema"],
        "additionalProperties": False,
    },
}

_BATCH_SIZE = 5


class SchemaArchitectAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(system_prompt=_SYSTEM_PROMPT)

    async def hydrate(self, nodes: list[SkillNode]) -> list[SkillNode]:
        """Add input_schema and output_schema to a batch of atomic nodes."""
        if not nodes:
            return nodes

        for batch_start in range(0, len(nodes), _BATCH_SIZE):
            batch = nodes[batch_start : batch_start + _BATCH_SIZE]
            await self._hydrate_batch(batch)

        self.log_usage()
        return nodes

    async def _hydrate_batch(self, nodes: list[SkillNode]) -> None:
        node_dicts = [
            {
                "name": n.name,
                "description": n.description,
                "exec_type": n.exec_type.value if n.exec_type else None,
            }
            for n in nodes
        ]
        user_content = (
            "Define the I/O interface for each of these atomic skill nodes:\n"
            + json.dumps(node_dicts, indent=2)
        )

        message = await self._call(
            messages=[{"role": "user", "content": user_content}],
            output_schema=_OUTPUT_SCHEMA,
        )

        text_block = _find_text_block(message)
        results: list[dict[str, Any]] = json.loads(text_block.text)

        if len(results) != len(nodes):
            raise ValueError(
                f"SchemaArchitect returned {len(results)} results for {len(nodes)} nodes"
            )

        for node, result in zip(nodes, results):
            node.input_schema = result["input_schema"]
            node.output_schema = result["output_schema"]
