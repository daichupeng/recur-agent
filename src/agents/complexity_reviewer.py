"""Complexity Reviewer agent: flags over-decomposed nodes before human review."""
from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base_agent import BaseAgent, _find_text_block
from src.orchestrator.state import NodeType, SkillNode

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Complexity Reviewer in a recursive skill-tree generation system.

You receive the result of one decomposition pass: a list of nodes that were just classified
as COMPOSITE or ATOMIC, together with any children that were drafted for composite nodes.

Your job is to identify nodes whose decomposition looks unnecessarily complex, and to leave a
brief advisory note that the human reviewer will see before approving.

Flag a node when you notice ANY of the following:
- A node was classified COMPOSITE but could plausibly be a single LLM_PROMPT or
  DETERMINISTIC_CODE call (potential over-split).
- A COMPOSITE node has 4-5 children where 2-3 tightly-scoped children would suffice.
- A child of a COMPOSITE node is itself COMPOSITE with only 2 children, creating an
  unnecessary extra layer of indirection.
- A child's description is almost identical to its parent's, suggesting the decomposition
  added no real value.

Do NOT flag nodes that are legitimately complex. When in doubt, leave the note empty.
The note is advisory only — the human may ignore it. Keep each note under 80 characters,
plain text, no markdown. Start with the concern keyword: OVER-SPLIT, TOO-MANY-CHILDREN,
SHALLOW-LAYER, or REDUNDANT.

You MUST respond with a JSON array parallel to the input array — one entry per input node:
[
  { "review_note": "<short note or empty string>" },
  ...
]
"""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "review_note": {"type": "string"},
        },
        "required": ["review_note"],
        "additionalProperties": False,
    },
}


class ComplexityReviewAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(system_prompt=_SYSTEM_PROMPT)

    async def review(self, nodes: list[SkillNode]) -> list[SkillNode]:
        """Annotate nodes with complexity warnings. Returns the same nodes mutated in-place."""
        if not nodes:
            return nodes

        node_dicts = [
            {
                "name": n.name,
                "description": n.description,
                "node_type": n.node_type.value,
                "exec_type": n.exec_type.value if n.exec_type else None,
                "children": [
                    {"name": c.name, "description": c.description, "node_type": c.node_type.value}
                    for c in n.children
                ],
            }
            for n in nodes
        ]
        user_content = (
            "Review the following decomposed nodes for unnecessary complexity:\n"
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
                f"ComplexityReviewer returned {len(results)} results for {len(nodes)} nodes"
            )

        flagged = 0
        for node, result in zip(nodes, results):
            note = result.get("review_note", "").strip()
            node.review_note = note if note else None
            if node.review_note:
                flagged += 1
                logger.info("Complexity flag on '%s': %s", node.name, node.review_note)

        logger.info(
            "ComplexityReview: %d/%d nodes flagged.", flagged, len(nodes)
        )
        self.log_usage()
        return nodes
