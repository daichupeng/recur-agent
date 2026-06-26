"""Decomposer agent: classifies nodes as COMPOSITE or ATOMIC and drafts children."""
from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base_agent import BaseAgent, _find_text_block
from src.orchestrator.state import CompositionType, ExecType, NodeType, SkillNode

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Decomposer Agent in a recursive skill-tree generation system.

Your job is to classify each skill node as either COMPOSITE or ATOMIC, and for COMPOSITE nodes
to draft their immediate sub-skills.

ATOMIC DEFINITION — a skill is atomic if its core logic can ENTIRELY be executed using exactly
ONE of the following three patterns, with NO further routing, branching, or sub-coordination:
1. DETERMINISTIC_CODE — purely standard code (SQL query, regex parse, math formula, JSON mapping)
2. EXTERNAL_API — a single external call (Stripe charge, Slack webhook, weather fetch)
3. OPENSOURCE_LIBRARY - a standard code utilizing well-known open-source libraries (pandas, yfinance, scikit-learn, etc.) Used for specialized tasks where specific libraries are good at.
4. LLM_PROMPT — a single-turn LLM prompt with zero tool-calling loops (sentiment classify,
   schema extraction, single-paragraph summarize)

A skill is COMPOSITE if it requires multiple of the above, orchestration, routing, or
conditional logic that spans more than one of those patterns.

## ADK runtime constraints (you MUST follow these when choosing composition_type)

SEQUENTIAL — use by default for any ordered multi-step flow.
  Generated as: SequentialAgent(sub_agents=[child1, child2, ...])
  Children run in sequence; each child's output is passed to the next via session state.

PARALLEL — use ONLY when children are fully independent (no data dependency between them).
  Generated as: ParallelAgent(sub_agents=[child1, child2, ...])
  All children run concurrently; their outputs are merged.

LOOP — use ONLY when:
  (a) exactly ONE child agent is being repeated, AND
  (b) that child agent can detect a terminal condition and signal it via ADK escalation.
  Never use LOOP for multi-step flows or when the exit condition is external.
  If unsure, use SEQUENTIAL with a dedicated termination-check step instead.
  Generated as: LoopAgent(sub_agents=[one_agent], max_iterations=10)
  The single sub-agent must call actions.escalate() to terminate the loop.

LLM_COORDINATOR — use for routing/dispatch where the decision is ambiguous at design time.
  The coordinator is an LlmAgent that routes to children based on context.
  Its routing instruction is auto-generated from each child's description.
  Generated as: LlmAgent(sub_agents=[...], instruction="Route based on ...")
  Use this only when the routing logic cannot be expressed as a predictable sequence.

## Simplicity bias (important)
Prefer ATOMIC over COMPOSITE whenever there is reasonable doubt. If a skill *could* be
expressed as a single LLM prompt or a single deterministic transform, classify it ATOMIC even
if it feels slightly richer than a textbook example. Over-decomposition creates unnecessary
layers that are harder to maintain and test.

When you do classify a node COMPOSITE, prefer 2-3 tightly-scoped children over 4-5 broader
ones. Each child should itself be obviously atomic or obviously composite — avoid children
whose classification is ambiguous. Avoid introducing a composite child solely to group two
atomic siblings; flatten when possible.

Rules of thumb:
- Atomic interfaces: inputs and outputs are strict data structures (Dict, bool, str, int, etc.)
- Composite interfaces: outputs are state transitions or ambiguous objects feeding downstream engines

For COMPOSITE nodes, generate 2-5 direct sub-skills at the next level of granularity.
Sub-skill names must be snake_case function-style identifiers.

You MUST respond with a JSON array parallel to the input array — one entry per input node:
[
  {
    "node_type": "composite" | "atomic",
    "exec_type": "DETERMINISTIC_CODE" | "EXTERNAL_API" | "LLM_PROMPT" | "OPENSOURCE_LIBRARY" | null,
    "composition_type": "SEQUENTIAL" | "PARALLEL" | "LOOP" | "LLM_COORDINATOR" | null,
    "children": [
      {"name": "...", "description": "..."},
      ...
    ]
  },
  ...
]

- If node_type is "atomic": exec_type must be set, composition_type must be null, children must be []
- If node_type is "composite": exec_type must be null, composition_type must be set, children must be non-empty
- The array must have exactly the same length as the input array
"""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "node_type": {"type": "string", "enum": ["composite", "atomic"]},
            "exec_type": {
                "anyOf": [
                    {"type": "string", "enum": ["DETERMINISTIC_CODE", "EXTERNAL_API", "LLM_PROMPT", "OPENSOURCE_LIBRARY"]},
                    {"type": "null"},
                ]
            },
            "composition_type": {
                "anyOf": [
                    {"type": "string", "enum": ["SEQUENTIAL", "PARALLEL", "LOOP", "LLM_COORDINATOR"]},
                    {"type": "null"},
                ]
            },
            "children": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "description"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["node_type", "children"],
        "additionalProperties": False,
    },
}


class DecomposerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(system_prompt=_SYSTEM_PROMPT)

    async def decompose(self, nodes: list[SkillNode], hint: str | None = None) -> list[SkillNode]:
        """Classify and expand a batch of nodes. Returns the same nodes mutated in-place.

        Args:
            nodes: Nodes to classify and expand.
            hint: Optional correction hint prepended to the user turn (used by HITL
                  per-node re-decompose to steer the LLM without a full layer rollback).
        """
        if not nodes:
            return nodes

        node_dicts = [
            {"name": n.name, "description": n.description} for n in nodes
        ]
        classify_text = (
            "Classify and expand the following skill nodes:\n"
            + json.dumps(node_dicts, indent=2)
        )
        user_content = f"{hint}\n\n{classify_text}" if hint else classify_text

        results = await self._call_with_count_check(user_content, nodes)

        for node, result in zip(nodes, results):
            node.node_type = NodeType(result["node_type"])
            if result.get("exec_type"):
                node.exec_type = ExecType(result["exec_type"])
            if node.node_type == NodeType.COMPOSITE:
                if result.get("composition_type"):
                    node.composition_type = CompositionType(result["composition_type"])
                for child_dict in result.get("children", []):
                    child = SkillNode(
                        name=child_dict["name"],
                        description=child_dict["description"],
                        parent_id=node.id,
                        depth=node.depth + 1,
                    )
                    node.children.append(child)

        self.log_usage()
        return nodes

    async def _call_with_count_check(
        self, user_content: str, nodes: list[SkillNode]
    ) -> list[dict[str, Any]]:
        """Call the LLM and ensure the result has exactly len(nodes) entries.

        Retry once with an explicit count reminder if the first response is short.
        Fall back to one-node-at-a-time if the retry also mismatches.
        """
        expected = len(nodes)

        # First attempt
        message = await self._call(
            messages=[{"role": "user", "content": user_content}],
            output_schema=_OUTPUT_SCHEMA,
        )
        results: list[dict[str, Any]] = json.loads(_find_text_block(message).text)
        if len(results) == expected:
            return results

        logger.warning(
            "Decomposer returned %d results for %d nodes — retrying with explicit count.",
            len(results), expected,
        )

        # Second attempt: reinforce the expected count
        node_dicts = [{"name": n.name, "description": n.description} for n in nodes]
        retry_content = (
            f"IMPORTANT: You must return EXACTLY {expected} JSON objects — one per input node. "
            f"Your previous response had {len(results)} entries, which is wrong.\n\n"
            "Classify and expand the following skill nodes:\n"
            + json.dumps(node_dicts, indent=2)
        )
        message = await self._call(
            messages=[{"role": "user", "content": retry_content}],
            output_schema=_OUTPUT_SCHEMA,
        )
        results = json.loads(_find_text_block(message).text)
        if len(results) == expected:
            return results

        logger.warning(
            "Retry also returned %d results — falling back to one-at-a-time.", len(results)
        )

        # Last resort: process each node individually and concatenate
        all_results: list[dict[str, Any]] = []
        for node in nodes:
            single_content = (
                "Classify and expand the following skill nodes:\n"
                + json.dumps([{"name": node.name, "description": node.description}], indent=2)
            )
            msg = await self._call(
                messages=[{"role": "user", "content": single_content}],
                output_schema=_OUTPUT_SCHEMA,
            )
            single = json.loads(_find_text_block(msg).text)
            if not single:
                raise ValueError(f"Decomposer returned empty result for node '{node.name}'")
            all_results.append(single[0])

        return all_results
