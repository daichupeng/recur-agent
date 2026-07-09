"""Decomposer agent: classifies nodes as COMPOSITE or ATOMIC and drafts children."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.agents.base_agent import BaseAgent, _find_text_block
from src.orchestrator.state import Contract, CompositionType, ExecType, NodeType, SkillNode
from src.skill_lib import SkillLib

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Decomposer Agent in a recursive skill-tree generation system.

Your job is to classify each skill node as either COMPOSITE or ATOMIC, and for COMPOSITE nodes
to draft their immediate sub-skills.

## Default to ATOMIC — the key insight

Skills in this system are NOT simple one-liners. Each atomic skill is a fully-implemented
Python function (or a richly-instructed LLM agent) that can contain imports, helper logic,
loops, conditionals, error handling, and multi-step computation — all within a single skill
unit. This means a skill that "does several things" but stays within one domain and one
execution pattern is STILL atomic.

ATOMIC exec types:
1. DETERMINISTIC_CODE — pure Python logic (string/math/data transforms, parsing, calculations,
   filtering, aggregation, sorting — even multi-step pipelines within one domain).
2. EXTERNAL_API — one external service (a single API client call, even if it includes
   pagination, retries, or response normalization).
3. OPENSOURCE_LIBRARY — specialized computation using well-known libraries (pandas, numpy,
   scikit-learn, yfinance, etc.) — even complex multi-step workflows within that library.
4. LLM_PROMPT — a single LLM call (classification, extraction, summarization, generation,
   scoring — even over many items in one prompt).

A skill MUST stay ATOMIC if it operates within one of the above patterns, even if its
implementation would be dozens of lines of code.

## Only go COMPOSITE when you MUST

A skill is COMPOSITE only when it genuinely requires:
  (a) A LOOP where one agent must repeat until a dynamic exit condition is met at runtime, OR
  (b) Crossing fundamentally different domains/execution patterns where the sub-tasks are
      independently useful AND the combination cannot be expressed as a single function
      (e.g. "fetch tweets from API, then analyze with an LLM" — these are truly different
      agents and the split has clear value).

Do NOT go composite for:
  - Multi-step logic within one domain (e.g. "compute similarity then cluster" → ATOMIC OPENSOURCE_LIBRARY)
  - Sequential transforms over the same data type (e.g. "normalize then score then aggregate" → ATOMIC DETERMINISTIC_CODE)
  - Aggregation + distribution + ranking → one ATOMIC DETERMINISTIC_CODE
  - Any task expressible as a single Python function, even a complex one

## ADK runtime constraints (only relevant when you DO choose COMPOSITE)

SEQUENTIAL — use by default for ordered multi-step flow across truly different agents.
  Generated as: SequentialAgent(sub_agents=[child1, child2, ...])

PARALLEL — use ONLY when children are fully independent (no data dependency between them).
  Generated as: ParallelAgent(sub_agents=[child1, child2, ...])

LOOP — use ONLY when exactly ONE child repeats until it self-terminates via ADK escalation.
  Generated as: LoopAgent(sub_agents=[one_agent], max_iterations=10)
  The single sub-agent must call actions.escalate() to terminate.

LLM_COORDINATOR — use ONLY for dynamic routing where the routing logic is ambiguous at
  design time and cannot be expressed as a sequence.

## Composite children limit
When you do make a node COMPOSITE, generate 2-3 children maximum. If 4-5 children feel
necessary, that usually means the scope is too broad — collapse related steps into atoms.

Rules of thumb:
- Atomic interfaces: inputs and outputs are strict data structures (Dict, bool, str, int, etc.)
- Composite interfaces: outputs are state transitions or ambiguous objects feeding downstream engines

For COMPOSITE nodes, generate 2-3 direct sub-skills at the next level of granularity.
Sub-skill names must be snake_case function-style identifiers.

## Data-flow contract (every node)

For EVERY node you classify, declare a `contract`: the ADK session-state keys it consumes
(`reads`) and produces (`writes`). Each is an object mapping a snake_case state key to a
short type/description string, e.g. {"stripe_event": "dict — raw webhook payload"}.

For a COMPOSITE node you must ALSO propose a `contract` for each child such that the children
collectively realize the parent's contract given the chosen composition_type. This wiring is
linted deterministically before human review, so make the keys chain correctly:

- SEQUENTIAL: children run in order. The parent's `reads` are available to the first child;
  each later child may read any key an earlier sibling wrote. Together the children must end
  up writing every key in the parent's `writes`.
- PARALLEL: every child reads only from the parent's `reads` (children cannot see each other's
  writes). Children must write DISJOINT key sets, and their union must cover the parent's
  `writes`.
- LOOP: the single child's `reads` and `writes` must be the SAME key set (shape-stable across
  iterations) and must include an explicit termination-condition key (e.g. "is_done").
- LLM_COORDINATOR: each child independently reads a subset of the parent's `reads` and writes a
  superset of the parent's `writes` (each child is a valid standalone path).

Use consistent snake_case key names so a producer's write key exactly matches a consumer's read key.

## Persistence scope (every state key)

For each state key in a node's contract, also declare its durability in a `scopes` object
mapping the key to "EPHEMERAL" or "PERSISTENT". Default every state key to EPHEMERAL. Only
mark a key PERSISTENT if the node description explicitly implies durability across separate
invocations (words like 'remember', 'history', 'next time', 'over time', 'previously seen').
Do not mark PERSISTENT speculatively. You may omit a key from `scopes` entirely to leave it
EPHEMERAL — only list the keys you are marking PERSISTENT.

You MUST respond with a JSON array parallel to the input array — one entry per input node:
[
  {
    "node_type": "composite" | "atomic",
    "exec_type": "DETERMINISTIC_CODE" | "EXTERNAL_API" | "LLM_PROMPT" | "OPENSOURCE_LIBRARY" | null,
    "composition_type": "SEQUENTIAL" | "PARALLEL" | "LOOP" | "LLM_COORDINATOR" | null,
    "contract": {"reads": {"key": "type/desc", ...}, "writes": {"key": "type/desc", ...}},
    "children": [
      {"name": "...", "description": "...", "contract": {"reads": {...}, "writes": {...}}},
      ...
    ]
  },
  ...
]

- If node_type is "atomic": exec_type must be set, composition_type must be null, children must be []
- If node_type is "composite": exec_type must be null, composition_type must be set, children must be non-empty
- Every node (and every child) must include a "contract" with "reads" and "writes" objects (use {} when empty)
- The array must have exactly the same length as the input array
"""

# A contract is {reads: {key: type/desc}, writes: {key: type/desc}}. Values are free-form
# strings; additionalProperties lets the model name arbitrary state keys.
_CONTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reads": {"type": "object", "additionalProperties": {"type": "string"}},
        "writes": {"type": "object", "additionalProperties": {"type": "string"}},
        "scopes": {
            "type": "object",
            "additionalProperties": {"type": "string", "enum": ["EPHEMERAL", "PERSISTENT"]},
        },
    },
    "required": ["reads", "writes"],
    "additionalProperties": False,
}

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
            "contract": _CONTRACT_SCHEMA,
            "children": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "contract": _CONTRACT_SCHEMA,
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


def _parse_contract(raw: Optional[dict[str, Any]]) -> Optional[Contract]:
    """Build a Contract from an LLM contract dict, tolerating missing/partial data."""
    if not raw:
        return None
    reads = {str(k): str(v) for k, v in (raw.get("reads") or {}).items()}
    writes = {str(k): str(v) for k, v in (raw.get("writes") or {}).items()}
    # Only keep explicitly PERSISTENT scopes; EPHEMERAL is the implied default.
    scopes = {
        str(k): str(v)
        for k, v in (raw.get("scopes") or {}).items()
        if str(v) == "PERSISTENT"
    }
    return Contract(reads=reads, writes=writes, scopes=scopes)


class DecomposerAgent(BaseAgent):
    def __init__(self, skill_lib: Optional[SkillLib] = None) -> None:
        super().__init__(system_prompt=_SYSTEM_PROMPT)
        self._skill_lib = skill_lib

    async def decompose(self, nodes: list[SkillNode], hint: str | None = None) -> list[SkillNode]:
        """Classify and expand a batch of nodes. Returns the same nodes mutated in-place.

        Args:
            nodes: Nodes to classify and expand.
            hint: Optional correction hint prepended to the user turn (used by HITL
                  per-node re-decompose to steer the LLM without a full layer rollback).
        """
        if not nodes:
            return nodes

        # Build skill_lib context to inject into the prompt when matches exist
        skill_context = ""
        if self._skill_lib:
            all_query = " ".join(f"{n.name} {n.description}" for n in nodes)
            matches = self._skill_lib.search("", all_query, top_k=5)
            if matches:
                refs = "\n".join(
                    f"  - {e.name} ({e.exec_type}): {e.description}" for e in matches
                )
                skill_context = (
                    "\n\nThe following skills already exist in the shared skill library. "
                    "Reuse them as atomic children (using exactly their listed name) when "
                    "they match a sub-task, rather than proposing brand-new equivalents:\n"
                    + refs
                    + "\n"
                )

        node_dicts = [
            {"name": n.name, "description": n.description} for n in nodes
        ]
        classify_text = (
            "Classify and expand the following skill nodes:\n"
            + json.dumps(node_dicts, indent=2)
            + skill_context
        )
        user_content = f"{hint}\n\n{classify_text}" if hint else classify_text

        results = await self._call_with_count_check(user_content, nodes)

        for node, result in zip(nodes, results):
            node.node_type = NodeType(result["node_type"])
            if result.get("exec_type"):
                node.exec_type = ExecType(result["exec_type"])
            node.contract = _parse_contract(result.get("contract"))
            if node.node_type == NodeType.COMPOSITE:
                if result.get("composition_type"):
                    node.composition_type = CompositionType(result["composition_type"])
                for child_dict in result.get("children", []):
                    child = SkillNode(
                        name=child_dict["name"],
                        description=child_dict["description"],
                        parent_id=node.id,
                        depth=node.depth + 1,
                        contract=_parse_contract(child_dict.get("contract")),
                    )
                    # If this child name matches a skill_lib entry, pre-fill
                    # its fields and mark the reference so we skip re-implementation.
                    if self._skill_lib:
                        lib_entry = self._skill_lib.get(child_dict["name"])
                        if lib_entry is None:
                            # Fuzzy-match by description as a fallback
                            candidates = self._skill_lib.search(
                                child_dict["name"], child_dict["description"], top_k=1
                            )
                            lib_entry = candidates[0] if candidates else None
                        if lib_entry:
                            child.skill_lib_ref = lib_entry.name
                            child.exec_type = ExecType(lib_entry.exec_type) if lib_entry.exec_type else None
                            child.node_type = NodeType.ATOMIC
                            if lib_entry.input_schema:
                                child.input_schema = lib_entry.input_schema
                            if lib_entry.output_schema:
                                child.output_schema = lib_entry.output_schema
                            if lib_entry.implementation:
                                child.implementation = lib_entry.implementation
                            if lib_entry.instruction:
                                child.instruction = lib_entry.instruction
                            logger.info(
                                "Child '%s' resolved from skill_lib entry '%s'.",
                                child.name, lib_entry.name,
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
