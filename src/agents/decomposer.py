"""Decomposer agent: classifies nodes as COMPOSITE or ATOMIC and drafts children."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.agents.base_agent import BaseAgent, _find_text_block
from src.orchestrator.state import (
    Contract,
    CompositionType,
    ExecType,
    NodeType,
    RouteRule,
    RoutingSpec,
    SkillNode,
)
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

LLM_COORDINATOR — a conversational orchestrator at the product boundary. It calls its
  children as tools (AgentTool), reads their results, and authors ONE final reply to the user
  in a single voice. Use it for a conversational product whose top-level capabilities are
  heterogeneous (the user asks for different things on different turns). It can also decline to
  call anything and ask the user a clarifying question. NOT a fixed sequence; children are
  independent capabilities, and cross-capability data flows through session state.

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
- LLM_COORDINATOR: each child independently reads a subset of the parent's `reads` (each child
  is a valid standalone capability). Children need NOT each reproduce the parent's full `writes`;
  the coordinator calls one (or few) capabilities per user turn and authors the reply itself, and
  cross-capability data flows through session state, not structural sequencing.

Use consistent snake_case key names so a producer's write key exactly matches a consumer's read key.

## Routing (LLM_COORDINATOR nodes only)

When you classify a node as COMPOSITE + LLM_COORDINATOR, ALSO return a `routing` object so a
human can review how user intents map to capabilities:
- `routes`: you MUST include EXACTLY ONE entry per child (never leave `routes` empty when there
  are children). Each is `{child_name, trigger, examples}` where:
  - `child_name` MUST exactly match one of the children's names.
  - `trigger`: a short natural-language description of the USER INTENT that maps to this child
    (what the user is asking for — distinct from what the child does).
  - `examples`: 1-3 example user utterances that should route here (optional but encouraged).
- `fallback`: how to handle chit-chat or an unmatched request — either the name of a default
  child, or an instruction to ask the user a clarifying question. Never leave this empty.
- `clarify_when`: a natural-language condition under which the coordinator should ask the user
  for more detail INSTEAD of routing (e.g. "the request is missing a required file or target").
  Leave "" if the coordinator never needs to clarify.

Only include `routing` for LLM_COORDINATOR nodes; omit it for all other node types.

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
    "routing": {  // LLM_COORDINATOR only; omit otherwise
      "routes": [{"child_name": "...", "trigger": "...", "examples": ["..."]}],
      "fallback": "...",
      "clarify_when": "..."
    },
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

# Routing metadata for an LLM_COORDINATOR node. Optional at the schema level so non-coordinator
# nodes simply omit it.
_ROUTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "routes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "child_name": {"type": "string"},
                    "trigger": {"type": "string"},
                    "examples": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["child_name", "trigger"],
                "additionalProperties": False,
            },
        },
        "fallback": {"type": "string"},
        "clarify_when": {"type": "string"},
    },
    "required": ["routes"],
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
            "routing": _ROUTING_SCHEMA,
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


def _parse_routing(raw: Optional[dict[str, Any]]) -> Optional[RoutingSpec]:
    """Build a RoutingSpec from an LLM routing dict, tolerating missing/partial data."""
    if not raw:
        return None
    routes = [
        RouteRule(
            child_name=str(r.get("child_name", "")),
            trigger=str(r.get("trigger", "")),
            examples=[str(e) for e in (r.get("examples") or [])],
        )
        for r in (raw.get("routes") or [])
        if r.get("child_name")
    ]
    return RoutingSpec(
        routes=routes,
        fallback=str(raw.get("fallback") or ""),
        clarify_when=str(raw.get("clarify_when") or ""),
    )


# Root-only instruction: bias the boundary node toward a conversational coordinator. Injected
# only when decomposing a depth-0 node; depth ≥ 1 guidance is unchanged.
_ROOT_BIAS_HINT = """ROOT NODE GUIDANCE (this node is the product's entry point / boundary):
Choose composition_type by product shape:
- Prefer COMPOSITE + LLM_COORDINATOR when the product is CONVERSATIONAL and its top-level
  capabilities are HETEROGENEOUS (the user may ask for different things on different turns —
  e.g. "analyze this file" vs "just answer a question" vs "run the whole thing"). A coordinator
  routes each user message to the right capability and can ask a clarifying question when needed.
- Choose ATOMIC when the product is single-purpose (one job, no routing).
- Choose COMPOSITE + SEQUENTIAL only for a genuinely fixed A→B→C pipeline where every input
  always flows through the same ordered stages.
When you pick LLM_COORDINATOR, you MUST also return the `routing` object described below. The
human confirms or overrides this shape at review, so bias toward the conversational coordinator
for anything that reads like a chatbot / assistant / multi-capability product."""


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
        # Root-only bias: when the boundary node (depth 0) is in this batch, prepend the
        # coordinator-root guidance. Composes with the HITL hint channel; depth ≥ 1 unaffected.
        prefix_parts = [p for p in (hint, _ROOT_BIAS_HINT if any(n.depth == 0 for n in nodes) else None) if p]
        user_content = "\n\n".join(prefix_parts + [classify_text]) if prefix_parts else classify_text

        results = await self._call_with_count_check(user_content, nodes)

        for node, result in zip(nodes, results):
            node.node_type = NodeType(result["node_type"])
            if result.get("exec_type"):
                node.exec_type = ExecType(result["exec_type"])
            node.contract = _parse_contract(result.get("contract"))
            if node.node_type == NodeType.COMPOSITE:
                if result.get("composition_type"):
                    node.composition_type = CompositionType(result["composition_type"])
                # A coordinator carries reviewable routing metadata; other composites don't.
                if node.composition_type == CompositionType.LLM_COORDINATOR:
                    node.routing = _parse_routing(result.get("routing"))
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
