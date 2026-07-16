"""Contract Linter: deterministic data-flow check across a composite's children.

Runs after complexity review and before HITL-1. For each COMPOSITE node it verifies that
the children's declared contracts (reads/writes over ADK session-state keys) correctly chain
together and discharge the parent's contract, keyed off the parent's composition_type.

Detection is PURE PYTHON — no LLM. Violations are attached to nodes as `contract_note`
(the same field/UI pattern the Complexity Reviewer uses for `review_note`), surfaced to the
human at HITL-1. The LLM is only invoked, optionally, to suggest repairs — never to detect.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base_agent import BaseAgent, _find_text_block
from src.orchestrator.state import CompositionType, NodeType, SkillNode

logger = logging.getLogger(__name__)

# Substrings that mark a LOOP termination-condition state key.
_TERMINATION_HINTS = (
    "done", "complete", "finished", "terminate", "stop", "converged",
    "should_continue", "continue", "is_final", "exit", "escalate",
)

_SYSTEM_PROMPT = """You are the Contract Repair Advisor in a recursive skill-tree system.

You are given a parent composite node, its composition_type, its declared data-flow contract
(reads/writes over session-state keys), its children's proposed contracts, and a list of
detected wiring violations. Suggest the smallest change to the children's reads/writes (or the
parent's contract) that would resolve the violations.

Keep it to 1-2 short sentences, plain text, no markdown. Do not restate the violation; propose
the fix (e.g. "have normalize_input write 'clean_text' so score_text can read it").
"""

_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"suggestion": {"type": "string"}},
    "required": ["suggestion"],
    "additionalProperties": False,
}


def _reads(node: SkillNode) -> set[str]:
    return set(node.contract.reads.keys()) if node.contract else set()


def _writes(node: SkillNode) -> set[str]:
    return set(node.contract.writes.keys()) if node.contract else set()


def _has_termination_key(keys: set[str]) -> bool:
    return any(any(h in k.lower() for h in _TERMINATION_HINTS) for k in keys)


class ContractLinterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(system_prompt=_SYSTEM_PROMPT)

    # ── Public entry points ──────────────────────────────────────────────────

    def lint_layer(self, nodes: list[SkillNode]) -> None:
        """Lint every COMPOSITE node in `nodes` against its children (in place).

        Non-composite nodes are left untouched. Idempotent: each parent's contract_note is
        recomputed from scratch so re-runs never accumulate stale notes.
        """
        flagged = 0
        for node in nodes:
            if node.node_type == NodeType.COMPOSITE and node.children:
                if self.lint_group(node):
                    flagged += 1
        logger.info("ContractLint: %d composite group(s) flagged.", flagged)

    def lint_group(self, parent: SkillNode) -> bool:
        """Lint a single (parent, children) group. Returns True if any violation was found.

        Clears and recomputes parent.contract_note. Does NOT touch child contract_notes so
        the note stays a single, group-level summary (matching the Complexity Reviewer UI).
        """
        parent.contract_note = None
        if not parent.children:
            return False

        violations = self._detect(parent)
        if not violations:
            return False

        parent.contract_note = "CONTRACT: " + "; ".join(violations)
        logger.info("Contract violation on '%s': %s", parent.name, parent.contract_note)
        return True

    # ── Pure-logic detection ─────────────────────────────────────────────────

    def _detect(self, parent: SkillNode) -> list[str]:
        """Return a list of human-readable violation strings for this group."""
        if parent.contract is None:
            return ["parent contract not declared"]

        missing_child = [c.name for c in parent.children if c.contract is None]
        if missing_child:
            return [f"child(ren) missing a contract: {', '.join(missing_child)}"]

        comp = parent.composition_type
        if comp == CompositionType.SEQUENTIAL:
            return self._detect_sequential(parent)
        if comp == CompositionType.PARALLEL:
            return self._detect_parallel(parent)
        if comp == CompositionType.LOOP:
            return self._detect_loop(parent)
        if comp == CompositionType.LLM_COORDINATOR:
            return self._detect_coordinator(parent)
        # No composition_type set yet — structural check elsewhere flags it; skip data-flow.
        return []

    def _detect_sequential(self, parent: SkillNode) -> list[str]:
        violations: list[str] = []
        available = set(parent.contract.reads.keys())
        for child in parent.children:
            unmet = _reads(child) - available
            for key in sorted(unmet):
                violations.append(
                    f"'{child.name}' reads '{key}' but no prior sibling (or parent input) writes it"
                )
            available |= _writes(child)
        unproduced = set(parent.contract.writes.keys()) - available
        for key in sorted(unproduced):
            violations.append(f"parent promises to write '{key}' but no child produces it")
        return violations

    def _detect_parallel(self, parent: SkillNode) -> list[str]:
        violations: list[str] = []
        parent_reads = set(parent.contract.reads.keys())
        for child in parent.children:
            unmet = _reads(child) - parent_reads
            for key in sorted(unmet):
                violations.append(
                    f"'{child.name}' reads '{key}' but parallel children can only read the parent's inputs"
                )
        # Pairwise-disjoint writes
        seen: dict[str, str] = {}
        for child in parent.children:
            for key in sorted(_writes(child)):
                if key in seen:
                    violations.append(
                        f"'{child.name}' and '{seen[key]}' both write '{key}' (parallel writes must be disjoint)"
                    )
                else:
                    seen[key] = child.name
        union_writes = set().union(*[_writes(c) for c in parent.children]) if parent.children else set()
        unproduced = set(parent.contract.writes.keys()) - union_writes
        for key in sorted(unproduced):
            violations.append(f"parent promises to write '{key}' but no child produces it")
        return violations

    def _detect_loop(self, parent: SkillNode) -> list[str]:
        violations: list[str] = []
        if len(parent.children) != 1:
            violations.append(f"LOOP must have exactly one child (found {len(parent.children)})")
            return violations
        child = parent.children[0]
        reads, writes = _reads(child), _writes(child)
        if reads != writes:
            only_read = sorted(reads - writes)
            only_write = sorted(writes - reads)
            detail = []
            if only_read:
                detail.append(f"reads-only {only_read}")
            if only_write:
                detail.append(f"writes-only {only_write}")
            violations.append(
                f"'{child.name}' reads/writes differ ({'; '.join(detail)}); a LOOP body must be shape-stable"
            )
        if not _has_termination_key(reads | writes):
            violations.append(
                f"'{child.name}' has no explicit termination-condition key (e.g. 'is_done')"
            )
        return violations

    def _detect_coordinator(self, parent: SkillNode) -> list[str]:
        """Per-capability standalone check for a routing coordinator.

        A coordinator routes to ONE capability per user turn; cross-capability data flows
        through session state, not structural sequencing. So — unlike SEQUENTIAL/PARALLEL —
        we do NOT require each child to reproduce the parent's full `writes`. We only require
        each child to be a valid standalone path (reads a subset of the parent's inputs), plus
        RoutingSpec-level checks (unresolved child_name, uncovered child, undefined fallback).
        """
        violations: list[str] = []
        parent_reads = set(parent.contract.reads.keys())
        child_names = {c.name for c in parent.children}

        for child in parent.children:
            unmet = _reads(child) - parent_reads
            for key in sorted(unmet):
                violations.append(
                    f"'{child.name}' reads '{key}' which is not in the parent's inputs"
                )

        # RoutingSpec-aware checks (only when routing metadata is present).
        routing = parent.routing
        if routing is not None:
            routed_children: set[str] = set()
            for rule in routing.routes:
                if rule.child_name not in child_names:
                    violations.append(
                        f"route target '{rule.child_name}' does not match any child"
                    )
                else:
                    routed_children.add(rule.child_name)
            fallback = (routing.fallback or "").strip()
            uncovered = child_names - routed_children
            if fallback in child_names:
                uncovered.discard(fallback)
            for name in sorted(uncovered):
                violations.append(f"child '{name}' has no route and is not the fallback")
            if not fallback and not (routing.clarify_when or "").strip():
                violations.append(
                    "no fallback and no clarify_when: unmatched user input has no defined behavior"
                )
        return violations

    # ── Optional LLM repair suggestion ───────────────────────────────────────

    async def suggest_repair(self, parent: SkillNode) -> str | None:
        """One small LLM call proposing a fix for a flagged group. Non-blocking, best-effort.

        Only meaningful when parent.contract_note is set. Appends the suggestion to the note.
        Never mutates structure. Returns the suggestion text, or None on failure/no note.
        """
        if not parent.contract_note:
            return None
        payload = {
            "parent": parent.name,
            "composition_type": parent.composition_type.value if parent.composition_type else None,
            "parent_contract": parent.contract.model_dump() if parent.contract else None,
            "children": [
                {"name": c.name, "contract": c.contract.model_dump() if c.contract else None}
                for c in parent.children
            ],
            "violations": parent.contract_note,
        }
        try:
            message = await self._call(
                messages=[{"role": "user", "content": "Suggest a repair:\n" + json.dumps(payload, indent=2)}],
                output_schema=_REPAIR_SCHEMA,
            )
            suggestion = json.loads(_find_text_block(message).text).get("suggestion", "").strip()
        except Exception as exc:  # best-effort; detection already stands on its own
            logger.warning("Contract repair suggestion failed for '%s': %s", parent.name, exc)
            return None
        if suggestion:
            parent.contract_note = f"{parent.contract_note} | FIX: {suggestion}"
        self.log_usage()
        return suggestion or None
