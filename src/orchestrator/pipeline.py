"""Main orchestration loop: snapshot → decompose → HITL-1 → schema → prompt-engineer
→ implement → HITL-2 → compile → verify-repair → debug-loop."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable, Optional

from src.agents.compiler import CompilerAgent, _snake_name
from src.agents.complexity_reviewer import ComplexityReviewAgent
from src.agents.contract_linter import ContractLinterAgent
from src.agents.debug import DebugAgent, EnvProvider
from src.agents.decomposer import DecomposerAgent
from src.agents.memory_architect import MemoryArchitectAgent
from src.agents.prompt_engineer import PromptEngineerAgent
from src.agents.schema_architect import SchemaArchitectAgent
from src.agents.tool_implementor import ToolImplementorAgent
from src.agents.ui_designer import UIDesignerAgent
from src.orchestrator.state import ExecType, NodeType, SkillNode, SkillTree
from src.skill_lib import SkillEntry, SkillLib

logger = logging.getLogger(__name__)

_MAX_REPAIR_ITERATIONS = 3


@dataclass
class RedecomposeRequest:
    """Carry a per-node re-decompose hint from the UI to the pipeline."""
    node_id: str
    new_description: Optional[str]
    hint: Optional[str]
    force_renegotiate: bool = False  # allow redecomposing a node with a FROZEN contract


@dataclass
class RetrySkillRequest:
    """Carry a per-skill retry request with optional feedback from the UI to the pipeline."""
    node_id: str
    feedback: Optional[str]


class PipelineEvents:
    """Shared async events between the pipeline and the HITL UI server."""

    def __init__(self) -> None:
        # HITL-1: structure review (after decompose)
        self.approve: asyncio.Event = asyncio.Event()
        self.rollback: asyncio.Event = asyncio.Event()
        # HITL-1 per-node re-decompose: queue of RedecomposeRequest
        self.redecompose: asyncio.Queue[RedecomposeRequest] = asyncio.Queue()

        # HITL-2: implementation review (after schema + tool implementation)
        self.approve_impl: asyncio.Event = asyncio.Event()
        self.rollback_impl: asyncio.Event = asyncio.Event()
        # HITL-2 per-skill retry: queue of RetrySkillRequest
        self.retry_skill: asyncio.Queue[RetrySkillRequest] = asyncio.Queue()

        # HITL-4: memory/persistence review (whole-tree, before UI design)
        self.approve_memory: asyncio.Event = asyncio.Event()
        self.rollback_memory: asyncio.Event = asyncio.Event()

        # HITL-3: UI/interaction review (after the whole tree is finalized)
        self.approve_ui: asyncio.Event = asyncio.Event()
        self.rollback_ui: asyncio.Event = asyncio.Event()

        # Updated by the pipeline to tell the UI what it's currently waiting on
        self.current_tree: SkillTree | None = None
        self.status: str = "idle"


async def run_pipeline(
    tree: SkillTree,
    events: PipelineEvents,
    output_dir: Path,
    *,
    env_provider: EnvProvider | None = None,
    skip_debug: bool = False,
    skill_lib_dir: Path | None = None,
) -> Path:
    """Run the full recursive decomposition and compilation pipeline.

    Flow per layer:
      snapshot → decompose → complexity-review → HITL-1 (approve / rollback / redecompose)
      → schema-hydration → prompt-engineer → tool-implementation → HITL-2 (approve / rollback)
      → advance layer → save new skills to skill_lib

    After all layers: compile → verify-repair → debug-loop → done.

    Args:
        tree: The skill tree, initialised with a root node for the requirement.
        events: Shared asyncio events for HITL communication.
        output_dir: Directory where the compiled ADK project will be written.
        env_provider: Async callback for collecting missing env vars during the debug phase.
                      Signature: async (missing_names: list[str]) -> dict[str, str].
                      If None, missing vars are logged but not fatal.
        skip_debug: Set True to bypass the end-to-end debug loop (e.g. dry-run mode).
        skill_lib_dir: Path to the skill_lib directory. Defaults to <output_dir>/../skill_lib.

    Returns:
        Path to the generated project directory.
    """
    if skill_lib_dir is None:
        skill_lib_dir = output_dir.parent / "skill_lib"
    skill_lib = SkillLib(skill_lib_dir)

    decomposer = DecomposerAgent(skill_lib=skill_lib)
    complexity_reviewer = ComplexityReviewAgent()
    contract_linter = ContractLinterAgent()
    schema_architect = SchemaArchitectAgent()
    prompt_engineer = PromptEngineerAgent()
    tool_implementor = ToolImplementorAgent()
    memory_architect = MemoryArchitectAgent()
    ui_designer = UIDesignerAgent()
    compiler = CompilerAgent()

    raw_path = output_dir / tree.project_name / "blueprint_raw.json"
    verified_path = output_dir / tree.project_name / "blueprint_verified.json"

    while True:
        layer_nodes = tree.get_layer_nodes()
        non_atomic = [n for n in layer_nodes if n.node_type != NodeType.ATOMIC]

        if not non_atomic:
            logger.info("All nodes at layer %d are atomic. Advancing.", tree.current_layer)
            next_layer_nodes = tree.get_layer_nodes(tree.current_layer + 1)
            if not next_layer_nodes:
                logger.info("Tree fully decomposed. Proceeding to compilation.")
                break
            tree.current_layer += 1
            continue

        # ── PRE-DECOMPOSE SNAPSHOT ─────────────────────────────────────────
        logger.info("Snapshotting layer %d before decomposition.", tree.current_layer)
        tree.snapshot_current_layer()

        # ── STEP 1: DECOMPOSE ──────────────────────────────────────────────
        events.status = f"decomposing_layer_{tree.current_layer}"
        logger.info("Decomposing %d nodes at layer %d...", len(non_atomic), tree.current_layer)
        await decomposer.decompose(non_atomic)

        # ── STEP 1b: COMPLEXITY REVIEW ─────────────────────────────────────
        events.status = f"complexity_review_layer_{tree.current_layer}"
        layer_nodes = tree.get_layer_nodes()
        await complexity_reviewer.review(layer_nodes)

        # ── STEP 1c: CONTRACT LINT ─────────────────────────────────────────
        # Deterministic data-flow check on the just-decomposed composites. Runs before
        # HITL-1 so wiring violations surface next to complexity warnings, not at runtime.
        events.status = f"contract_lint_layer_{tree.current_layer}"
        contract_linter.lint_layer(layer_nodes)

        tree.save_json(raw_path)
        events.current_tree = tree

        # ── STEP 2: HITL-1 (structure review) ─────────────────────────────
        events.status = f"awaiting_review_layer_{tree.current_layer}"
        events.approve.clear()
        events.rollback.clear()
        logger.info("Layer %d ready for human review at http://127.0.0.1:8000", tree.current_layer)

        approved = await _hitl1_loop(tree, events, decomposer, contract_linter, raw_path)
        if not approved:
            # Rollback requested
            tree.rollback()
            tree.save_json(raw_path)
            if verified_path.exists():
                verified_path.unlink()
            events.current_tree = tree
            logger.info("Rollback complete. Retrying layer %d.", tree.current_layer)
            continue

        # Freeze the contracts of every node in the approved layer group: the composites
        # just reviewed and their children. Approved siblings now depend on this wiring, so
        # a later redecompose of a frozen node requires explicit force_renegotiate.
        _freeze_layer_contracts(layer_nodes)
        tree.save_json(raw_path)

        # Do NOT advance the layer here. The Decomposer classifies the INPUT
        # nodes in place: a node that turned ATOMIC stays at its current depth,
        # and only a COMPOSITE node spawns UNKNOWN children one level deeper.
        # So the atomics that now need schema/impl live at `current_layer`,
        # not current_layer + 1. We advance only after this layer is fully
        # implemented and reviewed (see "ADVANCE TO NEXT LAYER" at loop end).
        events.current_tree = tree

        # ── STEP 3: SCHEMA HYDRATION ───────────────────────────────────────
        events.status = f"schema_hydration_layer_{tree.current_layer}"
        layer_atomics = [n for n in tree.get_layer_nodes() if n.node_type == NodeType.ATOMIC]
        unhydrated = [n for n in layer_atomics if n.input_schema is None]
        if unhydrated:
            logger.info("Running Schema Architect on %d unhydrated atomic nodes...", len(unhydrated))
            await schema_architect.hydrate(unhydrated)

        # ── STEP 3b: PROMPT ENGINEERING ────────────────────────────────────
        events.status = f"prompt_engineering_layer_{tree.current_layer}"
        llm_atomics = [
            n for n in layer_atomics
            if n.exec_type == ExecType.LLM_PROMPT and n.instruction is None
        ]
        if llm_atomics:
            logger.info("Running Prompt Engineer on %d LLM atomic nodes...", len(llm_atomics))
            persistent_hint = _build_persistent_keys_hint(tree, llm_atomics)
            await prompt_engineer.engineer(llm_atomics, persistent_keys_hint=persistent_hint)

        # ── STEP 3c: TOOL IMPLEMENTATION ───────────────────────────────────
        events.status = f"tool_implementation_layer_{tree.current_layer}"
        tool_atomics = [
            n for n in layer_atomics
            if n.exec_type in (ExecType.DETERMINISTIC_CODE, ExecType.EXTERNAL_API, ExecType.OPENSOURCE_LIBRARY)
            and n.implementation is None
        ]
        if tool_atomics:
            logger.info("Running Tool Implementor on %d tool nodes...", len(tool_atomics))
            await tool_implementor.implement(tool_atomics)

        tree.save_json(verified_path)
        events.current_tree = tree

        # ── STEP 4: HITL-2 (implementation review) ────────────────────────
        all_atomics = layer_atomics
        if not all_atomics:
            logger.info("No atomic nodes at layer %d — auto-approving implementation review.", tree.current_layer)
            impl_approved = True
        else:
            events.status = f"awaiting_impl_review_layer_{tree.current_layer}"
            events.approve_impl.clear()
            events.rollback_impl.clear()
            logger.info("Implementation ready for review at http://127.0.0.1:8000")

            impl_approved = await _hitl2(events, tree, prompt_engineer, tool_implementor, verified_path)
        if not impl_approved:
            # Roll back to the snapshot taken at the start of THIS layer and retry.
            # current_layer was not advanced, so the snapshot for it still exists.
            logger.info("Implementation rollback requested for layer %d.", tree.current_layer)
            events.status = "rolling_back"
            tree.rollback()
            tree.save_json(raw_path)
            if verified_path.exists():
                verified_path.unlink()
            events.current_tree = tree
            logger.info("Implementation rollback complete. Retrying layer %d.", tree.current_layer)
            continue

        # ── SAVE NEW SKILLS TO SKILL LIB ──────────────────────────────────
        _save_new_skills_to_lib(tree, skill_lib)

        # ── SAVE PER-LAYER SNAPSHOT ────────────────────────────────────────
        layer_dir = output_dir / tree.project_name / "layers" / f"layer_{tree.current_layer}"
        tree.save_json(layer_dir / "blueprint_verified.json")
        logger.info("Layer %d artifacts saved to %s", tree.current_layer, layer_dir)

        # ── ADVANCE TO NEXT LAYER ──────────────────────────────────────────
        # This layer's atomics are now hydrated/implemented and approved. Any
        # COMPOSITE nodes at this layer spawned UNKNOWN children one level deeper,
        # which the next iteration will pick up and classify.
        tree.current_layer += 1
        events.current_tree = tree

    # ── MEMORY DESIGN (whole-tree, once) ─────────────────────────────────────
    # Runs after the per-layer loop and BEFORE UI design (spec §4): if any state
    # key was flagged PERSISTENT, design storage for it so the UI Designer can then
    # offer renderers/affordances bound to a known entity. Triage-empty → no-op.
    await _design_memory(tree, events, memory_architect, verified_path)

    # ── UI DESIGN (whole-tree, once) ─────────────────────────────────────────
    # Needs the complete tree structure, so it runs after the per-layer loop and
    # before compile. Selects the frontend/interaction contract and marks which
    # agents are user-facing and which nodes emit media artifacts.
    await _design_ui(tree, events, ui_designer, tool_implementor, verified_path)

    # ── COMPILE ────────────────────────────────────────────────────────────
    events.status = "compiling"
    logger.info("Compiling Google ADK project...")
    project_dir = compiler.compile(tree, output_dir, skill_lib=skill_lib)

    # ── VALIDATE MEMORY WIRING (deterministic, no LLM) ──────────────────────
    # Catch broken bindings BEFORE spending any LLM call in the debug loop.
    if tree.memory_spec is not None:
        problems = _validate_memory_wiring(tree, project_dir)
        if problems:
            events.status = f"memory_wiring_invalid: {problems[0][:100]}"
            logger.error(
                "Memory wiring validation failed (%d problem(s)):\n  - %s",
                len(problems), "\n  - ".join(problems),
            )
            return project_dir

    # ── VERIFY AND REPAIR ──────────────────────────────────────────────────
    events.status = "verifying"
    logger.info("Running import verification on generated project...")
    await _verify_and_repair(project_dir, tool_implementor, events)

    # ── DEBUG LOOP ─────────────────────────────────────────────────────────
    if not skip_debug:
        events.status = "debugging"
        logger.info("Starting end-to-end debug loop for %s …", project_dir.name)
        debug_agent = DebugAgent()
        debug_result = await debug_agent.run(project_dir, tree, env_provider=env_provider)
        if debug_result.success:
            logger.info(
                "[debug] Project validated successfully in %d iteration(s).",
                debug_result.iterations,
            )
            events.status = "done"
        else:
            logger.warning(
                "[debug] Debug loop exhausted (%d iterations) without passing tests.",
                debug_result.iterations,
            )
            last_err = debug_result.errors[-1] if debug_result.errors else "unknown error"
            events.status = f"debug_failed: {last_err[:120]}"
    else:
        events.status = "done"

    logger.info("Pipeline complete → %s", project_dir)
    return project_dir


# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------

def _freeze_layer_contracts(nodes: list[SkillNode]) -> None:
    """Freeze the declared contracts of the approved layer's composites and their children."""
    for node in nodes:
        if node.contract is not None:
            node.contract.frozen = True
        for child in node.children:
            if child.contract is not None:
                child.contract.frozen = True


async def _hitl1_loop(
    tree: SkillTree,
    events: PipelineEvents,
    decomposer: DecomposerAgent,
    contract_linter: ContractLinterAgent,
    raw_path: Path,
) -> bool:
    """HITL-1: structure review loop.

    Handles three outcomes:
    - approve → return True
    - rollback → return False
    - redecompose(node_id, hint) → re-decompose the target node in-place, re-lint, save, loop

    A node with a FROZEN contract is refused unless the request carries force_renegotiate;
    when forced, the node's sibling group is unfrozen and re-linted before re-presenting.

    Returns True if approved, False if full-layer rollback was requested.
    """
    while True:
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(_wait_event(events.approve)),
                asyncio.create_task(_wait_event(events.rollback)),
                asyncio.create_task(events.redecompose.get()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

        if events.rollback.is_set():
            events.status = "rolling_back"
            return False

        if events.approve.is_set():
            return True

        # Redecompose request arrived
        req: RedecomposeRequest = next(t.result() for t in done if not t.cancelled())
        node = tree.root.find_node_by_id(req.node_id)
        if node is None:
            logger.warning("Redecompose: node %s not found; ignoring.", req.node_id)
            # Reset events so the UI loop continues
            events.approve.clear()
            events.rollback.clear()
            events.status = f"awaiting_review_layer_{tree.current_layer}"
            continue

        # Freeze guard: refuse to redecompose a node whose contract approved siblings depend
        # on, unless the human explicitly forced a renegotiation.
        if node.contract is not None and node.contract.frozen and not req.force_renegotiate:
            logger.info("Redecompose of frozen node '%s' blocked (no force_renegotiate).", node.name)
            node.contract_note = (
                "FROZEN: redecomposing this will renegotiate the contract your approved "
                "siblings depend on. Re-request with force_renegotiate to proceed."
            )
            tree.save_json(raw_path)
            events.current_tree = tree
            events.approve.clear()
            events.rollback.clear()
            events.status = f"awaiting_review_layer_{tree.current_layer}"
            continue

        parent = tree.root.find_parent_of(node.id)

        # Forced renegotiation: unfreeze the node's whole sibling group before redecomposing.
        if req.force_renegotiate:
            group = ([parent] + parent.children) if parent is not None else [node]
            for n in group:
                if n.contract is not None:
                    n.contract.frozen = False

        # Apply the description override if provided
        if req.new_description:
            node.description = req.new_description
        # Clear the node's existing decomposition
        node.children.clear()
        node.node_type = NodeType.UNKNOWN
        node.exec_type = None
        node.composition_type = None
        node.contract_note = None
        node.routing = None  # re-decompose may change the node's shape; drop stale routing

        logger.info("Re-decomposing node '%s' (hint: %s)", node.name, req.hint or "none")
        events.status = f"redecomposing_{node.name}"
        await decomposer.decompose([node], hint=req.hint)

        # Scoped re-lint: only the affected group(s). If the node itself became composite,
        # lint it; also re-lint its parent group since the node's contract may have changed.
        if node.node_type == NodeType.COMPOSITE:
            contract_linter.lint_group(node)
        if parent is not None:
            contract_linter.lint_group(parent)

        tree.save_json(raw_path)
        events.current_tree = tree
        events.approve.clear()
        events.rollback.clear()
        events.status = f"awaiting_review_layer_{tree.current_layer}"


def _check_contract_drift(node: SkillNode) -> None:
    """Warn if a re-implemented atomic's schema no longer satisfies its frozen contract.

    Compares the node's frozen declared contract (state keys) against the view derived
    from its input/output schemas. Sets node.contract_note on drift; clears it otherwise.
    Only meaningful for atomics with a frozen contract and hydrated schemas.
    """
    if node.contract is None or not node.contract.frozen:
        return
    derived = node.schema_contract()
    if derived is None:
        return  # not hydrated (e.g. LLM node before schemas) — nothing to compare
    dropped_writes = set(node.contract.writes.keys()) - set(derived.writes.keys())
    extra_reads = set(derived.reads.keys()) - set(node.contract.reads.keys())
    problems: list[str] = []
    if dropped_writes:
        problems.append(f"no longer produces {sorted(dropped_writes)}")
    if extra_reads:
        problems.append(f"now requires unexpected input(s) {sorted(extra_reads)}")
    if problems:
        node.contract_note = "DRIFT: regenerated signature " + "; ".join(problems)
        logger.info("Contract drift on '%s': %s", node.name, node.contract_note)
    else:
        node.contract_note = None


async def _hitl2(
    events: PipelineEvents,
    tree,
    prompt_engineer,
    tool_implementor,
    verified_path,
) -> bool:
    """HITL-2: implementation review loop.

    Handles three outcomes:
    - approve → return True
    - rollback_impl → return False
    - retry_skill(node_id, feedback) → re-implement that single skill, save, loop

    Returns True if approved, False if full-layer rollback was requested.
    """
    from src.orchestrator.state import ExecType, NodeType
    while True:
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(_wait_event(events.approve_impl)),
                asyncio.create_task(_wait_event(events.rollback_impl)),
                asyncio.create_task(events.retry_skill.get()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

        if events.rollback_impl.is_set():
            return False

        if events.approve_impl.is_set():
            return True

        # Per-skill retry request arrived
        req: RetrySkillRequest = next(t.result() for t in done if not t.cancelled())
        node = tree.root.find_node_by_id(req.node_id)
        if node is None:
            logger.warning("RetrySkill: node %s not found; ignoring.", req.node_id)
            events.approve_impl.clear()
            events.rollback_impl.clear()
            events.status = f"awaiting_impl_review_layer_{tree.current_layer}"
            continue

        logger.info("Re-implementing skill '%s' (feedback: %s)", node.name, req.feedback or "none")
        events.status = f"retrying_skill_{node.name}"

        # Clear existing implementation/instruction so it gets regenerated
        node.implementation = None
        node.instruction = None

        if node.exec_type == ExecType.LLM_PROMPT:
            await prompt_engineer.engineer([node], feedback=req.feedback)
        else:
            await tool_implementor.implement([node], feedback=req.feedback)

        # Drift check: the frozen declared contract must still match the regenerated
        # signature. This is the smaller signature-vs-frozen-contract check (NOT the
        # tree-structure lint), surfaced as a warning; it never blocks or rewrites.
        _check_contract_drift(node)

        tree.save_json(verified_path)
        events.current_tree = tree
        events.approve_impl.clear()
        events.rollback_impl.clear()
        events.status = f"awaiting_impl_review_layer_{tree.current_layer}"


async def _hitl_ui(events: PipelineEvents) -> bool:
    """HITL-3: UI/interaction review. Returns True if approved, False if re-design requested."""
    done, pending = await asyncio.wait(
        [
            asyncio.create_task(_wait_event(events.approve_ui)),
            asyncio.create_task(_wait_event(events.rollback_ui)),
        ],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()
    return events.approve_ui.is_set()


async def _hitl_memory(events: PipelineEvents) -> bool:
    """HITL-4: memory/persistence review. Returns True if approved, False if regenerate requested."""
    done, pending = await asyncio.wait(
        [
            asyncio.create_task(_wait_event(events.approve_memory)),
            asyncio.create_task(_wait_event(events.rollback_memory)),
        ],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()
    return events.approve_memory.is_set()


# ---------------------------------------------------------------------------
# Memory design (HITL-4)
# ---------------------------------------------------------------------------

async def _design_memory(
    tree: SkillTree,
    events: PipelineEvents,
    memory_architect: MemoryArchitectAgent,
    verified_path: Path,
) -> None:
    """Design the persistent-memory contract (declarative bindings), then HITL-4.

    Mirrors _design_ui. Triage-empty (tree.memory_spec is None after design) skips HITL
    entirely — no dashboard pause for a product that needs no persistence. A HITL-4
    rollback re-runs the Memory Architect (cheap, single call).

    Tool bodies are NOT re-implemented for memory: persistence is wired by the compiler as
    deterministic before/after_agent callbacks (memory/_bindings.py), never as adapter
    calls inside a tool body. Set env SKIP_MEMORY_REVIEW=1 to auto-approve.
    """
    skip_review = bool(os.environ.get("SKIP_MEMORY_REVIEW"))

    while True:
        events.status = "memory_design"
        logger.info("Designing persistent-memory contract for %s …", tree.project_name)
        await memory_architect.design(tree)

        # Triage found nothing → no persistence required. Skip HITL-4 (spec §3/§4).
        if tree.memory_spec is None:
            logger.info("No persistent memory required — skipping HITL-4.")
            tree.save_json(verified_path)
            events.current_tree = tree
            return

        tree.save_json(verified_path)
        events.current_tree = tree

        if skip_review:
            logger.info("SKIP_MEMORY_REVIEW set — auto-approving memory design.")
            return

        events.status = "awaiting_memory_review"
        events.approve_memory.clear()
        events.rollback_memory.clear()
        logger.info("Memory design ready for review at http://127.0.0.1:8000")

        approved = await _hitl_memory(events)
        if approved:
            return
        logger.info("Memory design rollback requested — regenerating.")


def _validate_memory_wiring(tree: SkillTree, project_dir: Path) -> list[str]:
    """Deterministically check every memory binding resolves. Returns problem strings.

    Runs BEFORE the LLM debug loop so a broken binding (dangling node id, missing adapter
    module, empty load+save) fails fast with a clear message instead of burning LLM calls
    on an unfixable wiring bug. No LLM, no subprocess.
    """
    problems: list[str] = []
    spec = tree.memory_spec
    if spec is None:
        return problems

    nodes = tree.root.topological_order()
    by_id = {n.id: n for n in nodes}
    mem_dir = project_dir / "memory"

    def _llm_writers(key: str) -> list[SkillNode]:
        out = []
        for n in nodes:
            if n.exec_type != ExecType.LLM_PROMPT:
                continue
            writes = set(n.state_writes or []) | (set(n.contract.writes) if n.contract else set())
            if key in writes:
                out.append(n)
        return out

    for entity in spec.entities:
        table = _snake_name(entity.name)
        if not (mem_dir / f"{table}.py").exists():
            problems.append(f"entity '{entity.name}': adapter module memory/{table}.py not emitted")
        if not entity.bindings:
            problems.append(f"entity '{entity.name}': no bindings (nothing loads or saves it)")
        if not entity.fields:
            problems.append(f"entity '{entity.name}': no derived fields")
        for b in entity.bindings:
            if b.node_id not in by_id:
                problems.append(
                    f"entity '{entity.name}': binding references unknown node id {b.node_id[:8]}…"
                )
            if not b.save_source_key and not b.load_target_key:
                problems.append(f"entity '{entity.name}': a binding has neither save nor load key")
            # A save_source_key must be produced by an LLM node, else it can never reach
            # state (tool returns aren't captured) — this is exactly the runner_coach bug.
            if b.save_source_key and not _llm_writers(b.save_source_key):
                problems.append(
                    f"entity '{entity.name}': save_source_key '{b.save_source_key}' is written by "
                    "no LLM node — it will never be captured to state (make the binding load-only)."
                )
            if b.key_field and b.key_field not in {f.name for f in entity.fields}:
                problems.append(
                    f"entity '{entity.name}': key_field '{b.key_field}' is not an entity field"
                )

    if not (mem_dir / "_bindings.py").exists():
        problems.append("memory/_bindings.py not emitted")

    return problems


# ---------------------------------------------------------------------------
# UI design (HITL-3)
# ---------------------------------------------------------------------------

async def _design_ui(
    tree: SkillTree,
    events: PipelineEvents,
    ui_designer: UIDesignerAgent,
    tool_implementor: ToolImplementorAgent,
    verified_path: Path,
) -> None:
    """Select the frontend/interaction contract, re-implement media nodes, HITL-3.

    Loops so that a HITL-3 rollback re-runs the UI Designer (cheap, single call).
    Set env SKIP_UI_REVIEW=1 to auto-approve (headless / fast runs).
    """
    skip_review = bool(os.environ.get("SKIP_UI_REVIEW"))

    while True:
        events.status = "ui_design"
        logger.info("Designing frontend/interaction contract for %s …", tree.project_name)
        await ui_designer.design(tree)

        # A coordinator root is an orchestrator: it calls capabilities as AgentTools and authors
        # the single user-facing reply itself (sub-agents run inside the tool call and do NOT
        # speak to the user). So the root MUST be user-facing, and it is normally the ONLY
        # user-facing agent. Guarantee it here; otherwise the frontend author filter would hide
        # the product's own answers ("(No response)"). Guard only; visibility model unchanged.
        from src.orchestrator.state import CompositionType, NodeVisibility
        if (
            tree.root.composition_type == CompositionType.LLM_COORDINATOR
            and tree.root.visibility != NodeVisibility.USER_FACING
        ):
            tree.root.visibility = NodeVisibility.USER_FACING
            if tree.ui_spec is not None and tree.root.name not in tree.ui_spec.user_facing_nodes:
                tree.ui_spec.user_facing_nodes.append(tree.root.name)
            logger.info("Forced coordinator root '%s' user-facing.", tree.root.name)

        # Media re-implementation: nodes the designer flagged as media producers need
        # their tool bodies regenerated so they call save_artifact (see tool_implementor
        # media mode). Reuses the null-and-reimplement pattern from _repair_from_traceback.
        media_nodes = [
            n for n in tree.root.topological_order()
            if n.node_type == NodeType.ATOMIC
            and n.output_media_types
            and n.exec_type in (
                ExecType.DETERMINISTIC_CODE, ExecType.EXTERNAL_API, ExecType.OPENSOURCE_LIBRARY
            )
            and not _implementation_saves_artifact(n.implementation)
        ]
        if media_nodes:
            logger.info(
                "Re-implementing %d media-producing node(s) with artifact emission…",
                len(media_nodes),
            )
            events.status = "media_implementation"
            for node in media_nodes:
                node.implementation = None
            await tool_implementor.implement(media_nodes)

        tree.save_json(verified_path)
        events.current_tree = tree

        if skip_review:
            logger.info("SKIP_UI_REVIEW set — auto-approving UI design.")
            return

        events.status = "awaiting_ui_review"
        events.approve_ui.clear()
        events.rollback_ui.clear()
        logger.info("UI design ready for review at http://127.0.0.1:8000")

        approved = await _hitl_ui(events)
        if approved:
            return
        logger.info("UI design rollback requested — re-designing.")


def _implementation_saves_artifact(impl: Optional[str]) -> bool:
    """True if a tool body already emits an artifact (avoids double re-implementation)."""
    return bool(impl) and "save_artifact" in impl


# ---------------------------------------------------------------------------
# Post-compile verify-and-repair
# ---------------------------------------------------------------------------

async def _verify_and_repair(
    project_dir: Path,
    tool_implementor: ToolImplementorAgent,
    events: PipelineEvents,
) -> None:
    """Attempt to import the generated project; repair tool bodies on failure.

    Runs up to _MAX_REPAIR_ITERATIONS times. If still broken, surfaces the error
    in events.status so the UI can show it rather than silently returning a broken project.
    """
    # Try to load the tree to be able to repair specific nodes
    tree: SkillTree | None = None
    blueprint_candidate = project_dir.parent / project_dir.name / "blueprint_verified.json"
    if blueprint_candidate.exists():
        try:
            tree = SkillTree.load_json(blueprint_candidate)
        except Exception:
            tree = None

    # Ensure project deps are installed before verifying
    subprocess.run(["uv", "sync"], cwd=str(project_dir), capture_output=True)

    for iteration in range(1, _MAX_REPAIR_ITERATIONS + 1):
        # Run inside the project venv via `uv run` so google-adk and other
        # project deps are importable.  sys.executable (the host Python) does
        # not have those packages installed.
        result = subprocess.run(
            ["uv", "run", "python", "-c",
             "import sys; sys.path.insert(0, '.'); import run"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Import verification passed on iteration %d.", iteration)
            return

        stderr = result.stderr
        logger.warning("Import check failed (iteration %d):\n%s", iteration, stderr)

        if tree is None:
            logger.warning("No blueprint available for repair — skipping.")
            break

        repaired = await _repair_from_traceback(stderr, project_dir, tree, tool_implementor)
        if not repaired:
            logger.warning("No repairable node identified from traceback.")
            break

    # Surface unresolved errors in status so the UI can show them
    events.status = f"verify_failed: {_first_error_line(result.stderr)}"
    logger.error("Verification failed after %d iterations. Last error:\n%s",
                 _MAX_REPAIR_ITERATIONS, result.stderr)


async def _repair_from_traceback(
    stderr: str,
    project_dir: Path,
    tree: SkillTree,
    tool_implementor: ToolImplementorAgent,
) -> bool:
    """Parse the traceback to find the offending node and re-implement it.

    Returns True if a repair was attempted, False if no node could be identified.
    """
    import re

    # Memory adapters are template-generated (not SkillNodes), so a failure in
    # memory/<x>.py cannot be fixed by re-implementing a node. Regenerate the whole
    # memory package from the current (possibly human-edited) memory_spec instead.
    if re.search(r'(?:File ".*)?memory/([^"]+)\.py', stderr) and tree.memory_spec is not None:
        logger.info("Traceback points at a memory adapter — regenerating memory package.")
        compiler = CompilerAgent()
        compiler._compile_memory(tree, project_dir)  # type: ignore[attr-defined]
        return True

    # Look for a File "...atomics/<name>.py" reference in the traceback
    match = re.search(r'File ".*atomics/([^"]+)\.py"', stderr)
    if not match:
        return False

    module_name = match.group(1)
    node = next(
        (n for n in tree.root.topological_order()
         if n.name == module_name
         and n.node_type == NodeType.ATOMIC
         and n.exec_type in (ExecType.DETERMINISTIC_CODE, ExecType.EXTERNAL_API)),
        None,
    )
    if node is None:
        return False

    logger.info("Repairing node '%s' based on traceback.", node.name)
    node.implementation = None
    await tool_implementor.implement([node])

    atomics_dir = project_dir / "atomics"
    compiler = CompilerAgent()
    compiler._compile_atomic(node, atomics_dir)  # type: ignore[attr-defined]

    return True


def _first_error_line(stderr: str) -> str:
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("Traceback") and not stripped.startswith("File "):
            return stripped[:120]
    return "unknown error"


async def _wait_event(event: asyncio.Event) -> None:
    await event.wait()


def _build_persistent_keys_hint(tree: SkillTree, nodes: list[SkillNode]) -> dict[str, set[str]]:
    """Return {node_id: set_of_persistent_keys} for each node in `nodes`.

    For each node, unions the PERSISTENT keys declared in the contracts of all its ancestor
    composites. These are the keys the PromptEngineer MUST mark PERSISTENT in state_scopes
    so the MemoryArchitect can later find LLM producers and wire save bindings.
    """
    hint: dict[str, set[str]] = {}
    for node in nodes:
        keys: set[str] = set()
        # Walk up via parent_id links to collect ancestor contracts' PERSISTENT scopes.
        current_id = node.parent_id
        while current_id is not None:
            ancestor = tree.root.find_node_by_id(current_id)
            if ancestor is None:
                break
            if ancestor.contract:
                for key, scope in ancestor.contract.scopes.items():
                    if scope == "PERSISTENT":
                        keys.add(key)
            current_id = ancestor.parent_id
        if keys:
            hint[node.id] = keys
    return hint


# ---------------------------------------------------------------------------
# Skill library helpers
# ---------------------------------------------------------------------------

def _save_new_skills_to_lib(tree: SkillTree, skill_lib: SkillLib) -> None:
    """Persist fully-hydrated atomic nodes to the skill_lib if not already there.

    Nodes that were sourced from the skill_lib (skill_lib_ref is set) are skipped
    because they already exist. Only nodes that were newly implemented are saved.
    """
    saved = 0
    for node in tree.root.topological_order():
        if node.node_type != NodeType.ATOMIC:
            continue
        if node.skill_lib_ref:
            continue  # already in the library
        if node.memory_entity_ref is not None:
            continue  # impl coupled to this project's memory schema; not portable (spec §5)
        if not node.input_schema or not node.output_schema:
            continue  # not fully hydrated yet

        # Only save if we have an implementation or instruction
        has_content = bool(node.implementation or node.instruction)
        if not has_content:
            continue

        # Skip if an entry with the same name already exists
        if skill_lib.get(node.name):
            continue

        exec_type_str = node.exec_type.value if node.exec_type else "LLM_PROMPT"
        entry = SkillEntry(
            name=node.name,
            description=node.description,
            exec_type=exec_type_str,
            implementation=node.implementation or "",
            instruction=node.instruction or "",
            input_schema=node.input_schema,
            output_schema=node.output_schema,
        )
        skill_lib.save(entry)
        node.skill_lib_ref = node.name  # mark it so we don't re-save next layer
        saved += 1

    if saved:
        logger.info("Saved %d new skill(s) to skill_lib.", saved)
