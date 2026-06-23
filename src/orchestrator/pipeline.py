"""Main orchestration loop: snapshot → decompose → HITL-1 → schema → prompt-engineer
→ implement → HITL-2 → compile → verify-repair."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.agents.compiler import CompilerAgent
from src.agents.complexity_reviewer import ComplexityReviewAgent
from src.agents.decomposer import DecomposerAgent
from src.agents.prompt_engineer import PromptEngineerAgent
from src.agents.schema_architect import SchemaArchitectAgent
from src.agents.tool_implementor import ToolImplementorAgent
from src.orchestrator.state import ExecType, NodeType, SkillNode, SkillTree

logger = logging.getLogger(__name__)

_MAX_REPAIR_ITERATIONS = 3


@dataclass
class RedecomposeRequest:
    """Carry a per-node re-decompose hint from the UI to the pipeline."""
    node_id: str
    new_description: Optional[str]
    hint: Optional[str]


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

        # Updated by the pipeline to tell the UI what it's currently waiting on
        self.current_tree: SkillTree | None = None
        self.status: str = "idle"


async def run_pipeline(
    tree: SkillTree,
    events: PipelineEvents,
    output_dir: Path,
) -> Path:
    """Run the full recursive decomposition and compilation pipeline.

    Flow per layer:
      snapshot → decompose → complexity-review → HITL-1 (approve / rollback / redecompose)
      → schema-hydration → prompt-engineer → tool-implementation → HITL-2 (approve / rollback)
      → advance layer

    After all layers: compile → verify-repair → done.

    Args:
        tree: The skill tree, initialised with a root node for the requirement.
        events: Shared asyncio events for HITL communication.
        output_dir: Directory where the compiled ADK project will be written.

    Returns:
        Path to the generated project directory.
    """
    decomposer = DecomposerAgent()
    complexity_reviewer = ComplexityReviewAgent()
    schema_architect = SchemaArchitectAgent()
    prompt_engineer = PromptEngineerAgent()
    tool_implementor = ToolImplementorAgent()
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

        tree.save_json(raw_path)
        events.current_tree = tree

        # ── STEP 2: HITL-1 (structure review) ─────────────────────────────
        events.status = f"awaiting_review_layer_{tree.current_layer}"
        events.approve.clear()
        events.rollback.clear()
        logger.info("Layer %d ready for human review at http://127.0.0.1:8000", tree.current_layer)

        approved = await _hitl1_loop(tree, events, decomposer, raw_path)
        if not approved:
            # Rollback requested
            tree.rollback()
            tree.save_json(raw_path)
            if verified_path.exists():
                verified_path.unlink()
            events.current_tree = tree
            logger.info("Rollback complete. Retrying layer %d.", tree.current_layer)
            continue

        # ── STEP 3: SCHEMA HYDRATION ───────────────────────────────────────
        events.status = f"schema_hydration_layer_{tree.current_layer}"
        unhydrated = [
            n for n in tree.root.topological_order()
            if n.node_type == NodeType.ATOMIC and n.input_schema is None
        ]
        if unhydrated:
            logger.info("Running Schema Architect on %d unhydrated atomic nodes...", len(unhydrated))
            await schema_architect.hydrate(unhydrated)

        # ── STEP 3b: PROMPT ENGINEERING ────────────────────────────────────
        events.status = f"prompt_engineering_layer_{tree.current_layer}"
        llm_atomics = [
            n for n in tree.root.topological_order()
            if n.node_type == NodeType.ATOMIC
            and n.exec_type == ExecType.LLM_PROMPT
            and n.instruction is None
        ]
        if llm_atomics:
            logger.info("Running Prompt Engineer on %d LLM atomic nodes...", len(llm_atomics))
            await prompt_engineer.engineer(llm_atomics)

        # ── STEP 3c: TOOL IMPLEMENTATION ───────────────────────────────────
        events.status = f"tool_implementation_layer_{tree.current_layer}"
        tool_atomics = [
            n for n in tree.root.topological_order()
            if n.node_type == NodeType.ATOMIC
            and n.exec_type in (ExecType.DETERMINISTIC_CODE, ExecType.EXTERNAL_API, ExecType.OPENSOURCE_LIBRARY)
            and n.implementation is None
        ]
        if tool_atomics:
            logger.info("Running Tool Implementor on %d tool nodes...", len(tool_atomics))
            await tool_implementor.implement(tool_atomics)

        tree.save_json(verified_path)
        events.current_tree = tree

        # ── STEP 4: HITL-2 (implementation review) ────────────────────────
        all_atomics = [n for n in tree.root.topological_order() if n.node_type == NodeType.ATOMIC]
        if not all_atomics:
            logger.info("No atomic nodes at layer %d — auto-approving implementation review.", tree.current_layer)
            impl_approved = True
        else:
            events.status = f"awaiting_impl_review_layer_{tree.current_layer}"
            events.approve_impl.clear()
            events.rollback_impl.clear()
            logger.info("Implementation ready for review at http://127.0.0.1:8000")

            impl_approved = await _hitl2(events)
        if not impl_approved:
            # Roll back the whole layer and retry from decompose
            logger.info("Implementation rollback requested for layer %d.", tree.current_layer)
            events.status = "rolling_back"
            tree.rollback()
            tree.save_json(raw_path)
            if verified_path.exists():
                verified_path.unlink()
            events.current_tree = tree
            logger.info("Implementation rollback complete. Retrying layer %d.", tree.current_layer)
            continue

        # ── SAVE PER-LAYER SNAPSHOT ────────────────────────────────────────
        layer_dir = output_dir / tree.project_name / "layers" / f"layer_{tree.current_layer}"
        tree.save_json(layer_dir / "blueprint_verified.json")
        logger.info("Layer %d artifacts saved to %s", tree.current_layer, layer_dir)

        tree.current_layer += 1

    # ── COMPILE ────────────────────────────────────────────────────────────
    events.status = "compiling"
    logger.info("Compiling Google ADK project...")
    project_dir = compiler.compile(tree, output_dir)

    # ── VERIFY AND REPAIR ──────────────────────────────────────────────────
    events.status = "verifying"
    logger.info("Running import verification on generated project...")
    await _verify_and_repair(project_dir, tool_implementor, events)

    events.status = "done"
    logger.info("Pipeline complete → %s", project_dir)
    return project_dir


# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------

async def _hitl1_loop(
    tree: SkillTree,
    events: PipelineEvents,
    decomposer: DecomposerAgent,
    raw_path: Path,
) -> bool:
    """HITL-1: structure review loop.

    Handles three outcomes:
    - approve → return True
    - rollback → return False
    - redecompose(node_id, hint) → re-decompose the target node in-place, save, loop

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

        # Apply the description override if provided
        if req.new_description:
            node.description = req.new_description
        # Clear the node's existing decomposition
        node.children.clear()
        node.node_type = NodeType.UNKNOWN
        node.exec_type = None
        node.composition_type = None

        logger.info("Re-decomposing node '%s' (hint: %s)", node.name, req.hint or "none")
        events.status = f"redecomposing_{node.name}"
        await decomposer.decompose([node], hint=req.hint)

        tree.save_json(raw_path)
        events.current_tree = tree
        events.approve.clear()
        events.rollback.clear()
        events.status = f"awaiting_review_layer_{tree.current_layer}"


async def _hitl2(events: PipelineEvents) -> bool:
    """HITL-2: implementation review. Returns True if approved, False if rollback."""
    done, pending = await asyncio.wait(
        [
            asyncio.create_task(_wait_event(events.approve_impl)),
            asyncio.create_task(_wait_event(events.rollback_impl)),
        ],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()
    return events.approve_impl.is_set()


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

    for iteration in range(1, _MAX_REPAIR_ITERATIONS + 1):
        result = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, '{project_dir}'); import run"],
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
