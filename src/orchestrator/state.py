"""Skill tree data structures and rollback state management."""
from __future__ import annotations

import json
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class ExecType(str, Enum):
    DETERMINISTIC_CODE = "DETERMINISTIC_CODE"
    EXTERNAL_API = "EXTERNAL_API"
    LLM_PROMPT = "LLM_PROMPT"


class NodeType(str, Enum):
    UNKNOWN = "unknown"
    COMPOSITE = "composite"
    ATOMIC = "atomic"


class CompositionType(str, Enum):
    SEQUENTIAL = "SEQUENTIAL"        # children run one after another in order
    PARALLEL = "PARALLEL"            # children run concurrently, outputs merged
    LOOP = "LOOP"                     # one child agent runs repeatedly until condition
    LLM_COORDINATOR = "LLM_COORDINATOR"  # LlmAgent decides which child to invoke


class SkillNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    node_type: NodeType = NodeType.UNKNOWN
    exec_type: Optional[ExecType] = None
    composition_type: Optional[CompositionType] = None
    input_schema: Optional[dict[str, Any]] = None
    output_schema: Optional[dict[str, Any]] = None
    children: list["SkillNode"] = Field(default_factory=list)
    parent_id: Optional[str] = None
    depth: int = 0
    approved: bool = False
    review_note: Optional[str] = None  # Set by ComplexityReviewAgent; shown to human before approval
    implementation: Optional[str] = None  # Set by ToolImplementorAgent; Python function body for tool atomics
    state_reads: list[str] = Field(default_factory=list)    # ADK session state keys this node reads
    state_writes: list[str] = Field(default_factory=list)   # ADK session state keys this node writes
    instruction: Optional[str] = None  # Set by PromptEngineerAgent; engineered system prompt for LLM agents

    def get_nodes_at_depth(self, target_depth: int) -> list["SkillNode"]:
        """Return all nodes (including self) at the given depth."""
        if self.depth == target_depth:
            return [self]
        results: list[SkillNode] = []
        for child in self.children:
            results.extend(child.get_nodes_at_depth(target_depth))
        return results

    def find_node_by_id(self, node_id: str) -> Optional["SkillNode"]:
        """Depth-first search for a node by id."""
        if self.id == node_id:
            return self
        for child in self.children:
            found = child.find_node_by_id(node_id)
            if found:
                return found
        return None

    def topological_order(self) -> list["SkillNode"]:
        """Return nodes in depth-descending order (leaves first)."""
        result: list[SkillNode] = []
        for child in self.children:
            result.extend(child.topological_order())
        result.append(self)
        return result


class LayerSnapshot(BaseModel):
    """Immutable snapshot taken BEFORE the Decomposer runs on a layer."""
    layer: int
    root_snapshot: dict[str, Any]  # root.model_dump() deep copy


class SkillTree(BaseModel):
    project_name: str
    requirement: str
    root: SkillNode
    current_layer: int = 0
    history: list[LayerSnapshot] = Field(default_factory=list)

    def snapshot_current_layer(self) -> None:
        """Capture the tree state before decomposing current_layer."""
        self.history.append(
            LayerSnapshot(
                layer=self.current_layer,
                root_snapshot=self.root.model_dump(),
            )
        )

    def rollback(self) -> None:
        """Restore to the snapshot taken before the current layer was processed.

        This purges all phantom children that the Decomposer generated because
        the snapshot was taken before they were created.
        """
        target = next(
            (s for s in reversed(self.history) if s.layer == self.current_layer),
            None,
        )
        if target is None:
            raise ValueError(
                f"No snapshot available for layer {self.current_layer}. "
                "Cannot roll back further."
            )
        self.root = SkillNode.model_validate(target.root_snapshot)
        self.current_layer = target.layer
        # Invalidate this snapshot and anything newer
        self.history = [s for s in self.history if s.layer < target.layer]

    def get_layer_nodes(self, depth: Optional[int] = None) -> list[SkillNode]:
        """Return all nodes at the given depth (defaults to current_layer)."""
        return self.root.get_nodes_at_depth(depth if depth is not None else self.current_layer)

    def has_unresolved_nodes(self) -> bool:
        """True if there are still UNKNOWN or COMPOSITE nodes at current_layer."""
        nodes = self.get_layer_nodes()
        return any(n.node_type != NodeType.ATOMIC for n in nodes)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load_json(cls, path: Path) -> "SkillTree":
        return cls.model_validate_json(path.read_text())
