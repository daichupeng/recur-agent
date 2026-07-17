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
    OPENSOURCE_LIBRARY = "OPENSOURCE_LIBRARY"
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


class MemoryScope(str, Enum):
    """Durability of a session-state key across separate product invocations.

    EPHEMERAL keys live only for one run (ADK session state). PERSISTENT keys must
    survive across runs — they are what the MemoryArchitectAgent designs storage for.
    """
    EPHEMERAL = "EPHEMERAL"    # default — lives for one run only
    PERSISTENT = "PERSISTENT"  # must be persisted across separate invocations


class Contract(BaseModel):
    """Data-flow contract for a node: the session-state keys it consumes and produces.

    Populated by the DecomposerAgent at classification time — for a composite AND a
    proposed contract for each of its children — so the ContractLinter can validate the
    wiring BEFORE schemas exist (i.e. before HITL-1). Frozen once the parent's layer is
    approved at HITL-1; a frozen contract may only be renegotiated via an explicit
    force_renegotiate redecompose.

    Distinct from SkillNode.schema_contract(), which is a read-only view derived on the
    fly from an atomic's input_schema/output_schema and used only for the HITL-2 drift
    check.
    """
    reads: dict[str, str] = Field(default_factory=dict)   # state_key -> type/description
    writes: dict[str, str] = Field(default_factory=dict)  # state_key -> type/description
    scopes: dict[str, str] = Field(default_factory=dict)  # state_key -> MemoryScope value; absent ⇒ EPHEMERAL
    frozen: bool = False  # locked once the parent's HITL-1 layer is approved


class RouteRule(BaseModel):
    """One routing rule for an LLM_COORDINATOR node: which child a user intent maps to.

    `trigger` describes the USER INTENT that should route here (distinct from the child's
    own `description`, which says what the child does). `examples` are optional user
    utterances that anchor the route.
    """
    child_name: str
    trigger: str
    examples: list[str] = Field(default_factory=list)


class RoutingSpec(BaseModel):
    """Reviewable routing metadata for an LLM_COORDINATOR node.

    Set by the DecomposerAgent for coordinator nodes, edited by the human at HITL-1, and
    rendered by the compiler into the coordinator's instruction. A node with routing=None
    falls back to today's thin routing (built from each child's description) — back-compat.
    """
    routes: list[RouteRule] = Field(default_factory=list)
    fallback: str = ""       # default child name OR an instruction to ask a clarifying question
    clarify_when: str = ""   # Feature 2: NL condition under which to ask the user instead of routing


class InputAffordance(str, Enum):
    """User input modalities the generated frontend can offer. TEXT is always available."""
    TEXT = "TEXT"                # free-text prompt box
    FILE_UPLOAD = "FILE_UPLOAD"  # arbitrary file → inline_data Blob
    IMAGE_UPLOAD = "IMAGE_UPLOAD"  # image file / paste → image/* Blob


class OutputRenderer(str, Enum):
    """How the generated frontend renders agent outputs."""
    TEXT = "TEXT"                    # plain text bubble (default fallback)
    MARKDOWN = "MARKDOWN"            # rich markdown (headings, tables, lists)
    TABLE = "TABLE"                  # structured tabular data
    IMAGE = "IMAGE"                  # inline image artifact
    FILE_DOWNLOAD = "FILE_DOWNLOAD"  # download link for a file artifact
    CODE = "CODE"                    # syntax-highlighted code block


class NodeVisibility(str, Enum):
    """Whether a node's agent output is shown to the end user in the generated frontend."""
    USER_FACING = "user_facing"  # this agent's output is shown to the user
    INTERNAL = "internal"        # hidden; intermediate/pipeline agent


class SkillNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    node_type: NodeType = NodeType.UNKNOWN
    exec_type: Optional[ExecType] = None
    composition_type: Optional[CompositionType] = None
    contract: Optional[Contract] = None  # Declared data-flow contract; set by DecomposerAgent, linted + frozen at HITL-1
    contract_note: Optional[str] = None  # Set by ContractLinterAgent; wiring violations shown to human before approval
    routing: Optional["RoutingSpec"] = None  # Set by DecomposerAgent for LLM_COORDINATOR nodes; edited at HITL-1, rendered by compiler. None ⇒ thin routing
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
    state_scopes: dict[str, str] = Field(default_factory=dict)  # Set by PromptEngineerAgent; state_key -> MemoryScope value (absent ⇒ EPHEMERAL)
    memory_entity_ref: Optional[str] = None  # Set by MemoryArchitectAgent; NAME of the MemorySpec entity this node reads/writes
    instruction: Optional[str] = None  # Set by PromptEngineerAgent; engineered system prompt for LLM agents
    skill_lib_ref: Optional[str] = None  # Name of the skill_lib entry this node was sourced from
    visibility: NodeVisibility = NodeVisibility.INTERNAL  # Set by UIDesignerAgent; governs frontend display
    output_media_types: list[str] = Field(default_factory=list)  # MIME types this node emits as artifacts; [] = text-only

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

    def find_parent_of(self, node_id: str) -> Optional["SkillNode"]:
        """Depth-first search for the parent of the node with the given id.

        Returns None if node_id is this subtree's root (no parent here) or absent.
        """
        for child in self.children:
            if child.id == node_id:
                return self
            found = child.find_parent_of(node_id)
            if found:
                return found
        return None

    def schema_contract(self) -> Optional["Contract"]:
        """Derive a Contract view from this atomic's input/output schemas.

        Flattens the top-level JSON-Schema property names into reads/writes (values are
        the stringified property type). Returns None when schemas are absent (e.g. a
        composite, or an unhydrated atomic). Used only for the HITL-2 drift check — never
        stored; the declared `contract` field remains the source of truth.
        """
        if self.input_schema is None or self.output_schema is None:
            return None

        def _props(schema: dict[str, Any]) -> dict[str, str]:
            props = schema.get("properties") or {}
            return {
                key: str(spec.get("type", "any")) if isinstance(spec, dict) else "any"
                for key, spec in props.items()
            }

        return Contract(reads=_props(self.input_schema), writes=_props(self.output_schema))


class LayerSnapshot(BaseModel):
    """Immutable snapshot taken BEFORE the Decomposer runs on a layer."""
    layer: int
    root_snapshot: dict[str, Any]  # root.model_dump() deep copy


class UISpec(BaseModel):
    """Frontend/interaction contract for the generated product.

    Selected by UIDesignerAgent from the interaction catalog (see
    src/interaction_catalog.py) after the tree is finalized; overridable in HITL-3.
    A tree with ui_spec=None compiles exactly as before (text-only CLI + adk web).
    """
    title: str
    tagline: str = ""
    inputs: list[InputAffordance] = Field(default_factory=lambda: [InputAffordance.TEXT])
    output_renderers: list[OutputRenderer] = Field(default_factory=lambda: [OutputRenderer.TEXT])
    example_prompts: list[str] = Field(default_factory=list)
    user_facing_nodes: list[str] = Field(default_factory=list)  # node NAMES marked user_facing
    accept_mime_types: list[str] = Field(default_factory=list)  # e.g. ["image/*", "application/pdf"]
    show_agent_trace: bool = True  # collapsed-by-default trace of internal agent steps in the generated UI


class MemoryBackend(str, Enum):
    """Storage backend for a persistent memory entity (fixed catalog; see src/memory_catalog.py)."""
    KEY_VALUE = "KEY_VALUE"    # default — point lookups, profiles, dedup flags, settings
    APPEND_LOG = "APPEND_LOG"  # growing history / audit trail
    SEMANTIC = "SEMANTIC"      # fuzzy / similarity recall (never a default)


class MemoryField(BaseModel):
    """One flat field in a memory entity's schema."""
    name: str
    type: str  # "str" | "int" | "float" | "bool" | "datetime" | "json"


class MemoryBinding(BaseModel):
    """Declarative wiring: when to load/save an entity around one node's lifecycle.

    The compiler turns each binding into deterministic ADK before_agent_callback /
    after_agent_callback functions on the node (`node_id`, a stable uuid). No LLM decides
    when to persist, and no adapter calls are written into tool bodies.

    Persistence flows through ONE session-state key holding a dict (whose keys are the
    entity's fields), not N top-level keys:

    - save_source_key: the state key (a dict, produced during the run and enforced via the
      producer's output_key/output_schema) PERSISTED into the entity after the node runs.
      None = load-only (recall) binding.
    - load_target_key: the state key POPULATED from storage before the node runs (so a
      consumer can read prior state). None = save-only binding.
    - key_field: KEY_VALUE only — which entity field is the primary key. When null, the
      callback uses the ADK user_id (per-user memory).
    """
    node_id: str
    save_source_key: Optional[str] = None
    load_target_key: Optional[str] = None
    key_field: Optional[str] = None


class MemoryEntity(BaseModel):
    """A single persisted entity designed by MemoryArchitectAgent.

    Compiled into memory/<snake(name)>.py by the CompilerAgent using the template
    selected from its backend. Persistence is wired via `bindings` (deterministic
    before/after_agent callbacks); `owner_nodes` is derived from bindings for display.
    """
    name: str
    backend: MemoryBackend = MemoryBackend.KEY_VALUE
    fields: list[MemoryField] = Field(default_factory=list)
    keys_covered: list[str] = Field(default_factory=list)   # PERSISTENT state keys this entity backs
    bindings: list[MemoryBinding] = Field(default_factory=list)  # deterministic load/save wiring per node
    owner_nodes: list[str] = Field(default_factory=list)    # node IDs (uuid); derived from bindings, for display
    retention: Optional[str] = None                          # e.g. "90 days"; null = indefinite
    deletion_scope: str = "entire entity"                    # which fields constitute "a user's data" (§6)
    deletion_confirmed: bool = False                         # HITL-4 forcing-function: human confirmed deletion scope


class MemorySpec(BaseModel):
    """Whole-product persistence contract; None on SkillTree = no persistent memory.

    Selected by MemoryArchitectAgent after the tree is finalized and before the UI
    Designer runs; overridable in HITL-4. A tree with memory_spec=None compiles exactly
    as before (no memory/ package, no init_memory, no adapter injection).
    """
    entities: list[MemoryEntity] = Field(default_factory=list)


class SkillTree(BaseModel):
    project_name: str
    requirement: str
    root: SkillNode
    current_layer: int = 0
    history: list[LayerSnapshot] = Field(default_factory=list)
    required_env_vars: list[str] = Field(default_factory=list)
    ui_spec: Optional[UISpec] = None  # None → back-compat: no generated frontend
    memory_spec: Optional[MemorySpec] = None  # None → back-compat: no persistent memory

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
