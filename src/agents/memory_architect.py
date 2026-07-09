"""Memory Architect agent: designs the persistent-memory contract for a finalized tree.

Runs once, after the whole skill tree is decomposed/implemented and BEFORE the UI
Designer (so the UI can offer renderers/affordances bound to a known entity). Works in
two steps:

  Step A — Triage (cheap, no LLM): collect every state key marked PERSISTENT anywhere in
    the tree (from LLM nodes' state_scopes and every node's contract.scopes). If none,
    set tree.memory_spec = None and skip the design step + HITL-4 entirely.

  Step B — Design (one LLM call, only if triage found PERSISTENT keys): choose, from the
    fixed backend catalog, a set of MemoryEntity objects that back those keys.

Produces STRUCTURED DATA only (a MemorySpec + per-node memory_entity_ref); the compiler
turns that into memory/ adapters via Jinja2. Mirrors UIDesignerAgent in shape and model
choice (default Haiku, tool-use output).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base_agent import BaseAgent, _find_text_block
from src.memory_catalog import memory_catalog_prompt_block
from src.orchestrator.state import (
    ExecType,
    MemoryBackend,
    MemoryBinding,
    MemoryEntity,
    MemoryField,
    MemorySpec,
    SkillNode,
    SkillTree,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Memory Architect Agent in a recursive skill-tree generation system.

The product being generated must persist some state ACROSS separate runs/sessions (not just
within one run). You receive the persistent state keys the pipeline flagged, the nodes that
own them, and the product requirement. Your job is to design the storage for that state by
SELECTING a backend from a fixed catalog for each entity. You do not write code — you return
structured choices only.

You decide the WIRING only — you do NOT invent field lists. The storage columns for each
entity are DERIVED automatically from the real input/output schemas of the nodes you wire.

You are given, for each persistent state key, the nodes that touch it, and for each node its
role for that key ("writes" or "reads"), whether it is an LLM node or a tool node, and the
relevant schema (a producer's output_schema, or a consumer's input_schema for that key).

Design a small set of memory ENTITIES. For each entity decide:

1. NAME — a PascalCase entity name (e.g. "AlertHistory", "UserProfile").
2. BACKEND — pick exactly one from the catalog below.
3. KEYS_COVERED — which of the given PERSISTENT state keys this entity backs.
4. BINDINGS — REQUIRED. How this entity is loaded/saved AROUND a node's execution. Each
   binding names ONE node and declares up to two directions:
   - node: the node whose start/end triggers load/save.
   - save_source_key: the state key (whose value is a structured object) to PERSIST into
     this entity AFTER the node runs. This key MUST be one that an **LLM node WRITES**
     (its output_schema becomes the entity's columns, and we enforce that the agent emits
     it). A tool node's return is NOT capturable as state, so never pick a tool-only key
     as a save_source_key. Omit (null) for a load-only recall binding.
   - load_target_key: the state key to POPULATE from storage BEFORE the node runs, so a
     consumer can read prior state. Omit (null) for a save-only binding.
   - key_field (KEY_VALUE only): the entity field to use as the primary key. Omit for
     per-user memory (the product's user id is used automatically).
   Guidance: the COORDINATING / ROOT agent is the usual node to bind — load before the run,
   save after. To persist a produced result AND recall it next run, set save_source_key to
   the producer's output key and load_target_key to the key the consumer reads (they may be
   the same or different keys). If a persistent key is ONLY read (no LLM writes it), make a
   load-only binding (save_source_key null) — it will be recalled when present.
5. RETENTION — a human-readable retention window (e.g. "90 days") or null for indefinite.
6. DELETION_SCOPE — REQUIRED. Describe which fields constitute "a user's data" for this
   entity (used to build the mandatory deletion path). If the whole entity is a single
   user's data, say "entire entity". This is never optional — always fill it.

Group related keys into as few entities as make sense. Prefer ONE entity per coherent
concept. Do not invent entities for keys that were not given to you.

## Backend catalog

{catalog}

Respond with a single JSON object (not an array) matching the provided schema. Use exactly
the node names given.
"""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "backend": {
                        "type": "string",
                        "enum": [b.value for b in MemoryBackend],
                    },
                    "keys_covered": {"type": "array", "items": {"type": "string"}},
                    "bindings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node": {"type": "string"},
                                "save_source_key": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                "load_target_key": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                "key_field": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            },
                            "required": ["node"],
                            "additionalProperties": False,
                        },
                    },
                    "retention": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "deletion_scope": {"type": "string"},
                },
                "required": ["name", "backend", "keys_covered", "bindings", "deletion_scope"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities"],
    "additionalProperties": False,
}


class MemoryArchitectAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            system_prompt=_SYSTEM_PROMPT.format(catalog=memory_catalog_prompt_block())
        )

    async def design(self, tree: SkillTree) -> MemorySpec | None:
        """Triage + design the persistence contract for `tree`, writing it back in-place.

        Sets tree.memory_spec (None if no persistent memory is required) and sets
        memory_entity_ref on every owner node. Returns the MemorySpec or None.
        """
        # ── Step A: Triage ────────────────────────────────────────────────
        persistent = _collect_persistent_keys(tree)
        if not persistent:
            tree.memory_spec = None
            # Clear any stale refs from a previous design pass (e.g. HITL-4 rollback).
            for node in tree.root.topological_order():
                node.memory_entity_ref = None
            logger.info("MemoryArchitect: no persistent memory required.")
            return None

        logger.info(
            "MemoryArchitect: %d persistent key(s) across %d node(s) — designing storage.",
            len(persistent),
            len({nid for keys in persistent.values() for nid in keys}),
        )

        # ── Step B: Design ────────────────────────────────────────────────
        summary = _summarize_persistent(tree, persistent)
        user_content = (
            "Design the persistent memory for this product.\n\n"
            f"Requirement: {tree.requirement}\n\n"
            "Persistent state keys and the nodes that own them:\n"
            + json.dumps(summary, indent=2)
        )

        message = await self._call(
            messages=[{"role": "user", "content": user_content}],
            output_schema=_OUTPUT_SCHEMA,
        )
        result: dict[str, Any] = json.loads(_find_text_block(message).text)

        spec = self._apply_result(tree, result)
        self.log_usage()
        return spec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_result(self, tree: SkillTree, result: dict[str, Any]) -> MemorySpec | None:
        """Validate the LLM result against the real tree; DERIVE fields from real schemas."""
        nodes = tree.root.topological_order()
        name_to_id = {n.name: n.id for n in nodes}
        by_id = {n.id: n for n in nodes}

        entities: list[MemoryEntity] = []
        for raw in result.get("entities", []):
            name = raw.get("name")
            if not name:
                continue
            try:
                backend = MemoryBackend(raw.get("backend", "KEY_VALUE"))
            except ValueError:
                logger.warning("MemoryArchitect: unknown backend %r; defaulting to KEY_VALUE", raw.get("backend"))
                backend = MemoryBackend.KEY_VALUE

            # Build bindings: map node NAME → node ID; validate save/load keys against the
            # node's actual contract (a save key must be WRITTEN by an LLM node to be
            # capturable; else downgrade the binding to load-only).
            bindings: list[MemoryBinding] = []
            for b in raw.get("bindings", []):
                node = by_id.get(name_to_id.get(b.get("node")))
                if node is None:
                    logger.warning("MemoryArchitect: dropping binding for unknown node '%s'", b.get("node"))
                    continue
                save_key = b.get("save_source_key") or None
                load_key = b.get("load_target_key") or None
                # Save is only meaningful if some LLM node actually writes that key.
                if save_key and not _has_capturable_writer(nodes, save_key):
                    logger.info(
                        "MemoryArchitect: save_source_key '%s' has no LLM producer; making load-only.",
                        save_key,
                    )
                    save_key = None
                if load_key and load_key not in _node_reads(node) and load_key not in _node_writes(node):
                    # tolerate: still allow recall into a key the node will read at runtime
                    pass
                if not save_key and not load_key:
                    continue
                bindings.append(
                    MemoryBinding(
                        node_id=node.id,
                        save_source_key=save_key,
                        load_target_key=load_key,
                        key_field=b.get("key_field") or None,
                    )
                )

            if not bindings:
                logger.info("MemoryArchitect: entity '%s' has no usable binding; dropping.", name)
                continue

            # DERIVE fields from real schemas (never invented):
            #  - prefer the output_schema of the LLM node producing a save_source_key,
            #  - else the input_schema shape of a load_target_key's consumer,
            #  - else a single json `value` column.
            fields, key_field = _derive_fields(nodes, bindings, backend)
            if not fields:
                logger.info("MemoryArchitect: entity '%s' derived no fields; dropping.", name)
                continue

            # Fill each binding's key_field default from the derived pk when KEY_VALUE.
            valid_field_names = {f.name for f in fields}
            for bd in bindings:
                if bd.key_field and bd.key_field not in valid_field_names:
                    bd.key_field = None
                if backend == MemoryBackend.KEY_VALUE and bd.key_field is None and key_field:
                    bd.key_field = key_field

            owner_ids = list(dict.fromkeys(bd.node_id for bd in bindings))
            entities.append(
                MemoryEntity(
                    name=name,
                    backend=backend,
                    fields=fields,
                    keys_covered=[str(k) for k in raw.get("keys_covered", [])],
                    bindings=bindings,
                    owner_nodes=owner_ids,
                    retention=raw.get("retention"),
                    deletion_scope=raw.get("deletion_scope") or "entire entity",
                )
            )

        # Drop entities that ended up with no usable binding (nothing to wire).
        entities = [e for e in entities if e.bindings]

        if not entities:
            # Triage said persistent, but the model produced nothing usable. Treat as
            # no-memory rather than emitting an empty (uncompilable) spec.
            logger.warning("MemoryArchitect: design produced no bindable entities; treating as no-memory.")
            tree.memory_spec = None
            for node in tree.root.topological_order():
                node.memory_entity_ref = None
            return None

        spec = MemorySpec(entities=entities)

        # Write back per-node memory_entity_ref (by id) for display/back-compat. Reset all
        # first so a HITL-4 rollback/regenerate does not leave stale refs on dropped nodes.
        id_to_entity: dict[str, str] = {}
        for entity in entities:
            for nid in entity.owner_nodes:
                id_to_entity.setdefault(nid, entity.name)
        for node in tree.root.topological_order():
            node.memory_entity_ref = id_to_entity.get(node.id)

        tree.memory_spec = spec
        logger.info(
            "MemoryArchitect: %d entit(y/ies): %s",
            len(entities),
            ", ".join(f"{e.name}({e.backend.value}, {len(e.bindings)} binding(s))" for e in entities),
        )
        return spec


def _collect_persistent_keys(tree: SkillTree) -> dict[str, list[str]]:
    """Return {state_key: [owner node ids]} for every key marked PERSISTENT anywhere.

    Unions two sources (spec §2): LLM nodes' `state_scopes` and every node's
    `contract.scopes`. A key is persistent if ANY node marks it PERSISTENT.
    """
    keys: dict[str, list[str]] = {}
    for node in tree.root.topological_order():
        node_persistent: set[str] = set()
        for key, scope in (node.state_scopes or {}).items():
            if scope == "PERSISTENT":
                node_persistent.add(key)
        if node.contract is not None:
            for key, scope in (node.contract.scopes or {}).items():
                if scope == "PERSISTENT":
                    node_persistent.add(key)
        for key in node_persistent:
            keys.setdefault(key, [])
            if node.id not in keys[key]:
                keys[key].append(node.id)
    return keys


def _has_capturable_writer(nodes: list[SkillNode], key: str) -> bool:
    """True if some LLM node writes `key` (so output_key can capture it to state)."""
    for n in nodes:
        if n.exec_type == ExecType.LLM_PROMPT and key in _node_writes(n):
            return True
    return False


def _derive_fields(
    nodes: list[SkillNode],
    bindings: list["MemoryBinding"],
    backend: MemoryBackend,
) -> tuple[list[MemoryField], Optional[str]]:
    """Derive entity fields from real node schemas. Returns (fields, key_field or None).

    Priority: the output_schema of the LLM node that produces a binding's save_source_key
    (flattened); else the input_schema shape of a load_target_key's consumer; else a single
    json `value` column. Fields are DERIVED, never invented by the LLM.
    """
    by_id = {n.id: n for n in nodes}

    # 1) Producer output_schema for any save_source_key.
    for b in bindings:
        if not b.save_source_key:
            continue
        producer = next(
            (n for n in nodes if n.exec_type == ExecType.LLM_PROMPT and b.save_source_key in _node_writes(n)),
            None,
        )
        if producer is not None:
            props = _schema_props(producer.output_schema)
            if props:
                fields = [MemoryField(name=k, type=t) for k, t in props.items()]
                return fields, _first_str_field(fields)

    # 2) Consumer input_schema shape for a load_target_key.
    for b in bindings:
        if not b.load_target_key:
            continue
        consumer = by_id.get(b.node_id)
        shape = _input_shape_for_key(consumer.input_schema if consumer else None, b.load_target_key)
        if isinstance(shape, dict) and shape.get("kind") == "object" and shape.get("properties"):
            fields = [MemoryField(name=k, type=t) for k, t in shape["properties"].items()]
            return fields, _first_str_field(fields)
        if isinstance(shape, dict):
            # scalar / list / unknown → single value column typed by the shape
            kind = shape.get("kind")
            t = "json" if kind in ("array", "object", "unknown", None) else _json_to_memory_type({"type": kind})
            return [MemoryField(name="value", type=t)], None

    # 3) Fallback: opaque single json column.
    return [MemoryField(name="value", type="json")], None


def _first_str_field(fields: list[MemoryField]) -> Optional[str]:
    return next((f.name for f in fields if f.type in ("str", "int")), fields[0].name if fields else None)


def _node_reads(node: SkillNode) -> set[str]:
    reads = set(node.state_reads or [])
    if node.contract is not None:
        reads |= set(node.contract.reads.keys())
    return reads


def _node_writes(node: SkillNode) -> set[str]:
    writes = set(node.state_writes or [])
    if node.contract is not None:
        writes |= set(node.contract.writes.keys())
    return writes


def _summarize_persistent(tree: SkillTree, persistent: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Schema-aware summary: for each persistent key, every node that reads/writes it, with
    the relevant schema so the architect can wire bindings (fields are derived, not chosen).

    A producer entry carries the node's output_schema; a consumer entry carries its
    input_schema. is_llm flags whether the node's writes are capturable via output_key.
    """
    all_nodes = tree.root.topological_order()
    out: list[dict[str, Any]] = []
    for key in persistent:
        producers: list[dict[str, Any]] = []
        consumers: list[dict[str, Any]] = []
        for n in all_nodes:
            is_llm = n.exec_type == ExecType.LLM_PROMPT
            if key in _node_writes(n):
                producers.append({
                    "node": n.name,
                    "is_llm": is_llm,
                    "capturable": is_llm,  # only LLM output_key writes reach state deterministically
                    "output_schema_properties": _schema_props(n.output_schema),
                })
            if key in _node_reads(n):
                consumers.append({
                    "node": n.name,
                    "is_llm": is_llm,
                    "input_schema_for_key": _input_shape_for_key(n.input_schema, key),
                })
        out.append({
            "state_key": key,
            "written_by": producers,     # candidates for save_source_key (prefer capturable=true)
            "read_by": consumers,        # candidates for load_target_key
        })
    return out


def _schema_props(schema: Optional[dict[str, Any]]) -> dict[str, str]:
    """Flatten a JSON-Schema object's top-level properties to {name: memory_type}."""
    if not schema or schema.get("type") != "object":
        return {}
    props = schema.get("properties") or {}
    return {name: _json_to_memory_type(spec) for name, spec in props.items() if isinstance(spec, dict)}


def _input_shape_for_key(input_schema: Optional[dict[str, Any]], key: str) -> Any:
    """Describe the shape of one input property (the value bound to `key`)."""
    if not input_schema:
        return None
    spec = (input_schema.get("properties") or {}).get(key)
    if not isinstance(spec, dict):
        return None
    if spec.get("type") == "object":
        return {"kind": "object", "properties": _schema_props(spec)}
    return {"kind": spec.get("type", "unknown")}


def _json_to_memory_type(spec: dict[str, Any]) -> str:
    """Map a JSON-Schema property to a MemoryField type (str/int/float/bool/datetime/json)."""
    t = spec.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if t in ("object", "array"):
        return "json"
    if t == "integer":
        return "int"
    if t == "number":
        return "float"
    if t == "boolean":
        return "bool"
    if t == "string" and spec.get("format") in ("date-time", "date"):
        return "datetime"
    return "str"
