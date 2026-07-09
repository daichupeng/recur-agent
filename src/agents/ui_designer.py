"""UI Designer agent: selects the frontend/interaction contract for a finalized tree.

Runs once, after the whole skill tree is decomposed and implemented, and before
compilation. Chooses — from the fixed interaction catalog — which input affordances
and output renderers the generated product needs, which agents are user-facing vs
internal (so intermediate output is hidden), and which nodes emit binary artifacts.

Produces STRUCTURED DATA only (a UISpec + per-node flags); the compiler turns that
into the actual frontend via Jinja2 templates. Mirrors SchemaArchitectAgent /
PromptEngineerAgent in shape and model choice (default Haiku, tool-use output).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base_agent import BaseAgent, _find_text_block
from src.interaction_catalog import (
    MEDIA_RENDERERS,
    RENDERER_MEDIA_TYPES,
    catalog_prompt_block,
)
from src.orchestrator.state import (
    InputAffordance,
    NodeVisibility,
    OutputRenderer,
    SkillNode,
    SkillTree,
    UISpec,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the UI Designer Agent in a recursive skill-tree generation system.

You receive a finalized skill tree (the product requirement plus every agent node that will
run) and must design the product's frontend by SELECTING from a fixed catalog. You do not
write any code — you return structured choices only.

Decide four things:

1. INPUTS — which input affordances the frontend offers. TEXT is MANDATORY (always include it).
   Add FILE_UPLOAD or IMAGE_UPLOAD ONLY when the requirement or a node's input schema clearly
   consumes that modality. Do not add an input "just in case".

2. OUTPUT RENDERERS — how the product displays answers. TEXT is a safe default. Add MARKDOWN
   when the answer is a report/explanation; TABLE for datasets; CODE for source code; IMAGE only
   if an agent genuinely produces an image; FILE_DOWNLOAD only if an agent genuinely produces a
   downloadable file. Prefer the smallest set that serves the requirement.

3. USER-FACING AGENTS — the names of the nodes whose output the end user should SEE. Mark a node
   user-facing ONLY if its result is meant for the user (typically the final/answer-producing
   agent, or a coordinator that replies directly). Fetchers, parsers, transformers, validators,
   and intermediate steps are INTERNAL and must NOT be listed — their output is hidden. Usually
   1-2 nodes are user-facing. Prefer leaf/answer agents over orchestrator wrappers.

4. NODE MEDIA — for each node that produces a BINARY artifact (an image, a generated file),
   list the MIME types it emits (e.g. "image/png", "text/csv", "application/pdf"). Only include a
   node here if it truly outputs binary media; text/JSON answers do NOT go here. If you enable the
   IMAGE or FILE_DOWNLOAD renderer, at least one node should have matching media types.

Also write a short product `title` and one-line `tagline`, and 2-3 `example_prompts` a user could
try (phrased for the end user, grounded in the requirement).

## Persistent memory (if present)
If the product persists memory entities, they are listed under "Persistent memory" in the input.
When present you MAY enable a TABLE or MARKDOWN renderer to display an entity's read-out (e.g.
"show my alert history"), and you SHOULD add at least one example prompt exercising it. The product
also always supports clearing persisted data via CLI/REST, so a "clear my data" example prompt is
reasonable when an entity holds user-specific data.

## Catalog

{catalog}

Respond with a single JSON object (not an array) matching the provided schema. Use exactly the
node names given in the tree summary for user_facing_node_names and node_media[].name.
"""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "tagline": {"type": "string"},
        "inputs": {
            "type": "array",
            "items": {"type": "string", "enum": [e.value for e in InputAffordance]},
        },
        "output_renderers": {
            "type": "array",
            "items": {"type": "string", "enum": [r.value for r in OutputRenderer]},
        },
        "example_prompts": {"type": "array", "items": {"type": "string"}},
        "user_facing_node_names": {"type": "array", "items": {"type": "string"}},
        "node_media": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "media_types": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "media_types"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "title",
        "inputs",
        "output_renderers",
        "user_facing_node_names",
        "node_media",
    ],
    "additionalProperties": False,
}


class UIDesignerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            system_prompt=_SYSTEM_PROMPT.format(catalog=catalog_prompt_block())
        )

    async def design(self, tree: SkillTree) -> UISpec:
        """Select the interaction contract for `tree` and write it back in-place.

        Sets tree.ui_spec, and on each node sets `visibility` and `output_media_types`.
        Returns the created UISpec.
        """
        summary = _summarize_tree(tree)
        user_content = (
            "Design the frontend for this product.\n\n"
            f"Requirement: {tree.requirement}\n\n"
            "Skill tree (every node that will run):\n"
            + json.dumps(summary, indent=2)
        )
        if tree.memory_spec is not None and tree.memory_spec.entities:
            mem = [
                {
                    "name": e.name,
                    "backend": e.backend.value,
                    "fields": [{"name": f.name, "type": f.type} for f in e.fields],
                    "retention": e.retention,
                }
                for e in tree.memory_spec.entities
            ]
            user_content += (
                "\n\nPersistent memory (entities this product stores across runs):\n"
                + json.dumps(mem, indent=2)
            )

        message = await self._call(
            messages=[{"role": "user", "content": user_content}],
            output_schema=_OUTPUT_SCHEMA,
        )
        result: dict[str, Any] = json.loads(_find_text_block(message).text)

        ui_spec = self._apply_result(tree, result)
        self.log_usage()
        return ui_spec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_result(self, tree: SkillTree, result: dict[str, Any]) -> UISpec:
        """Validate the LLM result against the real tree and write it back."""
        all_names = {n.name for n in tree.root.topological_order()}

        inputs = _coerce_enum(result.get("inputs"), InputAffordance, InputAffordance.TEXT)
        if InputAffordance.TEXT not in inputs:
            inputs.insert(0, InputAffordance.TEXT)  # TEXT is mandatory

        renderers = _coerce_enum(
            result.get("output_renderers"), OutputRenderer, OutputRenderer.TEXT
        )
        if OutputRenderer.TEXT not in renderers:
            renderers.append(OutputRenderer.TEXT)  # always keep a text fallback

        # Validate node names; drop any the LLM invented.
        user_facing = [
            name for name in result.get("user_facing_node_names", []) if name in all_names
        ]
        dropped = [
            name for name in result.get("user_facing_node_names", []) if name not in all_names
        ]
        if dropped:
            logger.warning("UIDesigner: dropping unknown user-facing node names: %s", dropped)

        media_by_name: dict[str, list[str]] = {}
        for entry in result.get("node_media", []):
            name = entry.get("name")
            media = [m for m in entry.get("media_types", []) if m]
            if name in all_names and media:
                media_by_name[name] = media
            elif name not in all_names:
                logger.warning("UIDesigner: dropping media for unknown node '%s'", name)

        # Ensure media renderers have at least one producing node; if the LLM enabled
        # IMAGE/FILE_DOWNLOAD but named no node, fall back to marking user-facing nodes.
        for renderer in renderers:
            if renderer in MEDIA_RENDERERS and not any(
                _mime_matches(renderer, mimes) for mimes in media_by_name.values()
            ):
                default_mimes = RENDERER_MEDIA_TYPES.get(renderer, [])
                targets = user_facing or [n.name for n in _leaf_nodes(tree)]
                for name in targets:
                    media_by_name.setdefault(name, [])
                    for mime in default_mimes:
                        if mime not in media_by_name[name]:
                            media_by_name[name].append(mime)
                logger.info(
                    "UIDesigner: renderer %s enabled but no node emitted it; seeded %s on %s",
                    renderer.value, default_mimes, targets,
                )

        # Write back per-node flags.
        for node in tree.root.topological_order():
            node.visibility = (
                NodeVisibility.USER_FACING
                if node.name in user_facing
                else NodeVisibility.INTERNAL
            )
            node.output_media_types = media_by_name.get(node.name, [])

        accept = _accept_mime_types(inputs)
        ui_spec = UISpec(
            title=result.get("title") or tree.project_name,
            tagline=result.get("tagline", ""),
            inputs=inputs,
            output_renderers=renderers,
            example_prompts=result.get("example_prompts", []),
            user_facing_nodes=user_facing,
            accept_mime_types=accept,
        )
        tree.ui_spec = ui_spec
        logger.info(
            "UIDesigner: inputs=%s renderers=%s user_facing=%s media_nodes=%s",
            [i.value for i in inputs],
            [r.value for r in renderers],
            user_facing,
            list(media_by_name.keys()),
        )
        return ui_spec


def _summarize_tree(tree: SkillTree) -> dict[str, Any]:
    """Compact JSON summary of the tree for the UIDesigner prompt."""
    nodes = []
    for n in tree.root.topological_order():
        nodes.append(
            {
                "name": n.name,
                "description": n.description,
                "node_type": n.node_type.value,
                "exec_type": n.exec_type.value if n.exec_type else None,
                "composition_type": n.composition_type.value if n.composition_type else None,
                "output_schema": n.output_schema,
            }
        )
    return {
        "root": {
            "name": tree.root.name,
            "description": tree.root.description,
            "node_type": tree.root.node_type.value,
            "composition_type": (
                tree.root.composition_type.value if tree.root.composition_type else None
            ),
        },
        "nodes": nodes,
    }


def _leaf_nodes(tree: SkillTree) -> list[SkillNode]:
    return [n for n in tree.root.topological_order() if not n.children]


def _coerce_enum(values: Any, enum_cls: type, default: Any) -> list:
    """Turn a list of enum-value strings into enum members, skipping invalid ones."""
    if not values:
        return [default]
    out = []
    for v in values:
        try:
            member = enum_cls(v)
        except ValueError:
            logger.warning("UIDesigner: ignoring invalid %s value %r", enum_cls.__name__, v)
            continue
        if member not in out:
            out.append(member)
    return out or [default]


def _mime_matches(renderer: OutputRenderer, mimes: list[str]) -> bool:
    from src.interaction_catalog import OUTPUT_CATALOG

    prefix = OUTPUT_CATALOG.get(renderer, {}).get("mime_prefix")
    if prefix:
        return any(m.startswith(prefix) for m in mimes)
    # FILE_DOWNLOAD handles any non-image artifact.
    if renderer == OutputRenderer.FILE_DOWNLOAD:
        return any(not m.startswith("image/") for m in mimes)
    return bool(mimes)


def _accept_mime_types(inputs: list[InputAffordance]) -> list[str]:
    """Derive the HTML file-input accept list from the enabled inputs."""
    from src.interaction_catalog import INPUT_CATALOG

    accept: list[str] = []
    for aff in inputs:
        a = INPUT_CATALOG.get(aff, {}).get("accept")
        if a and a not in accept:
            accept.append(a)
    return accept
