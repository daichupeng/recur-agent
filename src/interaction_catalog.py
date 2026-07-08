"""Interaction catalog: the fixed menu of frontend input/output capabilities.

This is the "template of potential interactions" the UIDesignerAgent selects from,
analogous to the ExecType / CompositionType enums the Decomposer selects from.

Kept as typed Python (not YAML) so it is importable by both:
  - src/agents/ui_designer.py  — to build the enum lists + prompt text
  - src/agents/compiler.py      — to map an enum → the frontend widget/renderer id

The frontend template (frontend_index.html.j2) switches on the `widget_id` /
`renderer_id` strings stored here, so those ids are a stable contract with the JS.
"""
from __future__ import annotations

from src.orchestrator.state import InputAffordance, OutputRenderer

# ── Input affordances ────────────────────────────────────────────────────────
# widget_id: the value the frontend JS switches on to render the input control.
INPUT_CATALOG: dict[InputAffordance, dict] = {
    InputAffordance.TEXT: {
        "label": "Text box",
        "widget_id": "text",
        "desc": "Free-form text prompt. ALWAYS available; include it in every design.",
    },
    InputAffordance.FILE_UPLOAD: {
        "label": "File upload",
        "widget_id": "file",
        "desc": (
            "User attaches an arbitrary file (CSV, PDF, docx, JSON). Choose when the "
            "task consumes documents or data files as input."
        ),
        "accept": "*/*",
    },
    InputAffordance.IMAGE_UPLOAD: {
        "label": "Image upload / paste",
        "widget_id": "image",
        "desc": (
            "User uploads or pastes an image. Choose for vision, OCR, or "
            "image-analysis tasks."
        ),
        "accept": "image/*",
    },
}

# ── Output renderers ─────────────────────────────────────────────────────────
# renderer_id: the value the frontend JS switches on to render an agent output.
# mime_prefix (optional): artifact MIME prefix this renderer handles.
OUTPUT_CATALOG: dict[OutputRenderer, dict] = {
    OutputRenderer.TEXT: {
        "label": "Plain text",
        "renderer_id": "text",
        "desc": "Plain text answer. Default; always safe to include.",
    },
    OutputRenderer.MARKDOWN: {
        "label": "Markdown",
        "renderer_id": "markdown",
        "desc": (
            "Rich text with headings, lists, tables, and inline code. Choose when the "
            "product's answer is a report or formatted explanation."
        ),
    },
    OutputRenderer.TABLE: {
        "label": "Table",
        "renderer_id": "table",
        "desc": "Structured tabular data (rows/columns). Choose for datasets or comparisons.",
    },
    OutputRenderer.IMAGE: {
        "label": "Image",
        "renderer_id": "image",
        "mime_prefix": "image/",
        "desc": (
            "Display an image the agent produces (chart, diagram, generated image). "
            "Requires a node that emits an image artifact."
        ),
    },
    OutputRenderer.FILE_DOWNLOAD: {
        "label": "File download",
        "renderer_id": "file",
        "desc": (
            "Offer a generated file for download (CSV, PDF, xlsx). Requires a node "
            "that emits a file artifact."
        ),
    },
    OutputRenderer.CODE: {
        "label": "Code block",
        "renderer_id": "code",
        "desc": "Syntax-highlighted source code output.",
    },
}

# ── OutputRenderer → default artifact MIME types ─────────────────────────────
# Used by UIDesignerAgent as a hint when seeding a node's output_media_types, and
# as the default when the LLM enables a media renderer without naming a MIME type.
RENDERER_MEDIA_TYPES: dict[OutputRenderer, list[str]] = {
    OutputRenderer.IMAGE: ["image/png"],
    OutputRenderer.FILE_DOWNLOAD: ["application/octet-stream"],
    OutputRenderer.TEXT: [],
    OutputRenderer.MARKDOWN: [],
    OutputRenderer.TABLE: [],
    OutputRenderer.CODE: [],
}

# Renderers that require a node to emit a binary artifact (vs. the text channel).
MEDIA_RENDERERS: frozenset[OutputRenderer] = frozenset(
    {OutputRenderer.IMAGE, OutputRenderer.FILE_DOWNLOAD}
)


def catalog_prompt_block() -> str:
    """Return the formatted catalog text injected into the UIDesigner system prompt.

    Mirrors how decomposer.py inlines its exec-type list into its prompt.
    """
    lines: list[str] = ["INPUT AFFORDANCES (choose a subset; TEXT is mandatory):"]
    for aff, meta in INPUT_CATALOG.items():
        lines.append(f"  - {aff.value}: {meta['desc']}")
    lines.append("")
    lines.append("OUTPUT RENDERERS (choose a subset; TEXT is a safe default):")
    for rnd, meta in OUTPUT_CATALOG.items():
        lines.append(f"  - {rnd.value}: {meta['desc']}")
    return "\n".join(lines)
