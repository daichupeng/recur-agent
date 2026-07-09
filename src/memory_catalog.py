"""Memory backend catalog: the fixed menu of persistence backends.

This is the "template of potential storage backends" the MemoryArchitectAgent selects
from, analogous to the ExecType / CompositionType enums the Decomposer selects from, and
directly modeled on src/interaction_catalog.py.

Kept as typed Python (not YAML) so it is importable by both:
  - src/agents/memory_architect.py — to build the backend-selection prompt text
  - src/agents/compiler.py          — to map a MemoryBackend enum → its Jinja template

The compiler switches on the template filename stored here, so those filenames are a
stable contract with src/compiler_templates/.
"""
from __future__ import annotations

from src.orchestrator.state import MemoryBackend

# ── Backend catalog ──────────────────────────────────────────────────────────
# template: the Jinja file in src/compiler_templates/ that emits this adapter.
# ops:      the adapter methods the tool implementor may call.
# desc:     the selection guidance injected into the MemoryArchitect prompt.
MEMORY_BACKEND_CATALOG: dict[MemoryBackend, dict] = {
    MemoryBackend.KEY_VALUE: {
        "template": "memory_keyvalue.py.j2",
        "ops": ["get", "set", "delete"],
        "desc": (
            "DEFAULT. Point lookups keyed by a primary field: user profiles, dedup "
            "flags, settings, last-seen markers. Choose this unless the description "
            "clearly needs one of the others."
        ),
    },
    MemoryBackend.APPEND_LOG: {
        "template": "memory_appendlog.py.j2",
        "ops": ["append", "query"],
        "desc": (
            "Growing history / audit trail. Choose when the description implies "
            "accumulation over time (history, log, events over time) rather than "
            "point lookups. Rows are appended and queried, never updated."
        ),
    },
    MemoryBackend.SEMANTIC: {
        "template": "memory_semantic.py.j2",
        "ops": ["add", "search"],
        "desc": (
            "Fuzzy / similarity recall only ('remember what we discussed', 'find "
            "related past cases'). NEVER a default — requires a local embedding store. "
            "Choose only when the description explicitly implies similarity search."
        ),
    },
}

# MemoryBackend → template filename (used by CompilerAgent._compile_memory).
TEMPLATE_FOR_BACKEND: dict[MemoryBackend, str] = {
    backend: meta["template"] for backend, meta in MEMORY_BACKEND_CATALOG.items()
}

# MemoryBackend → adapter ops (used by ToolImplementorAgent to steer memory-mode code).
OPS_FOR_BACKEND: dict[MemoryBackend, list[str]] = {
    backend: meta["ops"] for backend, meta in MEMORY_BACKEND_CATALOG.items()
}


def memory_catalog_prompt_block() -> str:
    """Return the formatted backend catalog injected into the MemoryArchitect prompt.

    Mirrors interaction_catalog.catalog_prompt_block().
    """
    lines: list[str] = ["MEMORY BACKENDS (choose exactly one per entity):"]
    for backend, meta in MEMORY_BACKEND_CATALOG.items():
        lines.append(f"  - {backend.value}: {meta['desc']}")
        lines.append(f"      operations: {', '.join(meta['ops'])}")
    return "\n".join(lines)
