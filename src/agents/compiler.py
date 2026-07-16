"""Compiler agent: walk the fully-hydrated SkillTree and emit a Google ADK project."""
from __future__ import annotations

import ast
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.memory_catalog import TEMPLATE_FOR_BACKEND
from src.orchestrator.state import (
    CompositionType,
    ExecType,
    MemoryBackend,
    NodeType,
    NodeVisibility,
    SkillNode,
    SkillTree,
)
from src.skill_lib import SkillLib

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent.parent / "compiler_templates"

_DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

# MemoryField type → Python type hint for a generated pydantic output_schema model.
_MEM_PYTYPE: dict[str, str] = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "datetime": "str",   # store timestamps as ISO strings (JSON-safe, markdown-legible)
    "json": "Any",
}


def _mem_pytype(mem_type: str) -> str:
    """Map a MemoryField type to a Python type hint for the pydantic output model."""
    return _MEM_PYTYPE.get(mem_type, "str")


# JSON Schema type → Python type hint mapping
_TYPE_MAP: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    "null": "None",
}

# composition_type → template filename
_COMPOSITE_TEMPLATES: dict[CompositionType, str] = {
    CompositionType.SEQUENTIAL: "adk_sequential_agent.py.j2",
    CompositionType.PARALLEL: "adk_parallel_agent.py.j2",
    CompositionType.LOOP: "adk_loop_agent.py.j2",
    CompositionType.LLM_COORDINATOR: "adk_coordinator_agent.py.j2",
}


def _pydantic_type(schema: dict[str, Any]) -> str:
    t = schema.get("type", "Any")
    if isinstance(t, list):
        types = [_TYPE_MAP.get(x, "Any") for x in t if x != "null"]
        nullable = "null" in t
        base = " | ".join(types) if types else "Any"
        return f"{base} | None" if nullable else base
    return _TYPE_MAP.get(t, "Any")


def _schema_summary(schema: dict[str, Any]) -> str:
    props = schema.get("properties", {})
    if not props:
        return json.dumps(schema)
    parts = []
    for k, v in props.items():
        t = v.get("type", "Any")
        if isinstance(t, list):
            types = [_TYPE_MAP.get(x, "Any") for x in t if x != "null"]
            nullable = "null" in t
            base = " | ".join(types) if types else "Any"
            type_str = f"{base} | None" if nullable else base
        else:
            type_str = _TYPE_MAP.get(t, "Any")
        parts.append(f"{k}: {type_str}")
    return "{" + ", ".join(parts) + "}"


_STDLIB_TOP_LEVEL = frozenset(
    sys.stdlib_module_names  # type: ignore[attr-defined]  # available in 3.10+
) if hasattr(sys, "stdlib_module_names") else frozenset()

# Packages already listed as base dependencies (or local packages) — never add them.
# `memory` is the generated project's own persistence package, not a PyPI distribution.
_BASE_DEPS = frozenset({"google", "dotenv", "adk", "memory"})

# Maps import-time module names to their correct PyPI distribution names where they differ.
_IMPORT_TO_PYPI: dict[str, str] = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "usaddress": "usaddress",
    "dateutil": "python-dateutil",
    "Crypto": "pycryptodome",
    "jwt": "PyJWT",
    "google.cloud": "google-cloud",
}


def _collect_third_party_deps(tree: SkillTree) -> list[str]:
    """Scan all node implementations for top-level imports; return third-party package names."""
    import_re = re.compile(
        r"^\s{4}(?:import\s+([\w]+)|from\s+([\w]+)\s+import)"
    )
    seen: set[str] = set()
    for node in tree.root.topological_order():
        if not node.implementation:
            continue
        for line in node.implementation.splitlines():
            m = import_re.match(line)
            if not m:
                continue
            pkg = (m.group(1) or m.group(2)).split(".")[0]
            if (
                pkg not in _STDLIB_TOP_LEVEL
                and pkg not in _BASE_DEPS
                and pkg not in seen
            ):
                seen.add(_IMPORT_TO_PYPI.get(pkg, pkg))
    return sorted(seen)


def _collect_required_env_vars(tree: SkillTree) -> list[str]:
    """Scan implementations for os.environ["KEY"] accesses; return sorted key names."""
    env_re = re.compile(r'os\.environ\["([^"]+)"\]|os\.environ\[\'([^\']+)\'\]')
    seen: set[str] = set()
    for node in tree.root.topological_order():
        if not node.implementation:
            continue
        for m in env_re.finditer(node.implementation):
            seen.add(m.group(1) or m.group(2))
    # Memory storage lives at a FIXED default dir (<project>/memory_store/), so no env
    # var is required. MEMORY_STORAGE_DIR is an optional override, intentionally not
    # listed here — we don't want the credential form to demand it.
    return sorted(seen)


def _validate_tree_for_compile(tree: SkillTree) -> None:
    """Ensure every node is fully resolved before compilation.

    Raises ValueError with a consolidated list of problems so the pipeline can
    surface an actionable message instead of a cryptic Jinja/template crash
    (e.g. "'None' has no attribute 'get'" when input_schema is None).
    """
    problems: list[str] = []

    for node in tree.root.topological_order():
        if node.node_type == NodeType.UNKNOWN:
            problems.append(f"'{node.name}': unresolved node_type (never classified)")
            continue

        if node.node_type == NodeType.COMPOSITE:
            if node.composition_type is None:
                problems.append(f"'{node.name}': composite node has no composition_type")
            if not node.children:
                problems.append(f"'{node.name}': composite node has no children")
            continue

        # ATOMIC — must be fully hydrated
        if node.exec_type is None:
            problems.append(f"'{node.name}': atomic node has no exec_type")
            continue
        if node.input_schema is None or node.output_schema is None:
            problems.append(f"'{node.name}': atomic node missing input/output schema (not hydrated)")
        if node.exec_type == ExecType.LLM_PROMPT:
            if not node.instruction:
                problems.append(f"'{node.name}': LLM atomic has no instruction (prompt engineering skipped)")
        else:
            if not node.implementation:
                problems.append(f"'{node.name}': tool atomic has no implementation (implementation skipped)")

    if problems:
        detail = "\n  - ".join(problems)
        raise ValueError(
            f"Cannot compile '{tree.project_name}': {len(problems)} node(s) are not "
            f"fully implemented:\n  - {detail}\n"
            "This usually means the decomposition/implementation pipeline advanced past "
            "a layer without hydrating its atomics. Re-run the pipeline with a fresh blueprint."
        )


def _indent_body(body: str) -> str:
    """Jinja2 filter: ensure implementation body has exactly 4-space base indent.

    Re-bases the LLM-stored body so its first statement sits at 4 spaces, preserving each
    line's indentation RELATIVE to that first line. We key off the first code line's indent
    rather than textwrap.dedent's minimum-over-all-lines, because a multi-line string literal
    (e.g. a triple-quoted prompt) whose interior lines are indented LESS than the surrounding
    code would corrupt the minimum and leave real statements over-indented — producing an
    `unexpected indent` SyntaxError inside the generated `def`.
    """
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return "    raise NotImplementedError()"
    first = lines[0]
    base = len(first) - len(first.lstrip())
    result = []
    for line in lines:
        if not line.strip():
            result.append("")
            continue
        indent = len(line) - len(line.lstrip())
        # Shift so the first line lands at 4 spaces; keep relative depth for deeper lines.
        result.append(" " * (max(0, indent - base) + 4) + line.lstrip())
    return "\n".join(result)


def _snake_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed_skill"


class _Import(NamedTuple):
    """A single import line in a generated file."""
    module: str        # dotted module path, e.g. "atomics.ask_clarifying_question"
    symbol: str        # symbol to import, e.g. "ask_clarifying_question_agent"
    description: str   # human-readable purpose, used in coordinator routing instruction


class CompilerAgent:
    """Generates a complete Google ADK project from a fully-hydrated SkillTree."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape(["html"]),
            keep_trailing_newline=True,
        )
        self._env.filters["pydantic_type"] = _pydantic_type
        self._env.filters["schema_summary"] = _schema_summary
        self._env.filters["tojson"] = lambda v, indent=None: json.dumps(v, indent=indent)
        self._env.filters["indent_body"] = _indent_body
        self._env.filters["snake"] = _snake_name
        self._binding_node_ids: set[str] = set()
        self._producer_enforcement: dict[str, Any] = {}
        # repr() → a valid Python literal (None, 'x'), unlike tojson which emits `null`.
        self._env.filters["pyliteral"] = repr
        # MemoryField type → Python type hint for a generated pydantic output_schema model.
        self._env.filters["mem_pytype"] = _mem_pytype
        # Set per-compile; lets _compile_atomic decide whether to wire the in-agent
        # "forget me" clear tool into a user-facing LlmAgent (see _compile_atomic).
        self._has_memory: bool = False
        self._clear_entities: list[dict[str, str]] = []  # [{name, table}] for clear-tool wiring

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def compile(self, tree: SkillTree, output_dir: Path, skill_lib: SkillLib | None = None) -> Path:
        """Write the Google ADK project to output_dir/{project_name}/."""
        # Fail fast with a clear message if the tree still has unresolved or
        # unhydrated nodes — otherwise the Jinja templates crash deep inside
        # with opaque errors like "'None' has no attribute 'get'".
        _validate_tree_for_compile(tree)

        self._has_memory = tree.memory_spec is not None
        self._clear_entities = (
            [{"name": e.name, "table": _snake_name(e.name)} for e in tree.memory_spec.entities]
            if tree.memory_spec is not None else []
        )
        # Node ids that have any memory binding (before/after load/save callbacks wired).
        self._binding_node_ids: set[str] = set()
        # Producer enforcement: node id → {output_key, fields} for the LLM node that must
        # emit a save_source_key. Its output_key/output_schema are set so the value is
        # written to state deterministically for the after-callback to persist.
        self._producer_enforcement: dict[str, dict[str, Any]] = {}
        if tree.memory_spec is not None:
            nodes_by_id = {n.id: n for n in tree.root.topological_order()}
            for e in tree.memory_spec.entities:
                for b in e.bindings:
                    self._binding_node_ids.add(b.node_id)
                    if not b.save_source_key:
                        continue
                    # Find the LLM node that writes this key; enforce its output there.
                    producer = next(
                        (
                            n for n in nodes_by_id.values()
                            if n.exec_type == ExecType.LLM_PROMPT
                            and (b.save_source_key in (n.state_writes or [])
                                 or (n.contract and b.save_source_key in n.contract.writes))
                        ),
                        None,
                    )
                    if producer is not None:
                        self._producer_enforcement[producer.id] = {
                            "output_key": b.save_source_key,
                            "fields": [{"name": f.name, "type": f.type} for f in e.fields],
                        }

        project_dir = output_dir / tree.project_name
        atomics_dir = project_dir / "atomics"
        orchestrators_dir = project_dir / "orchestrators"

        atomics_dir.mkdir(parents=True, exist_ok=True)
        orchestrators_dir.mkdir(parents=True, exist_ok=True)
        (atomics_dir / "__init__.py").write_text("")
        (orchestrators_dir / "__init__.py").write_text("")

        # Normalise all names to valid Python identifiers before compilation.
        for node in tree.root.topological_order():
            node.name = _snake_name(node.name)

        # ADK enforces a single-parent rule: an agent object may only appear in one
        # sub_agents list. If the blueprint contains the same node name under multiple
        # parents (duplicate subtrees), rename each occurrence after the first so each
        # gets its own Python file and a distinct ADK agent name.
        _seen_names: dict[str, int] = {}
        for node in tree.root.topological_order():
            if node.name in _seen_names:
                _seen_names[node.name] += 1
                node.name = f"{node.name}_{_seen_names[node.name]}"
            else:
                _seen_names[node.name] = 0

        root_symbol = self._compile_node(tree.root, atomics_dir, orchestrators_dir)

        # Determine root module path: either atomics/ or orchestrators/
        if tree.root.node_type == NodeType.ATOMIC:
            root_module = f"atomics.{tree.root.name}"
        else:
            root_module = f"orchestrators.{tree.root.name}"

        has_memory = tree.memory_spec is not None

        # Render interactive entry point (run.py)
        run_tmpl = self._env.get_template("adk_root_orchestrator.py.j2")
        run_code = run_tmpl.render(
            project_name=tree.project_name,
            requirement=tree.requirement,
            root_module=root_module,
            root_symbol=root_symbol,
            has_memory=has_memory,
        )
        (project_dir / "run.py").write_text(run_code)

        # Render ADK web entry point (agent.py) — `adk web .` discovers root_agent here.
        # We must insert this file's parent onto sys.path because adk web runs in
        # single-agent mode and adds agents_dir (the *parent* of this project) to
        # sys.path, making `orchestrators` and `atomics` unimportable without the fix.
        # init_memory() runs AFTER load_dotenv() and BEFORE the root-agent import, so it
        # ensures the Markdown store files exist (and surfaces any error) in the same
        # `import run`/`import agent` smoke test as any other import failure.
        memory_init = "from memory import init_memory\ninit_memory()\n\n" if has_memory else ""
        agent_entry = (
            f'"""{tree.project_name} — ADK web entry point.\n\n'
            '`adk web .` discovers this file and imports `root_agent` from it.\n'
            '"""\n'
            "import sys\n"
            "from pathlib import Path\n\n"
            "_HERE = Path(__file__).parent.resolve()\n"
            "if str(_HERE) not in sys.path:\n"
            "    sys.path.insert(0, str(_HERE))\n\n"
            "from dotenv import load_dotenv\n"
            "load_dotenv()\n\n"
            f"{memory_init}"
            f"from {root_module} import {root_symbol} as root_agent\n\n"
            '__all__ = ["root_agent"]\n'
        )
        (project_dir / "agent.py").write_text(agent_entry)

        # Persistent memory package (only when the Memory Architect designed one).
        # Legacy trees (memory_spec is None) skip this entirely and compile as before.
        if has_memory:
            self._compile_memory(tree, project_dir)
            self._write_ignore_files(project_dir)

        # Generated frontend + interaction contract (only when the UI Designer ran).
        # Legacy trees (ui_spec is None) skip this entirely and compile as before.
        if tree.ui_spec is not None:
            self._compile_frontend(tree, project_dir, has_memory=has_memory)

        # Scan implementations for third-party packages and required env vars
        third_party_deps = _collect_third_party_deps(tree)
        tree.required_env_vars = _collect_required_env_vars(tree)
        pyproject_tmpl = self._env.get_template("adk_pyproject.toml.j2")
        pyproject_content = pyproject_tmpl.render(
            project_name=tree.project_name,
            requirement=tree.requirement,
            third_party_deps=third_party_deps,
        )
        (project_dir / "pyproject.toml").write_text(pyproject_content)

        # Always copy root .env so the generated project has fresh keys.
        # Overwrite any stale copy — the root .env is authoritative.
        root_env = Path(__file__).parent.parent.parent / ".env"
        target_env = project_dir / ".env"
        if root_env.exists():
            import shutil
            shutil.copy2(root_env, target_env)
            logger.info("Copied .env to %s", project_dir)

        # Copy referenced skill_lib entries into the output project as SKILL.md files
        if skill_lib:
            self._copy_skill_lib(tree, project_dir, skill_lib)

        logger.info("Compilation complete → %s", project_dir)
        return project_dir

    # ------------------------------------------------------------------
    # Skill library copy
    # ------------------------------------------------------------------

    def _copy_skill_lib(self, tree: SkillTree, project_dir: Path, skill_lib: SkillLib) -> None:
        """Copy SKILL.md files for all skill_lib-referenced nodes into the project.

        Destination: <project_dir>/skill_lib/<skill_name>/SKILL.md
        Also writes a skill_reader.py so the project can load skills at runtime.
        """
        import shutil

        referenced: set[str] = set()
        for node in tree.root.topological_order():
            if node.skill_lib_ref:
                referenced.add(node.skill_lib_ref)
        # Also save every atomic node that has an implementation (new skills), EXCEPT
        # memory-coupled nodes whose bodies import this project's memory adapters (§5).
        for node in tree.root.topological_order():
            if (
                node.node_type.value == "atomic"
                and not node.skill_lib_ref
                and node.memory_entity_ref is None
            ):
                referenced.add(node.name)

        if not referenced:
            return

        dest_lib = project_dir / "skill_lib"
        dest_lib.mkdir(exist_ok=True)

        for ref_name in sorted(referenced):
            entry = skill_lib.get(ref_name)
            if entry and entry.skill_dir:
                src = entry.skill_dir / "SKILL.md"
                if src.exists():
                    dest_dir = dest_lib / ref_name
                    dest_dir.mkdir(exist_ok=True)
                    shutil.copy2(src, dest_dir / "SKILL.md")
                    logger.debug("Copied skill_lib/%s/SKILL.md to project.", ref_name)

        # Write a thin skill_reader.py so the generated project can load skills
        skill_reader_code = (
            '"""skill_reader — load SKILL.md entries from the bundled skill_lib."""\n'
            "from __future__ import annotations\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "_HERE = Path(__file__).parent.resolve()\n"
            "sys.path.insert(0, str(_HERE.parent.parent))  # ensure src/ is importable if needed\n\n"
            "from src.skill_lib import SkillLib, SkillEntry  # noqa: E402\n\n"
            "_skill_lib: SkillLib | None = None\n\n\n"
            "def get_skill_lib() -> SkillLib:\n"
            '    """Return the singleton SkillLib loaded from this project\'s skill_lib/ dir."""\n'
            "    global _skill_lib\n"
            "    if _skill_lib is None:\n"
            "        _skill_lib = SkillLib(Path(__file__).parent / 'skill_lib')\n"
            "    return _skill_lib\n\n\n"
            "def get_skill(name: str) -> SkillEntry | None:\n"
            '    """Retrieve one skill by name, or None if not found."""\n'
            "    return get_skill_lib().get(name)\n"
        )
        (project_dir / "skill_reader.py").write_text(skill_reader_code)
        logger.info("Wrote skill_reader.py and copied %d skill(s) to %s.", len(referenced), dest_lib)

    # ------------------------------------------------------------------
    # Generated frontend + interaction contract
    # ------------------------------------------------------------------

    def _compile_frontend(self, tree: SkillTree, project_dir: Path, *, has_memory: bool = False) -> None:
        """Emit web/index.html, web/ui_manifest.json, and serve.py from tree.ui_spec.

        user_facing_agents is computed from node.visibility using the already
        snake_cased node.name — that value equals the ADK LlmAgent name, which is
        exactly the `event.author` string the frontend filters on at runtime.
        """
        ui = tree.ui_spec
        assert ui is not None

        user_facing_agents = [
            node.name
            for node in tree.root.topological_order()
            if node.visibility == NodeVisibility.USER_FACING
        ]

        web_dir = project_dir / "web"
        web_dir.mkdir(exist_ok=True)

        manifest_tmpl = self._env.get_template("ui_manifest.json.j2")
        manifest = manifest_tmpl.render(
            project_name=tree.project_name,
            title=ui.title,
            tagline=ui.tagline,
            inputs=[i.value for i in ui.inputs],
            accept_mime_types=ui.accept_mime_types,
            output_renderers=[r.value for r in ui.output_renderers],
            example_prompts=ui.example_prompts,
            user_facing_agents=user_facing_agents,
        )
        # Validate the rendered manifest is well-formed JSON before writing.
        json.loads(manifest)
        (web_dir / "ui_manifest.json").write_text(manifest)

        frontend_tmpl = self._env.get_template("frontend_index.html.j2")
        (web_dir / "index.html").write_text(frontend_tmpl.render())

        serve_tmpl = self._env.get_template("serve.py.j2")
        serve_code = serve_tmpl.render(project_name=tree.project_name, has_memory=has_memory)
        ast.parse(serve_code)  # fail loudly if the template ever produces bad Python
        (project_dir / "serve.py").write_text(serve_code)

        logger.info(
            "Compiled frontend for %s (inputs=%s, renderers=%s, user_facing=%s)",
            tree.project_name,
            [i.value for i in ui.inputs],
            [r.value for r in ui.output_renderers],
            user_facing_agents,
        )

    # ------------------------------------------------------------------
    # Persistent memory package
    # ------------------------------------------------------------------

    def _compile_memory(self, tree: SkillTree, project_dir: Path) -> None:
        """Emit the memory/ package from tree.memory_spec (Markdown-backed storage).

        Emits memory/_store.py (shared MarkdownStore), memory/__init__.py (init_memory()
        + clear registry), and one memory/<snake(entity)>.py adapter per entity from the
        backend's template. State lives as human-readable Markdown tables under the FIXED
        directory <project>/memory_store/. Every adapter also gets a mechanically-derived
        clear_<entity>() deletion function (spec §6/§7). Idempotent — safe to re-run
        (used by the debug repair path).
        """
        spec = tree.memory_spec
        assert spec is not None

        mem_dir = project_dir / "memory"
        mem_dir.mkdir(exist_ok=True)

        entity_views = [self._memory_entity_view(e) for e in spec.entities]

        # Shared markdown store (all IO lives here; adapters are thin config wrappers).
        store_code = self._env.get_template("memory_store.py.j2").render()
        ast.parse(store_code)
        (mem_dir / "_store.py").write_text(store_code)

        # __init__.py — init_memory() (ensures each entity's .md exists) + clear registry.
        init_tmpl = self._env.get_template("memory_init.py.j2")
        init_code = init_tmpl.render(project_name=tree.project_name, entities=entity_views)
        ast.parse(init_code)
        (mem_dir / "__init__.py").write_text(init_code)

        # One adapter module per entity, selected by backend.
        for view in entity_views:
            template_name = TEMPLATE_FOR_BACKEND[view["entity"].backend]
            tmpl = self._env.get_template(template_name)
            code = tmpl.render(**view)
            try:
                ast.parse(code)
            except SyntaxError as exc:
                raise ValueError(
                    f"Compiled memory adapter for '{view['entity'].name}' has a syntax "
                    f"error: {exc}."
                ) from exc
            (mem_dir / f"{view['table']}.py").write_text(code)

        # _bindings.py — deterministic load/save callbacks keyed by node id.
        bindings_code = self._env.get_template("memory_bindings.py.j2").render(
            project_name=tree.project_name, entities=entity_views
        )
        ast.parse(bindings_code)
        (mem_dir / "_bindings.py").write_text(bindings_code)

        logger.info(
            "Compiled memory package for %s (%d entit(y/ies): %s) → memory_store/*.md",
            tree.project_name,
            len(spec.entities),
            ", ".join(f"{e.name}({e.backend.value})" for e in spec.entities),
        )

    def _memory_entity_view(self, entity) -> dict[str, Any]:
        """Build the Jinja render context for one MemoryEntity (Markdown-backed)."""
        table = _snake_name(entity.name)
        class_name = "".join(part.capitalize() for part in table.split("_")) or "Entity"

        data_fields = [f.name for f in entity.fields]
        field_types = {f.name: f.type for f in entity.fields}

        # KEY_VALUE gets a RESERVED "_key" primary column, decoupled from data fields, so
        # the record key (an explicit field value OR the ADK user_id) never collides with
        # a data field of the same name. APPEND_LOG/SEMANTIC have no key column.
        if entity.backend == MemoryBackend.KEY_VALUE:
            pk = "_key"
            field_names = ["_key"] + data_fields
            field_types = {"_key": "str", **field_types}
        else:
            field_names = data_fields or ["value"]
            field_types = field_types or {"value": "str"}
            pk = field_names[0]

        # SEMANTIC: pick the text field to embed/search (first str field, else pk).
        text_field = next((f.name for f in entity.fields if f.type == "str"), pk)

        # Binding views for _bindings.py. Persistence flows through one state key holding a
        # dict; key_field is an entity field used as the KV primary key (else per-user id).
        bindings = [
            {
                "node_id": b.node_id,
                "save_source_key": b.save_source_key,
                "load_target_key": b.load_target_key,
                "key_field": b.key_field,
            }
            for b in entity.bindings
        ]

        return {
            "entity": entity,
            "name": entity.name,
            "backend_value": entity.backend.value,
            "table": table,
            "class_name": class_name,
            "field_names": field_names,
            "field_types": field_types,
            "pk": pk,
            "text_field": text_field,
            "bindings": bindings,
        }

    def _write_ignore_files(self, project_dir: Path) -> None:
        """Emit .gitignore / .dockerignore so persisted user data never enters VCS/images."""
        ignore = self._env.get_template("dotignore.j2").render()
        (project_dir / ".gitignore").write_text(ignore)
        (project_dir / ".dockerignore").write_text(ignore)

    # ------------------------------------------------------------------
    # Recursive compiler core
    # ------------------------------------------------------------------

    def _compile_node(
        self,
        node: SkillNode,
        atomics_dir: Path,
        orchestrators_dir: Path,
    ) -> str:
        """Compile one node recursively. Returns the agent symbol name."""
        if node.node_type == NodeType.ATOMIC:
            return self._compile_atomic(node, atomics_dir)
        elif node.node_type == NodeType.COMPOSITE:
            return self._compile_composite(node, atomics_dir, orchestrators_dir)
        else:
            raise ValueError(
                f"Node '{node.name}' has unresolved node_type='{node.node_type}'. "
                "Re-run the decomposition pipeline with a fresh blueprint."
            )

    def _compile_atomic(self, node: SkillNode, atomics_dir: Path) -> str:
        # If this node has memory bindings, wire deterministic before/after callbacks from
        # memory._bindings (looked up by node id). Persistence is NEVER an LLM tool call.
        memory_node_id = node.id if node.id in self._binding_node_ids else None
        if node.exec_type == ExecType.LLM_PROMPT:
            # If this LLM node produces a save_source_key, enforce output_key + a generated
            # output_schema model so the value is written to state deterministically.
            enforce = self._producer_enforcement.get(node.id)
            tmpl = self._env.get_template("adk_llm_agent_stub.py.j2")
            code = tmpl.render(
                node=node,
                gemini_model=_DEFAULT_GEMINI_MODEL,
                memory_node_id=memory_node_id,
                output_key=(enforce or {}).get("output_key"),
                output_fields=(enforce or {}).get("fields"),
            )
        else:
            tmpl = self._env.get_template("adk_tool_stub.py.j2")
            code = tmpl.render(node=node, gemini_model=_DEFAULT_GEMINI_MODEL, memory_node_id=memory_node_id)

        try:
            ast.parse(code)
        except SyntaxError as exc:
            logger.error(
                "Syntax error in compiled output for '%s': %s — implementation may need re-generation",
                node.name, exc,
            )
            raise ValueError(
                f"Compiled file for '{node.name}' has a syntax error: {exc}. "
                "Re-run tool implementation to regenerate."
            ) from exc

        (atomics_dir / f"{node.name}.py").write_text(code)
        logger.debug("Compiled %s stub: %s", node.exec_type, node.name)
        return f"{node.name}_agent"

    def _compile_composite(
        self,
        node: SkillNode,
        atomics_dir: Path,
        orchestrators_dir: Path,
    ) -> str:
        if node.composition_type is None:
            raise ValueError(
                f"Composite node '{node.name}' has no composition_type. "
                "Re-run the decomposition pipeline with the updated Decomposer to populate it."
            )

        # Depth-first: compile all children first
        child_imports: list[_Import] = []
        for child in node.children:
            symbol = self._compile_node(child, atomics_dir, orchestrators_dir)
            if child.node_type == NodeType.ATOMIC:
                module = f"atomics.{child.name}"
            else:
                module = f"orchestrators.{child.name}"
            child_imports.append(_Import(module=module, symbol=symbol, description=child.description))

        # A composite can carry memory callbacks too (e.g. the root loading at start /
        # saving at end wraps the whole run) — ADK supports before/after on all agents.
        memory_node_id = node.id if node.id in self._binding_node_ids else None

        template_name = _COMPOSITE_TEMPLATES[node.composition_type]
        tmpl = self._env.get_template(template_name)
        code = tmpl.render(
            node=node,
            child_imports=child_imports,
            gemini_model=_DEFAULT_GEMINI_MODEL,
            memory_node_id=memory_node_id,
            routing=node.routing,  # LLM_COORDINATOR routing instruction; None ⇒ thin routing
        )
        try:
            ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(
                f"Compiled composite for '{node.name}' has a syntax error: {exc}."
            ) from exc
        (orchestrators_dir / f"{node.name}.py").write_text(code)
        logger.debug("Compiled composite (%s): %s", node.composition_type, node.name)
        return f"{node.name}_agent"
