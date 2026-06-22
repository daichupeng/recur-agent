"""Compiler agent: walk the fully-hydrated SkillTree and emit a Google ADK project."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, NamedTuple

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.orchestrator.state import CompositionType, ExecType, NodeType, SkillNode, SkillTree

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent.parent / "compiler_templates"

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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def compile(self, tree: SkillTree, output_dir: Path) -> Path:
        """Write the Google ADK project to output_dir/{project_name}/."""
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

        root_symbol = self._compile_node(tree.root, atomics_dir, orchestrators_dir)

        # Determine root module path: either atomics/ or orchestrators/
        if tree.root.node_type == NodeType.ATOMIC:
            root_module = f"atomics.{tree.root.name}"
        else:
            root_module = f"orchestrators.{tree.root.name}"

        # Render interactive entry point (run.py)
        run_tmpl = self._env.get_template("adk_root_orchestrator.py.j2")
        run_code = run_tmpl.render(
            project_name=tree.project_name,
            requirement=tree.requirement,
            root_module=root_module,
            root_symbol=root_symbol,
        )
        (project_dir / "run.py").write_text(run_code)

        # Render pyproject.toml
        pyproject_tmpl = self._env.get_template("adk_pyproject.toml.j2")
        pyproject_content = pyproject_tmpl.render(
            project_name=tree.project_name,
            requirement=tree.requirement,
        )
        (project_dir / "pyproject.toml").write_text(pyproject_content)

        logger.info("Compilation complete → %s", project_dir)
        return project_dir

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
        if node.exec_type == ExecType.LLM_PROMPT:
            tmpl = self._env.get_template("adk_llm_agent_stub.py.j2")
            code = tmpl.render(node=node)
            (atomics_dir / f"{node.name}.py").write_text(code)
            logger.debug("Compiled LLM agent stub: %s", node.name)
            return f"{node.name}_agent"
        else:
            tmpl = self._env.get_template("adk_tool_stub.py.j2")
            code = tmpl.render(node=node)
            (atomics_dir / f"{node.name}.py").write_text(code)
            logger.debug("Compiled tool stub: %s", node.name)
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

        template_name = _COMPOSITE_TEMPLATES[node.composition_type]
        tmpl = self._env.get_template(template_name)
        code = tmpl.render(node=node, child_imports=child_imports)
        (orchestrators_dir / f"{node.name}.py").write_text(code)
        logger.debug("Compiled composite (%s): %s", node.composition_type, node.name)
        return f"{node.name}_agent"
