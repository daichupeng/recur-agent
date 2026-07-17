#!/usr/bin/env python3
"""Tests for Coordinator-root routing (Feature 1) + conversational clarification (Feature 2).

Covers: RoutingSpec round-trip, decomposer routing parse, linter per-capability checks,
coordinator template rendering (routing vs thin/back-compat), and a full coordinator-root
compile. No LLM / no Gemini calls — pure logic + Jinja rendering.
"""
import ast
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.agents.compiler import CompilerAgent
from src.agents.contract_linter import ContractLinterAgent
from src.agents.decomposer import _parse_routing
from src.orchestrator.state import (
    CompositionType,
    Contract,
    ExecType,
    NodeType,
    NodeVisibility,
    RouteRule,
    RoutingSpec,
    SkillNode,
    SkillTree,
)

_TEMPLATES = Path(__file__).parent / "src" / "compiler_templates"
LINTER = ContractLinterAgent()


# ── RoutingSpec round-trip ────────────────────────────────────────────────────

def test_routing_spec_round_trip():
    root = SkillNode(
        name="root", description="d", depth=0, node_type=NodeType.COMPOSITE,
        composition_type=CompositionType.LLM_COORDINATOR,
        routing=RoutingSpec(
            routes=[RouteRule(child_name="a", trigger="do a", examples=["ex1", "ex2"])],
            fallback="ask", clarify_when="missing file",
        ),
    )
    tree = SkillTree(project_name="p", requirement="r", root=root)
    reloaded = SkillTree.model_validate_json(tree.model_dump_json())
    r = reloaded.root.routing
    assert r is not None
    assert r.routes[0].child_name == "a"
    assert r.routes[0].examples == ["ex1", "ex2"]
    assert r.fallback == "ask"
    assert r.clarify_when == "missing file"
    print("✓ RoutingSpec round-trips through save/load JSON")


def test_routing_none_round_trip_back_compat():
    root = SkillNode(name="root", description="d", depth=0)
    tree = SkillTree(project_name="p", requirement="r", root=root)
    reloaded = SkillTree.model_validate_json(tree.model_dump_json())
    assert reloaded.root.routing is None
    print("✓ routing=None round-trips (back-compat)")


# ── Decomposer routing parse ──────────────────────────────────────────────────

def test_parse_routing_full():
    raw = {
        "routes": [
            {"child_name": "faq", "trigger": "answer a question", "examples": ["hours?"]},
            {"child_name": "ticket", "trigger": "file a ticket"},
            {"child_name": "", "trigger": "ignored — no name"},  # dropped
        ],
        "fallback": "ask a clarifying question",
        "clarify_when": "the order id is missing",
    }
    spec = _parse_routing(raw)
    assert spec is not None
    assert [r.child_name for r in spec.routes] == ["faq", "ticket"]
    assert spec.routes[1].examples == []
    assert spec.fallback == "ask a clarifying question"
    assert spec.clarify_when == "the order id is missing"
    print("✓ _parse_routing builds a spec and drops nameless routes")


def test_parse_routing_none():
    assert _parse_routing(None) is None
    assert _parse_routing({}) is None
    print("✓ _parse_routing tolerates missing data")


# ── Linter: per-capability standalone ─────────────────────────────────────────

def _coordinator(reads, routing=None):
    p = SkillNode(
        name="root", description="d", node_type=NodeType.COMPOSITE,
        composition_type=CompositionType.LLM_COORDINATOR,
        contract=Contract(reads=reads, writes={}), routing=routing,
    )
    return p


def _child(name, reads=None, writes=None):
    return SkillNode(
        name=name, description=name, node_type=NodeType.ATOMIC,
        exec_type=ExecType.LLM_PROMPT, contract=Contract(reads=reads or {}, writes=writes or {}),
    )


def test_coordinator_clean():
    p = _coordinator(
        {"msg": "str"},
        RoutingSpec(routes=[RouteRule(child_name="a", trigger="t"),
                            RouteRule(child_name="b", trigger="t")], fallback="a"),
    )
    p.children = [_child("a", reads={"msg": "str"}, writes={"reply": "str"}),
                  _child("b", reads={"msg": "str"}, writes={"x": "str"})]
    assert LINTER.lint_group(p) is False, p.contract_note
    print("✓ clean coordinator → no note (relaxed superset rule)")


def test_coordinator_unresolved_and_uncovered():
    p = _coordinator(
        {"msg": "str"},
        RoutingSpec(routes=[RouteRule(child_name="zzz", trigger="t")], fallback=""),
    )
    p.children = [_child("a", reads={"msg": "str"}), _child("b", reads={"msg": "str"})]
    assert LINTER.lint_group(p) is True
    note = p.contract_note
    assert "zzz" in note and "no route" in note and "no fallback and no clarify" in note, note
    print("✓ unresolved child_name + uncovered child + undefined fallback all flagged")


def test_coordinator_read_subset_violation():
    p = _coordinator({"msg": "str"})
    p.children = [_child("a", reads={"other": "str"})]
    assert LINTER.lint_group(p) is True
    assert "other" in p.contract_note
    print("✓ child reading a non-parent input flagged")


def test_coordinator_fallback_covers_child():
    # 'b' has no route but IS the fallback → not flagged as uncovered.
    p = _coordinator(
        {"msg": "str"},
        RoutingSpec(routes=[RouteRule(child_name="a", trigger="t")], fallback="b"),
    )
    p.children = [_child("a", reads={"msg": "str"}), _child("b", reads={"msg": "str"})]
    assert LINTER.lint_group(p) is False, p.contract_note
    print("✓ fallback child counts as covered")


# ── Template rendering ────────────────────────────────────────────────────────

def _render_coordinator(routing):
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES)))
    tmpl = env.get_template("adk_coordinator_agent.py.j2")
    node = SkillNode(name="helpdesk", description="A helpdesk bot")
    imps = [
        type("I", (), {"module": "orchestrators.faq", "symbol": "faq_agent", "description": "FAQ"})(),
        type("I", (), {"module": "atomics.ticket", "symbol": "ticket_agent", "description": "ticket"})(),
    ]
    code = tmpl.render(node=node, child_imports=imps, gemini_model="gemini-3.1-flash-lite",
                       memory_node_id=None, routing=routing)
    ast.parse(code)  # must be valid Python
    return code


def test_template_with_routing():
    routing = RoutingSpec(
        routes=[RouteRule(child_name="faq_agent", trigger="answer a question", examples=["hours?"]),
                RouteRule(child_name="ticket_agent", trigger="file a ticket")],
        fallback="ask a clarifying question", clarify_when="the order id is missing",
    )
    code = _render_coordinator(routing)
    assert "faq_agent: answer a question" in code
    assert '"hours?"' in code
    assert "If nothing matches, ask a clarifying question." in code
    assert "If the order id is missing, call request_clarification instead of any capability tool" in code  # Feature 2 clause
    assert "from clarify_tool import request_clarification_tool" in code
    assert "request_clarification_tool," in code
    # Model A: children wrapped as AgentTools, coordinator authors the reply (no transfer).
    assert "AgentTool(agent=faq_agent)" in code
    assert "tools=[" in code and "sub_agents=[" not in code
    print("✓ routing instruction renders triggers + fallback + clarify + AgentTool wiring")


def test_template_no_routes_falls_back_to_child_descriptions():
    # Bug B guard: model may emit an empty routes list; template must still list capabilities.
    routing = RoutingSpec(routes=[], fallback="ask", clarify_when="")
    code = _render_coordinator(routing)
    assert "faq_agent: FAQ" in code  # from child_imports[].description
    assert "AgentTool(agent=faq_agent)" in code
    print("✓ empty routes falls back to child descriptions (no blank routing block)")


def test_template_back_compat_thin_routing():
    code = _render_coordinator(None)
    # No routing metadata → still an orchestrator, capabilities listed from child descriptions.
    assert "faq_agent: FAQ" in code
    assert "AgentTool(agent=faq_agent)" in code
    assert "sub_agents=[" not in code
    print("✓ routing=None renders capability list from child descriptions (back-compat)")


def test_template_clarify_tool_wired_before_capabilities():
    # No live LLM call is made here — this test only checks that the compiled
    # source structurally forces a clarify call to be distinguishable from a
    # capability call: request_clarification_tool is a separate tools[] entry
    # (not folded into a capability's AgentTool), listed before them, and the
    # instruction explicitly tells the model never to call both in one turn.
    routing = RoutingSpec(
        routes=[RouteRule(child_name="faq_agent", trigger="answer a question")],
        fallback="", clarify_when="the order id is missing",
    )
    code = _render_coordinator(routing)
    clarify_idx = code.index("request_clarification_tool,")
    agent_tool_idx = code.index("AgentTool(agent=faq_agent)")
    assert clarify_idx < agent_tool_idx
    assert "Never call request_clarification in the same turn as a capability tool." in code
    print("✓ request_clarification_tool is wired as its own tool before capability AgentTools")


def test_clarify_tool_template_renders_valid_stateless_python():
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES)))
    code = env.get_template("clarify_tool.py.j2").render()
    tree = ast.parse(code)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "request_clarification" in names
    assert "request_clarification_tool" in code
    # Stateless: the function only builds a dict from its own arguments.
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "request_clarification")
    assert len(fn.args.args) == 2  # question, missing_info — no hidden agent/session params
    print("✓ clarify_tool.py.j2 compiles to a single stateless FunctionTool")


# ── Full coordinator-root compile ─────────────────────────────────────────────

def _atomic(name, desc):
    return SkillNode(
        name=name, description=desc, node_type=NodeType.ATOMIC, exec_type=ExecType.LLM_PROMPT,
        input_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"reply": {"type": "string"}}},
        instruction=f"You handle {desc}.",
    )


def test_coordinator_root_compiles(tmp_path):
    root = SkillNode(
        name="assistant", description="A multi-capability assistant", depth=0,
        node_type=NodeType.COMPOSITE, composition_type=CompositionType.LLM_COORDINATOR,
        contract=Contract(reads={"msg": "str"}, writes={}),
        routing=RoutingSpec(
            routes=[RouteRule(child_name="answer_question", trigger="answer a general question"),
                    RouteRule(child_name="analyze_file", trigger="analyze an uploaded file")],
            fallback="ask a clarifying question", clarify_when="no input is provided",
        ),
    )
    root.children = [_atomic("answer_question", "general Q&A"),
                     _atomic("analyze_file", "file analysis")]
    for c in root.children:
        c.depth = 1
        c.parent_id = root.id
    tree = SkillTree(project_name="assistant", requirement="A multi-capability assistant", root=root)

    project_dir = CompilerAgent().compile(tree, tmp_path)
    coord = (project_dir / "orchestrators" / "assistant.py").read_text()
    ast.parse(coord)
    assert "answer_question: answer a general question" in coord
    assert "If nothing matches, ask a clarifying question." in coord
    assert "If no input is provided, call request_clarification instead of any capability tool" in coord
    assert "AgentTool(agent=answer_question_agent)" in coord
    assert "tools=[" in coord and "sub_agents=[" not in coord
    assert (project_dir / "clarify_tool.py").exists()
    print("✓ coordinator-root project compiles with routing + AgentTool wiring")


# ── Coordinator-root enforcement: SEQUENTIAL/PARALLEL/LOOP roots get wrapped ──

def _sequential_root_tree(project_name="pipeline"):
    root = SkillNode(
        name="pipeline", description="A two-step pipeline", depth=0,
        node_type=NodeType.COMPOSITE, composition_type=CompositionType.SEQUENTIAL,
        input_schema={"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
    )
    root.children = [_atomic("step_one", "first step"), _atomic("step_two", "second step")]
    for c in root.children:
        c.depth = 1
        c.parent_id = root.id
    return SkillTree(project_name=project_name, requirement="A two-step pipeline", root=root)


def test_sequential_root_gets_wrapped(tmp_path):
    tree = _sequential_root_tree()
    project_dir = CompilerAgent().compile(tree, tmp_path)

    wrapper_path = project_dir / "orchestrators" / "__root_coordinator.py"
    assert wrapper_path.exists()
    wrapper = wrapper_path.read_text()
    ast.parse(wrapper)
    assert "from orchestrators.pipeline import pipeline_agent" in wrapper
    assert "AgentTool(agent=pipeline_agent)" in wrapper
    assert "request_clarification_tool" in wrapper
    assert wrapper.index("request_clarification_tool,") < wrapper.index("AgentTool(agent=pipeline_agent)")
    assert "the user's message doesn't provide values for: topic" in wrapper

    agent_entry = (project_dir / "agent.py").read_text()
    assert "from orchestrators.__root_coordinator import __root_coordinator_agent as root_agent" in agent_entry
    assert (project_dir / "clarify_tool.py").exists()
    print("✓ SEQUENTIAL root gets wrapped by a synthesized __root_coordinator")


def test_llm_coordinator_root_not_wrapped(tmp_path):
    root = SkillNode(
        name="assistant", description="A multi-capability assistant", depth=0,
        node_type=NodeType.COMPOSITE, composition_type=CompositionType.LLM_COORDINATOR,
        contract=Contract(reads={"msg": "str"}, writes={}),
        routing=RoutingSpec(
            routes=[RouteRule(child_name="answer_question", trigger="answer a general question")],
            fallback="ask a clarifying question", clarify_when="no input is provided",
        ),
    )
    root.children = [_atomic("answer_question", "general Q&A")]
    root.children[0].depth = 1
    root.children[0].parent_id = root.id
    tree = SkillTree(project_name="assistant2", requirement="req", root=root)

    project_dir = CompilerAgent().compile(tree, tmp_path)
    assert not (project_dir / "orchestrators" / "__root_coordinator.py").exists()
    agent_entry = (project_dir / "agent.py").read_text()
    assert "from orchestrators.assistant import assistant_agent as root_agent" in agent_entry
    print("✓ LLM_COORDINATOR root is not double-wrapped")


def test_atomic_root_not_wrapped(tmp_path):
    root = _atomic("solo", "does everything")
    tree = SkillTree(project_name="solo_proj", requirement="req", root=root)

    project_dir = CompilerAgent().compile(tree, tmp_path)
    assert not (project_dir / "orchestrators").exists() or not list(
        (project_dir / "orchestrators").glob("__root_coordinator*")
    )
    agent_entry = (project_dir / "agent.py").read_text()
    assert "from atomics.solo import solo_agent as root_agent" in agent_entry
    print("✓ ATOMIC root is not wrapped")


def test_wrapped_root_manifest_single_voice(tmp_path):
    from src.orchestrator.state import UISpec

    tree = _sequential_root_tree(project_name="pipeline_ui")
    tree.ui_spec = UISpec(title="Pipeline")
    # Simulate a UIDesigner that (incorrectly, pre-enforcement) marked multiple nodes
    # user_facing — the manifest must still gate on the single wrapper voice.
    for node in tree.root.topological_order():
        node.visibility = NodeVisibility.USER_FACING

    project_dir = CompilerAgent().compile(tree, tmp_path)
    manifest = json.loads((project_dir / "web" / "ui_manifest.json").read_text())
    assert manifest["user_facing_agents"] == ["__root_coordinator"]
    print("✓ wrapped root's manifest user_facing_agents is exactly the single wrapper voice")


def test_wrapper_clarify_when_falls_back_without_required(tmp_path):
    root = SkillNode(
        name="pipeline", description="A pipeline", depth=0,
        node_type=NodeType.COMPOSITE, composition_type=CompositionType.PARALLEL,
        input_schema=None,
    )
    root.children = [_atomic("a", "a"), _atomic("b", "b")]
    for c in root.children:
        c.depth = 1
        c.parent_id = root.id
    tree = SkillTree(project_name="parallel_proj", requirement="req", root=root)

    project_dir = CompilerAgent().compile(tree, tmp_path)
    wrapper = (project_dir / "orchestrators" / "__root_coordinator.py").read_text()
    assert "the user's request is missing information this capability needs to run" in wrapper
    print("✓ wrapper clarify_when falls back to a generic phrase when input_schema has no required list")


# ── _indent_body: multi-line string interior must not corrupt dedent ──────────

def test_indent_body_mixed_indentation():
    from src.agents.compiler import _indent_body

    # Code at 6 spaces; a triple-quoted string's interior lines at 4 spaces (less than the
    # code). textwrap.dedent would key off the 4-space minimum and leave `import os` at 6
    # spaces inside a 4-space def → SyntaxError. First-line-based re-basing must not.
    body = (
        "      import os\n"
        "      x = f\"\"\"header\n"
        "    interior line indented less than code\n"
        "      \"\"\"\n"
        "      return x"
    )
    out = _indent_body(body)
    wrapped = "def f():\n" + out
    ast.parse(wrapped)  # must not raise
    assert out.splitlines()[0] == "    import os"
    print("✓ _indent_body handles a string interior indented less than surrounding code")


def test_indent_body_clean_four_space_preserved():
    from src.agents.compiler import _indent_body

    body = "    x = 1\n    if x:\n        y = 2\n    return y"
    assert _indent_body(body) == body
    print("✓ _indent_body leaves an already-4-space body unchanged")


if __name__ == "__main__":
    import sys
    import tempfile
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
    print(f"\nAll {len(fns)} tests passed.")
    sys.exit(0)
