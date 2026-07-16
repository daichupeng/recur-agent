#!/usr/bin/env python3
"""Tests for Coordinator-root routing (Feature 1) + conversational clarification (Feature 2).

Covers: RoutingSpec round-trip, decomposer routing parse, linter per-capability checks,
coordinator template rendering (routing vs thin/back-compat), and a full coordinator-root
compile. No LLM / no Gemini calls — pure logic + Jinja rendering.
"""
import ast
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
    assert "If the order id is missing, do NOT call a capability" in code  # Feature 2 clause
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
    assert "If no input is provided, do NOT call a capability" in coord
    assert "AgentTool(agent=answer_question_agent)" in coord
    assert "tools=[" in coord and "sub_agents=[" not in coord
    print("✓ coordinator-root project compiles with routing + AgentTool wiring")


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
