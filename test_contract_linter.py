#!/usr/bin/env python3
"""Tests for the ContractLinter's pure-logic detection across composition types."""

from src.agents.contract_linter import ContractLinterAgent
from src.orchestrator.state import (
    CompositionType,
    Contract,
    ExecType,
    NodeType,
    SkillNode,
)


def _composite(comp: CompositionType, reads: dict, writes: dict) -> SkillNode:
    return SkillNode(
        name="parent",
        description="parent composite",
        node_type=NodeType.COMPOSITE,
        composition_type=comp,
        contract=Contract(reads=reads, writes=writes),
    )


def _child(name: str, reads: dict, writes: dict) -> SkillNode:
    return SkillNode(
        name=name,
        description=name,
        node_type=NodeType.ATOMIC,
        exec_type=ExecType.DETERMINISTIC_CODE,
        contract=Contract(reads=reads, writes=writes),
    )


# The linter never calls the LLM for detection, so instantiating it is cheap and offline.
LINTER = ContractLinterAgent()


def test_sequential_clean():
    parent = _composite(CompositionType.SEQUENTIAL, {"raw": "str"}, {"score": "float"})
    parent.children = [
        _child("normalize", {"raw": "str"}, {"clean": "str"}),
        _child("score", {"clean": "str"}, {"score": "float"}),
    ]
    flagged = LINTER.lint_group(parent)
    assert flagged is False, parent.contract_note
    assert parent.contract_note is None
    print("✓ SEQUENTIAL clean → no note")


def test_sequential_missing_upstream_write():
    parent = _composite(CompositionType.SEQUENTIAL, {"raw": "str"}, {"score": "float"})
    parent.children = [
        _child("normalize", {"raw": "str"}, {"clean": "str"}),
        # reads a key nothing upstream (or the parent) writes
        _child("score", {"stripe_event": "dict"}, {"score": "float"}),
    ]
    flagged = LINTER.lint_group(parent)
    assert flagged is True
    assert "stripe_event" in parent.contract_note
    print("✓ SEQUENTIAL missing upstream write → flagged")


def test_sequential_parent_write_unproduced():
    parent = _composite(CompositionType.SEQUENTIAL, {"raw": "str"}, {"score": "float"})
    parent.children = [
        _child("normalize", {"raw": "str"}, {"clean": "str"}),
        _child("score", {"clean": "str"}, {"other": "float"}),  # never writes 'score'
    ]
    flagged = LINTER.lint_group(parent)
    assert flagged is True
    assert "score" in parent.contract_note
    print("✓ SEQUENTIAL unproduced parent write → flagged")


def test_parallel_write_collision():
    parent = _composite(CompositionType.PARALLEL, {"img": "bytes"}, {"a": "str", "b": "str"})
    parent.children = [
        _child("x", {"img": "bytes"}, {"a": "str"}),
        _child("y", {"img": "bytes"}, {"a": "str", "b": "str"}),  # collides on 'a'
    ]
    flagged = LINTER.lint_group(parent)
    assert flagged is True
    assert "disjoint" in parent.contract_note
    print("✓ PARALLEL write collision → flagged")


def test_parallel_reads_sibling_write():
    parent = _composite(CompositionType.PARALLEL, {"img": "bytes"}, {"a": "str", "b": "str"})
    parent.children = [
        _child("x", {"img": "bytes"}, {"a": "str"}),
        _child("y", {"a": "str"}, {"b": "str"}),  # can't read sibling's write in PARALLEL
    ]
    flagged = LINTER.lint_group(parent)
    assert flagged is True
    assert "parent's inputs" in parent.contract_note
    print("✓ PARALLEL child reading sibling write → flagged")


def test_parallel_union_misses_parent_write():
    parent = _composite(CompositionType.PARALLEL, {"img": "bytes"}, {"a": "str", "b": "str"})
    parent.children = [
        _child("x", {"img": "bytes"}, {"a": "str"}),
        _child("y", {"img": "bytes"}, {"c": "str"}),  # union {a,c} misses 'b'
    ]
    flagged = LINTER.lint_group(parent)
    assert flagged is True
    assert "'b'" in parent.contract_note
    print("✓ PARALLEL union misses parent write → flagged")


def test_loop_clean():
    parent = _composite(CompositionType.LOOP, {"state": "dict"}, {"state": "dict"})
    parent.children = [
        _child("step", {"state": "dict", "is_done": "bool"}, {"state": "dict", "is_done": "bool"}),
    ]
    flagged = LINTER.lint_group(parent)
    assert flagged is False, parent.contract_note
    print("✓ LOOP shape-stable with termination key → no note")


def test_loop_shape_unstable():
    parent = _composite(CompositionType.LOOP, {"state": "dict"}, {"state": "dict"})
    parent.children = [
        _child("step", {"state": "dict", "is_done": "bool"}, {"state": "dict"}),  # writes != reads
    ]
    flagged = LINTER.lint_group(parent)
    assert flagged is True
    assert "shape-stable" in parent.contract_note
    print("✓ LOOP shape-unstable → flagged")


def test_loop_missing_termination():
    parent = _composite(CompositionType.LOOP, {"state": "dict"}, {"state": "dict"})
    parent.children = [
        _child("step", {"state": "dict"}, {"state": "dict"}),  # no termination key
    ]
    flagged = LINTER.lint_group(parent)
    assert flagged is True
    assert "termination" in parent.contract_note
    print("✓ LOOP missing termination key → flagged")


def test_coordinator_underproduces():
    parent = _composite(CompositionType.LLM_COORDINATOR, {"q": "str"}, {"answer": "str"})
    parent.children = [
        _child("path_a", {"q": "str"}, {"answer": "str"}),
        _child("path_b", {"q": "str"}, {"other": "str"}),  # doesn't write 'answer'
    ]
    flagged = LINTER.lint_group(parent)
    assert flagged is True
    assert "answer" in parent.contract_note
    print("✓ LLM_COORDINATOR branch underproduces → flagged")


def test_missing_contract_flagged():
    parent = _composite(CompositionType.SEQUENTIAL, {"raw": "str"}, {"score": "float"})
    parent.children = [
        _child("normalize", {"raw": "str"}, {"clean": "str"}),
        SkillNode(name="score", description="score", node_type=NodeType.ATOMIC),  # no contract
    ]
    flagged = LINTER.lint_group(parent)
    assert flagged is True
    assert "missing a contract" in parent.contract_note
    print("✓ Child missing contract → flagged")


def test_lint_layer_only_composites():
    atomic = _child("solo", {"a": "str"}, {"b": "str"})
    atomic.contract_note = "stale"
    parent = _composite(CompositionType.SEQUENTIAL, {"raw": "str"}, {"clean": "str"})
    parent.children = [_child("normalize", {"raw": "str"}, {"clean": "str"})]
    LINTER.lint_layer([atomic, parent])
    # Atomic (non-composite) is left untouched; composite recomputed clean → None
    assert atomic.contract_note == "stale"
    assert parent.contract_note is None
    print("✓ lint_layer only touches composites")


def test_schema_contract_derivation():
    node = SkillNode(
        name="n",
        description="n",
        node_type=NodeType.ATOMIC,
        exec_type=ExecType.DETERMINISTIC_CODE,
        input_schema={"properties": {"raw": {"type": "string"}}},
        output_schema={"properties": {"clean": {"type": "string"}}},
    )
    derived = node.schema_contract()
    assert derived is not None
    assert set(derived.reads.keys()) == {"raw"}
    assert set(derived.writes.keys()) == {"clean"}
    # Unhydrated → None
    assert SkillNode(name="u", description="u").schema_contract() is None
    print("✓ schema_contract() derives from schemas / None when unhydrated")


def test_find_parent_of():
    root = SkillNode(name="root", description="root")
    child = SkillNode(name="c", description="c")
    grand = SkillNode(name="g", description="g")
    child.children.append(grand)
    root.children.append(child)
    assert root.find_parent_of(grand.id) is child
    assert root.find_parent_of(child.id) is root
    assert root.find_parent_of(root.id) is None
    assert root.find_parent_of("nonexistent") is None
    print("✓ find_parent_of resolves parents / None at root")


if __name__ == "__main__":
    test_sequential_clean()
    test_sequential_missing_upstream_write()
    test_sequential_parent_write_unproduced()
    test_parallel_write_collision()
    test_parallel_reads_sibling_write()
    test_parallel_union_misses_parent_write()
    test_loop_clean()
    test_loop_shape_unstable()
    test_loop_missing_termination()
    test_coordinator_underproduces()
    test_missing_contract_flagged()
    test_lint_layer_only_composites()
    test_schema_contract_derivation()
    test_find_parent_of()
    print("\n✅ All contract linter tests passed!")
