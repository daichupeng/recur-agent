# Diagnosis Report: recur-agent Generated Output

**Date:** 2026-06-19

## Overview

The recur-agent platform successfully decomposes a product requirement into a multi-layer skill tree (7 layers deep, 28 atomic skills across 9 composite nodes). However, the compiler that converts this tree into a Google ADK project has fundamental architectural flaws that make the generated output non-functional. The output cannot be imported, executed, or debugged in its current state.

---

## Issue 1: The Compiler Discards the Entire Intermediate Tree

**Location:** [`compiler.py:88-89`](src/agents/compiler.py#L88-L89)

**What happens:** 
The compiler collects `atomic_nodes` only — all composite nodes are silently dropped. The tree that was laboriously decomposed across 7+ layers is reduced to a flat bag of leaves.

**Current output:** 
The template [`adk_root_orchestrator.py.j2`](src/compiler_templates/adk_root_orchestrator.py.j2) emits a single `root_agent` with all 19 tool atoms in `tools=[]` and all 9 LLM atoms in `sub_agents=[]`. The five levels of composite structure (e.g., `profile_user_coverage_needs → conduct_multi_turn_conversation → ask_clarifying_question`) are invisible in the output.

**Example from generated code:** [`run.py:16-30`](output/life_insurance_recommender/run.py#L16-L30)
```python
root_agent = Agent(
    name="life_insurance_recommender",
    tools=[parse_user_response, store_indexed_entry, ..., augment_recommendation_object],
    sub_agents=[ask_clarifying_question_agent, ..., generate_cost_benefit_analysis_agent],
)
```

**Expected behavior:**
Composite nodes should compile into ADK orchestration agents (`SequentialAgent`, `LoopAgent`, `ParallelAgent`, or a coordinator `LlmAgent`) that explicitly invoke their children. The nesting from the blueprint tree needs a corresponding nesting in the ADK agent tree. As it stands, a root Gemini agent with 28 tools/sub-agents is being asked to internally divine the pipeline logic that was carefully designed during decomposition.

**Impact:** **Critical** — The generated project has no orchestration layer; all control flow is lost.

---

## Issue 2: All DETERMINISTIC_CODE and EXTERNAL_API Tool Bodies Are Unimplemented Stubs

**Location:** [`adk_tool_stub.py.j2:25-26`](src/compiler_templates/adk_tool_stub.py.j2#L25-L26)

**What happens:** 
The template unconditionally emits:
```python
# TODO: Implement atomic skill — {{ node.name }}
raise NotImplementedError("{{ node.name }} is not yet implemented")
```

**Examples in generated output:**
- [`parse_user_response.py`](output/life_insurance_recommender/atomics/parse_user_response.py)
- [`fetch_singapore_insurance_products.py`](output/life_insurance_recommender/atomics/fetch_singapore_insurance_products.py)
- All 19 other DETERMINISTIC_CODE / EXTERNAL_API tool files

Every one of these crashes immediately if called.

**Expected behavior:**
At minimum, the compiler should generate meaningful scaffolding code from the node's description and schema:
- For **DETERMINISTIC_CODE** nodes: boilerplate that maps inputs to outputs using the schema
- For **EXTERNAL_API** nodes: scaffolded HTTP/SDK call code with placeholder endpoints
- For **LLM_PROMPT** tools (incorrectly classified; see Issue 3): an LLM call wrapper

**Impact:** **Critical** — 19/28 skills are non-functional stubs. Running `adk run` will fail the moment any tool is invoked.

---

## Issue 3: LLM Agent Instructions Contain Python Syntax Errors

**Location:** [`adk_llm_agent_stub.py.j2:8-10`](src/compiler_templates/adk_llm_agent_stub.py.j2#L8-L10)

**What happens:** 
The template embeds `{{ node.output_schema | tojson }}` directly inside a double-quoted Python string. The `tojson` filter produces raw JSON whose interior double quotes are not escaped, breaking the string literal.

**Example from generated code:** [`ask_clarifying_question.py:8-10`](output/life_insurance_recommender/atomics/ask_clarifying_question.py#L8-L10)
```python
instruction=(
    "Generate a single targeted question based on conversation state to progressively uncover a specific gap in user profile (demographics, coverage, goals, risk tolerance). "
    "Return a JSON object matching the output schema: {"type": "object", "properties": {"clarifying_question": {"type": "string", ...}."
)
```

The unescaped `"` characters on line 3 terminate the string literal mid-JSON, producing:
```
SyntaxError: unterminated string literal
```

**Expected behavior:** 
The template must escape the embedded JSON or use a different string escaping method:
```jinja2
"Return a JSON object matching the output schema: {{ node.output_schema | tojson | replace('"', '\\"') }}."
```
Or use a triple-quoted raw string:
```jinja2
r"""Return a JSON object matching the output schema: {{ node.output_schema | tojson }}."""
```

**Impact:** **Critical** — All 9 LLM agent files are unparseable. The project cannot be imported without SyntaxError.

---

## Issue 4: LLM Agents Have No Prompt Engineering — Only a Description

**Location:** [`adk_llm_agent_stub.py.j2:7-10`](src/compiler_templates/adk_llm_agent_stub.py.j2#L7-L10)

**What happens:** 
Even if Issue 3 were fixed, the `LlmAgent` instruction is just the node's one-line description plus a schema request. Example for `ask_clarifying_question`:
> "Generate a single targeted question based on conversation state to progressively uncover a specific gap in user profile... Return a JSON object matching the output schema: {…}."

There is no wiring of the `input_schema` fields (`conversation_history`, `current_profile`) into the instruction. The agent is never told how to receive these inputs, their format, or how to use them.

**Expected behavior:**
The instruction should be a real prompt engineering artifact that:
1. Explicitly lists available inputs from `input_schema` with examples or type hints
2. Specifies the reasoning steps the LLM should follow
3. States constraints (e.g., "ask only one question")
4. Provides concrete output format instructions with examples
5. References the `output_schema` with field descriptions, not just raw JSON

Example:
```
You will receive:
- conversation_history: array of {"role": "user|assistant", "content": "..."} messages
- current_profile: object with demographics, coverage, goals, risk_tolerance fields

Your task:
1. Analyze what gaps remain in the user profile based on the conversation so far
2. Select the single most important gap to address
3. Formulate a natural, conversational question to fill that gap
4. Output a JSON object: {"clarifying_question": "...", "target_category": "...", "expected_response_type": "..."}
```

The Schema Architect generates detailed I/O specs — none of that makes it into the LLM instruction.

**Impact:** **High** — LLM agents are underpowered and will likely fail to produce correctly-structured outputs.

---

## Issue 5: The Entry Point Has No Conversation Loop or I/O Scaffolding

**Location:** [`run.py:136-138`](output/life_insurance_recommender/run.py#L136-L138)

**What happens:**
```python
if __name__ == "__main__":
    runner = InMemoryRunner(agent=root_agent)
    runner.run()
```

`InMemoryRunner.run()` called with no arguments does not present a user-facing interface. There is no STDIN loop, no threading of user input through the agent, and no session management.

**Expected behavior:**
A working entry point needs:
- Interactive CLI with STDIN/STDOUT wiring
- Session persistence (storing conversation history, user profile state)
- Proper runner initialization (e.g., `runner.run_interactive()` or equivalent ADK API)
- Or a web/API server mode (e.g., `adk web` scaffold)

Without this, the user cannot interact with the generated agent.

**Impact:** **High** — The application has no user-facing interface.

---

## Issue 6: Schema Hydration Only Runs on Current-Layer Atomics

**Location:** [`pipeline.py:118-127`](src/orchestrator/pipeline.py#L118-L127)

**What happens:**
```python
approved_atomics = [
    n for n in tree.get_layer_nodes()
    if n.node_type == NodeType.ATOMIC
]
```

This only hydrates nodes at `current_layer`. If an atomic node was created deeper in the tree (e.g., at depth 5) during a layer-2 decomposition run, but the pipeline never explicitly returns to that depth, the node may never be hydrated.

**Expected behavior:**
Schema hydration should traverse the entire tree and hydrate every node with `node_type == ATOMIC` and `input_schema is None`, regardless of depth or when it was created.

**Impact:** **Medium** — Some atomic nodes may lack I/O schemas, making their contracts undefined. This is masked by Issue 7.

---

## Issue 7: Composite Nodes Have No Input/Output Schemas

**Location:** Schema Architect design; all composite nodes in `blueprint_verified.json`

**What happens:**
The Schema Architect only processes atomic nodes. Composite nodes in the final blueprint all have `"input_schema": null, "output_schema": null`. For example, in [`layers/layer_7/blueprint_verified.json`](output/life_insurance_recommender/layers/layer_7/blueprint_verified.json), `profile_user_coverage_needs` (composite) has null schemas, while its atomic descendant `ask_clarifying_question` has full I/O specs.

**Why it matters:**
The data contract between layers is undefined. When the orchestration layer (Issue 1) tries to route output from `profile_user_coverage_needs` to `research_singapore_life_insurance_market`, there is no type contract specifying what data should flow between them.

**Expected behavior:**
For the generated pipeline to be coherent, the Schema Architect should also synthesize I/O schemas for composite nodes based on:
1. The composite's description
2. The aggregate I/O of its children
3. The composition pattern (sequential, parallel, loop, etc.)

This ensures end-to-end type safety from leaf atoms to the root.

**Impact:** **Medium** — Data flow between composed nodes is undefined and untraceable.

---

## Impact Summary Table

| # | Severity | Blockers | Problem |
|---|----------|----------|---------|
| 1 | **Critical** | Orchestration logic lost | All hierarchy discarded; generated structure is wrong |
| 2 | **Critical** | 19/28 skills non-functional | Stubs raise `NotImplementedError` |
| 3 | **Critical** | Import fails | 9/28 files have `SyntaxError` in string literals |
| 4 | **High** | LLM quality unknown | Agents lack proper prompting; schemas not wired |
| 5 | **High** | No user interface | No entry point for interaction |
| 6 | **Medium** | Edge case gaps | Schemas may be missed for deeply-nested atomics |
| 7 | **Medium** | Type safety undefined | Composite→composite contracts missing |

---

## What Currently Works

- ✅ Decomposition (7 layers, tree structure correctly formed)
- ✅ Complexity review (annotations generated)
- ✅ Schema hydration on leaf nodes (19 atomics have schemas)
- ✅ Blueprint JSON serialization (snapshots & rollback functional)
- ✅ UI/HITL approval flow (manual override mechanism works)

---

## What Needs to Be Built

1. **Orchestration Code Generation** — Compile composite nodes into explicit orchestrator agents with routing logic
2. **Deterministic Tool Implementation** — Generate meaningful code scaffolds from descriptions + schemas
3. **Template Syntax Fix** — Escape or re-quote the JSON in LLM instructions
4. **Prompt Engineering Template** — Embed input_schema details into LLM instructions
5. **Interactive Entry Point** — Wire up a working CLI or web interface
6. **Full-Tree Schema Hydration** — Apply SchemaArchitect to composites + ensure all atomics get schemas
7. **End-to-End Type Safety** — Propagate schemas up the tree and validate cross-node data flow

---

## Root Cause Analysis

The compiler was designed to be a **thin code-gen pass** that assumes:
- All nodes will eventually be decomposed into atomics (true in happy path)
- Atomics don't need orchestration (false — they need routing)
- Stub tools are acceptable scaffolding (false — they block all execution)
- JSON can be embedded in Python strings without escaping (false — bug in templating)

The actual requirement is a **full-fidelity compilation step** that:
- Preserves the tree structure as a hierarchy of orchestration agents
- Generates executable code, not stubs
- Properly escapes and engineers LLM prompts
- Wires up a complete application entry point

