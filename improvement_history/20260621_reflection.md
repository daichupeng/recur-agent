# Reflection: recur-agent Engine Problems

**Date:** 2026-06-21
**Status:** Analysis only — no solutions.
**Scope:** Problems in `src/` that explain why the generated `life_insurance_recommender` does not work.

---

## Category 1 — Big Picture: Architecture, Orchestration, Workflow

### A1. The engine has no model of ADK's runtime semantics — it generates structurally valid but semantically broken agents

The Decomposer, compiler, and all templates treat the target framework (Google ADK) as an opaque code-generation target. The engine knows the four composition types (`SEQUENTIAL`, `LOOP`, `PARALLEL`, `LLM_COORDINATOR`) as abstract patterns, but none of the agents encode what those patterns mean *inside ADK*:

- ADK's `LoopAgent` requires exactly one sub-agent and an explicit escalation signal to terminate; `adk_loop_agent.py.j2` emits `LoopAgent` with N sub-agents and no `max_iterations`, making every `LOOP` node either infinite or immediately dead.
- ADK's `LLM_COORDINATOR` pattern is an `LlmAgent` with `sub_agents=[]`, but routing only works if the `instruction` field describes *when to call which sub-agent*; `adk_coordinator_agent.py.j2` emits no `instruction`.
- ADK agents communicate through `ctx.session.state` keys, not function arguments; the engine generates `input_schema`/`output_schema` as function signatures for `FunctionTool` wrappers and never addresses how `LlmAgent` nodes read or write session state.

These are not template bugs — they reflect a design gap: the engine must encode ADK-specific composition rules in the Decomposer system prompt, not only in the templates.

### A2. `pipeline.py` runs tool implementation and schema hydration inside the HITL loop with no corresponding HITL step for those phases

The pipeline exposes exactly one HITL checkpoint: approve or rollback the decomposition. After approval, `SchemaArchitectAgent` and `ToolImplementorAgent` run silently and the tree is immediately compiled. The human sees the node structure but never sees — and cannot correct — the generated schemas or function bodies before the project is written to disk. Implementation errors (indentation bugs, wrong API calls) and schema mismatches are only discovered when running the compiled project. The engine's pipeline layout treats schema hydration and tool implementation as internal steps when they are in practice the steps most likely to produce incorrect output.

### A3. The engine decomposes ADK targets without giving the Decomposer any knowledge of ADK composition semantics

`decomposer.py`'s `_SYSTEM_PROMPT` defines `LOOP` as "one child agent runs repeatedly until a condition is met" without mentioning how the condition is implemented in ADK (escalation). It defines `LLM_COORDINATOR` as "an LLM decides which child to invoke" without mentioning that the routing requires an `instruction`. The Decomposer produces structurally plausible trees that are semantically broken when rendered into ADK, because the ADK-specific constraints were never part of its decision context.

### A4. Binary HITL model (approve whole layer / rollback whole layer) provides no granular correction path

`pipeline.py:98-116` implements a two-event `asyncio.wait` — either `events.approve` or `events.rollback`. There is no way to:
- Edit a single node's description to guide better re-decomposition
- Accept most of a layer but re-decompose one subtree
- Pass a correction hint (e.g., "use SEQUENTIAL here, not LOOP") to the Decomposer for a retry

The practical consequence is that a human reviewer who sees one bad node must either accept the whole layer (accumulating technical debt in the tree) or rollback and hope the LLM produces something different. Both options are costly. The HITL design stops exactly where granular control matters most.

---

## Category 2 — Critical Errors in Local Implementation

### B1. `_normalise_indent()` in `tool_implementor.py` compounds LLM indentation inconsistency into a hard `IndentationError`

`tool_implementor.py:118-135`:

```python
def _normalise_indent(body: str) -> str:
    dedented = textwrap.dedent(body)
    lines = dedented.splitlines()
    ...
    return "\n".join("    " + line if line.strip() else "" for line in lines)
```

`textwrap.dedent` strips only the *minimum common leading whitespace*. The LLM system prompt tells it to emit "the lines that go INSIDE the function, indented with 4 spaces" but says "all stdlib imports must be inside the function body" without specifying their indentation. The LLM emits `import` statements at column 0 (module-level style) and function-body lines at 4 spaces. The minimum common indent is 0 — `textwrap.dedent` strips nothing. The flat `+4` prepend then produces `import` at 4 spaces and all other lines at 8 spaces, which is `IndentationError` inside a function definition. Every tool file processed by this agent is unimportable. The normaliser cannot fix an ambiguity the system prompt introduced.

### B2. `adk_tool_stub.py.j2` renders `node.exec_type` as a Python enum repr, not its string value

`adk_tool_stub.py.j2:2`:
```jinja2
# SANDBOX: This tool executes {{ node.exec_type }} logic.
```

`node.exec_type` is a Python `ExecType` enum instance. Jinja renders it as `ExecType.EXTERNAL_API` (Python `str()` of the enum) rather than `EXTERNAL_API` (the `.value`). The template should use `{{ node.exec_type.value }}`. This pattern repeats wherever enum fields are rendered directly in templates — none of the templates reference `.value`.

### B3. `adk_loop_agent.py.j2` emits `LoopAgent` with no `max_iterations` and accepts N sub-agents

`src/compiler_templates/adk_loop_agent.py.j2` generates:
```python
{{ node.name }}_agent = LoopAgent(
    name="{{ node.name }}",
    sub_agents=[...],
)
```
Without `max_iterations`, the loop never terminates unless a sub-agent escalates. No sub-agent in the generated project emits an escalation. The complete fix is two-part: (1) the template must emit `max_iterations`, and (2) the Decomposer must be told that `LOOP` nodes require exactly one sub-agent and a termination condition — but neither is present in any engine file.

### B4. `adk_coordinator_agent.py.j2` emits `LlmAgent` with `sub_agents` but no `instruction`

`src/compiler_templates/adk_coordinator_agent.py.j2` generates:
```python
{{ node.name }}_agent = LlmAgent(
    name="{{ node.name }}",
    model="gemini-2.0-flash",
    description="{{ node.description }}",
    sub_agents=[...],
)
```
ADK routes sub-agent calls based on the `LlmAgent`'s `instruction`. Without an `instruction`, the coordinator LLM has no system prompt and routing decisions are arbitrary. The template must generate an `instruction` that describes each sub-agent and when to invoke it — which requires the compiler to pass child descriptions to the template, which it currently does not.

### B5. `adk_llm_agent_stub.py.j2` never wires `input_schema` into the instruction

`src/compiler_templates/adk_llm_agent_stub.py.j2`:
```jinja2
instruction="""{{ node.description }}

Return a JSON object matching this schema:
{{ node.output_schema | tojson(indent=2) }}
""",
```
The node's `input_schema` — which describes exactly what arguments the agent receives — is never rendered. The LLM agent has no idea what inputs are available or how to access them from session state. The template has access to `node.input_schema` but doesn't use it.

### B6. `adk_tool_stub.py.j2` wraps the `FunctionTool` in an `LlmAgent` with no `instruction`

Every tool atomic generates both a `FunctionTool` and a wrapping `LlmAgent`. The wrapper:
```python
{{ node.name }}_agent = LlmAgent(
    name="{{ node.name }}",
    model="gemini-2.0-flash",
    description="{{ node.description }}",
    tools=[{{ node.name }}],
)
```
has no `instruction`. The LLM driving the wrapper has no guidance on when or how to call the tool, what arguments to pass, or what to do with the result. The `description` field is for ADK's sub-agent routing metadata, not the agent's own operating instruction.

---

## Category 3 — Bugs and Unoptimized Flows

### C1. `decomposer.py` sends only `{name, description}` with no ancestor or sibling context

`decomposer.py:120-123`:
```python
node_dicts = [
    {"name": n.name, "description": n.description} for n in nodes
]
```
The Decomposer sees a flat list of sibling nodes with no knowledge of: the root-to-node path, the parent's composition type, what siblings exist at the same level, or what the requirements document says about adjacent responsibilities. Two siblings in a `SEQUENTIAL` parent may be decomposed independently with incompatible data shapes, duplicated logic, or overlapping concerns because the Decomposer has no structural context.

### C2. `schema_architect.py` sends nodes without sibling context — schemas are generated in isolation

`schema_architect.py:63-70`:
```python
node_dicts = [
    {"name": n.name, "description": n.description, "exec_type": n.exec_type}
    for n in nodes
]
```
In a `SEQUENTIAL` orchestrator, node N's `output_schema` must be compatible with node N+1's `input_schema`. But the Schema Architect generates each schema in isolation without knowing what adjacent nodes produce or consume. A field called `user_profile` in one schema and `profile` in the next will silently break the pipeline.

### C3. `tool_implementor.py` system prompt is ambiguous about import indentation

`_SYSTEM_PROMPT` says: "all stdlib imports must be inside the function body." This is interpreted by the LLM as "place `import` at column 0 of the body block" (module-level convention) rather than "each line, including imports, must begin with exactly 4 spaces." The ambiguity is the root cause of the `_normalise_indent` failure. The system prompt should read: "every line of the function body, including import statements, must be indented with exactly 4 leading spaces."

### C4. `base_agent.py` hardcodes `MODEL = "claude-haiku-4-5"` for all agents with no per-agent override

`base_agent.py:16`: `MODEL = "claude-haiku-4-5"`. Every agent in the engine — Decomposer, SchemaArchitect, ToolImplementor, ComplexityReviewer, Compiler — uses Haiku. There is no override mechanism. Code generation tasks (ToolImplementor) benefit significantly from a more capable model; structural tasks (ComplexityReviewer) are fine on Haiku. A single hardcoded class variable prevents appropriate model selection per agent.

### C5. `complexity_reviewer.py` flags go into `review_note` but the pipeline never acts on them

`complexity_reviewer.py` annotates nodes and logs flagged counts, but `pipeline.py` never reads `review_note` to take any action. The notes are stored in `SkillNode.review_note` and the UI theoretically shows them, but the pipeline has no code path to automatically re-decompose flagged nodes, suggest a rollback, or escalate. A reviewer that can never act on its findings adds only noise.

### C6. Blank lines between imports in all composite orchestrator templates — unstripped Jinja whitespace

All four orchestrator templates (`adk_sequential_agent.py.j2`, `adk_loop_agent.py.j2`, `adk_parallel_agent.py.j2`, `adk_coordinator_agent.py.j2`) use:
```jinja2
{% for imp in child_imports %}
from {{ imp.module }} import {{ imp.symbol }}
{% endfor %}
```
The newline after each `{% endfor %}` block emits a blank line between every import statement. This was flagged in the previous debug cycle (C1) and was not fixed. The correct pattern is `{%- for ... -%}` with explicit `\n` where needed.

### C7. ToolImplementor batches up to 10 code-generation tasks in a single API call

`tool_implementor.py:74`: `_BATCH_SIZE = 10`. Generating 10 function bodies in a single call forces the LLM to maintain all 10 contexts simultaneously. For code generation — where precise, correct implementation matters more than throughput — smaller batches or single-node calls produce better output. The batch design was copied from the decomposer pattern where it is appropriate, but code generation has different quality characteristics.

### C8. `pipeline.py` runs the full topological scan every layer, even when no new tool nodes were added

`pipeline.py:135-146` scans `tree.root.topological_order()` for `n.implementation is None` on every layer approval. For a 4-layer tree, the same already-processed nodes are scanned 4 times (filtered by `is None`, so no re-work, but the scan itself runs). More importantly, the `ToolImplementorAgent` is instantiated fresh each layer (`tool_implementor = ToolImplementorAgent()` at `pipeline.py:47`) — token usage is reset, losing the accumulated cost tracking that `log_usage()` is meant to provide.

---

## Category 4 — Fundamental Design Flaws

### D1. `SkillNode` has no concept of ADK session state keys — the engine cannot model inter-agent data flow

`state.py` defines `input_schema` and `output_schema` on `SkillNode`. These map correctly to `FunctionTool` argument signatures but have no analogue for `LlmAgent` nodes that communicate through `ctx.session.state`. The engine conflates two distinct communication models — function arguments and session state — into a single schema field, and no layer of the engine resolves the difference. The result: compiled `LlmAgent` nodes have schemas that describe function signatures no one calls.

A complete data model would add `state_reads: list[str]` and `state_writes: list[str]` to `SkillNode`, populated during decomposition and rendered into both the `instruction` field ("read `user_profile` from session state") and the compiler's session-state wiring logic.

### D2. The engine has no "verify and repair" phase after compilation

After `compiler.compile()` the pipeline sets `events.status = "done"` and returns. There is no step that attempts to import the generated project, run a syntax check, or execute a smoke test. Errors that are deterministically detectable — `SyntaxError`, `IndentationError`, `ImportError` — are left for the user to find manually. A `verify → diagnose → repair` loop (attempt `python -c "import run"`, capture tracebacks, feed them back to `ToolImplementorAgent`) would allow the engine to self-heal the most common class of generation errors.

### D3. `SkillNode` has no `instruction` field — there is no storage for prompt-engineered LLM instructions

`state.py` defines `SkillNode` with `implementation: Optional[str]` for tool bodies, but there is no `instruction: Optional[str]` for LLM agent instructions. The `20260619_solution.md` planned a `PromptEngineerAgent` (P3) but the data model was never extended to hold its output. Without this field, even if a `PromptEngineerAgent` were added, it has nowhere to store its result, and the `adk_llm_agent_stub.py.j2` has no conditional to render it.

### D4. The Decomposer prompt has no ADK-specific rules — structure and semantics must be learned separately

The Decomposer decides `LOOP` vs `SEQUENTIAL` vs `LLM_COORDINATOR` based on generic descriptions with no knowledge of how ADK implements them. The gap between the abstract composition type and the concrete ADK implementation is bridged by the templates, but the templates cannot correct a wrong `composition_type` assignment — they only render what they receive. The correct design is to encode ADK constraints directly in the Decomposer's decision rules: e.g., "use LOOP only if there is a clear, self-contained termination condition that a sub-agent can signal; otherwise use SEQUENTIAL."

### D5. No per-node re-decomposition — the only recovery from a wrong subtree is a full layer rollback

`pipeline.py` provides rollback at the layer granularity only. If a single node was given the wrong `composition_type` or decomposed into the wrong children, the human must rollback all layer-N work and retry the entire layer. For large trees (10+ nodes per layer), this is extremely costly. A targeted "re-decompose this node" action — which clears only the subtree rooted at that node and retries its decomposition with a user-supplied hint — would make recovery cheap.

### D6. The engine generates code without co-generating any test harness

No part of the engine produces tests alongside the generated code. The pipeline generates tool stubs, orchestrators, and an entry point, but no smoke tests (`import atomics.parse_user_response`), no unit test skeletons, and no integration test harness. The first feedback the developer gets about whether the generated project is correct comes from running it in production. A minimal `tests/test_imports.py` generated by the compiler — which only verifies that every generated module imports without error — would catch the `IndentationError` and `ImportError` classes of failure automatically.
