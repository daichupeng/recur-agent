# recur-agent

An LLM-driven platform engine that **recursively decomposes** a high-level software requirement into a granular, human-approved skill tree, then **compiles** that tree into an executable **Google ADK** agent framework.

---

## How It Works

The engine runs one `while True` loop over **depth layers** of the skill tree. Each iteration
processes all non-atomic nodes at the current depth. Two checkpoints inside each iteration
(HITL-1 and HITL-2) can both `continue` back to the top of the loop — same layer, same depth
— causing that layer to be retried from the Decomposer. The outer loop exits only when every
node in the tree is ATOMIC and no deeper layer has any nodes left.

```
[Product Requirement]
        |
        v
  init tree: root = SkillNode(requirement), current_layer = 0
        |
        | <------------------------------ OUTER LOOP (while True) ---------------------------------+
        v                                                                                          |
  get nodes at current_layer depth                                                                 |
        |                                                                                          |
        +-- all ATOMIC? --> any nodes at layer+1? --no--> EXIT LOOP (all layers done)              |
        |                          |                                                               |
        |                         yes                                                              |
        |                          v                                                               |
        |                   current_layer += 1 ------------------------------------------------->+
        |                                                                                          (continue)
        | has non-atomic nodes
        v
  snapshot tree state  <-- taken BEFORE decompose so rollback is correct-by-construction
        |
        v
  +------------------------------------------+
  | Decomposer Agent  (claude-haiku)         |
  |  inject skill_lib context                |
  |  classify each non-atomic node:          |
  |    COMPOSITE -> draft 2-5 sub-skills,    |
  |                 assign composition_type: |
  |                   SEQUENTIAL             |
  |                   PARALLEL               |
  |                   LOOP                   |
  |                   LLM_COORDINATOR        |
  |    ATOMIC    -> assign exec_type:        |
  |                   DETERMINISTIC_CODE     |
  |                   EXTERNAL_API           |
  |                   LLM_PROMPT             |
  |                 (pre-fill from skill_lib  |
  |                  if matching found)      |
  +------------------------------------------+
        |
        v
  +------------------------------------------+
  | Complexity Reviewer  (claude-haiku)      |
  |  scan all nodes at current layer;        |
  |  attach review_note to flagged nodes:    |
  |    OVER-SPLIT, TOO-MANY-CHILDREN,        |
  |    REDUNDANT                             |
  |  (shown as warnings in dashboard)        |
  +------------------------------------------+
        |
        v
  +------------------------------------------+
  | Contract Linter  (pure Python logic)     |
  |  deterministic data-flow wiring check:   |
  |  for each COMPOSITE node, verify that    |
  |  children's declared contracts chain     |
  |  correctly per composition_type:         |
  |    SEQUENTIAL: running state available   |
  |    PARALLEL: disjoint writes, coverage   |
  |    LOOP: shape-stable, termination key   |
  |    LLM_COORDINATOR: each path complete   |
  |  attach contract_note to flagged nodes   |
  |  (violations shown as warnings in UI)    |
  +------------------------------------------+
        |
        v
  ##################################################
  #  HITL-1: Structure Review  (dashboard)         #
  ##################################################
        |
        +-- ROLLBACK? --> restore snapshot -> current_layer unchanged --------------------------->+
        |                 (phantom children gone, no cascading purge needed)                      (continue)
        |
        +-- RE-DECOMPOSE NODE? -------+
        |   (human edits description  |
        |    + optional hint;
        |     if contract frozen:
        |     confirm force_renegotiate) v
        |                     reset that node only:
        |                       children.clear()
        |                       node_type = UNKNOWN
        |                     re-call Decomposer on
        |                     that node with hint
        |                     | (if forced unfreeze: 
        |                     |  re-lint parent group)
        |                             |
        |                     back to HITL-1 <-- INNER LOOP (siblings untouched)
        |
        +-- APPROVE?
        |   -> freeze all node contracts in
        |      approved layer group
        |
        v
  +------------------------------------------+
  | Schema Architect  (claude-haiku, batch=5)|
  |  walk ENTIRE tree for unhydrated ATOMICs |
  |  (not just current layer; catches nodes  |
  |   from any prior approved layer)         |
  |  generate input_schema + output_schema   |
  +------------------------------------------+
        |
        v
  +------------------------------------------+
  | Prompt Engineer  (claude-haiku)          |
  |  for each LLM_PROMPT atomic without      |
  |  instruction yet:                        |
  |    generate full system prompt           |
  |    populate state_reads, state_writes    |
  +------------------------------------------+
        |
        v
  +------------------------------------------+
  | Tool Implementor  (claude-sonnet, 1/call)|
  |  for each DETERMINISTIC_CODE /           |
  |  EXTERNAL_API atomic without impl yet:   |
  |    generate Python function body         |
  |    _normalise_indent                     |
  |    _check_syntax (ast.parse)             |
  |    if SyntaxError: retry up to x2        |
  +------------------------------------------+
        |
        v
  ##################################################
  #  HITL-2: Implementation Review  (dashboard)    #
  ##################################################
        |
        +-- (optional per-skill retry: re-implements
        |    one node, checks contract drift against
        |    frozen declared contract)
        |
        +-- ROLLBACK? --> restore snapshot -> schemas + impls discarded ----------------------->+
        |                 (same layer retried from Decomposer)                                   (continue)
        |
        +-- APPROVE?
        |
        v
  save layer checkpoint; current_layer += 1 ----------------------------------------------------+
                                                                                                  (continue)

  (exit loop when all layers done)
        |
        v
  +------------------------------------------+
  | Memory Architect Agent  (claude-haiku)   |
  |  design persistent-memory contract:      |
  |    triage: collect all PERSISTENT keys   |
  |             (from LLM state_scopes +     |
  |              ancestor contract scopes)   |
  |    design: select backend per entity:    |
  |             KEY_VALUE, APPEND_LOG, or    |
  |             SEMANTIC (embedding-backed)  |
  |    wire: bindings (load/save nodes +     |
  |           derived fields from schemas)   |
  |  produce MemorySpec (structured only)    |
  +------------------------------------------+
        |
        v
  ##################################################
  #  HITL-4: Memory Review (dashboard, skippable) #
  ##################################################
        |
        +-- ROLLBACK? --> re-run MemoryArchitect +
        |                                        |
        +-- APPROVE? ----+
        |                v
        |           continue
        |
        v
  +------------------------------------------+
  | UI Designer Agent  (claude-haiku)        |
  |  select frontend/interaction contract:   |
  |    input affordances                     |
  |      TEXT (mandatory)                    |
  |      FILE_UPLOAD, IMAGE_UPLOAD (opt)     |
  |    output renderers                      |
  |      TEXT, MARKDOWN, TABLE, CODE,        |
  |      IMAGE, FILE_DOWNLOAD (select ≥1)    |
  |    user-facing vs internal agents        |
  |    nodes with binary artifact output     |
  |  produce UISpec (structured data only)   |
  +------------------------------------------+
        |
        v
  ##################################################
  #  HITL-3: UI Review  (dashboard, skippable)     #
  ##################################################
        |
        +-- ROLLBACK? --> re-run UIDesigner -----+
        |                                        |
        +-- APPROVE? ----+
        |                v
        |           continue
        |
        v
  +------------------------------------------+
  | Compiler  (Jinja2, no LLM)               |
  |  normalise names to snake_case           |
  |  scan impls -> collect third-party deps  |
  |  scan impls -> collect required env vars |
  |  copy root .env -> project dir           |
  |  if ui_spec is not None:                 |
  |    compile frontend_index.html.j2        |
  |      -> web/index.html (manifest SPA)    |
  |    compile ui_manifest.json.j2           |
  |      -> web/ui_manifest.json (UI config) |
  |    compile serve.py.j2                   |
  |      -> serve.py (StaticFiles mounted)   |
  |  depth-first recursive walk:             |
  |                                          |
  |  ATOMIC -> LLM_PROMPT                    |
  |    adk_llm_agent_stub.py.j2              |
  |    -> atomics/{name}.py  (LlmAgent)      |
  |    (async if media_types set)            |
  |                                          |
  |  ATOMIC -> DETERMINISTIC_CODE /          |
  |            EXTERNAL_API                  |
  |    adk_tool_stub.py.j2                   |
  |    -> atomics/{name}.py                  |
  |       (FunctionTool + LlmAgent wrapper)  |
  |       (async if media_types set)         |
  |                                          |
  |  COMPOSITE -> SEQUENTIAL                 |
  |    -> orchestrators/{name}.py            |
  |       (SequentialAgent)                  |
  |                                          |
  |  COMPOSITE -> PARALLEL                   |
  |    -> orchestrators/{name}.py            |
  |       (ParallelAgent)                    |
  |                                          |
  |  COMPOSITE -> LOOP                       |
  |    -> orchestrators/{name}.py            |
  |       (LoopAgent, max_iterations=10)     |
  |                                          |
  |  COMPOSITE -> LLM_COORDINATOR            |
  |    -> orchestrators/{name}.py            |
  |       (LlmAgent with routing prompt)     |
  |                                          |
  |  root -> run.py (interactive CLI)        |
  |  root -> agent.py (adk web entry point)  |
  +------------------------------------------+
        |
        v
  +------------------------------------------+
  | Verify-and-Repair  (subprocess, up to x3)|
  |  python -c "import run"                  |
  |  on failure:                             |
  |    +-----------------------------+       |
  |    | Debug Agent (claude-haiku) |       |
  |    | analyze traceback:         |       |
  |    |  - parse error location    |       |
  |    |  - identify root cause     |       |
  |    |  - generate patch code     |       |
  |    +-----------------------------+       |
  |    clear node.implementation             |
  |    apply DebugAgent's patch              |
  |    rewrite file -> retry import          |
  |  after x3 failures: surface in UI status |
  +------------------------------------------+
        |
        v
     import success? ----yes----> +--------------------+
        |                         | output/            |
        no                        | {project_name}/    |
        |                         | (ADK project)      |
        +-- surface error in UI   +--------------------+
             + save debug logs
```

---

## UI Design (HITL-3)

After the entire skill tree is decomposed and implemented, the **UIDesignerAgent** designs the product's frontend by selecting from a fixed interaction catalog:

- **Input affordances**: TEXT (mandatory), FILE_UPLOAD, IMAGE_UPLOAD
- **Output renderers**: TEXT, MARKDOWN (for reports), TABLE (for datasets), CODE, IMAGE, FILE_DOWNLOAD
- **User-facing agents**: which node outputs the user sees; intermediate agents (parsers, fetchers, validators) are marked INTERNAL and their output is hidden
- **Media output**: which nodes emit binary artifacts (images, downloads) and their MIME types

The UIDesigner produces **structured data only** (a `UISpec`); the Compiler uses Jinja2 to turn that into actual HTML/JavaScript:
- `web/index.html` — single-page app with manifest-driven UI
- `web/ui_manifest.json` — UI configuration (affordances, renderers, user-facing agents)
- `serve.py` — FastAPI app that serves both the generated UI **and** the ADK API from one process on port 8080

**HITL-3** (optional, skippable with `SKIP_UI_REVIEW=1`) lets you review the UI design and either approve it or rollback to re-run the designer.

Nodes with media output are re-implemented as `async` functions with a `ToolContext` parameter to call `save_artifact()`. The ADK strips this parameter from the Gemini function declaration so the model never sees it.

---

## Memory Generation (HITL-4)

After the skill tree is fully decomposed and implemented, **MemoryArchitectAgent** designs persistent state storage for any session-state keys marked PERSISTENT across ancestor contracts and LLM node scopes.

### How Memory Scoping Works

When the **Decomposer** classifies a composite node, it declares a `contract.scopes` mapping: which state keys must survive across separate invocations (PERSISTENT) vs. live only for one run (EPHEMERAL). For example, a `coach` agent compositeNode might declare:
```
contract.scopes = {
  "current_training_plan": "PERSISTENT",
  "updated_training_plan": "PERSISTENT",
  "generated_plan": "PERSISTENT",
  ...
}
```

When **PromptEngineer** generates instructions for LLM leaf nodes under that composite, it receives a `persistent_keys_required` hint containing all ancestor PERSISTENT keys. The engineer is instructed to honour these by including them in the node's `state_scopes` when it writes them. This ensures:
- A leaf LLM node under `coach` that writes the plan summary will mark that key PERSISTENT (matching the ancestor's declared scope)
- The MemoryArchitect can later find an LLM producer and wire a **save binding** (not just load-only recall)

### Memory Design: Triage and Wiring

The MemoryArchitectAgent runs in two steps:

1. **Triage** (pure logic): Walk the entire tree and collect every state key marked PERSISTENT (union of LLM node `state_scopes` and ancestor `contract.scopes`). If none, set `tree.memory_spec=None` → skip HITL-4 entirely (no persistence needed).

2. **Design** (one LLM call): For found PERSISTENT keys, ask the LLM to:
   - Select a backend per logical entity (KEY_VALUE, APPEND_LOG, or SEMANTIC)
   - Wire **bindings**: which LLM node writes the source key (save_source_key) and which node loads it (load_target_key)
   - The LLM produces structured MemoryEntity objects only; the Compiler turns these into adapter code

### Memory Backends

| Backend | Use Case | Storage |
|---|---|---|
| `KEY_VALUE` | Per-user config (settings, preferences, one-per-user data) | `{_key: value, field1: val1, ...}` |
| `APPEND_LOG` | Time-series events (history, logs, audit trail) | Rows appended per invocation |
| `SEMANTIC` | High-dimensional recall (embeddings, similarity search) | Uses token-overlap, no external DB |

All backends write to **human-readable Markdown** in `memory_store/<entity>.md` (not SQLite). Atomicity is guaranteed by `os.replace()` + RLock.

### Memory Wiring: Enforcement and Callbacks

The **Compiler** enforces output_key + generated Pydantic schema on every LLM node that writes a PERSISTENT key, and generates deterministic before/after callbacks (not LLM-driven):

1. **before-callback** (before LLM runs): loads prior entity data into a state key so the LLM can read it
2. **after-callback** (after LLM runs): reads the state key written by the LLM's `output_key`, coerces it to a flat record, and persists it

This design keeps tool bodies 100% PURE — memory persistence is not an LLM responsibility.

### Validation and Repair

The pipeline validates memory wiring **before** the debug loop (`_validate_memory_wiring`):
- Every binding's node id resolves
- Save sources have at least one LLM producer (only LLM `output_key` writes reach state)
- Load/save keys are real fields
- Failures report immediately with clear messages; no LLM calls wasted

If a memory adapter fails at runtime (e.g. corruption in memory_store/ file), the DebugAgent re-runs `_compile_memory` to regenerate the adapters from the current spec.

### Ancestor PERSISTENT-Key Propagation

A critical insight: **PERSISTENT scopes are declared on composite nodes, but must be honoured by LLM leaf nodes.**

The fix: Before `PromptEngineer` generates instructions for LLM atomics, the pipeline computes `persistent_keys_hint` — the union of all PERSISTENT keys from every ancestor composite. Each node receives this as `persistent_keys_required` in its JSON input:

```python
# In pipeline.py
persistent_hint = _build_persistent_keys_hint(tree, llm_atomics)
await prompt_engineer.engineer(llm_atomics, persistent_keys_hint=persistent_hint)
```

The PromptEngineer's system prompt is updated to instruct: if a key appears in `persistent_keys_required` AND the node writes it (to match `state_writes` from output_schema), mark it PERSISTENT in `state_scopes` unconditionally. This ensures:
- The MemoryArchitect finds the LLM producer: `_has_capturable_writer` returns True
- A save binding (not load-only) gets wired: the Compiler sets `output_key` on the node
- The after-callback persists the value to storage

### HITL-4

Optional (skippable with `SKIP_MEMORY_REVIEW=1`). The dashboard shows:
- Each designed entity: backend, fields, bindings, retention window
- Deletion scope declaration (REQUIRED for GDPR/privacy compliance)
- "Edit delete_confirmed checkbox" before approval

A rollback re-runs the MemoryArchitectAgent (cheap, single LLM call).

---

## Data-Flow Contracts

**Contract Linter** fixes a critical gap: catching skill-to-skill wiring errors **before code is generated** rather than at runtime.

### Why Contracts Matter

When decomposing a skill into children, you must ensure they form a coherent pipeline:
- A child cannot read a state key no sibling produces.
- Parallel children cannot write the same state key (collision).
- A LOOP body (child) must be shape-stable: same keys in and out.
- A parent cannot promise outputs it never receives from children.

Today these errors only surface when the compiled project fails to run. **Contracts make them visible at HITL-1.**

### How They Work

Each node has a `contract`:
- **reads**: state keys the node consumes (e.g. `{stripe_event: "dict — raw webhook payload"}`)
- **writes**: state keys the node produces (e.g. `{alert_sent: "bool — whether alert succeeded"}`)

The `DecomposerAgent` emits contracts at classification time — for the composite node itself and each proposed child. The **Contract Linter** then performs a deterministic, pure-logic check:

| Composition | Check |
|---|---|
| **SEQUENTIAL** | Child reads must be subset of parent reads + what earlier siblings wrote. Final union of writes must satisfy parent writes. |
| **PARALLEL** | Each child's reads ⊆ parent reads (no inter-sibling data). Children's writes must be disjoint. Union of writes must cover parent writes. |
| **LOOP** | Single child's reads and writes must be identical key set (shape-stable). Must include a termination-condition key (e.g. `is_done`). |
| **LLM_COORDINATOR** | Each child is an independent path: child reads ⊆ parent reads, child writes ⊇ parent writes. |

**Violations** are attached to composite nodes as `contract_note` and shown as red banners in the HITL-1 UI, alongside complexity warnings.

### Contract Freezing

Once a layer is approved at HITL-1, all contracts in that group (the layer's composites and their children) are **frozen**. This prevents a later redecompose from silently breaking the wiring siblings depend on:
- If you try to redecompose a **frozen node**, the UI shows a confirmation dialog warning that you're changing an approved contract your siblings rely on.
- Confirming unlocks the node and its sibling group; the linter re-runs after redecompose.

### Drift Detection (HITL-2)

When a skill is re-implemented at HITL-2, its new signature (from the regenerated schema) is checked against the frozen declared contract:
- If the new schema **drops a promised output**, the frontend shows a **drift warning**.
- If the new schema **adds an unexpected input**, the frontend shows a **drift warning**.
- Drift is a warning, not a blocker — you can approve despite it, or retry the implementation.

---

## Skill Library (skill_lib)

**recur-agent** maintains a cross-project skill library that captures fully-hydrated atomic skills:

After each layer is approved (HITL-2), the pipeline saves every completed atomic node to `skill_lib/{name}.md`:
- Full node metadata (exec_type, input/output schema, implementation)
- Instruction text (for LLM_PROMPT nodes)
- Dependencies and required env vars

On subsequent projects, the **Decomposer** injects skill_lib context into its prompt:
- When decomposing a new node, the model sees matching skills from the library
- Pre-filled children allow the decomposer to skip redundant discovery
- Human can still override; nothing is auto-accepted

**Compilation**: The compiler scans the tree for `skill_lib_ref` markers and copies referenced SKILL.md files into the generated ADK project's `skill_lib/` directory, with a `skill_reader.py` utility to load them.

---

## Atomic Skill Definition

A skill is **atomic** if its core logic can be entirely executed using exactly **one** of:

| Pattern | Description | Example |
|---|---|---|
| `DETERMINISTIC_CODE` | Pure standard code, no external calls | SQL query, regex parse, math formula |
| `EXTERNAL_API` | A single external service call | Stripe charge, Slack webhook, weather fetch |
| `LLM_PROMPT` | A single-turn LLM prompt with no tool loops | Sentiment classification, entity extraction |

Everything else is **composite** and will be further decomposed.

---

## Architecture

### Platform Engine (this repo)
Uses the **Anthropic SDK** for internal agents:
- [src/agents/decomposer.py](src/agents/decomposer.py) — classifies and expands nodes; enforces ADK composition rules in system prompt; injects skill_lib context for pre-filling from reusable library; emits declared data-flow contracts (reads/writes) for every node at classification time
- [src/agents/complexity_reviewer.py](src/agents/complexity_reviewer.py) — flags over-split or structurally odd decompositions
- [src/agents/contract_linter.py](src/agents/contract_linter.py) — pure-logic deterministic wiring check: for each composite, verifies children's contracts chain correctly per composition_type; no auto-fix; surfaces violations to human at HITL-1
- [src/agents/schema_architect.py](src/agents/schema_architect.py) — hydrates I/O JSON Schema for every unhydrated atomic in the tree
- [src/agents/prompt_engineer.py](src/agents/prompt_engineer.py) — writes ADK session-state-aware instructions for LLM_PROMPT nodes
- [src/agents/tool_implementor.py](src/agents/tool_implementor.py) — generates Python function bodies; prefers domain client libraries over raw HTTP; mandates `os.environ` for credentials; uses claude-sonnet-4-6 with syntax validation and retry
- [src/agents/debug.py](src/agents/debug.py) — analyzes failed import traces, auto-repair failures, and generates patch code for broken implementations
- [src/agents/memory_architect.py](src/agents/memory_architect.py) — designs persistent-memory contract: triage to collect PERSISTENT keys, select backends (KEY_VALUE / APPEND_LOG / SEMANTIC), wire save/load bindings. Produces MemorySpec (structured only); Compiler transforms into adapters
- [src/agents/ui_designer.py](src/agents/ui_designer.py) — selects input affordances, output renderers, user-facing agents, and media outputs from a fixed catalog; produces UISpec for Compiler
- [src/agents/compiler.py](src/agents/compiler.py) — tree-recursive Jinja2 compiler; auto-collects third-party deps and required env vars from generated implementations; emits memory adapters (memory/_store.py, memory/<entity>.py, memory/_bindings.py) and before/after callbacks; emits frontend (web/index.html, ui_manifest.json, serve.py) when UISpec is present; copies referenced skill_lib files into output
- [src/orchestrator/pipeline.py](src/orchestrator/pipeline.py) — main async loop with four HITL checkpoints (HITL-1 structure, HITL-2 implementation, HITL-4 memory, HITL-3 UI design), per-node re-decompose with frozen-contract guard and scoped re-lint, contract freeze on HITL-1 approve, drift check on HITL-2 retry, ancestor PERSISTENT-key hint passed to PromptEngineer, memory validation before debug loop, saves fully-hydrated atomics to skill_lib after each approved layer
- [src/ui/server.py](src/ui/server.py) — FastAPI HITL dashboard + sandbox management + credential injection; real-time job status polling + debug logs
- [src/skill_lib.py](src/skill_lib.py) — SkillLib class for reading, writing, and searching reusable atomic skills persisted across projects
- [src/interaction_catalog.py](src/interaction_catalog.py) — fixed catalog of input affordances (TEXT, FILE_UPLOAD, IMAGE_UPLOAD) and output renderers (TEXT, MARKDOWN, TABLE, CODE, IMAGE, FILE_DOWNLOAD) that UIDesigner selects from

### Generated Output (Google ADK)
The compiler emits a fully self-contained **Google ADK** project:
```
output/{project_name}/
├── agent.py                     # adk web entry point (root_agent, sys.path self-injection)
├── run.py                       # interactive CLI entrypoint (InMemoryRunner)
├── pyproject.toml               # google-adk + auto-detected third-party deps
├── .env                         # copied from repo root at compile time (always overwritten)
├── blueprint_raw.json           # tree snapshot after each decomposition round
├── blueprint_verified.json      # tree after HITL-2 approval (schemas + implementations)
├── layers/
│   └── layer_{N}/
│       └── blueprint_verified.json   # per-layer checkpoint
├── atomics/
│   ├── __init__.py
│   ├── {skill_name}.py          # LlmAgent  (LLM_PROMPT)
│   └── {skill_name}.py          # def {name}(...) + FunctionTool + LlmAgent wrapper
│                                #   (DETERMINISTIC_CODE / EXTERNAL_API)
└── orchestrators/
    ├── __init__.py
    └── {composite_name}.py      # SequentialAgent / ParallelAgent / LoopAgent / LlmAgent (coordinator)
```

Every node (atomic or composite) exports a `{name}_agent` symbol. Orchestrators import only their direct children — the tree hierarchy is preserved as a hierarchy of ADK agent wrapping.

---

## Compiler Details

### Dependency auto-detection
After `ToolImplementorAgent` finishes, the compiler scans every `node.implementation` for
`import X` / `from X import` statements at function-body indentation. Third-party packages
(not stdlib, not `google.*`/`dotenv`) are collected and injected into `pyproject.toml` as
additional dependencies. No manual manifest editing required.

### Required env var detection
The compiler also scans implementations for `os.environ["KEY_NAME"]` patterns and stores the
discovered key names on `tree.required_env_vars`. The dashboard uses this list to show a
credential form before the sandbox can be started.

### FunctionTool registration
Tool nodes define the implementation function under its **public name** (`def {name}(...)`).
The `FunctionTool` wrapper is stored in a private `_{name}_tool` variable so that ADK
registers the tool under the same name the `LlmAgent` instruction tells the model to call.

### Gemini model
All generated `LlmAgent` instances use `_DEFAULT_GEMINI_MODEL` from `compiler.py`
(`"gemini-2.0-flash"`). Changing one constant updates every generated project.

### .env propagation
The root `.env` is **always** copied into the generated project directory at compile time,
overwriting any stale copy. The generated `agent.py` calls `load_dotenv()` before importing
the root agent so credentials are available to all `os.environ` lookups at ADK web startup.

---

## Rollback Design

Rollback is **correct-by-construction**. Before the Decomposer runs on layer N, a full snapshot of the tree is saved. Rolling back restores from that snapshot — since the snapshot existed before children were generated, all phantom branches are eliminated atomically. No explicit cascading purge algorithm is needed.

```
snapshot(layer=N, root=tree.root.copy())   <- taken BEFORE decompose
decompose(layer N nodes) -> phantom children created
--- human clicks Rollback ---
tree.root = restore(snapshot(layer=N))     <- phantom children gone
tree.current_layer = N
```

For **per-node re-decompose**, the pipeline does not touch the snapshot. Instead:
1. The targeted node's `children` are cleared and its `node_type` reset to `UNKNOWN`.
2. The Decomposer is called on just that node with an optional human-supplied hint prepended as a user-turn prefix.
3. Sibling nodes are untouched; the layer does not need a full rollback.

---

## Repo Structure

```
recur-agent/
├── main.py                              # CLI / web entrypoint
├── pyproject.toml                       # Platform dependencies
├── .env                                 # ANTHROPIC_API_KEY, GOOGLE_API_KEY, and any
│                                        # service keys used by generated projects
├── skill_lib/                           # Shared skill library (SKILL.md files)
│   └── *.md                             # Persisted atomic skill definitions
├── config/
│   ├── settings.yaml                    # Model, server, output defaults
│   └── agent_profiles.yaml              # Agent descriptions
├── src/
│   ├── orchestrator/
│   │   ├── state.py                     # SkillNode (+ contract + contract_note + state_scopes),
│   │   │                                # SkillTree (+ required_env_vars + skill_lib_ref + memory_spec),
│   │   │                                # Contract (reads/writes/scopes/frozen), MemorySpec, LayerSnapshot, rollback()
│   │   └── pipeline.py                  # Main async loop (snapshot -> decompose ->
│   │                                    #   complexity-review -> contract-lint -> HITL-1 -> schema ->
│   │                                    #   prompt-engineer (with ancestor PERSISTENT hints) -> tool-implement -> HITL-2 ->
│   │                                    #   memory-architect -> HITL-4 -> ui-designer -> HITL-3 ->
│   │                                    #   compile (with memory callbacks + output_key enforcement) -> verify-repair); 
│   │                                    #   saves atomics to skill_lib; validates memory wiring before debug
│   ├── agents/
│   │   ├── base_agent.py                # Anthropic SDK wrapper + retry + token tracking
│   │   ├── decomposer.py                # DecomposerAgent (ADK constraints + skill_lib + contract emission + scope tagging)
│   │   ├── complexity_reviewer.py       # ComplexityReviewAgent
│   │   ├── contract_linter.py           # ContractLinterAgent (pure-logic wiring check, optional LLM repair)
│   │   ├── schema_architect.py          # SchemaArchitectAgent (batches of 5)
│   │   ├── prompt_engineer.py           # PromptEngineerAgent (state_reads / state_writes / state_scopes; honors ancestor PERSISTENT hints)
│   │   ├── tool_implementor.py          # ToolImplementorAgent (sonnet, 1 node/call, syntax retry)
│   │   ├── memory_architect.py          # MemoryArchitectAgent (triage PERSISTENT keys, select backends, wire bindings)
│   │   ├── debug.py                     # DebugAgent (analyze import failures, auto-repair, memory adapter regeneration)
│   │   ├── ui_designer.py               # UIDesignerAgent (catalog selection, UISpec generation)
│   │   └── compiler.py                  # CompilerAgent (Jinja2 walk, dep/env/memory/UI scan, skill_lib copy, enforces output_key + schema on LLM producers)
│   ├── ui/
│   │   ├── server.py                    # FastAPI HITL dashboard + sandbox + job polling + debug logs
│   │   └── templates/
│   │       ├── index.html               # Landing page
│   │       ├── dashboard.html           # HITL-1 + HITL-2 review UI + credential form + debug panel
│   │       └── chat.html                # Embedded chat for testing generated agents
│   ├── skill_lib.py                     # SkillLib: read/write/search reusable skill definitions
│   ├── interaction_catalog.py           # Fixed catalog of input affordances + output renderers
│   ├── memory_catalog.py                # Fixed catalog of memory backends: KEY_VALUE, APPEND_LOG, SEMANTIC
│   └── compiler_templates/
│       ├── adk_tool_stub.py.j2          # def {name}() + FunctionTool + LlmAgent wrapper (+ memory callbacks)
│       ├── adk_llm_agent_stub.py.j2     # LlmAgent with instruction + state wiring + output_key + output_schema (+ memory callbacks)
│       ├── adk_sequential_agent.py.j2   # SequentialAgent orchestrator (+ memory callbacks)
│       ├── adk_parallel_agent.py.j2     # ParallelAgent orchestrator (+ memory callbacks)
│       ├── adk_loop_agent.py.j2         # LoopAgent orchestrator max_iterations=10 (+ memory callbacks)
│       ├── adk_coordinator_agent.py.j2  # LlmAgent coordinator with routing instruction (+ memory callbacks)
│       ├── adk_root_orchestrator.py.j2  # run.py (interactive CLI)
│       ├── frontend_index.html.j2       # Landing page template for UI feature
│       ├── serve.py.j2                  # Serve.py template for static web serving (initializes memory via init_memory())
│       ├── ui_manifest.json.j2          # UI manifest template
│       ├── memory_store.py.j2            # Shared MarkdownStore class (atomic writes via os.replace + RLock)
│       ├── memory_keyvalue.py.j2         # KEY_VALUE adapter (set by key / get by key)
│       ├── memory_appendlog.py.j2        # APPEND_LOG adapter (append row / query all)
│       ├── memory_semantic.py.j2         # SEMANTIC adapter (add by embedding / query by similarity)
│       ├── memory_init.py.j2             # init_memory() function (called before agents load)
│       ├── memory_bindings.py.j2         # BEFORE_CALLBACKS / AFTER_CALLBACKS (wired to agents)
│       ├── dotignore.j2                  # .gitignore + .dockerignore (blocks memory_store/)
│       └── adk_pyproject.toml.j2        # Target project manifest (auto-deps injected)
├── memory_catalog.py                    # Built-in backend catalog for MemoryArchitect
├── memory.md                            # (generated by memory_architect) — memory blueprint snapshot
└── output/
    └── {project_name}/                  # Generated ADK project
        ├── memory/                      # (if memory_spec is not None)
        │   ├── _store.py                # MarkdownStore class (shared)
        │   ├── _bindings.py             # BEFORE_CALLBACKS / AFTER_CALLBACKS per node
        │   ├── _init.py                 # init_memory() + clear_<entity>() registry
        │   ├── {entity_name}.py         # MemoryAdapter subclass (KEY_VALUE / APPEND_LOG / SEMANTIC)
        │   └── __init__.py
        └── memory_store/                # (in .gitignore / .dockerignore)
            └── {entity_name}.md         # Markdown table (one per entity, human-readable)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- `uv` package manager
- `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` in `.env`

### Install

```bash
uv sync
```

### Run

```bash
python main.py \
  --requirement "Build a Slack bot that alerts on Stripe payment failures" \
  --project-name stripe_alerter
```

Then open **http://127.0.0.1:8000** to review and approve each decomposition layer.

### Options

| Flag | Default | Description |
|---|---|---|
| `--requirement` / `-r` | required | High-level product requirement |
| `--project-name` / `-p` | required | Output project name (Python identifier) |
| `--output-dir` | `./output` | Where to write the generated project |
| `--host` | `127.0.0.1` | Dashboard host |
| `--port` | `8000` | Dashboard port |

---

## HITL Workflow

### HITL-1 — Structure Review

The pipeline pauses after each decomposition layer. Open the dashboard to:

1. Review node cards — each shows name, type badge, exec_type, composition_type, and children
2. Read advisory notes:
   - Orange `review_note` badge (Complexity Reviewer): warnings about over-decomposition or structural oddities
   - Red `contract_note` badge (Contract Linter): data-flow wiring violations (reads/writes mismatches)
3. **View contracts** on composite nodes (when declared):
   - A "Data-flow contract" panel shows the parent node's promised **reads** (green) and **writes** (amber)
   - Each child's row shows its own **reads** (↓ keys) and **writes** (↑ keys) as inline chips
   - A **FROZEN** badge appears once the layer is approved (contracts are locked)
4. **Edit inline**: click any name or description and type to override (no LLM re-run)
5. Click **"Save edits"** on a card to persist changes
6. Click **"Re-decompose"** on a composite node to retry only that node's subtree with your edits as context:
   - Normal nodes: re-decompose immediately
   - Frozen nodes: confirm that you want to change the contract approved siblings depend on
7. Click **"Approve Layer & Advance"** to freeze all contracts in this layer, then run Schema Architect, Prompt Engineer, and Tool Implementor, then proceed to HITL-2
8. Click **"Rollback This Layer"** to discard all LLM-generated children for this layer and retry from scratch

### HITL-2 — Implementation Review

After schemas, instructions, and function bodies are generated, the pipeline pauses again:

1. Review the generated schemas and function bodies per atomic node
2. Edit any function body or schema inline if needed
3. (Optional) Click **"Retry with feedback"** on an atomic node to re-implement only that node:
   - After regeneration, a drift check compares the frozen declared contract against the new schema
   - Any dropped outputs or unexpected new inputs are flagged as a contract-drift warning
4. Click **"Approve Implementation"** to compile the full ADK project
5. Click **"Rollback Implementation"** to discard this layer entirely and retry from decompose

---

## Sandbox

After compilation completes, the **Sandbox** tab becomes available. It installs the generated
project's dependencies (`uv sync`) and launches `adk web .` — the Google ADK chat UI — at
`http://localhost:7860`.

### Credential form

If the generated project requires API keys (detected from `os.environ["KEY_NAME"]` calls in
the implementation), the dashboard shows a credential form **before** the sandbox can start:

- Keys already present in the project's `.env` appear as `✓ satisfied`.
- Missing keys are shown as required password inputs.
- Submitting the form writes the values to `.env` on the server. Values are never echoed
  back after submission.
- **"Run in Sandbox" is disabled until all required keys are satisfied.**

Keys you need for your project should be added to the root `.env` before compiling so they
are copied automatically. For keys only known after compile time, enter them in the
credential form.

### Sandbox API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/sandbox/start` | Start the sandbox for the current compiled project |
| `POST` | `/sandbox/stop` | Terminate the running sandbox |
| `GET` | `/sandbox/status` | `{status, port, ready_url}` |
| `GET` | `/sandbox/logs` | SSE stream of setup + adk web stdout |
| `GET` | `/sandbox/required_env` | `{required, satisfied, missing}` key lists |
| `POST` | `/sandbox/env` | Write `{env_vars: {KEY: value}}` into the project `.env` |

---

## Agent Model Selection

| Agent | Model | Rationale |
|---|---|---|
| `DecomposerAgent` | `claude-haiku-4-5` (default) | High-throughput classification; skill_lib injection; many calls per run |
| `ComplexityReviewAgent` | `claude-haiku-4-5` (default) | Light scan; speed over depth |
| `SchemaArchitectAgent` | `claude-haiku-4-5` (default) | Structured JSON output; fast |
| `PromptEngineerAgent` | `claude-haiku-4-5` (default) | Instruction writing; honours ancestor PERSISTENT-key hints |
| `ToolImplementorAgent` | `claude-sonnet-4-6` | Code generation requires highest quality; 1 node/call |
| `MemoryArchitectAgent` | `claude-haiku-4-5` (default) | Memory backend selection; runs once after tree is hydrated+implemented |
| `DebugAgent` | `claude-haiku-4-5` (default) | Analyze import failures; generate patches; regenerate memory adapters on adapter errors |
| `UIDesignerAgent` | `claude-haiku-4-5` (default) | Select UI affordances from catalog; structured output (UISpec); runs once after memory design |

---

## Tool Implementation Rules

`ToolImplementorAgent` follows this hierarchy for `EXTERNAL_API` nodes:

1. **Use a domain client library** if one exists (e.g. `yfinance`, `stripe`, `twilio`, `boto3`, `newsapi-python`). Import it inside the function body.
2. **Fall back to `httpx`** only when no stable client exists. Use only documented, stable endpoints — never undocumented internal paths (e.g. paths with `/v7/`, `/v8/` slugs and no public spec).
3. **Credentials via `os.environ`** — `os.environ["KEY_NAME"]` always. No literal placeholder strings. A missing key raises `KeyError` immediately on first call rather than silently sending wrong auth headers.

For `DETERMINISTIC_CODE` nodes, only Python stdlib is used; all imports are inside the function body.
