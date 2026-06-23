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
  ##################################################
  #  HITL-1: Structure Review  (dashboard)         #
  ##################################################
        |
        +-- ROLLBACK? --> restore snapshot -> current_layer unchanged --------------------------->+
        |                 (phantom children gone, no cascading purge needed)                      (continue)
        |
        +-- RE-DECOMPOSE NODE? -------+
        |   (human edits description  |
        |    + optional hint)         v
        |                     reset that node only:
        |                       children.clear()
        |                       node_type = UNKNOWN
        |                     re-call Decomposer on
        |                     that node with hint
        |                             |
        |                     back to HITL-1 <-- INNER LOOP (siblings untouched)
        |
        +-- APPROVE?
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
  | Compiler  (Jinja2, no LLM)               |
  |  normalise names to snake_case           |
  |  scan impls -> collect third-party deps  |
  |  scan impls -> collect required env vars |
  |  copy root .env -> project dir           |
  |  depth-first recursive walk:             |
  |                                          |
  |  ATOMIC -> LLM_PROMPT                    |
  |    adk_llm_agent_stub.py.j2              |
  |    -> atomics/{name}.py  (LlmAgent)      |
  |                                          |
  |  ATOMIC -> DETERMINISTIC_CODE /          |
  |            EXTERNAL_API                  |
  |    adk_tool_stub.py.j2                   |
  |    -> atomics/{name}.py                  |
  |       (FunctionTool + LlmAgent wrapper)  |
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
  +------------------------------------------+    import     +--------------------+
  | Verify-and-Repair  (subprocess, up to x3)|   passes  -> | output/             |
  |  python -c "import run"                  |              | {project_name}/     |
  |  on failure:                             |              | (ADK project)       |
  |    parse traceback -> atomics/{x}.py     |              +---------------------+
  |    clear node.implementation             |
  |    re-call Tool Implementor (that node)  |
  |    rewrite file -> retry import          |
  |  after x3 failures: surface in UI status |
  +------------------------------------------+
```

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
- [src/agents/decomposer.py](src/agents/decomposer.py) — classifies and expands nodes; enforces ADK composition rules in system prompt
- [src/agents/complexity_reviewer.py](src/agents/complexity_reviewer.py) — flags over-split or structurally odd decompositions
- [src/agents/schema_architect.py](src/agents/schema_architect.py) — hydrates I/O JSON Schema for every unhydrated atomic in the tree
- [src/agents/prompt_engineer.py](src/agents/prompt_engineer.py) — writes ADK session-state-aware instructions for LLM_PROMPT nodes
- [src/agents/tool_implementor.py](src/agents/tool_implementor.py) — generates Python function bodies; prefers domain client libraries over raw HTTP; mandates `os.environ` for credentials; uses claude-sonnet-4-6 with syntax validation and retry
- [src/agents/compiler.py](src/agents/compiler.py) — tree-recursive Jinja2 compiler; auto-collects third-party deps and required env vars from generated implementations
- [src/orchestrator/pipeline.py](src/orchestrator/pipeline.py) — main async loop with two HITL checkpoints and per-node re-decompose
- [src/ui/server.py](src/ui/server.py) — FastAPI HITL dashboard + sandbox management + credential injection

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
├── config/
│   ├── settings.yaml                    # Model, server, output defaults
│   └── agent_profiles.yaml              # Agent descriptions
├── src/
│   ├── orchestrator/
│   │   ├── state.py                     # SkillNode, SkillTree (+ required_env_vars),
│   │   │                                # LayerSnapshot, rollback()
│   │   └── pipeline.py                  # Main async loop (snapshot -> decompose ->
│   │                                    #   complexity-review -> HITL-1 -> schema ->
│   │                                    #   prompt-engineer -> tool-implement -> HITL-2 ->
│   │                                    #   compile -> verify-repair)
│   ├── agents/
│   │   ├── base_agent.py                # Anthropic SDK wrapper + retry + token tracking
│   │   ├── decomposer.py                # DecomposerAgent (ADK constraints baked into prompt)
│   │   ├── complexity_reviewer.py       # ComplexityReviewAgent
│   │   ├── schema_architect.py          # SchemaArchitectAgent (batches of 5)
│   │   ├── prompt_engineer.py           # PromptEngineerAgent (state_reads / state_writes)
│   │   ├── tool_implementor.py          # ToolImplementorAgent (sonnet, 1 node/call, syntax retry)
│   │   └── compiler.py                  # CompilerAgent (Jinja2 tree-recursive walk,
│   │                                    #   dep scan, env-var scan, .env copy)
│   ├── ui/
│   │   ├── server.py                    # FastAPI HITL dashboard + sandbox endpoints
│   │   └── templates/
│   │       ├── index.html               # Landing page
│   │       ├── dashboard.html           # HITL-1 + HITL-2 review UI + credential form
│   │       └── chat.html                # Embedded chat for testing generated agents
│   └── compiler_templates/
│       ├── adk_tool_stub.py.j2          # def {name}() + FunctionTool + LlmAgent wrapper
│       ├── adk_llm_agent_stub.py.j2     # LlmAgent with instruction + state wiring
│       ├── adk_sequential_agent.py.j2   # SequentialAgent orchestrator
│       ├── adk_parallel_agent.py.j2     # ParallelAgent orchestrator
│       ├── adk_loop_agent.py.j2         # LoopAgent orchestrator (max_iterations=10)
│       ├── adk_coordinator_agent.py.j2  # LlmAgent coordinator with routing instruction
│       ├── adk_root_orchestrator.py.j2  # run.py (interactive CLI)
│       └── adk_pyproject.toml.j2        # Target project manifest (auto-deps injected)
└── output/
    └── {project_name}/                  # Generated ADK project
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
2. Read `review_note` badges (orange) if the Complexity Reviewer flagged a node
3. **Edit inline**: click any name or description and type to override (no LLM re-run)
4. Click **"Save edits"** on a card to persist changes
5. Click **"Re-decompose"** on a node to have the pipeline retry _only that node_'s subtree with your edits as context
6. Click **"Approve Layer & Advance"** to run Schema Architect, Prompt Engineer, and Tool Implementor, then proceed to HITL-2
7. Click **"Rollback This Layer"** to discard all LLM-generated children for this layer and retry from scratch

### HITL-2 — Implementation Review

After schemas, instructions, and function bodies are generated, the pipeline pauses again:

1. Review the generated schemas and function bodies per atomic node
2. Edit any function body or schema inline if needed
3. Click **"Approve Implementation"** to compile the full ADK project
4. Click **"Rollback Implementation"** to discard this layer entirely and retry from decompose

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
| `DecomposerAgent` | `claude-haiku-4-5` (default) | High-throughput classification; many calls per run |
| `ComplexityReviewAgent` | `claude-haiku-4-5` (default) | Light scan; speed over depth |
| `SchemaArchitectAgent` | `claude-haiku-4-5` (default) | Structured JSON output; fast |
| `PromptEngineerAgent` | `claude-haiku-4-5` (default) | Instruction writing; moderate complexity |
| `ToolImplementorAgent` | `claude-sonnet-4-6` | Code generation requires highest quality; 1 node/call |

---

## Tool Implementation Rules

`ToolImplementorAgent` follows this hierarchy for `EXTERNAL_API` nodes:

1. **Use a domain client library** if one exists (e.g. `yfinance`, `stripe`, `twilio`, `boto3`, `newsapi-python`). Import it inside the function body.
2. **Fall back to `httpx`** only when no stable client exists. Use only documented, stable endpoints — never undocumented internal paths (e.g. paths with `/v7/`, `/v8/` slugs and no public spec).
3. **Credentials via `os.environ`** — `os.environ["KEY_NAME"]` always. No literal placeholder strings. A missing key raises `KeyError` immediately on first call rather than silently sending wrong auth headers.

For `DETERMINISTIC_CODE` nodes, only Python stdlib is used; all imports are inside the function body.
