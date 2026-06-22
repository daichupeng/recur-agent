# recur-agent

An LLM-driven platform engine that **recursively decomposes** a high-level software requirement into a granular, human-approved skill tree, then **compiles** that tree into an executable **Google ADK** agent framework.

---

## How It Works

```
[Product Requirement]
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 1. Decomposer Agent  (claude-haiku)                                  │◄─────────────────────────────────┐
│    • Classifies each node as COMPOSITE or ATOMIC                     │                                  │
│    • For COMPOSITE → drafts 2–5 immediate sub-skills                 │                                  │
│    • Assigns composition_type:                                        │                                  │
│        SEQUENTIAL   — children run in order, state passed along      │                                  │
│        PARALLEL     — children run concurrently (no data dependency) │                                  │
│        LOOP         — single child repeated until escalation signal  │                                  │
│        LLM_COORDINATOR — LlmAgent routes to children by context      │                                  │
│    • For ATOMIC → assigns exec_type (no further split):              │                                  │
│        DETERMINISTIC_CODE — pure stdlib logic                        │                                  │
│        EXTERNAL_API       — single external service call             │                                  │
│        LLM_PROMPT         — single-turn LLM call                    │                                  │
└──────────────────────────────────────────────────────────────────────┘                                  │
        │                                                                                                  │   Recursion:
        ▼                                                                                                  │   snapshot → decompose
┌──────────────────────────────────────────────────────────────────────┐                                  │   → review → HITL-1
│ 1b. Complexity Reviewer  (claude-haiku)                              │                                  │   loop until all nodes
│    • Scans all nodes at the current layer                            │                                  │   at this depth are
│    • Flags structural anomalies with advisory notes:                 │                                  │   ATOMIC
│        OVER-SPLIT — node could be one atomic skill                   │                                  │
│        TOO-MANY-CHILDREN — composite has > 5 children                │                                  │
│        SHALLOW-LAYER — entire layer is atomics (nothing to do)       │                                  │
│        REDUNDANT — duplicates sibling or ancestor scope              │                                  │
│    • Notes are surfaced as warnings in the HITL dashboard            │                                  │
└──────────────────────────────────────────────────────────────────────┘                                  │
        │                                                                                                  │
        ▼                                                                                                  │
┌──────────────────────────────────────────────────────────────────────┐                                  │
│ 2. HITL-1 Dashboard  (FastAPI)           [Approval Checkpoint]       │                                  │
│    • Human reviews the current decomposition layer                   │                                  │
│    • Three actions per review:                                        │                                  │
│                                                                        │                                  │
│      ① APPROVE LAYER → advance to schema hydration                   │                                  │
│                                                                        │                                  │
│      ② ROLLBACK LAYER → restore pre-decompose snapshot;              │──────────────────────────────────┘
│                          all LLM-generated children purged atomically │
│                          (no cascading purge needed)                  │
│                                                                        │
│      ③ RE-DECOMPOSE NODE → human edits a single node's description   │──┐
│                             (or types a hint), pipeline re-decomposes │  │ Per-node retry:
│                             only that subtree; siblings untouched     │  │ reset node, re-call
│                             (no full-layer rollback required)         │◄─┘ Decomposer with hint
└──────────────────────────────────────────────────────────────────────┘
        │ (all nodes at current depth approved)
        ▼ (repeat from step 1 for the next depth layer)
        │
        │ (all depths fully decomposed — every leaf is ATOMIC)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 3. Schema Architect  (claude-haiku, batches of 5)                    │
│    • Walks the entire tree for unhydrated ATOMIC nodes               │
│      (not just the current layer — catches deeply-buried atomics     │
│       from any prior layer)                                          │
│    • Defines JSON Schema (input_schema + output_schema) per node     │
│    • Schemas drive function signatures and LLM instructions          │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 3b. Prompt Engineer  (claude-haiku)                                  │
│    • Runs on every LLM_PROMPT atomic that has no instruction yet     │
│    • Generates a full system prompt (instruction) per node:          │
│        – States the node's objective precisely                       │
│        – Lists which ADK session-state keys to READ at turn start    │
│        – Lists which ADK session-state keys to WRITE before return   │
│    • Populates state_reads / state_writes lists on each node         │
│    • Used later by adk_llm_agent_stub.py.j2 to wire state correctly  │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 3c. Tool Implementor  (claude-sonnet-4-6, one node per call)         │
│    • Runs on every DETERMINISTIC_CODE and EXTERNAL_API atomic        │
│      that has no implementation yet                                  │
│    • Generates the Python function body (lines inside the def):      │
│        DETERMINISTIC_CODE — full logic, stdlib only                  │
│        EXTERNAL_API       — httpx call with # CONFIGURE: placeholders│
│    • Returns only the indented function body (no def line)           │
│    • Post-processes output:                                          │
│        – _normalise_indent: anchors on first non-empty line's        │
│          indent level, re-maps to 4-space base                       │
│        – _check_syntax: wraps in dummy def, runs ast.parse()         │
│        – _retry_until_valid: re-prompts with error up to 2×          │
│          if syntax check fails                                        │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 4. HITL-2 Dashboard  (FastAPI)           [Approval Checkpoint]       │
│    • Human reviews generated schemas and function bodies             │
│    • APPROVE → proceed to compilation                                │
│    • ROLLBACK → discard entire layer and retry from decompose        │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 5. Compiler  (Jinja2 templates, no LLM)                              │
│    • Normalises all node names to snake_case Python identifiers      │
│    • Depth-first recursive walk of the tree (leaves first):          │
│                                                                        │
│      ATOMIC → LLM_PROMPT:                                            │
│        renders adk_llm_agent_stub.py.j2                              │
│        → atomics/{name}.py  (LlmAgent with instruction from node)   │
│                                                                        │
│      ATOMIC → DETERMINISTIC_CODE / EXTERNAL_API:                    │
│        renders adk_tool_stub.py.j2                                   │
│        → atomics/{name}.py  (FunctionTool + thin LlmAgent wrapper)  │
│                                                                        │
│      COMPOSITE → SEQUENTIAL:                                          │
│        renders adk_sequential_agent.py.j2                            │
│        → orchestrators/{name}.py  (SequentialAgent)                 │
│                                                                        │
│      COMPOSITE → PARALLEL:                                           │
│        renders adk_parallel_agent.py.j2                              │
│        → orchestrators/{name}.py  (ParallelAgent)                   │
│                                                                        │
│      COMPOSITE → LOOP:                                               │
│        renders adk_loop_agent.py.j2                                  │
│        → orchestrators/{name}.py  (LoopAgent, max_iterations=10)    │
│                                                                        │
│      COMPOSITE → LLM_COORDINATOR:                                    │
│        renders adk_coordinator_agent.py.j2                           │
│        → orchestrators/{name}.py  (LlmAgent routing to sub_agents)  │
│                                                                        │
│    • Every compiled node exports a {name}_agent symbol               │
│    • Orchestrators import only their direct children                 │
│    • Root orchestrator imported by run.py (interactive CLI entrypoint│
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 6. Verify-and-Repair Loop  (subprocess + ToolImplementorAgent)       │
│    • Runs: python -c "import run" in the generated project dir       │
│    • On failure: parses traceback for  atomics/<name>.py  reference  │
│    • Finds the corresponding SkillNode, clears implementation,       │
│      re-calls ToolImplementorAgent for that node only                │
│    • Rewrites the file and retries import (up to 3 iterations)       │
│    • If still failing: surfaces error in events.status for the UI    │
│      rather than silently returning a broken project                 │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
 [output/{project_name}/]  ← executable Google ADK project
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
- [src/agents/tool_implementor.py](src/agents/tool_implementor.py) — generates Python function bodies; uses claude-sonnet-4-6 with syntax validation and retry
- [src/agents/compiler.py](src/agents/compiler.py) — tree-recursive Jinja2 compiler; depth-first walk, leaves compiled first
- [src/orchestrator/pipeline.py](src/orchestrator/pipeline.py) — main async loop with two HITL checkpoints and per-node re-decompose
- [src/ui/server.py](src/ui/server.py) — FastAPI HITL dashboard

### Generated Output (Google ADK)
The compiler emits a fully self-contained **Google ADK** project:
```
output/{project_name}/
├── run.py                       # interactive CLI entrypoint (InMemoryRunner)
├── pyproject.toml               # google-adk dependency
├── blueprint_raw.json           # Tree snapshot after each decomposition round
├── blueprint_verified.json      # Tree after HITL-2 approval (schemas + implementations)
├── layers/
│   └── layer_{N}/
│       └── blueprint_verified.json   # Per-layer checkpoint
├── atomics/
│   ├── __init__.py
│   ├── {skill_name}.py          # LlmAgent definition (LLM_PROMPT)
│   └── {skill_name}.py          # FunctionTool + LlmAgent wrapper (DETERMINISTIC_CODE / EXTERNAL_API)
└── orchestrators/
    ├── __init__.py
    └── {composite_name}.py      # SequentialAgent / ParallelAgent / LoopAgent / LlmAgent (coordinator)
```

Every node (atomic or composite) exports a `{name}_agent` symbol. Orchestrators import only their direct children — the tree hierarchy is preserved as a hierarchy of ADK agent wrapping.

---

## Rollback Design

Rollback is **correct-by-construction**. Before the Decomposer runs on layer N, a full snapshot of the tree is saved. Rolling back restores from that snapshot — since the snapshot existed before children were generated, all phantom branches are eliminated atomically. No explicit cascading purge algorithm is needed.

```
snapshot(layer=N, root=tree.root.copy())   ← taken BEFORE decompose
decompose(layer N nodes) → phantom children created
─── human clicks Rollback ───
tree.root = restore(snapshot(layer=N))     ← phantom children gone
tree.current_layer = N
```

For **per-node re-decompose** (added in the 2026-06-21 update), the pipeline does not use the snapshot. Instead:
1. The targeted node's `children` are cleared and its `node_type` reset to `UNKNOWN`.
2. The Decomposer is called on just that node with an optional human-supplied hint prepended as a user-turn prefix.
3. Sibling nodes are untouched; the layer does not need a full rollback.

---

## Repo Structure

```
recur-agent/
├── main.py                              # CLI / web entrypoint
├── pyproject.toml                       # Platform dependencies
├── config/
│   ├── settings.yaml                    # Model, server, output defaults
│   └── agent_profiles.yaml              # Agent descriptions
├── src/
│   ├── orchestrator/
│   │   ├── state.py                     # SkillNode, SkillTree, LayerSnapshot, rollback()
│   │   └── pipeline.py                  # Main async loop (snapshot → decompose →
│   │                                    #   complexity-review → HITL-1 → schema →
│   │                                    #   prompt-engineer → tool-implement → HITL-2 →
│   │                                    #   compile → verify-repair)
│   ├── agents/
│   │   ├── base_agent.py                # Anthropic SDK wrapper + retry + token tracking
│   │   ├── decomposer.py                # DecomposerAgent (ADK constraints baked into prompt)
│   │   ├── complexity_reviewer.py       # ComplexityReviewAgent
│   │   ├── schema_architect.py          # SchemaArchitectAgent (batches of 5)
│   │   ├── prompt_engineer.py           # PromptEngineerAgent (state_reads / state_writes)
│   │   ├── tool_implementor.py          # ToolImplementorAgent (sonnet, 1 node/call, syntax retry)
│   │   └── compiler.py                  # CompilerAgent (Jinja2 tree-recursive walk)
│   ├── ui/
│   │   ├── server.py                    # FastAPI HITL dashboard
│   │   └── templates/
│   │       ├── index.html               # Landing page
│   │       ├── dashboard.html           # HITL-1 + HITL-2 review UI
│   │       └── chat.html                # Embedded chat for testing generated agents
│   └── compiler_templates/
│       ├── adk_tool_stub.py.j2          # FunctionTool + LlmAgent wrapper
│       ├── adk_llm_agent_stub.py.j2     # LlmAgent with instruction + state wiring
│       ├── adk_sequential_agent.py.j2   # SequentialAgent orchestrator
│       ├── adk_parallel_agent.py.j2     # ParallelAgent orchestrator
│       ├── adk_loop_agent.py.j2         # LoopAgent orchestrator (max_iterations=10)
│       ├── adk_coordinator_agent.py.j2  # LlmAgent coordinator with routing instruction
│       ├── adk_root_orchestrator.py.j2  # run.py (interactive CLI)
│       └── adk_pyproject.toml.j2        # Target project manifest
├── improvement_history/
│   ├── 20260619_solution.md             # P1/P2 fixes: compiler rewrite, tool implementor, etc.
│   └── 20260621_solution.md             # P1 fixes: ADK constraints, HITL-2, verify-repair, etc.
└── output/
    └── {project_name}/                  # Generated ADK project
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- `uv` package manager
- `ANTHROPIC_API_KEY` environment variable set

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

## Agent Model Selection

| Agent | Model | Rationale |
|---|---|---|
| `DecomposerAgent` | `claude-haiku-4-5` (default) | High-throughput classification; many calls per run |
| `ComplexityReviewAgent` | `claude-haiku-4-5` (default) | Light scan; speed over depth |
| `SchemaArchitectAgent` | `claude-haiku-4-5` (default) | Structured JSON output; fast |
| `PromptEngineerAgent` | `claude-haiku-4-5` (default) | Instruction writing; moderate complexity |
| `ToolImplementorAgent` | `claude-sonnet-4-6` | Code generation requires highest quality; 1 node/call |

---

## MVP Scope

The MVP focuses on **generation and structural compilation** only:
- The platform does **not** dynamically execute the generated target code
- Sandbox wiring (GKE/gVisor) is declared as comments in generated stubs — implementation is post-MVP
- The skill tree is a **tree** (not a DAG); duplicate atomics across branches are allowed and generate separate stubs
- `EXTERNAL_API` stubs contain `# CONFIGURE: ...` placeholders for base URLs and API keys that the developer must fill in
