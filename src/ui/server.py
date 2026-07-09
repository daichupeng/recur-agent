"""FastAPI HITL dashboard server."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.orchestrator.pipeline import PipelineEvents, RedecomposeRequest, RetrySkillRequest
from src.orchestrator.state import SkillNode, SkillTree

_TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="recur-agent HITL Dashboard")
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Injected at startup by main.py
_events: PipelineEvents | None = None
# Callback set by main.py to kick off a new pipeline run from a web request
_start_callback = None

# ── Sandbox state ──────────────────────────────────────────────────────────
_SANDBOX_PORT = 7860
_sandbox: dict = {
    "status": "idle",   # idle | setting_up | ready | error
    "port": _SANDBOX_PORT,
    "proc": None,       # subprocess.Popen | None
    "output": [],       # list[str] — accumulated log lines
    "notify": None,     # asyncio.Event — set when new log line added
    "mode": None,       # "serve" (generated frontend) | "adk_web" (legacy) | None
}

# ── Debug env-var gate ─────────────────────────────────────────────────────
# The debug loop calls `debug_env_provider(missing_names)` which suspends until
# the frontend POSTs to /debug/env.  A fresh gate is created per request.
_debug_env_gate: dict = {
    "pending": [],      # list[str] — missing var names currently being requested
    "provided": {},     # dict[str, str] — values submitted by the UI
    "event": None,      # asyncio.Event — set when /debug/env is posted
}


def _sandbox_log(line: str) -> None:
    _sandbox["output"].append(line)
    if _sandbox["notify"]:
        _sandbox["notify"].set()


async def _stream_subprocess(cmd: list[str], cwd: Path, env: dict | None = None) -> int:
    """Run a subprocess, stream output to sandbox log. Returns exit code."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env if env is not None else {**os.environ},
    )
    assert proc.stdout is not None
    async for raw in proc.stdout:
        _sandbox_log(raw.decode(errors="replace").rstrip())
    return await proc.wait()


async def _run_sandbox(project_path: Path, port: int) -> None:
    _sandbox["status"] = "setting_up"
    _sandbox["output"] = []
    _sandbox["proc"] = None
    _sandbox["port"] = port
    _sandbox["notify"] = asyncio.Event()

    _sandbox_log(f"▶ Setting up sandbox for {project_path.name}…")

    venv_dir = project_path / ".venv"

    # Pin the venv location so uv ignores any ambient VIRTUAL_ENV in the shell.
    uv_env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv_dir)}

    # Install generated project dependencies. Use --verbose so uv emits lines
    # while downloading (without a TTY it would otherwise be silent).
    _sandbox_log("▶ Running: uv sync")
    rc = await _stream_subprocess(
        [sys.executable, "-m", "uv", "sync", "--verbose"],
        project_path,
        env=uv_env,
    )
    if rc != 0:
        # Try plain uv if python -m uv fails
        _sandbox_log("  (retrying with bare uv)")
        rc = await _stream_subprocess(["uv", "sync", "--verbose"], project_path, env=uv_env)
    if rc != 0:
        _sandbox["status"] = "error"
        _sandbox_log(f"✗ uv sync failed (exit {rc})")
        return

    # Prefer the generated self-contained server (serve.py) when present: it serves
    # the manifest-driven product frontend AND the ADK API from one process. Legacy
    # projects (no serve.py) fall back to ADK's generic `adk web` chat UI.
    serve_py = project_path / "serve.py"
    if serve_py.exists():
        launch_cmd = ["uv", "run", "python", "serve.py", "--port", str(port), "--host", "0.0.0.0"]
        _sandbox["mode"] = "serve"
        _sandbox_log(f"▶ Starting: uv run python serve.py --port {port} --host 0.0.0.0")
    else:
        launch_cmd = ["uv", "run", "adk", "web", ".", "--port", str(port), "--host", "0.0.0.0"]
        _sandbox["mode"] = "adk_web"
        _sandbox_log(f"▶ Starting: uv run adk web . --port {port} --host 0.0.0.0")
    try:
        adk_proc = await asyncio.create_subprocess_exec(
            *launch_cmd,
            cwd=str(project_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ},
        )
        _sandbox["proc"] = adk_proc

        assert adk_proc.stdout is not None
        async for raw in adk_proc.stdout:
            stripped = raw.decode(errors="replace").rstrip()
            _sandbox_log(stripped)
            if "Application startup complete" in stripped or "Uvicorn running" in stripped:
                _sandbox["status"] = "ready"
                _sandbox_log(f"✓ Ready at http://127.0.0.1:{port}")
                # Keep draining stdout so the pipe never fills and deadlocks the server
                asyncio.create_task(_drain_stdout(adk_proc))
                return

        # stdout closed without startup signal
        if _sandbox["status"] != "ready":
            _sandbox["status"] = "error"
            _sandbox_log("✗ adk web did not signal startup")
    except FileNotFoundError:
        _sandbox["status"] = "error"
        _sandbox_log("✗ uv not found — ensure uv is on PATH")


async def _drain_stdout(proc: asyncio.subprocess.Process) -> None:
    """Silently drain adk web stdout so the OS pipe never fills and deadlocks the server."""
    assert proc.stdout is not None
    try:
        async for raw in proc.stdout:
            _sandbox_log(raw.decode(errors="replace").rstrip())
    except Exception:
        pass


def set_events(events: PipelineEvents) -> None:
    global _events
    _events = events


def set_start_callback(cb) -> None:
    global _start_callback
    _start_callback = cb


def _require_events() -> PipelineEvents:
    if _events is None:
        raise HTTPException(status_code=503, detail="Pipeline not started")
    return _events


# ── Routes ──────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Landing page — enter requirement and start a run."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """HITL review dashboard."""
    # If no pipeline has started yet, redirect to landing
    if _events is None:
        return RedirectResponse("/")

    ev = _events
    tree = ev.current_tree
    nodes: list[SkillNode] = []
    if tree:
        nodes = tree.get_layer_nodes()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "status": ev.status,
            "current_layer": tree.current_layer if tree else 0,
            "nodes": nodes,
            "tree": tree,
            "sandbox_project_name": tree.project_name if tree else None,
        },
    )


@app.get("/status")
async def get_status() -> dict:
    """Lightweight poll endpoint — returns current pipeline status."""
    if _events is None:
        return {"status": "idle"}
    return {
        "status": _events.status,
        "current_layer": _events.current_tree.current_layer if _events.current_tree else 0,
        "project_name": _events.current_tree.project_name if _events.current_tree else None,
    }


@app.get("/tree")
async def get_tree() -> dict:
    ev = _require_events()
    if ev.current_tree is None:
        return {"tree": None}
    return ev.current_tree.model_dump()


class StartPayload(BaseModel):
    requirement: str
    project_name: str
    output_dir: Optional[str] = None


@app.post("/start")
async def start_pipeline(payload: StartPayload) -> dict:
    """Start a new pipeline run from the web UI."""
    global _events

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", payload.project_name):
        raise HTTPException(
            status_code=422,
            detail="project_name must be a valid Python identifier (letters/digits/underscore, start with a letter).",
        )

    # If a pipeline is already running (not idle/done), refuse
    if _events is not None and _events.status not in ("idle", "done"):
        raise HTTPException(
            status_code=409,
            detail=f"A pipeline is already running (status={_events.status}). Finish or reload first.",
        )

    if _start_callback is None:
        raise HTTPException(status_code=503, detail="Server not fully initialised yet.")

    # Kick off the pipeline asynchronously (non-blocking)
    asyncio.create_task(
        _start_callback(payload.requirement, payload.project_name, payload.output_dir)
    )

    return {"ok": True, "project_name": payload.project_name}


@app.post("/approve")
async def approve() -> dict:
    ev = _require_events()
    if not ev.status.startswith("awaiting_review"):
        raise HTTPException(
            status_code=409,
            detail=f"Nothing to approve right now (status={ev.status})",
        )
    ev.rollback.clear()
    ev.approve.set()
    return {"ok": True, "action": "approved"}


@app.post("/rollback")
async def rollback() -> dict:
    ev = _require_events()
    if not ev.status.startswith("awaiting_review"):
        raise HTTPException(
            status_code=409,
            detail=f"Nothing to roll back right now (status={ev.status})",
        )
    ev.approve.clear()
    ev.rollback.set()
    return {"ok": True, "action": "rollback"}


class RedecomposePayload(BaseModel):
    node_id: str
    new_description: Optional[str] = None
    hint: Optional[str] = None
    force_renegotiate: bool = False  # required to redecompose a node with a FROZEN contract


@app.post("/redecompose")
async def redecompose(payload: RedecomposePayload) -> dict:
    """Request a per-node re-decompose without rolling back the whole layer."""
    ev = _require_events()
    if not ev.status.startswith("awaiting_review"):
        raise HTTPException(
            status_code=409,
            detail=f"No structure review in progress (status={ev.status})",
        )
    await ev.redecompose.put(
        RedecomposeRequest(
            node_id=payload.node_id,
            new_description=payload.new_description,
            hint=payload.hint,
            force_renegotiate=payload.force_renegotiate,
        )
    )
    return {"ok": True, "node_id": payload.node_id}


@app.post("/approve_impl")
async def approve_impl() -> dict:
    """Approve the implementation review (HITL-2) and proceed to compilation."""
    ev = _require_events()
    if not ev.status.startswith("awaiting_impl_review"):
        raise HTTPException(
            status_code=409,
            detail=f"No implementation review in progress (status={ev.status})",
        )
    ev.rollback_impl.clear()
    ev.approve_impl.set()
    return {"ok": True, "action": "approved_impl"}


@app.post("/rollback_impl")
async def rollback_impl() -> dict:
    """Rollback the implementation review (HITL-2) and retry the entire layer."""
    ev = _require_events()
    if not ev.status.startswith("awaiting_impl_review"):
        raise HTTPException(
            status_code=409,
            detail=f"No implementation review in progress (status={ev.status})",
        )
    ev.approve_impl.clear()
    ev.rollback_impl.set()
    return {"ok": True, "action": "rollback_impl"}


class RetrySkillPayload(BaseModel):
    feedback: Optional[str] = None


@app.post("/retry_skill/{node_id}")
async def retry_skill(node_id: str, payload: RetrySkillPayload) -> dict:
    """Request per-skill retry during HITL-2 without rolling back the whole layer."""
    ev = _require_events()
    if not ev.status.startswith("awaiting_impl_review"):
        raise HTTPException(
            status_code=409,
            detail=f"No implementation review in progress (status={ev.status})",
        )
    await ev.retry_skill.put(
        RetrySkillRequest(node_id=node_id, feedback=payload.feedback)
    )
    return {"ok": True, "node_id": node_id}


# ── HITL-3: UI / interaction review ──────────────────────────────────────────


@app.post("/approve_ui")
async def approve_ui() -> dict:
    """Approve the generated UI/interaction contract (HITL-3) and proceed to compile."""
    ev = _require_events()
    if not ev.status.startswith("awaiting_ui_review"):
        raise HTTPException(
            status_code=409,
            detail=f"No UI review in progress (status={ev.status})",
        )
    ev.rollback_ui.clear()
    ev.approve_ui.set()
    return {"ok": True, "action": "approved_ui"}


@app.post("/rollback_ui")
async def rollback_ui() -> dict:
    """Reject the generated UI design (HITL-3); the UI Designer re-runs."""
    ev = _require_events()
    if not ev.status.startswith("awaiting_ui_review"):
        raise HTTPException(
            status_code=409,
            detail=f"No UI review in progress (status={ev.status})",
        )
    ev.approve_ui.clear()
    ev.rollback_ui.set()
    return {"ok": True, "action": "rollback_ui"}


class EditUIPayload(BaseModel):
    title: Optional[str] = None
    tagline: Optional[str] = None
    inputs: Optional[list[str]] = None
    output_renderers: Optional[list[str]] = None
    example_prompts: Optional[list[str]] = None
    user_facing_nodes: Optional[list[str]] = None


@app.post("/edit_ui")
async def edit_ui(payload: EditUIPayload) -> dict:
    """Manual override of the generated UISpec (no LLM re-run).

    Editing user_facing_nodes also re-syncs each node's `visibility` flag so the
    compiled frontend's author filter matches the human's choice.
    """
    from src.orchestrator.state import (
        InputAffordance,
        NodeVisibility,
        OutputRenderer,
        UISpec,
    )

    ev = _require_events()
    if ev.current_tree is None:
        raise HTTPException(status_code=404, detail="No active tree")
    tree = ev.current_tree
    if tree.ui_spec is None:
        tree.ui_spec = UISpec(title=tree.project_name)
    spec = tree.ui_spec

    if payload.title is not None:
        spec.title = payload.title
    if payload.tagline is not None:
        spec.tagline = payload.tagline
    if payload.inputs is not None:
        spec.inputs = [InputAffordance(i) for i in payload.inputs if i in InputAffordance._value2member_map_]
        if InputAffordance.TEXT not in spec.inputs:
            spec.inputs.insert(0, InputAffordance.TEXT)
    if payload.output_renderers is not None:
        spec.output_renderers = [
            OutputRenderer(r) for r in payload.output_renderers if r in OutputRenderer._value2member_map_
        ]
        if OutputRenderer.TEXT not in spec.output_renderers:
            spec.output_renderers.append(OutputRenderer.TEXT)
    if payload.example_prompts is not None:
        spec.example_prompts = payload.example_prompts
    if payload.user_facing_nodes is not None:
        all_names = {n.name for n in tree.root.topological_order()}
        chosen = [n for n in payload.user_facing_nodes if n in all_names]
        spec.user_facing_nodes = chosen
        # Re-sync per-node visibility so the compiler's author list matches.
        for node in tree.root.topological_order():
            node.visibility = (
                NodeVisibility.USER_FACING if node.name in chosen else NodeVisibility.INTERNAL
            )

    return {"ok": True}


# ── HITL-4: memory / persistence review ──────────────────────────────────────


@app.post("/approve_memory")
async def approve_memory() -> dict:
    """Approve the persistent-memory design (HITL-4) and proceed to UI design.

    Refuses unless every entity's deletion scope has been explicitly confirmed — the
    forcing function that makes a human look at the deletion path before approving.
    """
    ev = _require_events()
    if not ev.status.startswith("awaiting_memory_review"):
        raise HTTPException(
            status_code=409,
            detail=f"No memory review in progress (status={ev.status})",
        )
    tree = ev.current_tree
    if tree is not None and tree.memory_spec is not None:
        unconfirmed = [e.name for e in tree.memory_spec.entities if not e.deletion_confirmed]
        if unconfirmed:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Confirm the deletion scope for every entity before approving: "
                    + ", ".join(unconfirmed)
                ),
            )
    ev.rollback_memory.clear()
    ev.approve_memory.set()
    return {"ok": True, "action": "approved_memory"}


@app.post("/rollback_memory")
async def rollback_memory() -> dict:
    """Reject the generated memory design (HITL-4); the Memory Architect re-runs."""
    ev = _require_events()
    if not ev.status.startswith("awaiting_memory_review"):
        raise HTTPException(
            status_code=409,
            detail=f"No memory review in progress (status={ev.status})",
        )
    ev.approve_memory.clear()
    ev.rollback_memory.set()
    return {"ok": True, "action": "rollback_memory"}


class MemoryEntityEdit(BaseModel):
    name: str  # match key — identifies which entity to edit
    backend: Optional[str] = None
    retention: Optional[str] = None
    deletion_scope: Optional[str] = None
    deletion_confirmed: Optional[bool] = None


class EditMemoryPayload(BaseModel):
    entities: list[MemoryEntityEdit]


@app.post("/edit_memory")
async def edit_memory(payload: EditMemoryPayload) -> dict:
    """Manual override of the generated MemorySpec (no LLM re-run).

    Matches each edit to an entity by name and applies backend/retention/deletion_scope/
    deletion_confirmed. Backend is validated against MemoryBackend.
    """
    from src.orchestrator.state import MemoryBackend

    ev = _require_events()
    if ev.current_tree is None or ev.current_tree.memory_spec is None:
        raise HTTPException(status_code=404, detail="No memory spec to edit")
    spec = ev.current_tree.memory_spec
    by_name = {e.name: e for e in spec.entities}

    for edit in payload.entities:
        entity = by_name.get(edit.name)
        if entity is None:
            continue
        if edit.backend is not None and edit.backend in MemoryBackend._value2member_map_:
            entity.backend = MemoryBackend(edit.backend)
        if edit.retention is not None:
            entity.retention = edit.retention or None
        if edit.deletion_scope is not None:
            entity.deletion_scope = edit.deletion_scope or "entire entity"
        if edit.deletion_confirmed is not None:
            entity.deletion_confirmed = edit.deletion_confirmed

    return {"ok": True}


# ── Debug env-var endpoints ─────────────────────────────────────────────────


@app.get("/debug/missing_env")
async def debug_missing_env() -> dict:
    """Return the env var names the debug loop is currently waiting on."""
    return {
        "pending": _debug_env_gate["pending"],
        "provided": list(_debug_env_gate["provided"].keys()),
    }


class DebugEnvPayload(BaseModel):
    env_vars: dict[str, str]


@app.post("/debug/env")
async def debug_provide_env(payload: DebugEnvPayload) -> dict:
    """Supply missing env var values so the debug loop can continue.

    Values are written to the project .env and the waiting debug_env_provider is unblocked.
    """
    if not payload.env_vars:
        return {"ok": True, "written": 0}

    _debug_env_gate["provided"].update(payload.env_vars)
    if _debug_env_gate["event"] is not None:
        _debug_env_gate["event"].set()

    return {"ok": True, "written": len(payload.env_vars)}


async def debug_env_provider(missing_names: list[str]) -> dict[str, str]:
    """Async callback passed to DebugAgent: suspends until /debug/env is posted.

    Sets pipeline status to "debug_awaiting_env" so the dashboard renders an env-var
    collection form instead of the generic "Processing…" spinner.
    Times out after 300 s so a headless run is not blocked forever.
    """
    if not missing_names:
        return {}

    gate = _debug_env_gate
    gate["pending"] = list(missing_names)
    gate["provided"] = {}
    gate["event"] = asyncio.Event()

    # Flip pipeline status so the dashboard stops auto-refreshing and shows the form
    if _events is not None:
        _events.status = "debug_awaiting_env"

    logger.info("[debug] Waiting for env vars via UI: %s", missing_names)

    try:
        await asyncio.wait_for(gate["event"].wait(), timeout=300)
    except asyncio.TimeoutError:
        logger.warning("[debug] Timed out waiting for env vars; continuing without them.")

    # Restore status to debugging so normal flow resumes
    if _events is not None and _events.status == "debug_awaiting_env":
        _events.status = "debugging"

    result = {k: v for k, v in gate["provided"].items() if k in missing_names and v}
    gate["pending"] = []
    return result


# ── Sandbox endpoints ───────────────────────────────────────────────────────


class SandboxEnvPayload(BaseModel):
    env_vars: dict[str, str]
    project_name: Optional[str] = None


class SandboxStartPayload(BaseModel):
    project_name: Optional[str] = None


@app.post("/sandbox/env")
async def sandbox_set_env(payload: SandboxEnvPayload) -> dict:
    """Write key-value pairs into the compiled project's .env before sandbox launch."""
    project_path = _resolve_sandbox_project(payload.project_name)
    if project_path is None:
        raise HTTPException(status_code=404, detail="No compiled project found.")
    if not payload.env_vars:
        return {"ok": True, "written": 0}

    env_file = project_path / ".env"
    existing: dict[str, str] = {}
    if env_file.exists():
        for raw_line in env_file.read_text().splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    existing.update(payload.env_vars)
    lines = [f'{k}={v}' for k, v in existing.items()]
    env_file.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %d env vars to %s", len(payload.env_vars), env_file)
    return {"ok": True, "written": len(payload.env_vars)}


@app.get("/sandbox/required_env")
async def sandbox_required_env(project_name: Optional[str] = None) -> dict:
    """Return the list of env var names required by the compiled project."""
    project_path = _resolve_sandbox_project(project_name)

    # Read required_env_vars from the tree on disk or from the active pipeline.
    required: list[str] = []
    if project_path:
        tree_json = project_path / "blueprint_verified.json"
        if tree_json.exists():
            tree = SkillTree.load_json(tree_json)
            required = tree.required_env_vars if hasattr(tree, "required_env_vars") else []
    elif _events and _events.current_tree:
        tree = _events.current_tree
        required = tree.required_env_vars if hasattr(tree, "required_env_vars") else []

    satisfied: list[str] = []
    missing: list[str] = list(required)
    if project_path:
        env_file = project_path / ".env"
        present: set[str] = set()
        if env_file.exists():
            for raw_line in env_file.read_text().splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    if v.strip():
                        present.add(k.strip())
        satisfied = [k for k in required if k in present]
        missing = [k for k in required if k not in present]

    return {"required": required, "satisfied": satisfied, "missing": missing}


@app.post("/sandbox/start")
async def sandbox_start(payload: SandboxStartPayload = SandboxStartPayload()) -> dict:
    """Start the sandbox for a project. Pass project_name in the body, or fall back
    to the active pipeline run when status is 'done'."""
    project_path = _resolve_sandbox_project(payload.project_name)
    if project_path is None:
        raise HTTPException(status_code=404, detail="No compiled project found to run.")

    if _sandbox["status"] in ("setting_up", "ready"):
        # Stop any running sandbox before starting a new one
        await _sandbox_stop_proc()

    asyncio.create_task(_run_sandbox(project_path, _SANDBOX_PORT))
    return {"ok": True, "port": _SANDBOX_PORT}


@app.post("/sandbox/stop")
async def sandbox_stop() -> dict:
    await _sandbox_stop_proc()
    return {"ok": True}


@app.get("/sandbox/status")
async def sandbox_status_endpoint() -> dict:
    return {
        "status": _sandbox["status"],
        "port": _sandbox["port"],
        "mode": _sandbox.get("mode"),
        "ready_url": f"http://localhost:{_sandbox['port']}" if _sandbox["status"] == "ready" else None,
    }


@app.get("/sandbox/logs")
async def sandbox_logs(request: Request) -> StreamingResponse:
    """Server-Sent Events stream of sandbox log lines."""
    snapshot_idx = 0

    async def event_stream():
        nonlocal snapshot_idx
        # First, replay all existing lines
        lines = _sandbox["output"]
        while snapshot_idx < len(lines):
            yield f"data: {lines[snapshot_idx]}\n\n"
            snapshot_idx += 1

        # Then stream new lines as they arrive
        while True:
            if await request.is_disconnected():
                break
            if _sandbox["notify"]:
                _sandbox["notify"].clear()
            lines = _sandbox["output"]
            while snapshot_idx < len(lines):
                yield f"data: {lines[snapshot_idx]}\n\n"
                snapshot_idx += 1
            if _sandbox["status"] in ("ready", "error"):
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/projects")
async def list_projects() -> dict:
    """List existing compiled projects in the output directory."""
    output_dir = _get_output_dir()
    projects = []
    if output_dir.exists():
        for p in sorted(output_dir.iterdir()):
            if p.is_dir() and (p / "run.py").exists():
                projects.append({"name": p.name, "path": str(p)})
    return {"projects": projects}


@app.get("/project/{project_name}", response_class=HTMLResponse)
async def project_view(request: Request, project_name: str) -> HTMLResponse:
    """View a completed project — shows its Layer View / Tree View / Sandbox tabs."""
    project_path = _get_output_dir() / project_name
    if not project_path.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    tree_json = project_path / "blueprint_verified.json"
    if not tree_json.exists():
        raise HTTPException(status_code=404, detail="No blueprint found for this project.")

    tree = SkillTree.load_json(tree_json)
    nodes = tree.get_layer_nodes()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "status": "done",
            "current_layer": tree.current_layer,
            "nodes": nodes,
            "tree": tree,
            "tree_endpoint": f"/project/{project_name}/tree",
            "sandbox_project_name": project_name,
        },
    )


@app.get("/project/{project_name}/tree")
async def project_tree(project_name: str) -> dict:
    """Return the skill tree JSON for a compiled project."""
    project_path = _get_output_dir() / project_name
    tree_json = project_path / "blueprint_verified.json"
    if not tree_json.exists():
        raise HTTPException(status_code=404, detail="No blueprint found for this project.")
    tree = SkillTree.load_json(tree_json)
    return tree.model_dump()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_output_dir() -> Path:
    # Resolve relative to repo root (two levels up from this file)
    return Path(__file__).parent.parent.parent / "output"


def _resolve_sandbox_project(project_name: Optional[str] = None) -> Optional[Path]:
    """Return the project path to use for sandbox operations.

    Resolution order:
    1. Explicit project_name (from template context or request body).
    2. Active pipeline run when status == "done".
    """
    if project_name:
        p = _get_output_dir() / project_name
        return p if p.exists() else None
    if _events and _events.current_tree and _events.status == "done":
        p = _get_output_dir() / _events.current_tree.project_name
        return p if p.exists() else None
    return None


async def _sandbox_stop_proc() -> None:
    proc = _sandbox.get("proc")
    if proc is not None:
        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                proc.kill()
        except ProcessLookupError:
            pass
    _sandbox["status"] = "idle"
    _sandbox["proc"] = None
    _sandbox["output"] = []
    _sandbox["notify"] = None




class EditPayload(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    exec_type: Optional[str] = None


@app.post("/edit/{node_id}")
async def edit_node(node_id: str, payload: EditPayload) -> dict:
    """Manual override of a node's name/description/exec_type. No LLM re-run."""
    from src.orchestrator.state import ExecType

    ev = _require_events()
    if ev.current_tree is None:
        raise HTTPException(status_code=404, detail="No active tree")
    node = ev.current_tree.root.find_node_by_id(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    if payload.name is not None:
        node.name = payload.name
    if payload.description is not None:
        node.description = payload.description
    if payload.exec_type is not None:
        try:
            node.exec_type = ExecType(payload.exec_type)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid exec_type: {payload.exec_type}")
    return {"ok": True, "node_id": node_id}


class EditChildrenPayload(BaseModel):
    description: Optional[str] = None
    children: list[dict]  # list of {id?: str, name: str, description: str}
    composition_type: Optional[str] = None


@app.post("/edit_children/{node_id}")
async def edit_children(node_id: str, payload: EditChildrenPayload) -> dict:
    """Edit a composite node's description, children (add/remove/update), and composition type.

    Requires at least 2 children in the final result. Entries with `id` matching an existing child
    are updated; entries without `id` are created as new children. Existing children omitted from
    the list are removed. Finally, all children are synced.
    """
    from src.orchestrator.state import CompositionType, NodeType, SkillNode

    ev = _require_events()
    if ev.current_tree is None:
        raise HTTPException(status_code=404, detail="No active tree")
    node = ev.current_tree.root.find_node_by_id(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    if len(payload.children) < 2:
        raise HTTPException(status_code=422, detail="A composite node must have at least 2 sub-skills.")

    if payload.description is not None:
        node.description = payload.description

    if payload.composition_type is not None:
        try:
            node.composition_type = CompositionType(payload.composition_type)
        except ValueError:
            raise HTTPException(
                status_code=422, detail=f"Invalid composition_type: {payload.composition_type}"
            )

    # Build a map of existing children by id
    child_by_id = {c.id: c for c in node.children}

    # Collect ids of children to keep (from the submission)
    submitted_ids = set()
    for entry in payload.children:
        child_id = entry.get("id")
        if child_id:
            submitted_ids.add(child_id)

    # Remove children that are not in the submission
    node.children = [c for c in node.children if c.id in submitted_ids]

    # Update existing children and create new ones
    for entry in payload.children:
        child_id = entry.get("id")
        child_name = entry.get("name", "Unnamed").strip()
        child_desc = entry.get("description", "").strip()

        if child_id:
            # Update existing child
            child = child_by_id.get(child_id)
            if child is None:
                raise HTTPException(status_code=404, detail=f"Child node {child_id} not found")
            if child_name:
                child.name = child_name
            if child_desc:
                child.description = child_desc
        else:
            # Create new child
            new_child = SkillNode(
                name=child_name or "New Sub-skill",
                description=child_desc,
                node_type=NodeType.UNKNOWN,
                depth=node.depth + 1,
                parent_id=node.id,
            )
            node.children.append(new_child)

    return {"ok": True, "node_id": node_id}


class ConvertNodePayload(BaseModel):
    target_type: str  # "atomic" | "composite"
    exec_type: Optional[str] = None  # required/used when target_type == "atomic"
    composition_type: Optional[str] = None  # used when target_type == "composite"


@app.post("/convert_node/{node_id}")
async def convert_node(node_id: str, payload: ConvertNodePayload) -> dict:
    """Convert a node between atomic and composite types.

    Atomic → Composite: adds 2 placeholder children, clears exec_type/implementation/instruction.
    Composite → Atomic: clears children/composition_type, sets exec_type (default LLM_PROMPT).
    """
    from src.orchestrator.state import CompositionType, ExecType, NodeType, SkillNode

    ev = _require_events()
    if ev.current_tree is None:
        raise HTTPException(status_code=404, detail="No active tree")
    node = ev.current_tree.root.find_node_by_id(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    target_type = payload.target_type.lower()

    if target_type == "composite":
        # Atomic → Composite
        node.node_type = NodeType.COMPOSITE
        node.composition_type = CompositionType(
            payload.composition_type or "SEQUENTIAL"
        ) if payload.composition_type else CompositionType.SEQUENTIAL
        node.exec_type = None
        node.implementation = None
        node.instruction = None

        # Add 2 placeholder children if not enough
        if len(node.children) < 2:
            for i in range(2 - len(node.children)):
                placeholder = SkillNode(
                    name=f"Sub-skill {len(node.children) + 1}",
                    description="",
                    node_type=NodeType.UNKNOWN,
                    depth=node.depth + 1,
                    parent_id=node.id,
                )
                node.children.append(placeholder)

    elif target_type == "atomic":
        # Composite → Atomic
        node.node_type = NodeType.ATOMIC
        node.children = []
        node.composition_type = None
        node.exec_type = ExecType(payload.exec_type or "LLM_PROMPT")

    else:
        raise HTTPException(
            status_code=422, detail=f"Invalid target_type: {target_type}. Use 'atomic' or 'composite'."
        )

    return {"ok": True, "node_id": node_id, "new_type": target_type}


class EditImplPayload(BaseModel):
    description: Optional[str] = None
    implementation: Optional[str] = None
    instruction: Optional[str] = None
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None


@app.post("/edit_impl/{node_id}")
async def edit_impl(node_id: str, payload: EditImplPayload) -> dict:
    """Edit an atomic node's implementation, instruction, schemas, or description."""
    ev = _require_events()
    if ev.current_tree is None:
        raise HTTPException(status_code=404, detail="No active tree")
    node = ev.current_tree.root.find_node_by_id(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    if payload.description is not None:
        node.description = payload.description
    if payload.implementation is not None:
        node.implementation = payload.implementation
    if payload.instruction is not None:
        node.instruction = payload.instruction
    if payload.input_schema is not None:
        node.input_schema = payload.input_schema
    if payload.output_schema is not None:
        node.output_schema = payload.output_schema
    return {"ok": True, "node_id": node_id}
