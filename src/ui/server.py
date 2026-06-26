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

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.orchestrator.pipeline import PipelineEvents, RedecomposeRequest
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


async def _stream_subprocess(cmd: list[str], cwd: Path) -> int:
    """Run a subprocess, stream output to sandbox log. Returns exit code."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ},
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

    # Remove any existing .venv (may have been created on a different OS/arch).
    # Use subprocess rm -rf instead of shutil.rmtree because broken symlinks
    # inside the venv cause rmtree to fail silently on Linux, leaving a corrupt
    # directory that makes `uv sync` error with "not a valid Python environment".
    venv_dir = project_path / ".venv"
    if venv_dir.exists() or venv_dir.is_symlink():
        await _stream_subprocess(["rm", "-rf", str(venv_dir)], project_path)

    # Install generated project dependencies
    _sandbox_log("▶ Running: uv sync")
    rc = await _stream_subprocess(
        [sys.executable, "-m", "uv", "sync"],
        project_path,
    )
    if rc != 0:
        # Try plain uv if python -m uv fails
        _sandbox_log("  (retrying with bare uv)")
        rc = await _stream_subprocess(["uv", "sync"], project_path)
    if rc != 0:
        _sandbox["status"] = "error"
        _sandbox_log(f"✗ uv sync failed (exit {rc})")
        return

    _sandbox_log(f"▶ Starting: uv run adk web . --port {port} --host 0.0.0.0")
    try:
        adk_proc = await asyncio.create_subprocess_exec(
            "uv", "run", "adk", "web", ".", "--port", str(port), "--host", "0.0.0.0",
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
                _sandbox_log(f"✓ Agent chat ready at http://127.0.0.1:{port}")
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


@app.post("/sandbox/env")
async def sandbox_set_env(payload: SandboxEnvPayload) -> dict:
    """Write key-value pairs into the compiled project's .env before sandbox launch.

    Values are written to disk only and are never returned after submission.
    """
    project_path = _resolve_sandbox_project()
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
async def sandbox_required_env() -> dict:
    """Return the list of env var names required by the compiled project."""
    if _events is None or _events.current_tree is None:
        return {"required": []}
    tree = _events.current_tree
    required = tree.required_env_vars if hasattr(tree, "required_env_vars") else []

    project_path = _resolve_sandbox_project()
    satisfied: list[str] = []
    missing: list[str] = []
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
        for key in required:
            (satisfied if key in present else missing).append(key)
    else:
        missing = list(required)

    return {"required": required, "satisfied": satisfied, "missing": missing}


@app.post("/sandbox/start")
async def sandbox_start() -> dict:
    """Start the sandbox for the current (done) project, or a named existing project."""
    # Resolve project path
    project_path = _resolve_sandbox_project()
    if project_path is None:
        raise HTTPException(status_code=404, detail="No compiled project found to run.")

    if _sandbox["status"] in ("setting_up", "ready"):
        raise HTTPException(
            status_code=409,
            detail=f"Sandbox already running (status={_sandbox['status']})",
        )

    asyncio.create_task(_run_sandbox(project_path, _SANDBOX_PORT))
    return {"ok": True, "port": _SANDBOX_PORT}


@app.post("/sandbox/start/{project_name}")
async def sandbox_start_named(project_name: str) -> dict:
    """Start sandbox for an existing output project by name."""
    project_path = _get_output_dir() / project_name
    if not project_path.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found in output.")

    if _sandbox["status"] in ("setting_up", "ready"):
        # Stop the previous one first
        await _sandbox_stop_proc()

    asyncio.create_task(_run_sandbox(project_path, _SANDBOX_PORT))
    return {"ok": True, "project_name": project_name, "port": _SANDBOX_PORT}


@app.post("/sandbox/stop")
async def sandbox_stop() -> dict:
    await _sandbox_stop_proc()
    return {"ok": True}


@app.get("/sandbox/status")
async def sandbox_status_endpoint() -> dict:
    return {
        "status": _sandbox["status"],
        "port": _sandbox["port"],
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


def _resolve_sandbox_project() -> Optional[Path]:
    """Return the project path for the current pipeline run, if compiled."""
    if _events and _events.current_tree and _events.status == "done":
        p = _get_output_dir() / _events.current_tree.project_name
        if p.exists():
            return p
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


# ── Chat page + ADK proxy ────────────────────────────────────────────────────

def _adk_base_url() -> str:
    return f"http://127.0.0.1:{_sandbox['port']}"


@app.get("/chat/{project_name}", response_class=HTMLResponse)
async def chat_page(request: Request, project_name: str) -> HTMLResponse:
    """Serve the embedded chat UI for a compiled project."""
    return templates.TemplateResponse(
        request,
        "chat.html",
        {"project_name": project_name},
    )


@app.post("/chat/{project_name}/session")
async def chat_create_session(project_name: str) -> dict:
    """Create a new ADK session for this project and return its ID."""
    url = f"{_adk_base_url()}/apps/{project_name}/users/user/sessions"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json={})
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Sandbox not running — start it first.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


@app.post("/chat/{project_name}/send")
async def chat_send(project_name: str, request: Request) -> StreamingResponse:
    """Proxy a user message to ADK /run_sse and stream events back."""
    body = await request.json()
    session_id = body.get("session_id")
    text = body.get("text", "")

    if not session_id:
        raise HTTPException(status_code=422, detail="session_id required")

    adk_payload = {
        "app_name": project_name,
        "user_id": "user",
        "session_id": session_id,
        "new_message": {
            "role": "user",
            "parts": [{"text": text}],
        },
        "streaming": True,
    }

    async def stream_events():
        url = f"{_adk_base_url()}/run_sse"
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, json=adk_payload) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data:"):
                            yield f"{line}\n\n"
        except httpx.ConnectError:
            yield 'data: {"error": "Sandbox not running"}\n\n'

    return StreamingResponse(stream_events(), media_type="text/event-stream")


class EditPayload(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@app.post("/edit/{node_id}")
async def edit_node(node_id: str, payload: EditPayload) -> dict:
    """Manual override of a node's name/description. No LLM re-run."""
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
    return {"ok": True, "node_id": node_id}


class EditChildrenPayload(BaseModel):
    description: Optional[str] = None
    children: list[dict]  # list of {id, name, description}


@app.post("/edit_children/{node_id}")
async def edit_children(node_id: str, payload: EditChildrenPayload) -> dict:
    """Edit a composite node's description and its children's names/descriptions.

    Requires at least 2 children. Each entry in `children` must have `id` matching
    an existing child node; only `name` and `description` are updated.
    """
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

    child_by_id = {c.id: c for c in node.children}
    for entry in payload.children:
        child_id = entry.get("id")
        child = child_by_id.get(child_id)
        if child is None:
            raise HTTPException(status_code=404, detail=f"Child node {child_id} not found")
        if entry.get("name") is not None:
            child.name = entry["name"]
        if entry.get("description") is not None:
            child.description = entry["description"]

    return {"ok": True, "node_id": node_id}


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
