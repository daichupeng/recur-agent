"""Debug agent: end-to-end test generation, execution, and repair loop for generated ADK projects.

Flow:
  1. Scan the compiled project for required env vars; invoke `env_provider` callback if any are missing.
  2. Generate a minimal pytest test case via LLM (single conversation turn).
  3. Install project deps with `uv sync`.
  4. Run the test under `uv run pytest -x` inside the project directory.
  5. On failure: parse traceback, ask LLM to patch the offending atomic, rewrite the file, repeat.
  6. Declare success when pytest exits 0 (or when MAX_ITERATIONS is exhausted).

The agent is designed to run after the compiler + import-verify stage.  It operates independently:
no asyncio.Event HITL gates — the caller awaits `run()` and receives a DebugResult.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

from src.agents.base_agent import BaseAgent, _find_text_block
from src.agents.compiler import CompilerAgent, _snake_name
from src.orchestrator.state import ExecType, NodeType, SkillNode, SkillTree

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8
_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "implementation": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["implementation", "explanation"],
    "additionalProperties": False,
}

_TEST_GEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "test_code": {"type": "string"},
        "test_input": {"type": "string"},
    },
    "required": ["test_code", "test_input"],
    "additionalProperties": False,
}

_SYSTEM_TEST_GEN = """You are a senior Python test engineer helping to validate an auto-generated Google ADK multi-agent project.

You will receive:
- project_name: the name of the generated project
- requirement: the original product requirement the project implements
- root_module: the Python import path for the root agent (e.g. orchestrators.stock_analyzer)
- root_symbol: the exported agent variable name (e.g. stock_analyzer_agent)
- required_env_vars: list of environment variable names the project uses

Your job is to produce a pytest test file. Follow these rules EXACTLY:

IMPORTS:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

load_dotenv()
```

CRITICAL API RULES — the google-genai library uses CONSTRUCTORS, not class methods:
- CORRECT:   Part(text="some text")
- WRONG:     Part.from_text("some text")   ← DO NOT USE THIS
- CORRECT:   Content(role="user", parts=[Part(text="hello")])
- WRONG:     Content.from_text(...)        ← DO NOT USE THIS

TEST PATTERN:
```python
@pytest.mark.asyncio
async def test_<name>():
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID
    )
    message = Content(role="user", parts=[Part(text="<realistic input>")])
    responses = []
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    responses.append(part.text)
    assert responses, "Agent returned no text response"
```

Also produce a one-sentence `test_input` — the realistic user message to send to the agent (e.g. "Analyze AAPL").

Respond with JSON matching the schema: {"test_code": "<full pytest file as string>", "test_input": "<one sentence user message>"}
"""

_SYSTEM_PATCH = """You are a senior Python engineer debugging an auto-generated Google ADK tool.

You will receive:
- node_name: the name of the failing atomic skill function
- node_description: what the function is supposed to do
- exec_type: DETERMINISTIC_CODE | EXTERNAL_API | OPENSOURCE_LIBRARY
- input_schema: JSON Schema of the function's inputs
- output_schema: JSON Schema the function must return
- current_implementation: the current function body (4-space indented, no def line)
- error_output: the full pytest / traceback output

Your job:
1. Identify the root cause of the error.
2. Produce a corrected function body.

Return JSON: {"implementation": "<fixed function body>", "explanation": "<one-line summary of fix>"}

Rules for the function body:
- Every line must be indented exactly 4 spaces (or more for nested blocks).
- Do NOT include the `def` line, decorators, or any outer-scope code.
- All imports must be inside the body, each indented 4 spaces.
- The function must return a dict matching output_schema.
- For EXTERNAL_API: use `os.environ["KEY_NAME"]` for credentials (import os inside body).
- Never use hardcoded credential strings.
"""


@dataclass
class DebugResult:
    """Summary of the debug run."""
    success: bool
    iterations: int
    test_file: Path | None = None
    final_output: str = ""
    errors: list[str] = field(default_factory=list)


# Callback type: given a list of missing env-var names, returns a dict of {name: value}.
# The caller (UI or CLI) is responsible for prompting the user.
EnvProvider = Callable[[list[str]], Awaitable[dict[str, str]]]


class DebugAgent(BaseAgent):
    """End-to-end test generation and repair loop for a compiled ADK project."""

    MODEL = "claude-sonnet-4-6"
    MAX_TOKENS = 16000

    def __init__(self) -> None:
        super().__init__(system_prompt=_SYSTEM_TEST_GEN)
        self._patch_agent = _PatchAgent()
        self._test_patch_agent = _TestPatchAgent()
        self._compiler = CompilerAgent()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        project_dir: Path,
        tree: SkillTree,
        *,
        env_provider: EnvProvider | None = None,
    ) -> DebugResult:
        """Run the full debug loop for the compiled project at `project_dir`.

        Args:
            project_dir: Path to the generated ADK project directory.
            tree: The SkillTree that was compiled into project_dir (used for repair).
            env_provider: Optional async callback to collect missing env var values.
                          Signature: async (missing_names: list[str]) -> dict[str, value]
                          If None, missing vars are logged as warnings but not fatal.

        Returns:
            DebugResult with success status, iteration count, and last output.
        """
        result = DebugResult(success=False, iterations=0)

        # ── 1. Collect missing env vars ────────────────────────────────────
        await self._ensure_env_vars(project_dir, tree, env_provider)

        # ── 2. Install dependencies ────────────────────────────────────────
        logger.info("[debug] Installing project dependencies via uv sync …")
        install_ok = await self._uv_sync(project_dir)
        if not install_ok:
            result.final_output = "uv sync failed — cannot run tests."
            result.errors.append(result.final_output)
            return result

        # ── 3. Generate test file ──────────────────────────────────────────
        test_file = await self._generate_test(project_dir, tree)
        result.test_file = test_file
        if test_file is None:
            result.final_output = "Test generation failed."
            result.errors.append(result.final_output)
            return result

        # ── 4. Debug loop ──────────────────────────────────────────────────
        consecutive_unpatched = 0
        for iteration in range(1, MAX_ITERATIONS + 1):
            result.iterations = iteration
            logger.info("[debug] Running pytest (iteration %d/%d) …", iteration, MAX_ITERATIONS)

            exit_code, output = await self._run_pytest(project_dir, test_file)
            result.final_output = output

            if exit_code == 0:
                logger.info("[debug] All tests passed on iteration %d.", iteration)
                result.success = True
                return result

            logger.warning("[debug] Tests failed (iteration %d). Output:\n%s", iteration, output[-3000:])
            result.errors.append(output[-2000:])

            patched = await self._patch_from_error(output, project_dir, tree, test_file)
            if patched:
                consecutive_unpatched = 0
            else:
                consecutive_unpatched += 1
                logger.warning(
                    "[debug] No repair made (consecutive unpatched: %d).", consecutive_unpatched
                )
                if consecutive_unpatched >= 2:
                    logger.warning("[debug] Stopping after 2 consecutive unrepaired failures.")
                    break

        return result

    # ------------------------------------------------------------------
    # Env var collection
    # ------------------------------------------------------------------

    async def _ensure_env_vars(
        self,
        project_dir: Path,
        tree: SkillTree,
        env_provider: EnvProvider | None,
    ) -> None:
        """Check for missing env vars; fill .env via env_provider if needed."""
        env_file = project_dir / ".env"
        existing: dict[str, str] = {}

        # Parse existing .env (key=value lines)
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    existing[k.strip()] = v.strip().strip('"').strip("'")

        # Also check process environment
        import os
        missing = [
            var for var in tree.required_env_vars
            if not existing.get(var) and not os.environ.get(var)
        ]

        if not missing:
            return

        logger.warning("[debug] Missing env vars: %s", missing)

        if env_provider is None:
            logger.warning("[debug] No env_provider configured — missing vars may cause runtime errors.")
            return

        provided = await env_provider(missing)
        if not provided:
            return

        # Append new values to .env
        new_lines = [f'{k}="{v}"' for k, v in provided.items() if v]
        if new_lines:
            with env_file.open("a") as f:
                f.write("\n" + "\n".join(new_lines) + "\n")
            logger.info("[debug] Wrote %d env var(s) to .env", len(new_lines))

    # ------------------------------------------------------------------
    # Dep install
    # ------------------------------------------------------------------

    async def _uv_sync(self, project_dir: Path) -> bool:
        """Run `uv sync` in project_dir. Returns True on success."""
        proc = await asyncio.create_subprocess_exec(
            "uv", "sync",
            cwd=str(project_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout_bytes, _ = await proc.communicate()
        output = stdout_bytes.decode(errors="replace")
        if proc.returncode != 0:
            logger.error("[debug] uv sync failed:\n%s", output)
            return False
        logger.debug("[debug] uv sync OK:\n%s", output)
        return True

    # ------------------------------------------------------------------
    # Test generation
    # ------------------------------------------------------------------

    async def _generate_test(self, project_dir: Path, tree: SkillTree) -> Path | None:
        """Ask the LLM to write a smoke-test; write it as test_smoke.py."""
        from src.orchestrator.state import NodeType

        root = tree.root
        root_name = _snake_name(root.name)
        if root.node_type == NodeType.ATOMIC:
            root_module = f"atomics.{root_name}"
        else:
            root_module = f"orchestrators.{root_name}"
        root_symbol = f"{root_name}_agent"

        payload = {
            "project_name": tree.project_name,
            "requirement": tree.requirement,
            "root_module": root_module,
            "root_symbol": root_symbol,
            "required_env_vars": tree.required_env_vars or [],
        }

        try:
            message = await self._call(
                messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
                output_schema=_TEST_GEN_SCHEMA,
            )
            data = json.loads(_find_text_block(message).text)
            test_code: str = data["test_code"]
            test_input: str = data.get("test_input", "Hello")
        except Exception as exc:
            logger.error("[debug] Test generation failed: %s", exc)
            # Fall back to a minimal template
            test_code = _minimal_test_template(
                project_name=tree.project_name,
                root_module=root_module,
                root_symbol=root_symbol,
                test_input="Hello, please run a basic test.",
            )

        # Always inject the test_input into the code if the placeholder exists
        test_code = test_code.replace("__TEST_INPUT__", test_input if "test_input" in dir() else "Hello")

        # Sanitize known ADK API misuses that LLMs frequently generate
        test_code = _sanitize_test_code(test_code)

        # Ensure pytest-asyncio is added to pyproject if missing
        await self._ensure_pytest_deps(project_dir)

        test_file = project_dir / "test_smoke.py"
        test_file.write_text(test_code)
        logger.info("[debug] Wrote test file: %s", test_file)
        return test_file

    async def _ensure_pytest_deps(self, project_dir: Path) -> None:
        """Add pytest + pytest-asyncio to pyproject.toml as dev deps if missing."""
        pyproject = project_dir / "pyproject.toml"
        if not pyproject.exists():
            return
        content = pyproject.read_text()
        changed = False

        # Append missing test packages to [project] dependencies (simplest approach —
        # avoids inventing new TOML sections that uv may not recognise).
        if "pytest-asyncio" not in content:
            import re
            pkg_lines = '    "pytest>=8.0",\n    "pytest-asyncio>=0.24",\n'
            content = re.sub(
                r'(dependencies\s*=\s*\[)(.*?)(\])',
                lambda m: m.group(1) + m.group(2) + pkg_lines + m.group(3),
                content,
                count=1,
                flags=re.DOTALL,
            )
            changed = True

        if "[tool.pytest.ini_options]" not in content:
            content += '\n[tool.pytest.ini_options]\nasyncio_mode = "auto"\n'
            changed = True

        if changed:
            pyproject.write_text(content)
            await self._uv_sync(project_dir)

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------

    async def _run_pytest(self, project_dir: Path, test_file: Path) -> tuple[int, str]:
        """Run pytest on `test_file` inside project_dir. Returns (exit_code, combined_output)."""
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", "pytest", str(test_file.name), "-x", "-v",
            "--tb=short", "--no-header",
            cwd=str(project_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**__import__("os").environ},
        )
        stdout_bytes, _ = await proc.communicate()
        output = stdout_bytes.decode(errors="replace")
        return proc.returncode or 0, output

    # ------------------------------------------------------------------
    # Error parsing and repair
    # ------------------------------------------------------------------

    async def _patch_from_error(
        self, error_output: str, project_dir: Path, tree: SkillTree, test_file: Path | None = None
    ) -> bool:
        """Identify the failing atomic node or test file and patch it. Returns True if patched."""

        # ── Priority 0: deterministic sanitize of the test file ───────────────
        # Catches known API misuses (Part.from_text etc.) without an LLM call.
        if test_file is not None and test_file.exists():
            original = test_file.read_text()
            sanitized = _sanitize_test_code(original)
            if sanitized != original:
                test_file.write_text(sanitized)
                logger.info("[debug] Deterministic sanitize fixed test file.")
                return True

        # ── Priority 1: atomic node repair ────────────────────────────────────
        node = self._identify_failing_node(error_output, project_dir, tree)
        if node is not None:
            logger.info("[debug] Patching node '%s' …", node.name)
            new_impl = await self._patch_agent.patch(node, error_output)
            if not new_impl:
                logger.warning("[debug] Patch agent returned nothing for '%s'.", node.name)
                # Fall through to test repair below
            else:
                node.implementation = new_impl
                atomics_dir = project_dir / "atomics"
                try:
                    self._compiler._compile_atomic(node, atomics_dir)  # type: ignore[attr-defined]
                    logger.info("[debug] Patched and recompiled '%s'.", node.name)
                    return True
                except ValueError as exc:
                    logger.error("[debug] Recompile after patch failed for '%s': %s", node.name, exc)
                    # Fall through to test repair below

        # ── Priority 2: LLM rewrite of the test file ──────────────────────────
        if test_file is not None and test_file.exists():
            logger.info("[debug] Attempting LLM test-file repair …")
            new_test = await self._test_patch_agent.patch(
                test_file.read_text(), error_output
            )
            if new_test:
                new_test = _sanitize_test_code(new_test)
                test_file.write_text(new_test)
                logger.info("[debug] Rewrote test file after LLM repair.")
                return True

        return False

    def _identify_failing_node(
        self, error_output: str, project_dir: Path, tree: SkillTree
    ) -> SkillNode | None:
        """Parse error output to find which atomic node caused the failure."""
        # Priority 1: explicit File "…/atomics/<name>.py" references in tracebacks
        for match in re.finditer(r'File ".*?atomics[/\\\\]([^"]+)\.py"', error_output):
            candidate = match.group(1)
            node = self._find_node_by_module(candidate, tree)
            if node:
                return node

        # Priority 2: ImportError mentioning a module
        import_match = re.search(r"ImportError.*?'([^']+)'", error_output)
        if import_match:
            mod = import_match.group(1).split(".")[-1]
            node = self._find_node_by_module(mod, tree)
            if node:
                return node

        # Priority 3: NameError / AttributeError inside an atomic file named in the path
        for match in re.finditer(r"(atomics[/\\\\][^\s:]+\.py)", error_output):
            candidate = Path(match.group(1)).stem
            node = self._find_node_by_module(candidate, tree)
            if node:
                return node

        return None

    @staticmethod
    def _find_node_by_module(module_name: str, tree: SkillTree) -> SkillNode | None:
        return next(
            (
                n for n in tree.root.topological_order()
                if n.name == module_name
                and n.node_type == NodeType.ATOMIC
                and n.exec_type in (
                    ExecType.DETERMINISTIC_CODE,
                    ExecType.EXTERNAL_API,
                    ExecType.OPENSOURCE_LIBRARY,
                )
            ),
            None,
        )


class _PatchAgent(BaseAgent):
    """Single-purpose agent: fix an atomic tool body given an error."""

    MODEL = "claude-sonnet-4-6"
    MAX_TOKENS = 16000

    def __init__(self) -> None:
        super().__init__(system_prompt=_SYSTEM_PATCH)

    async def patch(self, node: SkillNode, error_output: str) -> str | None:
        payload = {
            "node_name": node.name,
            "node_description": node.description,
            "exec_type": node.exec_type.value if node.exec_type else "UNKNOWN",
            "input_schema": node.input_schema,
            "output_schema": node.output_schema,
            "current_implementation": node.implementation or "",
            "error_output": error_output[-4000:],  # trim to avoid token overflow
        }
        try:
            message = await self._call(
                messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
                output_schema=_PATCH_SCHEMA,
            )
            data = json.loads(_find_text_block(message).text)
            impl = data.get("implementation", "")
            explanation = data.get("explanation", "")
            logger.info("[debug] Patch explanation for '%s': %s", node.name, explanation)
            return impl if impl else None
        except Exception as exc:
            logger.error("[debug] Patch LLM call failed for '%s': %s", node.name, exc)
            return None


# ------------------------------------------------------------------
# Test code sanitizer — fixes known LLM API misuses deterministically
# ------------------------------------------------------------------

def _extract_call_arg(code: str, pos: int) -> str:
    """Return the substring inside the outermost parentheses starting at `pos`.

    `pos` must point to the opening '('.  Handles nested parens and quoted strings.
    Returns the inner content (without the enclosing parens) or "" on parse error.
    """
    assert code[pos] == "("
    depth = 0
    in_str: str | None = None
    start = pos + 1
    i = pos
    while i < len(code):
        ch = code[i]
        if in_str:
            if ch == "\\" and in_str != "`":
                i += 2
                continue
            if ch == in_str:
                in_str = None
        else:
            if ch in ('"', "'"):
                in_str = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return code[start:i]
        i += 1
    return ""


def _sanitize_test_code(code: str) -> str:
    """Fix common google-genai API misuses emitted by LLMs.

    Uses a balanced-paren scanner so nested parens (e.g. Part.from_text(
    "Find the most fitting poem sentence(s)")) are handled correctly.
    """
    # --- Part.from_text(<arg>) → Part(text=<arg>) ---
    # In google-genai 2.x, Part.from_text is keyword-only (text=...); LLMs often
    # call it positionally. Rewrite to the Part(text=...) constructor. If the arg
    # is already `text=...`, keep it as-is to avoid `Part(text=text=...)`.
    result = []
    search_from = 0
    for m in re.finditer(r'Part\.from_text\(', code):
        result.append(code[search_from:m.start()])
        arg = _extract_call_arg(code, m.end() - 1)
        inner = arg if _arg_is_keyword(arg, "text") else f"text={arg}"
        result.append(f"Part({inner})")
        search_from = m.end() + len(arg) + 1  # skip past the closing ')'
    result.append(code[search_from:])
    code = "".join(result)

    # --- Content.from_text(<arg>) → Content(role="user", parts=[Part(text=<arg>)]) ---
    result = []
    search_from = 0
    for m in re.finditer(r'Content\.from_text\(', code):
        result.append(code[search_from:m.start()])
        arg = _extract_call_arg(code, m.end() - 1)
        inner = arg if _arg_is_keyword(arg, "text") else f"text={arg}"
        result.append(f'Content(role="user", parts=[Part({inner})])')
        search_from = m.end() + len(arg) + 1
    result.append(code[search_from:])
    code = "".join(result)

    return code


def _arg_is_keyword(arg: str, name: str) -> bool:
    """True if `arg` already opens with the keyword `name=` (not `name==` / `name>=`)."""
    stripped = arg.lstrip()
    if not stripped.startswith(f"{name}="):
        return False
    after = stripped[len(name) + 1:]
    return not after.startswith("=")  # guard against `text==` comparisons


# ------------------------------------------------------------------
# Agent that rewrites a broken test file given its error output
# ------------------------------------------------------------------

_SYSTEM_TEST_PATCH = """You are a senior Python test engineer debugging a pytest smoke test for a Google ADK project.

You will receive:
- current_test_code: the full content of the broken test file
- error_output: the pytest / traceback output showing what failed

Your job: produce a corrected version of the entire test file.

CRITICAL API RULES — the google-genai library uses CONSTRUCTORS, not class methods:
- CORRECT:   Part(text="some text")
- WRONG:     Part.from_text("some text")   ← DO NOT USE THIS
- CORRECT:   Content(role="user", parts=[Part(text="hello")])
- WRONG:     Content.from_text(...)        ← DO NOT USE THIS

Return JSON: {"test_code": "<full corrected test file>", "explanation": "<one-line summary of fix>"}
"""

_TEST_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "test_code": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["test_code", "explanation"],
    "additionalProperties": False,
}


class _TestPatchAgent(BaseAgent):
    """Rewrites a broken smoke-test file given the pytest error output."""

    MODEL = "claude-sonnet-4-6"
    MAX_TOKENS = 8000

    def __init__(self) -> None:
        super().__init__(system_prompt=_SYSTEM_TEST_PATCH)

    async def patch(self, current_test_code: str, error_output: str) -> str | None:
        payload = {
            "current_test_code": current_test_code,
            "error_output": error_output[-4000:],
        }
        try:
            message = await self._call(
                messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
                output_schema=_TEST_PATCH_SCHEMA,
            )
            data = json.loads(_find_text_block(message).text)
            explanation = data.get("explanation", "")
            logger.info("[debug] Test patch explanation: %s", explanation)
            return data.get("test_code") or None
        except Exception as exc:
            logger.error("[debug] Test patch LLM call failed: %s", exc)
            return None


# ------------------------------------------------------------------
# Fallback minimal test template
# ------------------------------------------------------------------

def _minimal_test_template(
    project_name: str,
    root_module: str,
    root_symbol: str,
    test_input: str,
) -> str:
    return textwrap.dedent(f'''\
        """Smoke test for {project_name} — auto-generated by recur-agent debug agent."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))

        import asyncio
        import pytest
        from dotenv import load_dotenv
        load_dotenv()

        from google.adk.runners import InMemoryRunner
        from google.genai.types import Content, Part

        from {root_module} import {root_symbol} as root_agent


        APP_NAME = "{project_name}"
        USER_ID = "debug_user"


        @pytest.mark.asyncio
        async def test_smoke():
            runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
            session = await runner.session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID
            )
            message = Content(role="user", parts=[Part(text={test_input!r})])
            responses = []
            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=session.id,
                new_message=message,
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            responses.append(part.text)
            assert responses, "Agent returned no text response"
    ''')
