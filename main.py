"""recur-agent CLI / web entrypoint.

CLI mode (original):
    python main.py --requirement "Build a Slack bot that alerts on Stripe failures" \\
                   --project-name stripe_alerter

Web mode (start the UI, submit from browser):
    python main.py --web

Environment:
    ANTHROPIC_API_KEY  required for the platform engine agents
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("recur-agent")


def _load_settings() -> dict:
    settings_path = Path(__file__).parent / "config" / "settings.yaml"
    if settings_path.exists():
        with settings_path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursively decompose a product requirement into a Google ADK framework."
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Start in web mode: open the browser UI to enter requirement and project name.",
    )
    parser.add_argument(
        "--requirement", "-r",
        default=None,
        help="High-level product requirement (CLI mode only).",
    )
    parser.add_argument(
        "--project-name", "-p",
        default=None,
        help="Snake-case project name used as the output directory (CLI mode only).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the generated project (default: ./output)",
    )
    parser.add_argument(
        "--host", default=None,
        help="Dashboard host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Dashboard port (default: 8000)",
    )
    return parser


async def _run_pipeline_for(
    requirement: str,
    project_name: str,
    output_dir: Path,
    *,
    set_events_fn,
    env_provider=None,
) -> Path:
    """Create a fresh SkillTree + PipelineEvents and run the pipeline."""
    from src.orchestrator.pipeline import PipelineEvents, run_pipeline
    from src.orchestrator.state import SkillNode, SkillTree

    root = SkillNode(
        id=str(uuid.uuid4()),
        name=project_name,
        description=requirement,
        depth=0,
    )
    tree = SkillTree(
        project_name=project_name,
        requirement=requirement,
        root=root,
    )
    events = PipelineEvents()
    events.current_tree = tree
    set_events_fn(events)

    return await run_pipeline(tree, events, output_dir, env_provider=env_provider)


async def _main_web(settings: dict, host: str, port: int, output_dir: Path) -> None:
    """Web mode: serve the landing page; pipeline starts when user submits the form."""
    from src.ui.server import app, debug_env_provider, set_events, set_start_callback

    async def start_callback(
        requirement: str,
        project_name: str,
        custom_output_dir: Optional[str],
    ) -> None:
        out = Path(custom_output_dir) if custom_output_dir else output_dir
        try:
            result = await _run_pipeline_for(
                requirement, project_name, out,
                set_events_fn=set_events,
                env_provider=debug_env_provider,
            )
            logger.info("Done! Generated project at: %s", result)
        except Exception:
            logger.exception("Pipeline failed")

    set_start_callback(start_callback)

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    logger.info("recur-agent web UI running at http://%s:%d", host, port)
    await server.serve()


async def _cli_env_provider(missing_names: list[str]) -> dict[str, str]:
    """Collect missing env vars interactively from stdin in CLI mode."""
    if not missing_names:
        return {}
    print(f"\n[debug] The generated project requires these API keys/env vars: {missing_names}")
    provided: dict[str, str] = {}
    for name in missing_names:
        try:
            value = input(f"  Enter value for {name} (leave blank to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if value:
            provided[name] = value
    return provided


async def _main_cli(args: argparse.Namespace, settings: dict) -> None:
    """CLI mode: pipeline starts immediately from command-line args."""
    from src.ui.server import app, set_events

    output_dir = Path(args.output_dir or settings.get("output_dir", "./output"))
    host = args.host or settings.get("server", {}).get("host", "127.0.0.1")
    port = args.port or settings.get("server", {}).get("port", 8000)

    # We need to wire this so that when the pipeline creates events we share them
    events_holder: list = []

    def _capture_events(ev):
        set_events(ev)
        events_holder.append(ev)

    logger.info("Starting HITL dashboard at http://%s:%d/dashboard", host, port)

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    async def pipeline_task():
        result = await _run_pipeline_for(
            args.requirement,
            args.project_name,
            output_dir,
            set_events_fn=_capture_events,
            env_provider=_cli_env_provider,
        )
        logger.info("Done! Generated project at: %s", result)
        server.should_exit = True

    async with asyncio.TaskGroup() as tg:
        tg.create_task(server.serve(), name="uvicorn")
        tg.create_task(pipeline_task(), name="pipeline")


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    settings = _load_settings()
    output_dir = Path(args.output_dir or settings.get("output_dir", "./output"))
    host = args.host or settings.get("server", {}).get("host", "127.0.0.1")
    port = args.port or settings.get("server", {}).get("port", 8000)

    if args.web:
        asyncio.run(_main_web(settings, host, port, output_dir))
        return

    # CLI mode — require --requirement and --project-name
    if not args.requirement or not args.project_name:
        parser.error("--requirement and --project-name are required in CLI mode (or use --web)")

    import re
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", args.project_name):
        print(
            f"Error: --project-name must be a valid Python identifier, got: {args.project_name!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(_main_cli(args, settings))


if __name__ == "__main__":
    main()
