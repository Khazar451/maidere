"""FastAPI application for Maidere.

Lifespan manages:
- Directory creation (db/, workspace/, logs/)
- Database initialization (tables, WAL mode)
- LangGraph agent compilation with AsyncSqliteSaver
- Cleanup on shutdown
"""

import logging
import logging.handlers
import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from maidere import __version__
from maidere.api.auth import AuthMiddleware
from maidere.api.routes import router, set_graph
from maidere.core.agent import create_graph
from maidere.core.config import settings
from maidere.core.db import get_db, init_tables
from maidere.core.tokens import load_or_create_token, token_file_path, token_strength_warning
from maidere.paths import resolve_config_path, resolve_db_path, runtime_root, static_dir, workspace_dir

_LOG_MAX_BYTES = 5 * 1024 * 1024
_LOG_BACKUPS = 3
_STDLIB_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "langgraph", "apscheduler")


def setup_logging() -> None:
    """JSON logs to ``log_path`` (rotated) and the console. stdlib loggers included.

    The file is always JSON. The console is JSON too, except at DEBUG where it
    uses structlog's console renderer. Handlers are replaced on each call so
    tests can reconfigure safely.
    """
    log_path = resolve_config_path(settings.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    debug = settings.log_level.upper() == "DEBUG"

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    structlog.configure(
        processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    def _formatter(console: bool) -> structlog.stdlib.ProcessorFormatter:
        renderer = structlog.dev.ConsoleRenderer() if console and debug else structlog.processors.JSONRenderer()
        return structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUPS, encoding="utf-8"
    )
    file_handler.setFormatter(_formatter(console=False))
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_formatter(console=True))
    root.addHandler(console)

    for name in _STDLIB_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: setup on startup, cleanup on shutdown."""
    log = structlog.get_logger()

    # Ensure runtime directories exist (anchored, not the process cwd).
    db_path = resolve_db_path(settings.db_path)
    workspace_dir()
    resolve_config_path(settings.log_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize database tables
    db = await get_db(db_path)
    await init_tables(db)
    await db.close()

    # Reconcile Tier 2 Auto-Memory index and initialize Tier 1 rules
    from maidere.core.memory_tiers import load_instruction_memory, reconcile_memory_index
    load_instruction_memory()
    reconcile_memory_index()

    # Build and compile the agent graph
    graph, checkpointer_ctx = await create_graph(db_path)
    set_graph(graph)

    # Start persistent scheduler
    from maidere.core.scheduler import shutdown_scheduler, start_scheduler

    start_scheduler(db_path)

    # Resolve (and if needed create) the API token before serving traffic.
    if settings.api_auth_enabled:
        load_or_create_token()
        token_source = "env" if settings.api_token.strip() else str(token_file_path())
        await log.ainfo("api_auth_enabled", token_source=token_source)
        warning = token_strength_warning()
        if warning:
            await log.awarn("api_token_weak", detail=warning)
    else:
        await log.awarn(
            "api_auth_disabled",
            detail="API_AUTH_ENABLED=false: anyone who can reach this port can run maidere.tools.",
        )
    # Report which tool sandbox is in effect; restricted mode is a weaker guarantee.
    from maidere.tools.sandbox import describe as describe_sandbox

    sandbox_info = describe_sandbox()
    if sandbox_info["active"] == "bwrap":
        await log.ainfo("tool_sandbox", **sandbox_info)
    else:
        await log.awarn(
            "tool_sandbox_degraded",
            **sandbox_info,
            detail=(
                "Tool subprocesses are not namespace-isolated. Install `bubblewrap` "
                "(and allow unprivileged user namespaces) for full isolation."
            ),
        )

    if settings.api_host not in ("127.0.0.1", "localhost", "::1"):
        await log.awarn(
            "api_bound_to_non_loopback",
            host=settings.api_host,
            detail="Ensure a TLS-terminating reverse proxy or firewall is in front of this port.",
        )

    await log.ainfo(
        "maidere_started",
        model=settings.ollama_model,
        num_ctx=settings.ollama_num_ctx,
        db_path=db_path,
        workspace=str(workspace_dir()),
        runtime_root=str(runtime_root()),
    )

    yield

    # Cleanup
    shutdown_scheduler()
    await checkpointer_ctx.__aexit__(None, None, None)
    await log.ainfo("maidere_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging()

    app = FastAPI(
        title="Maidere",
        description="A self-hosted autonomous AI agent",
        version=__version__,
        lifespan=lifespan,
    )

    # Middleware order: Starlette wraps in reverse order of add_middleware, so
    # CORS (added last) is outermost and answers preflights itself; the auth
    # middleware sits directly inside it, so no unauthenticated request ever
    # reaches the router.
    app.add_middleware(AuthMiddleware)

    # CORS: only the origins we serve the UI from (loopback for the configured
    # port) plus any explicitly configured extras. No wildcard, no credentials:
    # auth is a bearer header, so cookies are never relied upon.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Maidere-Token"],
    )

    # Include API routes
    app.include_router(router)

    # Instrument Prometheus metrics exporter
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    # Bundled UI. Mounted from the package so it works regardless of cwd.
    ui_dir = static_dir()
    if ui_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = ui_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Maidere Agent API is running. Web UI not found in static/."}

    return app



app = create_app()

