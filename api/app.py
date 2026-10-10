"""FastAPI application for Maidere.

Lifespan manages:
- Directory creation (db/, workspace/, logs/)
- Database initialization (tables, WAL mode)
- LangGraph agent compilation with AsyncSqliteSaver
- Cleanup on shutdown
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.auth import AuthMiddleware
from api.routes import router, set_graph
from core.agent import create_graph
from core.config import settings
from core.db import get_db, init_tables
from core.tokens import load_or_create_token, token_file_path, token_strength_warning


def setup_logging() -> None:
    """Configure structlog for JSON output to logs/maidere.jsonl."""
    log_path = Path(settings.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Map string log level to numeric
    numeric_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if settings.log_level == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: setup on startup, cleanup on shutdown."""
    log = structlog.get_logger()

    # Ensure directories exist
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.agent_workspace).mkdir(parents=True, exist_ok=True)
    Path(settings.log_path).parent.mkdir(parents=True, exist_ok=True)
    Path("static").mkdir(parents=True, exist_ok=True)

    # Initialize database tables
    db = await get_db(settings.db_path)
    await init_tables(db)
    await db.close()

    # Reconcile Tier 2 Auto-Memory index and initialize Tier 1 rules
    from core.memory_tiers import load_instruction_memory, reconcile_memory_index
    load_instruction_memory()
    reconcile_memory_index()

    # Build and compile the agent graph
    graph, checkpointer_ctx = await create_graph(settings.db_path)
    set_graph(graph)

    # Start persistent scheduler
    from core.scheduler import shutdown_scheduler, start_scheduler

    start_scheduler(settings.db_path)

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
            detail="API_AUTH_ENABLED=false: anyone who can reach this port can run tools.",
        )
    # Report which tool sandbox is in effect; restricted mode is a weaker guarantee.
    from tools.sandbox import describe as describe_sandbox

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
        db_path=settings.db_path,
        workspace=settings.agent_workspace,
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
        version="0.1.0",
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

    # Mount static assets if directory exists
    static_dir = Path("static")
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Maidere Agent API is running. Web UI not found in static/."}

    return app



app = create_app()

