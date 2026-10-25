"""Maidere configuration via environment variables."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_fast_model: str = "qwen2.5:3b"
    ollama_thinking_model: str = "deepseek-r1:7b"
    ollama_vision_model: str = "qwen2.5vl:7b"
    ollama_num_ctx: int = 8192
    enable_model_routing: bool = True
    ollama_timeout: float = 180.0
    ollama_connect_timeout: float = 10.0

    # Database
    db_path: str = "db/maidere.db"

    # Seconds between background WAL checkpoints (0 disables the loop).
    db_checkpoint_interval_seconds: float = 3600.0

    # Visual memory: how many user turns (the upload turn counts as the first) an
    # uploaded image stays available for follow-up questions, and the most images
    # kept in the checkpoint. 0 turns disables it.
    image_memory_turns: int = 2
    image_memory_max_images: int = 4

    # A small local model sometimes emits a tool call Ollama cannot parse, which
    # leaves a reply with no text and no tool call. Ask again this many times
    # before telling the user the model produced nothing.
    empty_response_retries: int = 2

    # Embeddings (fastembed, CPU-only, zero VRAM)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    # Agent
    agent_workspace: str = "workspace"
    agent_token_limit: int = 6000
    agent_chars_per_token: int = 4
    enable_agentic_deliberation: bool = True
    enable_deep_reasoning: bool = True

    # Logging
    log_path: str = "logs/maidere.jsonl"
    log_level: str = "INFO"

    # API
    # Bind to loopback by default. The agent can run shell commands and read
    # the workspace, so exposing it on 0.0.0.0 without a reverse proxy is unsafe.
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # API authentication
    # Every HTTP route except /health, /live, /ready, / and /static requires
    # `Authorization: Bearer <token>`. The WebSocket requires an auth frame as
    # its first message. If `api_token` is empty, a random token is generated
    # once and persisted to `api_token_file` (mode 0600).
    api_auth_enabled: bool = True
    api_token: str = ""
    api_token_file: str = "~/.config/maidere/token"
    # Extra browser origins allowed for CORS and WebSocket handshakes,
    # comma-separated (e.g. "https://maidere.example.com,http://10.0.0.5:8000").
    # Loopback origins for the configured port are always allowed.
    allowed_origins: str = ""
    # Allow unauthenticated /metrics scrapes from loopback clients
    # (Prometheus runs with network_mode: host in docker-compose.yml).
    metrics_allow_loopback: bool = True

    # Upload limits
    max_upload_bytes: int = 25 * 1024 * 1024

    # Tool sandbox (shell + code_runner subprocesses)
    # auto       -> bwrap if bubblewrap works on this host, else restricted (with a warning)
    # bwrap      -> require bubblewrap; tools refuse to run if it is unavailable
    # restricted -> argument-level confinement only (no namespaces); weaker
    # off        -> no confinement beyond cwd (development only)
    tool_sandbox_mode: str = "auto"
    tool_sandbox_allow_network: bool = False
    tool_sandbox_memory_mb: int = 1024
    tool_sandbox_max_file_mb: int = 64
    tool_sandbox_max_output_chars: int = 20000
    tool_sandbox_timeout_seconds: float = 30.0
    # Extra host paths to expose read-only inside the sandbox, comma-separated
    # (e.g. a Node.js install outside /usr).
    tool_sandbox_extra_ro_paths: str = ""

    # Extra binaries the shell tool may run, comma-separated bare names
    # (e.g. EXTRA_ALLOWED_COMMANDS=uv,pytest,cargo,go,ruff). Empty by default.
    # Still confined to the workspace sandbox; shells, sudo and network clients
    # are always refused.
    extra_allowed_commands: str = Field(
        default="",
        validation_alias=AliasChoices("extra_allowed_commands", "maidere_extra_allowed_commands"),
    )

    # Web Search (SearXNG)
    searxng_url: str = "http://localhost:8080"
    # When SearXNG is unreachable, fall back to Tavily/Serper (if a key is set)
    # and then DuckDuckGo. Queries leave the machine in that case; set to false
    # to keep search strictly local.
    search_fallback_enabled: bool = True
    tavily_api_key: str = ""
    serper_api_key: str = ""

    # Obsidian Vault Integration
    obsidian_vault_path: str = ""
    obsidian_vault_name: str = ""

    # GitHub Integration
    github_token: str = ""
    github_default_repo: str = ""

    # How many chat generations may run at once, and how large one WebSocket
    # frame may be (attachments included).
    max_concurrent_generations: int = 2
    max_ws_message_bytes: int = 2_000_000

    # Refuse /ready when the database volume has less free space than this.
    min_free_disk_mb: int = 100

    # Cloud AI Provider (NVIDIA NIM / OpenAI-compatible endpoints)
    cloud_api_key: str = ""
    cloud_api_base: str = "https://integrate.api.nvidia.com/v1"
    cloud_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    cloud_num_ctx: int = 65536
    cloud_timeout: float = 120.0
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    openai_api_key: str = ""
    openai_base_url: str = ""

    def get_allowed_origins(self) -> list[str]:
        """Browser origins permitted for CORS and WebSocket handshakes.

        Always includes the loopback origins for the configured port, plus any
        comma-separated extras from `allowed_origins`.
        """
        origins = [
            f"http://127.0.0.1:{self.api_port}",
            f"http://localhost:{self.api_port}",
            f"http://[::1]:{self.api_port}",
        ]
        for raw in self.allowed_origins.split(","):
            origin = raw.strip().rstrip("/")
            if origin and origin not in origins:
                origins.append(origin)
        return origins

    def get_effective_cloud_config(self) -> tuple[str, str, str]:
        """Resolve effective cloud API key, base URL, and default model."""
        api_key = self.cloud_api_key or self.nvidia_api_key or self.openai_api_key
        base_url = (
            self.cloud_api_base or self.nvidia_base_url or self.openai_base_url or "https://integrate.api.nvidia.com/v1"
        )
        model = self.cloud_model or "nvidia/nemotron-3-ultra-550b-a55b"
        return api_key.strip(), base_url.strip().rstrip("/"), model.strip()

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()


def get_obsidian_vault_info() -> tuple[str, str]:
    """Resolve active Obsidian vault path and name.

    Returns:
        (vault_path, vault_name) as absolute path and string name.
    """
    import json
    from pathlib import Path

    # 1. User configured settings
    if settings.obsidian_vault_path and Path(settings.obsidian_vault_path).exists():
        path_obj = Path(settings.obsidian_vault_path).expanduser().resolve()
        name = settings.obsidian_vault_name or path_obj.name
        return str(path_obj), name

    # 2. Inspect Flatpak & Native Obsidian configuration files
    candidate_configs = [
        Path.home() / ".var/app/md.obsidian.Obsidian/config/obsidian/obsidian.json",
        Path.home() / ".config/obsidian/obsidian.json",
    ]
    for cfg in candidate_configs:
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
                vaults = data.get("vaults", {})
                for _, vinfo in vaults.items():
                    vpath = vinfo.get("path")
                    if vpath and Path(vpath).exists():
                        p = Path(vpath).resolve()
                        return str(p), p.name
            except Exception:
                pass

    # 3. Common fallback locations
    from maidere.paths import workspace_dir

    common_paths = [
        Path.home() / "Documents/Obsidian Vault",
        Path.home() / "Documents/Obsidian",
        Path.home() / "Obsidian Vault",
        Path.home() / "Obsidian",
        workspace_dir() / "notes",
    ]
    for p in common_paths:
        if p.exists():
            return str(p.resolve()), p.name

    # Default to creating/using standard Documents/Obsidian Vault
    default_path = Path.home() / "Documents/Obsidian Vault"
    return str(default_path), "Obsidian Vault"
