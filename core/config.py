"""Maidere configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_fast_model: str = "qwen2.5:3b"
    ollama_thinking_model: str = "deepseek-r1:7b"
    ollama_vision_model: str = "qwen2.5-vl:7b"
    ollama_num_ctx: int = 8192
    enable_model_routing: bool = True
    ollama_timeout: float = 180.0
    ollama_connect_timeout: float = 10.0



    # Database
    db_path: str = "db/maidere.db"

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
    # Every HTTP route except /health, / and /static requires
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

    # Web Search (SearXNG)
    searxng_url: str = "http://localhost:8080"

    # Obsidian Vault Integration
    obsidian_vault_path: str = ""
    obsidian_vault_name: str = ""

    # GitHub Integration
    github_token: str = ""
    github_default_repo: str = "Khazar451/maidere"

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
            self.cloud_api_base
            or self.nvidia_base_url
            or self.openai_base_url
            or "https://integrate.api.nvidia.com/v1"
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
    common_paths = [
        Path.home() / "Documents/Obsidian Vault",
        Path.home() / "Documents/Obsidian",
        Path.home() / "Obsidian Vault",
        Path.home() / "Obsidian",
        (Path.cwd() / settings.agent_workspace / "notes").resolve(),
    ]
    for p in common_paths:
        if p.exists():
            return str(p.resolve()), p.name

    # Default to creating/using standard Documents/Obsidian Vault
    default_path = Path.home() / "Documents/Obsidian Vault"
    return str(default_path), "Obsidian Vault"


