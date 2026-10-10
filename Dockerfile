# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /uvx /bin/
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin maidere

WORKDIR /app
COPY --from=build /src/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MAIDERE_DATA_DIR=/data \
    MAIDERE_RUNTIME_ROOT=/data \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_TOKEN_FILE=/data/api-token \
    TOOL_SANDBOX_MODE=restricted

RUN mkdir -p /data && chown maidere:maidere /data
USER maidere
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/live', timeout=3)"

CMD ["/app/.venv/bin/python", "-m", "uvicorn", "maidere.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
