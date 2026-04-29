FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    KITSUNE_DATA_DIR=/app/data \
    KITSUNE_MEMORY_DB_DIR=/app/memory_db \
    ENABLE_LOCAL_TOOLS=false \
    ENABLE_SHELL_TOOL=false

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

VOLUME ["/app/data", "/app/memory_db"]

CMD ["uv", "run", "python", "main.py"]
