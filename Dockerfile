FROM python:3.11-slim-bookworm

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# curl is used by the container healthcheck. Node/npm keeps the existing
# stdio filesystem MCP server available inside the backend container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        file \
        libgl1 \
        libglib2.0-0 \
        libmagic1 \
        nodejs \
        npm \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip --index-url "${PIP_INDEX_URL}" \
    && python -m pip install \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url "${PYTORCH_INDEX_URL}" \
    && python -m pip install -r requirements.txt --index-url "${PIP_INDEX_URL}"

COPY app ./app
COPY backend ./backend
COPY skills ./skills
COPY mcp_servers.docker.json ./mcp_servers.json

RUN mkdir -p /app/volumes/checkpoints /app/volumes/user-skills \
    /app/volumes/milvus-metadata /app/volumes/chroma /app/volumes/workspace

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=10 \
    CMD curl --fail --silent http://127.0.0.1:8000/api/v1/health >/dev/null || exit 1

CMD ["uvicorn", "backend.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
