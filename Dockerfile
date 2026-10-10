FROM python:3.11-slim-bookworm

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Use Aliyun Debian mirror（清华 TUNA 会对部分网络/出口 IP 返回 403 Forbidden）
RUN sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g' \
        /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        file \
        fonts-wqy-zenhei \
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
    && python -m pip install --default-timeout=1000 --retries=20 \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url "${PYTORCH_INDEX_URL}" \
    && python -m pip install -r requirements.txt --index-url "${PIP_INDEX_URL}"

# uv/uvx：MCP 广场上相当一部分服务以 `uvx <package>` 启动（单独一层，
# 避免改动上一层的重型依赖缓存）。镜像内提供 uvx 后即可直接安装这类服务。
RUN python -m pip install --no-cache-dir --default-timeout=1000 --retries=20 uv \
    --index-url "${PIP_INDEX_URL}"

ARG NPM_REGISTRY=https://registry.npmmirror.com
# Node 运行时：Debian bookworm 自带的 nodejs 18.20.4 缺少 Headers.getSetCookie
# （undici 新 API），12306-mcp 这类 MCP 包会直接报 "get cookie failed"。
# 这里换成官方 Node 22 静态包，走 npmmirror 镜像（nodejs.org 在部分网络不可达）。
# 单独成层：不触碰上面的重型 pip 层缓存。
ARG NODE_VERSION=22.14.0
ARG NODE_MIRROR=https://npmmirror.com/mirrors/node
RUN curl -fsSL "${NODE_MIRROR}/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz" -o /tmp/node.tar.gz \
    && tar -xzf /tmp/node.tar.gz -C /usr/local --strip-components=1 \
    && rm -f /tmp/node.tar.gz \
    && test "$(node --version)" = "v${NODE_VERSION}" \
    && npm --version \
    && npm config set registry "${NPM_REGISTRY}" \
    && npm install -g @modelcontextprotocol/server-filesystem @modelcontextprotocol/server-postgres

COPY app ./app
COPY backend ./backend
COPY skills ./skills
COPY config ./config

RUN mkdir -p /app/volumes/checkpoints /app/volumes/user-skills \
    /app/volumes/milvus-metadata /app/volumes/chroma /app/volumes/workspace

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=10 \
    CMD curl --fail --silent http://127.0.0.1:8000/api/v1/health >/dev/null || exit 1

CMD ["uvicorn", "backend.server.main:app", "--host", "0.0.0.0", "--port", "8000"]