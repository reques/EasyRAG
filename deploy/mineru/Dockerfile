FROM python:3.12-slim-bookworm

ARG MINERU_VERSION=3.4.4
ARG MINERU_MODEL_SOURCE=modelscope

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# 切到清华 Debian 镜像：deb.debian.org 走 Fastly CDN，国内慢且易断连
RUN sed -i \
        -e 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
        -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
        /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        fontconfig \
        fonts-noto-cjk \
        fonts-noto-core \
        libgl1 \
        libglib2.0-0 && \
    fc-cache -f && \
    rm -rf /var/lib/apt/lists/*

# Pipeline is intentionally installed without vLLM. It supports PDF, images,
# DOCX, PPTX and XLSX while fitting the 8 GB GPU available on this host.
# 走清华 PyPI 镜像，避免 pypi.org 国内直连慢
RUN python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    python -m pip install "mineru[pipeline]==${MINERU_VERSION}" -i https://pypi.tuna.tsinghua.edu.cn/simple

# Bake the pipeline models into the image so runtime parsing is independent of
# external model registries. ModelScope is more reliable from mainland China.
RUN mineru-models-download -s "${MINERU_MODEL_SOURCE}" -m pipeline

# MinerU 3.4.4 imports six from its bundled PaddleOCR-derived preprocessing
# code but does not declare it in the published package metadata.
RUN python -m pip install "six==1.17.0" -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV MINERU_MODEL_SOURCE=local \
    MINERU_API_OUTPUT_ROOT=/opt/mineru/output \
    MINERU_API_TASK_RETENTION_SECONDS=86400 \
    MINERU_API_TASK_CLEANUP_INTERVAL_SECONDS=300 \
    MINERU_API_MAX_CONCURRENT_REQUESTS=1 \
    MINERU_PROCESSING_WINDOW_SIZE=16 \
    MINERU_API_DISABLE_ACCESS_LOG=0 \
    CUDA_VISIBLE_DEVICES=0

WORKDIR /opt/mineru
RUN mkdir -p /opt/mineru/output

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=10 \
    CMD curl --fail --silent http://127.0.0.1:8000/health > /dev/null || exit 1

ENTRYPOINT ["mineru-api"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
