#!/bin/sh
# =============================================================================
# 容器入口：补齐 ./volumes 下的运行期目录。
#
# 为什么必须在启动时建、而不是构建时建：
#   docker-compose 用 bind mount 把宿主机的 ./volumes 挂到 /app/volumes，
#   会**覆盖**镜像里 RUN mkdir -p 建好的目录。只要宿主机缺 workspace/ 等目录，
#   容器里就不存在，MCP 文件系统服务会直接连不上：
#     Warning: Cannot access directory /app/volumes/workspace, skipping
#     Error: None of the specified directories are accessible
# =============================================================================
set -e

for dir in workspace checkpoints user-skills milvus-metadata chroma logs; do
    mkdir -p "/app/volumes/$dir"
done

exec "$@"
