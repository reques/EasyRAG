# MinerU 独立解析服务

这是 EasyRAG 旁路部署的 MinerU Pipeline API，不会向 EasyRAG 的 Python 环境安装任何 MinerU 依赖。

## 当前配置

- MinerU：`3.4.4`
- API：`http://127.0.0.1:18000`
- Swagger：`http://127.0.0.1:18000/docs`
- 默认调用后端：请求中显式传递 `backend=pipeline`
- 模型：构建时从 ModelScope 下载并固化到镜像
- 输入：PDF、图片、DOCX、PPTX、XLSX
- 输出：Markdown、content list JSON、图片和其他可选结构化结果
- 并发：单任务执行，其余任务在 MinerU 内部排队
- 长文档窗口：每批最多 16 页，降低 8 GB 显存下的峰值占用

服务只绑定本机回环地址，局域网和公网无法直接访问。如果 EasyRAG 后续运行在 Docker 中，可通过 `host.docker.internal:18000` 访问。

MinerU API 本身没有业务鉴权，因此不要在未增加网关鉴权的情况下把端口改成公网监听。异步任务状态保存在 MinerU 进程内，服务重启后任务 ID 不再可查询；调用方应持久化自己的入库状态，并在任务完成后及时下载结果。当前 MinerU 结果保留时间为 24 小时。

## 运维命令

以下命令都在本目录执行：

```powershell
docker compose -f compose.yml build
docker compose -f compose.yml up -d
docker compose -f compose.yml ps
docker compose -f compose.yml logs -f --tail 200
docker compose -f compose.yml restart
docker compose -f compose.yml down
```

`down` 不会删除解析结果卷。只有明确不再需要历史结果时才执行：

```powershell
docker compose -f compose.yml down --volumes
```

## 验证

仅检查服务：

```powershell
.\smoke-test.ps1
```

解析一份真实文件：

```powershell
.\smoke-test.ps1 -InputFile "D:\docs\sample.pdf"
```

解析结果压缩包写入本目录的 `test-output`。

## EasyRAG 接入参数

宿主机运行 EasyRAG：

```text
MINERU_API_URL=http://127.0.0.1:18000
MINERU_BACKEND=pipeline
```

EasyRAG 容器中运行：

```text
MINERU_API_URL=http://host.docker.internal:18000
MINERU_BACKEND=pipeline
```

不要省略 `backend=pipeline`。MinerU API 的上游默认后端可能是 Hybrid，而本镜像刻意不包含 vLLM/VLM 模型。
