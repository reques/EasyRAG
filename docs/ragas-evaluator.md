# 可选 RagasEvaluator

EasyRAG 默认只运行本地确定性指标，不导入 Ragas。推荐将 Ragas 安装到独立虚拟环境，通过 JSON stdin/stdout worker 执行，避免升级主服务中的 `datasets`、LangChain 和 OpenAI SDK。

## 1. 创建独立环境

Windows PowerShell 示例：

```powershell
python -m venv .venv-ragas
.\.venv-ragas\Scripts\python.exe -m pip install -r requirements-ragas.txt
```

## 2. 启用无 LLM 指标

在 `.env` 中配置：

```dotenv
RAGAS_ENABLED=true
RAGAS_EXECUTION_MODE=process
RAGAS_PYTHON_EXECUTABLE=E:\Learn_Agent\develop\EasyRAG\.venv-ragas\Scripts\python.exe
RAGAS_METRICS=id_context_precision,id_context_recall
RAGAS_TIMEOUT=300
```

这两个 ID 指标不会调用 LLM。Ragas 结果保存在评估运行的 `metrics_json.ragas` 中；Ragas 不可用或执行失败时，本地指标仍会正常保存。

## 3. 可选 LLM 指标

支持的指标名称：

- `context_precision`
- `context_recall`

示例：

```dotenv
RAGAS_METRICS=id_context_precision,id_context_recall,context_precision,context_recall
RAGAS_LLM_BASE_URL=https://api.deepseek.com/v1
RAGAS_LLM_API_KEY=your-key
RAGAS_LLM_MODEL=deepseek-chat
```

LLM 配置留空时回退到 EasyRAG 主 LLM 配置。LLM 指标会增加调用成本并可能产生小幅非确定性，建议只在离线评估运行中启用。

## 4. 同进程模式

仅当主后端环境已经安装兼容版本的 Ragas 时，才可使用：

```dotenv
RAGAS_EXECUTION_MODE=in_process
```

生产环境推荐继续使用 `process`。
