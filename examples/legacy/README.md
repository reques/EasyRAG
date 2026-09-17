# 旧版入口

这里保留早期 FastAPI 入口和配套 Gradio 页面，供查看旧接口和调试使用。
当前 Vue 前端使用 `backend.server.main:app`，日常启动方式见根目录 README。

在项目根目录运行旧版服务：

```bash
python -m examples.legacy.run
# 开发时可加 --reload
```

另外打开终端，安装可选的 `gradio` 后运行配套页面：

```bash
python -m examples.legacy.gradio_app
```

旧服务默认端口由 `.env` 的 `PORT` 决定。Gradio 默认连接
`http://127.0.0.1:8000/api/v1`，可用 `API_BASE` 覆盖。
根目录启动命令 `python run.py` 和 `python gradio_app.py` 已改为上述模块命令。
