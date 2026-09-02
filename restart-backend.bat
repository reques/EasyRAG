@echo off
chcp 65001 >nul
echo === 重启 EasyRAG 后端 ===
echo.

REM 1. 杀掉所有 python.exe 中运行 uvicorn 的进程
echo [1/3] 停止旧进程...
for /f "tokens=2" %%a in ('tasklist ^| findstr "python.exe"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | findstr "uvicorn" >nul
    if not errorlevel 1 (
        echo   终止 PID %%a
        taskkill /F /PID %%a 2>nul
    )
)

timeout /t 2 /nobreak >nul

REM 2. 清理 Python 缓存
echo [2/3] 清理缓存...
cd /d E:\Learn_Agent\EasyRAG
for /d /r %%d in (__pycache__) do @rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul

REM 3. 启动新服务
echo [3/3] 启动服务...
echo.
echo 服务将在 http://localhost:8000 启动
echo 按 Ctrl+C 停止
echo.
D:\Anaconda3\envs\stage1-agent\python.exe -m uvicorn backend.server.main:app --host 0.0.0.0 --port 8000
