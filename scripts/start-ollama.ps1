# 以「容器可访问」的方式启动宿主机 Ollama。
#
# 背景：docker 部署（docker-compose.override.yml）里 backend 通过
# host.docker.internal:11434 访问宿主 Ollama，而 Ollama 默认只监听
# 127.0.0.1:11434 —— 容器连不上。这里显式用 OLLAMA_HOST=0.0.0.0:11434 启动。
#
# 幂等：已经在监听且监听地址覆盖所有网卡时，只做探活并退出。
#
# 用法：powershell -ExecutionPolicy Bypass -File scripts\start-ollama.ps1

$ErrorActionPreference = 'Stop'

$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) {
    foreach ($p in @('D:\Programs\Ollama\ollama.exe', "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")) {
        if (Test-Path $p) { $ollama = $p; break }
    }
}
if (-not $ollama) { Write-Error '未找到 ollama.exe，请先安装 Ollama'; exit 1 }

function Show-Models {
    try {
        $tags = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 10
        Write-Host '本地模型：'
        $tags.models | ForEach-Object { Write-Host "  - $($_.name)" }
    } catch {
        Write-Host "探活失败：$($_.Exception.Message)" -ForegroundColor Yellow
    }
}

$listening = Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue
if ($listening) {
    $addrs = ($listening.LocalAddress | Sort-Object -Unique) -join ', '
    Write-Host "Ollama 已在监听 11434：$addrs"
    if ($addrs -notmatch '0\.0\.0\.0|::') {
        Write-Warning '它只监听了回环地址，容器连不上。请先退出 Ollama，再运行本脚本。'
        exit 1
    }
    Show-Models
    exit 0
}

$env:OLLAMA_HOST = '0.0.0.0:11434'
Write-Host "启动 Ollama：$ollama  (OLLAMA_HOST=$env:OLLAMA_HOST)"
Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden
Start-Sleep -Seconds 8
Show-Models
Write-Host ''
Write-Host '提示：需要在宿主机长期生效可执行  setx OLLAMA_HOST 0.0.0.0:11434  （之后重启 Ollama）'
