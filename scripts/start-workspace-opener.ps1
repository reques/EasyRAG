$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$openerPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$openerScript = Join-Path $PSScriptRoot 'workspace_opener.py'
$openerLogs = Join-Path $workspaceRoot 'tmp'
try {
    $existingOpener = Invoke-RestMethod 'http://127.0.0.1:18765/health' -TimeoutSec 2
    if ($existingOpener.service -ne 'easyrag-workspace-opener') { throw 'Port 18765 is used by another service' }
    Write-Output 'EasyRAG workspace opener is already running on 127.0.0.1:18765'
    exit 0
} catch { }
if (-not (Test-Path -LiteralPath $openerPython)) { throw 'Missing .venv Python. Install project dependencies first.' }
New-Item -ItemType Directory -Path $openerLogs -Force | Out-Null
$openerProcess = Start-Process -FilePath $openerPython -ArgumentList @('-X', 'utf8', '-u', ('"' + $openerScript + '"')) -WorkingDirectory $workspaceRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $openerLogs 'workspace-opener.log') -RedirectStandardError (Join-Path $openerLogs 'workspace-opener-error.log')
$openerProcess.Id | Set-Content -LiteralPath (Join-Path $openerLogs 'workspace-opener.pid')
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    try {
        $openerHealth = Invoke-RestMethod 'http://127.0.0.1:18765/health' -TimeoutSec 1
        if ($openerHealth.service -eq 'easyrag-workspace-opener') {
            Write-Output 'EasyRAG workspace opener ready on 127.0.0.1:18765'
            exit 0
        }
    } catch { }
    Start-Sleep -Milliseconds 250
}
throw 'Workspace opener failed to start. Check tmp/workspace-opener-error.log.'
