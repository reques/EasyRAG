param(
    [string]$ApiBaseUrl = "http://127.0.0.1:18000",
    [string]$InputFile
)

$ErrorActionPreference = "Stop"

$health = Invoke-RestMethod -Uri "$ApiBaseUrl/health" -Method Get
Write-Host "MinerU health: $($health.status), version: $($health.version)"

if (-not $InputFile) {
    exit 0
}

$resolvedInput = (Resolve-Path -LiteralPath $InputFile).Path
$outputPath = Join-Path $PSScriptRoot "test-output"
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$resultZip = Join-Path $outputPath "mineru-smoke-result.zip"

$curlArgs = @(
    "--fail-with-body",
    "--silent",
    "--show-error",
    "--request", "POST",
    "$ApiBaseUrl/file_parse",
    "--form", "files=@$resolvedInput",
    "--form", "backend=pipeline",
    "--form", "parse_method=auto",
    "--form", "lang_list=ch",
    "--form", "formula_enable=true",
    "--form", "table_enable=true",
    "--form", "return_md=true",
    "--form", "return_content_list=true",
    "--form", "return_images=true",
    "--form", "response_format_zip=true",
    "--output", $resultZip
)

& curl.exe @curlArgs
if ($LASTEXITCODE -ne 0) {
    throw "MinerU parsing request failed with exit code $LASTEXITCODE"
}

Write-Host "Parsing succeeded: $resultZip"
