$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found at $python"
}

Set-Location $projectRoot
$env:PYTHONPATH = $projectRoot
& $python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
