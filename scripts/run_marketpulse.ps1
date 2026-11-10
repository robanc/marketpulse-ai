$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pipeline = Join-Path $projectRoot "src\run_pipeline.py"
$logDir = Join-Path $projectRoot "logs"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "pipeline_$timestamp.log"

Set-Location $projectRoot

$env:MPLBACKEND = "Agg"

& $python $pipeline --mode cached *> $logFile

$exitCode = $LASTEXITCODE

Write-Host "Pipeline exit code: $exitCode"
Write-Host "Log file: $logFile"

exit $exitCode
