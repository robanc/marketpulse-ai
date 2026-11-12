$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pipeline = Join-Path $projectRoot "src\run_pipeline.py"
$logDir = Join-Path $projectRoot "logs"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss_fffffff"
$logFile = Join-Path $logDir "pipeline_$timestamp.log"

$mutex = [System.Threading.Mutex]::new(
    $false,
    "Local\MarketPulseAI_IncrementalPipeline"
)

$lockAcquired = $false
$exitCode = 1

try {
    try {
        $lockAcquired = $mutex.WaitOne(0)
    }
    catch [System.Threading.AbandonedMutexException] {
        $lockAcquired = $true
        Write-Warning "Recovered an abandoned pipeline mutex."
    }

    if (-not $lockAcquired) {
        Write-Warning "Another MarketPulse AI pipeline is running."
        $exitCode = 2
    }
    else {
        if (-not (Test-Path $python)) {
            throw "Python executable not found: $python"
        }

        if (-not (Test-Path $pipeline)) {
            throw "Pipeline orchestrator not found: $pipeline"
        }

        Set-Location $projectRoot
        $env:MPLBACKEND = "Agg"

        & $python $pipeline --mode incremental *> $logFile
        $exitCode = $LASTEXITCODE

        Write-Host "Pipeline exit code: $exitCode"
        Write-Host "Log file: $logFile"
    }
}
catch {
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    $exitCode = 1
}
finally {
    if ($lockAcquired) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}

exit $exitCode
