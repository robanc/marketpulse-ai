$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $projectRoot "scripts\run_marketpulse.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$launcher`""

$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Sunday `
    -At "6:00PM"

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Write-Host "Schedule configuration prepared."
Write-Host "Task: MarketPulseAI-Weekly"
Write-Host "Frequency: Sunday at 6:00 PM (Windows local time)"
Write-Host "Launcher: $launcher"
Write-Host "No scheduled task has been registered."

# Registration will be added after approval.
