[CmdletBinding()]
param(
    [switch]$KeepModel,
    [switch]$KeepBge,
    [switch]$KeepPhotoBench
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PidDir = Join-Path $RootDir "data\run"

function Stop-ManagedProcess {
    param([string]$Name, [string]$PidFile)
    if (-not (Test-Path -LiteralPath $PidFile)) {
        Write-Host "$Name was not started by this deployment."
        return
    }
    $processId = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
        & taskkill.exe /PID $processId /T /F | Out-Null
        Write-Host "$Name stopped (PID $processId)."
    } else {
        Write-Host "$Name is already stopped."
    }
    Remove-Item -LiteralPath $PidFile -Force
}

Stop-ManagedProcess "Web" (Join-Path $PidDir "web.pid")
Stop-ManagedProcess "API" (Join-Path $PidDir "api.pid")
if (-not $KeepPhotoBench) { Stop-ManagedProcess "PhotoBench" (Join-Path $PidDir "photobench.pid") }
if (-not $KeepBge) { Stop-ManagedProcess "BGE" (Join-Path $PidDir "bge.pid") }
if (-not $KeepModel) { Stop-ManagedProcess "Qwen" (Join-Path $PidDir "qwen.pid") }
