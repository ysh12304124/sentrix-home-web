<#
.SYNOPSIS
  Control the WSL Qwen3-VL vLLM runtime as user hpq.

.EXAMPLE
  .\vllm.ps1 start
  .\vllm.ps1 status
  .\vllm.ps1 logs
  .\vllm.ps1 stop
#>
[CmdletBinding()]
param(
  [ValidateSet('start', 'stop', 'restart', 'status', 'logs')]
  [string]$Action = 'status',
  [int]$Tail = 40
)

$ErrorActionPreference = 'Stop'
$runtimeRoot = 'D:\vllm-runtime'
$manager = Join-Path $runtimeRoot 'start-gemma-vllm.ps1'
$pidFile = Join-Path $runtimeRoot 'qwen3-vl-vllm.windows.pid'
$distro = 'Ubuntu-22.04-vllm-backup'
$logPath = "\\wsl.localhost\$distro\home\hpq\vllm-runtime\logs\qwen3-vl-vllm.log"

if (-not (Test-Path -LiteralPath $manager)) {
  throw "vLLM manager script not found: $manager"
}

function Show-Status {
  & $manager -Status
  $listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
  if ($listener) {
    Write-Host "TCP 8000: LISTEN (PID $($listener[0].OwningProcess))" -ForegroundColor Green
  } else {
    Write-Host 'TCP 8000: not listening' -ForegroundColor Yellow
  }
}

function Wait-Ready {
  param([int]$TimeoutSeconds = 120)
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  do {
    $ready = & $manager -Status 2>$null
    if ($ready -and ($ready -notmatch 'not ready')) {
      Write-Host 'vLLM is ready.' -ForegroundColor Green
      $ready
      return $true
    }
    Start-Sleep -Seconds 5
  } while ((Get-Date) -lt $deadline)

  Write-Warning "vLLM did not become ready within $TimeoutSeconds seconds."
  if (Test-Path -LiteralPath $logPath) {
    Write-Host 'Last vLLM log lines:' -ForegroundColor Yellow
    Get-Content -LiteralPath $logPath -Tail 30
  }
  return $false
}

switch ($Action) {
  'start' {
    if (Test-Path -LiteralPath $pidFile) {
      $rawPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
      $existing = Get-Process -Id ([int]$rawPid) -ErrorAction SilentlyContinue
      if ($existing) {
        Write-Host "Launcher already exists (Windows PID $rawPid); not starting a duplicate." -ForegroundColor Yellow
        [void](Wait-Ready)
        break
      }
    }
    & $manager
    [void](Wait-Ready)
  }
  'stop' {
    & $manager -Stop
    Write-Host 'Stop command sent.' -ForegroundColor Cyan
  }
  'restart' {
    & $manager -Stop
    Start-Sleep -Seconds 2
    & $manager
    [void](Wait-Ready)
  }
  'status' {
    Show-Status
  }
  'logs' {
    if (-not (Test-Path -LiteralPath $logPath)) {
      Write-Host "Log file does not exist yet: $logPath" -ForegroundColor Yellow
    } else {
      Get-Content -LiteralPath $logPath -Tail $Tail
    }
  }
}
