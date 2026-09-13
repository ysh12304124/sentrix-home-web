# Sentrix Home one-click start (Windows PowerShell)
# Usage:  .\start.ps1                                      connect to independently started local WSL vLLM
#         .\start.ps1 -Restart                             reconnect to local WSL vLLM
#         .\start.ps1 -Restart -LlmBackend ollama           use local Ollama explicitly
#         .\start.ps1 -Restart -GraphRetrievalMode off      disable graph traversal for A/B baseline
#         .\start.ps1 -Status                              show status only
param(
  [switch]$Restart,
  [switch]$Status,
  [ValidateSet("ollama", "vllm")]
  [string]$LlmBackend = "vllm",
  [string]$VllmBaseUrl = "http://127.0.0.1:8000/v1",
  [string]$VllmModel = "gemma4-12b-it",
  [string]$VllmManagerUrl = "",
  [ValidateSet("auto", "on", "off")]
  [string]$GraphRetrievalMode = "auto"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
Set-Location $root

# Keep Python/API logs and Chinese metadata readable on Windows terminals.
# The application data is UTF-8 already; this only controls the launcher
# console and child-process stdio, so it does not change memory semantics.
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# ---------- env ----------
$env:PYTHONPATH = "."
$env:SENTRIX_DATA_DIR  = Join-Path $root "data"
$env:SENTRIX_DB_PATH   = Join-Path $env:SENTRIX_DATA_DIR "sentrix.db"
$env:SENTRIX_ANN_DIR   = Join-Path $env:SENTRIX_DATA_DIR "ann"
$env:SENTRIX_API_PORT  = "11001"
$env:SENTRIX_BACKEND_URL = "http://127.0.0.1:$env:SENTRIX_API_PORT"
$env:PORT              = "11000"
# Local model calls must bypass any desktop/enterprise HTTP proxy. Without this,
# httpx in the API process turns 127.0.0.1 Ollama requests into a 502 Bad Gateway.
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = "127.0.0.1,localhost"

# Qdrant realtime vector chain
$env:SENTRIX_VECTOR_BACKEND = "qdrant"
$env:SENTRIX_QDRANT_PATH    = Join-Path $env:SENTRIX_DATA_DIR "qdrant"

# CLIP embedding.  The project Python has CUDA support on this host; keeping
# image and text embeddings on the 4090 releases the CPU for face detection
# and video extraction.  Face ONNX remains CPU below until a CUDA execution
# provider is installed, which preserves the existing clustering behaviour.
$env:CLIP_ENABLED        = "true"
$env:CLIP_DEVICE         = "cuda:0"
$env:CLIP_ALLOW_DOWNLOAD = "true"
$env:CLIP_MODEL_NAME     = "ViT-B-32"
$env:SENTRIX_IMAGE_EMBEDDER = "clip"
$env:SENTRIX_TEXT_EMBEDDER  = "clip"

# Face recognition (CPU; models auto-download to data/face-models)
$env:FACE_ENABLED      = "true"
$env:FACE_MODEL_ROOT   = Join-Path $env:SENTRIX_DATA_DIR "face-models"
$env:FACE_MODEL_NAME   = "buffalo_l"
$env:FACE_PROVIDERS    = "CPUExecutionProvider"
$env:FACE_EMBEDDING_MODE = "legacy"
$env:RETINAFACE_MODEL_PATH = Join-Path $env:FACE_MODEL_ROOT "retinaface_r50.onnx"

# vLLM is an independently managed local WSL service.  Sentrix only connects
# to its OpenAI-compatible endpoint; it never owns, starts, or stops vLLM.
if ($LlmBackend -eq "vllm") {
  $env:SENTRIX_LLM_BACKEND = "openai"
  $env:SENTRIX_OPENAI_API_MODE = "vllm"
  $env:SENTRIX_RUNTIME_SOURCE = "external"
  if ($VllmManagerUrl) { $env:SENTRIX_RUNTIME_SOURCE = "managed" }
  $env:SENTRIX_VLLM_BASE_URL = $VllmBaseUrl.TrimEnd('/')
  $env:SENTRIX_VLLM_MODEL = $VllmModel
  $env:SENTRIX_VLLM_MANAGER_API = $VllmManagerUrl.TrimEnd('/')
  $env:BENCH_VLLM_BASE_URL = $env:SENTRIX_VLLM_BASE_URL
  $env:BENCH_VLLM_API_URL = $env:SENTRIX_VLLM_MANAGER_API
  # Keep the main-branch split: Agent uses vLLM, while the configured Judge
  # provider remains separate and therefore does not consume vLLM GPU slots.
  Remove-Item Env:BENCH_JUDGE_URL -ErrorAction SilentlyContinue
  Remove-Item Env:BENCH_JUDGE_MODEL -ErrorAction SilentlyContinue
  $env:PHOTOBENCH_QA_CONCURRENCY = "16"
  $env:PHOTOBENCH_JUDGE_CONCURRENCY = "8"
  $env:SENTRIX_ASSISTANT_TURN_WORKERS = "16"
} else {
  # Local Ollama model available on this Windows host.
  $env:SENTRIX_LLM_BACKEND = "ollama"
  $env:SENTRIX_RUNTIME_SOURCE = "external"
  $env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
  $env:OLLAMA_MODEL = "qwen3-vl:4b-instruct"
  $env:OLLAMA_KEEP_ALIVE = "-1"
  $env:OLLAMA_TIMEOUT_SECONDS = "240"
  # Survive transient Ollama connection resets/refusals during long QA runs.
  $env:OLLAMA_RETRY_COUNT = "3"
  # Keep the Ollama server's scheduler/cache bounded for long evaluations.
  # Four is the ingestion ceiling requested for full graph rebuilds.  It is
  # deliberately below the historical 12-way benchmark burst that produced
  # WinError 10054/10061.
  $env:OLLAMA_NUM_PARALLEL = "4"
  $env:OLLAMA_MAX_QUEUE = "64"
  $env:OLLAMA_CONTEXT_LENGTH = "8192"
  # Do not inherit a previous vLLM max_num_seqs snapshot (commonly 12). Two
  # local Agent turns are a bounded concurrency level for the Ollama endpoint.
  $env:PHOTOBENCH_QA_CONCURRENCY = "2"
  $env:PHOTOBENCH_JUDGE_CONCURRENCY = "1"
  $env:SENTRIX_ASSISTANT_TURN_WORKERS = "2"
}
# Face identity assignment is order-sensitive: the online cluster prototype
# is updated as assets are committed.  A multi-worker full rebuild can finish
# the same images in a different detector/SQLite timing order and changes
# precision/recall even when the thresholds are identical.  Keep the memory
# build deterministic (the old 80.9% baseline was produced this way); QA
# concurrency remains independent below.
$env:SENTRIX_PIPELINE_MAX_WORKERS = "1"
$env:SENTRIX_EVENT_SUMMARY_MAX_WORKERS = "1"
$env:SENTRIX_GRAPH_RETRIEVAL_MODE = $GraphRetrievalMode

# Use the existing hybrid extractor, which merges related video frames into
# one memory event and preserves multiple representative keyframes per event.
$env:SENTRIX_VIDEO_KEYFRAME_ALGORITHM = "hybrid_webp"
# Hybrid event analysis sends up to three evidence images plus the temporal
# detector trace; 4096 is just below the prompt size accepted by Ollama.
$env:VISION_CORE_NUM_CTX = "8192"

# Enable the existing incremental person-insight trigger for the default
# household space after an ingest batch completes.  This only enables the
# configured pipeline; ranking and portrait-generation logic remain unchanged.
$env:SENTRIX_PERSON_INSIGHT_SCOPES = "home-default"

# FFmpeg is required by video metadata, preview transcoding, and the vendored
# keyframe workers.  WinGet updates PATH for new shells only, so discover its
# binary directory explicitly when starting the API and propagate it to every
# child process launched by the video pipeline.
$ffmpegBin = $null
try {
  $ffmpegCommand = Get-Command ffmpeg -ErrorAction Stop
  $ffmpegBin = Split-Path $ffmpegCommand.Source
} catch { }
if (-not $ffmpegBin) {
  $wingetPackages = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
  if (Test-Path $wingetPackages) {
    $ffmpegExe = Get-ChildItem -Path $wingetPackages -Recurse -File -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ffmpegExe) { $ffmpegBin = $ffmpegExe.DirectoryName }
  }
}
if ($ffmpegBin) {
  $env:SENTRIX_FFMPEG_BIN = $ffmpegBin
  if (-not (($env:Path -split ';') -contains $ffmpegBin)) { $env:Path = "$ffmpegBin;$env:Path" }
}

# ---------- runtime detection ----------
# Python 3.13 (the original Anaconda-backed .venv runtime) has repeatedly
# crashed inside python313.dll under a long QA run (Windows APPCRASH
# 0xc0000005).  Use the project-compatible 3.11 environment when present;
# dependencies are installed there as well.  Keep .venv as a portable fallback.
$python = "E:\anaconda\envs\magma\python.exe"
if (-not (Test-Path $python)) { $python = Join-Path $root ".venv\Scripts\python.exe" }
if (-not (Test-Path $python)) { Write-Error "python not found: $python" }
$env:PYTHONFAULTHANDLER = "1"

# vLLM is deliberately not started by this project.  Start or inspect it from
# D:\vllm-runtime\start-gemma-vllm.ps1, then start Sentrix normally.
if (-not $Status -and $LlmBackend -eq "vllm" -and $VllmBaseUrl -match "127\.0\.0\.1:8000") {
  try {
    Invoke-RestMethod -Uri "$($env:SENTRIX_VLLM_BASE_URL)/models" -TimeoutSec 3 | Out-Null
  } catch {
    Write-Warning "Local WSL vLLM is not ready. Start it separately: D:\vllm-runtime\start-gemma-vllm.ps1"
  }
}
$node = $null
try { $node = (Get-Command node -ErrorAction Stop).Source } catch { }
if (-not $node -and (Test-Path "C:\Users\VCC\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")) {
  $node = "C:\Users\VCC\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
}
if (-not $node) { Write-Error "node not found. Install Node.js or use the Codex runtime." }

$apiPort = [int]$env:SENTRIX_API_PORT
$webPort = [int]$env:PORT
$photobenchPort = 8771
$logDir  = Join-Path $env:SENTRIX_DATA_DIR "logs"
New-Item -ItemType Directory -Force -Path $logDir, $env:SENTRIX_ANN_DIR, (Join-Path $env:SENTRIX_DATA_DIR "media") | Out-Null

function Test-Port($port) {
  (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) -ne $null
}
function Stop-Port($port) {
  Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}
function Test-Ollama {
  try {
    return (Invoke-WebRequest "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200
  } catch { return $false }
}
function Restart-OllamaWithScheduler {
  # OLLAMA_NUM_PARALLEL is read by `ollama serve`, not by Sentrix clients.
  # Applying it here makes -Restart deterministic instead of retaining the
  # desktop server's old scheduler setting.
  Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Milliseconds 800
  $ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
  if (-not $ollama) { throw "ollama executable not found" }
  $proc = Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden -PassThru
  for ($i = 0; $i -lt 60; $i++) {
    if (Test-Ollama) { "Ollama started (PID $($proc.Id), parallel=$env:OLLAMA_NUM_PARALLEL)"; return }
    Start-Sleep -Milliseconds 500
  }
  throw "Ollama did not become ready on http://127.0.0.1:11434 within 30 seconds"
}

if ($Status) {
  $api = if (Test-Port $apiPort) { "listen" } else { "down" }
  $web = if (Test-Port $webPort) { "listen" } else { "down" }
  "Web  :$webPort $web"
  "API  :$apiPort $api  ($env:SENTRIX_DB_PATH)"
  if ($api -eq "listen") { try { (Invoke-WebRequest "http://127.0.0.1:$apiPort/api/health" -UseBasicParsing -TimeoutSec 3).Content.Substring(0,200) } catch { "health request failed" } }
  exit 0
}

if ($Restart) {
  # PhotoBench is a separate long-lived process.  It must be restarted too,
  # otherwise it keeps the previous SENTRIX_GRAPH_RETRIEVAL_MODE (for example
  # `off`) even after the API/Web processes receive the new mode.
  Stop-Port $apiPort
  Stop-Port $webPort
  Stop-Port $photobenchPort
  if ($LlmBackend -eq "ollama") { Restart-OllamaWithScheduler }
  Start-Sleep -Seconds 1
}
if (Test-Port $apiPort) { Write-Warning "API port $apiPort in use; use -r to restart"; exit 1 }
if (Test-Port $webPort) { Write-Warning "Web port $webPort in use; use -r to restart"; exit 1 }

# ---------- start API ----------
$apiLog  = Join-Path $logDir "api-$apiPort.log"
$apiErr  = Join-Path $logDir "api-$apiPort.err.log"
$apiArgs = @("-m","uvicorn","backend.app:app","--host","127.0.0.1","--port",[string]$apiPort)
$apiProc = Start-Process -FilePath $python -ArgumentList $apiArgs -WorkingDirectory $root `
  -WindowStyle Hidden -RedirectStandardOutput $apiLog -RedirectStandardError $apiErr -PassThru
"API started (PID $($apiProc.Id)): uvicorn 127.0.0.1:$apiPort  log: $apiLog"

# ---------- start Web ----------
$webLog = Join-Path $logDir "web-$webPort.log"
$webProc = Start-Process -FilePath $node -ArgumentList "server.js" -WorkingDirectory $root `
  -WindowStyle Hidden -RedirectStandardOutput $webLog -RedirectStandardError (Join-Path $logDir "web-$webPort.err.log") -PassThru
"Web started (PID $($webProc.Id)): node server.js :$webPort log: $webLog"

# ---------- wait health ----------
$ok = $false
for ($i = 0; $i -lt 40; $i++) {
  Start-Sleep -Milliseconds 500
  try {
    $r = Invoke-WebRequest "http://127.0.0.1:$apiPort/api/health" -UseBasicParsing -TimeoutSec 2
    if ($r.StatusCode -eq 200) { $ok = $true; break }
  } catch { }
}
""
if (-not $ok) {
  "API not ready in 20s; tail of error log:"
  Get-Content $apiErr -Tail 20 -ErrorAction SilentlyContinue
  exit 1
}
$webOk = $false
for ($i = 0; $i -lt 20; $i++) {
  Start-Sleep -Milliseconds 300
  try { if ((Invoke-WebRequest "http://127.0.0.1:$webPort/" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $webOk = $true; break } } catch { }
}

# PhotoBench used to be started only after the UI's asynchronous
# /api/photobench/ensure request.  That made the evaluation page fail whenever
# the browser blocked the new tab or the request was interrupted.  Start the
# local evaluator as part of the normal project boot, while keeping the UI
# ensure endpoint as a safe retry path.  A missing Judge/Ollama endpoint must
# not prevent the core Sentrix Web/API from starting.
$photobenchScript = Join-Path $root "services\photobench\scripts\start-local.ps1"
$photobenchReady = $false
if (Test-Path $photobenchScript) {
  try {
    # Do not pipe the child PowerShell through Out-Host.  In an interactive
    # console that pipeline can remain open after PhotoBench has printed its
    # ready message, making the launcher appear frozen without a prompt.
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $photobenchScript
    $photobenchReady = Test-Port $photobenchPort
    if (-not $photobenchReady) {
      Write-Warning "PhotoBench start command returned, but port 8771 is not listening"
    }
  } catch {
    Write-Warning "PhotoBench evaluator did not start: $($_.Exception.Message)"
  }
}
"Ready:"
"  Web  http://127.0.0.1:$webPort  $(if ($webOk) {'OK'} else {'no response; check log'})"
"  API  http://127.0.0.1:$apiPort/api/health  OK"
"  LLM  $LlmBackend $(if ($LlmBackend -eq 'vllm') { $env:SENTRIX_VLLM_BASE_URL } else { "$env:OLLAMA_BASE_URL / $env:OLLAMA_MODEL" })"
"  QA   http://127.0.0.1:$photobenchPort/  $(if ($photobenchReady) {'OK'} else {'down; use /api/photobench/ensure to retry'})"
"Open browser: http://127.0.0.1:$webPort/"
