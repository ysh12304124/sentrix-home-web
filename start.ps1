# Sentrix Home one-click start (Windows PowerShell)
# Usage:  .\start.ps1            start Web(11000) + API(11001)
#         .\start.ps1 -r         stop ports first then start
#         .\start.ps1 -Status    show status only
param(
  [switch]$Restart,
  [switch]$Status
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
Set-Location $root

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

# CLIP embedding (CPU; first run downloads weights)
$env:CLIP_ENABLED        = "true"
$env:CLIP_DEVICE         = "cpu"
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

# Local Ollama model available on this Windows host.  Mark it external so a
# stale vLLM registry cannot replace the live client during FastAPI startup.
$env:SENTRIX_LLM_BACKEND = "ollama"
$env:SENTRIX_RUNTIME_SOURCE = "external"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:OLLAMA_MODEL = "qwen3-vl:4b-instruct"
$env:OLLAMA_KEEP_ALIVE = "-1"
$env:OLLAMA_TIMEOUT_SECONDS = "240"
$env:SENTRIX_PIPELINE_MAX_WORKERS = "1"
$env:SENTRIX_EVENT_SUMMARY_MAX_WORKERS = "1"

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
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { Write-Error "python not found: $python" }
$node = $null
try { $node = (Get-Command node -ErrorAction Stop).Source } catch { }
if (-not $node -and (Test-Path "C:\Users\VCC\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")) {
  $node = "C:\Users\VCC\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
}
if (-not $node) { Write-Error "node not found. Install Node.js or use the Codex runtime." }

$apiPort = [int]$env:SENTRIX_API_PORT
$webPort = [int]$env:PORT
$logDir  = Join-Path $env:SENTRIX_DATA_DIR "logs"
New-Item -ItemType Directory -Force -Path $logDir, $env:SENTRIX_ANN_DIR, (Join-Path $env:SENTRIX_DATA_DIR "media") | Out-Null

function Test-Port($port) {
  (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) -ne $null
}
function Stop-Port($port) {
  Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}

if ($Status) {
  $api = if (Test-Port $apiPort) { "listen" } else { "down" }
  $web = if (Test-Port $webPort) { "listen" } else { "down" }
  "Web  :$webPort $web"
  "API  :$apiPort $api  ($env:SENTRIX_DB_PATH)"
  if ($api -eq "listen") { try { (Invoke-WebRequest "http://127.0.0.1:$apiPort/api/health" -UseBasicParsing -TimeoutSec 3).Content.Substring(0,200) } catch { "health request failed" } }
  exit 0
}

if ($Restart) { Stop-Port $apiPort; Stop-Port $webPort; Start-Sleep -Seconds 1 }
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
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $photobenchScript | Out-Host
    $photobenchReady = $true
  } catch {
    Write-Warning "PhotoBench evaluator did not start: $($_.Exception.Message)"
  }
}
"Ready:"
"  Web  http://127.0.0.1:$webPort  $(if ($webOk) {'OK'} else {'no response; check log'})"
"  API  http://127.0.0.1:$apiPort/api/health  OK"
"  QA   http://127.0.0.1:8771/  $(if ($photobenchReady) {'OK'} else {'down; use /api/photobench/ensure to retry'})"
"Open browser: http://127.0.0.1:$webPort/"
