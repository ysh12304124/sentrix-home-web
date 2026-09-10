[CmdletBinding()]
param(
    [string]$CondaEnv = "memory",
    [string]$PythonPath = "",
    [int]$ApiPort = 0,
    [int]$WebPort = 0,
    [int]$ModelPort = 0,
    [int]$BgePort = 0,
    [int]$PhotoBenchPort = 0,
    [int]$ModelIdleUnloadSeconds = 0,
    [switch]$EagerLoadModel,
    [switch]$SkipBge,
    [switch]$SkipPhotoBench,
    [switch]$Restart,
    [switch]$Status
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$DataDir = Join-Path $RootDir "data"
$LogDir = Join-Path $DataDir "logs"
$PidDir = Join-Path $DataDir "run"

function Import-DotEnv {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) { continue }
        $key, $value = $trimmed.Split("=", 2)
        $key = $key.Trim()
        $value = $value.Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

function Set-DefaultEnv {
    param([string]$Name, [string]$Value)
    $current = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($current)) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

function Resolve-CondaRuntime {
    if ($PythonPath) {
        $python = (Resolve-Path -LiteralPath $PythonPath).Path
        $environmentNode = Join-Path (Split-Path $python -Parent) "node.exe"
        $node = if (Test-Path -LiteralPath $environmentNode) {
            $environmentNode
        } else {
            (Get-Command node.exe -ErrorAction Stop).Source
        }
        return @{ Python = $python; Node = $node }
    }

    # ``conda activate`` exposes the selected environment through
    # CONDA_PREFIX, even when ``conda`` itself is a PowerShell function or a
    # conda.bat shim rather than conda.exe on PATH. Prefer that authoritative
    # location so the normal `(memory)` prompt works without extra flags.
    $activePrefix = [Environment]::GetEnvironmentVariable("CONDA_PREFIX", "Process")
    if ($activePrefix -and (Split-Path $activePrefix -Leaf) -eq $CondaEnv) {
        $python = Join-Path $activePrefix "python.exe"
        $node = Join-Path $activePrefix "node.exe"
        if (-not (Test-Path -LiteralPath $python)) {
            throw "Python is missing from active Conda environment '$CondaEnv': $python"
        }
        if (-not (Test-Path -LiteralPath $node)) {
            throw "Node.js is missing from active Conda environment '$CondaEnv'. Run: conda install -n $CondaEnv -c conda-forge nodejs=22"
        }
        return @{ Python = $python; Node = $node }
    }

    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $conda) { $conda = Get-Command conda.exe -ErrorAction SilentlyContinue }
    if (-not $conda) { throw "Conda was not found. Open an Anaconda PowerShell terminal or pass -PythonPath." }
    $condaCommand = if ($conda.CommandType -eq "Application") { $conda.Source } else { $conda.Name }
    $envPaths = (& $condaCommand env list --json | ConvertFrom-Json).envs
    $envPath = $envPaths | Where-Object { (Split-Path $_ -Leaf) -eq $CondaEnv } | Select-Object -First 1
    if (-not $envPath) { throw "Conda environment '$CondaEnv' was not found." }
    $python = Join-Path $envPath "python.exe"
    $node = Join-Path $envPath "node.exe"
    if (-not (Test-Path -LiteralPath $python)) { throw "Python is missing from '$CondaEnv': $python" }
    if (-not (Test-Path -LiteralPath $node)) { throw "Node.js is missing from '$CondaEnv'. Run: conda install -n $CondaEnv -c conda-forge nodejs=22" }
    return @{ Python = $python; Node = $node }
}

function Test-Url {
    param([string]$Url, [int]$TimeoutMilliseconds = 2500)
    try {
        $request = [System.Net.HttpWebRequest]::Create($Url)
        $request.Method = "GET"
        $request.Proxy = $null
        $request.Timeout = $TimeoutMilliseconds
        $response = $request.GetResponse()
        $response.Close()
        return $true
    }
    catch { return $false }
}

function Test-PortInUse {
    param([int]$Port)
    # Get-NetTCPConnection can block for minutes on some Windows hosts while
    # the networking CIM provider is busy.  Reading the active TCP listeners
    # through .NET is local, fast, and sufficient for this startup guard.
    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    return [bool]($listeners | Where-Object { $_.Port -eq $Port } | Select-Object -First 1)
}

function Wait-ForUrl {
    param(
        [string]$Url,
        [System.Diagnostics.Process]$Process,
        [string]$ErrorLog,
        [int]$TimeoutSeconds = 60
    )
    $attempts = [Math]::Max(1, $TimeoutSeconds * 2)
    for ($attempt = 0; $attempt -lt $attempts; $attempt++) {
        if (Test-Url $Url) { return }
        $Process.Refresh()
        if ($Process.HasExited) {
            $details = if (Test-Path -LiteralPath $ErrorLog) {
                (Get-Content -LiteralPath $ErrorLog -Tail 60) -join [Environment]::NewLine
            } else { "No error log was created." }
            throw "Process exited before $Url became available.`n$details"
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Timed out waiting for $Url. See $ErrorLog"
}

function Start-ManagedProcess {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory,
        [string]$PidFile,
        [string]$HealthUrl,
        [int]$Port,
        [int]$TimeoutSeconds = 60
    )
    if (Test-Url $HealthUrl) {
        Write-Host "$Name already available: $HealthUrl"
        return $null
    }
    if (Test-PortInUse $Port) {
        throw "$Name cannot start because port $Port is occupied by a different service."
    }
    $outLog = Join-Path $LogDir "$Name.log"
    $errorLog = Join-Path $LogDir "$Name.error.log"
    $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $outLog `
        -RedirectStandardError $errorLog -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $PidFile -Value $process.Id
    try {
        Wait-ForUrl $HealthUrl $process $errorLog $TimeoutSeconds
    }
    catch {
        if (Test-Path -LiteralPath $PidFile) { Remove-Item -LiteralPath $PidFile -Force }
        throw
    }
    Write-Host "$Name started: $HealthUrl (PID $($process.Id))"
    return $process
}

function Show-Status {
    $rows = @(
        @{ Name = "Qwen"; Url = "http://127.0.0.1:$ModelPort/v1/models" },
        @{ Name = "BGE"; Url = "http://127.0.0.1:$BgePort/health" },
        @{ Name = "API"; Url = "http://127.0.0.1:$ApiPort/api/health" },
        @{ Name = "Web"; Url = "http://127.0.0.1:$WebPort/" },
        @{ Name = "PhotoBench"; Url = "http://127.0.0.1:$PhotoBenchPort/api/config" }
    )
    foreach ($row in $rows) {
        $state = if (Test-Url $row.Url) { "ready" } else { "stopped" }
        Write-Host ("{0,-11} {1,-7} {2}" -f $row.Name, $state, $row.Url)
    }
    Write-Host "Data        $DataDir"
}

Import-DotEnv (Join-Path $RootDir ".env.local")

if ($ApiPort -le 0) { $ApiPort = [int]$(if ($env:SENTRIX_API_PORT) { $env:SENTRIX_API_PORT } else { "8090" }) }
if ($WebPort -le 0) { $WebPort = [int]$(if ($env:PORT) { $env:PORT } else { "4174" }) }
if ($ModelPort -le 0) { $ModelPort = [int]$(if ($env:LOCAL_QWEN_PORT) { $env:LOCAL_QWEN_PORT } else { "11434" }) }
if ($BgePort -le 0) { $BgePort = [int]$(if ($env:SENTRIX_TEXT_EMBEDDER_PORT) { $env:SENTRIX_TEXT_EMBEDDER_PORT } else { "8101" }) }
if ($PhotoBenchPort -le 0) { $PhotoBenchPort = [int]$(if ($env:PHOTOBENCH_PORT) { $env:PHOTOBENCH_PORT } else { "8772" }) }
if ($ModelIdleUnloadSeconds -le 0) { $ModelIdleUnloadSeconds = [int]$(if ($env:LOCAL_QWEN_IDLE_UNLOAD_SECONDS) { $env:LOCAL_QWEN_IDLE_UNLOAD_SECONDS } else { "60" }) }

if ($Status) { Show-Status; exit 0 }

New-Item -ItemType Directory -Path $DataDir, (Join-Path $DataDir "media"), (Join-Path $DataDir "ann"), (Join-Path $DataDir "qdrant"), $LogDir, $PidDir -Force | Out-Null
if ($Restart) { & (Join-Path $PSScriptRoot "stop_local.ps1") }

$runtime = Resolve-CondaRuntime
$Python = $runtime.Python
$Node = $runtime.Node
$ModelName = if ($env:LOCAL_QWEN_MODEL_NAME) { $env:LOCAL_QWEN_MODEL_NAME } else { "Qwen3-VL-4B-Instruct" }
$ModelUrl = "http://127.0.0.1:$ModelPort"
$BgeUrl = "http://127.0.0.1:$BgePort"
$ApiUrl = "http://127.0.0.1:$ApiPort"

Set-DefaultEnv "NO_PROXY" "127.0.0.1,localhost"
Set-DefaultEnv "no_proxy" "127.0.0.1,localhost"
Set-DefaultEnv "HF_HUB_OFFLINE" "1"
Set-DefaultEnv "TRANSFORMERS_OFFLINE" "1"
$env:PYTHONPATH = $RootDir

if (-not (Test-Url "$ModelUrl/v1/models")) {
    $ModelPath = $env:LOCAL_QWEN_MODEL_PATH
    if (-not $ModelPath -or -not (Test-Path -LiteralPath $ModelPath -PathType Container)) {
        throw "LOCAL_QWEN_MODEL_PATH in .env.local must point to the complete local Qwen3-VL model directory."
    }
    $modelArgs = @(
        "-m", "backend.local_qwen_server", "--model-path", $ModelPath,
        "--model-name", $ModelName, "--host", "127.0.0.1", "--port", "$ModelPort",
        "--idle-unload-seconds", "$ModelIdleUnloadSeconds"
    )
    if ($EagerLoadModel) { $modelArgs += "--eager-load" }
    Start-ManagedProcess "qwen" $Python $modelArgs $RootDir (Join-Path $PidDir "qwen.pid") "$ModelUrl/v1/models" $ModelPort 30 | Out-Null
} else {
    Write-Host "Qwen endpoint reused: $ModelUrl/v1 (idle unloading remains enabled)."
}

if ($SkipBge) {
    $env:SENTRIX_TEXT_EMBEDDER = "clip"
} else {
    $BgeModel = $env:LOCAL_BGE_MODEL_PATH
    if (-not $BgeModel -or -not (Test-Path -LiteralPath $BgeModel -PathType Container)) {
        throw "LOCAL_BGE_MODEL_PATH in .env.local must point to the complete local BGE-M3 model directory."
    }
    $env:SENTRIX_TEXT_EMBEDDER = "bge"
    $env:SENTRIX_TEXT_EMBED_MODEL = $BgeModel
    $env:SENTRIX_TEXT_EMBEDDER_DEVICE = "cpu"
    $env:SENTRIX_TEXT_EMBEDDER_HOST = "127.0.0.1"
    $env:SENTRIX_TEXT_EMBEDDER_PORT = "$BgePort"
    $env:SENTRIX_TEXT_EMBEDDER_URL = $BgeUrl
    Start-ManagedProcess "bge" $Python @("scripts/maintenance/text_embedder_sidecar.py", "--host", "127.0.0.1", "--port", "$BgePort") $RootDir (Join-Path $PidDir "bge.pid") "$BgeUrl/health" $BgePort 180 | Out-Null
}

$env:SENTRIX_DATA_DIR = $DataDir
$env:SENTRIX_DB_PATH = Join-Path $DataDir "sentrix.db"
$env:SENTRIX_ANN_DIR = Join-Path $DataDir "ann"
$env:SENTRIX_VECTOR_BACKEND = "qdrant"
$env:SENTRIX_QDRANT_PATH = Join-Path $DataDir "qdrant"
$env:SENTRIX_QDRANT_COLLECTION_PREFIX = "sentrix_local"
$env:SENTRIX_LLM_BACKEND = "openai"
$env:SENTRIX_VLLM_BASE_URL = "$ModelUrl/v1"
$env:SENTRIX_VLLM_MODEL = $ModelName
$env:SENTRIX_OPENAI_API_MODE = "generic"
$env:SENTRIX_RUNTIME_SOURCE = "external"
$env:SENTRIX_VLLM_MANAGER_API = ""
$env:SENTRIX_VLLM_API_URL = ""
$env:SENTRIX_MODEL_SPLIT_V1 = "0"
$env:SENTRIX_PARSE_BACKEND = "openai"
$env:SENTRIX_PARSE_BASE_URL = "$ModelUrl/v1"
$env:SENTRIX_PARSE_MODEL = $ModelName
$env:SENTRIX_ANSWER_MODEL = $ModelName
$env:SENTRIX_VERIFY_MODEL = $ModelName
$env:SENTRIX_CLAIM_MODEL = $ModelName
$env:SENTRIX_REPAIR_MODEL = $ModelName
$env:OLLAMA_MODEL = $ModelName
$env:SENTRIX_IMAGE_EMBEDDER = "clip"
Set-DefaultEnv "CLIP_DEVICE" "cpu"
Set-DefaultEnv "CLIP_MODEL_NAME" "ViT-B-32"
Set-DefaultEnv "FACE_ENABLED" "false"
Set-DefaultEnv "FACE_PROVIDERS" "CPUExecutionProvider"
Set-DefaultEnv "FUNASR_DEVICE" "cpu"
Set-DefaultEnv "SENTRIX_PIPELINE_MAX_WORKERS" "1"
Set-DefaultEnv "SENTRIX_EVENT_SUMMARY_MAX_WORKERS" "1"
Set-DefaultEnv "SENTRIX_ASSISTANT_TURN_WORKERS" "2"
Set-DefaultEnv "SENTRIX_THIN_AGENT_V1" "1"
Set-DefaultEnv "SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1" "1"
Set-DefaultEnv "SENTRIX_EVIDENCE_RETRIEVAL_V1" "1"
Set-DefaultEnv "SENTRIX_ANN_INDEX_V1" "1"
Set-DefaultEnv "SENTRIX_CORE_MEMORY_V1" "1"
Set-DefaultEnv "SENTRIX_CONVERSATION_STORE_V1" "1"

Start-ManagedProcess "api" $Python @("-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "$ApiPort") $RootDir (Join-Path $PidDir "api.pid") "$ApiUrl/api/health" $ApiPort 90 | Out-Null

$env:BENCH_SENTRIX_URL = $ApiUrl
$env:BENCH_VLLM_API_URL = ""
$env:BENCH_VLLM_BASE_URL = "$ModelUrl/v1"
$env:BENCH_BIG_MODEL_ENABLED = "true"
$env:BENCH_BIG_MODEL_BASE_URL = "$ModelUrl/v1"
$env:BENCH_BIG_MODEL_MODEL = $ModelName
$env:BENCH_JUDGE_URL = "$ModelUrl/v1"
$env:BENCH_JUDGE_MODEL = $ModelName
$env:PHOTOBENCH_QA_CONCURRENCY = "1"
$env:PHOTOBENCH_JUDGE_CONCURRENCY = "1"
$env:PHOTOBENCH_PORT = "$PhotoBenchPort"
$env:PHOTOBENCH_PYTHON = $Python
if (-not $SkipPhotoBench) {
    $photoRoot = Join-Path $RootDir "services\photobench"
    $photoScript = Join-Path $photoRoot "backend\benchmark_orchestrator.py"
    Start-ManagedProcess "photobench" $Python @($photoScript, "--host", "127.0.0.1", "--port", "$PhotoBenchPort") $photoRoot (Join-Path $PidDir "photobench.pid") "http://127.0.0.1:$PhotoBenchPort/api/config" $PhotoBenchPort 60 | Out-Null
}

$env:PORT = "$WebPort"
$env:SENTRIX_BACKEND_URL = $ApiUrl
Start-ManagedProcess "web" $Node @("server.js") $RootDir (Join-Path $PidDir "web.pid") "http://127.0.0.1:$WebPort/" $WebPort 30 | Out-Null

Write-Host ""
Show-Status
Write-Host ""
Write-Host "Sentrix Home is ready: http://127.0.0.1:$WebPort/"
Write-Host "Qwen loads on the first model request and unloads after ${ModelIdleUnloadSeconds}s idle."
