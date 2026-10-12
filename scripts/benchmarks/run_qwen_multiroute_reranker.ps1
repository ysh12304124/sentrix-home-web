param(
    [string]$PythonPath = "C:\Users\Admin\miniconda3\envs\memory\python.exe",
    [int]$SentrixPort = 8091,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$liteRepo = "D:\RuiMeng\memory\_rag\new\_memory\sentrix-home-test"
$liteData = Join-Path $liteRepo "data\.sentrix-lite"
$baselineSnapshot = Join-Path $repo "data\snapshots\bge-baseline-20260916"
$textModel = "D:\RuiMeng\memory_rag\new_memory\models\Qwen3-Embedding-0.6B"
$rerankerModel = "D:\RuiMeng\memory_rag\new_memory\models\Qwen3-VL-Reranker-2B"
$scope = "scope_06a94e4483964088"
if (-not $Output) {
    $Output = Join-Path $repo "docs\reports\qwen-multiroute-qwen-reranker-balanced98-20260930.json"
}

$env:NO_PROXY = "127.0.0.1,localhost,::1"
$env:no_proxy = "127.0.0.1,localhost,::1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:SENTRIX_LITE_IMAGE_EMBEDDER = "qwen3_vl"
$env:SENTRIX_QWEN3_VL_EMBEDDER_URL = "http://127.0.0.1:8102"
$env:SENTRIX_QWEN3_VL_QUERY_INSTRUCTION = "Retrieve images or video keyframes relevant to the user's personal memory question."
$env:SENTRIX_LITE_EMBEDDING_INSTRUCTION = "Given a personal memory question, retrieve media descriptions that contain evidence relevant to answering the question."
$env:SENTRIX_RERANKER_URL = "http://127.0.0.1:8103"
$env:SENTRIX_RERANKER_MODEL_PATH = $rerankerModel
$env:SENTRIX_RERANKER_DEVICE = "cuda"
$env:SENTRIX_RERANKER_BATCH_SIZE = "1"
$env:SENTRIX_RERANKER_MAX_LENGTH = "4096"
$env:SENTRIX_RERANKER_IDLE_UNLOAD_SECONDS = "60"
$env:SENTRIX_RERANKER_TIMEOUT = "600"
$env:SENTRIX_RERANKER_COARSE_RANK_WEIGHT = "1.0"
$env:SENTRIX_RERANKER_BGE_RANK_WEIGHT = "0.1"
$env:SENTRIX_RERANKER_FUSION_RRF_K = "60"

foreach ($url in @("http://127.0.0.1:8102/health", "http://127.0.0.1:8103/health")) {
    try {
        Invoke-RestMethod -Uri $url -Proxy $null -TimeoutSec 5 | Out-Null
    } catch {
        throw "Required Qwen sidecar is unavailable: $url"
    }
}

$runtime = Join-Path $repo "data\runtime"
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$stdout = Join-Path $runtime "qwen-multiroute.stdout.log"
$stderr = Join-Path $runtime "qwen-multiroute.stderr.log"
$arguments = @(
    "-m", "backend.sentrix_lite",
    "--host", "127.0.0.1", "--port", "$SentrixPort",
    "--data-root", $liteData,
    "--qdrant-path", (Join-Path $liteData "qdrant-qwen3-vl-embedding-2b"),
    "--qdrant-collection", "sentrix_memory_units_v1",
    "--text-model-path", $textModel,
    "--text-device", "cpu",
    "--clip-device", "cpu",
    "--video-index-frames", "4"
)
$service = Start-Process -FilePath $PythonPath -ArgumentList $arguments `
    -WorkingDirectory $liteRepo -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru

try {
    $ready = $false
    for ($i = 0; $i -lt 180; $i++) {
        if ($service.HasExited) {
            $details = Get-Content $stderr -Raw -ErrorAction SilentlyContinue
            throw "Qwen multiroute service exited early: $details"
        }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$SentrixPort/health" `
                -Proxy $null -TimeoutSec 2
            if ($health.status -in @("ok", "ready")) {
                $ready = $true
                break
            }
        } catch {}
        Start-Sleep -Seconds 1
    }
    if (-not $ready) {
        throw "Qwen multiroute service did not become ready in 180 seconds."
    }
    Write-Output "Qwen multiroute service ready on port $SentrixPort"
    Write-Output ($health.vector_index | ConvertTo-Json -Compress -Depth 5)

    & $PythonPath scripts\benchmarks\evaluate_bge_reranker.py `
        --db (Join-Path $baselineSnapshot "sentrix.db") `
        --scope "album3-max-video10-migrated" `
        --selection media_only `
        --coarse-url "http://127.0.0.1:$SentrixPort" `
        --coarse-scope $scope `
        --video-limit 48 `
        --image-limit 50 `
        --out $Output
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluator exited with code $LASTEXITCODE"
    }
} finally {
    if ($service -and -not $service.HasExited) {
        Stop-Process -Id $service.Id -Force
    }
}
