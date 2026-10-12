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
$bgeModel = "D:\RuiMeng\memory_rag\new_memory\models\models\BAAI--bge-m3\snapshots\master"
$rerankerModel = "D:\RuiMeng\memory_rag\new_memory\models\models\BAAI--bge-reranker-v2-m3\snapshots\master"
$scope = "scope_06a94e4483964088"
if (-not $Output) {
    $Output = Join-Path $repo "docs\reports\clip-multiroute-bge-baseline-balanced98-20260930.json"
}

$env:NO_PROXY = "127.0.0.1,localhost,::1"
$env:no_proxy = "127.0.0.1,localhost,::1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:SENTRIX_LITE_IMAGE_EMBEDDER = "clip"
$env:SENTRIX_RERANKER_URL = ""
$env:SENTRIX_RERANKER_MODEL_PATH = $rerankerModel
$env:SENTRIX_RERANKER_DEVICE = "cuda"
$env:SENTRIX_RERANKER_BATCH_SIZE = "8"
$env:SENTRIX_RERANKER_MAX_LENGTH = "512"
$env:SENTRIX_RERANKER_IDLE_UNLOAD_SECONDS = "0"
$env:SENTRIX_RERANKER_COARSE_RANK_WEIGHT = "1.0"
$env:SENTRIX_RERANKER_BGE_RANK_WEIGHT = "0.1"
$env:SENTRIX_RERANKER_FUSION_RRF_K = "60"

$runtime = Join-Path $repo "data\runtime"
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$stdout = Join-Path $runtime "clip-multiroute-baseline.stdout.log"
$stderr = Join-Path $runtime "clip-multiroute-baseline.stderr.log"
$arguments = @(
    "-m", "backend.sentrix_lite",
    "--host", "127.0.0.1", "--port", "$SentrixPort",
    "--data-root", $liteData,
    "--qdrant-path", (Join-Path $liteData "qdrant"),
    "--qdrant-collection", "sentrix_memory_units_v1",
    "--text-model-path", $bgeModel,
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
            throw "Sentrix Lite baseline exited early: $details"
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
        throw "Sentrix Lite baseline did not become ready in 180 seconds."
    }
    Write-Output "Sentrix Lite baseline ready on port $SentrixPort"
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
