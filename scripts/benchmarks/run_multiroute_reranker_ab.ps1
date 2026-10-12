param(
    [string]$PythonPath = "C:\Users\Admin\miniconda3\envs\memory\python.exe",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$snapshot = Join-Path $repo "data\snapshots\bge-baseline-20260916"
$coarseDirectory = Join-Path $repo "data\experiments\multiroute-ab-v2-20261012"
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repo "data\experiments\multiroute-reranker-ab-v2-20261012"
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$bgeReranker = "D:\RuiMeng\memory_rag\new_memory\models\models\BAAI--bge-reranker-v2-m3\snapshots\master"
$qwenReranker = "D:\RuiMeng\memory_rag\new_memory\models\Qwen3-Reranker-0.6B"
foreach ($path in @($PythonPath, $bgeReranker, $qwenReranker)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path is missing: $path"
    }
}

$env:NO_PROXY = "127.0.0.1,localhost,::1"
$env:no_proxy = "127.0.0.1,localhost,::1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:SENTRIX_RERANKER_DEVICE = "cuda"
$env:SENTRIX_RERANKER_BATCH_SIZE = "4"
$env:SENTRIX_RERANKER_MAX_LENGTH = "512"
$env:SENTRIX_RERANKER_IDLE_UNLOAD_SECONDS = "0"
$env:SENTRIX_RERANKER_CLUSTER_WINDOW_SECONDS = "1"
$env:SENTRIX_RERANKER_COARSE_RANK_WEIGHT = "1.0"
$env:SENTRIX_RERANKER_RANK_WEIGHT = "1.0"
$env:SENTRIX_RERANKER_BGE_RANK_WEIGHT = "1.0"
$env:SENTRIX_RERANKER_FUSION_RRF_K = "60"
$env:SENTRIX_RERANKER_SKIP_WHOLE_VIDEO_ONLY = "false"
# Keep the variable present but empty so .env.local cannot repopulate the old
# remote Qwen-VL sidecar URL.  These experiments intentionally use local text
# rerankers selected per encoder family.
$env:SENTRIX_RERANKER_URL = ""

$variants = @(
    @{
        Name = "E0-clip-bge-plus-bge-reranker"
        Cache = "E0-clip-bge-multiroute-balanced98.json"
        Reranker = $bgeReranker
    },
    @{
        Name = "E1-clip-qwen06-plus-qwen06-reranker"
        Cache = "E1-clip-qwen06-multiroute-balanced98.json"
        Reranker = $qwenReranker
    },
    @{
        Name = "E2-qwenvl2b-bge-plus-bge-reranker"
        Cache = "E2-qwenvl2b-bge-multiroute-balanced98.json"
        Reranker = $bgeReranker
    },
    @{
        Name = "E3-qwenvl2b-qwen06-plus-qwen06-reranker"
        Cache = "E3-qwenvl2b-qwen06-multiroute-balanced98.json"
        Reranker = $qwenReranker
    }
)

foreach ($variant in $variants) {
    $cache = Join-Path $coarseDirectory $variant.Cache
    if (-not (Test-Path -LiteralPath $cache)) {
        throw "Missing coarse candidate cache: $cache"
    }
    $env:SENTRIX_RERANKER_MODEL_PATH = $variant.Reranker
    $output = Join-Path $OutputDirectory "$($variant.Name)-balanced98.json"
    Write-Output "Running $($variant.Name)"
    & $PythonPath scripts\benchmarks\evaluate_bge_reranker.py `
        --db (Join-Path $snapshot "sentrix.db") `
        --scope "album3-max-video10-migrated" `
        --selection media_only `
        --candidate-cache-in $cache `
        --video-limit 48 --image-limit 50 `
        --out $output
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluator failed for $($variant.Name): exit $LASTEXITCODE"
    }
}

Write-Output "Saved paired reranker A/B reports: $OutputDirectory"
