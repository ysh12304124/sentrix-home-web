param(
    [string]$PythonPath = "C:\Users\Admin\miniconda3\envs\memory\python.exe",
    [int]$Port = 8115,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Output) {
    $Output = Join-Path $repo "docs\reports\bge-baseline-retrieval-only-balanced98-20260930.json"
}

$bgeId = "D:\RuiMeng\memory_rag\new_memory\models\models\BAAI--bge-m3\snapshots\master"
$bgePath = $bgeId
$snapshot = Join-Path $repo "data\snapshots\bge-baseline-20260916"
$runtime = Join-Path $repo "data\runtime"
New-Item -ItemType Directory -Force -Path $runtime | Out-Null

$env:NO_PROXY = "127.0.0.1,localhost,::1"
$env:no_proxy = "127.0.0.1,localhost,::1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:SENTRIX_TEXT_EMBEDDER = "bge"
$env:SENTRIX_TEXT_EMBED_MODEL = $bgePath
$env:SENTRIX_TEXT_EMBED_MODEL_ID = $bgeId
$env:SENTRIX_TEXT_EMBEDDER_URL = "http://127.0.0.1:$Port"
$env:SENTRIX_TEXT_EMBEDDER_PORT = "$Port"
$env:SENTRIX_TEXT_EMBEDDER_DEVICE = "cpu"
$env:SENTRIX_TEXT_EMBEDDER_MAX_LENGTH = "512"
$env:SENTRIX_TEXT_EMBEDDING_INSTRUCTION = ""
$env:SENTRIX_IMAGE_EMBEDDER = "clip"
$env:CLIP_CHECKPOINT = "C:\Users\Admin\.cache\clip\ViT-B-32.pt"
$env:CLIP_DEVICE = "cpu"
$env:SENTRIX_VECTOR_BACKEND = "qdrant"
$env:SENTRIX_QDRANT_PATH = Join-Path $snapshot "native-qdrant"
$env:SENTRIX_QDRANT_COLLECTION_PREFIX = "sentrix_local"
$env:SENTRIX_RERANKER_MODEL_PATH = "D:\RuiMeng\memory_rag\new_memory\models\bge-reranker-v2-m3"
$env:SENTRIX_RERANKER_DEVICE = "cuda"
$env:SENTRIX_RERANKER_BATCH_SIZE = "2"
$env:SENTRIX_RERANKER_MAX_LENGTH = "512"
$env:SENTRIX_RERANKER_IDLE_UNLOAD_SECONDS = "0"
$env:SENTRIX_RERANKER_COARSE_RANK_WEIGHT = "1.0"
$env:SENTRIX_RERANKER_BGE_RANK_WEIGHT = "0.1"
$env:SENTRIX_RERANKER_FUSION_RRF_K = "60"

$stdout = Join-Path $runtime "bge-baseline-sidecar.stdout.log"
$stderr = Join-Path $runtime "bge-baseline-sidecar.stderr.log"
$sidecar = Start-Process -FilePath $PythonPath `
    -ArgumentList @("scripts\maintenance\text_embedder_sidecar.py", "--port", "$Port") `
    -WorkingDirectory $repo -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru

try {
    $ready = $false
    for ($i = 0; $i -lt 180; $i++) {
        if ($sidecar.HasExited) {
            $details = Get-Content $stderr -Raw -ErrorAction SilentlyContinue
            throw "BGE sidecar exited early: $details"
        }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" `
                -Proxy $null -TimeoutSec 2
            if ($health.status -eq "ok") {
                $ready = $true
                break
            }
        } catch {}
        Start-Sleep -Seconds 1
    }
    if (-not $ready) {
        throw "BGE sidecar did not become ready in 180 seconds."
    }
    Write-Output "BGE sidecar ready: $($health.model), dim=$($health.dimension)"

    & $PythonPath scripts\benchmarks\evaluate_bge_reranker.py `
        --db (Join-Path $snapshot "sentrix.db") `
        --selection media_only `
        --video-limit 48 `
        --image-limit 50 `
        --out $Output
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluator exited with code $LASTEXITCODE"
    }
} finally {
    if ($sidecar -and -not $sidecar.HasExited) {
        Stop-Process -Id $sidecar.Id -Force
    }
}
