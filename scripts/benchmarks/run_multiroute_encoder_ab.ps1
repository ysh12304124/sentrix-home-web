param(
    [string]$PythonPath = "C:\Users\Admin\miniconda3\envs\memory\python.exe",
    [string]$QwenPythonPath = "D:\RuiMeng\memory_rag\new_memory\models\qwen3-vl-runtime\Scripts\python.exe",
    [int]$SentrixPort = 8122,
    [int]$QwenVlPort = 8102,
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$liteRepo = "D:\RuiMeng\memory\_rag\new\_memory\sentrix-home-test"
$liteData = Join-Path $liteRepo "data\.sentrix-lite"
$indexRoot = Join-Path $liteData "experiments\multiroute-ab-v2-20261012"
$snapshot = Join-Path $repo "data\snapshots\bge-baseline-20260916"
$runtime = Join-Path $repo "data\runtime\multiroute-ab-v2-20261012"
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repo "data\experiments\multiroute-ab-v2-20261012"
}
New-Item -ItemType Directory -Force -Path $runtime, $OutputDirectory | Out-Null

$bgeModel = "D:\RuiMeng\memory_rag\new_memory\models\models\BAAI--bge-m3\snapshots\master"
$qwenTextModel = "D:\RuiMeng\memory_rag\new_memory\models\Qwen3-Embedding-0.6B"
$qwenVlModel = "D:\RuiMeng\memory_rag\new_memory\models\Qwen3-VL-Embedding-2B"
$qwenVlCode = "D:\RuiMeng\memory_rag\new_memory\models\Qwen3-VL-Embedding-code"
$scope = "scope_06a94e4483964088"

foreach ($path in @($PythonPath, $QwenPythonPath, $bgeModel, $qwenTextModel, $qwenVlModel, $qwenVlCode)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path is missing: $path"
    }
}

$env:NO_PROXY = "127.0.0.1,localhost,::1"
$env:no_proxy = "127.0.0.1,localhost,::1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:SENTRIX_LITE_RRF_K = "60"
$env:SENTRIX_LITE_LEXICAL_RRF_WEIGHT = "1.0"
$env:SENTRIX_QWEN3_VL_EMBEDDER_URL = "http://127.0.0.1:$QwenVlPort"
$env:SENTRIX_QWEN3_VL_QUERY_INSTRUCTION = "Retrieve images or video keyframes relevant to the user's personal memory question."
$env:SENTRIX_LITE_EMBEDDING_INSTRUCTION = "Given a personal memory question, retrieve media descriptions that contain evidence relevant to answering the question."

$variants = @(
    @{ Name = "E0-clip-bge"; Index = "E0-clip-bge"; Image = "clip"; Text = $bgeModel },
    @{ Name = "E1-clip-qwen06"; Index = "E1-clip-qwen06"; Image = "clip"; Text = $qwenTextModel },
    @{ Name = "E2-qwenvl2b-bge"; Index = "E2-qwenvl2b-bge"; Image = "qwen3_vl"; Text = $bgeModel },
    @{ Name = "E3-qwenvl2b-qwen06"; Index = "E3-qwenvl2b-qwen06"; Image = "qwen3_vl"; Text = $qwenTextModel }
)

function Wait-Health([string]$Url, [System.Diagnostics.Process]$Process, [int]$Attempts = 180) {
    for ($attempt = 0; $attempt -lt $Attempts; $attempt++) {
        if ($Process -and $Process.HasExited) {
            throw "Service exited before becoming healthy: $Url"
        }
        try {
            return Invoke-RestMethod -Uri $Url -Proxy $null -TimeoutSec 3
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    throw "Service did not become healthy: $Url"
}

$qwenSidecar = $null
try {
    foreach ($variant in $variants) {
        if ($variant.Image -eq "qwen3_vl" -and -not $qwenSidecar) {
            $env:SENTRIX_QWEN3_VL_MODE = "embedding"
            $env:SENTRIX_QWEN3_VL_MODEL_PATH = $qwenVlModel
            $env:SENTRIX_QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
            $env:SENTRIX_QWEN3_VL_CODE_PATH = $qwenVlCode
            $env:SENTRIX_QWEN3_VL_DEVICE = "cuda"
            $env:SENTRIX_QWEN3_VL_IDLE_UNLOAD_SECONDS = "300"
            $qwenSidecar = Start-Process -FilePath $QwenPythonPath -ArgumentList @(
                "scripts\maintenance\qwen3_vl_retrieval_sidecar.py",
                "--host", "127.0.0.1", "--port", "$QwenVlPort"
            ) -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
              -RedirectStandardOutput (Join-Path $runtime "qwen-vl.stdout.log") `
              -RedirectStandardError (Join-Path $runtime "qwen-vl.stderr.log")
            Wait-Health "http://127.0.0.1:$QwenVlPort/health" $qwenSidecar | Out-Null
        }

        $env:SENTRIX_LITE_IMAGE_EMBEDDER = $variant.Image
        $stdout = Join-Path $runtime "$($variant.Name).stdout.log"
        $stderr = Join-Path $runtime "$($variant.Name).stderr.log"
        $service = Start-Process -FilePath $PythonPath -ArgumentList @(
            "-m", "backend.sentrix_lite",
            "--host", "127.0.0.1", "--port", "$SentrixPort",
            "--data-root", $liteData,
            "--qdrant-path", (Join-Path $indexRoot $variant.Index),
            "--qdrant-collection", "sentrix_memory_units_v1",
            "--text-model-path", $variant.Text,
            "--text-device", "cpu", "--clip-device", "cpu",
            "--video-index-frames", "4"
        ) -WorkingDirectory $liteRepo -WindowStyle Hidden -PassThru `
          -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        try {
            $health = Wait-Health "http://127.0.0.1:$SentrixPort/health" $service
            Write-Output "Running $($variant.Name)"
            Write-Output ($health.vector_index | ConvertTo-Json -Compress -Depth 6)
            $output = Join-Path $OutputDirectory "$($variant.Name)-multiroute-balanced98.json"
            & $PythonPath scripts\benchmarks\evaluate_bge_reranker.py `
                --db (Join-Path $snapshot "sentrix.db") `
                --scope "album3-max-video10-migrated" `
                --selection media_only `
                --coarse-url "http://127.0.0.1:$SentrixPort" `
                --coarse-backend multiroute_rrf `
                --coarse-scope $scope `
                --video-limit 48 --image-limit 50 `
                --skip-rerank --out $output
            if ($LASTEXITCODE -ne 0) {
                throw "Evaluator failed for $($variant.Name): exit $LASTEXITCODE"
            }
        } finally {
            if ($service -and -not $service.HasExited) {
                Stop-Process -Id $service.Id -Force
                $service.WaitForExit()
            }
        }
    }
} finally {
    if ($qwenSidecar -and -not $qwenSidecar.HasExited) {
        Stop-Process -Id $qwenSidecar.Id -Force
    }
}

Write-Output "Saved multiroute A/B reports: $OutputDirectory"
