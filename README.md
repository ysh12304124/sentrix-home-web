# Sentrix Home

Sentrix Home is a local-first household memory system. It turns original media
into evidence-backed episodic, semantic, and visual memory. The canonical
product definition, architecture, data contracts, acceptance results, and
current work queue are maintained in [docs/PROJECT_MEMORY.md](docs/PROJECT_MEMORY.md).

## Repository Layout

- `backend/`: FastAPI application, SQLite memory store, media pipeline, model
  adapters, identity clustering, and Python regression tests.
- `src/`: browser application code and styles.
- `scripts/runtime/`: project-local runtime utilities.
- `scripts/maintenance/`: destructive or long-running maintenance commands.
- `scripts/benchmarks/`: controlled clustering evaluation tools.
- `scripts/fixtures/`: reproducible public test-data and metadata generators.
- `test/`: Node-based frontend and repository-layout regression tests.
- `docs/`: live project memory and approved design/implementation records.

## Windows local deployment

The Windows launcher uses one Conda environment for the API, local model
gateway, BGE sidecar, PhotoBench, and Node.js.  Copy `.env.local.example` to
`.env.local`, then point the model paths at local Qwen3-VL, BGE-M3,
BGE-reranker-v2-m3, and OpenAI CLIP weights.

One-time setup:

```powershell
conda activate memory
python -m pip install -r backend\requirements.txt
conda install -c conda-forge nodejs=22
```

Start, inspect, and stop the local stack:

```powershell
conda activate memory
powershell -ExecutionPolicy Bypass -File .\scripts\runtime\start_local.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\runtime\start_local.ps1 -Status
powershell -ExecutionPolicy Bypass -File .\scripts\runtime\stop_local.ps1
```

Defaults are Web `4174`, API `8090`, BGE `8101`, local Qwen `11434`, and
PhotoBench `8772`.  The API stores SQLite, media, ANN data, and embedded Qdrant
under `data/`.  Qwen loads on the first request and unloads after 60 idle
seconds.  Face recognition is disabled in the example configuration because
it additionally requires local InsightFace model weights.

### Hybrid retrieval and reranking

When `SENTRIX_RERANKER_ENABLED=true`, retrieval keeps the Top-50 candidates
from the multi-channel RRF coarse stage, groups keyframes from the same video
within one second, and reranks cluster text with the local
`bge-reranker-v2-m3` cross-encoder.  The text contains caption, OCR, object,
object description, relation, and action/event fields; `visual_text` is used
only when those semantic fields are empty.  The reranker loads lazily and is
unloaded after the configured idle interval.  Set the flag to `false` to
return to the original coarse ranking.

A migrated Sentrix Lite named-vector collection can be configured with the
three `SENTRIX_LITE_NAMED_VECTOR_*` variables.  The main API then reads its
CLIP image, visual/caption, object, and relation vectors directly and fuses
them with lexical retrieval; no service on port `8091` is required.

Run the controlled media-retrieval ablation with:

```powershell
conda activate memory
powershell -ExecutionPolicy Bypass -File .\scripts\runtime\stop_local.ps1 `
  -KeepModel -KeepBge -KeepPhotoBench
python scripts\benchmarks\evaluate_bge_reranker.py `
  --out docs\reports\bge-reranker-ablation.json
powershell -ExecutionPolicy Bypass -File .\scripts\runtime\start_local.ps1
```

The API is stopped during this command because embedded Qdrant permits only
one process to own a local collection directory at a time.

## Legacy network deployment (reference only)

The commands below describe the upstream team's former LAN deployment. They
are not used by the Windows local stack above and can be ignored for local use.

```bash
cd /home/asus/Github/Sentrix-Home-Web
scripts/runtime/start_sentrix_ollama.sh

FACE_MODEL_ROOT=$PWD/data/face-models \
FACE_MODEL_NAME=buffalo_l \
FACE_PROVIDERS=CPUExecutionProvider \
FACE_EMBEDDING_MODE=adaface \
ADAFACE_ARCHITECTURE=ir_50 \
ADAFACE_DEVICE=cuda \
ADAFACE_MODEL_PATH=/home/asus/models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt \
ADAFACE_REPO_ROOT=/home/asus/models/AdaFace \
OLLAMA_BASE_URL=http://127.0.0.1:11435 \
OLLAMA_MODEL=gemma4:12b \
OLLAMA_KEEP_ALIVE=-1 \
.venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8090
```

Start the web gateway separately:

```bash
SENTRIX_BACKEND_URL=http://127.0.0.1:8090 PORT=4174 npm run dev
```

That deployment previously used a separate LAN host for the WorldMM timeline.
The Windows launcher replaces those connections with the local API and local
model services configured in `.env.local`.

## Verify

```bash
.venv/bin/python -m unittest discover -s backend/tests -v
npm test
node --check src/app.js
node --check src/api.js
.venv/bin/python -m compileall -q backend scripts
```

Video imports require `ffprobe` on `PATH`. The repository vendors the supplied
WorldMM-a pipeline and its fixed YOLO/Pose weights under `tools/video_keyframe/`;
runtime tuning uses `SENTRIX_VIDEO_WIDTH`, `SENTRIX_VIDEO_SAMPLE_FPS`,
`SENTRIX_VIDEO_ANALYSIS_FPS`, `SENTRIX_VIDEO_DEVICE`, and
`SENTRIX_VIDEO_MAX_KEYFRAMES`. WorldMM's complete `memory_keyframes` remain
available in the derived output; Sentrix uses the package's recommended
`research/summary_keyframes.json` and imports at most the configured number of
representative frames (160 by default).

The maintenance rebuild is intentionally explicit because it replaces derived
memory data:

```bash
.venv/bin/python scripts/maintenance/rebuild_memory.py --root . --source /path/to/source-album
```
