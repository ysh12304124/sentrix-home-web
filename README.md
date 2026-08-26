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

## Run on 153

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

The WorldMM video-timeline deployment is `http://192.168.0.200:4174` with its
project-local API on `192.168.0.200:8091`. Existing services and shared model
processes on that host are not stopped or reconfigured by this deployment.

## Verify

```bash
.venv/bin/python -m unittest discover -s backend/tests -v
npm test
node --check src/app.js
node --check src/api.js
.venv/bin/python -m compileall -q backend scripts
```

Video imports require `ffprobe` and a CUDA-enabled `ffmpeg` on `PATH`. The
frozen keyframe method is `sentrix-keyframe-hybrid-v2.0.0`, recorded in
`tools/video_keyframe/METHOD_VERSION`. It runs 10 FPS batched YOLO/Pose
prefiltering, merges stable spans, sends only high-change windows through
targeted Katna, decodes final representatives with NVDEC, and writes one
full-resolution WebP per event. Stable samples are retained in `semantic.json`
as semantic substitutions; there is no 160-frame cap and no JPEG intermediate.

The backend uses this method by default through `SENTRIX_VIDEO_METHOD=hybrid_v2`.
Use `SENTRIX_VIDEO_METHOD=legacy` only for an explicit old WorldMM comparison.
Runtime tuning uses `SENTRIX_VIDEO_SCAN_FPS`, `SENTRIX_VIDEO_YOLO_BATCH_SIZE`,
`SENTRIX_VIDEO_KATNA_SCAN_FPS`, `SENTRIX_VIDEO_KATNA_UNSTABLE_PERCENTILE`,
`SENTRIX_VIDEO_MERGE_MAX_SEC`, `SENTRIX_VIDEO_TARGET_DECODE_WORKERS`,
`SENTRIX_VIDEO_WEBP_QUALITY`, and `SENTRIX_VIDEO_DEVICE`.

The maintenance rebuild is intentionally explicit because it replaces derived
memory data:

```bash
.venv/bin/python scripts/maintenance/rebuild_memory.py --root . --source /path/to/source-album
```
