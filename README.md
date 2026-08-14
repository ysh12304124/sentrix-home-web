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

Video imports require `ffprobe` on `PATH`. The repository vendors the supplied
WorldMM-a pipeline and its fixed YOLO/Pose weights under `tools/video_keyframe/`;
runtime tuning uses `SENTRIX_VIDEO_WIDTH`, `SENTRIX_VIDEO_SAMPLE_FPS`,
`SENTRIX_VIDEO_ANALYSIS_FPS`, and `SENTRIX_VIDEO_DEVICE`.

The maintenance rebuild is intentionally explicit because it replaces derived
memory data:

```bash
.venv/bin/python scripts/maintenance/rebuild_memory.py --root . --source /path/to/source-album
```
