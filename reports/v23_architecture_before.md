# Sentrix Home v2.3 MTSW — architecture before

审计时间：2026-08-20

## 线上调用链

```text
POST /api/import/server-directory
  -> backend.app.import_assets
  -> background process_ingest_batch
  -> backend.video.processor.VideoMemoryAdapter.process
  -> backend.video.worldmm_adapter.WorldMMAdapter.run
  -> tools/video_keyframe/katna/run_yolo_prefilter_event_webp.py
  -> YOLO 10 FPS batch prefilter
  -> target-window Katna quality selection
  -> FFmpeg CUDA/NVDEC target decode
  -> WebP + semantic.json/frame_map.json/stats.json
  -> 180 baseline video_scene rows + derived WebP assets
  -> optional v2.1 EventAgg DINO cache/index
  -> Timeline and Performance APIs
```

## Existing modules and observed boundaries

| Area | Current implementation | v2.3 impact |
|---|---|---|
| Video decode | `WorldMMAdapter._run_hybrid`; helper `_decode_one_target` uses FFmpeg CUDA/NVDEC | retain for dense transition windows; add independent MTSW runner |
| YOLO | `run_yolo_prefilter_event_webp.py::_predict_yolo_batch`; 10 FPS batch | v2.3 low-rate state scan at 2 FPS, cached state vectors |
| Pose | existing `yolo11n-pose.pt` is wired into the vendored semantic analyzer/package path | optional transition-local pose state; no VLM |
| Katna | targeted unstable-segment quality gate | v2.3 does not rerun Katna over the whole video; dense windows are selected by state transitions |
| DINOv2 | `backend.video.event_aggregator.DINOEmbedder`; hash/model/version-bound cache | reuse cache for fixed-frame experiment, candidate-only embeddings for MTSW |
| Current keyframes | `HybridResult.from_output` groups package frames; current baseline has 180 WebP | v2.3 writes separate method/version and never edits v2.0/v2.1 |
| EventAgg | `event_aggregate_events` / `event_aggregate_frames` and `method_runs` | v2.3 uses the same additive index tables with method version `keyframe-hybrid-v2.3-mtsw` |
| Timeline | `/api/events?method=baseline/eventagg`; `src/app.js::timelineView` | add `mtsw` as a third switchable method |
| Performance | `/api/performance`; video metadata and method runs | add v2.3 stage/wall metrics and comparison rows |
| Database | SQLite `MemoryStore`; baseline `events` and derived assets | MTSW rows are additive; evidence assets are never deleted |

## Baseline facts preserved

`loan-shoe-v2` / `asset_c9cca662209f`: 1521.72 s, 180 WebP, 180 baseline scenes.
The v2.1 EventAgg run remains in `method_runs` and `event_aggregate_*` with 131 events.

## Current bottleneck

The frozen hybrid extraction is the dominant wall-clock stage (`77.827 s` keyframe
extraction in the v2.1 full run, `293.26 s` total processing including evidence
ingestion). v2.1 EventAgg itself is only `3.9315 s`; it can compress the event
index but cannot recover state transitions that never became keyframes. v2.3
therefore adds a separate low-rate state scan and local dense decode scheduler.

## v2.3 invariants

- VLM/LLM calls are disabled in the MTSW runner (`0` calls).
- No baseline rows, WebP files, or v2.1 runs are overwritten.
- Model results and state vectors are cached by source hash/config version.
- Stage timings and total wall-clock are recorded separately.
