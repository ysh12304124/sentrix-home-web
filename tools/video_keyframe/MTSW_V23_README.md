# keyframe-hybrid-v2.3-mtsw

This is an additive, VLM-free keyframe/event method. The frozen v2.0
evidence package and v2.1 EventAgg index are not rewritten.

## Runtime chain

1. Sequential low-rate scan at 2 FPS (`LowRateScanner`). YOLO inference is
   batched on the 3090; Pose inference is sampled every fourth scan item.
2. Each scan item becomes a `FrameState`: objects/bounding boxes, person count,
   pose state, interaction relations, appearance histogram, pHash and the
   cached visual embedding.
3. The MTSW engine computes a weighted state-change score, adaptive Z-score
   boundary and persistence decision. Strong boundaries retain pre/transition/
   post state evidence.
4. pHash/visual-similarity deduplication is state-aware, so frames with a new
   interaction or pose are not removed just because the image is similar.
5. Short continuity grouping and bridge merge produce events. A local dense
   window around each boundary is decoded with FFmpeg CUDA/NVDEC at the source
   resolution and written as WebP.
6. Selected WebP assets and event evidence are written as an independent
   `method_runs`/`event_aggregate_events` record with method version
   `keyframe-hybrid-v2.3-mtsw`. Timeline can switch among Baseline, EventAgg
   v2.1 and MTSW v2.3.

## Default parameters

`scan_fps=2`, `micro_window_sec=2`, event windows `3/10` seconds, adaptive
history `30`, `z_threshold=1.7`, strong threshold `3.0`, persistence `2`,
local dense window `±1.5` seconds, and weights interaction/person/object/pose/
scene/appearance = `0.25/0.20/0.15/0.15/0.15/0.10`.

## Commands

Full source-video run:

```bash
SENTRIX_VIDEO_DEVICE=0 python tools/video_keyframe/run_mtsw.py \
  --db /ssd/sscy/sentrix-home-v2-data/sentrix.db \
  --media-id <video-asset-id> --video <source.mp4> \
  --output /ssd/sscy/sentrix-home-v2-data/reports/mtsw/full
```

Fair event-only run over an existing 180-WebP package:

```bash
python tools/video_keyframe/run_mtsw.py --semantic <semantic.json> ...
```

Parameter and component sweep from a measured state cache:

```bash
python tools/video_keyframe/run_mtsw_sweep.py \
  --cache <full>/state_cache.json --output <sweep>
```

The sweep reuses detector output and performs no VLM/LLM call (`0/0`).
