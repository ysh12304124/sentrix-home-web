# Sentrix / WorldMM HippoVlog benchmark

This is an isolated experiment for the 200 server. The source is tracked on
`yxw2`; server data, models, caches, and results are not part of the repository.
Its source base is Sentrix `e9c923be73cfd1bb5d36f09c103ac14ffece9810`;
WorldMM is read from the HippoVlog dataset's pinned `source/WorldMM` checkout.

## Experimental variable

The two arms share the same 25 HippoVlog videos, 1,000 four-choice questions,
transcripts, Qwen3-VL-4B model, WorldMM caption prompt, 10/30/180/600-second
episodic hierarchy, OpenIE triples, semantic consolidation, VLM2Vec visual
memory, dynamic memory retrieval, and official answer scoring. The visual
memory is encoded once with WorldMM's 16-frame clip protocol and shared by
both arms because it is not the experimental variable.

- `worldmm_uniform`: WorldMM's original 1 fps sampling for each 10-second
  fine-caption window.
- `svd_esvd`: Sentrix's current two-pass SVD/eSVD algorithm with required CUDA
  YOLO semantic events; its selected frame(s) are placed into the same
  WorldMM fine-caption prompt. A window with no selected frame uses one frame
  from that window's midpoint, and these fallback windows are counted
  separately. This preserves temporal correctness in WorldMM's fixed windows.

The project uses Qwen3-VL-4B for local inference rather than the paper's
GPT-5-based setup. This is a *within-server A/B comparison*, not a claim to
reproduce the paper's published score.

## Paths on the 200 server

- Project: `/home/sscy/evaluations/sentrix-worldmm-hippovlog-20260928`
- Service: `services/worldmm_hippovlog`
- Dataset: `/ssd/sscy/datasets/HippoVlog-svd-worldmm-20260928`
- Model: `services/worldmm_hippovlog/models/Qwen3-VL-4B-Instruct`
- Results: `services/worldmm_hippovlog/results/<run-id>`
- UI: `http://192.168.0.200:8776/`

The dataset mirror is byte-identical to the supplied local folder. Its 25
MP4 files are hardlinked to the existing read-only source dataset on 200 to
avoid a second 17 GB copy. The experiment writes only under its own project.

## Runtime

The Python environment at `services/worldmm_hippovlog/.venv` uses Python 3.10
and reads the CUDA-capable Torch 2.6.0+cu124 installation from the existing
`d4rt` environment; additional packages are installed only in this project's
venv. `runtime.py` adapts WorldMM's Qwen3-VL model from its hardcoded `cuda:1`
and FlashAttention 2 to the server's single `cuda:0` GPU and PyTorch SDPA.
WorldMM's source files in the dataset remain unmodified.

Start the UI with `WORLDMM_BENCH_PYTHON=<venv>/bin/python python3
services/worldmm_hippovlog/app.py --host 0.0.0.0 --port 8776` from the project
root. Use the UI to start or resume a run. For a manual resume, invoke
`<venv>/bin/python services/worldmm_hippovlog/runner.py --run-dir <run-path>`.
Fine captions and QA are checkpointed per segment/question. A completed run
contains `summary.json`, `eval.json`, per-video keyframe manifests, caption
JSON files, WorldMM memory directories, and per-question responses.

## Validation rule

The report is valid only when both arms have 25 videos and 1,000 answered
questions, with exactly 250 in each HippoVlog category. Missing memory or QA
responses cause the run to fail instead of silently reducing the denominator.

The separate two-video fault-attribution diagnostic is documented under
`diagnostics/hint_sd_hippovlog_20261009/`. Its 80-question result must not be
reported as the full 1,000-question A/B score.
