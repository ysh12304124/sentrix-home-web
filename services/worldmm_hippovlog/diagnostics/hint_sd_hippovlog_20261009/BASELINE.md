# HINT-SD-style HippoVlog diagnostic baseline

Experiment directory on 200: `/home/sscy/evaluations/hint-sd-hippovlog-20261009`.
Source project: `/home/sscy/evaluations/sentrix-worldmm-hippovlog-20260928`, Git `e9c923be73cfd1bb5d36f09c103ac14ffece9810`.
Existing source tree `git diff --stat` was empty before this experiment. The WorldMM service directory is untracked in that worktree, so file hashes are recorded below. No source, online service, model, or dataset file is edited by this experiment.

Pre-experiment SHA-256:

| Source file | SHA-256 |
| --- | --- |
| `services/worldmm_hippovlog/qa.py` | `171717eeb1e7db71cb3009883025f2c4335acbf9fceb048da3c2779ea2591d72` |
| `services/worldmm_hippovlog/runtime.py` | `fcd202fa2e48a10bf92f191015cf5756fec92a816285703ae75110c70f40e664` |
| `backend/video/svd_keyframe.py` | `8e5216339da98c18ef390a426e6b5e69f101dffe5a678a1fb0656a4c7dd63166` |
| `backend/video/yolo_event_worker.py` | `1a62286dfec6b72a538758f30563bb2bd23599561a20501336af8a4364cd86d6` |

The diagnostic sidecar reuses completed, immutable upstream outputs from run `20260928-163006` and reruns WorldMM retrieval/QA for exactly two videos with isolated HippoRAG caches. Thus historical upstream timing is measured by original manifests and logs, while new retrieval/QA resource samples come from this diagnostic run. It is not a new extraction or memory build. Missing historical resource values must be reported as unavailable, never estimated as measured data.

The `runtime.py` checksum above was corrected after the final audit: the first version of this note contained a transcription error (61 hex characters). The file mtime remained 2026-09-30 and the final 64-character SHA-256 is shown above; the source file was not edited by this experiment.
