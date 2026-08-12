# Sentrix vLLM Manager Service

单实例 vLLM 模型管理服务（原 8500 端口服务，已纳入仓库）。

## 职责

- 常驻管理端口，不参与推理：负责按 registry profile 拉起/停止/切换 vLLM 子进程，并写入 `state/current.json`。
- 暴露 REST 接口供后端与远程主机管理模型，无需 SSH 拼接命令。

## 启动

```bash
python3 services/vllm_manager/app.py --host 0.0.0.0 --port 8500
```

监听端口与模型均来自 `configs/sentrix_vllm_registry_192_168_0_153.json`（`default_port`，当前 `8100`）。
可用环境变量覆盖：`SENTRIX_VLLM_MANAGER`、`SENTRIX_VLLM_REGISTRY`。

## API

| 路径 | 说明 |
|---|---|
| `GET /state` | 当前运行 profile / pid / port / base_url |
| `GET /registry` | registry 原文 |
| `GET /profiles` | profile 列表 + 本地路径可用性 |
| `POST /switch` | 切换模型（杀旧进程 → 起新 profile → wait-ready） |
| `POST /start` | 启动指定 profile |
| `POST /stop` | 停止当前实例 |
| `GET /gpu-stats` | nvidia-smi GPU 统计 |

`/switch`、`/start` 支持 `wait_ready`、`ready_timeout`、`dry_run` 以及
`max_model_len/max_num_seqs/gpu_memory_utilization/quantization/load_format/dtype/cuda_visible_devices` 覆盖。

## 与后端的关系

- 后端 `POST /api/model-profiles/switch` 直接调用 `services/vllm_manager/manager.py`（CLI），
  与 8500 的 `POST /switch` 走同一套 manager 逻辑、同一份 registry 与 state 文件。
- 切换成功后，后端 `_apply_vllm_profile_to_runtime` 重建全局 `gamma`/`pipeline`，
  使后续 agent turn 指向新模型的推理端口。
