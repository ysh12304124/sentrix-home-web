# Sentrix E2B Server

E2B LoRA 模型服务器，兼容 Ollama `/api/chat` 协议。

## 启动

```bash
scripts/runtime/start_sentrix_e2b.sh
```

默认监听 `127.0.0.1:8100`，可通过 `E2B_HOST` / `E2B_PORT` 环境变量覆盖。

## 环境变量

| 变量 | 默认值 |
|---|---|
| `E2B_BASE_MODEL` | `/home/asus/models/gemma-4-E2B-it` |
| `E2B_ADAPTER` | `/home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47` |
| `E2B_DTYPE` | `bf16` |
| `E2B_DEVICE_MAP` | `auto` |
| `E2B_HOST` | `127.0.0.1` |
| `E2B_PORT` | `8100` |

## API

### GET /api/health

```json
{"status":"ok","model":"gemma-4-E2B-it","adapter":"V2_student_step47","dtype":"bf16","loaded":true,"error":null}
```

### POST /api/chat

Ollama 兼容的 chat endpoint。详见 `ollama_shape.py`。

### POST /api/embeddings

永远返回 501。E2B 服务器不提供向量嵌入。

### POST /admin/load

加载模型到 GPU。

### POST /admin/unload

从 GPU 卸载模型。
