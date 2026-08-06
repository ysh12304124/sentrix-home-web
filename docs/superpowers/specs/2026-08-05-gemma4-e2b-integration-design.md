# Gemma-4-E2B-it + LoRA V2 集成设计

- Status: Approved for implementation
- Author: Codex (Kiro session)
- Date: 2026-08-05
- Target branch: `psh` on 153 (`/home/asus/Github/Sentrix-Home-Web`)
- Related delivery: `model_merge_gemma4e2b_cross_family_v2_lora_delivery_20260805`
- Related MCP project: `sentrix-home-web`

## 1. 背景与动机

Sentrix Home 目前通过 Ollama HTTP API(`http://127.0.0.1:11435`)调用 `gemma4:12b` 完成全部多模态大模型任务:图像描述(`analyze_image`)、维度聚焦(`analyze_image_focus`)、人物外观(`analyze_person_appearance`)、文本分析(`analyze_text`)、事件摘要(`summarize_event`)、对话(`chat`)、Agent 回答(`answer`)、文本嵌入(`embed_text`)。所有调用集中在 `backend/model_clients.py::GammaClient`,通过 5 处实例化点(`app.py:28`、`agent.py:57`、`pipeline.py:34`、`scripts/maintenance/rebuild_memory.py`、`scripts/maintenance/backfill_scene_types.py`)注入到业务层。

现在需要把训练过的 `Gemma-4-E2B-it + LoRA V2 (student_step47)` 作为**可切换的备选**接入进来,与 12B 并存,由用户在设置页显式选择。E2B + LoRA 是跨家族 Trust-OPD 蒸馏产物,基于 Qwen3-VL-8B-Instruct 教师训练,只在 language decoder 上挂 410 个 LoRA 张量,视觉编码器/projector/merger 未训练。

**核心矛盾**:交付包只提供 LoRA adapter(92 MB),必须挂载在 Gemma-4-E2B-it 基模上并通过 PEFT 加载。Ollama 只支持 GGUF,把 LoRA 合并回 safetensors 再转 GGUF 会破坏 adapter 灵活性,且训练脚本没有对应的 GGUF 兼容路径。因此 E2B 必须走 HuggingFace `transformers + peft` 直接推理,不能复用 Ollama 服务。这带来了双运行时(Ollama on :11435 + 新 HF 服务 on :8100)。

## 2. 目标

1. E2B + LoRA 作为一等公民接入,7 个生成类方法(chat/analyze_image/analyze_image_focus/analyze_person_appearance/analyze_text/summarize_event/answer)由前端设置页全局切换。
2. `embed_text` 硬钉 Ollama 12B,保护 SQLite `memory_vectors` 表(714 条 3840 维向量)不因模型切换失效。
3. 现有 5 处 `GammaClient()` 实例化点和 20+ 处调用点**零改动**,通过 facade 内部路由实现切换。
4. GPU 单卡驻留策略:只驻留当前选中模型;切换时旧模型释放显存。
5. 切换状态持久化到 SQLite,API 重启不丢。
6. 切换失败(目标 backend 不可达)显式报错,不做静默 fallback。
7. 前端在 settings 页 AI MODEL ROUTER 卡片内嵌切换器。

## 3. Non-goals

1. **不做**全量 `memory_vectors` 重嵌入迁移。E2B 与 12B 向量空间不兼容,`embed_text` 仍走 12B。未来若需要独立 embedder(BGE-M3/jina),另起工程。
2. **不做** GGUF 转换或把 LoRA 合并进 Ollama。保留 adapter 形态以便回滚。
3. **不做**批量并发优化的 E2B 推理。单卡单请求,通过 asyncio Lock 串行化;批量场景推荐用 12B。
4. **不改** FMA :5173(外部服务边界)、AdaFace、buffalo_l、FunASR、CLIP 的调用链。
5. **不做**多用户/多会话的 per-request 模型选择;全局单选。

## 4. 总体架构

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Browser (src/app.js)                        │
│  settingsView(): AI MODEL ROUTER 卡片增加 <select> 切换器        │
│  GET  /api/vlm-backend  → 显示当前选中 + 可选列表                │
│  POST /api/vlm-backend  → { backend: "ollama_12b" | "e2b_lora"} │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         Node Proxy :4174 → FastAPI backend :8091 (unchanged)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  backend/app.py (FastAPI :8091)                 │
│                                                                 │
│  gamma = GammaClient()  ← 全局单例(facade)                     │
│  gamma.bind_store(store)                                        │
│                                                                 │
│  新增: /api/vlm-backend GET/POST                                │
│  修改: /api/health.models.gamma4_12B → .models.vlm              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
┌──────────────────────────┐   ┌──────────────────────────────────┐
│  GammaClient (facade)    │   │  runtime_settings 表             │
│  __init__ 时读一次        │   │  key='vlm_backend'               │
│  active_backend 缓存 5s   │◄──│  value='ollama_12b'|'e2b_lora'   │
│                          │   │  updated_at                      │
│  chat/analyze_image/...  │   └──────────────────────────────────┘
│  → router.dispatch(fn)   │
│                          │
│  embed_text              │
│  → 硬编码走 OllamaBackend │
└──────────────────────────┘
     │              │
     ▼              ▼
┌──────────────┐  ┌────────────────────────────────────────┐
│OllamaBackend │  │  E2BBackend                            │
│HTTP :11435   │  │  HTTP :8100/api/chat (mimic Ollama)    │
│gemma4:12b    │  │                                        │
└──────────────┘  └────────────────────────────────────────┘
                                        │
                                        ▼
                        ┌────────────────────────────────┐
                        │ services/e2b_server/           │
                        │ FastAPI 独立进程 :8100         │
                        │ 加载 base=Gemma-4-E2B-it       │
                        │  + LoRA(V2_student_step47)    │
                        │ 提供 /api/chat /api/generate   │
                        │      /api/health /admin/*      │
                        │ 独立 conda env: sentrix-e2b   │
                        │ transformers 5.13.1           │
                        │ peft 0.19.1                   │
                        │ torch 2.8.0+cu126             │
                        └────────────────────────────────┘
```

**驻留切换**:切换到 E2B 时,`POST /api/vlm-backend` 处理器异步:(1) 对 Ollama 发 `keep_alive=0` 空请求触发卸载;(2) 对 E2B 发 `POST /admin/load` 预热。切回 12B 反之。切换本身立即返回(SQLite 已更新),驻留 transition 是 fire-and-forget。

## 5. 组件设计

### 5.1 E2B 独立服务 (`services/e2b_server/`)

**目录结构**:
```
services/e2b_server/
  app.py              # FastAPI 主入口
  model.py            # 加载器: base + LoRA, generate 封装
  ollama_shape.py     # 响应格式转换(mimic Ollama)
  __init__.py
  README.md
  tests/
    test_ollama_shape.py
    test_model_smoke.py
```

**加载流程**(严格按交付脚本 `deploy_gemma4e2b_lora.py`):
1. `AutoProcessor.from_pretrained(BASE)` → 得到 processor(含 chat_template / image tokenizer)
2. `AutoModelForMultimodalLM.from_pretrained(BASE, dtype=bf16, device_map="cuda:0", low_cpu_mem_usage=True)`,不可用则 fallback 到 `AutoModelForImageTextToText`
3. `PeftModel.from_pretrained(base_model, ADAPTER)` 挂 LoRA
4. `model.eval()`
5. 保留 processor 的 `image_position_ids`/`mm_token_type_ids`,`images=[[pil_image]]` 嵌套结构
6. 记录 `loaded=True` 状态供 `/api/health` 上报

**HTTP 接口**:

| 路径 | 方法 | 说明 |
|---|---|---|
| `/api/health` | GET | `{"status":"ok","model":"gemma-4-e2b-it+lora-v2","adapter":"V2_student_step47","dtype":"bf16","loaded":bool,"error":null}` |
| `/api/chat` | POST | 请求体 `{model, messages, images?, format?, stream:false, options?}`;响应 `{"message":{"role":"assistant","content":"..."},"done":true,"model":"..."}` |
| `/api/generate` | POST | 简单 prompt 场景;响应 `{"response":"...","done":true,"model":"..."}` |
| `/api/embeddings` | POST | **不实现**,返回 501。Embed 强制走 Ollama 12B。 |
| `/admin/load` | POST | 幂等,触发懒加载 |
| `/admin/unload` | POST | 释放显存,`del model; torch.cuda.empty_cache()` |

**Request 兼容层**:
- 图片: `messages[i].content` 支持 `[{"type":"image","image":"base64:..."}, {"type":"text","text":"..."}]`;或复用 Ollama 顶层 `"images":["base64..."]` 字段(GammaClient 现有做法)。两种都支持,减少 GammaClient 分支。
- `format: "json"` → 不做强约束(HF PEFT 无 grammar 支持),仅提示 prompt 追加 "仅输出 JSON,不要包裹在代码块中。";返回原文由 GammaClient 端 `parse_json_response()` 兜底。
- `options.temperature` / `options.num_predict` → 映射到 `generation_config`(temperature, max_new_tokens)
- `keep_alive` → 忽略。E2B 常驻,通过 `/admin/unload` 手动释放。

**并发**:`--np 1` 单请求。`asyncio.Lock` 序列化。批量场景推荐 12B。

**启动脚本** `scripts/runtime/start_sentrix_e2b.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /home/asus/miniconda3/etc/profile.d/conda.sh
conda activate sentrix-e2b
export E2B_BASE_MODEL="${E2B_BASE_MODEL:-/home/asus/models/gemma-4-E2B-it}"
export E2B_ADAPTER="${E2B_ADAPTER:-/home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47}"
export E2B_HOST="${E2B_HOST:-127.0.0.1}"
export E2B_PORT="${E2B_PORT:-8100}"
exec uvicorn services.e2b_server.app:app --host "$E2B_HOST" --port "$E2B_PORT" --workers 1
```

`start_sentrix_api.sh` 增加一行 env:
```bash
export E2B_BASE_URL="${E2B_BASE_URL:-http://127.0.0.1:8100}"
```

### 5.2 GammaClient facade (`backend/model_clients.py`)

**目标结构**:

```python
class VLMBackend(Protocol):
    name: str
    endpoint: str
    def chat(prompt, images=None, vision_options=None, json_mode=True) -> str: ...
    def analyze_image(path, metadata=None) -> dict: ...
    def analyze_image_focus(path, dimension, metadata=None) -> dict: ...
    def analyze_person_appearance(path, metadata=None) -> dict: ...
    def analyze_text(text, source_type="text") -> dict: ...
    def summarize_event(event, observations) -> dict: ...
    def answer(query, context) -> dict: ...
    def embed_text(text) -> list[float]: ...
    def health() -> dict: ...

class OllamaBackend:
    """现有 GammaClient 主体代码搬进来,行为不变。"""
    name = "ollama_12b"
    # ... 与当前 GammaClient 同

class E2BBackend:
    """新增,HTTP 打到 :8100,mimic Ollama 响应格式。"""
    name = "e2b_lora"
    # ...

class GammaClient:  # facade,类名和构造签名保持不变
    def __init__(self, base_url=None, model=None, timeout=None, keep_alive=None):
        self._ollama = OllamaBackend(base_url, model, timeout, keep_alive)
        self._e2b = E2BBackend(os.getenv("E2B_BASE_URL", "http://127.0.0.1:8100"), timeout)
        self._store = None
        self._active_cache = None
        self._cache_ttl = 5.0
        self._cache_ts = 0.0

    def bind_store(self, store):
        self._store = store

    def invalidate_backend_cache(self):
        self._active_cache = None
        self._cache_ts = 0.0

    def _active(self) -> "VLMBackend":
        now = time.monotonic()
        if self._active_cache and (now - self._cache_ts) < self._cache_ttl:
            return self._active_cache
        name = self._read_active_name()  # SQLite runtime_settings
        backend = self._e2b if name == "e2b_lora" else self._ollama
        self._active_cache = backend
        self._cache_ts = now
        return backend

    def _read_active_name(self) -> str:
        if self._store is None:
            return os.getenv("SENTRIX_VLM_BACKEND", "ollama_12b")
        return self._store.get_setting("vlm_backend", "ollama_12b")

    # facade 方法 —— 全部委托到 _active(),除 embed_text
    def chat(self, *a, **kw):                       return self._active().chat(*a, **kw)
    def analyze_image(self, *a, **kw):              return self._active().analyze_image(*a, **kw)
    def analyze_image_focus(self, *a, **kw):        return self._active().analyze_image_focus(*a, **kw)
    def analyze_person_appearance(self, *a, **kw):  return self._active().analyze_person_appearance(*a, **kw)
    def analyze_text(self, *a, **kw):               return self._active().analyze_text(*a, **kw)
    def summarize_event(self, *a, **kw):            return self._active().summarize_event(*a, **kw)
    def answer(self, *a, **kw):                     return self._active().answer(*a, **kw)

    def embed_text(self, text):
        # 保护向量库:memory_vectors 累积用的是 gemma4:12b (3840 维),不能跟随切换。
        # 未来若接入独立 embedder,在此改为 self._embedder.embed_text(text)。
        return self._ollama.embed_text(text)

    @property
    def model(self):        return self._active().model_name
    @property
    def base_url(self):     return self._active().endpoint
    @property
    def active_name(self):  return self._active().name
```

**要点**:

1. **构造签名不变** → 现有 5 处 `GammaClient()` 实例化点零改动。
2. **`model`/`base_url` 属性保留** → 现有 `/api/health` payload 结构不破;仅 key 从 `gamma4_12B` 换成 `vlm`(见 5.4)。
3. **`_read_active_name()` 缓存 5 秒** → 避免热路径每次查 SQLite;切换后 `invalidate_backend_cache()` 立即失效。
4. **不做 warmup** → 首次调用触发 backend 加载;显式 warmup 由 `/api/vlm-backend` POST 处理器发起。
5. **`bind_store(store)` 显式注入** → 避免模块级全局耦合;`app.py` 里 `gamma = GammaClient(); gamma.bind_store(store)` 两行。

### 5.3 数据模型 (`backend/db.py`)

**新表**:
```sql
CREATE TABLE IF NOT EXISTS runtime_settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO runtime_settings (key, value) VALUES
    ('vlm_backend', 'ollama_12b');
```

DDL 加进 `MemoryStore._ensure_schema()` 的幂等序列。新增方法:
```python
def get_setting(self, key: str, default: str | None = None) -> str | None
def set_setting(self, key: str, value: str) -> None
def list_settings(self) -> dict[str, str]
```

**不用现有表**:项目现有 domain 表(assets/observations/events/entities/...)承载业务实体;运行时开关是横切配置,单独一张 KV 表避免污染 domain schema,便于未来加"当前 embedder"、"批量事件摘要开关"等。

### 5.4 后端 API 端点 (`backend/app.py`)

**新增**(靠近 `/api/health`,即 line 69 附近):

```python
VLM_BACKENDS = ("ollama_12b", "e2b_lora")

@app.get("/api/vlm-backend")
def get_vlm_backend():
    active = store.get_setting("vlm_backend", "ollama_12b")
    return {
        "active": active,
        "available": [
            {
                "id": "ollama_12b",
                "label": "Gemma-4 12B (Ollama)",
                "endpoint": os.getenv("OLLAMA_BASE_URL"),
                "model": os.getenv("OLLAMA_MODEL", "gemma4:12b"),
                "capabilities": ["completion", "vision", "tools", "thinking"],
                "healthy": _check_ollama_health(),
            },
            {
                "id": "e2b_lora",
                "label": "Gemma-4 E2B-it + LoRA V2",
                "endpoint": os.getenv("E2B_BASE_URL", "http://127.0.0.1:8100"),
                "adapter": "V2_student_step47",
                "capabilities": ["completion", "vision"],
                "healthy": _check_e2b_health(),
            },
        ],
    }

class SetVLMBackend(BaseModel):
    backend: Literal["ollama_12b", "e2b_lora"]

@app.post("/api/vlm-backend")
def set_vlm_backend(payload: SetVLMBackend):
    if payload.backend not in VLM_BACKENDS:
        raise HTTPException(400, "invalid backend")
    if payload.backend == "e2b_lora" and not _check_e2b_health(hard=True):
        raise HTTPException(503, "e2b service :8100 unreachable")
    if payload.backend == "ollama_12b" and not _check_ollama_health(hard=True):
        raise HTTPException(503, "ollama service :11435 unreachable")

    old = store.get_setting("vlm_backend", "ollama_12b")
    store.set_setting("vlm_backend", payload.backend)
    gamma.invalidate_backend_cache()
    _schedule_backend_transition(old, payload.backend)  # async fire-and-forget
    return {"active": payload.backend, "previous": old}
```

**修改 `/api/health`**:
```python
"models": {
    "vlm": {
        "active": gamma.active_name,   # "ollama_12b" | "e2b_lora"
        "name": gamma.model,           # 当前激活的模型名
        "endpoint": gamma.base_url,    # 当前激活的端点
    },
    "asr": {...},
    ...
}
```

**探活**:
- `_check_ollama_health()`:GET `{OLLAMA_BASE_URL}/api/tags`,2 秒超时,任意 20x 视为健康。
- `_check_e2b_health()`:GET `{E2B_BASE_URL}/api/health`,2 秒超时,`loaded=true` 视为健康。
- `hard=True` 变体超时 5 秒,允许冷启动。

**切换 transition** `_schedule_backend_transition(old, new)`(用 `asyncio.create_task` 后台跑):
- `old == "ollama_12b"` → 对 `{OLLAMA_BASE_URL}/api/generate` POST `{"model":"gemma4:12b","prompt":"","keep_alive":0}`
- `old == "e2b_lora"` → 对 `{E2B_BASE_URL}/admin/unload` POST
- `new == "e2b_lora"` → 对 `{E2B_BASE_URL}/admin/load` POST 预热
- `new == "ollama_12b"` → 一个短 prompt 触发 Ollama 加载
- 任何失败只 log warning,不影响切换本身(已经生效于 SQLite)

### 5.5 前端 UI (`src/app.js` + `src/api.js`)

**位置**:`settingsView()` 里 AI MODEL ROUTER 卡片(line 341 附近)。

**卡片结构**:
```html
<article class="health-card">
  <div class="health-title">
    <span>AI MODEL ROUTER</span>
    <span class="ready-label ${vlm.healthy ? '' : 'warn'}">
      ${vlm.healthy ? 'READY' : 'OFFLINE'}
    </span>
  </div>
  <label class="model-switcher">
    <span>视觉推理</span>
    <select id="vlm-backend-select" data-action="switch-vlm">
      ${available.map(b => `
        <option value="${b.id}"
                ${b.id === vlm.active ? 'selected' : ''}
                ${!b.healthy ? 'disabled' : ''}>
          ${escapeHtml(b.label)}${b.healthy ? '' : ' · 离线'}
        </option>
      `).join('')}
    </select>
    <small>${escapeHtml(active.endpoint)}</small>
  </label>
  <div class="model-row"><span>语音转写</span> ...</div>
  <div class="model-row"><span>人物识别</span> ...</div>
</article>
```

**交互**:
```javascript
document.addEventListener('change', async (event) => {
  const select = event.target.closest('[data-action="switch-vlm"]');
  if (!select) return;
  const target = select.value;
  const previous = state.vlm?.active;
  select.disabled = true;
  toast(`切换到 ${select.selectedOptions[0].textContent}...`, 'info');
  try {
    const result = await api.setVlmBackend(target);
    state.vlm.active = result.active;
    toast(`已切换到 ${select.selectedOptions[0].textContent}`, 'success');
    await refreshHealth();
  } catch (err) {
    select.value = previous;
    toast(`切换失败: ${err.message}`, 'error');
  } finally {
    select.disabled = false;
  }
});
```

**API 封装** `src/api.js`:
```javascript
getVlmBackend: () => fetch('/api/vlm-backend').then(handle),
setVlmBackend: (backend) => fetch('/api/vlm-backend', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({backend}),
}).then(async r => {
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${r.status}`);
  }
  return r.json();
}),
```

**Overview 页面副作用**:`overview()` 里第 181 行 `state.health?.models?.gamma4_12B?.name` 改成 `state.health?.models?.vlm?.name`,一处一行改动。

**CSS**:复用现有 `.model-row` 与 `.space-switcher` 样式,不新增设计 token。

## 6. 错误处理与边界

### 6.1 E2B 服务未启动或崩溃

- **业务代码调用**:`E2BBackend` 抛 `ModelError("E2B backend unreachable: ...")`;业务层已有的 try/except 会降级(agent.py 返回 "证据不足",pipeline 记 failed_asset)。
- **前端 select 尝试切到 E2B**:`POST /api/vlm-backend` 前的探活拒绝(503),select 自动回退到 previous。
- **切换成功后 E2B 中途挂**:`/api/health.vlm.healthy=false`,前端 badge 从 READY 变 OFFLINE;调用继续走 E2B 直到用户手动切回。

### 6.2 base 模型或 adapter 目录丢失

- 启动脚本前置检查 `E2B_BASE_MODEL` 与 `E2B_ADAPTER` 目录存在,并断言关键文件:`config.json`、`processor_config.json`、`tokenizer.json`、`adapter_config.json`、`adapter_model.safetensors`。缺则打印明确错误,退出码 1,uvicorn 不起。
- SHA256 校验只在首次部署做一次(用交付包的 `PACKAGE_CONTENTS.clean.sha256`),运行时不做以免拖启动。

### 6.3 GPU OOM

- 切换 transition 先 unload old,等 200-500ms 让 CUDA 内存归还,再 load new。
- E2B 加载 OOM → 服务捕获异常返回 500,`/api/health.loaded=false, error="OOM"`,前端 badge OFFLINE。
- 首次切到 E2B 卡在加载(10-30 s):select 已 disabled + toast,`refreshHealth` 轮询 `loaded` 直到 ready 或 45 s 超时。超时提示 "加载超时,请检查服务日志"。

### 6.4 embed_text 与 12B 唤醒副作用

- `GammaClient.embed_text` 硬编码走 OllamaBackend。选中 E2B 期间发起 embed(如用户查询)会让 Ollama 短暂唤醒 12B,占一段显存。
- MVP 阶段接受此副作用。若实测 OOM,备选:
  - P1:Ollama 走 CPU (`OLLAMA_HOST_DEVICE=cpu`) 仅 embed,慢但不抢显存。
  - P2:引入独立 embedding 模型(BGE-M3 CPU),对应第 5.2 节里的 `_embedder` 抽象。

### 6.5 JSON 解析失败

- Ollama 有 `format:"json"` 强约束,E2B 没有。
- 现有 `parse_json_response()` 兜底:严格 parse 失败 → 提取 markdown 代码块 / 首个平衡 `{...}` → 再失败 → 返回 `{"raw": text}`。业务层已有 "insufficient_evidence" 分支。
- E2BBackend 在 prompt 末尾追加 "仅输出 JSON,不要包裹在代码块中。" 提升成功率。此追加只对 E2B。

### 6.6 并发请求

- Ollama 端:有内置队列,并发到 :11435 无问题(现状)。
- E2B 端:`asyncio.Lock` 串行化。批量导入图片时 E2B 慢(单请求),这是单卡单请求的物理限制,非 bug。
- 文档:runbook 里明确 "批量导入建议用 12B,单次问答或演示可用 E2B"。

### 6.7 现有 test 破坏面

- `test_model_clients.py` 现有 4 处直接 `GammaClient(base_url=..., model=...).chat("测试")`,mock `requests.post`。`GammaClient` facade 委托到 `OllamaBackend`,构造签名不变,`.chat("测试")` 默认走 Ollama(因为 `runtime_settings` 无值时默认 `ollama_12b`)。**现有测试不改**,新增测试见 §7.3。

## 7. 部署、测试、迁移

### 7.1 一次性部署

**Step 1 — 下载 E2B 基模到 153**:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  mkdir -p /home/asus/models/gemma-4-E2B-it &&
  python -c \"
from modelscope import snapshot_download
snapshot_download(
    'google/gemma-4-E2B-it',
    cache_dir='/home/asus/.cache/modelscope',
    local_dir='/home/asus/models/gemma-4-E2B-it'
)
\"
"
```
校验:10 个文件、总 9.57 GB、`model.safetensors` 约 9.7 GB。

**Step 2 — 传 LoRA adapter 到 153**:

使用 rsync 把 delivery 目录的**内容**(注意源路径末尾的斜杠)复制到 153 上的目标目录,避免 scp -r 语义歧义:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "mkdir -p /home/asus/models/gemma-4-e2b-lora-v2"
sshpass -p 'Abc123' rsync -av --info=progress2 \
  /Users/rm001/Downloads/model_merge_gemma4e2b_cross_family_v2_lora_delivery_20260805/ \
  asus@192.168.0.153:/home/asus/models/gemma-4-e2b-lora-v2/
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47 &&
  sha256sum adapter_model.safetensors adapter_config.json
"
```
期望:
- `225e4f70ed3cc1098e43bcc00ea571649e14994c33353ba6745ee386fdd31ce5  adapter_model.safetensors`
- `7df6e3514dd04df283ed9b00291bf5e2a0c8ac01fe49a4833730694c2dc94eca  adapter_config.json`

复制后目录结构:
```
/home/asus/models/gemma-4-e2b-lora-v2/
├── artifacts/lora/V2_student_step47/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── deploy_gemma4e2b_lora.py
├── requirements-tested.txt
├── README_DEPLOY.md
├── DEPLOYMENT_MANIFEST.json
└── PACKAGE_CONTENTS.clean.sha256
```

因此 `E2B_ADAPTER` 指向 `/home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47`。

**Step 3 — 建 conda env**:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda create -n sentrix-e2b python=3.11 -y &&
  conda activate sentrix-e2b &&
  pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126 &&
  pip install transformers==5.13.1 peft==0.19.1 safetensors Pillow accelerate fastapi 'uvicorn[standard]' requests
"
```

**Step 3.5 — 冒烟(硬门禁)**:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate sentrix-e2b &&
  python /home/asus/models/gemma-4-e2b-lora-v2/deploy_gemma4e2b_lora.py \
    --base-model /home/asus/models/gemma-4-E2B-it \
    --adapter /home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47 \
    --image <album1 里任意一张> \
    --prompt '请描述图片中的主要内容。' \
    --max-new-tokens 128
"
```
必须打印一段中文描述。失败则回退查环境;`transformers==5.13.1` 若 pypi 上不存在,备选 `transformers>=5.11,<6`,同步测试。

### 7.2 代码提交顺序(11 步,每步可回滚)

| # | 提交 | 含义 | 验证 |
|---|---|---|---|
| 1 | `feat(db): add runtime_settings kv table` | 只加表 + get/set/list_setting | Python 测试通过 |
| 2 | `refactor(model_clients): extract OllamaBackend from GammaClient` | 行为不变,搬代码 | 现有 test_model_clients.py 通过 |
| 3 | `feat(model_clients): add VLMBackend protocol and E2BBackend stub` | 加接口 + E2B 空实现(返回 501) | 新 test_e2b_backend 桩 |
| 4 | `feat(services): add e2b_server FastAPI mimicking ollama shape` | 独立服务 :8100 完整实现 | curl 冒烟 + 单测 |
| 5 | `feat(model_clients): wire E2BBackend to :8100 and GammaClient facade routing` | 打通真调用 + facade 分发 | 集成测试 |
| 6 | `feat(api): add /api/vlm-backend GET/POST endpoints` | 后端切换路由 + 探活拒切换 | test_vlm_backend_api |
| 7 | `refactor(api): migrate health.models.gamma4_12B to health.models.vlm` | 后端 key 换名 | 现有测试改 fixture |
| 8 | `feat(web): add vlm backend switcher in settings AI MODEL ROUTER card` | 前端 select + 交互 + api.js | Node test + 手动 |
| 9 | `refactor(web): use health.models.vlm in overview and settings views` | 前端 key 换名(2 处) | Node test |
| 10 | `feat(runtime): add start_sentrix_e2b.sh + start_sentrix_api.sh env` | 启动脚本 + env 声明 | project-structure.test.js 增补 |
| 11 | `docs: add e2b integration operator runbook` | 部署/切换/回滚步骤文档 | — |

第 7、9 步是耦合的 breaking(health JSON schema),必须一起部署;其他前向兼容。

### 7.3 测试策略

**Python `unittest`**:
- `test_model_clients.py` → 重命名 `test_ollama_backend.py`(或保留原文件加 shim)。
- `test_e2b_backend.py`(新增):mock HTTP :8100,覆盖 chat/analyze_image/health/error 分支。
- `test_vlm_router.py`(新增):facade 分发、`embed_text` 硬编码不受切换影响、TTL 缓存、`invalidate_backend_cache`。
- `test_vlm_backend_api.py`(新增):GET 返回 available 列表、POST 探活失败 503、POST 成功更新 SQLite + invalidate cache。
- `test_memory_store.py` 补 `runtime_settings` DDL 幂等 + set/get 一致性。

**Node test**:
- `test/project-structure.test.js` 增补 `start_sentrix_e2b.sh` 存在断言、API 启动脚本必须 export `E2B_BASE_URL`。
- `test/api-contract.test.js`:`/api/vlm-backend` 契约测试。

**E2B 独立服务测试**:
- `services/e2b_server/tests/test_ollama_shape.py`:响应格式匹配 Ollama 关键字段。
- `services/e2b_server/tests/test_model_smoke.py`:真跑一张小图(需 GPU,markers 跳过 CI)。

**手动验证清单**(部署完成后逐项跑):
1. 后端启动脚本正常拉起 :8091 + :8100,健康端点返回 200。
2. `GET /api/vlm-backend` 返回 active + available[2]。
3. 前端 settings 页看到切换器,当前选中 "Gemma-4 12B (Ollama)"。
4. 切到 E2B → toast 提示 → 30 s 内 badge 变 READY。
5. 导一张图 → observations 表新增记录 → `raw.gamma.models.vision == "gemma-4-e2b-it+lora-v2"`。
6. Agent 问答一次 → answer 返回、trace 里 model 字段是 E2B。
7. 切回 12B → GPU 释放 → 导图 raw.gamma 是 `gemma4:12b`。
8. 关掉 :8100 → `/api/health.vlm.healthy=false` → 切到 E2B 被拒 503。

### 7.4 迁移与回滚

**首次部署顺序**:
1. Step 1-3(数据 + 环境)。
2. Code 1-11 合并到 psh 分支。
3. 启动 e2b 服务 → 启动 API 服务(顺序无所谓)。
4. 前端加载,过 §7.3 手动清单。

**回滚**:
- 前端问题 → 单独回滚 Code 8-9,后端保留;老前端 `models.gamma4_12B` 找不到 key 时读到 undefined,退化为 "未知"(不 crash)。
- 后端 API 问题 → Code 7 + 9 一起回滚。
- E2B 服务问题 → 停 :8100,SQLite `UPDATE runtime_settings SET value='ollama_12b' WHERE key='vlm_backend'`,GammaClient 5 秒内切回。

**数据兼容**:`runtime_settings` 新表不动老表;`observations.raw.gamma.models.vision` 新出现 `gemma-4-e2b-it+lora-v2` 值(现只 `gemma4:12b`),前端只展示字符串,无影响。

## 8. 未定项 / 未来工作

1. **独立 embedder**:当有需求换掉 12B 做 embed 时,把 `GammaClient.embed_text` 委托到 `self._embedder`;新增 `BGEBackend` 或 `JinaBackend`;写一次性迁移脚本重嵌入 `memory_vectors`。
2. **E2B 的 JSON 稳定性**:上线后跟踪 `parse_json_response()` 兜底触发率,>10% 时考虑接入 `outlines` 或 `lm-format-enforcer`。
3. **多 backend 扩展**:如果未来接入 Qwen3-VL 或其它模型,现在的 `VLMBackend` protocol 已经预留;`VLM_BACKENDS` 常量和前端 `available` 列表都是可扩展的。
4. **切换后异常观测**:接 `/api/metrics`(现有 P0-P9 落地记录里的可观测性)暴露 backend 分布、失败率、切换事件。

## 9. 验收标准

- [ ] 部署冒烟脚本一次成功(§7.1 Step 3.5)。
- [ ] 手动验证清单 8 项全过(§7.3)。
- [ ] Python 测试全绿:现有 236/236 + 新增 4 个测试文件全绿。
- [ ] Node 测试全绿:现有 27/27 + 新增契约测试。
- [ ] `/api/health` 与 `/api/vlm-backend` 契约文档更新到 README 或 API 文档。
- [ ] 部署 runbook 说明:环境变量、启动顺序、切换和回滚步骤(Code 11)。
- [ ] 至少完成一次真数据端到端:导入一张 album1 图,`observations.raw.gamma.models.vision` 记录 E2B 名字,SQLite 里可查。

