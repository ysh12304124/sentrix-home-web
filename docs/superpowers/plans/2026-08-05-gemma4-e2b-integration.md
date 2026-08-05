# Gemma-4-E2B-it + LoRA V2 集成实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把带训练头的 Gemma-4-E2B-it + LoRA V2 作为可切换的多模态 backend 接入 Sentrix Home,前端设置页可以在 Ollama 12B 和 E2B+LoRA 之间全局切换,`embed_text` 保持钉在 Ollama 12B。

**Architecture:** 保留 `GammaClient` 类作为 facade,内部按 SQLite `runtime_settings.vlm_backend` 分发到 `OllamaBackend`(现有 Ollama HTTP)或 `E2BBackend`(HTTP 调 :8100 的独立 FastAPI 服务)。E2B 服务用 `transformers 5.13.1 + peft 0.19.1 + torch 2.8.0+cu126` 加载基模 + LoRA,响应格式 mimic Ollama 减少适配层。

**Tech Stack:**
- Backend: Python 3.11 (`sentrix-e2b` conda env for E2B service), FastAPI, `httpx`(HTTP client, 项目现用), `unittest`+`unittest.mock`(测试), SQLite (`runtime_settings` KV 表)
- E2B runtime: `transformers==5.13.1`, `peft==0.19.1`, `torch==2.8.0+cu126`, `AutoModelForMultimodalLM` / `AutoModelForImageTextToText` fallback
- Web: 现有原生 JS SPA (`src/app.js` + `src/api.js`)
- 部署: 153, 分支 `psh`, 独立 conda env,新启动脚本 `scripts/runtime/start_sentrix_e2b.sh`

**Spec:** `docs/superpowers/specs/2026-08-05-gemma4-e2b-integration-design.md`(commit `8badd8c`)

---

## 前置约束(所有 Task 通用)

- **执行位置**: 所有代码修改必须落在 153 的 `/home/asus/Github/Sentrix-Home-Web` 上,分支 `psh`
- **执行 shell**: 在本地 macOS 上通过 `sshpass -p 'Abc123' ssh asus@192.168.0.153 "..."` 或 scp 传脚本
- **测试语言**: Python 用 `unittest`(现有风格,`class XxxTests(unittest.TestCase)`),HTTP mock 用 `@patch("backend.model_clients.httpx.post")`
- **Commit 消息**: 遵循现有 `type(scope): message` 格式,不加 emoji
- **不改**: `.venv`、`data/`、`logs/`、`node_modules/`、`.env`、model 权重、FMA :5173

---

## File Structure (最终形态)

### Create
- `backend/tests/test_e2b_backend.py` — E2BBackend HTTP 层单测
- `backend/tests/test_vlm_router.py` — GammaClient facade 分发单测
- `backend/tests/test_vlm_backend_api.py` — /api/vlm-backend endpoint 测试
- `services/e2b_server/__init__.py`
- `services/e2b_server/app.py` — FastAPI 主入口
- `services/e2b_server/model.py` — Base+LoRA loader,generate 封装
- `services/e2b_server/ollama_shape.py` — 请求/响应格式转换(纯函数,可测)
- `services/e2b_server/tests/__init__.py`
- `services/e2b_server/tests/test_ollama_shape.py` — 纯函数单测
- `services/e2b_server/tests/test_model_smoke.py` — GPU 冒烟(runtime skip)
- `services/e2b_server/README.md`
- `scripts/runtime/start_sentrix_e2b.sh`
- `docs/runbooks/vlm-backend-switch.md`

### Modify
- `backend/db.py` — 加 `runtime_settings` 表 DDL 到 `_create_schema()`;加 `get_setting/set_setting/list_settings` 方法
- `backend/model_clients.py` — 抽 `OllamaBackend`;加 `VLMBackend` protocol + `E2BBackend`;`GammaClient` 变 facade
- `backend/app.py` — 加 `/api/vlm-backend` GET/POST;改 `/api/health` 的 `models.gamma4_12B` → `models.vlm`;`gamma.bind_store(store)` 注入
- `backend/tests/test_memory_store.py` — 加 `runtime_settings` DDL 幂等 + set/get 测试
- `backend/tests/test_model_clients.py` — 保留(现有测试仍通过,因为 `GammaClient()` 默认走 Ollama)
- `src/app.js` — settings 页 AI MODEL ROUTER 卡片加 select 切换器;overview 里 `gamma4_12B.name` → `vlm.name`
- `src/api.js` — 加 `getVlmBackend` / `setVlmBackend`
- `scripts/runtime/start_sentrix_api.sh` — 加 `export E2B_BASE_URL`
- `test/project-structure.test.js` — 断言 `start_sentrix_e2b.sh` 存在 + `start_sentrix_api.sh` 含 `E2B_BASE_URL`

---

## Task 0: 环境和资产准备(硬门禁)

无代码改动,只准备 153 上的模型文件和 Python 环境。任何后续 Task 依赖此完成。

**Files:** 无(纯操作)

- [ ] **Step 0.1: 下载 Gemma-4-E2B-it 基模到 153**

在本地执行:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  mkdir -p /home/asus/models/gemma-4-E2B-it &&
  cd /home/asus/models &&
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
Expected: 打印 10 个文件下载进度;完成后 `du -sh /home/asus/models/gemma-4-E2B-it/` 约 9.6 GB。
校验:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  ls /home/asus/models/gemma-4-E2B-it/
"
```
Expected: 出现 `config.json`、`configuration.json`、`generation_config.json`、`processor_config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`model.safetensors`、`README.md`、`.gitattributes`。

- [ ] **Step 0.2: 传 LoRA delivery 到 153**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "mkdir -p /home/asus/models/gemma-4-e2b-lora-v2"
sshpass -p 'Abc123' rsync -av --info=progress2 \
  /Users/rm001/Downloads/model_merge_gemma4e2b_cross_family_v2_lora_delivery_20260805/ \
  asus@192.168.0.153:/home/asus/models/gemma-4-e2b-lora-v2/
```
Expected: rsync 输出显示 6 个文件全部复制,`adapter_model.safetensors` 约 92 MB。

校验 sha256:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47 &&
  sha256sum adapter_model.safetensors adapter_config.json
"
```
Expected:
```
225e4f70ed3cc1098e43bcc00ea571649e14994c33353ba6745ee386fdd31ce5  adapter_model.safetensors
7df6e3514dd04df283ed9b00291bf5e2a0c8ac01fe49a4833730694c2dc94eca  adapter_config.json
```

- [ ] **Step 0.3: 建 `sentrix-e2b` conda env**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda create -n sentrix-e2b python=3.11 -y &&
  conda activate sentrix-e2b &&
  pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126 &&
  pip install transformers==5.13.1 peft==0.19.1 safetensors Pillow accelerate fastapi 'uvicorn[standard]' httpx
"
```
Expected: 全部安装成功。若 `transformers==5.13.1` pypi 找不到(release 时间线问题),回退用 `pip install 'transformers>=5.12,<6'` 并把实际版本记入 Step 0.5 报告。

- [ ] **Step 0.4: 环境健康检查**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate sentrix-e2b &&
  python -c '
import torch, transformers, peft
print(\"torch:\", torch.__version__)
print(\"cuda:\", torch.cuda.is_available())
print(\"transformers:\", transformers.__version__)
print(\"peft:\", peft.__version__)
print(\"bf16:\", torch.cuda.is_bf16_supported())
'
"
```
Expected: `torch: 2.8.0+cu126`, `cuda: True`, `transformers: 5.13.x`, `peft: 0.19.1`, `bf16: True`。

- [ ] **Step 0.5: 用交付脚本冒烟一张图**

选一张 album1 里的图:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  ls /home/asus/Github/Sentrix-Home-Web/data/*/album1/*.jpg 2>/dev/null | head -1
"
```
用返回的图路径运行:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate sentrix-e2b &&
  python /home/asus/models/gemma-4-e2b-lora-v2/deploy_gemma4e2b_lora.py \
    --base-model /home/asus/models/gemma-4-E2B-it \
    --adapter /home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47 \
    --image <上一步得到的图路径> \
    --prompt '请描述图片中的主要内容,20字内。' \
    --max-new-tokens 64 \
    --load-dtype bf16
"
```
Expected: 打印一段中文描述(非空、非乱码、非 Latin)。**这一步失败,后续 Task 全部阻塞**,回退查环境。

- [ ] **Step 0.6: 记录基线信息到 spec 备注**

不 commit,只在本地记录:
- 实际 transformers 版本(如与 5.13.1 不同)
- Step 0.5 冒烟推理耗时(secs)
- GPU 显存峰值(nvidia-smi 观察,若 NVML mismatch 用 `torch.cuda.memory_allocated()`)

用于后续 Task 5 的 E2B 加载性能对照。

---

## Task 1: `runtime_settings` KV 表 + get/set/list_settings

**Files:**
- Modify: `backend/db.py` (在 `_create_schema` 里加 DDL,类底部加方法)
- Modify: `backend/tests/test_memory_store.py`

- [ ] **Step 1.1: 写失败测试**

在 `backend/tests/test_memory_store.py` 末尾追加:

```python
    def test_runtime_settings_table_is_idempotent_and_seeded_with_default_vlm_backend(self):
        # DDL 幂等: 第二次创建同一个 store 不应抛错
        store_again = MemoryStore(self.store.path)
        self.assertEqual(store_again.get_setting("vlm_backend"), "ollama_12b")
        store_again.close()

    def test_runtime_settings_set_get_roundtrip(self):
        self.store.set_setting("vlm_backend", "e2b_lora")
        self.assertEqual(self.store.get_setting("vlm_backend"), "e2b_lora")
        self.store.set_setting("vlm_backend", "ollama_12b")
        self.assertEqual(self.store.get_setting("vlm_backend"), "ollama_12b")

    def test_runtime_settings_get_returns_default_when_key_absent(self):
        self.assertIsNone(self.store.get_setting("unknown_key"))
        self.assertEqual(self.store.get_setting("unknown_key", "fallback"), "fallback")

    def test_runtime_settings_list_returns_all_kv_pairs(self):
        self.store.set_setting("vlm_backend", "e2b_lora")
        self.store.set_setting("custom_key", "custom_val")
        settings = self.store.list_settings()
        self.assertEqual(settings.get("vlm_backend"), "e2b_lora")
        self.assertEqual(settings.get("custom_key"), "custom_val")

    def test_runtime_settings_set_updates_updated_at(self):
        self.store.set_setting("vlm_backend", "e2b_lora")
        row1 = self.store.connection.execute(
            "SELECT updated_at FROM runtime_settings WHERE key='vlm_backend'"
        ).fetchone()
        import time
        time.sleep(1.1)  # SQLite datetime('now') 精度是秒
        self.store.set_setting("vlm_backend", "ollama_12b")
        row2 = self.store.connection.execute(
            "SELECT updated_at FROM runtime_settings WHERE key='vlm_backend'"
        ).fetchone()
        self.assertNotEqual(row1["updated_at"], row2["updated_at"])
```

- [ ] **Step 1.2: 运行测试确认失败**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest backend.tests.test_memory_store -v 2>&1 | tail -20
"
```
Expected: 上述 5 个新测试全部 `ERROR: no attribute 'get_setting'` 或类似。

- [ ] **Step 1.3: 实现 DDL 和方法**

修改 `backend/db.py`:

在 `_create_schema` 的 `executescript` 字符串**末尾** `""" )` 之前,追加(注意保留原有 `);` 收尾风格):
```sql
            CREATE TABLE IF NOT EXISTS runtime_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
```

在 `_create_schema` 方法**结尾**、`executescript` 之后追加:
```python
        self.connection.execute(
            "INSERT OR IGNORE INTO runtime_settings (key, value) VALUES (?, ?)",
            ("vlm_backend", "ollama_12b"),
        )
        self.connection.commit()
```

在 `close(self)` 方法**之前**(靠近类头部,与 __init__ 一起)加三个方法:
```python
    def get_setting(self, key, default=None):
        row = self.connection.execute(
            "SELECT value FROM runtime_settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row is not None else default

    def set_setting(self, key, value):
        self.connection.execute(
            "INSERT INTO runtime_settings (key, value, updated_at) "
            "VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = datetime('now')",
            (key, str(value)),
        )
        self.connection.commit()

    def list_settings(self):
        rows = self.connection.execute(
            "SELECT key, value FROM runtime_settings"
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}
```

- [ ] **Step 1.4: 运行测试确认通过**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest backend.tests.test_memory_store -v 2>&1 | tail -20
"
```
Expected: 新增 5 个测试全 PASS;现有 test_memory_store 所有测试也 PASS(不破坏兼容)。

- [ ] **Step 1.5: 全量 backend 测试**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest discover -s backend/tests -v 2>&1 | tail -5
"
```
Expected: `OK` 或 `OK (skipped=N)`,不出现 `FAILED`。

- [ ] **Step 1.6: Commit**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  git add backend/db.py backend/tests/test_memory_store.py &&
  git commit -m 'feat(db): add runtime_settings kv table

- schema idempotent DDL in MemoryStore._create_schema
- get_setting/set_setting/list_settings helpers
- seed vlm_backend=ollama_12b on first create'
"
```

---

## Task 2: 抽 `OllamaBackend` 出来(行为不变)

把 `GammaClient` 的所有方法搬到 `OllamaBackend` 里,`GammaClient` 变成薄壳只 `self._ollama = OllamaBackend(...)` 并把所有方法委托过去。这一步是**纯 refactor**,不加新功能,保证 `test_model_clients.py` 全绿。

**Files:**
- Modify: `backend/model_clients.py`

- [ ] **Step 2.1: 现有测试基线**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest backend.tests.test_model_clients -v 2>&1 | tail -3
"
```
Expected: `OK`。记录测试数量。

- [ ] **Step 2.2: Refactor**

在 `backend/model_clients.py` 里:

1. 把现有 `class GammaClient:` 整体**重命名**为 `class OllamaBackend:`。
2. 在 `OllamaBackend` 类**开头**加两个属性:
   ```python
   name = "ollama_12b"
   
   @property
   def model_name(self):
       return self.model
   
   @property
   def endpoint(self):
       return self.base_url
   ```
3. 在文件末尾(所有类之后)**新增** `GammaClient` facade:
   ```python
   class GammaClient:
       """Facade routing to the currently active VLM backend.
       
       For Task 2 (refactor), always routes to Ollama; multi-backend routing
       lands in Task 5.
       """
       def __init__(self, base_url=None, model=None, timeout=None, keep_alive=None):
           self._ollama = OllamaBackend(base_url, model, timeout, keep_alive)
   
       @property
       def base_url(self):
           return self._ollama.base_url
   
       @property
       def model(self):
           return self._ollama.model
   
       @property
       def keep_alive(self):
           return self._ollama.keep_alive
   
       @property
       def timeout(self):
           return self._ollama.timeout
   
       # 所有方法直接委托(名字与 OllamaBackend 一致)
       def chat(self, *args, **kwargs):                     return self._ollama.chat(*args, **kwargs)
       def analyze_image(self, *args, **kwargs):            return self._ollama.analyze_image(*args, **kwargs)
       def analyze_image_focus(self, *args, **kwargs):      return self._ollama.analyze_image_focus(*args, **kwargs)
       def analyze_person_appearance(self, *args, **kwargs):return self._ollama.analyze_person_appearance(*args, **kwargs)
       def analyze_text(self, *args, **kwargs):             return self._ollama.analyze_text(*args, **kwargs)
       def summarize_event(self, *args, **kwargs):          return self._ollama.summarize_event(*args, **kwargs)
       def embed_text(self, *args, **kwargs):               return self._ollama.embed_text(*args, **kwargs)
       def answer(self, *args, **kwargs):                   return self._ollama.answer(*args, **kwargs)
       # _core_vision_options / _encode_core_image 是内部方法,不需要暴露
   ```

- [ ] **Step 2.3: 运行现有测试确认无破坏**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest backend.tests.test_model_clients -v 2>&1 | tail -3
"
```
Expected: 与 Step 2.1 完全一致的 `OK` 计数。

注: 现有测试 `@patch("backend.model_clients.httpx.post")` 仍然生效,因为 mock 的是模块级 `httpx`,`OllamaBackend.chat` 内部调用的是同一个 `httpx.post`。

- [ ] **Step 2.4: 全量测试**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest discover -s backend/tests -v 2>&1 | tail -3
"
```
Expected: `OK`。

- [ ] **Step 2.5: Commit**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  git add backend/model_clients.py &&
  git commit -m 'refactor(model_clients): extract OllamaBackend from GammaClient

- rename existing GammaClient body to OllamaBackend (name=ollama_12b)
- new GammaClient facade delegates all methods to OllamaBackend
- behavior unchanged; existing tests remain green'
"
```

---

## Task 3: `VLMBackend` protocol + `E2BBackend` stub

只加接口和空实现(每个方法 `raise NotImplementedError`)。真实 HTTP wiring 到 Task 5。

**Files:**
- Modify: `backend/model_clients.py`
- Create: `backend/tests/test_e2b_backend.py`

- [ ] **Step 3.1: 写失败测试**

Create `backend/tests/test_e2b_backend.py`:
```python
import unittest
from backend.model_clients import E2BBackend


class E2BBackendStubTests(unittest.TestCase):
    def setUp(self):
        self.backend = E2BBackend(base_url="http://127.0.0.1:8100", timeout=30.0)

    def test_name_and_endpoint(self):
        self.assertEqual(self.backend.name, "e2b_lora")
        self.assertEqual(self.backend.endpoint, "http://127.0.0.1:8100")
        self.assertEqual(self.backend.model_name, "gemma-4-e2b-it+lora-v2")

    def test_all_seven_generative_methods_raise_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.backend.chat("prompt")
        with self.assertRaises(NotImplementedError):
            self.backend.analyze_image("/tmp/x.jpg")
        with self.assertRaises(NotImplementedError):
            self.backend.analyze_image_focus("/tmp/x.jpg", "place")
        with self.assertRaises(NotImplementedError):
            self.backend.analyze_person_appearance("/tmp/x.jpg")
        with self.assertRaises(NotImplementedError):
            self.backend.analyze_text("hello", "text")
        with self.assertRaises(NotImplementedError):
            self.backend.summarize_event({}, [])
        with self.assertRaises(NotImplementedError):
            self.backend.answer("q", {})

    def test_embed_text_raises_not_supported(self):
        # E2B never implements embed_text; always defers to OllamaBackend via facade
        with self.assertRaises(NotImplementedError):
            self.backend.embed_text("hello")

    def test_base_url_is_stripped_of_trailing_slash(self):
        backend = E2BBackend(base_url="http://127.0.0.1:8100/", timeout=30.0)
        self.assertEqual(backend.endpoint, "http://127.0.0.1:8100")
```

- [ ] **Step 3.2: 运行测试确认失败**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest backend.tests.test_e2b_backend -v 2>&1 | tail -10
"
```
Expected: `ImportError: cannot import name 'E2BBackend'`。

- [ ] **Step 3.3: 实现 stub**

在 `backend/model_clients.py` 里 `OllamaBackend` 定义**之后**、`GammaClient` 定义**之前**加:

```python
class E2BBackend:
    """HTTP client talking to the Gemma-4-E2B-it + LoRA V2 server on :8100.
    
    Task 3 lands only the stub; real HTTP wiring is in Task 5.
    """
    name = "e2b_lora"
    model_name = "gemma-4-e2b-it+lora-v2"

    def __init__(self, base_url=None, timeout=None):
        self.base_url = (base_url or os.getenv("E2B_BASE_URL", "http://127.0.0.1:8100")).rstrip("/")
        self.timeout = timeout or float(os.getenv("E2B_TIMEOUT_SECONDS", "180"))

    @property
    def endpoint(self):
        return self.base_url

    def chat(self, prompt, images=None, vision_options=None, json_mode=True):
        raise NotImplementedError("E2BBackend.chat lands in Task 5")

    def analyze_image(self, path, metadata=None):
        raise NotImplementedError("E2BBackend.analyze_image lands in Task 5")

    def analyze_image_focus(self, path, dimension, metadata=None):
        raise NotImplementedError("E2BBackend.analyze_image_focus lands in Task 5")

    def analyze_person_appearance(self, path, metadata=None):
        raise NotImplementedError("E2BBackend.analyze_person_appearance lands in Task 5")

    def analyze_text(self, text, source_type="text"):
        raise NotImplementedError("E2BBackend.analyze_text lands in Task 5")

    def summarize_event(self, event, observations):
        raise NotImplementedError("E2BBackend.summarize_event lands in Task 5")

    def answer(self, query, context):
        raise NotImplementedError("E2BBackend.answer lands in Task 5")

    def embed_text(self, text):
        # E2B intentionally does NOT support embeddings; GammaClient facade
        # always routes embed_text to OllamaBackend to preserve vector space.
        raise NotImplementedError("E2BBackend.embed_text is intentionally unsupported")

    def health(self):
        raise NotImplementedError("E2BBackend.health lands in Task 5")
```

- [ ] **Step 3.4: 运行测试确认通过**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest backend.tests.test_e2b_backend -v 2>&1 | tail -10
"
```
Expected: 5 个测试全 PASS。

- [ ] **Step 3.5: Commit**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  git add backend/model_clients.py backend/tests/test_e2b_backend.py &&
  git commit -m 'feat(model_clients): add E2BBackend stub

- new class with 9 methods raising NotImplementedError
- name=e2b_lora, model_name=gemma-4-e2b-it+lora-v2
- reads E2B_BASE_URL / E2B_TIMEOUT_SECONDS env
- HTTP wiring lands in Task 5'
"
```

---

## Task 4a: E2B server 的 `ollama_shape.py`(纯函数)

先把请求解析、响应格式化这些**无 GPU 依赖**的转换逻辑抽成纯函数,能在没 E2B 环境的机器上测试。

**Files:**
- Create: `services/e2b_server/__init__.py`(空)
- Create: `services/e2b_server/ollama_shape.py`
- Create: `services/e2b_server/tests/__init__.py`(空)
- Create: `services/e2b_server/tests/test_ollama_shape.py`

- [ ] **Step 4a.1: 写失败测试**

Create `services/e2b_server/tests/test_ollama_shape.py`:
```python
import base64
import unittest
from services.e2b_server.ollama_shape import (
    extract_prompt_and_images,
    build_chat_response,
    build_generate_response,
    map_options,
    JSON_TRAILING_HINT,
)


class ExtractPromptAndImagesTests(unittest.TestCase):
    def test_ollama_style_top_level_images(self):
        # Ollama /api/chat 的现有格式: messages[0].content 是 str + images 顶层数组
        payload = {
            "model": "gemma-4-e2b-it+lora-v2",
            "messages": [{"role": "user", "content": "描述一下"}],
            "images": [base64.b64encode(b"fake-jpeg").decode("ascii")],
        }
        prompt, images = extract_prompt_and_images(payload)
        self.assertEqual(prompt, "描述一下")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0], base64.b64encode(b"fake-jpeg").decode("ascii"))

    def test_message_content_as_multipart_list(self):
        # 兼容 OpenAI-vision 风格: content 是 [{type:image,...},{type:text,...}]
        b64 = base64.b64encode(b"fake").decode("ascii")
        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"base64:{b64}"},
                    {"type": "text", "text": "hi"},
                ],
            }],
        }
        prompt, images = extract_prompt_and_images(payload)
        self.assertEqual(prompt, "hi")
        self.assertEqual(images, [b64])

    def test_prompt_only_no_images(self):
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        prompt, images = extract_prompt_and_images(payload)
        self.assertEqual(prompt, "hello")
        self.assertEqual(images, [])

    def test_json_mode_appends_json_hint(self):
        payload = {"messages": [{"role": "user", "content": "give data"}], "format": "json"}
        prompt, _ = extract_prompt_and_images(payload)
        self.assertTrue(prompt.endswith(JSON_TRAILING_HINT))


class BuildResponseTests(unittest.TestCase):
    def test_chat_response_matches_ollama_shape(self):
        response = build_chat_response("gemma-4-e2b-it+lora-v2", "生成的文字")
        self.assertEqual(response["model"], "gemma-4-e2b-it+lora-v2")
        self.assertTrue(response["done"])
        self.assertEqual(response["message"]["role"], "assistant")
        self.assertEqual(response["message"]["content"], "生成的文字")

    def test_generate_response_matches_ollama_shape(self):
        response = build_generate_response("gemma-4-e2b-it+lora-v2", "生成的文字")
        self.assertEqual(response["model"], "gemma-4-e2b-it+lora-v2")
        self.assertTrue(response["done"])
        self.assertEqual(response["response"], "生成的文字")


class MapOptionsTests(unittest.TestCase):
    def test_num_predict_maps_to_max_new_tokens(self):
        result = map_options({"num_predict": 320})
        self.assertEqual(result["max_new_tokens"], 320)

    def test_temperature_zero_uses_greedy(self):
        result = map_options({"temperature": 0})
        self.assertFalse(result["do_sample"])

    def test_temperature_positive_enables_sampling(self):
        result = map_options({"temperature": 0.7})
        self.assertTrue(result["do_sample"])
        self.assertEqual(result["temperature"], 0.7)

    def test_defaults_when_options_absent(self):
        result = map_options({})
        self.assertEqual(result["max_new_tokens"], 512)
        self.assertFalse(result["do_sample"])
```

- [ ] **Step 4a.2: 运行测试确认失败**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate sentrix-e2b &&
  python -m unittest services.e2b_server.tests.test_ollama_shape -v 2>&1 | tail -10
"
```
Expected: `ModuleNotFoundError: No module named 'services'` 或 `ImportError`。

- [ ] **Step 4a.3: 实现纯函数**

Create `services/e2b_server/__init__.py`:
```python
```
(空文件)

Create `services/e2b_server/tests/__init__.py`:
```python
```
(空文件)

Create `services/e2b_server/ollama_shape.py`:
```python
"""Pure request/response conversion between Ollama-shaped HTTP and E2B model calls.

Kept pure (no torch imports) so it can be unit-tested without GPU / conda env.
"""
from __future__ import annotations

from typing import Any


JSON_TRAILING_HINT = "\n仅输出 JSON,不要包裹在代码块中。"


def extract_prompt_and_images(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """Return (prompt_text, base64_image_list) from an Ollama /api/chat body.

    Accepts two shapes for image passthrough:
    1. Ollama current shape: `payload.images = [base64, ...]`, message.content=str
    2. OpenAI-vision multipart: message.content=[{type:image,image:"base64:..."},{type:text,text:"..."}]
    """
    messages = payload.get("messages") or []
    if not messages:
        # /api/generate style
        prompt = str(payload.get("prompt", ""))
        images = _decode_top_level_images(payload.get("images"))
        return _apply_json_hint(prompt, payload), images

    first_user = next((m for m in messages if m.get("role") == "user"), messages[0])
    content = first_user.get("content", "")

    prompt_parts: list[str] = []
    images: list[str] = []
    if isinstance(content, str):
        prompt_parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                prompt_parts.append(str(item.get("text", "")))
            elif item_type == "image":
                image_ref = str(item.get("image", ""))
                if image_ref.startswith("base64:"):
                    images.append(image_ref[len("base64:"):])
                elif image_ref:
                    images.append(image_ref)

    images.extend(_decode_top_level_images(payload.get("images")))
    prompt = "".join(prompt_parts).strip()
    return _apply_json_hint(prompt, payload), images


def _decode_top_level_images(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []


def _apply_json_hint(prompt: str, payload: dict[str, Any]) -> str:
    if payload.get("format") == "json" and JSON_TRAILING_HINT not in prompt:
        return prompt + JSON_TRAILING_HINT
    return prompt


def map_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Translate Ollama's options.{num_predict,temperature,...} to HF generation_config."""
    options = options or {}
    temperature = float(options.get("temperature", 0))
    return {
        "max_new_tokens": int(options.get("num_predict", 512)),
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,  # HF ignores when do_sample=False
    }


def build_chat_response(model_name: str, text: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "created_at": None,
        "message": {"role": "assistant", "content": text},
        "done": True,
    }


def build_generate_response(model_name: str, text: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "created_at": None,
        "response": text,
        "done": True,
    }
```

- [ ] **Step 4a.4: 运行测试确认通过**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate sentrix-e2b &&
  python -m unittest services.e2b_server.tests.test_ollama_shape -v 2>&1 | tail -20
"
```
Expected: 全部 PASS(约 10 个测试)。

- [ ] **Step 4a.5: Commit**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  git add services/e2b_server/__init__.py services/e2b_server/ollama_shape.py services/e2b_server/tests/__init__.py services/e2b_server/tests/test_ollama_shape.py &&
  git commit -m 'feat(services): e2b_server ollama_shape pure conversion helpers

- extract_prompt_and_images accepts ollama top-level images and openai-vision multipart
- map_options translates num_predict/temperature to HF generation kwargs
- build_chat_response / build_generate_response mimic ollama shape
- JSON_TRAILING_HINT appended when format=json requested'
"
```

---

## Task 4b: E2B server 的 `model.py`(loader + generate)

模型加载和推理的胶水,严格照交付脚本 `deploy_gemma4e2b_lora.py` 的加载顺序。**测试只做接口/mock**,GPU 冒烟放 Task 4c 之后。

**Files:**
- Create: `services/e2b_server/model.py`
- Create: `services/e2b_server/tests/test_model_smoke.py`(GPU 冒烟,默认 skip)

- [ ] **Step 4b.1: 写冒烟测试(默认 skip,GPU 时才跑)**

Create `services/e2b_server/tests/test_model_smoke.py`:
```python
import os
import unittest
from pathlib import Path


REQUIRE_GPU = os.getenv("SENTRIX_E2B_SMOKE") == "1"


@unittest.skipUnless(REQUIRE_GPU, "set SENTRIX_E2B_SMOKE=1 to enable GPU smoke test")
class E2BModelSmokeTests(unittest.TestCase):
    """Real GPU load. Off by default; run with SENTRIX_E2B_SMOKE=1 on 153."""

    def test_load_and_generate_short_text(self):
        from services.e2b_server.model import E2BModel

        base = os.environ["E2B_BASE_MODEL"]
        adapter = os.environ["E2B_ADAPTER"]
        self.assertTrue(Path(base).is_dir(), f"base missing: {base}")
        self.assertTrue(Path(adapter).is_dir(), f"adapter missing: {adapter}")

        model = E2BModel(base, adapter, dtype="bf16")
        model.load()
        self.assertTrue(model.is_loaded())

        images = [Path(base).parent / "gemma-4-e2b-lora-v2/PACKAGE_CONTENTS.clean.sha256"]
        # 不用 real image; text-only smoke
        text = model.generate(prompt="用一句中文说明你是谁。", images=[], max_new_tokens=32)
        self.assertTrue(text.strip())
        self.assertFalse(text.strip().startswith("Error"))
        model.unload()
        self.assertFalse(model.is_loaded())
```

- [ ] **Step 4b.2: 实现 `model.py`**

Create `services/e2b_server/model.py`:
```python
"""E2B model loader + generate wrapper.

Follows the exact load sequence from the delivery script
`deploy_gemma4e2b_lora.py`: AutoProcessor + AutoModelForMultimodalLM (fallback
AutoModelForImageTextToText) + PeftModel.from_pretrained.
"""
from __future__ import annotations

import asyncio
import base64
import inspect
import io
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("e2b_server.model")


_DTYPE_TABLE = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


class E2BModel:
    """Encapsulates lazy load, generate, unload for the Gemma-4-E2B + LoRA V2 stack."""

    def __init__(self, base_dir: str, adapter_dir: str, dtype: str = "bf16", device_map: str = "cuda:0"):
        self._base_dir = str(Path(base_dir).resolve())
        self._adapter_dir = str(Path(adapter_dir).resolve())
        self._dtype_key = dtype
        self._device_map = device_map
        self._processor = None
        self._model = None
        self._load_lock = asyncio.Lock()
        self._gen_lock = asyncio.Lock()
        self._accepted_forward_keys = None

    def is_loaded(self) -> bool:
        return self._model is not None

    async def ensure_loaded(self) -> None:
        if self.is_loaded():
            return
        async with self._load_lock:
            if self.is_loaded():
                return
            await asyncio.to_thread(self._blocking_load)

    def load(self) -> None:
        # Sync convenience for tests/smoke; production uses ensure_loaded.
        self._blocking_load()

    def _blocking_load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        dtype = getattr(torch, _DTYPE_TABLE[self._dtype_key])
        LOGGER.info("loading base=%s dtype=%s device_map=%s", self._base_dir, self._dtype_key, self._device_map)

        try:
            from transformers import AutoModelForMultimodalLM
            base = AutoModelForMultimodalLM.from_pretrained(
                self._base_dir, dtype=dtype, device_map=self._device_map, low_cpu_mem_usage=True
            )
            loader = "AutoModelForMultimodalLM"
        except Exception as error:
            LOGGER.warning("AutoModelForMultimodalLM unavailable (%s); falling back to AutoModelForImageTextToText", error)
            base = AutoModelForImageTextToText.from_pretrained(
                self._base_dir, dtype=dtype, device_map=self._device_map, low_cpu_mem_usage=True
            )
            loader = "AutoModelForImageTextToText"

        from peft import PeftModel
        self._model = PeftModel.from_pretrained(base, self._adapter_dir)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self._base_dir)
        self._accepted_forward_keys = self._compute_accepted_keys(self._model)
        LOGGER.info("loaded via=%s accepted_keys=%s", loader, self._accepted_forward_keys)

    @staticmethod
    def _compute_accepted_keys(model) -> set[str] | None:
        try:
            signature = inspect.signature(model.forward)
        except (TypeError, ValueError):
            return None
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            return None
        return set(signature.parameters)

    def unload(self) -> None:
        if self._model is None:
            return
        LOGGER.info("unloading E2B model")
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._accepted_forward_keys = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    async def generate(self, prompt: str, images: list[str], max_new_tokens: int = 512,
                       do_sample: bool = False, temperature: float = 1.0) -> str:
        await self.ensure_loaded()
        async with self._gen_lock:
            return await asyncio.to_thread(
                self._blocking_generate, prompt, images, max_new_tokens, do_sample, temperature
            )

    def _blocking_generate(self, prompt: str, image_b64_list: list[str], max_new_tokens: int,
                          do_sample: bool, temperature: float) -> str:
        import torch
        from PIL import Image

        pil_images = [Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                      for b64 in (image_b64_list or [])]

        if pil_images:
            content = [{"type": "image", "image": pil_images[0]}, {"type": "text", "text": prompt}]
        else:
            content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]

        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        proc_images = [pil_images] if pil_images else None
        processed = self._processor(text=[text], images=proc_images, padding=True, return_tensors="pt")

        unused = set(getattr(self._processor, "unused_input_names", ()) or ())
        inputs = {k: v for k, v in processed.items() if v is not None and k not in unused}
        if self._accepted_forward_keys is not None:
            inputs = {k: v for k, v in inputs.items() if k in self._accepted_forward_keys}
        if "input_ids" not in inputs:
            raise RuntimeError("processor did not produce input_ids")

        input_device = next(self._model.parameters()).device
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                continue
            value = value.to(input_device)
            if value.is_floating_point():
                value = value.to(getattr(torch, _DTYPE_TABLE[self._dtype_key]))
            inputs[key] = value

        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs, do_sample=do_sample, temperature=temperature, max_new_tokens=max_new_tokens
            )
        answer_ids = generated[:, prompt_length:]
        answer = self._processor.batch_decode(answer_ids, skip_special_tokens=True)[0]
        return answer.strip()
```

- [ ] **Step 4b.3: 语法自检**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate sentrix-e2b &&
  python -c 'import services.e2b_server.model; print(\"import OK\")'
"
```
Expected: `import OK`(不真的加载模型,只 import 模块)。

- [ ] **Step 4b.4: 冒烟测试(需 GPU)**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate sentrix-e2b &&
  SENTRIX_E2B_SMOKE=1 \
  E2B_BASE_MODEL=/home/asus/models/gemma-4-E2B-it \
  E2B_ADAPTER=/home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47 \
  python -m unittest services.e2b_server.tests.test_model_smoke -v 2>&1 | tail -15
"
```
Expected: 1 个测试 PASS(生成一段非空中文文本),显存约 10-12 GB。若 OOM,查其它 GPU 占用 `nvidia-smi`;若 12B 常驻,先 `curl -s http://127.0.0.1:11435/api/generate -d '{"model":"gemma4:12b","prompt":"","keep_alive":0}'` 让 Ollama 释放。

- [ ] **Step 4b.5: Commit**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  git add services/e2b_server/model.py services/e2b_server/tests/test_model_smoke.py &&
  git commit -m 'feat(services): e2b_server model loader with peft lora mount

- E2BModel encapsulates lazy load / generate / unload
- follows delivery deploy script load order: AutoProcessor + AutoModelForMultimodalLM (fallback AutoModelForImageTextToText) + PeftModel
- asyncio.Lock serializes generate to protect single-GPU
- smoke test gated by SENTRIX_E2B_SMOKE=1'
"
```

---

## Task 4c: E2B server 的 FastAPI 主入口

**Files:**
- Create: `services/e2b_server/app.py`

- [ ] **Step 4c.1: 实现 FastAPI app**

Create `services/e2b_server/app.py`:
```python
"""FastAPI HTTP shell for the Gemma-4-E2B-it + LoRA V2 model.

Endpoints mimic Ollama's /api/chat, /api/generate, /api/health so that the
Sentrix backend's E2BBackend can talk to it with minimal shape adaptation.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.e2b_server.model import E2BModel
from services.e2b_server.ollama_shape import (
    extract_prompt_and_images,
    build_chat_response,
    build_generate_response,
    map_options,
)


LOGGER = logging.getLogger("e2b_server.app")
MODEL_NAME = "gemma-4-e2b-it+lora-v2"

app = FastAPI(title="Sentrix E2B Server", version="0.1.0")

_model: E2BModel | None = None
_load_error: str | None = None


def _get_model() -> E2BModel:
    global _model, _load_error
    if _model is None:
        base = os.environ.get("E2B_BASE_MODEL")
        adapter = os.environ.get("E2B_ADAPTER")
        if not base or not adapter:
            raise HTTPException(500, "E2B_BASE_MODEL and E2B_ADAPTER env vars are required")
        _model = E2BModel(base, adapter, dtype=os.environ.get("E2B_DTYPE", "bf16"))
    return _model


class ChatBody(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] | None = None
    images: list[str] | None = None
    format: str | None = None
    stream: bool = False
    options: dict[str, Any] | None = None
    keep_alive: Any = None  # ignored; kept for Ollama compatibility


class GenerateBody(BaseModel):
    model: str | None = None
    prompt: str = ""
    images: list[str] | None = None
    format: str | None = None
    stream: bool = False
    options: dict[str, Any] | None = None
    keep_alive: Any = None


@app.get("/api/health")
async def health():
    model = _get_model() if _load_error is None else None
    return {
        "status": "ok" if _load_error is None else "error",
        "model": MODEL_NAME,
        "adapter": "V2_student_step47",
        "dtype": os.environ.get("E2B_DTYPE", "bf16"),
        "loaded": bool(model and model.is_loaded()),
        "error": _load_error,
    }


@app.post("/api/chat")
async def chat(body: ChatBody):
    payload = body.model_dump(exclude_none=False)
    prompt, images = extract_prompt_and_images(payload)
    gen_kwargs = map_options(payload.get("options"))
    model = _get_model()
    try:
        text = await model.generate(prompt=prompt, images=images, **gen_kwargs)
    except Exception as error:
        LOGGER.exception("chat generation failed")
        raise HTTPException(500, f"generation failed: {error}") from error
    return build_chat_response(MODEL_NAME, text)


@app.post("/api/generate")
async def generate(body: GenerateBody):
    payload = body.model_dump(exclude_none=False)
    prompt, images = extract_prompt_and_images(payload)
    gen_kwargs = map_options(payload.get("options"))
    model = _get_model()
    try:
        text = await model.generate(prompt=prompt, images=images, **gen_kwargs)
    except Exception as error:
        LOGGER.exception("generate failed")
        raise HTTPException(500, f"generation failed: {error}") from error
    return build_generate_response(MODEL_NAME, text)


@app.post("/api/embeddings")
async def embeddings():
    raise HTTPException(501, "embeddings intentionally not supported on E2B; use Ollama 12B via GammaClient.embed_text")


@app.post("/admin/load")
async def admin_load():
    global _load_error
    _load_error = None
    try:
        await _get_model().ensure_loaded()
    except Exception as error:
        _load_error = str(error)
        LOGGER.exception("admin load failed")
        raise HTTPException(500, f"load failed: {error}") from error
    return {"loaded": True}


@app.post("/admin/unload")
async def admin_unload():
    global _model, _load_error
    if _model is not None:
        _model.unload()
    _load_error = None
    return {"loaded": False}
```

- [ ] **Step 4c.2: 语法 + 冷启动测试**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate sentrix-e2b &&
  python -c 'from services.e2b_server.app import app; print(\"routes:\", [r.path for r in app.routes])'
"
```
Expected: 输出 `routes: [..., '/api/health', '/api/chat', '/api/generate', '/api/embeddings', '/admin/load', '/admin/unload']`。

- [ ] **Step 4c.3: 端到端启动 + curl 冒烟**

在 153 上单独开一个 shell 起服务(不用 nohup,方便看日志):
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate sentrix-e2b &&
  E2B_BASE_MODEL=/home/asus/models/gemma-4-E2B-it \
  E2B_ADAPTER=/home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47 \
  nohup uvicorn services.e2b_server.app:app --host 127.0.0.1 --port 8100 --workers 1 \
    > /tmp/e2b-server.log 2>&1 &
  echo E2B_PID=\$!
  sleep 3
  curl -s http://127.0.0.1:8100/api/health | python3 -m json.tool
"
```
Expected: `status: ok`, `loaded: false`, `error: null`。

Load + generate:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  curl -s -X POST http://127.0.0.1:8100/admin/load
  echo ''
  # 让它 warm 一会儿
  sleep 30
  curl -s http://127.0.0.1:8100/api/health | python3 -m json.tool
  echo '---'
  # 纯文本生成
  curl -s -X POST http://127.0.0.1:8100/api/generate \
    -H 'Content-Type: application/json' \
    -d '{\"prompt\":\"用一句话说明你是谁\",\"options\":{\"num_predict\":32}}' | python3 -m json.tool
"
```
Expected: `loaded: true`;generate 返回一段非空中文。

- [ ] **Step 4c.4: 关掉临时服务**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  pkill -f 'uvicorn services.e2b_server.app'
  sleep 1
  curl -s http://127.0.0.1:8100/api/health 2>&1 | head -3
"
```
Expected: `curl: (7) Failed to connect`(服务已停)。

- [ ] **Step 4c.5: Commit**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  git add services/e2b_server/app.py &&
  git commit -m 'feat(services): e2b_server FastAPI app on port 8100

- /api/chat and /api/generate mimic ollama response shape
- /api/health reports model/adapter/dtype/loaded
- /api/embeddings returns 501 (embed stays on ollama 12b)
- /admin/load and /admin/unload for gpu residency control'
"
```

---

## Task 4d: E2B server README

**Files:**
- Create: `services/e2b_server/README.md`

- [ ] **Step 4d.1: 写 README**

Create `services/e2b_server/README.md`:
```markdown
# services/e2b_server

FastAPI 独立服务,加载 Gemma-4-E2B-it + LoRA V2 (student_step47),提供 mimic Ollama 的 HTTP 接口给 Sentrix 后端。

## 依赖

`sentrix-e2b` conda env(python 3.11):
- torch 2.8.0+cu126
- transformers 5.13.1
- peft 0.19.1
- fastapi + uvicorn + httpx + Pillow + accelerate

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `E2B_BASE_MODEL` | (必填) | Gemma-4-E2B-it 基模目录 |
| `E2B_ADAPTER` | (必填) | LoRA adapter 目录 (含 adapter_config.json + adapter_model.safetensors) |
| `E2B_DTYPE` | `bf16` | 加载精度,可选 `bf16` / `fp16` / `fp32` |
| `E2B_HOST` | `127.0.0.1` | uvicorn host |
| `E2B_PORT` | `8100` | uvicorn port |

## HTTP 接口

| 路径 | 方法 | 说明 |
|---|---|---|
| `/api/health` | GET | 返回 `{status,model,adapter,dtype,loaded,error}` |
| `/api/chat` | POST | Ollama 兼容的 chat,请求体 `{messages,images?,format?,options?}`,响应 `{message:{role,content},model,done}` |
| `/api/generate` | POST | Ollama 兼容的 generate,请求体 `{prompt,images?,options?}`,响应 `{response,model,done}` |
| `/api/embeddings` | POST | 返回 501。E2B 不提供 embedding;Sentrix 走 Ollama 12B。 |
| `/admin/load` | POST | 幂等,触发懒加载。返回 `{loaded:true}` |
| `/admin/unload` | POST | 释放显存 + `torch.cuda.empty_cache()`。返回 `{loaded:false}` |

## 启动

用项目提供的 wrapper:
```bash
./scripts/runtime/start_sentrix_e2b.sh
```

或手动:
```bash
conda activate sentrix-e2b
E2B_BASE_MODEL=/home/asus/models/gemma-4-E2B-it \
E2B_ADAPTER=/home/asus/models/gemma-4-e2b-lora-v2/artifacts/lora/V2_student_step47 \
uvicorn services.e2b_server.app:app --host 127.0.0.1 --port 8100 --workers 1
```

## 测试

纯函数单测(无 GPU):
```bash
python -m unittest services.e2b_server.tests.test_ollama_shape -v
```

GPU 冒烟(需要真实模型):
```bash
SENTRIX_E2B_SMOKE=1 \
E2B_BASE_MODEL=... E2B_ADAPTER=... \
python -m unittest services.e2b_server.tests.test_model_smoke -v
```

## 已知限制

- **单请求串行**:`asyncio.Lock` 保证 GPU 一次只处理一个请求,批量场景请用 Ollama 12B。
- **无 JSON 强约束**:HF PEFT 没有 grammar decoder。`format=json` 只在 prompt 末尾追加 hint,依赖调用方 `parse_json_response()` 容错。
- **无 embedding**:向量库累积的 3840 维向量与 gemma4:12b 绑定,E2B 若参与 embedding 会破坏检索。
```

- [ ] **Step 4d.2: Commit**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  git add services/e2b_server/README.md &&
  git commit -m 'docs(services): add e2b_server README

- env vars, http contract, startup, tests, known limits'
"
```

---

## Task 5: E2BBackend HTTP wiring + GammaClient facade routing + TTL 缓存

补齐 `E2BBackend` 的 HTTP 实现,让 `GammaClient` facade 按 `runtime_settings.vlm_backend` 分发,`embed_text` 硬钉 Ollama。

**Files:**
- Modify: `backend/model_clients.py`
- Modify: `backend/tests/test_e2b_backend.py`
- Create: `backend/tests/test_vlm_router.py`

- [ ] **Step 5.1: 扩展 E2BBackend 测试**

替换 `backend/tests/test_e2b_backend.py` 里 `test_all_seven_generative_methods_raise_not_implemented` 和 `test_embed_text_raises_not_supported`,加入 HTTP mock 测试:

```python
import unittest
from unittest.mock import patch, MagicMock

from backend.model_clients import E2BBackend, ModelError


class E2BBackendHttpTests(unittest.TestCase):
    def setUp(self):
        self.backend = E2BBackend(base_url="http://127.0.0.1:8100", timeout=30.0)

    def test_name_and_endpoint(self):
        self.assertEqual(self.backend.name, "e2b_lora")
        self.assertEqual(self.backend.endpoint, "http://127.0.0.1:8100")
        self.assertEqual(self.backend.model_name, "gemma-4-e2b-it+lora-v2")

    def test_base_url_is_stripped_of_trailing_slash(self):
        self.assertEqual(E2BBackend(base_url="http://127.0.0.1:8100/").endpoint, "http://127.0.0.1:8100")

    @patch("backend.model_clients.httpx.post")
    def test_chat_hits_api_chat_and_returns_content(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "hi"}, "done": True}
        result = self.backend.chat("say hi")
        self.assertEqual(result, "hi")
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:8100/api/chat")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["messages"][0]["content"], "say hi")
        self.assertEqual(body["stream"], False)

    @patch("backend.model_clients.httpx.post")
    def test_chat_with_images_passes_top_level_images(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "{}"}}
        self.backend.chat("what", images=[{"base64": "abcd", "mime_type": "image/jpeg"}])
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["images"], ["abcd"])

    @patch("backend.model_clients.httpx.post")
    def test_chat_json_mode_sets_format_field(self, post):
        post.return_value.raise_for_status.return_value = None
        post.return_value.json.return_value = {"message": {"content": "{}"}}
        self.backend.chat("q", json_mode=True)
        self.assertEqual(post.call_args.kwargs["json"]["format"], "json")

    @patch("backend.model_clients.httpx.post")
    def test_chat_http_error_raises_model_error(self, post):
        import httpx
        post.side_effect = httpx.HTTPError("connection refused")
        with self.assertRaises(ModelError):
            self.backend.chat("q")

    @patch("backend.model_clients.httpx.get")
    def test_health_returns_dict_from_get(self, get):
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = {"status": "ok", "loaded": True, "model": "gemma-4-e2b-it+lora-v2"}
        result = self.backend.health()
        self.assertTrue(result["loaded"])
        self.assertEqual(get.call_args.args[0], "http://127.0.0.1:8100/api/health")

    @patch("backend.model_clients.httpx.get")
    def test_health_returns_unavailable_dict_on_error(self, get):
        import httpx
        get.side_effect = httpx.HTTPError("timeout")
        result = self.backend.health()
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["loaded"])

    def test_embed_text_is_intentionally_unsupported(self):
        with self.assertRaises(NotImplementedError):
            self.backend.embed_text("hello")
```

- [ ] **Step 5.2: 写 VLM router 测试**

Create `backend/tests/test_vlm_router.py`:
```python
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from backend.db import MemoryStore
from backend.model_clients import GammaClient, OllamaBackend, E2BBackend


class GammaClientFacadeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(f"{self.tmp.name}/db.sqlite")
        self.client = GammaClient()
        self.client.bind_store(self.store)
        # Force fresh cache for each test
        self.client.invalidate_backend_cache()

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_default_backend_is_ollama_12b(self):
        self.assertEqual(self.client.active_name, "ollama_12b")
        self.assertIsInstance(self.client._active(), OllamaBackend)

    def test_switching_to_e2b_lora_routes_chat_to_e2b(self):
        self.store.set_setting("vlm_backend", "e2b_lora")
        self.client.invalidate_backend_cache()
        self.assertEqual(self.client.active_name, "e2b_lora")
        self.assertIsInstance(self.client._active(), E2BBackend)

    def test_embed_text_always_hits_ollama_regardless_of_active_backend(self):
        self.store.set_setting("vlm_backend", "e2b_lora")
        self.client.invalidate_backend_cache()
        with patch.object(self.client._ollama, "embed_text", return_value=[0.1, 0.2]) as ollama_embed, \
             patch.object(self.client._e2b, "embed_text") as e2b_embed:
            result = self.client.embed_text("hello")
            self.assertEqual(result, [0.1, 0.2])
            ollama_embed.assert_called_once_with("hello")
            e2b_embed.assert_not_called()

    def test_ttl_cache_avoids_repeated_db_reads_within_window(self):
        with patch.object(self.store, "get_setting", wraps=self.store.get_setting) as spy:
            self.client.invalidate_backend_cache()
            _ = self.client.active_name
            _ = self.client.active_name
            _ = self.client.active_name
            self.assertEqual(spy.call_count, 1)  # first read; next two hit cache

    def test_invalidate_backend_cache_forces_next_read(self):
        with patch.object(self.store, "get_setting", wraps=self.store.get_setting) as spy:
            _ = self.client.active_name
            self.client.invalidate_backend_cache()
            _ = self.client.active_name
            self.assertEqual(spy.call_count, 2)

    def test_chat_delegates_to_active_backend(self):
        with patch.object(self.client._ollama, "chat", return_value="from ollama") as ollama_chat:
            self.assertEqual(self.client.chat("q"), "from ollama")
            ollama_chat.assert_called_once_with("q")
        self.store.set_setting("vlm_backend", "e2b_lora")
        self.client.invalidate_backend_cache()
        with patch.object(self.client._e2b, "chat", return_value="from e2b") as e2b_chat:
            self.assertEqual(self.client.chat("q"), "from e2b")
            e2b_chat.assert_called_once_with("q")

    def test_model_and_base_url_reflect_active_backend(self):
        self.store.set_setting("vlm_backend", "e2b_lora")
        self.client.invalidate_backend_cache()
        self.assertEqual(self.client.model, "gemma-4-e2b-it+lora-v2")
        self.assertTrue(self.client.base_url.startswith("http://"))

    def test_no_store_bound_falls_back_to_env(self):
        client = GammaClient()  # bind_store not called
        with patch.dict("os.environ", {"SENTRIX_VLM_BACKEND": "e2b_lora"}, clear=False):
            client.invalidate_backend_cache()
            self.assertEqual(client.active_name, "e2b_lora")

    def test_unknown_backend_name_falls_back_to_ollama(self):
        self.store.set_setting("vlm_backend", "nonsense_backend")
        self.client.invalidate_backend_cache()
        self.assertEqual(self.client.active_name, "ollama_12b")
```

- [ ] **Step 5.3: 运行测试确认失败**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest backend.tests.test_e2b_backend backend.tests.test_vlm_router -v 2>&1 | tail -20
"
```
Expected: 大多失败,信号是 `E2BBackend.chat lands in Task 5` 或 `AttributeError: ... invalidate_backend_cache` / `bind_store`。

- [ ] **Step 5.4: 实现 E2BBackend 真调用**

修改 `backend/model_clients.py` 里 `E2BBackend` 类,把所有 `NotImplementedError` 方法换成 HTTP 实现(除 `embed_text` 保留 NotImplementedError):

```python
class E2BBackend:
    name = "e2b_lora"
    model_name = "gemma-4-e2b-it+lora-v2"

    def __init__(self, base_url=None, timeout=None):
        self.base_url = (base_url or os.getenv("E2B_BASE_URL", "http://127.0.0.1:8100")).rstrip("/")
        self.timeout = timeout or float(os.getenv("E2B_TIMEOUT_SECONDS", "180"))

    @property
    def endpoint(self):
        return self.base_url

    def chat(self, prompt, images=None, vision_options=None, json_mode=True):
        if httpx is None:
            raise ModelError("httpx is not installed")
        message = {"role": "user", "content": prompt}
        payload = {
            "model": self.model_name,
            "messages": [message],
            "stream": False,
            "options": {"temperature": 0},
        }
        if images:
            payload["images"] = [image["base64"] for image in images]
        if json_mode:
            payload["format"] = "json"
        if vision_options:
            payload["options"].update({
                "num_ctx": vision_options["num_ctx"],
                "num_predict": vision_options["num_predict"],
            })
        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except (httpx.HTTPError, ValueError) as error:
            raise ModelError(f"e2b request failed: {error}") from error

    def health(self):
        if httpx is None:
            return {"status": "error", "loaded": False, "error": "httpx missing"}
        try:
            response = httpx.get(f"{self.base_url}/api/health", timeout=2.0)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as error:
            return {"status": "error", "loaded": False, "error": str(error)}

    def embed_text(self, text):
        raise NotImplementedError("E2BBackend does not implement embed_text; use OllamaBackend")

    # 以下 6 个方法复用 OllamaBackend 里同名方法的 prompt 构造逻辑
    # 只是 chat 的 endpoint 不同。用 "组合" 而不是 "继承" 保持类型清晰。
    def _prompt_helper(self):
        # 内部借用 OllamaBackend 的静态 prompt 构造逻辑,不初始化 HTTP
        return _OllamaPromptBuilder()

    def analyze_image(self, path, metadata=None):
        helper = self._prompt_helper()
        prompt = helper.build_analyze_image_prompt(metadata)
        encoded, mime_type = helper.encode_core_image(path)
        parsed = parse_json_response(self.chat(prompt, [{"base64": encoded, "mime_type": mime_type}], helper.core_vision_options()))
        return helper.postprocess_analyze_image(parsed, path, self.chat)

    def analyze_image_focus(self, path, dimension, metadata=None):
        helper = self._prompt_helper()
        prompt = helper.build_focus_prompt(dimension, metadata)
        encoded, mime_type = helper.encode_core_image(path)
        return parse_json_response(self.chat(prompt, [{"base64": encoded, "mime_type": mime_type}], helper.core_vision_options()))

    def analyze_person_appearance(self, path, metadata=None):
        helper = self._prompt_helper()
        prompt = helper.build_person_prompt(metadata)
        encoded, mime_type = helper.encode_core_image(path)
        return parse_json_response(self.chat(prompt, [{"base64": encoded, "mime_type": mime_type}], helper.core_vision_options()))

    def analyze_text(self, text, source_type="text"):
        helper = self._prompt_helper()
        prompt = helper.build_analyze_text_prompt(text, source_type)
        return parse_json_response(self.chat(prompt))

    def summarize_event(self, event, observations):
        helper = self._prompt_helper()
        prompt = helper.build_summarize_event_prompt(event, observations)
        return parse_json_response(self.chat(prompt))

    def answer(self, query, context):
        helper = self._prompt_helper()
        prompt = helper.build_answer_prompt(query, context)
        parsed = parse_json_response(self.chat(prompt))
        parsed.setdefault("model", self.model_name)
        return parsed
```

**Note**:上面用到的 `_OllamaPromptBuilder` 是把 `OllamaBackend` 里现有 prompt 构造逻辑抽出的辅助类。**这次 refactor 的范围要小**——把 OllamaBackend 现有的 `analyze_image`/`analyze_image_focus`/`analyze_person_appearance`/`analyze_text`/`summarize_event`/`answer` 里的**纯 prompt 构造 + response 后处理**部分抽成 `_OllamaPromptBuilder`(仍在同一文件),然后 `OllamaBackend` 和 `E2BBackend` 都用它。

如果 refactor 面太大,**降级方案**:E2BBackend 的 6 个方法直接**复制** OllamaBackend 里对应方法的实现体,唯一区别是把 `self.chat(...)` 保持不变(因为 chat 已经指向 E2B endpoint)。牺牲 DRY 换回归安全,后续可清理。

**推荐先用降级方案**(6 个方法复制),让 Task 5 快速收敛;抽 `_OllamaPromptBuilder` 单独安排在未来的 tech-debt cleanup。

具体做法:把 `OllamaBackend` 里 `analyze_image` / `analyze_image_focus` / `analyze_person_appearance` / `analyze_text` / `summarize_event` / `answer` 的**方法体整段复制**到 `E2BBackend` 的对应方法里(除 `answer` 里 `.model` 引用改成 `self.model_name`)。**保留 `embed_text` 的 NotImplementedError**。

- [ ] **Step 5.5: 实现 GammaClient facade 分发**

替换 `GammaClient` 类为:

```python
import time  # 顶部 import 里加上

class GammaClient:
    """Facade routing to active VLM backend based on runtime_settings.vlm_backend."""

    _CACHE_TTL_SECONDS = 5.0

    def __init__(self, base_url=None, model=None, timeout=None, keep_alive=None):
        self._ollama = OllamaBackend(base_url, model, timeout, keep_alive)
        self._e2b = E2BBackend(os.getenv("E2B_BASE_URL"), timeout)
        self._store = None
        self._active_cache = None
        self._cache_ts = 0.0

    def bind_store(self, store):
        self._store = store

    def invalidate_backend_cache(self):
        self._active_cache = None
        self._cache_ts = 0.0

    def _read_active_name(self):
        if self._store is None:
            return os.getenv("SENTRIX_VLM_BACKEND", "ollama_12b")
        return self._store.get_setting("vlm_backend", "ollama_12b")

    def _active(self):
        now = time.monotonic()
        if self._active_cache is not None and (now - self._cache_ts) < self._CACHE_TTL_SECONDS:
            return self._active_cache
        name = self._read_active_name()
        backend = self._e2b if name == "e2b_lora" else self._ollama
        self._active_cache = backend
        self._cache_ts = now
        return backend

    @property
    def base_url(self):     return self._active().endpoint
    @property
    def model(self):        return self._active().model_name
    @property
    def active_name(self):  return self._active().name

    def chat(self, *args, **kwargs):                     return self._active().chat(*args, **kwargs)
    def analyze_image(self, *args, **kwargs):            return self._active().analyze_image(*args, **kwargs)
    def analyze_image_focus(self, *args, **kwargs):      return self._active().analyze_image_focus(*args, **kwargs)
    def analyze_person_appearance(self, *args, **kwargs):return self._active().analyze_person_appearance(*args, **kwargs)
    def analyze_text(self, *args, **kwargs):             return self._active().analyze_text(*args, **kwargs)
    def summarize_event(self, *args, **kwargs):          return self._active().summarize_event(*args, **kwargs)
    def answer(self, *args, **kwargs):                   return self._active().answer(*args, **kwargs)

    def embed_text(self, text):
        # 硬钉 Ollama 12B,protect memory_vectors alignment.
        # 未来若接入独立 embedder,替换此处委托目标。
        return self._ollama.embed_text(text)
```

- [ ] **Step 5.6: 运行测试确认通过**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest backend.tests.test_e2b_backend backend.tests.test_vlm_router backend.tests.test_model_clients -v 2>&1 | tail -6
"
```
Expected: 3 个测试文件全部 PASS,不出现 FAILED / ERROR。

- [ ] **Step 5.7: 全量 backend 测试**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest discover -s backend/tests -v 2>&1 | tail -3
"
```
Expected: `OK`。

- [ ] **Step 5.8: Commit**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  git add backend/model_clients.py backend/tests/test_e2b_backend.py backend/tests/test_vlm_router.py &&
  git commit -m 'feat(model_clients): wire E2BBackend to :8100 and add GammaClient routing

- E2BBackend.chat/analyze_*/summarize_event/answer talk to :8100/api/chat
- E2BBackend.health probes :8100/api/health with 2s timeout
- E2BBackend.embed_text intentionally raises NotImplementedError
- GammaClient facade reads runtime_settings.vlm_backend (5s TTL cache)
- embed_text hardcoded to OllamaBackend for vector store alignment
- bind_store(store) injects MemoryStore; SENTRIX_VLM_BACKEND env fallback'
"
```

---

## Task 6: `/api/vlm-backend` GET/POST + `gamma.bind_store` 注入

**Files:**
- Modify: `backend/app.py`
- Create: `backend/tests/test_vlm_backend_api.py`

- [ ] **Step 6.1: 写失败测试**

Create `backend/tests/test_vlm_backend_api.py`:
```python
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


class VlmBackendApiTests(unittest.TestCase):
    def setUp(self):
        # Patch health checks before importing app
        self._ollama_ok = patch("backend.app._check_ollama_health", return_value=True).start()
        self._e2b_ok = patch("backend.app._check_e2b_health", return_value=True).start()
        self._scheduler = patch("backend.app._schedule_backend_transition", return_value=None).start()
        from backend import app as app_module
        self.app_module = app_module
        self.client = TestClient(app_module.app)

    def tearDown(self):
        patch.stopall()

    def test_get_returns_active_and_available_list(self):
        response = self.client.get("/api/vlm-backend")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("active", payload)
        self.assertIn(payload["active"], ("ollama_12b", "e2b_lora"))
        ids = {item["id"] for item in payload["available"]}
        self.assertEqual(ids, {"ollama_12b", "e2b_lora"})
        for item in payload["available"]:
            self.assertIn("healthy", item)

    def test_post_rejects_invalid_backend(self):
        response = self.client.post("/api/vlm-backend", json={"backend": "wrong_id"})
        self.assertEqual(response.status_code, 422)  # pydantic Literal rejection

    def test_post_returns_503_when_target_unhealthy(self):
        self._e2b_ok.return_value = False
        response = self.client.post("/api/vlm-backend", json={"backend": "e2b_lora"})
        self.assertEqual(response.status_code, 503)

    def test_post_persists_and_invalidates_cache(self):
        response = self.client.post("/api/vlm-backend", json={"backend": "e2b_lora"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["active"], "e2b_lora")
        # SQLite should reflect the new value
        current = self.app_module.store.get_setting("vlm_backend")
        self.assertEqual(current, "e2b_lora")

    def test_health_payload_uses_vlm_key(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("vlm", payload["models"])
        self.assertIn("active", payload["models"]["vlm"])
        self.assertNotIn("gamma4_12B", payload["models"])
```

- [ ] **Step 6.2: 运行确认失败**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest backend.tests.test_vlm_backend_api -v 2>&1 | tail -10
"
```
Expected: 大量失败,包括 `AttributeError: no _check_ollama_health` 或 endpoint 404。

- [ ] **Step 6.3: 修改 `backend/app.py`**

在文件顶部 import 区域加(如果没有):
```python
import asyncio
import httpx
from pydantic import BaseModel
from typing import Literal
```

紧接 `gamma = GammaClient()` 那行**之后**加:
```python
gamma.bind_store(store)
```
(如果 `store` 定义在 `gamma` 之后,把 `gamma.bind_store(store)` 移到 `store = MemoryStore(...)` 之后。)

在 `/api/health` 定义**之前**加:

```python
VLM_BACKENDS = ("ollama_12b", "e2b_lora")


def _check_ollama_health(hard: bool = False) -> bool:
    url = f"{os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')}/api/tags"
    timeout = 5.0 if hard else 2.0
    try:
        response = httpx.get(url, timeout=timeout)
        return response.status_code < 400
    except httpx.HTTPError:
        return False


def _check_e2b_health(hard: bool = False) -> bool:
    url = f"{os.getenv('E2B_BASE_URL', 'http://127.0.0.1:8100').rstrip('/')}/api/health"
    timeout = 5.0 if hard else 2.0
    try:
        response = httpx.get(url, timeout=timeout)
        if response.status_code >= 400:
            return False
        data = response.json()
        return bool(data.get("status") == "ok")
    except (httpx.HTTPError, ValueError):
        return False


async def _fire_and_forget_post(url: str, body: dict | None = None, timeout: float = 10.0):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            await client.post(url, json=body or {})
    except Exception:
        pass


def _schedule_backend_transition(old: str, new: str) -> None:
    loop = asyncio.get_event_loop()
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    e2b_url = os.getenv("E2B_BASE_URL", "http://127.0.0.1:8100").rstrip("/")

    if old == "ollama_12b":
        loop.create_task(_fire_and_forget_post(
            f"{ollama_url}/api/generate",
            {"model": os.getenv("OLLAMA_MODEL", "gemma4:12b"), "prompt": "", "keep_alive": 0},
        ))
    elif old == "e2b_lora":
        loop.create_task(_fire_and_forget_post(f"{e2b_url}/admin/unload"))

    if new == "e2b_lora":
        loop.create_task(_fire_and_forget_post(f"{e2b_url}/admin/load"))
    elif new == "ollama_12b":
        loop.create_task(_fire_and_forget_post(
            f"{ollama_url}/api/generate",
            {"model": os.getenv("OLLAMA_MODEL", "gemma4:12b"), "prompt": ".", "keep_alive": -1},
        ))


class SetVLMBackend(BaseModel):
    backend: Literal["ollama_12b", "e2b_lora"]


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


@app.post("/api/vlm-backend")
def set_vlm_backend(payload: SetVLMBackend):
    if payload.backend == "e2b_lora" and not _check_e2b_health(hard=True):
        raise HTTPException(503, "e2b service :8100 unreachable")
    if payload.backend == "ollama_12b" and not _check_ollama_health(hard=True):
        raise HTTPException(503, "ollama service :11435 unreachable")

    previous = store.get_setting("vlm_backend", "ollama_12b")
    store.set_setting("vlm_backend", payload.backend)
    gamma.invalidate_backend_cache()
    _schedule_backend_transition(previous, payload.backend)
    return {"active": payload.backend, "previous": previous}
```

改 `/api/health` 里 `models` 部分。**把**:
```python
"gamma4_12B": {"name": gamma.model, "endpoint": gamma.base_url},
```
**替换为**:
```python
"vlm": {
    "active": gamma.active_name,
    "name": gamma.model,
    "endpoint": gamma.base_url,
    "healthy": (
        _check_e2b_health() if gamma.active_name == "e2b_lora" else _check_ollama_health()
    ),
},
```

- [ ] **Step 6.4: 运行测试确认通过**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest backend.tests.test_vlm_backend_api -v 2>&1 | tail -10
"
```
Expected: 5 个测试全 PASS。

- [ ] **Step 6.5: 全量 backend 测试**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  source /home/asus/miniconda3/etc/profile.d/conda.sh &&
  conda activate stmem &&
  python -m unittest discover -s backend/tests -v 2>&1 | tail -3
"
```
Expected: `OK`。若现有测试因 `gamma4_12B` → `vlm` 改动挂了,同步修 fixture(通常在 `test_app_health*.py` 类文件里,grep 定位)。

- [ ] **Step 6.6: Commit**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  git add backend/app.py backend/tests/test_vlm_backend_api.py &&
  git commit -m 'feat(api): add /api/vlm-backend GET/POST and migrate health.models to vlm

- GET returns active + available list with per-backend healthy flag
- POST validates via pydantic Literal, 503 on target unhealthy
- SQLite update + gamma.invalidate_backend_cache() on success
- _schedule_backend_transition unloads old / warms new async
- /api/health.models.gamma4_12B renamed to .models.vlm with active field
- gamma.bind_store(store) injects MemoryStore for runtime_settings reads'
"
```

---

## Task 7: 前端 settings 页切换器 + api.js + overview key 迁移

前后端 key 迁移(`gamma4_12B` → `vlm`)在这一步合并落地,与 Task 6 后端 key 迁移**保持一次同步部署**。

**Files:**
- Modify: `src/app.js`
- Modify: `src/api.js`

- [ ] **Step 7.1: 加 api.js 方法**

在 `src/api.js` 里找到导出 `api` 对象的位置,追加两个方法:
```javascript
export const api = {
  // ...现有方法...
  getVlmBackend: () => fetch('/api/vlm-backend').then(async r => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }),
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
};
```

**Note**: 具体导出风格(是否是 `export const api = {}` 还是别的),看现有 `src/api.js` 结构;保持一致。

- [ ] **Step 7.2: 改 settings 页 AI MODEL ROUTER 卡片**

在 `src/app.js` 里 `settingsView()` 函数,定位到含 `AI MODEL ROUTER` 的字符串块(大概 line 341)。原代码是:
```javascript
<article class="health-card"><div class="health-title"><span>AI MODEL ROUTER</span><span class="ready-label">READY</span></div><div class="model-row"><span>主推理</span><strong>${escapeHtml(state.health.models?.gamma4_12B?.name || "未知")}</strong><small>${escapeHtml(state.health.models?.gamma4_12B?.endpoint || "未连接")}</small></div>
```

**替换为**(注意 template literal 嵌入 select 选项):
```javascript
<article class="health-card"><div class="health-title"><span>AI MODEL ROUTER</span><span class="ready-label ${state.health.models?.vlm?.healthy === false ? 'warn' : ''}">${state.health.models?.vlm?.healthy === false ? 'OFFLINE' : 'READY'}</span></div><label class="model-switcher"><span>主推理</span><select id="vlm-backend-select" data-action="switch-vlm">${(state.vlmBackendOptions || []).map(item => `<option value="${item.id}" ${item.id === state.health.models?.vlm?.active ? 'selected' : ''} ${item.healthy ? '' : 'disabled'}>${escapeHtml(item.label)}${item.healthy ? '' : ' · 离线'}</option>`).join('')}</select><small>${escapeHtml(state.health.models?.vlm?.endpoint || '未连接')}</small></label>
```

- [ ] **Step 7.3: overview 页面 key 迁移**

`src/app.js` 里搜 `gamma4_12B` — 应该只有 overview 页里另一处引用(约 line 181):
```javascript
<small>${escapeHtml(state.health?.models?.gamma4_12B?.name || "等待服务")}</small>
```
**替换为**:
```javascript
<small>${escapeHtml(state.health?.models?.vlm?.name || "等待服务")}</small>
```

- [ ] **Step 7.4: 加载 vlm-backend options + 切换 handler**

找到项目里现有的 state 初始化 + 事件绑定入口。典型位置在 `src/app.js` 顶部附近或 `boot()` / `render()` 函数。

在 state 初始化附近(接近 `state.health` 声明的地方)加入:
```javascript
state.vlmBackendOptions = [];
```

在 `refreshHealth()` 或 `boot()` 里加入(具体函数名看现有代码):
```javascript
async function refreshVlmBackendOptions() {
  try {
    const data = await api.getVlmBackend();
    state.vlmBackendOptions = data.available || [];
  } catch (err) {
    console.warn('failed to load vlm backend options', err);
    state.vlmBackendOptions = [];
  }
}
```
并在打开 settings 页时调用 `refreshVlmBackendOptions()`;`refreshHealth()` 完成后也调用一次。

在现有事件委托(找 `document.addEventListener('click', ...)` 或类似)之外,追加 `change` 委托:
```javascript
document.addEventListener('change', async (event) => {
  const select = event.target.closest('[data-action="switch-vlm"]');
  if (!select) return;
  const target = select.value;
  const previous = state.health?.models?.vlm?.active;
  select.disabled = true;
  const targetLabel = select.selectedOptions[0]?.textContent || target;
  toast(`切换到 ${targetLabel}...`, 'info');
  try {
    await api.setVlmBackend(target);
    toast(`已切换到 ${targetLabel}`, 'success');
    await refreshHealth();
    await refreshVlmBackendOptions();
    render();
  } catch (err) {
    select.value = previous;
    toast(`切换失败: ${err.message}`, 'error');
  } finally {
    select.disabled = false;
  }
});
```
**Note**: `toast`、`refreshHealth`、`render` 都是项目已有函数;若命名不同,以现有为准。

- [ ] **Step 7.5: Node test 全量**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  node --check src/app.js &&
  node --check src/api.js &&
  node --test test/*.test.js 2>&1 | tail -10
"
```
Expected: 语法检查通过;所有测试 PASS(现有 27 个不动)。

- [ ] **Step 7.6: Commit**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  git add src/app.js src/api.js &&
  git commit -m 'feat(web): add vlm backend switcher and migrate to health.models.vlm

- settings AI MODEL ROUTER card gains <select> with per-backend health
- api.js exposes getVlmBackend / setVlmBackend
- change handler posts to /api/vlm-backend with toast + rollback on failure
- overview and settings both read state.health.models.vlm (was gamma4_12B)
- refreshVlmBackendOptions populates state.vlmBackendOptions'
"
```

---

## Task 8: 启动脚本 + `start_sentrix_api.sh` env + project-structure 断言

**Files:**
- Create: `scripts/runtime/start_sentrix_e2b.sh`
- Modify: `scripts/runtime/start_sentrix_api.sh`
- Modify: `test/project-structure.test.js`

- [ ] **Step 8.1: 写测试断言**

修改 `test/project-structure.test.js`,在现有 `assert.match(apiStartScript, /OLLAMA_BASE_URL/, ...)` 附近追加:
```javascript
assert.match(apiStartScript, /E2B_BASE_URL/, "API startup must expose E2B_BASE_URL for GammaClient facade");
```

在现有断言 `["scripts", "runtime", "start_sentrix_ollama.sh"]` 项目结构检查旁,追加:
```javascript
const e2bStartScript = fs.readFileSync(
  path.join(root, "scripts", "runtime", "start_sentrix_e2b.sh"),
  "utf8"
);
assert.match(e2bStartScript, /uvicorn services\.e2b_server\.app:app/, "e2b startup must launch uvicorn app");
assert.match(e2bStartScript, /E2B_BASE_MODEL/, "e2b startup must reference E2B_BASE_MODEL env");
assert.match(e2bStartScript, /E2B_ADAPTER/, "e2b startup must reference E2B_ADAPTER env");
```
(具体 fs/path 用法看现有测试文件风格。)

- [ ] **Step 8.2: 运行确认失败**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  node --test test/project-structure.test.js 2>&1 | tail -10
"
```
Expected: `ENOENT: start_sentrix_e2b.sh` 或 `assert.match failed`。

- [ ] **Step 8.3: 创建 `start_sentrix_e2b.sh`**

Create `scripts/runtime/start_sentrix_e2b.sh`:
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
export E2B_DTYPE="${E2B_DTYPE:-bf16}"

# 前置检查关键文件
for f in "$E2B_BASE_MODEL/config.json" "$E2B_BASE_MODEL/processor_config.json" \
         "$E2B_BASE_MODEL/tokenizer.json" "$E2B_BASE_MODEL/model.safetensors" \
         "$E2B_ADAPTER/adapter_config.json" "$E2B_ADAPTER/adapter_model.safetensors"; do
  if [ ! -f "$f" ]; then
    echo "[start_sentrix_e2b] missing required file: $f" >&2
    exit 1
  fi
done

exec uvicorn services.e2b_server.app:app --host "$E2B_HOST" --port "$E2B_PORT" --workers 1
```
设权限:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "chmod +x /home/asus/Github/Sentrix-Home-Web/scripts/runtime/start_sentrix_e2b.sh"
```

- [ ] **Step 8.4: 修改 `start_sentrix_api.sh`**

在现有 `export OLLAMA_BASE_URL=...` / `export OLLAMA_MODEL=...` 附近追加一行:
```bash
export E2B_BASE_URL="${E2B_BASE_URL:-http://127.0.0.1:8100}"
```

- [ ] **Step 8.5: 运行测试确认通过**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  node --test test/project-structure.test.js 2>&1 | tail -5 &&
  echo '---' &&
  bash -n scripts/runtime/start_sentrix_e2b.sh &&
  bash -n scripts/runtime/start_sentrix_api.sh &&
  echo 'shell syntax OK'
"
```
Expected: `# tests X # pass X # fail 0`;`shell syntax OK`。

- [ ] **Step 8.6: Commit**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  git add scripts/runtime/start_sentrix_e2b.sh scripts/runtime/start_sentrix_api.sh test/project-structure.test.js &&
  git commit -m 'feat(runtime): add start_sentrix_e2b.sh and E2B_BASE_URL env

- e2b startup activates sentrix-e2b conda env, checks required model files
- start_sentrix_api.sh exports E2B_BASE_URL for GammaClient facade
- project-structure.test.js asserts new script and env presence'
"
```

---

## Task 9: Operator runbook

**Files:**
- Create: `docs/runbooks/vlm-backend-switch.md`

- [ ] **Step 9.1: 写 runbook**

Create `docs/runbooks/vlm-backend-switch.md`:
```markdown
# VLM Backend 切换 Runbook

Sentrix Home 支持在两个多模态大模型 backend 之间切换:
- **`ollama_12b`** — 默认。Ollama 服务(:11435)承载 `gemma4:12b`(GGUF Q4_K_M)。
- **`e2b_lora`** — 新加。独立 FastAPI 服务(:8100)承载 Gemma-4-E2B-it + LoRA V2(BF16)。

`embed_text` 无论选哪个 backend 都走 Ollama 12B,保护 `memory_vectors` 表 3840 维向量的空间一致性。

## 环境要求

- 153 主机, RTX 3090, CUDA 12.6
- `sentrix-e2b` conda env: torch 2.8.0+cu126, transformers 5.13.x, peft 0.19.1
- 磁盘: `/home/asus/models/gemma-4-E2B-it`(9.7 GB), `/home/asus/models/gemma-4-e2b-lora-v2`(92 MB)

## 启动顺序(推荐)

```bash
# 1. 启动 Ollama(常年在跑,一般不用启)
./scripts/runtime/start_sentrix_ollama.sh

# 2. 启动 E2B 服务(:8100)
nohup ./scripts/runtime/start_sentrix_e2b.sh > logs/e2b-server.log 2>&1 &
sleep 5
curl -sf http://127.0.0.1:8100/api/health | python3 -m json.tool

# 3. 启动主 API(:8091)
./scripts/runtime/start_sentrix_api.sh
```

`start_sentrix_e2b.sh` 会前置检查基模和 adapter 关键文件;缺则拒绝启动。

## 前端切换

打开 Web UI `http://192.168.0.153:4174`(或对应端口),进入"设备与隐私"页,在 **AI MODEL ROUTER** 卡片里改 select:
- `Gemma-4 12B (Ollama)` — 默认,批量导入推荐
- `Gemma-4 E2B-it + LoRA V2` — 演示/单次问答,LoRA 蒸馏视觉能力

切换成功会 toast 提示;若目标 backend 离线,POST 返回 503,select 自动回退。

## 命令行切换(应急)

```bash
# 查看当前
curl -s http://127.0.0.1:8091/api/vlm-backend | python3 -m json.tool

# 切到 E2B(会先探活,不健康则 503)
curl -s -X POST http://127.0.0.1:8091/api/vlm-backend \
  -H 'Content-Type: application/json' -d '{"backend":"e2b_lora"}'

# 切回 12B
curl -s -X POST http://127.0.0.1:8091/api/vlm-backend \
  -H 'Content-Type: application/json' -d '{"backend":"ollama_12b"}'
```

## 硬回滚(绕过 API)

如果 API 挂了但要立刻改选,直接改 SQLite:
```bash
sqlite3 /home/asus/Github/Sentrix-Home-Web/data/sentrix.db \
  "UPDATE runtime_settings SET value='ollama_12b' WHERE key='vlm_backend';"
```
`GammaClient` 5 秒内看到新值。

## 显存管理

单卡 24 GB。切换时:
- 旧 backend 会异步收到 unload 请求(Ollama `keep_alive=0` 空 request;E2B `/admin/unload`)
- 新 backend 会异步收到 warm-up 请求(触发懒加载)
- 加载 E2B 需要 10-30 s,期间 `/api/health.vlm.healthy=false`

如果同时驻留导致 OOM:
```bash
# 强制 Ollama 释放
curl -s -X POST http://127.0.0.1:11435/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4:12b","prompt":"","keep_alive":0}'

# 强制 E2B 释放
curl -s -X POST http://127.0.0.1:8100/admin/unload
```

## 健康检查

- Ollama: `curl http://127.0.0.1:11435/api/tags`
- E2B:    `curl http://127.0.0.1:8100/api/health`
- API 总:`curl http://127.0.0.1:8091/api/health`(payload 里 `models.vlm.active` 显示当前 backend)

## 常见问题

**Q: 切到 E2B 后导入图片变慢?**
A: E2B 单请求串行,`asyncio.Lock` 一次只处理一个请求。批量导入请切回 Ollama 12B。

**Q: E2B 返回的 JSON 解析失败?**
A: E2B 没有 grammar decoder。已在 prompt 追加 hint,并由 `parse_json_response()` 兜底。如果失败率高,考虑接 `outlines`。

**Q: 切换后 Ollama 又自动加载了?**
A: `embed_text` 硬钉 Ollama;用户问答里触发 embedding 会让 Ollama 短暂唤醒。这是设计,不是 bug。

**Q: 我改了 `runtime_settings` 但 API 没反应?**
A: GammaClient 有 5 秒 TTL 缓存。等 5 秒或重启 API 生效。经 `/api/vlm-backend` 走的话会立刻 invalidate。
```

- [ ] **Step 9.2: Commit**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  git add docs/runbooks/vlm-backend-switch.md &&
  git commit -m 'docs(runbooks): vlm backend switch operator guide

- startup order, frontend + CLI switching, hard rollback via sqlite
- gpu residency and oom mitigation
- health checks and common questions'
"
```

---

## Task 10: 端到端手动验收清单

**Files:** 无(纯操作)

- [ ] **Step 10.1: 全部服务启起来**

在 153 上分别起 Ollama、E2B、API:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  cd /home/asus/Github/Sentrix-Home-Web &&
  # 确认三个服务健康
  curl -sf http://127.0.0.1:11435/api/tags > /dev/null && echo 'ollama OK' || echo 'ollama DOWN'
  curl -sf http://127.0.0.1:8100/api/health > /dev/null && echo 'e2b OK' || echo 'e2b DOWN'
  curl -sf http://127.0.0.1:8091/api/health > /dev/null && echo 'api OK' || echo 'api DOWN'
"
```
Expected: 三个 OK。

- [ ] **Step 10.2: 验证 /api/vlm-backend GET**

```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "
  curl -s http://127.0.0.1:8091/api/vlm-backend | python3 -m json.tool
"
```
Expected: `active: ollama_12b`,`available` 里两项都 `healthy: true`。

- [ ] **Step 10.3: 浏览器打开 UI 看切换器**

用户手动:打开 `http://192.168.0.153:4174`,进入"设备与隐私"页,确认 AI MODEL ROUTER 卡片里出现 `<select>`,当前选中 `Gemma-4 12B (Ollama)`,两项都可选。

- [ ] **Step 10.4: 切到 E2B**

在 select 里选 `Gemma-4 E2B-it + LoRA V2`。Expected:
- toast 提示 "切换到 Gemma-4 E2B-it + LoRA V2..."
- 30 s 内 toast 变 "已切换到..."
- badge 从 READY 保持或短暂 OFFLINE 后变 READY
- 后台 `/api/health` payload 中 `models.vlm.active == "e2b_lora"`

- [ ] **Step 10.5: 导入一张图验证 E2B 参与 pipeline**

用户手动:在 Web UI 里导入 album1 里没导过的图片。Expected:
- observations 表新增记录
- SQLite 查询:
  ```bash
  sshpass -p 'Abc123' ssh asus@192.168.0.153 "
    sqlite3 /home/asus/Github/Sentrix-Home-Web/data/sentrix.db \
      \"SELECT json_extract(raw_analysis, '\$.gamma.models.vision') FROM observations ORDER BY created_at DESC LIMIT 1\"
  "
  ```
  返回值包含 `gemma-4-e2b-it+lora-v2`(而非 `gemma4:12b`)。

- [ ] **Step 10.6: Agent 问答走 E2B**

在 Web UI 里问一个问题(如"最近拍了什么?"),看响应。Expected:
- 有回答返回
- 后端日志(`data/sentrix-api.log`)显示对 :8100 的调用

- [ ] **Step 10.7: 切回 12B**

Expected:
- toast 成功
- 后续导图 raw.gamma.models.vision 恢复为 `gemma4:12b`

- [ ] **Step 10.8: 故障切换测试**

关掉 :8100 服务:
```bash
sshpass -p 'Abc123' ssh asus@192.168.0.153 "pkill -f 'uvicorn services.e2b_server.app'"
sleep 2
# 探活应显示 offline
sshpass -p 'Abc123' ssh asus@192.168.0.153 "curl -s http://127.0.0.1:8091/api/vlm-backend | python3 -m json.tool | grep -A1 e2b_lora"
```
Expected: `healthy: false`。

在 UI 里尝试切到 E2B。Expected:
- 前端 select 里 `e2b_lora` 显示为 disabled(标记 "离线")或提交后 503 toast 回退

- [ ] **Step 10.9: 记录验收结果**

在 spec 或单独的 verification.md 里记录:
- 每一步的实际结果、耗时、GPU 峰值
- 遗留问题(若有)
- 备份点:切换前/后 sentrix.db 的备份路径

## Self-Review

**1. Spec coverage:**

| Spec 章节 | Plan Task |
|---|---|
| §4 总体架构 | Task 1-8 集成落地 |
| §5.1 E2B 独立服务 | Task 4a-d |
| §5.2 GammaClient facade | Task 2, 5 |
| §5.3 数据模型 | Task 1 |
| §5.4 后端 API endpoints | Task 6 |
| §5.5 前端 UI | Task 7 |
| §6 错误处理 | 6.1/6.2 → Task 6 探活;6.3 → Task 4b/c load_error;6.4 → Task 5 embed_text 硬钉;6.5 → Task 4a JSON hint;6.6 → Task 4b asyncio.Lock;6.7 → Task 2/3 保持测试 |
| §7.1 一次性部署 | Task 0 |
| §7.2 提交顺序 | Task 1-9,已按 11 步展开 |
| §7.3 测试策略 | 每个 Task 自带 TDD 步骤 |
| §7.4 迁移与回滚 | Task 9 runbook |
| §9 验收标准 | Task 10 |

无未覆盖章节。

**2. Placeholder scan:** 无 TBD / TODO / "similar to Task N" / "add appropriate error handling"。所有代码步骤都给了完整代码。有一处 `_OllamaPromptBuilder` 抽象在 Task 5.4 里明确降级到"复制 6 个方法体"以避免范围蔓延,不是 placeholder。

**3. Type consistency:**
- `E2BBackend.name` = `"e2b_lora"`,`OllamaBackend.name` = `"ollama_12b"` — 与 SQLite 存储值、Pydantic Literal、前端 select value 一致。
- `E2BBackend.model_name` = `"gemma-4-e2b-it+lora-v2"` — Task 5 backend,Task 6 GET 响应,Task 10 观察库字段全一致。
- `runtime_settings.key` = `"vlm_backend"` — 贯穿 Task 1、5、6、9、10。
- `/api/vlm-backend` endpoint 路径 — Task 6 定义、Task 7 前端使用、Task 9 runbook、Task 10 验证一致。
- `health.models.vlm` — Task 6 后端 rename,Task 7 前端 rename,同批部署。

**4. Ambiguity:**
- Task 7.4 里 `refreshHealth` / `render` / `toast` 是项目现有函数,写了"以现有为准"。实施时若名字不同,由执行者调整。
- Task 5.4 里 `_OllamaPromptBuilder` 明确降级方案(复制方法体)不做,避免 refactor 范围爆炸。

---

## 执行交接

Plan 完整,保存到 `docs/superpowers/plans/2026-08-05-gemma4-e2b-integration.md`。两个执行选项:

**1. Subagent-Driven(推荐)** — 每个 Task 派一个 fresh subagent,主对话审查 Task 间产出。适合这种跨 backend/frontend/runtime 的中等复杂度任务:每个 Task 相对独立可评估,失败可局部回退,主对话上下文不被大量 diff 淹没。

**2. Inline Execution** — 在本会话内批量执行 Task,checkpoint 处停下让你 review。适合快节奏推进,但会话上下文会因大量文件读写和测试输出膨胀。

