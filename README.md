# Sentrix Home Web

Sentrix Home 是本地优先的家庭多模态数字记忆系统网页端。当前版本独立维护自己的 `Asset -> Observation -> Event -> Fact` 数据链路：图片和音频进入共享事件/语义记忆，视频只登记原始资产并保留 `video_memory_adapter` 接口。

## 当前能力

- 图片：Ollama `gemma4:12b` 视觉观察抽取，InsightFace 人脸候选，人、地点、活动、物体和事件落库。
- 音频：Sentrix 自己的 FunASR 适配器组合 Paraformer、FSMN-VAD 和 CT-Punc 保存转写，再由 Gemma 归纳事件和事实。
- 语义记忆：事实有 `active`、`pending`、`superseded`、`retracted` 状态，冲突事实不会静默覆盖，可以在网页确认或驳回。
- Agent：本地 SQLite 事件/观察/实体/事实检索，结合 Sentrix 原生向量索引，再由 Gemma 返回答案、置信度、证据 ID 和检索轨迹。
- 视频：保存原始 Asset，返回 `video-extraction-reserved`，不解码、不切片、不生成视频编码记忆。

## 153 启动

```bash
cd /home/asus/Github/Sentrix-Home-Web
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
mkdir -p data/face-models/models

SENTRIX_DATA_DIR=$PWD/data \
FACE_MODEL_ROOT=$PWD/data/face-models \
OLLAMA_BASE_URL=http://127.0.0.1:11434 \
OLLAMA_MODEL=gemma4:12b \
FUNASR_MODEL=paraformer-zh \
FUNASR_VAD_MODEL=fsmn-vad \
FUNASR_PUNC_MODEL=ct-punc \
CLIP_MODEL_NAME=ViT-B-32 \
CLIP_CHECKPOINT=$PWD/data/models/clip/ViT-B-32.bin \
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8090
```

另开终端启动网页：

```bash
cd /home/asus/Github/Sentrix-Home-Web
SENTRIX_BACKEND_URL=http://127.0.0.1:8090 PORT=4174 npm run dev
```

浏览器访问 `http://192.168.0.153:4174`。

## API

- `GET /api/health`
- `GET /api/dashboard`
- `GET /api/events`
- `GET /api/events/{id}`：事件与 Observation、Asset、Fact 原始证据
- `POST /api/events`、`PATCH /api/events/{id}`：人工事件创建和修订
- `GET /api/persons`、`GET /api/entities`、`GET /api/face-clusters`
- `POST /api/persons/{id}/confirm`、`POST /api/persons/{id}/reject`
- `POST /api/face-clusters/{id}/confirm`、`POST /api/face-clusters/{id}/reject`
- `GET /api/observations`
- `GET /api/observations/{id}`
- `GET /api/assets`、`GET /api/assets/{id}`、`GET /api/assets/{id}/file`
- `POST /api/ingest`：multipart 字段 `file`
- `POST /api/search`：`{"query":"图片里有冰箱吗？"}`
- `GET /api/stories`、`POST /api/stories`、`PATCH /api/stories/{id}`
- `POST /api/invites`：生成局域网邀请 token
- `POST /api/maintenance/recheck`
- `POST /api/facts/{id}/confirm`
- `POST /api/facts/{id}/reject`

Node 网页服务只代理同源 `/api`，模型和原生记忆存储地址不会暴露给浏览器。

## 测试

```bash
npm test
PYTHONPATH=. python3 -m unittest discover -s backend/tests -v
```

测试图片、数据库和运行日志都在 `data/`、`logs/` 中，已被 Git 忽略。模型、密钥和本地路径通过环境变量配置。
