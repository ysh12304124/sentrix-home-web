# 语义实体层与语义目录界面实现计划

## 目标

在 Sentrix `psh` 分支中实现已批准的语义实体层和语义优先网页目录：地点、物品和氛围使用受控主类与细节词表；地点主名称来自图像语义而不是 GPS；语义组只作为非破坏性浏览投影；网页默认只展示语义段和原始证据缩略图。Agent 规划、检索和回答逻辑不在本计划范围内。

## 架构

```text
Image Observation
  -> semantic taxonomy normalization
  -> semantic primary + detail values
  -> stable entity + evidence-backed properties
  -> primary semantic group projection
  -> semantic directory card/detail + original evidence
```

- `backend/semantic_taxonomy.py` 成为模型适配器和 MemoryStore 共用的唯一词表边界。
- `backend/model_clients.py` 负责让视觉模型在受控词表中选择，并保留自由描述为原始观察。
- `backend/pipeline.py` 负责把快速路径和延迟语义丰富路径统一写入同一规范字段。
- `backend/db.py` 负责派生实体、属性、语义组和证据聚合；GPS 仅是地点属性，不是语义组键。
- `src/app.js` 和 `src/styles.css` 负责语义优先卡片、细节筛选、证据图库和技术信息折叠入口。
- `backend/agent.py` 不修改。

## 技术栈

- Python 3、FastAPI、SQLite、现有 MemoryStore 和本地 Gemma/AdaFace/CLIP 适配器。
- Plain browser JavaScript、现有同源 API 和 CSS。
- Python `unittest`、Node `node:test`、`git diff --check`。
- 开发在本地从 153 `psh` 创建的干净副本完成；正式提交只在 153 `psh`。

## 实现步骤

### 1. 建立共享词表和规范化测试

文件：

- `backend/semantic_taxonomy.py`（新增）
- `backend/model_clients.py`
- `backend/tests/test_semantic_taxonomy.py`（新增）
- `backend/tests/test_model_clients.py`

步骤：

1. 新增地点、物品、氛围主类和细节词表，包含设计文档中的 v1 词项以及统一的 `其他或不确定`。
2. 新增 `normalize_semantic_analysis()`，将模型输出规范化为：

   ```json
   {
     "place": {"primary": "餐饮空间", "details": ["室内", "有餐桌"]},
     "objects": [{"primary": "食品与饮品", "label": "蛋糕", "details": ["摆放在桌面"]}],
     "atmosphere": {"labels": ["温馨"], "details": ["暖色光线", "多人聚集"]}
   }
   ```

3. 未知主类统一变成 `其他或不确定`；未知自由文本不进入受控实体列表，但保留在 raw/canonical 证据。
4. 保留 `scene_type`、`objects` 和 `emotions` 旧字段作为兼容读取字段，新增规范字段为主写入路径；对外把 emotion 映射为 atmosphere。
5. 先写失败测试，覆盖：合法值保留、非法主类兜底、细节去重、物品多项、氛围多标签、原始标签不丢失。

命令：

```bash
cd /path/to/local/clone
PYTHONPATH=. python3 -m unittest backend.tests.test_semantic_taxonomy backend.tests.test_model_clients
```

预期：新增测试先失败；规范化实现完成后全部通过。

### 2. 统一视觉模型输出合同

文件：

- `backend/model_clients.py`
- `backend/tests/test_model_clients.py`

步骤：

1. 将视觉 prompt 的地点主类替换为批准的 18 项地点词表。
2. 要求模型输出 `semantic.place.primary/details`、`semantic.objects` 和 `semantic.atmosphere.labels/details`。
3. 要求每张图片地点主类单选；物品记录可多项；氛围标签可多项；看不清时选择 `其他或不确定`。
4. 将模型自由 `place` 作为 `visual_place_descriptions` 候选，不让它直接成为地点实体主名称。
5. 在快速图片路径、完整视觉路径和 JSON 规范化路径调用同一规范化函数，避免两条管线产生不同结构。
6. 测试 prompt 包含词表、单选/多选约束和禁止猜测要求；测试解析结果与旧字段兼容。

命令：

```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_model_clients
python3 -m py_compile backend/model_clients.py
```

预期：模型客户端测试通过，Python 编译无输出。

### 3. 修正 Observation 和实体派生

文件：

- `backend/pipeline.py`
- `backend/db.py`
- `backend/tests/test_entities.py`
- `backend/tests/test_pipeline.py`

步骤：

1. 在 `add_observation`/`enrich_observation` 的 canonical 数据中持久化规范语义对象，并继续保存 raw 模型返回。
2. 修改 `maintain_observation_entities()`：
   - 地点实体名称始终取 `semantic.place.primary`；
   - GPS 写入地点的 `geo` 属性；
   - 自由 `place` 写入 `visual_place_descriptions`；
   - 地点细节写入 `semantic_details`；
   - 物品按主类、label 和细节写入证据属性；
   - 氛围使用 `atmosphere` 语义名称和属性，不再把它当作人物情绪事实。
3. 对既有 `emotion` 实体提供一次可审计的派生迁移到 `atmosphere` 的路径，保留原始 raw mood 标签和证据链接。
4. 删除地点实体主名称对 GPS 的优先级；GPS 只保留在详情和事件/行程定位所需的内部属性中。
5. 保持用户别名、私密地点和属性 revision 优先级，重建不得覆盖 user property。
6. 新增失败测试：GPS + `滨水空间` 的实体名称必须是 `滨水空间`；自由地点描述和坐标仍可从属性中读取；语义细节和原始标签均有 Observation evidence ID。

命令：

```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_entities backend.tests.test_pipeline
python3 -m py_compile backend/db.py backend/pipeline.py
```

预期：测试覆盖地点主名称、GPS 分离、物品/氛围属性和重建幂等性。

### 4. 改造语义组 API 和派生维护

文件：

- `backend/db.py`
- `backend/app.py`
- `backend/tests/test_entities.py`
- `backend/tests/test_app.py`（若现有测试布局需要新增）
- `scripts/maintenance/reindex_semantic_entities.py`（新增）

步骤：

1. 修改 `_semantic_entity_key()` 和 `list_semantic_entity_groups()`，使用规范主类作为地点、物品和氛围分组键，不使用 GPS 网格作为默认语义组。
2. 语义组返回聚合后的 `semantic_details`、短摘要、代表性原始证据和成员关系；不删除成员实体。
3. 保留 `/api/entity-groups` 和 `/api/entity-groups/{id}` 路由，扩充响应字段，不改变 scope 过滤。
4. 增加仅重建派生语义投影的维护命令：默认 dry-run，`--apply` 必须要求 `--backup`，并打印每个 MemorySpace 的前后计数。
5. 维护命令只更新语义字段、实体链接和语义组投影，不修改 Asset 原文件、原始 Observation raw 内容或用户属性。
6. 测试同一主类下多个具体描述的聚合、不同 MemorySpace 不互相聚合、GPS 不创建语义组名称、原始证据数量不重复膨胀。

命令：

```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_entities
PYTHONPATH=. python3 scripts/maintenance/reindex_semantic_entities.py --database /tmp/sentrix-semantic-test.db --dry-run
```

预期：测试通过；dry-run 只报告变更，不写入数据库。

### 5. 重做语义目录和证据详情

文件：

- `src/app.js`
- `src/api.js`（仅在响应字段需要显式归一化时修改）
- `src/styles.css`
- `test/project-structure.test.js`
- `test/normalizers.test.js`（若加入前端归一化）

步骤：

1. 将 `semanticKnowledgeView()` 的非人物区域改成地点、物品、氛围三组语义目录。
2. 卡片只显示原始证据缩略图、主类、细节标签和证据驱动的短语义段；移除文件名、Asset ID、Observation ID、属性键、模型名和置信度数字。
3. 将语义组详情改成“语义摘要 -> 细节筛选 -> 原始证据缩略图网格 -> 技术审计入口”的固定顺序。
4. 原始证据缩略图点击打开现有 Asset 原图；只有技术审计入口显示文件名、时间、坐标、Observation ID、原始模型 JSON 和版本。
5. 对 `assetCard`、导入队列、通用 `evidenceCard`、实体详情和人物证据页做同一信息层级审查：默认用缩略图和语义说明，不用文件名充当标题。
6. 保留屏幕阅读器可用的图片 alt 文本，但 alt 使用“餐饮空间的原始证据”等语义描述，不使用文件名。
7. 增加移动端响应式样式：证据网格固定缩略图比例，细节标签换行，长语义段不遮挡按钮。
8. Node 测试验证语义目录源代码不再把 `file_name`、内部 ID 或 confidence 作为默认卡片主文本，并保持现有交互动作和 API 调用。

命令：

```bash
node --test test/*.test.js
node --check src/app.js
node --check src/api.js
```

预期：Node 测试和 JS 语法检查通过；语义卡片仍可打开原始证据和详情。

### 6. 隔离库重建和展示验收

文件：

- `scripts/maintenance/reindex_semantic_entities.py`
- `backend/tests/`
- `test/`
- `docs/PROJECT_MEMORY.md`

步骤：

1. 用三相册数据的隔离 SQLite 副本运行 dry-run，检查地点主类覆盖率、`其他或不确定` 数量、物品/氛围细节分布和 evidence link 完整性。
2. 在隔离副本执行 apply，比较 Asset、Observation、Event、Entity、EntityObservation 和原始 raw JSON 的计数/哈希。
3. 使用本地 Web 服务打开语义目录，确认地点卡片不出现 GPS 名称、文件名或内部 ID；确认缩略图能打开原始 Asset。
4. 运行完整 Python、Node、编译和 diff 检查。
5. 记录实际词表覆盖、兜底比例、失败资产和已知限制；不把语义分类覆盖率当作事实准确率。

命令：

```bash
.venv/bin/python -m unittest discover -s backend/tests -v
node --test test/*.test.js
node --check src/app.js
node --check src/api.js
.venv/bin/python -m compileall -q backend scripts
git diff --check
```

预期：完整回归结果被记录；任何失败先定位，不跳过或修改既有测试结论。

### 7. 传输到 153、在线验证和正式提交

步骤：

1. 本地记录 Git 状态、HEAD、测试输出和只涉及的文件清单。
2. 再次检查 153 `psh` 是否干净；只传输源代码、测试、维护脚本和文档，不传输 `.env`、凭据、数据库、备份、日志或模型。
3. 在 153 运行完整回归和语义维护 dry-run。
4. 仅在明确维护窗口使用 SQLite backup 执行生产派生重建；执行前后检查 integrity、scope 数量、raw JSON 哈希和 evidence link。
5. 只重启 Sentrix 8090 以加载后端变更；不停止、修改或重启 FMA 5173。
6. 验证 `8090/api/health`、Web 4174、FMA 5173 和实际语义 API。
7. 后端及网页改动统一在 153 `psh` 提交，并更新 `docs/PROJECT_MEMORY.md` 的实际状态。

## 不变约束

- 不修改 `backend/agent.py` 的规划、检索和回答逻辑。
- 不把 GPS、文件名、Asset ID 或匿名模型标签当作用户可见语义主名称。
- 不跨 MemorySpace 聚合。
- 不物理删除或覆盖原始 Observation、Asset、raw 模型输出和用户属性。
- 不将视觉氛围写成个人情绪、人物关系、所有权或家庭事实。
- 不宣称语义层准确率，除非有可审计标注和独立评估结果。
