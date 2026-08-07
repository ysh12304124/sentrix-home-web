# Offline Geocoded Image Context Implementation Plan

## Goal

让 GPS 反向地理编码以离线、低精度方式进入单图描述上下文，同时保证视觉语义地点分类不受其覆盖。

## Architecture

新增 `backend/geocoding.py`，封装离线城市查询和可选本地点位索引；`backend/pipeline.py` 在调用 Gamma 前生成 `location_context` 并写回 Asset metadata；`backend/model_clients.py` 增加严格的 prompt 边界；使用现有 `semantic_taxonomy.py` 保存 `semantic.place.primary/details`，不改数据库 schema。

## Technology

Python、SQLite JSON metadata、`reverse_geocoder` 可选离线索引、现有 Gamma/Ollama 客户端、Python `unittest`。

## Steps

1. 在 `backend/tests/test_geocoding.py` 编写离线城市和 POI 匹配测试。
2. 添加 `backend/geocoding.py`，实现坐标校验、离线城市查询、可选 POI JSON 查询、距离和置信度计算，并让无依赖/无数据时安全返回空结果。
3. 在 `backend/tests/test_pipeline.py` 验证 GPS 会写入 `reverse_geocode`、传入 Gamma metadata，且不会改变 `semantic.place.primary`。
4. 修改 `backend/pipeline.py`，在单图 Gamma 调用前生成地理上下文，保留批处理、并行视觉任务和现有 Asset provenance。
5. 在 `backend/tests/test_model_clients.py` 增加 prompt 边界断言。
6. 修改 `backend/model_clients.py`，更新单图提示词；不把 `reverse_geocode` 写入 `place` 或 `semantic.place`。
7. 运行完整 Python 测试、语法检查和 `git diff --check`。
8. 将只涉及本功能的文件同步回 153 `psh`；重新运行测试并确认另一位 agent 的脚本与评估文件未被修改。
