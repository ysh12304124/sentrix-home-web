# 人物级衣物证据链实施计划

## 目标

将“人物穿着”从场景级图片描述中拆出，只在已确认的人脸实例拥有可追溯的
视觉证据时，才生成该人物的 `clothing` 语义声明。每一条声明必须能回到
`SemanticClaim -> PersonAppearanceEvidence -> FaceInstance -> Observation -> Asset`。

## 架构

```text
confirmed face cluster
  -> one high-quality face instance per related event
  -> deterministic head-and-upper-body crop
  -> Gemma visual clothing extraction with target-only instruction
  -> person_appearance_evidence
  -> rebuild_person_memory
  -> SemanticClaim(clothing) and SemanticProfile.appearance_summary_zh
  -> Agent evidence-backed clothing answer
```

场景 `Observation.clothing` 保持不变，仅作为场景事实；它不能进入人物衣物
声明。模型无法把衣物明确归属给目标人物时返回空数组，仍可保存“已检查但未
确认”的证据记录，避免把不确定性伪装成事实。

## 实施步骤

1. 在 `backend/tests/test_model_clients.py`、`backend/tests/test_entities.py` 和
   `backend/tests/test_agent.py` 添加失败测试：人体裁剪分析的输出契约、脸级
   衣物证据生成声明、Agent 返回该证据及原始 Asset。
2. 运行新增测试，确认失败原因是目标方法和表尚不存在。
3. 新增 `backend/person_appearance.py`，实现有边界保护的头部与上半身扩展
   裁剪；裁剪大小不足时明确失败，绝不使用整张图片替代。
4. 在 `backend/model_clients.py` 增加 `analyze_person_appearance`：输入裁剪
   JPEG、目标脸框元数据，输出仅能归属给该人物的简体中文衣物数组。
5. 在 `backend/db.py` 增加 `person_appearance_evidence` 表、索引、写入、查询
   和候选选择；修改 `rebuild_person_memory` 只以该表重建衣物 Claim 与画像。
6. 在 `backend/app.py` 的确认人物流程中，按每个关联事件选择最高质量人脸，
   分析人体裁剪、写入证据并再次重建人物记忆；单次失败不写空的虚假衣物。
7. 运行新增测试及完整后端、前端、语法、编译检查；用 153 当前测试数据确认
   一名已确认人物，检查 Claim、Agent 和原图证据链。
8. 更新 `docs/PROJECT_MEMORY.md` 的数据契约和验收状态，并在 153 正式分支
   提交已验证的功能。

## 验收标准

- 没有 `person_appearance_evidence` 时，场景衣物不会写入人物 Claim。
- 每个衣物 Claim 的 `evidence_ids_json` 指向脸级证据；该记录包含 face、
  observation、asset 和扩展裁剪坐标。
- 人物衣物查询答案和证据层同时包含 Claim、人物外观证据和原始 Asset。
- 只有已确认人物触发模型分析；未确认簇不产生人物画像事实。
