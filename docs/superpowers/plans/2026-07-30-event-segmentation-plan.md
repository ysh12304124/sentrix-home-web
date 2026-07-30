# 事件切分质量实施计划

## 目标

让 Sentrix 在时间地点相同或相近时，使用真实图片向量、活动/对象语义和已确认
人物重合判断是否属于同一事件。外部 manifest 的 `event_id` 只用于离线评估，
不得进入 Asset、Observation、Event 或聚合评分。

## 架构

```text
Asset -> Observation + CLIP asset vector
                      |
                      v
Event candidate recall (time window + capture location)
  -> semantic score (activity, type, objects, confirmed people)
  -> visual score (candidate asset CLIP maximum cosine)
  -> conservative split guard
  -> Event / EventObservation + score breakdown

External manifest labels -> benchmark only -> split/merge pairwise metrics
```

## 技术栈

- Python 3、SQLite、unittest
- existing `ClipAdapter` and `memory_vectors` visual space
- existing `MemoryStore` event/observation records

## 实施步骤

1. Add failing pipeline and store tests covering candidate-vector availability,
   generic same-event merging, and a split when both visual and semantic
   evidence disagree. Confirm that visual difference alone does not split an
   event, because a family event normally contains close-ups and varied views.
2. Add a read-only benchmark test for pairwise split/merge metrics. Confirm it
   reads external labels only after grouping and cannot write them into the
   Sentrix database.
3. In `backend/pipeline.py`, persist the image asset vector immediately after
   the observation is created and before `merge_observation_into_event`; update
   its metadata with the selected event ID after merging.
4. In `backend/db.py`, add visual-vector lookup for an observation and its
   candidate event, calculate maximum cosine similarity, and store it in the
   aggregation breakdown. Reject a candidate only when low visual similarity,
   conflicting activity and event type, and the absence of confirmed-person/object
   corroboration all agree that it is a different event.
5. Add `scripts/benchmarks/evaluate_event_segmentation.py`, which accepts a
   database and external manifest, computes pairwise precision/recall/F1 plus
   split-event and merged-event counts, and never writes to the database.
6. Run focused tests, full Python and Node suites, syntax/compile checks, and
   execute the benchmark against the controlled 120-image database.
7. Update `docs/PROJECT_MEMORY.md` with the exact scoring contract, benchmark
   command and measured result; commit the verified change on 153 `psh`.

## Acceptance Criteria

- A candidate image vector exists before event selection.
- Dissimilar vectors alone do not split a same-time/place event.
- Same time/place images split only when low vector similarity, conflicting
  activity and type, and no confirmed-person/object bridge occur together.
- Same time/place images with similar vectors merge even when their captions
  differ in incidental detail.
- Every selected or rejected candidate records `visual_similarity`,
  `visual_available`, semantic conflict and split-guard state.
- The benchmark reads labels only from its manifest and reports pairwise F1,
  source-event splits and predicted-event merges.
