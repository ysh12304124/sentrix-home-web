# PhotoBench QA 三集合导出

- 导出时间：2026-08-28T18:53:01+08:00
- 在线来源：`http://192.168.1.65:8782`
- 图片相关范围：album3 当日事件前 30 个日期 + 全部跨日事件

| 集合 | QA 数量 | 含义 |
|---|---:|---|
| `image-related-439q.jsonl` | 439 | 当日事件与跨日事件审核页 QA |
| `single-video-48q.jsonl` | 48 | 视频审核页 QA，统一为单视频 typed refs |
| `mixed-image-video-487q.jsonl` | 487 | 上述两类的完整并集 |

三个文件均保留当前审核状态。图片相关集是按审核工作区来源命名，其中部分事件已经绑定视频或图视频混合 GT；详细媒体构成和状态分布见 `summary.json`。跨日导出与当日导出发生冲突的顺序型 QA ID 已确定性改写，原 ID 保存在 `source_original_qa_id`。
