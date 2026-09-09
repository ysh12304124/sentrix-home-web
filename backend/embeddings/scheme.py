"""单一检索方案（钉死，不再 env 切换）。

历史教训：IMAGE/TEXT embedder 曾多次被 env/.env 切换（clip / chinese_clip / bge），
造成 memory_vectors、ANN 索引、FTS 各套并存且互不匹配，检索时好时坏。
现在把系统**固定为一套**，所有写入/查询/索引都只认这套：

  IMAGE (visual/asset 通道) = chinese_clip   -> chinese-clip-ViT-L-14 (dim 768)
  TEXT  (semantic/episodic) = bge            -> BAAI/bge-m3            (dim 1024)

旧的其它模型行仍保留在库里不删（避免误删/回溯），但**检索与新建都不再触碰它们**：
- 新写入只用本方案的模型名；
- 查询 embedding 只用本方案的两个 slot；
- 重建索引只取本方案的模型行。
"""

IMAGE_EMBEDDER = "chinese_clip"
TEXT_EMBEDDER = "bge"
# 对应写入 memory_vectors.model_name
IMAGE_MODEL = "chinese-clip-ViT-L-14"
TEXT_MODEL = "BAAI/bge-m3"
