# 真实相册测试数据

## 当前批次：A Cloud Guru photo album

来源：

`https://github.com/ACloudGuru-Resources/Course_Go_Serverless_with_a_Graph_Database/tree/master/photos`

该仓库公开提供 `SR_AWS_N_0001.jpg` 到 `SR_AWS_N_0054.jpg`，并配套 `data/vertices-*.csv`、`data/edges-*.csv` 图关系数据。仓库代码采用 MIT License；图片版权边界不等同于代码版权，因此只下载到 153 的未跟踪测试目录，不提交到 Sentrix 仓库，也不对外分发。

下载位置：

`data/test-albums/acg-photo-album/photos/`

## 人物实体聚类批次：LFW

来源：Labeled Faces in the Wild（LFW），通过公开 Figshare 镜像下载。

- 约 13,000 张真实人脸图像；
- 目录名保留公开数据集的人物标签，只用于离线评估，不写入 Sentrix 人物实体；
- 用于验证 `buffalo_l` embedding 的同人聚类、异人分离和簇样本维护；
- 目录：`data/test-albums/lfw/`。

LFW 不是家庭相册，不用于推断家庭关系。它通过 `ingest_face_benchmark.py` 只写入 Asset、face instance、FaceCluster 和 pending Entity，不生成家庭 Event、Fact 或 Relationship。家庭事件测试仍以 A Cloud Guru 相册为主，关系候选只来自 Sentrix 的共现证据和用户确认。

测试目标：

- 批量图片上传是否稳定；
- 同一批相册图片是否能生成多个 Observation 和 Event；
- 人脸候选是否进入待确认人物；
- 多张图片的事实是否进入 active/pending 版本维护；
- Agent 是否能返回具体图片和 Observation 证据。

## 更贴合家庭相册的研究集：PIPA

PIPA（People In Photo Albums）元数据仓库：

`https://github.com/coallaoh/PIPA_dataset`

PIPA 专门面向个人照片集和相册划分，包含 album/day/time 等评估切分；仓库只提供 Photo ID 和标注，原图需要通过 Flickr API 获取，不能在当前环境无凭证下载。因此本阶段不把 PIPA 原图伪装成已接入数据，后续配置 Flickr 凭证后再加入同样的批量导入脚本。
