# Offline Geocoded Image Context Design

## Goal

将图片 EXIF 中的 GPS 坐标转换为低精度、可解释的离线地点上下文，作为单张图片描述的输入属性；同时保持视觉语义地点分类独立，避免 GPS 或反向地理编码结果覆盖图片中的语义地点。

## Data Contract

`assets.metadata_json` 保留原始 `exif.gps`，并增加独立的 `reverse_geocode` 对象：

```json
{
  "source": "offline",
  "precision": "city",
  "label": "上海市浦东新区",
  "city": "上海市",
  "district": "浦东新区",
  "admin1": "上海市",
  "admin2": "浦东新区",
  "country": "CN",
  "latitude": 31.2304,
  "longitude": 121.4737,
  "distance_km": 2.4,
  "confidence": 0.72
}
```

字段职责：

- `exif.gps`：原始证据，不改写。
- `reverse_geocode`：坐标推导的地理上下文，不是视觉事实。
- `observation.place`：模型从图片中判断的自然语言地点描述。
- `observation.semantic.place.primary`：从受控地点词表选择的地点主类。
- `observation.semantic.place.details`：从图片中观察到的地点特征多选项。
- `observation.scene_type`：规范化后的地点主类，和 `semantic.place.primary` 一致。
- `observation.caption`：可以引用可信的地理上下文来提高描述可读性，但必须保持不确定性。

没有 GPS、没有本地索引或匹配结果过远时，`reverse_geocode` 为空；系统不调用外部地图 API，也不编造具体地点。

## Processing Flow

```text
图片导入
  -> EXIF 解析出 GPS
  -> OfflineReverseGeocoder 查询城市/区县及可选精选 POI
  -> reverse_geocode 写入 Asset metadata
  -> 单图 Gamma 描述 prompt 获得地理上下文
  -> Gamma 只根据画面填写 place/semantic.place/scene_type
  -> Observation 保存视觉语义与原始模型响应
  -> 事件聚合使用 place/semantic 字段；地理上下文只作为独立证据保留
```

## Offline Strategy

第一阶段使用 `reverse_geocoder` 的离线城市索引，返回最近城市和行政层级。为支持有限的常用地名，可通过 `SENTRIX_GEO_POI_PATH` 加载本地 JSON 点位索引，并仅在距离阈值内返回最近 POI。后续可以替换为更完整的行政区 polygon 或 OSM 区域索引，而无需改动模型和数据库契约。

## Prompt Boundary

1. `location_context` 是由坐标推导的候选上下文，只能帮助描述拍摄背景。
2. `place` 和 `semantic.place.primary` 必须只依据图片视觉证据及受控词表填写。
3. `semantic.place.details` 只能来自图片可观察特征。
4. 如果画面和地理上下文冲突，保留视觉分类，不能用 GPS 改写地点主类。
5. 无法确认的地点特征返回空数组，不猜测 POI、门牌或人物。

## Failure and Compatibility

- 缺少可选依赖或索引时，地理编码器退化为空结果，图片处理继续完成。
- 已有数据库无需迁移；`reverse_geocode` 存入已有 `metadata_json`。
- 已有客户端继续使用 `place`、`semantic` 和 `scene_type`；新字段只增加信息，不改变旧接口字段含义。
- 原始 GPS 和模型原始响应继续保留，便于审计和后续重算。
