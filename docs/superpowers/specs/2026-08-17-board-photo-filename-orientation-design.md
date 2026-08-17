# 看板:照片信息框加"文件名" + 修复照片侧置

日期:2026-08-17
状态:已批准(zhx)

## 背景/问题

1. **无法定位具体照片**:看板照片信息框只显示"时间/地点/revision"三项。zhx 在看板反馈某张照片正置/侧置时,仅凭拍摄时间(如"0618 13:45:52")无法判断是哪张照片(是 frame1388 还是 frame1390),必须翻相册/素材库才能定位——不合理。
2. **照片侧置**:看板里竖拍照片横躺/倒置。根因:`asset_file` 对 jpg 照片直接 `FileResponse` 返回原文件,**未应用 EXIF orientation**(而人脸裁剪等路径用了 `exif_transpose`,方向正确)。

## 设计(方案 A)

### 1. 照片信息框加"文件名"
- 在"时间/地点/revision"三框基础上,增加第 4 项"**文件名**",显示照片原始文件名(如 `frame1388.jpg`);
- 数据源:`asset.file_name`(后端 assets 表已有该字段,事件/证据详情接口的 photo 对象已含 `file_name`);
- 前端从该数据渲染显示即可。

### 2. 修复照片侧置
- 根因:`backend/app.py` 的 `asset_file`(L1563-1588),当 `original=False` 且 `needs_browser_transcode` 返回 False(jpg 等)时,直接 `FileResponse` 返回原文件,没有 `exif_transpose`;
- 修复:非 original 的照片(含 jpg)统一生成方向正确的预览返回(复用 `encode_jpeg_preview`,它内部已 `ImageOps.exif_transpose`),或对直传分支也应用 `exif_transpose`。以"方向正确 + 不破坏原图下载(original=True 仍返回原文件)"为准。

## 改动范围

- 前端 `src/app.js`:照片/证据详情信息框(时间/地点/revision)渲染处,增加"文件名"项;
- 后端 `backend/app.py`:`asset_file` 的照片方向处理;
- 若事件/证据详情接口未返回 `file_name`,则补上(目前 photo 对象已含,确认即可)。

## 数据流

照片详情 → API(`/api/assets/{id}` 及事件证据,含 `file_name`/`captured_at`/`captured_location`)→ 前端信息框渲染(时间 / 地点 / revision / 文件名)。

## 测试

1. 看板打开照片详情,信息框显示原始文件名;
2. 竖拍照片(有 EXIF orientation)在概览/事件/照片详情中正置,不再侧置;
3. `original=true` 下载仍返回原始文件(不破坏原图);
4. 人脸裁剪、其他接口不受影响。

## 非目标(YAGNI)

- 不做文件名点击复制/高亮(方案 B,未选);
- 不做无 EXIF 标签照片的内容自动定向(Future_Plans,不在此范围);
- 不改动画板其他板块。
