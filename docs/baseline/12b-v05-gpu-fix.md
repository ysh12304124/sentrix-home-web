# 12B-FC · V0.5 GPU driver 修复报告

**日期**：2026-08-06
**性质**：独立维护窗口——修复 153 GPU driver mismatch，重启服务器，恢复服务，并发现/修复 2 个阻塞性代码 bug。

## 1. 根因

- 内核模块 `NVRM 595.71.05`（旧，已加载）vs 用户态 `NVML 595.84`（新）→ `nvidia-smi` 报 "Driver/library version mismatch"。
- `/var/run/reboot-required` 已置位（驱动已升级到 595.84 + 新内核 6.8.0-136 已装），需重启加载 595.84 内核模块。

## 2. 执行

1. **备份**（V0 完成）：`/home/asus/sentrix-backups/20260806-122813`（1.1G，SQLite backup API + checksum manifest）。
2. **诊断**：确认已装驱动 595.84（dpkg nvidia-driver-595 + dkms nvidia/595.84 for 6.8.0-124/136），旧 595.71 模块仍加载。
3. **重启**（用户确认）：`sudo reboot` → ~50s 恢复。
4. **恢复服务**：`restart_sentrix_services.sh` → 8091 API（health 200）、8081 ASR（up）；**e2b 8100 无法恢复**——`services/e2b_server` 模块在当前 repo 中不存在（R8 事故恢复时可能丢失，git 未跟踪），仅 experimental_2b 对照 profile 需要，**不阻塞 12B 验证**。

## 3. 修复后状态

| 项 | 修复前 | 修复后 |
|---|---|---|
| nvidia-smi | mismatch 失败 | **Driver 595.84 / CUDA 13.2 / RTX 3090 24GB** ✅ |
| 12B 模型 | CPU，62s/case | **GPU VRAM 8.4GB 全量驻留，GPU util 93%** ✅ |
| 12B parser warm | 62s（CPU） | **~4.1s**（GPU，含完整 prompt） |

## 4. 顺带发现并修复的 2 个代码 bug（阻塞真实 12B 参与）

1. **`RequestDeadline` 是进程级而非请求级**（`model_routing.py`）：deadline 在 ModelRouter 构造时创建（进程启动），进程运行 >20s 后 `remaining()=0` → 所有模型调用立即 fallback → **12B 永不参与**。修复：`thin_agent.answer_turn` 每请求重置 `router.deadline = RequestDeadline()`（并发重置只会延长预算，不会缩短，安全）。提交 `7c4dd80`。
2. **parser 阶段预算 4s 太紧**（12B GPU 需 ~4s）：被 4s budget 切掉 → 超时 → breaker 熔断。修复：parser budget 提至 **8s**（`SENTRIX_PARSER_BUDGET` 可配）。提交 `e718e4e`。

## 5. 验证

- parser 稳定 4.1s/请求，无熔断，`model_calls.parser=1` 每请求真实调用。
- 12B parser 正确产出 `summarize_person` action + person facet（"介绍一下明哥"）——2B 时代做不到。

## 6. 遗留（后续阶段处理）

- **e2b 8100 未恢复**（模块缺失；仅 experimental_2b 需要）。
- **明哥人物链数据**：明哥 confirmed entity 有 5 events，但 album2 scope 有 **0 条 observation**——需 V3 定位明哥可检索的 observation 所在 scope/资产，否则 V4 人物链无证据。
