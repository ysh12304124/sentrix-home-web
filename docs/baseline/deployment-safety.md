# 部署安全红线（Phase R8-8）

**背景**：2026-08-06 交付事故——一条多源 `rsync --delete-excluded` 命令误清了 153 生产工作树（`.git`/`backend/`/`data/`/`.venv`/前端全丢）。虽然 DB 从运行进程 fd 恢复、源码从本地重建，但这是比 Recall 差几个点更严重的工程风险。以下红线在继续自动部署前必须建立。

## 1. 同步命令红线

- **禁止**对生产仓库 / DB / 媒体 / ANN / 模型权重使用未经 `--dry-run` 的 `rsync --delete`（含 `--delete-excluded`、`--delete-after`）。
- 传输源码一律用 **git patch / bundle / 同 commit cherry-pick**，不再 `scp` 覆盖工作树。
- 单文件/单目录传输用 `scp <file> <dest>`；**禁止多源 scp 到单目录**（会错位）。
- 每次 `rsync` 必须 `--dry-run` 先预览，确认无误再加真实执行。

## 2. 备份分层

| 层 | 方式 | 脚本 |
|---|---|---|
| 源码 | git bundle（`git bundle create psh.bundle --all`） | `backup_sentrix.sh` |
| 数据库 | **SQLite backup API**（`conn.backup()`），不直接复制活动 WAL 数据库文件 | `backup_sentrix.sh` |
| 媒体 | 目录复制（源图在 repo 外 `/home/asus/samples/` + `data/household-benchmark-source`） | `backup_sentrix.sh` |
| ANN 索引 | 目录复制（可重建，但仍备份） | `backup_sentrix.sh` |
| 模型权重 | 目录复制（CLIP/Chinese-CLIP 权重） | `backup_sentrix.sh` |
| 生产日志 | 目录复制 | `backup_sentrix.sh` |

```bash
bash scripts/maintenance/backup_sentrix.sh          # -> /home/asus/sentrix-backups/<timestamp>/
bash scripts/maintenance/backup_sentrix.sh /tmp/backup-test   # 到指定目录
```

## 3. 部署脚本排除清单

任何部署/同步到 153 的操作必须排除：
```
data/                 （含 data/sentrix.db、媒体、ANN）
logs/
.env
*.bin / 模型权重
data/ann/
.venv / .venv-mac / node_modules
__pycache__ / *.pyc
```

## 4. 部署前后校验

- 每次部署前：文件清单（`git ls-files` + 目标文件列表）。
- 每次部署后：`git status` 干净、测试全绿、健康检查 8091/4174/5173 全 200。
- 关键文件（backend 核心、configs）在部署前后 `sha256sum` 记录在案。

## 5. 恢复流程（演练过）

1. **DB**：若文件被删但进程存活 → `cp /proc/<pid>/fd/3 data/sentrix.db`（+wal/shm fd 4/5），然后 `PRAGMA integrity_check`。
2. **源码**：git bundle 重建 → `git init -b psh && git fetch <bundle> && git reset --mixed origin/psh`。
3. **venv**：`python3.10 -m venv --system-site-packages .venv` + 补装缺失包。
4. **媒体**：`/home/asus/samples/album{1,2,3}/images` → `data/household-benchmark-source/`。
5. **ANN**：`rebuild_ann_indices.py --visual-embedder chinese_clip --apply`。
6. **检索投影**：`rebuild_retrieval_indexes.py data/sentrix.db --apply`。
7. **服务**：`nohup /home/asus/start_sentrix_api_8091.sh > /tmp/sentrix-api-8091.log 2>&1 < /dev/null &`（分两步：先 pkill，再独立 ssh nohup）。

**演练要求**：本流程至少完整演练一次（用测试备份到临时目录），并记录演练日期与结果。

## 6. 验收

- R8-7 关闭清单第 8 项：部署安全门槛通过 = 红线文档化 + `backup_sentrix.sh` 可用 + 恢复流程演练记录。
