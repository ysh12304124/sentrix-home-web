"""统一时间基准：Runtime / Tool / Resolver 共用同一 current datetime + timezone。

Phase C 把"相对时间"换算收敛到单一来源，避免 12B 模型自行估算年份。
时区固定 Asia/Shanghai（可用 SENTRIX_TIMEZONE 覆盖），不做多时区。
"""

from __future__ import annotations

import os
from datetime import datetime

try:
    from zoneinfo import ZoneInfo as _ZoneInfo
except ImportError:  # pragma: no cover - python < 3.9
    _ZoneInfo = None

_DEFAULT_TZ = "Asia/Shanghai"


def timezone_name() -> str:
    return os.environ.get("SENTRIX_TIMEZONE", _DEFAULT_TZ)


def now() -> datetime:
    """当前时刻（含时区）。zoneinfo 不可用时退回系统本地时间。"""
    if _ZoneInfo is not None:
        try:
            return datetime.now(_ZoneInfo(timezone_name()))
        except Exception:
            pass
    return datetime.now().astimezone()


def current_time_line() -> str:
    """注入系统提示的时间基准行。"""
    n = now()
    return (f"当前时间：{n.year}年{n.month}月{n.day}日（{timezone_name()}）。"
            "相对时间一律由系统按当前时间换算，你只需把用户原话中的相对时间原样写进工具 filters.time"
            "（如'去年'、'这两年'、'去年春天'、'上个月'），不要自己估算成具体年份。")
