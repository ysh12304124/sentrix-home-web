"""统一意图信号（Phase H 架构收敛）。

收敛 completion / runtime / tools 三处分散的意图正则，只承担"行为信号"：
是否需要检索、视觉复核、OCR、图片交付、多图预算。不参与"事实是否合格"的判断，
事实正确性由 guard/judge 模型层负责。
"""

from __future__ import annotations

import re

# 纯聊天/通用知识：不需要检索家庭记忆
_CHAT_ONLY_RE = re.compile(
    r"你好|谢谢|在吗|再见|哈哈|好的|嗯|哦|你是谁|你叫什么|你会什么|帮我写|"
    r"写一|写个|改一|翻译|解释一下什么是|什么是|怎么用|步骤|教程", re.I)

# 涉及家庭记忆/照片证据的问题信号
_EVIDENCE_INTENT_RE = re.compile(
    r"照片|图片|记录|回忆|拍|合影|去了|在哪里|哪里|什么|谁|多少|几|"
    r"时间|日期|年份|月份|几号|活动|旅游|旅行|度假|菜单|价格|招牌|店名|"
    r"电话|穿着|衣服|颜色|天气|人物|家人|地点|城市|哪个|哪张")

# 视觉细节意图
_VISUAL_RE = re.compile(
    r"桌上|桌面|颜色|几个|多少人|招牌|文字|天气|外套|衣服|猫|雪|小孩|穿着|穿|"
    r"在做什么|有没有|是什么|放着|写了|内容|细节|长什么样|什么颜色|什么动物|"
    r"拿着|道具|植物|场景|拍的什么|什么造型|什么样子|哪[一123]?张")

# OCR 文字读取意图
_OCR_RE = re.compile(
    r"菜单|价格|多少钱|售价|招牌|店名|电话|写了什么|什么字|文字|"
    r"价位|几块钱|多少钱一份|价格是|上面写了|数字|号码|年份|哪一年")

# 明确要求查看/交付照片
_IMAGE_RE = re.compile(
    r"给我看看|给我看|发我|发给我|发来|原图|都给我|全部给我|"
    r"展示|显示(?:一下|给我)?|让我看看|看看(?:这些|照片|图)?|"
    r"打开(?:照片|图片)|看图|给我图|把.{0,6}(?:照片|图片|图)", re.I)

# 多图/跨图对比意图
_MULTI_IMAGE_RE = re.compile(
    r"哪一张|哪张|哪些|哪几张|每一张|逐[一一张]|逐一|对比|还有吗|还有没有|都看|"
    r"全部|每张|所有照片|哪几张|哪几个|翻看")


def chat_only(message: str) -> bool:
    return bool(_CHAT_ONLY_RE.search(message or ""))


def evidence_intent(message: str) -> bool:
    text = message or ""
    return not chat_only(text) and bool(_EVIDENCE_INTENT_RE.search(text))


def visual_intent(message: str) -> bool:
    return bool(_VISUAL_RE.search(message or ""))


def ocr_intent(message: str) -> bool:
    return bool(_OCR_RE.search(message or ""))


def image_delivery_intent(message: str) -> bool:
    return bool(_IMAGE_RE.search(message or ""))


def multi_image_intent(message: str) -> bool:
    return bool(_MULTI_IMAGE_RE.search(message or ""))
