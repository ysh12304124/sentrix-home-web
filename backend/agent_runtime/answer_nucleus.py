"""Phase H — Answer Nucleus：确定性事实的结构化核心。

目标（H-A Deterministic Fact Delivery）：
- 把工具已经确认的硬值（数量/日期/布尔/结果总数/OCR 价格电话/地点/人物）从
  tool_results 中提取为结构化 NucleusValue，不依赖 LLM 复述。
- 简单确定性问题（数量/日期/布尔）→ 直接确定性渲染，不走 12B 改写。
- 复杂回答 → 硬值绑定：注入 final 约束 + guard 校验，禁止 LLM 改写硬值。

与 final_writer._trusted_facts 的关系：后者面向 recovery/judge 的字符串事实；
本模块面向 final 交付，提供结构化值 + 渲染 + 约束 + 校验。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# 硬值类型：出现这些类型的值必须原样保留
HARD_KINDS = {"count", "date", "first", "last", "result_total",
              "price", "phone", "year", "boolean", "place"}

# 价格/电话/年份提取（与 final_guard._check_ocr_hard_values 对齐）
_PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:元|块|¥|￥)")
_PHONE_RE = re.compile(r"(?<!\d)(\d{7,12})(?!\d)")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

# 简单确定性问题分类
_Q_COUNT = re.compile(r"多少张|几张|多少个|多少条|几段|几首|多少份|几份|几张照片|数量")
_Q_DATE = re.compile(r"哪天|哪一天|几号|什么日期|日期是|拍摄日期|哪天拍的|什么时候拍的|具体时间")
_Q_YEAR = re.compile(r"哪一年|哪年|什么年份|年份|成立于|创立于|开业|创建于|几几年")
_Q_BOOL = re.compile(r"有没有|是不是|是否|有没有拍|是否存在|有没有去过|有没有去过")
_Q_PRICE = re.compile(r"多少钱|售价|价格|费用|收费|卖多少钱|多少钱一杯|多少钱一份|多少钱一(?:杯|份|听|瓶)")
_Q_WHERE = re.compile(r"在哪里|在哪|哪儿|哪里|什么地点|哪个地方|在什么地方|在哪儿")


@dataclass
class NucleusValue:
    kind: str                 # count|date|first|last|place|price|phone|year|boolean|result_total|person
    value: Any
    unit: str = ""            # 张/个/条/元…
    certainty: str = "confirmed"   # confirmed|estimated
    source: str = ""          # 来源 tool / 字段
    label: str = ""           # 自然标签（如“汉堡单人套餐价格”）
    display: str = ""         # 直接可渲染的文本


class AnswerNucleus:
    def __init__(self, values: list[NucleusValue] | None = None):
        self.values: list[NucleusValue] = values or []

    def get(self, kind: str) -> NucleusValue | None:
        for v in self.values:
            if v.kind == kind:
                return v
        return None

    def all(self, kind: str) -> list[NucleusValue]:
        return [v for v in self.values if v.kind == kind]

    @property
    def empty(self) -> bool:
        return not self.values

    def hard_values(self) -> list[str]:
        """不可修改的硬值文本（去重保序）。"""
        out: list[str] = []
        for v in self.values:
            if v.kind not in HARD_KINDS:
                continue
            text = v.display or str(v.value)
            if text and text not in out:
                out.append(text)
        return out

    def constraint_text(self) -> str:
        """注入 LLM 的硬值约束（复杂回答时使用）。"""
        hard = self.hard_values()
        if not hard:
            return ""
        return ("以下数字/日期/结果总数是从工具结果确认的硬值，回答中必须原样使用、"
                "不得修改、不得遗漏、不得替换成其他数值：" + "、".join(hard) + "。")


# ---------------------------------------------------------------------------
# 提取
# ---------------------------------------------------------------------------

def _fmt_date(value: Any) -> str:
    """把 2022-06-23 / 2022/6/23 / 2022年6月23日 归一化为中文日期。"""
    s = str(value or "").strip()
    m = re.match(r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})", s)
    if m:
        return f"{m.group(1)}年{int(m.group(2))}月{int(m.group(3))}日"
    if re.fullmatch(r"\d{4}[-/年.]\d{1,2}[-/月.]\d{1,2}.*", s):
        return s
    return s


def build_nucleus(task_state: dict, question: str = "") -> AnswerNucleus:
    """从 task_state / tool_results 提取结构化确定性值。"""
    nucleus = AnswerNucleus()
    task_state = task_state or {}

    # 1) query_memory_facts 确定性操作
    op = task_state.get("fact_operation")
    value = task_state.get("fact_value")
    if op in {"count", "media"} and isinstance(value, int):
        nucleus.values.append(NucleusValue(
            kind="count", value=value, unit="条" if op == "count" else "个",
            certainty="confirmed", source="query_memory_facts", display=str(value)))
    elif op in {"first", "last", "date"} and value:
        d = _fmt_date(value)
        nucleus.values.append(NucleusValue(
            kind=op, value=value, certainty="confirmed",
            source="query_memory_facts", display=d))
    elif op == "exists" and isinstance(value, bool):
        nucleus.values.append(NucleusValue(
            kind="boolean", value=value, certainty="confirmed",
            source="query_memory_facts",
            display="有" if value else "无"))

    # 2) search_memories：结果总数（精确计数，非估计）
    for tr in task_state.get("tool_results") or []:
        total = tr.get("total")
        if tr.get("tool") == "search_memories" and isinstance(total, int) and total > 0:
            satisfaction = task_state.get("search_satisfaction")
            certainty = "confirmed" if satisfaction in ("full_support", "partial_support") \
                else "estimated"
            nucleus.values.append(NucleusValue(
                kind="result_total", value=total, unit="张",
                certainty=certainty, source=tr.get("tool_call_id") or "search_memories",
                display=str(total)))
            group_count = tr.get("group_photo_count")
            if isinstance(group_count, int) and group_count >= 0:
                sizes = tr.get("group_photo_sizes") or []
                nucleus.values.append(NucleusValue(
                    kind="count", value=group_count, unit="张",
                    certainty="confirmed", source=tr.get("tool_call_id") or "search_memories",
                    label="不同人数合影", display=str(group_count)))
        # 3) read_photo_text：OCR 硬值（价格/电话/年份/数字）
        #    优先用结构化 exact_values（带 label/provider/confidence），缺失时回退 ocr_text 正则。
        if tr.get("tool") == "read_photo_text":
            evs = tr.get("exact_values") or []
            if evs:
                for ev in evs:
                    kind = str(ev.get("type") or "")
                    if kind not in {"price", "phone", "year"}:
                        continue
                    value = str(ev.get("value") or "")
                    if not value:
                        continue
                    label = str(ev.get("label") or "")
                    display = str(ev.get("text") or value)
                    if kind == "price":
                        display = f"￥{value}"
                    if any(v.value == value and v.kind == kind and (v.label or "") == label
                           for v in nucleus.all(kind)):
                        continue
                    nucleus.values.append(NucleusValue(
                        kind=kind, value=value,
                        unit="元" if kind == "price" else "",
                        certainty="confirmed", source="ocr",
                        label=label, display=display))
            elif (tr.get("full_text") or tr.get("ocr_text") or "").strip():
                ocr = str(tr.get("full_text") or tr.get("ocr_text"))
                for m in _PRICE_RE.finditer(ocr):
                    nucleus.values.append(NucleusValue(
                        kind="price", value=m.group(1), unit="元",
                        certainty="confirmed", source="ocr", display=m.group(0).strip()))
                for m in _PHONE_RE.finditer(ocr):
                    nucleus.values.append(NucleusValue(
                        kind="phone", value=m.group(1), certainty="confirmed",
                        source="ocr", display=m.group(1)))
                for m in _YEAR_RE.finditer(ocr):
                    if m.group(1) not in [v.display for v in nucleus.all("year")]:
                        nucleus.values.append(NucleusValue(
                            kind="year", value=m.group(1), certainty="confirmed",
                            source="ocr", display=m.group(1)))
    # 4) 时间/地点（来自搜索结果的结构化媒体元数据）。 Bind these facts
    # to the query-recommended photo. A majority over a mixed preview can
    # confidently report another event's date/place.
    required_evidence_types = {
        str(req.get("evidence_type") or "")
        for req in (task_state.get("requirements") or [])
        if isinstance(req, dict) and req.get("required", True)
        and str(req.get("status") or "open") in {"open", "running", "partially_supported"}
    }
    date_requested = bool(
        _Q_DATE.search(question or "")
        or _Q_YEAR.search(question or "")
        or "temporal_metadata" in required_evidence_types
    )
    place_requested = bool(
        _Q_WHERE.search(question or "")
        or "location_metadata" in required_evidence_types
    )

    metadata_candidates = []
    if any((tr or {}).get("tool") == "search_memories"
           for tr in task_state.get("tool_results") or []):
        for tr in task_state.get("tool_results") or []:
            if tr.get("tool") != "search_memories":
                continue
            preview = [item for item in (tr.get("preview") or [])
                       if isinstance(item, dict)]
            if not preview:
                continue
            recommended = str(tr.get("recommended_handle") or "")
            selected = next((item for item in preview
                             if recommended and item.get("handle") == recommended), None)
            metadata_candidates.append(selected or preview[0])
        if metadata_candidates:
            selected = metadata_candidates[-1]
            captured_at = str(selected.get("captured_at") or "").strip()
            if captured_at and _Q_YEAR.search(question or ""):
                year_match = _YEAR_RE.search(captured_at)
                if year_match:
                    year = year_match.group(1)
                    nucleus.values.append(NucleusValue(
                        kind="year", value=year, certainty="confirmed",
                        source=f"search_memories.{selected.get('handle')}.captured_at",
                        display=year))
            elif captured_at and date_requested:
                match = re.match(r"(\d{4})[-/]([0-9]{1,2})[-/]([0-9]{1,2})", captured_at)
                if match:
                    date = f"{match.group(1)}年{int(match.group(2))}月{int(match.group(3))}日"
                    nucleus.values.append(NucleusValue(
                        kind="date", value=date, certainty="confirmed",
                        source=f"search_memories.{selected.get('handle')}.captured_at",
                        display=date))

    # Place metadata must come from the same recommended photo, not the most
    # frequent location among unrelated search candidates.
        if place_requested:
            selected = metadata_candidates[-1] if metadata_candidates else {}
            top_place = str(selected.get("place") or "").strip()
            if len(top_place) >= 2:
                nucleus.values.append(NucleusValue(
                    kind="place", value=top_place, certainty="confirmed",
                    source=f"search_memories.{selected.get('handle')}.place",
                    display=top_place))

    # 5) 当前关注人物
    active = task_state.get("active_person")
    if active:
        nucleus.values.append(NucleusValue(
            kind="person", value=active, certainty="confirmed",
            source="task_state", display=str(active)))
    return nucleus


# ---------------------------------------------------------------------------
# 简单确定性问题：直接渲染
# ---------------------------------------------------------------------------

def classify_deterministic(question: str) -> str | None:
    """返回 'count' | 'date' | 'year' | 'place' | 'price' | 'boolean' | None。"""
    q = question or ""
    if _Q_PRICE.search(q):
        return "price"
    if _Q_YEAR.search(q) and not _Q_COUNT.search(q):
        return "year"
    if _Q_DATE.search(q) and not _Q_COUNT.search(q):
        return "date"
    if _Q_WHERE.search(q):
        return "place"
    if _Q_COUNT.search(q):
        return "count"
    if _Q_BOOL.search(q):
        return "boolean"
    return None


def render_simple(nucleus: AnswerNucleus, kind: str, question: str = "") -> str | None:
    """简单确定性问题直接渲染；缺少确定值返回 None（交给正常 LLM 流程）。"""
    if kind == "count":
        v = nucleus.get("count") or nucleus.get("result_total")
        if v is None:
            return None
        unit = v.unit or "个"
        return f"一共 {v.value} {unit}。" if v.certainty == "confirmed" \
            else f"大约 {v.value} {unit}。"
    if kind == "date":
        v = nucleus.get("date") or nucleus.get("first") or nucleus.get("last")
        if v is None:
            return None
        label = {"first": "最早一次", "last": "最近一次"}.get(v.kind, "相关时间")
        d = v.display or _fmt_date(v.value)
        return f"{label}是 {d}。"
    if kind == "boolean":
        v = nucleus.get("boolean")
        if v is None:
            return None
        return "有的。" if v.value else "没有找到相关记录。"
    if kind == "year":
        vs = nucleus.all("year")
        if not vs:
            return None
        # 优先 label 含创立/成立/始于/开业/创建 的年份
        best = None
        for v in vs:
            if re.search(r"创立|成立|始于|开业|创建", v.label or ""):
                best = v
                break
        v = best or vs[0]
        return f"{v.value} 年。"
    if kind == "place":
        v = nucleus.get("place")
        if v is None:
            return None
        return f"地点是 {v.display or v.value}。"
    if kind == "price":
        # 只当问题里的商品词与核值 label 匹配时才确定性渲染，避免错配（如多个价格）
        q = question or ""
        labels = [v for v in nucleus.all("price") if v.label and len(v.label) >= 2]
        if not labels:
            return None
        qwords = re.findall(r"[\u4e00-\u9fff]{2,8}", q)
        best = None
        for v in labels:
            if any(v.label == w or v.label in w or w in v.label for w in qwords):
                best = v
                break
        if best is None:
            return None
        return f"{best.label}的售价是 {best.value} 元。"
    return None


# ---------------------------------------------------------------------------
# Guard 校验：final 不得改写硬值
# ---------------------------------------------------------------------------

def check_nucleus_preservation(answer: str, nucleus: AnswerNucleus,
                               question: str = "") -> list[str]:
    """校验 final 回答是否保留了硬值；返回问题代码列表（空=通过）。"""
    issues: list[str] = []
    if not answer:
        return issues
    # 1) 结果总数/数量：答案若出现“N 张/个/条”且与核值不符 → 改错
    rv = nucleus.get("result_total")
    if rv is not None:
        m = re.search(r"(?<!\d)(\d+)\s*张", answer)
        if m and int(m.group(1)) != int(rv.value):
            issues.append(f"count_conflict:expected={rv.value}")
    cv = nucleus.get("count")
    if cv is not None:
        for m in re.finditer(r"(?<!\d)(\d+)\s*(?:个|条|张)", answer):
            if int(m.group(1)) != int(cv.value):
                issues.append(f"count_conflict:expected={cv.value}")
                break
    # 2) 日期：答案写错日期 → date_conflict；问日期但答案无日期 → date_missing
    for kind in ("date", "first", "last"):
        dv = nucleus.get(kind)
        if dv is None:
            continue
        d = _fmt_date(dv.value)
        full_date = re.search(
            r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})", answer)
        if full_date:
            got = f"{full_date.group(1)}年{int(full_date.group(2))}月{int(full_date.group(3))}日"
            if got != d:
                issues.append(f"date_conflict:expected={d}")
            continue
        # 无完整日期：核值年月日任一出现在答案里但不同 → 冲突
        nums = re.findall(r"\d+", d)
        answer_nums = set(re.findall(r"\d+", answer))
        expected = set(nums)
        if expected & answer_nums and expected - answer_nums:
            issues.append(f"date_conflict:expected={d}")
        elif not expected & answer_nums and classify_deterministic(question) == "date":
            issues.append(f"date_missing:expected={d}")
    # 3) 价格/电话/年份：已有 final_guard._check_ocr_hard_values 兜底，
    #    这里补充“答案出现核值之外的同类硬值”场景由外层 guard 负责。
    return issues


def nucleus_preservation_issues(answer: str, task_state: dict,
                                question: str = "") -> list[str]:
    """供 runtime guard 调用：从 task_state 现场构建 nucleus 并校验。"""
    return check_nucleus_preservation(answer, build_nucleus(task_state, question), question)
