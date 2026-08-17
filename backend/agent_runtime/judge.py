"""L2 模型评审 guard — 12B Observation Faithfulness Judge（对照实验 18 例：bad_recall 90.9% / good_precision 100%）。

分层：L1 FinalGuard（确定性规则，只做结构性兜底：安全/占位符/交付完整性/流程结构）
→ L2 本模块（模型做语义级合格性判断：编造/矛盾/漏报/过度声称/缺口披露）。

设计原则（Phase H H7 拍板）：
- 回答"是否合格"由模型判断，代码不用词语/数字正则做等价匹配。
- 数字等价（"三个人" vs "3 个人" vs "3"）、中文数字 vs 阿拉伯数字、日期格式、
  同义词/量词差异一律视为等价表达，不算冲突。
- 只有"实质事实不同"（数值不同、存在性相反、观察明确没有却断言）才算问题。
- 模棱两可时倾向判 faithful（宁可漏报，不误伤事实正确的回答）。
- 代码只做结构性兜底：L2 异常/输出不可解析 → 放行；L1 已兜底安全与交付结构。
"""

from __future__ import annotations

import json

from .guard_types import (GuardIssue, GuardResult, REVISION_REWRITE_ONLY,
                          SEVERITY_STYLE, SEVERITY_TRUTH)

JUDGE_SYSTEM = """你是 Sentrix 家庭记忆助手的“事实一致性评审”。你的任务是核对模型的最终回答是否忠实于工具观察。

工具观察是唯一可信事实来源。请判断最终回答是否存在以下问题：
1. fabrication（编造）：回答断言了观察中明确没有的事实或细节。例如观察说“照片中没有猫”，回答却说“猫是白色的”。
2. contradiction（矛盾）：回答与观察直接冲突。例如观察说“多云天气”，回答说“晴天阳光充足”；工具确认数量是 5，回答写 3。
3. omission（漏报/否认）：工具返回了结果（total>0 或有观察），回答却整体否认“没有找到/没找到/未找到/不存在”。
4. certainty_upgrade（过度声称）：观察/检索只是 candidate_only 或 unknown，回答却断言“确认/确定是”。
5. missing_disclosure（缺口未披露）：检索只是 partial/candidate，回答完全没提“还不能确认”就当成确定事实。

判定标准（必须严格遵守，宁可漏报，不误伤正确回答）：
- 表达等价不算任何问题：
  - 数字等价：“三个人” == “3 个人” == “3 人” == “3”；“两张” == “2 张”。中文数字与阿拉伯数字等价。
  - 日期格式等价：“2024年3月2日” == “2024-03-02” == “2024.3.2”；“3月” == “03月”。
  - 量词/同义词差异不算冲突（“3 个人” vs “三人”；店名/地名表达不同但指同一事物）。
  - 只有在实质数值不同（工具 5、回答 3）或存在性相反时才算 contradiction。
- fabrication 的门槛：观察明确没有该事实、或与观察相反才判；观察未提及、含糊、缺失时一律不判 fabrication。
- certainty_upgrade 的门槛：观察明确是 candidate_only/unknown/无法判断，回答却用“确定是/确认是/就是”断言才判；
  “看起来是/可能是/应该是/不能完全确认”不算。
- omission 的门槛：工具 total>0 且回答整体否认存在才算；回答只答了一部分、遗漏了某个细节不算 omission。
- missing_disclosure 只是披露问题，不影响事实正确性；回答已给出核心事实并自然说明不确定性时不判。

特别注意：
- search_memories 返回 total=0（无候选）时，回答里出现“已为您找到/找到 N 张照片”这类明确交付断言是编造；
  但如实说“没有找到”正确。
- search_memories 返回 total>0（有候选）时，回答里出现“没有找到任何相关照片”是漏报。
- read_photo_text / inspect_photo 返回 supported 的观察中直接读出的内容（价格/文字/颜色/年份/电话/店名）
  是照片复核层证据，模型引用这些细节回答不算编造；不要因为检索层只是 partial/candidate
  就要求模型把这些已读出的细节改成“还不能确认”。
- 诚实的不确定性：回答已给出核心事实，同时说明“还不能完全确认/可能是…”，
  不算 certainty_upgrade 或 missing_disclosure。
- 回答不再复述检索过程（不出现 candidate_only/partial_support/候选照片/匹配程度等内部词汇）
  不代表信息缺失，不应因此判 unfaithful。

只输出一个 JSON 对象，不要 markdown、不要多余文字：
{"faithful": true 或 false, "problems": [{"type": "fabrication|contradiction|omission|certainty_upgrade|missing_disclosure", "detail": "具体问题"}], "reason": "一句话理由"}"""

_JUDGE_MESSAGE = {
    "fabrication": "评审认为回答编造了工具观察中没有的事实",
    "contradiction": "评审认为回答与工具观察直接矛盾",
    "omission": "评审认为工具返回了结果但回答却否认/漏报",
    "certainty_upgrade": "评审认为回答把未确认条件说成了确定事实",
    "missing_disclosure": "评审认为回答没有披露部分确认/相似候选的缺口",
}

# 事实性问题（可恢复，不能放行错误答案）；missing_disclosure 只是披露建议
_TRUTH_TYPES = {"fabrication", "contradiction", "omission", "certainty_upgrade"}


def parse_verdict(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    text = raw
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    start = text.find("{")
    if start < 0:
        return None
    depth, end = 0, None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    try:
        return json.loads(text[start:end])
    except Exception:
        return None


def judge_faithfulness(chat_fn, *, query: str, tool_results: list, answer: str,
                       trusted_facts: list[str] | None = None,
                       include_debug: bool = False):
    """返回 (faithful, issues[, debug])。任何异常/输出不可解析都降级为放行。"""
    try:
        obs_lines = []
        for tr in tool_results or []:
            compact = {k: tr.get(k) for k in (
                "tool", "total", "satisfaction", "blocked", "inspect_text", "certainty",
                "operation", "value", "rows", "answer_type", "filters_applied")}
            if compact.get("tool") == "read_photo_text":
                compact["ocr_text"] = (tr.get("ocr_text") or "")[:800]
            if compact.get("tool") and compact["tool"] == "inspect_photo" and compact.get("satisfaction") is None:
                compact.pop("satisfaction", None)
            for k in ("operation", "value", "rows", "answer_type", "filters_applied"):
                if compact.get(k) is None:
                    compact.pop(k, None)
            obs_lines.append("- " + json.dumps(compact, ensure_ascii=False))
        facts_block = ("\n".join(f"- {f}" for f in (trusted_facts or []))
                       or "(从观察中提取)")
        user = (
            f"用户问题：{query}\n"
            f"工具观察：\n" + ("\n".join(obs_lines) or "(无)") + "\n"
            f"可信事实（回答必须与之一致）：\n{facts_block}\n"
            f"模型最终回答：{answer}"
        )
        judge_messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ]
        raw = chat_fn(judge_messages)
        verdict = parse_verdict(raw)
        if not verdict or "faithful" not in verdict:
            if include_debug:
                return True, GuardResult([]), {"prompt": judge_messages, "raw": raw}
            return True, GuardResult([])
        if verdict.get("faithful") is False:
            issues = []
            for p in (verdict.get("problems") or []):
                ptype = str(p.get("type") or "unfaithful")
                detail = str(p.get("detail") or "")
                message = _JUDGE_MESSAGE.get(ptype,
                                             "评审认为回答与工具观察不一致")
                if detail:
                    message = f"{message}：{detail}"
                # 事实性问题 → truth recoverable（先重写，失败再 partial，不放错误答案）；
                # missing_disclosure → style advisory（只建议，不拦事实正确的回答）。
                severity = SEVERITY_TRUTH if ptype in _TRUTH_TYPES else SEVERITY_STYLE
                issues.append(GuardIssue(code=f"judge_{ptype}", message=message,
                                         revision=REVISION_REWRITE_ONLY,
                                         severity=severity,
                                         trusted_facts=list(trusted_facts or [])))
            if not issues:
                issues.append(GuardIssue(code="judge_unfaithful",
                                         message=_JUDGE_MESSAGE["contradiction"],
                                         revision=REVISION_REWRITE_ONLY,
                                         severity=SEVERITY_STYLE,
                                         trusted_facts=list(trusted_facts or [])))
            if include_debug:
                return False, GuardResult(issues), {"prompt": judge_messages, "raw": raw,
                                                     "verdict": verdict}
            return False, GuardResult(issues)
        if include_debug:
            return True, GuardResult([]), {"prompt": judge_messages, "raw": raw,
                                           "verdict": verdict}
        return True, GuardResult([])
    except Exception:
        if include_debug:
            return True, GuardResult([]), {"error": "judge call failed"}
        return True, GuardResult([])
