"""Guard 结果类型（Phase C C2 / Phase G G4）——结构化、可自然化、分严重级。

- GuardIssue：单条问题，带机器码 code、面向恢复的自然 message、恢复策略 revision、
  严重级 severity（G4 三层：hard_block / truth / style）。
- GuardResult：list[str] 子类，兼容旧调用方（iter/len/== []），同时携带结构化 issues 与整体 severity。
"""

from __future__ import annotations

from dataclasses import dataclass, field

REVISION_REWRITE_ONLY = "rewrite_only"      # 只用已有工具事实重写 final
REVISION_TOOL_REFETCH = "tool_refetch"      # 工具结果结构不足，需要补充查询
REVISION_ALTERNATE_TOOL = "alternate_tool"  # 换一个工具
REVISION_PARTIAL = "partial"                # 无法完全完成，给自然 partial
REVISION_HARD_BLOCK = "hard_block"          # 权限/安全，禁止编造

# Phase G G4：Guard 三层严重级
SEVERITY_HARD_BLOCK = "hard_block"  # 权限/隐私/内部泄漏/非法写：不可放行，不可修复
SEVERITY_TRUTH = "truth"            # 确定性事实冲突：先 rewrite-only，失败再 partial，不放错误答案
SEVERITY_STYLE = "style"            # 表达/披露问题：只能建议重写，绝不得 block 事实正确的答案

# 分层判定：style 只做建议；truth 可恢复；hard_block 直接终止
STYLE_CODES = {
    "missing_disclosure", "hedge_overuse", "not_direct", "too_verbose", "poor_tone",
    "judge_missing_disclosure", "judge_not_direct", "judge_too_verbose",
}
HARD_BLOCK_CODES = {
    "internal_id_leak", "table_name_leak", "scope_escape", "viewer_escape",
    "write_not_allowed", "forbidden_content",
}


def severity_for_code(code: str) -> str:
    """G4：按码确定严重级。未知码默认 truth（可恢复），保证新规则不误放行事实错误。"""
    if code in HARD_BLOCK_CODES:
        return SEVERITY_HARD_BLOCK
    if code in STYLE_CODES:
        return SEVERITY_STYLE
    return SEVERITY_TRUTH


@dataclass
class GuardIssue:
    code: str
    message: str
    revision: str = REVISION_REWRITE_ONLY
    severity: str = SEVERITY_TRUTH
    tool_ref: str | None = None
    trusted_facts: list[str] = field(default_factory=list)

    def __post_init__(self):
        # 显式未指定 severity 时按码推导，避免调用方漏标
        if self.severity == SEVERITY_TRUTH:
            self.severity = severity_for_code(self.code)


class GuardResult(list):
    """codes 列表兼容旧断言，同时携带结构化 issues 与整体 severity/status。"""

    def __init__(self, issues: list[GuardIssue]):
        super().__init__(issue.code for issue in issues)
        self.issues = issues
        self.severity = "pass" if not issues else (
            SEVERITY_HARD_BLOCK if any(i.severity == SEVERITY_HARD_BLOCK for i in issues) else
            SEVERITY_TRUTH if any(i.severity == SEVERITY_TRUTH for i in issues) else
            SEVERITY_STYLE)
        # legacy status：hard_block / recoverable / advisory / pass
        self.status = "pass" if not issues else (
            "hard_block" if self.severity == SEVERITY_HARD_BLOCK else
            "recoverable" if self.severity == SEVERITY_TRUTH else
            "advisory")

    @property
    def natural_messages(self) -> list[str]:
        return [i.message for i in self.issues]
