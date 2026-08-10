"""Guard 结果类型（Phase C C2）：从"规则名列表"升级为结构化、可自然化的结果。

- GuardIssue：单条问题，带机器码 code、面向恢复的自然 message、恢复策略 revision。
- GuardResult：list[str] 子类，兼容旧调用方（iter/len/== []），同时携带结构化 issues。
"""

from __future__ import annotations

from dataclasses import dataclass, field

REVISION_REWRITE_ONLY = "rewrite_only"      # 只用已有工具事实重写 final
REVISION_TOOL_REFETCH = "tool_refetch"      # 工具结果结构不足，需要补充查询
REVISION_ALTERNATE_TOOL = "alternate_tool"  # 换一个工具
REVISION_PARTIAL = "partial"                # 无法完全完成，给自然 partial
REVISION_HARD_BLOCK = "hard_block"          # 权限/安全，禁止编造


@dataclass
class GuardIssue:
    code: str
    message: str
    revision: str = REVISION_REWRITE_ONLY
    tool_ref: str | None = None
    trusted_facts: list[str] = field(default_factory=list)


class GuardResult(list):
    """codes 列表兼容旧断言，同时携带结构化 issues 与整体 status。"""

    def __init__(self, issues: list[GuardIssue]):
        super().__init__(issue.code for issue in issues)
        self.issues = issues
        self.status = "pass" if not issues else (
            "hard_block" if any(i.revision == REVISION_HARD_BLOCK for i in issues) else
            "recoverable" if any(i.revision in {REVISION_REWRITE_ONLY, REVISION_TOOL_REFETCH,
                                                REVISION_ALTERNATE_TOOL} for i in issues) else
            "partial")

    @property
    def natural_messages(self) -> list[str]:
        return [i.message for i in self.issues]
