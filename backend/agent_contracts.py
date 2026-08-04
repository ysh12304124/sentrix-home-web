"""Typed boundaries for the Agent orchestration layer.

PydanticAI is optional at import time so SQLite-only tests and offline
maintenance commands remain usable. The framework can supply a plan, but the
deterministic validator below always owns permissions and fallback behavior.
"""

from dataclasses import dataclass
import os


ALLOWED_MODES = {"chat", "memory", "feedback", "clarify"}
ALLOWED_TOOLS = {
    "resolve_constraints", "describe_entity", "find_events", "trace_timeline",
    "compare_memories", "suggest_recall", "open_evidence", "request_clarification",
    "record_feedback",
}


try:
    from pydantic import BaseModel, Field

    class TurnPlanModel(BaseModel):
        mode: str = "chat"
        tools: list[str] = Field(default_factory=list)
        show_images: bool = False
        reason: str = ""
except ImportError:  # pragma: no cover - exercised by the dependency-free test environment.
    TurnPlanModel = None


try:
    from pydantic_ai import Agent as PydanticAIAgent
except ImportError:  # pragma: no cover - the production dependency is optional during migration.
    PydanticAIAgent = None


@dataclass(frozen=True)
class PlanValidation:
    mode: str
    tools: tuple[str, ...]
    show_images: bool
    reason: str

    def as_dict(self, planner):
        return {
            "mode": self.mode,
            "tools": list(self.tools),
            "show_images": self.show_images,
            "reason": self.reason,
            "planner": planner,
        }


def _fallback_plan_values(fallback):
    return {
        "mode": fallback.get("mode", "chat"),
        "tools": list(fallback.get("tools", [])),
        "show_images": bool(fallback.get("show_images")),
        "reason": str(fallback.get("reason") or ""),
    }


def validate_turn_plan(parsed, fallback):
    """Validate a model/framework plan without allowing permission escalation."""
    fallback_values = _fallback_plan_values(fallback)
    parsed = parsed if isinstance(parsed, dict) else {}
    mode = parsed.get("mode") if parsed.get("mode") in ALLOWED_MODES else fallback_values["mode"]
    if fallback_values["mode"] in {"memory", "feedback", "clarify"} and mode == "chat":
        mode = fallback_values["mode"]
    tools = [item for item in parsed.get("tools", []) if item in ALLOWED_TOOLS]
    if mode == "chat":
        tools = []
    elif mode == "memory":
        tools = list(dict.fromkeys(["resolve_constraints", *tools]))
        if len(tools) == 1:
            tools.append("find_events")
    elif mode == "feedback":
        tools = ["record_feedback"]
    else:
        tools = list(dict.fromkeys(["resolve_constraints", *tools, "request_clarification"]))
    show_images = bool(parsed.get("show_images")) and mode == "memory"
    if fallback_values["show_images"]:
        show_images = True
    return PlanValidation(
        mode=mode,
        tools=tuple(tools),
        show_images=show_images,
        reason=str(parsed.get("reason") or fallback_values["reason"])[:240],
    )


class PydanticAIPlanner:
    """Optional structured planner; never receives or owns the memory store."""

    def __init__(self, model=None):
        self.model = model or os.getenv("SENTRIX_AGENT_FRAMEWORK_MODEL")
        self._agent = None
        if PydanticAIAgent and TurnPlanModel and self.model:
            try:
                self._agent = PydanticAIAgent(self.model, output_type=TurnPlanModel)
            except Exception:
                self._agent = None

    @property
    def available(self):
        return self._agent is not None

    def plan(self, prompt):
        if not self._agent:
            return None
        try:
            result = self._agent.run_sync(prompt)
            output = getattr(result, "output", result)
            if hasattr(output, "model_dump"):
                return output.model_dump()
            return dict(output) if isinstance(output, dict) else None
        except Exception:
            return None
