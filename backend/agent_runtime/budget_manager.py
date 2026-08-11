"""BudgetManager（v2 §22/§23）。

统一管理：model steps / model calls / tool calls / visual inspections /
wall time / final reserve。不再让 Parser/Writer/Verifier 各自持有互不一致的 deadline。
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class BudgetState:
    max_model_steps: int = 3
    max_tool_calls: int = 3
    max_inspections: int = 1
    wall_time_s: float = 60.0
    final_reserve_s: float = 10.0

    model_steps: int = 0
    tool_calls: int = 0
    inspections: int = 0
    _started: float = 0.0

    def start(self):
        self._started = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def wall_remaining(self) -> float:
        return max(0.0, self.wall_time_s - self.elapsed())

    def has_final_reserve(self) -> bool:
        return self.wall_remaining() > self.final_reserve_s

    def can_model_step(self) -> bool:
        return self.model_steps < self.max_model_steps and self.has_final_reserve()

    def can_tool_call(self, *, inspection: bool = False) -> bool:
        if self.tool_calls >= self.max_tool_calls:
            return False
        if inspection and self.inspections >= self.max_inspections:
            return False
        return self.has_final_reserve()

    def record_model_step(self):
        self.model_steps += 1

    def record_tool_call(self, *, inspection: bool = False):
        self.tool_calls += 1
        if inspection:
            self.inspections += 1

    def as_dict(self) -> dict:
        return {
            "model_steps": self.model_steps,
            "tool_calls": self.tool_calls,
            "inspections": self.inspections,
            "elapsed_s": round(self.elapsed(), 2),
            "wall_remaining_s": round(self.wall_remaining(), 2),
        }
