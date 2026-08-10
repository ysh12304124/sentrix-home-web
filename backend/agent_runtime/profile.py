"""Agent Profile 收敛（v2 §18/§24/§32）。

目标：把散落的 feature flags 收敛为少数 ``SENTRIX_AGENT_PROFILE``。
本模块只做解析与查询，不改变 thin_agent 现有行为。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    tools: tuple = ()
    max_model_steps: int = 3
    max_tool_calls: int = 3
    max_inspections: int = 1
    wall_time_s: float = 60.0
    final_reserve_s: float = 10.0
    features: dict = field(default_factory=dict)


PROFILES = {
    "pipeline": ProfileConfig(
        name="pipeline",
        tools=(),
        max_model_steps=1,
        max_tool_calls=0,
        wall_time_s=20.0,
        features={"rx": False, "tool_loop": False, "conversation_store": True},
    ),
    "tool_loop_shadow": ProfileConfig(
        name="tool_loop_shadow",
        tools=("query_memory_facts", "search_memories", "get_original_photos", "inspect_photo"),
        max_model_steps=4,
        max_tool_calls=4,
        max_inspections=1,
        wall_time_s=60.0,
        final_reserve_s=10.0,
        features={"rx": True, "tool_loop": True, "conversation_store": True},
    ),
    "tool_loop": ProfileConfig(
        name="tool_loop",
        tools=("query_memory_facts", "search_memories", "get_original_photos", "inspect_photo"),
        max_model_steps=5,
        max_tool_calls=5,
        max_inspections=1,
        wall_time_s=60.0,
        final_reserve_s=10.0,
        features={"rx": True, "tool_loop": True, "conversation_store": True},
    ),
}


def active_profile() -> str:
    return os.getenv("SENTRIX_AGENT_PROFILE", "pipeline").strip().lower()


def get_profile(name: str | None = None) -> ProfileConfig:
    return PROFILES.get(name or active_profile(), PROFILES["pipeline"])


def tool_loop_active() -> bool:
    return get_profile().features.get("tool_loop", False)


def tool_enabled(tool_name: str) -> bool:
    return tool_name in get_profile().tools
