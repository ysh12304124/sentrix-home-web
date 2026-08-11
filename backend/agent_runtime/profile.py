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
    "tool_loop_shadow": ProfileConfig(
        name="tool_loop_shadow",
        tools=("query_memory_facts", "search_memories", "get_original_photos", "get_result_page",
               "inspect_photo", "search_conversation_history"),
        max_model_steps=6,
        max_tool_calls=4,
        max_inspections=1,
        wall_time_s=60.0,
        final_reserve_s=10.0,
        features={"rx": True, "tool_loop": True, "conversation_store": True},
    ),
    "tool_loop": ProfileConfig(
        name="tool_loop",
        tools=("query_memory_facts", "search_memories", "get_original_photos", "get_result_page",
               "inspect_photo", "search_conversation_history"),
        max_model_steps=6,
        max_tool_calls=5,
        max_inspections=1,
        wall_time_s=60.0,
        final_reserve_s=10.0,
        features={"rx": True, "tool_loop": True, "conversation_store": True},
    ),
    "photo_inspector": ProfileConfig(
        name="photo_inspector",
        tools=("inspect_photo", "get_original_photos", "search_memories", "get_result_page"),
        max_model_steps=5,
        max_tool_calls=4,
        max_inspections=2,
        wall_time_s=60.0,
        final_reserve_s=8.0,
        features={"rx": True, "tool_loop": True, "photo_inspector": True,
                  "conversation_store": True},
    ),
}


def active_profile() -> str:
    return os.getenv("SENTRIX_AGENT_PROFILE", "tool_loop").strip().lower()


def get_profile(name: str | None = None) -> ProfileConfig:
    return PROFILES.get(name or active_profile(), PROFILES["tool_loop"])
