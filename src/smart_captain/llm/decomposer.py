"""Task decomposition layer for the restructured LLM interface.

This module currently uses deterministic heuristics so the architecture can be
validated before a real model backend is integrated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from urllib import request


@dataclass
class StructuredTaskDecomposer:
    """Convert a high-level mission instruction into structured subtask payloads."""

    keyword_map: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "导航": {"skill": "navigation"},
        "前往": {"skill": "navigation"},
        "到达": {"skill": "navigation"},
        "避障": {"skill": "obstacle_avoidance"},
        "躲避障碍": {"skill": "obstacle_avoidance"},
        "跟踪": {"skill": "target_tracking"},
        "搜索": {"skill": "search"},
        "搜寻": {"skill": "search"},
        "测绘": {"skill": "mapping"},
        "建图": {"skill": "mapping"},
    })

    default_skill_order: tuple[str, ...] = (
        "navigation",
        "obstacle_avoidance",
        "target_tracking",
        "search",
        "mapping",
    )
    api_url: str | None = field(default_factory=lambda: os.getenv("SMART_CAPTAIN_LLM_API_URL"))
    api_key: str | None = field(default_factory=lambda: os.getenv("SMART_CAPTAIN_LLM_API_KEY"))
    api_model: str = field(default_factory=lambda: os.getenv("SMART_CAPTAIN_LLM_MODEL", "deepseek-chat"))
    last_source: str = "heuristic"

    def decompose(
        self,
        command: str,
        world_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Produce a structured sequence of skill payloads.

        The output format is designed to be consumed directly by
        `TaskGraphPlanner`.
        """
        world_context = world_context or {}
        api_tasks = self._decompose_via_api(command=command, world_context=world_context)
        if api_tasks:
            self.last_source = "api"
            return api_tasks

        self.last_source = "heuristic"
        return self._decompose_heuristic(command=command, world_context=world_context)

    def _decompose_heuristic(
        self,
        command: str,
        world_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Deterministic fallback decomposition."""
        tasks: list[dict[str, Any]] = []
        seen_skills: set[str] = set()

        for keyword, template in self.keyword_map.items():
            if keyword in command:
                skill = template["skill"]
                if skill in seen_skills:
                    continue
                tasks.append(self._build_payload(skill=skill, command=command, world_context=world_context))
                seen_skills.add(skill)

        if not tasks:
            tasks.append(self._build_payload(skill="navigation", command=command, world_context=world_context))

        ordered = sorted(
            tasks,
            key=lambda payload: self.default_skill_order.index(payload["skill"])
            if payload["skill"] in self.default_skill_order
            else len(self.default_skill_order),
        )
        for index, payload in enumerate(ordered):
            payload["id"] = f"task_{index}"
        return ordered

    def _decompose_via_api(
        self,
        command: str,
        world_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Call a configured OpenAI-compatible LLM API, e.g. DeepSeek."""
        if not self.api_url or not self.api_key:
            return []

        body = {
            "model": self.api_model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是AUV任务分解器。"
                        "请把用户指令分解为任务数组，并只返回JSON。"
                        "返回格式必须是 {\"tasks\": [...]}。"
                        "每个任务对象必须包含字段："
                        "skill, scenario, sensor_mode, constraints, success_condition, description。"
                        "允许的skill只有：navigation, obstacle_avoidance, target_tracking, search, mapping。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"command": command, "world_context": world_context},
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            self.api_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            tasks = parsed.get("tasks", [])
            if not isinstance(tasks, list):
                return []
            normalized = []
            for index, task in enumerate(tasks):
                if not isinstance(task, dict) or "skill" not in task:
                    continue
                skill = str(task["skill"])
                normalized.append({
                    "id": task.get("id", f"task_{index}"),
                    "skill": skill,
                    "scenario": task.get("scenario") or self._infer_scenario(skill, command),
                    "sensor_mode": task.get("sensor_mode") or self._infer_sensor_mode(skill, command),
                    "params": {
                        "raw_command": command,
                        "world_context": world_context,
                    },
                    "constraints": task.get("constraints", self._infer_constraints(skill, command)),
                    "success_condition": task.get("success_condition") or self._infer_success_condition(skill),
                    "description": task.get("description") or self._describe_payload(skill, command),
                })
            return normalized
        except Exception:
            return []

    def _build_payload(
        self,
        skill: str,
        command: str,
        world_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a single structured task payload."""
        scenario = self._infer_scenario(skill, command)
        sensor_mode = self._infer_sensor_mode(skill, command)
        return {
            "skill": skill,
            "scenario": scenario,
            "sensor_mode": sensor_mode,
            "params": {
                "raw_command": command,
                "world_context": world_context,
            },
            "constraints": self._infer_constraints(skill, command),
            "success_condition": self._infer_success_condition(skill),
            "description": self._describe_payload(skill, command),
        }

    @staticmethod
    def _infer_scenario(skill: str, command: str) -> str | None:
        if "大坝" in command:
            return "dam"
        if "开阔水域" in command or skill == "obstacle_avoidance":
            return "open_water"
        return "pier_harbor"

    @staticmethod
    def _infer_sensor_mode(skill: str, command: str) -> str | None:
        if "声呐" in command or skill in {"search", "mapping"}:
            return "imaging_sonar"
        if "相机" in command:
            return "rgb_camera"
        return "radar"

    @staticmethod
    def _infer_constraints(skill: str, command: str) -> dict[str, Any]:
        constraints: dict[str, Any] = {}
        if "避障" in command or "躲避障碍" in command:
            constraints["avoid_obstacles"] = True
        if skill == "mapping":
            constraints["high_resolution"] = True
        return constraints

    @staticmethod
    def _infer_success_condition(skill: str) -> str:
        conditions = {
            "navigation": "reach_waypoint",
            "obstacle_avoidance": "path_clear",
            "target_tracking": "target_locked",
            "search": "target_candidate_detected",
            "mapping": "mapping_complete",
        }
        return conditions.get(skill, "task_complete")

    @staticmethod
    def _describe_payload(skill: str, command: str) -> str:
        return f"{skill} derived from command: {command}"
