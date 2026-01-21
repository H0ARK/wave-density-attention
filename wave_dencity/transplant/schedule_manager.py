from __future__ import annotations

from typing import Any


def _stage_end(stage: dict[str, Any]) -> float:
    if "end_step" in stage:
        return float(stage["end_step"])
    if "steps" in stage and "start_step" in stage:
        return float(stage["start_step"]) + float(stage["steps"])
    return float("inf")


class ScheduleManager:
    def __init__(self, stages: list[dict[str, Any]] | None):
        self.stages = stages or []

    def get_overrides(self, step: float) -> dict[str, Any] | None:
        for stage in self.stages:
            start = float(stage.get("start_step", 0.0))
            end = _stage_end(stage)
            if start <= step <= end:
                overrides = dict(stage)
                overrides.pop("name", None)
                overrides.pop("start_step", None)
                overrides.pop("end_step", None)
                overrides.pop("steps", None)
                return overrides
        return None
