from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dcase2026_task1.tasks.base import Task, TaskItem


@dataclass(frozen=True)
class ClassificationResponse:
    predicted_class_idx: int | None
    predicted_class_name: str | None
    raw_response: str
    parsed_label: str | None = None
    final_response: str | None = None
    reasoning: str | None = None


class ClassificationTask(Task):
    def __init__(self, candidate_classes: list[dict[str, Any]]) -> None:
        self.candidate_classes = [dict(candidate) for candidate in candidate_classes]

    def class_lines(self) -> list[str]:
        lines: list[str] = []
        for option_index, candidate in enumerate(self.candidate_classes, start=1):
            description_text = candidate["description"] or ""
            lines.append(
                f"{option_index}. "
                f'{candidate["description_top_level"]} -> '
                f'{candidate["description_second_level"]}: '
                f"{description_text}"
            )
        return lines

    def resolve_class_name(self, predicted_class_idx: int | None) -> str | None:
        if predicted_class_idx is None:
            return None
        for candidate in self.candidate_classes:
            if int(candidate["class_idx"]) == predicted_class_idx:
                return str(candidate["class_name"])
        return None

    def normalize_item(self, item: TaskItem) -> dict[str, Any]:
        normalized = super().normalize_item(item)
        normalized.setdefault("title", "")
        normalized.setdefault("tags", "")
        normalized.setdefault("description", "")
        return normalized
