from __future__ import annotations

from dataclasses import dataclass

from dcase2026_task1.tasks.base import Task, TaskItem


@dataclass(frozen=True)
class MetadataSummarizationResponse:
    audio_content: str
    metadata_details: dict[str, str]
    raw_response: str
    final_response: str | None = None
    reasoning: str | None = None


class MetadataSummarizationTask(Task):
    def normalize_item(self, item: TaskItem) -> dict[str, object]:
        normalized = super().normalize_item(item)
        normalized.setdefault("title", "")
        normalized.setdefault("tags", "")
        normalized.setdefault("description", "")
        return normalized
