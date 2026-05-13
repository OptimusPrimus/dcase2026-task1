from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from typing import Any


TaskItem = Mapping[str, Any]


class Task(ABC):
    def normalize_item(self, item: TaskItem) -> dict[str, Any]:
        return dict(item)

    def normalize_items(self, items: Sequence[TaskItem]) -> list[dict[str, Any]]:
        return [self.normalize_item(item) for item in items]
