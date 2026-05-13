from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from dcase2026_task1.tasks import Task, TaskItem


@dataclass(frozen=True)
class ModelInput:
    prompt: str
    audio_path: str | None = None


class ModelSkill(ABC):
    def __init__(self, task: Task) -> None:
        self.task = task

    @abstractmethod
    def build_input(self, item: TaskItem) -> ModelInput:
        raise NotImplementedError

    def build_inputs(self, items: list[TaskItem]) -> list[ModelInput]:
        return [self.build_input(item) for item in items]

    @abstractmethod
    def parse_output(
        self,
        raw_response: str,
        item: TaskItem,
    ) -> Any:
        raise NotImplementedError

    def parse_outputs(
        self,
        raw_responses: list[str],
        items: list[TaskItem],
    ) -> list[Any]:
        return [
            self.parse_output(raw_response=raw_response, item=item)
            for raw_response, item in zip(raw_responses, items, strict=True)
        ]


class AudioLanguageModel(ABC):
    def predict(
        self,
        item: TaskItem,
        skill: ModelSkill,
    ) -> Any:
        return self.predict_batch([item], skill)[0]

    @abstractmethod
    def predict_batch(
        self,
        items: list[TaskItem],
        skill: ModelSkill,
    ) -> list[Any]:
        raise NotImplementedError
