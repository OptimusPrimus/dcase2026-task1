from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PredictionResult:
    predicted_class_idx: int | None
    predicted_class_name: str | None
    raw_response: str
    parsed_label: str | None = None
    final_response: str | None = None
    reasoning: str | None = None


class AudioLanguageModel(ABC):
    @abstractmethod
    def predict(
        self,
        item: dict[str, Any],
        candidate_classes: list[dict[str, Any]],
    ) -> PredictionResult:
        raise NotImplementedError
