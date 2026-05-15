from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInput:
    prompt: str
    audio_path: str | None = None


class GenerativeModel(ABC):
    def generate(self, model_input: ModelInput) -> str:
        return self.generate_batch([model_input])[0]

    @abstractmethod
    def generate_batch(
        self,
        model_inputs: list[ModelInput],
    ) -> list[str]:
        raise NotImplementedError
