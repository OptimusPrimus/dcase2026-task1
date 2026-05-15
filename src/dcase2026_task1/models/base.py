from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInput:
    prompt: str
    audio_path: str | None = None


@dataclass(frozen=True)
class ModelOutput:
    text: str
    reasoning_summary: str | None = None


@dataclass(frozen=True)
class AudioTaggingInput:
    audio_path: str


@dataclass(frozen=True)
class AudioTagScore:
    index: int
    label: str
    score: float


@dataclass(frozen=True)
class AudioTaggingOutput:
    scores: list[AudioTagScore]


class GenerativeModel(ABC):
    def generate(self, model_input: ModelInput) -> str:
        return self.generate_batch([model_input])[0]

    def generate_batch_outputs(
        self,
        model_inputs: list[ModelInput],
    ) -> list[ModelOutput]:
        return [ModelOutput(text=text) for text in self.generate_batch(model_inputs)]

    @abstractmethod
    def generate_batch(
        self,
        model_inputs: list[ModelInput],
    ) -> list[str]:
        raise NotImplementedError


class AudioTaggingModel(ABC):
    def predict(self, model_input: AudioTaggingInput) -> AudioTaggingOutput:
        return self.predict_batch_outputs([model_input])[0]

    @abstractmethod
    def predict_batch_outputs(
        self,
        model_inputs: list[AudioTaggingInput],
    ) -> list[AudioTaggingOutput]:
        raise NotImplementedError
