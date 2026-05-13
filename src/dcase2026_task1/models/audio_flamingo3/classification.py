from __future__ import annotations

import re

from dcase2026_task1.models.base import ModelInput, ModelSkill
from dcase2026_task1.tasks import ClassificationResponse, ClassificationTask, TaskItem


class AudioFlamingo3ClassificationSkill(ModelSkill):
    def __init__(self, task: ClassificationTask) -> None:
        super().__init__(task)
        self.task = task

    def build_input(self, item: TaskItem) -> ModelInput:
        normalized = self.task.normalize_item(item)
        prompt = (
            "You are classifying an audio clip into one of the known dataset classes.\n"
            "Use the audio together with the provided metadata.\n"
            "Choose exactly one option from the list below.\n"
            "Output only the option index.\n"
            "Choose only from these classes:\n"
            f"{chr(10).join(self.task.class_lines())}\n\n"
            "Clip metadata:\n"
            f'- title="{normalized["title"]}"\n'
            f'- tags="{normalized["tags"]}"\n'
            f'- description="{normalized["description"]}"\n'
        )
        audio_path = normalized.get("audio_path")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(
                "AudioFlamingo3ClassificationSkill requires item['audio_path']."
            )
        return ModelInput(prompt=prompt, audio_path=audio_path)

    def parse_output(
        self,
        raw_response: str,
        item: TaskItem,
    ) -> ClassificationResponse:
        del item
        predicted_class_idx, parsed_label = self._parse_prediction(raw_response)
        return ClassificationResponse(
            predicted_class_idx=predicted_class_idx,
            predicted_class_name=self.task.resolve_class_name(predicted_class_idx),
            raw_response=raw_response,
            parsed_label=parsed_label,
        )

    def _parse_prediction(self, raw_response: str) -> tuple[int | None, str | None]:
        option_match = re.search(r"\b(\d+)\b", raw_response)
        if option_match:
            option_index = int(option_match.group(1))
            if 1 <= option_index <= len(self.task.candidate_classes):
                candidate = self.task.candidate_classes[option_index - 1]
                return int(candidate["class_idx"]), str(option_index)
        return None, raw_response.strip() or None
