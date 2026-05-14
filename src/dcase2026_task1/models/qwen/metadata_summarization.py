from __future__ import annotations

import re

from dcase2026_task1.models.base import ModelInput, ModelSkill
from dcase2026_task1.models.qwen.common import split_reasoning
from dcase2026_task1.tasks import (
    MetadataSummarizationResponse,
    MetadataSummarizationTask,
    TaskItem,
)


class QwenMetadataSummarizationSkill(ModelSkill):
    _DETAIL_KEYS = [
        "recording_device",
        "sampling_rate",
        "bitrate",
        "recording_location",
        "dataset_or_project",
        "additional_context",
    ]

    def __init__(self, task: MetadataSummarizationTask) -> None:
        super().__init__(task)
        self.task = task

    def build_input(self, item: TaskItem) -> ModelInput:
        normalized = self.task.normalize_item(item)
        prompt = (
            "Given the following audio clip metadata, write a short summary of the audible content.\n\n"
            "Describe only what is likely heard in the audio.\n\n"
            "Do not mention technical or contextual metadata such as recording device, location, bitrate, "
            "sample rate, file format, duration, timestamps, dataset name, project name, uploader, tags, "
            "IDs, filenames, channels, or other non-audible details.\n\n"
            'Use "unknown" if necessary.\n\n'
            "Return only the summary text.\n\n"
            "[AUDIO CLIP METADATA]\n\n"
            "Title:\n"
            f'{normalized["title"]}\n\n'
            "Tags:\n"
            f'{normalized["tags"]}\n\n'
            "Description:\n"
            f'{normalized["description"]}\n'
        )
        return ModelInput(prompt=prompt)

    def parse_output(
        self,
        raw_response: str,
        item: TaskItem,
    ) -> MetadataSummarizationResponse:
        del item
        reasoning, final_response = self._split_reasoning(raw_response)
        audio_content, metadata_details = self._parse_summary(final_response)
        return MetadataSummarizationResponse(
            audio_content=audio_content,
            metadata_details=metadata_details,
            raw_response=raw_response,
            final_response=final_response,
            reasoning=reasoning,
        )

    @staticmethod
    def _split_reasoning(text: str) -> tuple[str | None, str]:
        return split_reasoning(text)

    def _parse_summary(self, text: str) -> tuple[str, dict[str, str]]:
        audio_content = text.strip() or "unknown"
        metadata_details = {key: "unknown" for key in self._DETAIL_KEYS}
        return audio_content, metadata_details
