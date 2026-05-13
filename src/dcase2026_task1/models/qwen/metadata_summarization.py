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
            "Given the following audio clip metadata, separate the information into two sections.\n\n"
            "Section 1 should describe only the audible content of the recording.\n\n"
            "Section 2 should contain contextual or technical metadata, such as recording device, "
            "recording location, bitrate, sampling rate, dataset name, project name, or other non-audio details.\n\n"
            "If you use reasoning, keep it very short and focused on the key evidence only.\n\n"
            "Do not invent information that is not explicitly provided or cannot reasonably be inferred. "
            'Use "unknown" when necessary.\n\n'
            "Use the following output format exactly:\n\n"
            "AUDIO_CONTENT:\n"
            "<description of audible content or unknown>\n\n"
            "METADATA_DETAILS:\n"
            "- recording_device: <value or unknown>\n"
            "- sampling_rate: <value or unknown>\n"
            "- bitrate: <value or unknown>\n"
            "- recording_location: <value or unknown>\n"
            "- dataset_or_project: <value or unknown>\n"
            "- additional_context: <value or unknown>\n\n"
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
        audio_content = "unknown"
        section_match = re.search(
            r"AUDIO_CONTENT:\s*(.*?)\s*METADATA_DETAILS:",
            text,
            flags=re.DOTALL,
        )
        if section_match:
            extracted = section_match.group(1).strip()
            if extracted:
                audio_content = extracted
        elif text.strip():
            audio_content = text.strip()

        metadata_details = {key: "unknown" for key in self._DETAIL_KEYS}
        for key in self._DETAIL_KEYS:
            line_match = re.search(
                rf"-\s*{re.escape(key)}:\s*(.*)",
                text,
                flags=re.IGNORECASE,
            )
            if line_match:
                value = line_match.group(1).strip()
                metadata_details[key] = value or "unknown"

        return audio_content, metadata_details
