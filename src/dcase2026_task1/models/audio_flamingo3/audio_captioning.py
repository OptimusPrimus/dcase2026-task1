from __future__ import annotations

import re

from dcase2026_task1.models.base import ModelInput, ModelSkill
from dcase2026_task1.tasks import (
    AudioCaptioningResponse,
    AudioCaptioningTask,
    TaskItem,
)


class AudioFlamingo3AudioCaptioningSkill(ModelSkill):
    def __init__(self, task: AudioCaptioningTask) -> None:
        super().__init__(task)
        self.task = task

    def build_input(self, item: TaskItem) -> ModelInput:
        normalized = self.task.normalize_item(item)
        prompt = (
            "Analyze the provided audio recording together with the metadata.\n\n"
            "First, describe the actual audio content in a concise but specific way, "
            "including audible events, environment, speech, music, ambience, or notable sounds.\n\n"
            "Second, provide technical and contextual metadata inferred from the audio "
            "and supplied metadata where possible, including:\n"
            "- recording device or microphone type\n"
            "- bitrate\n"
            "- sampling rate\n"
            "- recording location\n"
            "- sound project or collection name\n"
            "- audio quality characteristics\n\n"
            "Use the following explicit output format exactly:\n\n"
            "DESCRIPTION: <content description>\n"
            "TECHNICAL_DETAILS:\n"
            "- device: <device or unknown>\n"
            "- bitrate: <value or unknown>\n"
            "- sample_rate: <value or unknown>\n"
            "- location: <location or unknown>\n"
            "- sound_project: <project name or unknown>\n"
            "- audio_quality: <quality notes>\n\n"
            "Rules:\n"
            "- Base the response on both the audio and metadata.\n"
            "- Do not invent details that cannot reasonably be inferred.\n"
            "- Use 'unknown' when information is unavailable.\n"
            "- Keep the DESCRIPTION concise and factual.\n"
            "- Return only the formatted result.\n\n"
            "Clip metadata:\n"
            f'- title=\"{normalized["title"]}\"\n'
            f'- tags=\"{normalized["tags"]}\"\n'
            f'- description=\"{normalized["description"]}\"\n'
        )
        audio_path = normalized.get("audio_path")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(
                "AudioFlamingo3AudioCaptioningSkill requires item['audio_path']."
            )
        return ModelInput(prompt=prompt, audio_path=audio_path)

    def parse_output(
        self,
        raw_response: str,
        item: TaskItem,
    ) -> AudioCaptioningResponse:
        del item
        reasoning, final_response = self._split_reasoning(raw_response)
        caption = final_response.strip()
        return AudioCaptioningResponse(
            caption=caption,
            raw_response=raw_response,
            final_response=final_response,
            reasoning=reasoning,
        )

    @staticmethod
    def _split_reasoning(text: str) -> tuple[str | None, str]:
        if "</think>" in text and "<think>" not in text:
            text = f"<think>{text}"
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else None
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return reasoning, answer
