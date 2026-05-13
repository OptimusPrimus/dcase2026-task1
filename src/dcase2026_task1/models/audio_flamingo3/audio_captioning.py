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
            "Clip metadata:\n"
            f'- title=\"{normalized["title"]}\"\n'
            f'- tags=\"{normalized["tags"]}\"\n'
            f'- description=\"{normalized["description"]}\"\n\n'
            "Task:\n"
            "Analyze the provided audio recording together with the metadata.\n\n"
            "Describe only the audible content of the recording in a concise and factual way.\n"
            "Focus on audible events, environment, speech, music, ambience, textures, "
            "or notable sounds.\n\n"
            "Do NOT include technical details such as bitrate, sampling rate, "
            "recording device, location, or file metadata.\n\n"
            "Use the following output format exactly:\n\n"
            "DESCRIPTION: <audio content description>\n\n"
            "Rules:\n"
            "- Base the response on both the audio and metadata.\n"
            "- Do not invent details that cannot reasonably be inferred.\n"
            "- Keep the description concise, specific, and factual.\n"
            "- Return only the formatted result.\n"
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
