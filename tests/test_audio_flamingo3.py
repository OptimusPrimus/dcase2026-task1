from __future__ import annotations

import pytest

from dcase2026_task1.models.audio_flamingo3 import (
    AudioFlamingo3AudioCaptioningSkill,
    AudioFlamingo3ClassificationSkill,
)
from dcase2026_task1.tasks import (
    AudioCaptioningResponse,
    AudioCaptioningTask,
    ClassificationResponse,
    ClassificationTask,
)


def _candidate_classes() -> list[dict[str, object]]:
    return [
        {
            "class_idx": 10,
            "class_name": "rain",
            "description_top_level": "weather",
            "description_second_level": "rain",
            "description": "steady rainfall",
        },
        {
            "class_idx": 20,
            "class_name": "speech",
            "description_top_level": "human",
            "description_second_level": "speech",
            "description": "spoken voice",
        },
    ]


def test_classification_task_builds_audio_input() -> None:
    task = ClassificationTask(_candidate_classes())
    skill = AudioFlamingo3ClassificationSkill(task)

    model_input = skill.build_input(
        {
            "title": "Street ambience",
            "tags": "outdoor, wet",
            "description": "Traffic and rainfall",
            "audio_path": "/tmp/example.wav",
        }
    )

    assert model_input.audio_path == "/tmp/example.wav"
    assert "Use the audio together with the provided metadata." in model_input.prompt


def test_classification_task_requires_audio_path() -> None:
    task = ClassificationTask(_candidate_classes())
    skill = AudioFlamingo3ClassificationSkill(task)

    with pytest.raises(ValueError, match="audio_path"):
        skill.build_input({"title": "missing audio"})


def test_classification_task_parses_prediction() -> None:
    task = ClassificationTask(_candidate_classes())
    skill = AudioFlamingo3ClassificationSkill(task)

    result = skill.parse_output("2", {})

    assert result == ClassificationResponse(
        predicted_class_idx=20,
        predicted_class_name="speech",
        raw_response="2",
        parsed_label="2",
    )


def test_audio_captioning_skill_builds_audio_input_with_metadata() -> None:
    task = AudioCaptioningTask()
    skill = AudioFlamingo3AudioCaptioningSkill(task)

    model_input = skill.build_input(
        {
            "title": "Street ambience",
            "tags": "outdoor, wet",
            "description": "Traffic and rainfall",
            "audio_path": "/tmp/example.wav",
        }
    )

    assert model_input.audio_path == "/tmp/example.wav"
    assert 'title="Street ambience"' in model_input.prompt
    assert 'tags="outdoor, wet"' in model_input.prompt
    assert 'description="Traffic and rainfall"' in model_input.prompt
    assert "no longer than two seconds" in model_input.prompt


def test_audio_captioning_skill_requires_audio_path() -> None:
    task = AudioCaptioningTask()
    skill = AudioFlamingo3AudioCaptioningSkill(task)

    with pytest.raises(ValueError, match="audio_path"):
        skill.build_input({"title": "missing audio"})


def test_audio_captioning_skill_parses_caption() -> None:
    task = AudioCaptioningTask()
    skill = AudioFlamingo3AudioCaptioningSkill(task)

    result = skill.parse_output("<think>\nlisten\n</think>\nSteady rain on a street.", {})

    assert result == AudioCaptioningResponse(
        caption="Steady rain on a street.",
        raw_response="<think>\nlisten\n</think>\nSteady rain on a street.",
        final_response="Steady rain on a street.",
        reasoning="listen",
    )
