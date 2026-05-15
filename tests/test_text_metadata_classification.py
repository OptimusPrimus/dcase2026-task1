from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from dcase2026_task1.experiments.text_metadata_classification import (
    CandidateClass,
    build_prompt,
    parse_prediction,
)
from dcase2026_task1.models.base import ModelInput, ModelOutput
from dcase2026_task1.models.openai.base import OpenAIModel


def _candidate_classes() -> list[CandidateClass]:
    return [
        CandidateClass(
            class_idx=10,
            class_name="rain",
            description_top_level="weather",
            description_second_level="rain",
            description="steady rainfall",
            examples="drizzle, storm",
        ),
        CandidateClass(
            class_idx=20,
            class_name="speech",
            description_top_level="human",
            description_second_level="speech",
            description="spoken voice",
            examples="monologue",
        ),
    ]


def test_build_prompt_includes_metadata_and_candidate_classes() -> None:
    with patch(
        "dcase2026_task1.experiments.text_metadata_classification.audio_metadata",
        return_value={"duration_sec": 3.5},
    ):
        prompt = build_prompt(
            {
                "title": "Street ambience",
                "tags": "outdoor, wet",
                "description": "Traffic and rainfall",
                "audio_path": str(Path("/tmp/example.wav")),
            },
            _candidate_classes(),
        )

    assert "weather -> rain" in prompt
    assert 'title="Street ambience"' in prompt
    assert 'tags="outdoor, wet"' in prompt
    assert "- duration=3.5 sec" in prompt


def test_parse_prediction_resolves_index() -> None:
    predicted_class_idx, parsed_label = parse_prediction("2", _candidate_classes())

    assert predicted_class_idx == 20
    assert parsed_label == "2"


def test_openai_model_uses_responses_api_and_extracts_reasoning_summary() -> None:
    class SummaryPart:
        def __init__(self, text: str) -> None:
            self.text = text

    class OutputItem:
        def __init__(self, item_type: str, summary: list[SummaryPart] | None = None) -> None:
            self.type = item_type
            self.summary = summary

    class Response:
        output_text = "fx-o"
        output = [OutputItem("reasoning", [SummaryPart("Picked object class from metadata.")])]

    class ResponsesClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> Response:
            self.calls.append(kwargs)
            return Response()

    class Client:
        def __init__(self) -> None:
            self.responses = ResponsesClient()

    model = OpenAIModel.__new__(OpenAIModel)
    model.model_id = "gpt-5-mini"
    model.max_new_tokens = 128
    model.temperature = 0.2
    model.top_p = 0.85
    model.enable_reasoning = True
    model.reasoning_effort = "medium"
    model._client = Client()

    output = model.generate_batch_outputs([ModelInput(prompt="classify this")])[0]

    assert output == ModelOutput(text="fx-o", reasoning_summary="Picked object class from metadata.")
    assert model._client.responses.calls == [
        {
            "model": "gpt-5-mini",
            "input": "classify this",
            "max_output_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.85,
            "reasoning": {"effort": "medium", "summary": "auto"},
        }
    ]
