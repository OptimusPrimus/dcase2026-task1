from __future__ import annotations

from dcase2026_task1.models.base import ModelInput, ModelSkill
from dcase2026_task1.models.qwen import QwenClassificationSkill, QwenModel
from dcase2026_task1.tasks import ClassificationResponse, ClassificationTask


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


def test_split_reasoning_with_explicit_think_tags() -> None:
    reasoning, answer = QwenClassificationSkill._split_reasoning(
        "<think>\nreasoning text\n</think>\n7"
    )

    assert reasoning == "reasoning text"
    assert answer == "7"


def test_split_reasoning_with_opening_think_tag_in_prompt() -> None:
    reasoning, answer = QwenClassificationSkill._split_reasoning(
        "reasoning text\n</think>\n7"
    )

    assert reasoning == "reasoning text"
    assert answer == "7"


def test_classification_task_builds_prompt_from_generic_query() -> None:
    task = ClassificationTask(_candidate_classes())
    skill = QwenClassificationSkill(task)

    model_input = skill.build_input(
        {
            "title": "Street ambience",
            "tags": "outdoor, wet",
            "description": "Traffic and rainfall",
        }
    )

    assert isinstance(model_input, ModelInput)
    assert "Street ambience" in model_input.prompt
    assert "weather -> rain" in model_input.prompt
    assert model_input.audio_path is None


def test_classification_task_parses_prediction() -> None:
    task = ClassificationTask(_candidate_classes())
    skill = QwenClassificationSkill(task)

    result = skill.parse_output("<think>\nchecking\n</think>\n2", {})

    assert result == ClassificationResponse(
        predicted_class_idx=20,
        predicted_class_name="speech",
        raw_response="<think>\nchecking\n</think>\n2",
        parsed_label="2",
        final_response="2",
        reasoning="checking",
    )


class DummySkill(ModelSkill):
    def __init__(self) -> None:
        super().__init__(ClassificationTask([]))

    def build_input(self, query: dict[str, object]) -> ModelInput:
        return ModelInput(prompt=f"prompt:{query['id']}")

    def parse_output(self, raw_response: str, query: dict[str, object]) -> ClassificationResponse:
        return ClassificationResponse(
            predicted_class_idx=int(query["id"]),
            predicted_class_name=raw_response,
            raw_response=raw_response,
        )


def test_predict_batch_uses_generic_task_and_batch_queries() -> None:
    classifier = QwenModel.__new__(QwenModel)
    classifier._generate_raw_response = lambda model_input: f"reply:{model_input.prompt}"

    results = classifier.predict_batch(
        [{"id": 1}, {"id": 2}],
        DummySkill(),
    )

    assert [result.predicted_class_idx for result in results] == [1, 2]
    assert [result.raw_response for result in results] == [
        "reply:prompt:1",
        "reply:prompt:2",
    ]

def test_predict_batch_uses_classification_skill() -> None:
    classifier = QwenModel.__new__(QwenModel)
    classifier._generate_raw_response = lambda model_input: "<think>\n...\n</think>\n1"
    task = ClassificationTask(_candidate_classes())

    results = classifier.predict_batch(
        [{"title": "clip", "tags": "", "description": ""}],
        QwenClassificationSkill(task),
    )

    assert len(results) == 1
    assert results[0].predicted_class_idx == 10
    assert results[0].predicted_class_name == "rain"
