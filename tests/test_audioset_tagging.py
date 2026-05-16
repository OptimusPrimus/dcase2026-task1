from __future__ import annotations

from types import SimpleNamespace

from dcase2026_task1.models.audioset_tagging.base import AudioSetTaggingModel
from dcase2026_task1.models.base import AudioTaggingInput


def test_predict_batch_outputs_returns_probability_for_every_label() -> None:
    class FakeTensor:
        def __init__(self, values: object) -> None:
            self.values = values

        def __iter__(self):
            for value in self.values:
                yield FakeTensor(value)

        def to(self, _device: object) -> "FakeTensor":
            return self

        def cpu(self) -> "FakeTensor":
            return self

        def tolist(self) -> object:
            return self.values

    class FakeTorch:
        float16 = object()
        bfloat16 = object()
        float32 = object()

        @staticmethod
        def no_grad() -> SimpleNamespace:
            class _Context:
                def __enter__(self) -> None:
                    return None

                def __exit__(self, exc_type, exc, tb) -> bool:
                    return False

            return _Context()

        @staticmethod
        def sigmoid(logits: FakeTensor) -> FakeTensor:
            assert logits.values == [[0.1, 0.9], [0.7, 0.3]]
            return FakeTensor([[0.1, 0.9], [0.7, 0.3]])

    class FakeFeatureExtractor:
        sampling_rate = 16000

        def __call__(self, audio_arrays: list[object], sampling_rate: int, return_tensors: str, padding: bool) -> dict[str, FakeTensor]:
            assert len(audio_arrays) == 2
            assert sampling_rate == 16000
            assert return_tensors == "pt"
            assert padding is True
            return {"input_values": FakeTensor(audio_arrays)}

    class FakeModel:
        config = SimpleNamespace(id2label={0: "Speech", 1: "Music"})

        def __call__(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(logits=FakeTensor([[0.1, 0.9], [0.7, 0.3]]))

    model = AudioSetTaggingModel.__new__(AudioSetTaggingModel)
    model.model_id = "fake/audioset"
    model._torch = FakeTorch()
    model._device = "cpu"
    model._feature_extractor = FakeFeatureExtractor()
    model._model = FakeModel()
    model._load_audio_array = lambda audio_path: [audio_path]  # type: ignore[method-assign]

    outputs = model.predict_batch_outputs(
        [
            AudioTaggingInput(audio_path="/tmp/a.wav"),
            AudioTaggingInput(audio_path="/tmp/b.wav"),
        ]
    )

    assert [score.label for score in outputs[0].scores] == ["Speech", "Music"]
    assert [score.score for score in outputs[0].scores] == [0.1, 0.9]
    assert [score.score for score in outputs[1].scores] == [0.7, 0.3]


def test_resolve_dtype_rejects_unknown_values() -> None:
    model = AudioSetTaggingModel.__new__(AudioSetTaggingModel)
    model._torch = SimpleNamespace(float16=object(), bfloat16=object(), float32=object())

    try:
        model._resolve_dtype("int8")
    except ValueError as exc:
        assert "Unsupported torch_dtype=int8" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported dtype.")
