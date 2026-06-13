from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from dcase2026_task1.experiments.generate_class_probabilities import (
    BASE_PREDICTIONS_FILENAME,
    BATCH_DIR_PREFIX,
    CONFIG_FILENAME,
    INPUT_ROWS_FILENAME,
    PREDICTIONS_FILENAME,
    REQUESTS_FILENAME,
    actual_num_batches,
    batch_dir_name,
    build_prediction_rows,
    clone_args_for_completion,
    merge_prediction_rows,
    normalize_raw_response,
    prediction_failed,
    resolve_target_batch_dir,
    run_complete,
    run_prepare,
    split_rows_into_batches,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _args(tmp_path: Path, *, dry_run: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        action="complete",
        dataset="BSD10k",
        dataset_root=None,
        experiment_dir=None,
        model_id="gpt-5.4-mini",
        api_base=None,
        api_key=None,
        max_new_tokens=1024,
        enable_reasoning=False,
        reasoning_effort="medium",
        max_items=None,
        num_batches=1,
        completion_window="24h",
        output_root=str(tmp_path / "outputs"),
        dry_run=dry_run,
    )


def test_prediction_failed_detects_missing_response_and_errors() -> None:
    assert prediction_failed({"status_code": 200, "raw_response": None, "error": None}) is True
    assert prediction_failed({"status_code": 500, "raw_response": "[]", "error": None}) is True
    assert prediction_failed({"status_code": 200, "raw_response": "[]", "error": {"message": "failed"}}) is True
    assert prediction_failed({"status_code": 200, "raw_response": "not json", "error": None}) is True
    assert (
        prediction_failed(
            {
                "status_code": 200,
                "raw_response": '[{"label":"fx-o","probability":0.48},{"label":"other","probability":0.52}]}',
                "error": None,
            }
        )
        is False
    )
    assert (
        prediction_failed(
            {
                "status_code": 200,
                "raw_response": json.dumps(
                    [
                        {"label": "fx-el", "probability": 0.46},
                        {"label": "m-si", "probability": 0.28},
                        {"label": "fx-o", "probability": 0.12},
                        {"label": "m-m", "probability": 0.08},
                        {"label": "is-e", "probability": 0.04},
                        {"label": "other", "probability": 0.02},
                    ]
                ),
                "error": None,
            }
        )
        is False
    )


def test_normalize_raw_response_trims_trailing_json_garbage() -> None:
    assert (
        normalize_raw_response('[{"label":"fx-o","probability":0.48},{"label":"other","probability":0.52}]}')
        == '[{"label": "fx-o", "probability": 0.48}, {"label": "other", "probability": 0.52}]'
    )


def test_build_prediction_rows_stores_normalized_raw_response() -> None:
    rows = build_prediction_rows(
        input_rows_by_custom_id={
            "dataset-index-0": {
                "custom_id": "dataset-index-0",
                "dataset_index": 0,
            }
        },
        raw_output_text=json.dumps(
            {
                "custom_id": "dataset-index-0",
                "id": "batch_req_123",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output_text": '[{"label":"fx-o","probability":0.48},{"label":"other","probability":0.52}]}'
                    },
                },
            }
        ),
    )

    assert rows[0]["raw_response"] == '[{"label": "fx-o", "probability": 0.48}, {"label": "other", "probability": 0.52}]'


def test_actual_num_batches_is_capped_by_num_rows() -> None:
    assert actual_num_batches(4, 2) == 2
    assert actual_num_batches(4, 0) == 1


def test_split_rows_into_batches_distributes_rows_evenly() -> None:
    batches = split_rows_into_batches(
        [{"custom_id": f"dataset-index-{index}", "dataset_index": index} for index in range(5)],
        num_batches=3,
    )

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert batches[0][0]["dataset_index"] == 0
    assert batches[-1][-1]["dataset_index"] == 4


def test_merge_prediction_rows_prefers_retry_rows_and_sorts() -> None:
    merged = merge_prediction_rows(
        [
            {"custom_id": "dataset-index-3", "dataset_index": 3, "raw_response": "old-3"},
            {"custom_id": "dataset-index-1", "dataset_index": 1, "raw_response": "old-1"},
        ],
        [
            {"custom_id": "dataset-index-3", "dataset_index": 3, "raw_response": "new-3"},
            {"custom_id": "dataset-index-2", "dataset_index": 2, "raw_response": "new-2"},
        ],
    )

    assert [row["dataset_index"] for row in merged] == [1, 2, 3]
    assert merged[-1]["raw_response"] == "new-3"


def test_clone_args_for_completion_uses_source_config() -> None:
    cloned = clone_args_for_completion(
        _args(Path("/tmp")),
        {
            "dataset": "BSD2k",
            "dataset_root": "/data/bsd2k",
            "model_id": "gpt-5.4-mini",
            "api_base": "https://example.test/v1",
            "max_new_tokens": 4096,
            "enable_reasoning": True,
            "reasoning_effort": "high",
            "completion_window": "48h",
        },
    )

    assert cloned.dataset == "BSD2k"
    assert cloned.dataset_root == "/data/bsd2k"
    assert cloned.api_base == "https://example.test/v1"
    assert cloned.max_new_tokens == 1024
    assert cloned.enable_reasoning is True
    assert cloned.reasoning_effort == "high"
    assert cloned.completion_window == "48h"


def test_run_complete_creates_retry_folder_with_failed_rows_and_base_predictions(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()

    input_rows = [
        {
            "custom_id": "dataset-index-0",
            "dataset_index": 0,
            "sound_id": 10,
            "source_dataset": "BSD10k",
            "audio_path": "/tmp/0.wav",
            "title": "ok",
            "tags": "",
            "description": "",
            "target_class_idx": 1,
            "target_class": "fx-o",
            "prompt": "prompt-0",
        },
        {
            "custom_id": "dataset-index-1",
            "dataset_index": 1,
            "sound_id": 11,
            "source_dataset": "BSD10k",
            "audio_path": "/tmp/1.wav",
            "title": "retry",
            "tags": "",
            "description": "",
            "target_class_idx": 2,
            "target_class": "fx-h",
            "prompt": "prompt-1",
        },
    ]
    predictions = [
        {**input_rows[0], "status_code": 200, "raw_response": "[{}]", "reasoning": None, "error": None},
        {**input_rows[1], "status_code": 200, "raw_response": None, "reasoning": None, "error": None},
    ]

    (parent_dir / CONFIG_FILENAME).write_text(
        json.dumps(
            {
                "dataset": "BSD10k",
                "dataset_root": "/data/bsd10k",
                "model": "openai",
                "model_id": "gpt-5.4-mini",
                "api_base": None,
                "max_new_tokens": 1024,
                "enable_reasoning": False,
                "reasoning_effort": "medium",
                "completion_window": "24h",
                "dry_run": False,
                "num_items": 2,
                "created_at": "2026-06-11T19:00:00",
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(parent_dir / INPUT_ROWS_FILENAME, input_rows)
    _write_jsonl(parent_dir / PREDICTIONS_FILENAME, predictions)

    completion_dir = run_complete(_args(tmp_path), parent_dir)

    base_predictions = [json.loads(line) for line in (completion_dir / BASE_PREDICTIONS_FILENAME).read_text().splitlines()]
    retry_input_rows = [json.loads(line) for line in (completion_dir / INPUT_ROWS_FILENAME).read_text().splitlines()]
    retry_requests = [json.loads(line) for line in (completion_dir / REQUESTS_FILENAME).read_text().splitlines()]
    completion_config = json.loads((completion_dir / CONFIG_FILENAME).read_text())

    assert len(base_predictions) == 1
    assert base_predictions[0]["custom_id"] == "dataset-index-0"
    assert len(retry_input_rows) == 1
    assert retry_input_rows[0]["custom_id"] == "dataset-index-1"
    assert retry_requests == [
        {
            "custom_id": "dataset-index-1",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": "gpt-5.4-mini",
                "input": "prompt-1",
                "max_output_tokens": 1024,
            },
        }
    ]
    assert completion_config["parent_experiment_dir"] == str(parent_dir)
    assert completion_config["completed_items_from_parent"] == 1
    assert completion_config["failed_items_from_parent"] == 1


def test_run_complete_splits_failed_rows_into_batch_subdirs(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()

    input_rows = [
        {"custom_id": f"dataset-index-{index}", "dataset_index": index, "prompt": f"prompt-{index}"}
        for index in range(3)
    ]
    predictions = [
        {**row, "status_code": 200, "raw_response": None, "reasoning": None, "error": None}
        for row in input_rows
    ]

    (parent_dir / CONFIG_FILENAME).write_text(
        json.dumps(
            {
                "dataset": "BSD10k",
                "dataset_root": "/data/bsd10k",
                "model_id": "gpt-5.4-mini",
                "api_base": None,
                "max_new_tokens": 1024,
                "enable_reasoning": False,
                "reasoning_effort": "medium",
                "completion_window": "24h",
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(parent_dir / INPUT_ROWS_FILENAME, input_rows)
    _write_jsonl(parent_dir / PREDICTIONS_FILENAME, predictions)

    args = _args(tmp_path)
    args.num_batches = 2
    completion_dir = run_complete(args, parent_dir)

    batch_dirs = sorted(path for path in completion_dir.iterdir() if path.is_dir() and path.name.startswith(BATCH_DIR_PREFIX))
    assert [path.name for path in batch_dirs] == [batch_dir_name(0), batch_dir_name(1)]
    assert len((batch_dirs[0] / INPUT_ROWS_FILENAME).read_text().splitlines()) == 2
    assert len((batch_dirs[1] / INPUT_ROWS_FILENAME).read_text().splitlines()) == 1
    assert json.loads((completion_dir / CONFIG_FILENAME).read_text())["num_batches_actual"] == 2


def test_resolve_target_batch_dir_requires_batch_index_for_multi_batch(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    (experiment_dir / batch_dir_name(0)).mkdir()
    (experiment_dir / batch_dir_name(1)).mkdir()

    with pytest.raises(ValueError, match="--batch-index is required"):
        resolve_target_batch_dir(experiment_dir, None)

    assert resolve_target_batch_dir(experiment_dir, 1) == experiment_dir / batch_dir_name(1)


def test_run_prepare_creates_split_batch_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyDataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [
                {
                    "sound_id": index,
                    "source_dataset": "BSD10k",
                    "audio_path": f"/tmp/{index}.wav",
                    "title": f"title-{index}",
                    "tags": "",
                    "description": "",
                    "class_idx": index,
                    "class": "fx-o",
                }
                for index in range(3)
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict[str, object]:
            return self.items[index]

    monkeypatch.setattr(
        "dcase2026_task1.experiments.generate_class_probabilities.BSDDataset",
        DummyDataset,
    )
    monkeypatch.setattr(
        "dcase2026_task1.experiments.generate_class_probabilities.audio_metadata",
        lambda _path: {"duration_sec": 1.0},
    )

    args = _args(tmp_path)
    args.action = "prepare"
    args.dataset_root = "/tmp/dataset"
    args.num_batches = 2
    args.max_items = None
    args.experiment_dir = None

    experiment_dir = run_prepare(args, tmp_path / "prepared")

    batch_dirs = sorted(path for path in experiment_dir.iterdir() if path.is_dir() and path.name.startswith(BATCH_DIR_PREFIX))
    assert [path.name for path in batch_dirs] == [batch_dir_name(0), batch_dir_name(1)]
    assert len((batch_dirs[0] / REQUESTS_FILENAME).read_text().splitlines()) == 2
    assert len((batch_dirs[1] / REQUESTS_FILENAME).read_text().splitlines()) == 1


def test_run_complete_raises_when_nothing_failed(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    (parent_dir / CONFIG_FILENAME).write_text(
        json.dumps(
            {
                "dataset": "BSD10k",
                "dataset_root": "/data/bsd10k",
                "model_id": "gpt-5.4-mini",
                "api_base": None,
                "max_new_tokens": 1024,
                "enable_reasoning": False,
                "reasoning_effort": "medium",
                "completion_window": "24h",
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(parent_dir / INPUT_ROWS_FILENAME, [{"custom_id": "dataset-index-0", "dataset_index": 0, "prompt": "p"}])
    _write_jsonl(
        parent_dir / PREDICTIONS_FILENAME,
        [{"custom_id": "dataset-index-0", "dataset_index": 0, "status_code": 200, "raw_response": "[]", "error": None}],
    )

    with pytest.raises(RuntimeError, match="No failed predictions found"):
        run_complete(_args(tmp_path), parent_dir)
