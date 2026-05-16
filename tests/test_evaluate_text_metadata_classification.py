from __future__ import annotations

import pytest

from dcase2026_task1.experiments.evaluate_text_metadata_classification import (
    aggregate_prediction_rows,
    collect_labels,
    majority_vote_labels,
)


def _candidate_map() -> dict[str, str]:
    return {
        "1": "sp-s",
        "2": "fx-h",
        "sp-s": "sp-s",
        "fx-h": "fx-h",
    }


def test_collect_labels_keeps_speech_prediction_when_audioset_speech_is_above_threshold() -> None:
    rows = [
        {
            "audio_path": "/tmp/example.wav",
            "target_class": "sp-s",
            "parsed_label": "1",
        }
    ]

    y_true, y_pred, parse_failures = collect_labels(
        rows,
        _candidate_map(),
        {"/tmp/example.wav": 0.31},
    )

    assert y_true == ["sp-s"]
    assert y_pred == ["sp-s"]
    assert parse_failures == 0


def test_collect_labels_rewrites_speech_prediction_when_audioset_speech_is_not_above_threshold() -> None:
    rows = [
        {
            "audio_path": "/tmp/example.wav",
            "target_class": "sp-s",
            "parsed_label": "1",
        }
    ]

    _, y_pred, _ = collect_labels(
        rows,
        _candidate_map(),
        {"/tmp/example.wav": 0.30},
    )

    assert y_pred == ["fx-h"]


def test_majority_vote_labels_returns_most_common_label() -> None:
    assert majority_vote_labels(["sp-s", "fx-h", "sp-s"]) == "sp-s"


def test_majority_vote_labels_breaks_ties_by_first_seen_label() -> None:
    assert majority_vote_labels(["fx-h", "sp-s", "sp-s", "fx-h"]) == "fx-h"


def test_aggregate_prediction_rows_uses_majority_vote_on_predictions() -> None:
    rows_per_prediction_file = [
        [
            {
                "audio_path": "/tmp/example.wav",
                "target_class": "sp-s",
                "parsed_label": "1",
            }
        ],
        [
            {
                "audio_path": "/tmp/example.wav",
                "target_class": "sp-s",
                "parsed_label": "2",
            }
        ],
        [
            {
                "audio_path": "/tmp/example.wav",
                "target_class": "sp-s",
                "parsed_label": "1",
            }
        ],
    ]

    aggregated_rows = aggregate_prediction_rows(rows_per_prediction_file, _candidate_map())

    assert len(aggregated_rows) == 1
    assert aggregated_rows[0]["parsed_label"] == "sp-s"


def test_aggregate_prediction_rows_rejects_misaligned_files() -> None:
    rows_per_prediction_file = [
        [
            {
                "audio_path": "/tmp/example.wav",
                "target_class": "sp-s",
                "parsed_label": "1",
            }
        ],
        [
            {
                "audio_path": "/tmp/other.wav",
                "target_class": "sp-s",
                "parsed_label": "1",
            }
        ],
    ]

    with pytest.raises(ValueError, match="same items in the same order"):
        aggregate_prediction_rows(rows_per_prediction_file, _candidate_map())
