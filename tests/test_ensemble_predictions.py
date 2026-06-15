from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from dcase2026_task1.experiments.ensemble_predictions import (
    PREDICTION_FILENAMES_BY_DATASET,
    average_logits_by_file_id,
    average_logits_for_records,
    build_bsd10k_eval_records,
    default_ensemble_dir,
    evaluate_ensemble,
    evaluate_model_combinations,
    format_combination_rankings,
    label_names_from_config,
    load_logits_npz,
    short_model_name,
    write_ensembled_prediction_files,
)


def _config() -> dict:
    return {
        "labels": [
            {"label_id": 0, "dataset_class_idx": 10, "class_name": "m-si"},
            {"label_id": 1, "dataset_class_idx": 20, "class_name": "fx-a"},
        ],
        "split": {
            "fold": 0,
            "n_splits": 5,
            "validation_size": 0.2,
            "split_seed": 566182,
            "val_size": 2,
            "test_size": 3,
        },
    }


def test_average_logits_for_records_uses_file_ids_and_simple_mean() -> None:
    records = [
        {"sound_id": 1},
        {"sound_id": 2},
    ]
    averaged = average_logits_for_records(
        records,
        [
            {"1": np.array([2.0, 0.0]), "2": np.array([0.0, 2.0])},
            {"1": np.array([4.0, 2.0]), "2": np.array([2.0, 4.0])},
        ],
    )

    assert np.allclose(averaged, np.array([[3.0, 1.0], [1.0, 3.0]]))


def test_load_logits_npz_checks_label_names(tmp_path: Path) -> None:
    path = tmp_path / "bsd10k_logits.npz"
    np.savez(
        path,
        label_names=np.asarray(["m-si", "fx-a"], dtype=np.str_),
        **{"1": np.asarray([1.0, 0.0], dtype=np.float32)},
    )

    logits = load_logits_npz(path, ["m-si", "fx-a"])

    assert list(logits) == ["1"]
    assert np.allclose(logits["1"], np.array([1.0, 0.0]))


def test_average_logits_by_file_id_checks_matching_keys() -> None:
    averaged = average_logits_by_file_id(
        [
            {"a": np.array([1.0, 3.0]), "b": np.array([3.0, 5.0])},
            {"a": np.array([3.0, 5.0]), "b": np.array([5.0, 7.0])},
        ],
        "BSD10k",
    )

    assert list(averaged) == ["a", "b"]
    assert np.allclose(averaged["a"], np.array([2.0, 4.0]))
    assert np.allclose(averaged["b"], np.array([4.0, 6.0]))


def test_default_ensemble_dir_uses_sorted_model_names(tmp_path: Path) -> None:
    output_dir = default_ensemble_dir(
        [
            tmp_path / "20260614_231922_BSD10k_clap_30005d33",
            tmp_path / "20260613_140743_BSD10k_beats_b958ff06",
        ],
        tmp_path / "outputs",
    )

    assert short_model_name(tmp_path / "20260614_231922_BSD10k_clap_30005d33") == "clap_30005d33"
    assert output_dir == tmp_path / "outputs" / "ensemble_beats_b958ff06__clap_30005d33"


def test_write_ensembled_prediction_files_writes_all_dataset_npzs(tmp_path: Path) -> None:
    model_a = tmp_path / "model_a"
    model_b = tmp_path / "model_b"
    label_names = ["m-si", "fx-a"]
    for model_dir, offset in [(model_a, 0.0), (model_b, 2.0)]:
        model_dir.mkdir()
        for filename in PREDICTION_FILENAMES_BY_DATASET.values():
            np.savez(
                model_dir / filename,
                label_names=np.asarray(label_names, dtype=np.str_),
                **{
                    "1": np.asarray([1.0 + offset, 3.0 + offset], dtype=np.float32),
                    "2": np.asarray([3.0 + offset, 5.0 + offset], dtype=np.float32),
                },
            )

    output_paths = write_ensembled_prediction_files(
        model_dirs=[model_a, model_b],
        output_dir=tmp_path / "ensemble_a__b",
        label_names=label_names,
    )

    assert set(output_paths) == {"BSD10k", "BSD35k-CS", "BSD2k"}
    for filename in PREDICTION_FILENAMES_BY_DATASET.values():
        logits = load_logits_npz(tmp_path / "ensemble_a__b" / filename, label_names)
        assert np.allclose(logits["1"], np.array([2.0, 4.0]))
        assert np.allclose(logits["2"], np.array([4.0, 6.0]))


def test_build_bsd10k_eval_records_uses_training_split_and_configured_limits() -> None:
    records = [
        {
            "sound_id": index,
            "class_idx": 10 if index % 2 == 0 else 20,
            "class": "m-si" if index % 2 == 0 else "fx-a",
            "source_dataset": "BSD10k",
        }
        for index in range(20)
    ]

    with patch(
        "dcase2026_task1.experiments.ensemble_predictions.load_records_by_dataset_name",
        return_value=records,
    ):
        val_records, test_records = build_bsd10k_eval_records(
            Path("/tmp/bsd10k"),
            _config()["split"],
        )

    assert len(val_records) == 2
    assert len(test_records) == 3
    assert all(record["source_dataset"] == "BSD10k" for record in val_records + test_records)


def test_evaluate_ensemble_reports_val_and_test_metrics() -> None:
    config = _config()
    val_records = [
        {"sound_id": 1, "class_idx": 10},
        {"sound_id": 2, "class_idx": 20},
    ]
    test_records = [
        {"sound_id": 3, "class_idx": 10},
        {"sound_id": 4, "class_idx": 20},
    ]
    logits_per_model = [
        {
            "1": np.array([4.0, 0.0]),
            "2": np.array([0.0, 4.0]),
            "3": np.array([4.0, 0.0]),
            "4": np.array([0.0, 4.0]),
        },
        {
            "1": np.array([2.0, 1.0]),
            "2": np.array([1.0, 2.0]),
            "3": np.array([2.0, 1.0]),
            "4": np.array([1.0, 2.0]),
        },
    ]

    results = evaluate_ensemble(
        val_records=val_records,
        test_records=test_records,
        logits_per_model=logits_per_model,
        config=config,
    )

    assert results["counts"] == {"val": 2, "test": 2}
    assert results["val"]["accuracy"] == 1.0
    assert results["test"]["accuracy"] == 1.0
    assert label_names_from_config(config) == ["m-si", "fx-a"]


def test_evaluate_model_combinations_reports_and_sorts_all_subsets(tmp_path: Path) -> None:
    config = _config()
    model_dirs = [
        tmp_path / "20260614_000000_BSD10k_a_11111111",
        tmp_path / "20260614_000000_BSD10k_b_22222222",
        tmp_path / "20260614_000000_BSD10k_c_33333333",
    ]
    val_records = [
        {"sound_id": 1, "class_idx": 10},
        {"sound_id": 2, "class_idx": 20},
    ]
    test_records = [
        {"sound_id": 3, "class_idx": 10},
        {"sound_id": 4, "class_idx": 20},
    ]
    logits_per_model = [
        {
            "1": np.array([3.0, 0.0]),
            "2": np.array([2.0, 1.0]),
            "3": np.array([3.0, 0.0]),
            "4": np.array([2.0, 1.0]),
        },
        {
            "1": np.array([1.0, 2.0]),
            "2": np.array([0.0, 3.0]),
            "3": np.array([1.0, 2.0]),
            "4": np.array([0.0, 3.0]),
        },
        {
            "1": np.array([0.0, 6.0]),
            "2": np.array([6.0, 0.0]),
            "3": np.array([0.0, 6.0]),
            "4": np.array([6.0, 0.0]),
        },
    ]

    results = evaluate_model_combinations(
        model_dirs=model_dirs,
        val_records=val_records,
        test_records=test_records,
        logits_per_model=logits_per_model,
        config=config,
    )

    assert len(results) == 7
    assert results[0]["short_model_names"] == ["a_11111111", "b_22222222"]
    assert results[0]["val"]["accuracy"] == 1.0
    assert results[0]["test"]["accuracy"] == 1.0
    assert results[0]["validation_score"] == 1.0
    assert results[0]["test_score"] == 1.0
    assert results[0]["combined_score"] == 2.0
    assert all("val" in result and "test" in result for result in results)


def test_format_combination_rankings_prints_three_top_lists() -> None:
    results = {
        "combinations": [
            {
                "short_model_names": ["a"],
                "size": 1,
                "validation_score": 0.7,
                "test_score": 0.6,
                "combined_score": 1.3,
            },
            {
                "short_model_names": ["b"],
                "size": 1,
                "validation_score": 0.6,
                "test_score": 0.9,
                "combined_score": 1.5,
            },
        ]
    }

    output = format_combination_rankings(results)

    assert "Top 5 systems by validation score" in output
    assert "Top 5 systems by test score" in output
    assert "Top 5 systems by combined validation+test score" in output
    assert "rank  val_score  test_score  combined  size  models" in output
    assert output.count("\n   1  ") == 3
