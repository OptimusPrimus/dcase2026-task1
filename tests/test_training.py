from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from dcase2026_task1.data.splits import (
    DEFAULT_BSD_SPLIT_SEED,
    build_experiment_split,
    get_experiment_records,
    load_records_by_dataset_name,
)
from dcase2026_task1.experiments.training import (
    DEFAULT_EMBEDDING_MODEL,
    WaveformClassificationDataset,
    WaveformInferenceDataset,
    apply_hard_llm_constraints,
    apply_soft_llm_constraints,
    build_class_frequency_loss_weights,
    build_embedding_model,
    build_llm_prior_weights,
    build_prediction_head,
    build_id2label,
    build_parser,
    build_label_map,
    build_lr_lambda,
    build_label_specs,
    collate_waveforms,
    collate_inference_waveforms,
    compute_classification_metrics,
    compute_hierarchical_metrics,
    epochs_to_update_steps,
    filter_bsd35k_records_by_pseudo_label_confidence,
    masked_mean_embedding_sequence,
    maybe_limit,
    pool_embedding_sequence,
    load_initial_training_state_dict,
    load_pseudo_labels,
    resolve_pseudo_label_dir,
    resolve_checkpoint_path,
    resolve_initial_checkpoint_path,
    resolve_training_run_checkpoint_path,
    resolve_embedding_sample_rate,
    resolve_record_file_id,
    resolve_dataset_roots,
    resolve_seed,
    run_experiment,
    write_logits_npz,
)
from dcase2026_task1.models.audio_wrappers import (
    ArbitraryLengthAudioWrapper,
    mean_segment_outputs,
    pack_segment_outputs,
    split_waveforms_into_segments,
)
from dcase2026_task1.models.M2D import (
    DEFAULT_CHECKPOINT_ARCHIVE_ALIAS as M2D_DEFAULT_CHECKPOINT_ARCHIVE_ALIAS,
    DEFAULT_CHECKPOINT_FILENAME as M2D_DEFAULT_CHECKPOINT_FILENAME,
    M2DTextEncoderEmbeddingModel,
    _build_m2d_token_padding_mask,
    _concatenate_segment_outputs,
    resolve_checkpoint_path as resolve_m2d_checkpoint_path,
)
from dcase2026_task1.models.beats import (
    DEFAULT_CHECKPOINT_ALIAS,
    BEATs,
    BEATsConfig,
    ChunkedBEATs,
)
from dcase2026_task1.models.clap import (
    CLAPEmbeddingModel,
    build_clap_embedding_model,
    metadata_to_keyword_texts,
    metadata_to_summary_texts,
)
from dcase2026_task1.models.clap.passt import CutInputIntoSegmentsWrapper
from dcase2026_task1.models.lclap import (
    LAIONCLAPAudioEncoder,
    LAIONCLAPEmbeddingModel,
    LAIONCLAPTextEncoder,
    build_lclap_embedding_model,
    int16_quantize_audio,
)
from dcase2026_task1.models.passt import _extract_passt_embeddings


def test_resolve_dataset_roots() -> None:
    roots = resolve_dataset_roots("/tmp/bsd10k", "/tmp/bsd35k", "/tmp/bsd2k")
    assert roots["BSD10k"].as_posix() == "/tmp/bsd10k"
    assert roots["BSD35k-CS"].as_posix() == "/tmp/bsd35k"
    assert roots["BSD2k"].as_posix() == "/tmp/bsd2k"


def test_build_label_specs_and_maps() -> None:
    records = [
        {"class_idx": 7, "class": "fx-a"},
        {"class_idx": 3, "class": "m-si"},
        {"class_idx": 7, "class": "fx-a"},
    ]
    label_specs = build_label_specs(records)

    assert [spec.dataset_class_idx for spec in label_specs] == [3, 7]
    assert build_label_map(label_specs) == {3: 0, 7: 1}
    assert build_id2label(label_specs) == {0: "m-si", 1: "fx-a"}


def test_build_class_frequency_loss_weights_uses_inverse_frequency() -> None:
    records = [
        {"class_idx": 3, "class": "m-si"},
        {"class_idx": 3, "class": "m-si"},
        {"class_idx": 7, "class": "fx-a"},
    ]
    label_specs = build_label_specs(records)
    label_map = build_label_map(label_specs)

    weights = build_class_frequency_loss_weights(records, [0, 1, 2], label_map)

    assert torch.allclose(weights, torch.tensor([0.75, 1.5]))


def test_maybe_limit() -> None:
    assert maybe_limit([1, 2, 3], None) == [1, 2, 3]
    assert maybe_limit([1, 2, 3], 2) == [1, 2]


def test_parser_seed_defaults_to_none() -> None:
    args = build_parser().parse_args([])
    assert args.seed is None
    assert args.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert args.use_llm_prior_embedding_fusion is False
    assert args.use_class_frequency_loss is False
    assert args.pseudo_label_dir is None
    assert args.pseudo_label_weight == 1.0
    assert args.bsd35k_pseudo_label_confidence_retention is None
    assert args.only_bsd35k_cs is False
    assert args.init_checkpoint_path is None
    assert args.save_checkpoints is False
    assert args.early_stopping_patience == 10
    assert args.label_smoothing == 0.0
    assert args.head_hidden_layers == 1
    assert args.head_hidden_dim is None
    assert not hasattr(args, "checkpoint_alias")


def test_resolve_seed_returns_explicit_seed() -> None:
    assert resolve_seed(1234) == 1234


def test_resolve_seed_generates_random_seed_when_missing() -> None:
    with patch(
        "dcase2026_task1.experiments.training.random.SystemRandom.randint",
        return_value=987654321,
    ) as randint:
        resolved = resolve_seed(None)

    assert resolved == 987654321
    randint.assert_called_once_with(0, (2**32) - 1)


def test_resolve_pseudo_label_dir_uses_output_root_for_missing_relative_path(tmp_path: Path) -> None:
    assert resolve_pseudo_label_dir("ensemble_a", tmp_path / "outputs") == tmp_path / "outputs" / "ensemble_a"


def test_resolve_pseudo_label_dir_keeps_existing_path(tmp_path: Path) -> None:
    pseudo_label_dir = tmp_path / "existing"
    pseudo_label_dir.mkdir()

    assert resolve_pseudo_label_dir(str(pseudo_label_dir), tmp_path / "outputs") == pseudo_label_dir


def test_resolve_initial_checkpoint_path_uses_output_root_for_run_name(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "outputs" / "model_a" / "checkpoints" / "best.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")

    resolved = resolve_initial_checkpoint_path("model_a", tmp_path / "outputs")

    assert resolved == checkpoint_path.resolve()


def test_resolve_training_run_checkpoint_path_uses_summary_best_model_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "model_a"
    checkpoint_path = run_dir / "checkpoints" / "epoch.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    (run_dir / "summary.json").write_text(
        json.dumps({"best_model_path": str(checkpoint_path)}),
        encoding="utf-8",
    )

    resolved = resolve_training_run_checkpoint_path(run_dir)

    assert resolved == checkpoint_path.resolve()


def test_resolve_embedding_sample_rate_supports_embedding_models() -> None:
    assert resolve_embedding_sample_rate("beats") == 16000
    assert resolve_embedding_sample_rate("clap") == 32000
    assert resolve_embedding_sample_rate("clap_kw") == 32000
    assert resolve_embedding_sample_rate("lclap") == 48000
    assert resolve_embedding_sample_rate("lclap_audio") == 48000
    assert resolve_embedding_sample_rate("lclap_text") == 48000
    assert resolve_embedding_sample_rate("lclap_kw") == 48000
    assert resolve_embedding_sample_rate("llm") == 16000
    assert resolve_embedding_sample_rate("m2d") == 16000
    assert resolve_embedding_sample_rate("m2d_te") == 16000
    assert resolve_embedding_sample_rate("passt") == 32000


def test_vendored_beats_package_exports_model_classes() -> None:
    assert BEATs.__name__ == "BEATs"
    assert BEATsConfig.__name__ == "BEATsConfig"
    assert ChunkedBEATs.__name__ == "ChunkedBEATs"


def test_resolve_checkpoint_path_from_explicit_path(tmp_path) -> None:
    checkpoint_dir = tmp_path / "cache"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / f"{DEFAULT_CHECKPOINT_ALIAS}.pt"
    checkpoint_path.write_text("x", encoding="utf-8")

    resolved = resolve_checkpoint_path(
        checkpoint_dir=checkpoint_dir,
        checkpoint_alias=DEFAULT_CHECKPOINT_ALIAS,
    )

    assert resolved == checkpoint_path.resolve()


def test_official_checkpoint_alias_is_available() -> None:
    assert DEFAULT_CHECKPOINT_ALIAS == "beats_iter3plus_as2m"


def test_m2d_checkpoint_resolution_stages_root_level_weights_file(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / M2D_DEFAULT_CHECKPOINT_FILENAME
    checkpoint_path.write_bytes(b"PKT")

    resolved = resolve_m2d_checkpoint_path(
        checkpoint_dir=tmp_path,
        checkpoint_archive_alias=M2D_DEFAULT_CHECKPOINT_ARCHIVE_ALIAS,
        checkpoint_filename=M2D_DEFAULT_CHECKPOINT_FILENAME,
    )

    assert resolved.name == M2D_DEFAULT_CHECKPOINT_FILENAME
    assert resolved.parent.name == Path(M2D_DEFAULT_CHECKPOINT_ARCHIVE_ALIAS).stem
    assert resolved.read_bytes() == b"PKT"


def test_build_experiment_split_keeps_noisy_records_out_of_val_and_test() -> None:
    clean_records = [
        {
            "sound_id": index,
            "class_idx": index % 2,
            "class": f"class-{index % 2}",
            "source_dataset": "BSD10k",
            "audio_path": Path(f"/tmp/clean_{index}.wav"),
        }
        for index in range(20)
    ]
    noisy_records = [
        {
            "sound_id": 100 + index,
            "class_idx": index % 2,
            "class": f"class-{index % 2}",
            "source_dataset": "BSD35k-CS",
            "audio_path": Path(f"/tmp/noisy_{index}.wav"),
        }
        for index in range(6)
    ]
    with patch(
        "dcase2026_task1.data.splits.load_records_by_dataset_name",
        side_effect=lambda dataset_name, root: clean_records if dataset_name == "BSD10k" else noisy_records,
    ):
        split = build_experiment_split(
            bsd10k_root=Path("/tmp/bsd10k"),
            bsd35k_root=Path("/tmp/bsd35k"),
            include_bsd35k_cs=True,
            only_bsd35k_cs=False,
            fold=0,
            n_splits=5,
            validation_size=0.2,
        )

    assert all(record["source_dataset"] == "BSD10k" for record in split.val_records)
    assert all(record["source_dataset"] == "BSD10k" for record in split.test_records)
    assert split.noisy_train_size == len(noisy_records)
    assert sum(record["source_dataset"] == "BSD35k-CS" for record in split.train_records) == len(noisy_records)


def test_load_records_by_dataset_name_filters_bsd35k_other_classes() -> None:
    dataset_records = [
        {"class": "fx-a", "sound_id": 1},
        {"class": "m-other", "sound_id": 2},
        {"class": "sp-s", "sound_id": 3},
        {"class": "fx-other", "sound_id": 4},
    ]

    class FakeDataset:
        def __init__(self, root: Path, dataset_name: str, load_audio: bool) -> None:
            self.records = dataset_records

    with patch("dcase2026_task1.data.datasets.BSDDataset", FakeDataset):
        bsd35k_records = load_records_by_dataset_name("BSD35k-CS", Path("/tmp/bsd35k"))
        bsd10k_records = load_records_by_dataset_name("BSD10k", Path("/tmp/bsd10k"))

    assert [record["sound_id"] for record in bsd35k_records] == [1, 3]
    assert [record["sound_id"] for record in bsd10k_records] == [1, 2, 3, 4]


def test_build_experiment_split_can_use_only_bsd35k_for_training() -> None:
    clean_records = [
        {
            "sound_id": index,
            "class_idx": index % 2,
            "class": f"class-{index % 2}",
            "source_dataset": "BSD10k",
            "audio_path": Path(f"/tmp/clean_{index}.wav"),
        }
        for index in range(20)
    ]
    noisy_records = [
        {
            "sound_id": 100 + index,
            "class_idx": index % 2,
            "class": f"class-{index % 2}",
            "source_dataset": "BSD35k-CS",
            "audio_path": Path(f"/tmp/noisy_{index}.wav"),
        }
        for index in range(6)
    ]

    with patch(
        "dcase2026_task1.data.splits.load_records_by_dataset_name",
        side_effect=lambda dataset_name, root: clean_records if dataset_name == "BSD10k" else noisy_records,
    ):
        split = build_experiment_split(
            bsd10k_root=Path("/tmp/bsd10k"),
            bsd35k_root=Path("/tmp/bsd35k"),
            include_bsd35k_cs=False,
            only_bsd35k_cs=True,
            fold=0,
            n_splits=5,
            validation_size=0.2,
        )

    assert split.clean_train_size == 0
    assert split.noisy_train_size == len(noisy_records)
    assert split.train_records == noisy_records
    assert all(record["source_dataset"] == "BSD10k" for record in split.val_records)
    assert all(record["source_dataset"] == "BSD10k" for record in split.test_records)


def test_build_experiment_split_is_reproducible() -> None:
    clean_records = [
        {
            "sound_id": index,
            "class_idx": index % 2,
            "class": f"class-{index % 2}",
            "source_dataset": "BSD10k",
            "audio_path": Path(f"/tmp/clean_{index}.wav"),
        }
        for index in range(20)
    ]

    with patch(
        "dcase2026_task1.data.splits.load_records_by_dataset_name",
        side_effect=lambda dataset_name, root: clean_records,
    ):
        split_a = build_experiment_split(
            bsd10k_root=Path("/tmp/bsd10k"),
            bsd35k_root=None,
            include_bsd35k_cs=False,
            only_bsd35k_cs=False,
            fold=1,
            n_splits=5,
            validation_size=0.2,
        )
        split_b = build_experiment_split(
            bsd10k_root=Path("/tmp/bsd10k"),
            bsd35k_root=None,
            include_bsd35k_cs=False,
            only_bsd35k_cs=False,
            fold=1,
            n_splits=5,
            validation_size=0.2,
        )

    assert [record["sound_id"] for record in split_a.train_records] == [
        record["sound_id"] for record in split_b.train_records
    ]
    assert [record["sound_id"] for record in split_a.val_records] == [
        record["sound_id"] for record in split_b.val_records
    ]
    assert [record["sound_id"] for record in split_a.test_records] == [
        record["sound_id"] for record in split_b.test_records
    ]
    assert split_a.split_seed == DEFAULT_BSD_SPLIT_SEED


def test_get_experiment_records_returns_split_records() -> None:
    clean_records = [
        {
            "sound_id": index,
            "class_idx": index % 2,
            "class": f"class-{index % 2}",
            "source_dataset": "BSD10k",
            "audio_path": Path(f"/tmp/clean_{index}.wav"),
        }
        for index in range(20)
    ]
    noisy_records = [
        {
            "sound_id": 100 + index,
            "class_idx": index % 2,
            "class": f"class-{index % 2}",
            "source_dataset": "BSD35k-CS",
            "audio_path": Path(f"/tmp/noisy_{index}.wav"),
        }
        for index in range(4)
    ]

    with patch(
        "dcase2026_task1.data.splits.load_records_by_dataset_name",
        side_effect=lambda dataset_name, root: clean_records if dataset_name == "BSD10k" else noisy_records,
    ):
        train_records, val_records, test_records = get_experiment_records(
            bsd10k_root=Path("/tmp/bsd10k"),
            bsd35k_root=Path("/tmp/bsd35k"),
            include_bsd35k_cs=True,
            only_bsd35k_cs=False,
            fold=0,
            n_splits=5,
            validation_size=0.2,
        )

    assert len(train_records) > 0
    assert len(val_records) > 0
    assert len(test_records) > 0
    assert all(record["source_dataset"] == "BSD10k" for record in val_records)
    assert all(record["source_dataset"] == "BSD10k" for record in test_records)


def test_split_waveforms_into_segments_splits_long_audio() -> None:
    waveforms = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0],
            [10.0, 11.0, 12.0, 13.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    padding_mask = torch.tensor(
        [
            [False, False, False, False, False, False, True, True],
            [False, False, False, False, True, True, True, True],
        ]
    )

    segmented_waveforms, segmented_padding_mask, segment_batch_indices = split_waveforms_into_segments(
        waveforms,
        padding_mask,
        max_segment_samples=4,
    )

    assert torch.equal(
        segmented_waveforms,
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 0.0, 0.0],
                [10.0, 11.0, 12.0, 13.0],
            ]
        ),
    )
    assert torch.equal(
        segmented_padding_mask,
        torch.tensor(
            [
                [False, False, False, False],
                [False, False, True, True],
                [False, False, False, False],
            ]
        ),
    )
    assert torch.equal(segment_batch_indices, torch.tensor([0, 0, 1]))


def test_mean_segment_outputs_averages_segments_per_sample() -> None:
    segment_outputs = torch.tensor(
        [
            [1.0, 3.0],
            [5.0, 7.0],
            [2.0, 4.0],
        ]
    )
    segment_batch_indices = torch.tensor([0, 0, 1])

    outputs = mean_segment_outputs(segment_outputs, segment_batch_indices, batch_size=2)

    assert torch.allclose(outputs, torch.tensor([[3.0, 5.0], [2.0, 4.0]]))


def test_mean_segment_outputs_preserves_sequence_shape() -> None:
    segment_outputs = torch.tensor(
        [
            [[1.0, 2.0]],
            [[5.0, 6.0]],
            [[3.0, 4.0]],
        ]
    )
    segment_batch_indices = torch.tensor([0, 0, 1])

    outputs = mean_segment_outputs(segment_outputs, segment_batch_indices, batch_size=2)

    assert torch.allclose(outputs, torch.tensor([[[3.0, 4.0]], [[3.0, 4.0]]]))


def test_pack_segment_outputs_returns_one_embedding_per_segment() -> None:
    segment_outputs = torch.tensor(
        [
            [1.0, 2.0],
            [5.0, 6.0],
            [3.0, 4.0],
        ]
    )
    segment_batch_indices = torch.tensor([0, 0, 1])

    outputs, padding_mask = pack_segment_outputs(segment_outputs, segment_batch_indices, batch_size=2)

    assert torch.allclose(outputs, torch.tensor([[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [0.0, 0.0]]]))
    assert torch.equal(padding_mask, torch.tensor([[False, False], [False, True]]))


def test_arbitrary_length_audio_wrapper_chunks_and_aggregates() -> None:
    class FakeEncoder(torch.nn.Module):
        pass

    def segment_forward(
        _model: torch.nn.Module,
        waveforms: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if padding_mask is None:
            valid = torch.ones_like(waveforms, dtype=torch.bool)
        else:
            valid = ~padding_mask
        return torch.stack(
            [
                (waveforms * valid).sum(dim=1),
                valid.sum(dim=1),
            ],
            dim=1,
        )

    wrapper = ArbitraryLengthAudioWrapper(
        FakeEncoder(),
        sample_rate=2,
        max_audio_seconds=2,
        segment_forward=segment_forward,
        aggregate_outputs=mean_segment_outputs,
    )

    waveforms = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0],
            [10.0, 11.0, 12.0, 13.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    padding_mask = torch.tensor(
        [
            [False, False, False, False, False, False, True, True],
            [False, False, False, False, True, True, True, True],
        ]
    )

    outputs = wrapper(waveforms, padding_mask)

    assert torch.allclose(outputs, torch.tensor([[10.5, 3.0], [46.0, 4.0]]))


def test_waveform_dataset_returns_metadata(monkeypatch) -> None:
    records = [
        {
            "audio_path": "/tmp/example.wav",
            "class_idx": 7,
            "sound_id": 11,
            "source_dataset": "BSD10k",
            "metadata": {"title": "clip", "description": "metal hit"},
        }
    ]

    monkeypatch.setattr(
        "dcase2026_task1.data.datasets.load_audio_waveform",
        lambda _path: (torch.tensor([[1.0, 2.0, 3.0]]).numpy(), 16000),
    )

    dataset = WaveformClassificationDataset(
        records=records,
        indices=[0],
        label_map={7: 1},
        target_sample_rate=16000,
    )

    item = dataset[0]

    assert item["label"] == 1
    assert item["metadata"] == {"title": "clip", "description": "metal hit"}


def test_waveform_inference_dataset_returns_file_id_and_metadata(monkeypatch) -> None:
    records = [
        {
            "audio_path": "/tmp/example.wav",
            "anonymous_id": "anon-1",
            "metadata": {"title": "clip"},
        }
    ]

    monkeypatch.setattr(
        "dcase2026_task1.data.datasets.load_audio_waveform",
        lambda _path: (torch.tensor([[1.0, 2.0, 3.0]]).numpy(), 16000),
    )

    dataset = WaveformInferenceDataset(records=records, target_sample_rate=16000)
    item = dataset[0]

    assert item["file_id"] == "anon-1"
    assert item["metadata"] == {"title": "clip"}


def test_collate_waveforms_keeps_metadata() -> None:
    batch = collate_waveforms(
        [
            {
                "waveform": torch.tensor([1.0, 2.0]).numpy(),
                "label": 0,
                "metadata": {"title": "a"},
            },
            {
                "waveform": torch.tensor([3.0]).numpy(),
                "label": 1,
                "metadata": {"title": "b"},
            },
        ]
    )

    assert batch["waveforms"].shape == (2, 2)
    assert batch["metadata"] == [{"title": "a"}, {"title": "b"}]


def test_collate_inference_waveforms_keeps_file_ids_and_metadata() -> None:
    batch = collate_inference_waveforms(
        [
            {
                "waveform": torch.tensor([1.0, 2.0]).numpy(),
                "file_id": "a",
                "metadata": {"title": "x"},
            },
            {
                "waveform": torch.tensor([3.0]).numpy(),
                "file_id": "b",
                "metadata": {"title": "y"},
            },
        ]
    )

    assert batch["waveforms"].shape == (2, 2)
    assert batch["file_ids"] == ["a", "b"]
    assert batch["metadata"] == [{"title": "x"}, {"title": "y"}]


def test_resolve_record_file_id_prefers_anonymous_id() -> None:
    assert resolve_record_file_id({"anonymous_id": "anon-9", "sound_id": 123}) == "anon-9"
    assert resolve_record_file_id({"sound_id": 123}) == "123"
    assert resolve_record_file_id({"audio_path": "/tmp/file.wav"}) == "file"


def test_write_logits_npz_stores_logits_by_file_id_and_label_names(tmp_path: Path) -> None:
    path = tmp_path / "predictions.npz"
    label_specs = build_label_specs(
        [
            {"class_idx": 401, "class": "fx-o"},
            {"class_idx": 203, "class": "is-w"},
        ]
    )

    write_logits_npz(
        path,
        file_ids=["file-a", "file-b"],
        logits=torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy(),
        label_specs=label_specs,
    )

    archive = np.load(path)
    assert set(archive.files) == {"file-a", "file-b", "label_names"}
    assert np.allclose(archive["file-a"], np.array([1.0, 2.0], dtype=np.float32))
    assert np.allclose(archive["file-b"], np.array([3.0, 4.0], dtype=np.float32))
    assert archive["label_names"].tolist() == ["is-w", "fx-o"]


def test_load_pseudo_labels_from_predictions_json(tmp_path: Path) -> None:
    label_specs = build_label_specs(
        [
            {"class_idx": 401, "class": "fx-o"},
            {"class_idx": 203, "class": "is-w"},
        ]
    )
    prediction_dir = tmp_path / "run"
    prediction_dir.mkdir()
    (prediction_dir / "predictions.json").write_text(
        """
        {
          "label_names": ["fx-o", "is-w"],
          "datasets": {
            "BSD10k": [
              {"file_id": "clip-a", "probabilities": [0.75, 0.25]}
            ]
          }
        }
        """,
        encoding="utf-8",
    )

    pseudo_labels = load_pseudo_labels(prediction_dir, label_specs)

    assert np.allclose(pseudo_labels["BSD10k:clip-a"], np.array([0.25, 0.75]))


def test_load_pseudo_labels_from_ensemble_logits(tmp_path: Path) -> None:
    label_specs = build_label_specs(
        [
            {"class_idx": 401, "class": "fx-o"},
            {"class_idx": 203, "class": "is-w"},
        ]
    )
    ensemble_dir = tmp_path / "ensemble_a__b"
    ensemble_dir.mkdir()
    np.savez(
        ensemble_dir / "bsd35k_cs_logits.npz",
        clip_b=np.array([2.0, 0.0], dtype=np.float32),
        label_names=np.asarray(["fx-o", "is-w"], dtype=np.str_),
    )

    pseudo_labels = load_pseudo_labels(ensemble_dir, label_specs)

    expected = torch.softmax(torch.tensor([0.0, 2.0]), dim=0).numpy()
    assert np.allclose(pseudo_labels["BSD35k-CS:clip_b"], expected)


def test_filter_bsd35k_records_by_pseudo_label_confidence_retains_top_fraction_per_predicted_class() -> None:
    records = [
        {"sound_id": 1, "source_dataset": "BSD10k"},
        {"sound_id": 10, "source_dataset": "BSD35k-CS"},
        {"sound_id": 11, "source_dataset": "BSD35k-CS"},
        {"sound_id": 12, "source_dataset": "BSD35k-CS"},
        {"sound_id": 13, "source_dataset": "BSD35k-CS"},
        {"sound_id": 14, "source_dataset": "BSD35k-CS"},
    ]
    pseudo_labels = {
        "BSD35k-CS:10": np.array([0.9, 0.1], dtype=np.float32),
        "BSD35k-CS:11": np.array([0.6, 0.4], dtype=np.float32),
        "BSD35k-CS:12": np.array([0.8, 0.2], dtype=np.float32),
        "BSD35k-CS:13": np.array([0.2, 0.8], dtype=np.float32),
        "BSD35k-CS:14": np.array([0.3, 0.7], dtype=np.float32),
    }

    filtered, stats = filter_bsd35k_records_by_pseudo_label_confidence(
        records,
        pseudo_labels,
        0.5,
    )

    assert [record["sound_id"] for record in filtered] == [1, 10, 12, 13, 10, 12]
    assert stats["enabled"] is True
    assert stats["bsd35k_before"] == 5
    assert stats["bsd35k_after"] == 5
    assert stats["bsd35k_unique_retained"] == 3
    assert stats["bsd35k_missing_pseudo_labels"] == 0
    assert stats["retained_by_pseudo_label_id"] == {"0": 2, "1": 1}


def test_filter_bsd35k_records_by_pseudo_label_confidence_drops_missing_pseudo_labels() -> None:
    records = [
        {"sound_id": 10, "source_dataset": "BSD35k-CS"},
        {"sound_id": 11, "source_dataset": "BSD35k-CS"},
    ]
    pseudo_labels = {
        "BSD35k-CS:10": np.array([0.9, 0.1], dtype=np.float32),
    }

    filtered, stats = filter_bsd35k_records_by_pseudo_label_confidence(
        records,
        pseudo_labels,
        1.0,
    )

    assert [record["sound_id"] for record in filtered] == [10, 10]
    assert stats["bsd35k_after"] == 2
    assert stats["bsd35k_unique_retained"] == 1
    assert stats["bsd35k_missing_pseudo_labels"] == 1


def test_filter_bsd35k_records_by_pseudo_label_confidence_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        filter_bsd35k_records_by_pseudo_label_confidence([], {}, 1.5)


def test_build_embedding_model_beats_accepts_metadata() -> None:
    args = Namespace(
        embedding_model="beats",
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
    )

    class FakeBeats(torch.nn.Module):
        def __init__(self, _config) -> None:
            super().__init__()

        def load_state_dict(self, _state_dict, strict=False):
            return [], []

    class FakeChunkedBeats(torch.nn.Module):
        def __init__(self, _model, *, sample_rate: int, max_audio_seconds: float) -> None:
            super().__init__()
            self.sample_rate = sample_rate
            self.max_audio_seconds = max_audio_seconds
            self.output_dim = 4
            self.received_metadata = None

        def forward(self, waveforms, padding_mask=None, metadata=None):
            self.received_metadata = metadata
            return (
                torch.ones((waveforms.shape[0], 2, self.output_dim)),
                torch.zeros((waveforms.shape[0], 2), dtype=torch.bool),
            )

    with (
        patch(
            "dcase2026_task1.models.beats.load_embedding_checkpoint",
            return_value={"cfg": {"encoder_embed_dim": 4}, "model": {}},
        ),
        patch("dcase2026_task1.models.beats.BEATsConfig", side_effect=lambda cfg: SimpleNamespace(**cfg)),
        patch("dcase2026_task1.models.beats.BEATs", FakeBeats),
        patch("dcase2026_task1.models.beats.ChunkedBEATs", FakeChunkedBeats),
    ):
        model = build_embedding_model(args, sample_rate=16000)
        embedding_sequence, embedding_padding_mask = model(
            torch.ones((2, 8)),
            torch.zeros((2, 8), dtype=torch.bool),
            metadata=[{"title": "x"}, {"title": "y"}],
        )

    assert embedding_sequence.shape == (2, 2, 4)
    assert torch.equal(embedding_padding_mask, torch.zeros((2, 2), dtype=torch.bool))
    assert model.received_metadata == [{"title": "x"}, {"title": "y"}]


def test_build_embedding_model_passt_accepts_metadata() -> None:
    args = Namespace(
        embedding_model="passt",
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
    )

    class FakeChunkedPaSST(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 6
            self.received_metadata = None

        def forward(self, waveforms, padding_mask=None, metadata=None):
            self.received_metadata = metadata
            return (
                torch.ones((waveforms.shape[0], 3, self.output_dim)),
                torch.zeros((waveforms.shape[0], 3), dtype=torch.bool),
            )

    fake_model = FakeChunkedPaSST()
    with patch(
        "dcase2026_task1.experiments.training.build_passt_embedding_model",
        return_value=fake_model,
    ) as build_passt:
        model = build_embedding_model(args, sample_rate=32000)
        embedding_sequence, embedding_padding_mask = model(
            torch.ones((2, 8)),
            torch.zeros((2, 8), dtype=torch.bool),
            metadata=[{"title": "x"}, {"title": "y"}],
        )

    build_passt.assert_called_once_with(
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
        sample_rate=32000,
    )
    assert embedding_sequence.shape == (2, 3, 6)
    assert torch.equal(embedding_padding_mask, torch.zeros((2, 3), dtype=torch.bool))
    assert model.received_metadata == [{"title": "x"}, {"title": "y"}]


def test_build_embedding_model_m2d_accepts_metadata() -> None:
    args = Namespace(
        embedding_model="m2d",
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
    )

    class FakeChunkedM2D(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 5
            self.received_metadata = None

        def forward(self, waveforms, padding_mask=None, metadata=None):
            self.received_metadata = metadata
            return (
                torch.ones((waveforms.shape[0], 4, self.output_dim)),
                torch.zeros((waveforms.shape[0], 4), dtype=torch.bool),
            )

    fake_model = FakeChunkedM2D()
    with patch(
        "dcase2026_task1.experiments.training.build_m2d_embedding_model",
        return_value=fake_model,
    ) as build_m2d:
        model = build_embedding_model(args, sample_rate=16000)
        embedding_sequence, embedding_padding_mask = model(
            torch.ones((2, 8)),
            torch.zeros((2, 8), dtype=torch.bool),
            metadata=[{"title": "x"}, {"title": "y"}],
        )

    build_m2d.assert_called_once_with(
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
        sample_rate=16000,
    )
    assert embedding_sequence.shape == (2, 4, 5)
    assert torch.equal(embedding_padding_mask, torch.zeros((2, 4), dtype=torch.bool))
    assert model.received_metadata == [{"title": "x"}, {"title": "y"}]


def test_build_embedding_model_m2d_te_accepts_metadata() -> None:
    args = Namespace(
        embedding_model="m2d_te",
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
    )

    class FakeM2DTE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 7
            self.received_metadata = None

        def forward(self, waveforms, padding_mask=None, metadata=None):
            self.received_metadata = metadata
            return (
                torch.ones((waveforms.shape[0], 1, self.output_dim)),
                torch.zeros((waveforms.shape[0], 1), dtype=torch.bool),
            )

    fake_model = FakeM2DTE()
    with patch(
        "dcase2026_task1.experiments.training.build_m2d_text_encoder_embedding_model",
        return_value=fake_model,
    ) as build_m2d_te:
        model = build_embedding_model(args, sample_rate=16000)
        embedding_sequence, embedding_padding_mask = model(
            torch.ones((2, 8)),
            torch.zeros((2, 8), dtype=torch.bool),
            metadata=[
                {"metadata_summary": "A metal impact."},
                {"metadata_summary": "Rain and water."},
            ],
        )

    build_m2d_te.assert_called_once_with(
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
        sample_rate=16000,
    )
    assert embedding_sequence.shape == (2, 1, 7)
    assert torch.equal(embedding_padding_mask, torch.zeros((2, 1), dtype=torch.bool))
    assert model.received_metadata == [
        {"metadata_summary": "A metal impact."},
        {"metadata_summary": "Rain and water."},
    ]


def test_build_embedding_model_clap_accepts_metadata() -> None:
    args = Namespace(
        embedding_model="clap",
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
    )

    class FakeCLAP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 8
            self.received_metadata = None

        def forward(self, waveforms, padding_mask=None, metadata=None):
            self.received_metadata = metadata
            return (
                torch.ones((waveforms.shape[0], 1, self.output_dim)),
                torch.zeros((waveforms.shape[0], 1), dtype=torch.bool),
            )

    fake_model = FakeCLAP()
    with patch(
        "dcase2026_task1.experiments.training.build_clap_embedding_model",
        return_value=fake_model,
    ) as build_clap:
        model = build_embedding_model(args, sample_rate=32000)
        embedding_sequence, embedding_padding_mask = model(
            torch.ones((2, 8)),
            torch.zeros((2, 8), dtype=torch.bool),
            metadata=[
                {"metadata_summary": "A metal impact."},
                {"metadata_summary": "Rain and water."},
            ],
        )

    build_clap.assert_called_once_with(
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
        sample_rate=32000,
    )
    assert embedding_sequence.shape == (2, 1, 8)
    assert torch.equal(embedding_padding_mask, torch.zeros((2, 1), dtype=torch.bool))
    assert model.received_metadata == [
        {"metadata_summary": "A metal impact."},
        {"metadata_summary": "Rain and water."},
    ]


def test_build_embedding_model_clap_kw_uses_keyword_metadata() -> None:
    args = Namespace(
        embedding_model="clap_kw",
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
    )

    fake_model = torch.nn.Module()
    fake_model.output_dim = 8
    with patch(
        "dcase2026_task1.experiments.training.build_clap_embedding_model",
        return_value=fake_model,
    ) as build_clap:
        model = build_embedding_model(args, sample_rate=32000)

    build_clap.assert_called_once_with(
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
        sample_rate=32000,
        metadata_text_key="tags",
        arch="clap_kw",
    )
    assert model is fake_model


def test_build_embedding_model_lclap_accepts_metadata() -> None:
    args = Namespace(
        embedding_model="lclap",
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
    )

    class FakeLAIONCLAP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 8
            self.received_metadata = None

        def forward(self, waveforms, padding_mask=None, metadata=None):
            self.received_metadata = metadata
            return (
                torch.ones((waveforms.shape[0], 1, self.output_dim)),
                torch.zeros((waveforms.shape[0], 1), dtype=torch.bool),
            )

    fake_model = FakeLAIONCLAP()
    with patch(
        "dcase2026_task1.experiments.training.build_lclap_embedding_model",
        return_value=fake_model,
    ) as build_lclap:
        model = build_embedding_model(args, sample_rate=48000)
        embedding_sequence, embedding_padding_mask = model(
            torch.ones((2, 8)),
            torch.zeros((2, 8), dtype=torch.bool),
            metadata=[
                {"metadata_summary": "A metal impact."},
                {"metadata_summary": "Rain and water."},
            ],
        )

    build_lclap.assert_called_once_with(
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
        sample_rate=48000,
    )
    assert embedding_sequence.shape == (2, 1, 8)
    assert torch.equal(embedding_padding_mask, torch.zeros((2, 1), dtype=torch.bool))
    assert model.received_metadata == [
        {"metadata_summary": "A metal impact."},
        {"metadata_summary": "Rain and water."},
    ]


def test_build_embedding_model_lclap_audio_accepts_metadata() -> None:
    args = Namespace(
        embedding_model="lclap_audio",
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
    )

    class FakeLAIONCLAPAudio(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 4
            self.received_metadata = None

        def forward(self, waveforms, padding_mask=None, metadata=None):
            self.received_metadata = metadata
            return (
                torch.ones((waveforms.shape[0], 1, self.output_dim)),
                torch.zeros((waveforms.shape[0], 1), dtype=torch.bool),
            )

    fake_model = FakeLAIONCLAPAudio()
    with patch(
        "dcase2026_task1.experiments.training.build_lclap_audio_encoder",
        return_value=fake_model,
    ) as build_lclap_audio:
        model = build_embedding_model(args, sample_rate=48000)
        embedding_sequence, embedding_padding_mask = model(
            torch.ones((2, 8)),
            torch.zeros((2, 8), dtype=torch.bool),
            metadata=[
                {"metadata_summary": "A metal impact."},
                {"metadata_summary": "Rain and water."},
            ],
        )

    build_lclap_audio.assert_called_once_with(
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
        sample_rate=48000,
    )
    assert embedding_sequence.shape == (2, 1, 4)
    assert torch.equal(embedding_padding_mask, torch.zeros((2, 1), dtype=torch.bool))
    assert model.received_metadata == [
        {"metadata_summary": "A metal impact."},
        {"metadata_summary": "Rain and water."},
    ]


def test_build_embedding_model_lclap_text_accepts_metadata() -> None:
    args = Namespace(
        embedding_model="lclap_text",
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
    )

    class FakeLAIONCLAPText(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 4
            self.received_metadata = None

        def forward(self, waveforms, padding_mask=None, metadata=None):
            self.received_metadata = metadata
            return (
                torch.ones((waveforms.shape[0], 1, self.output_dim)),
                torch.zeros((waveforms.shape[0], 1), dtype=torch.bool),
            )

    fake_model = FakeLAIONCLAPText()
    with patch(
        "dcase2026_task1.experiments.training.build_lclap_text_encoder",
        return_value=fake_model,
    ) as build_lclap_text:
        model = build_embedding_model(args, sample_rate=48000)
        embedding_sequence, embedding_padding_mask = model(
            torch.ones((2, 8)),
            torch.zeros((2, 8), dtype=torch.bool),
            metadata=[
                {"metadata_summary": "A metal impact."},
                {"metadata_summary": "Rain and water."},
            ],
        )

    build_lclap_text.assert_called_once_with(
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
        sample_rate=48000,
    )
    assert embedding_sequence.shape == (2, 1, 4)
    assert torch.equal(embedding_padding_mask, torch.zeros((2, 1), dtype=torch.bool))
    assert model.received_metadata == [
        {"metadata_summary": "A metal impact."},
        {"metadata_summary": "Rain and water."},
    ]


def test_build_embedding_model_lclap_kw_uses_keyword_metadata() -> None:
    args = Namespace(
        embedding_model="lclap_kw",
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
    )

    fake_model = torch.nn.Module()
    fake_model.output_dim = 8
    with patch(
        "dcase2026_task1.experiments.training.build_lclap_embedding_model",
        return_value=fake_model,
    ) as build_lclap:
        model = build_embedding_model(args, sample_rate=48000)

    build_lclap.assert_called_once_with(
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
        sample_rate=48000,
        metadata_text_key="tags",
        arch="lclap_kw",
    )
    assert model is fake_model


def test_build_embedding_model_llm_uses_metadata_prior_class_embeddings() -> None:
    args = Namespace(
        embedding_model="llm",
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
    )
    id2label = {0: "m-sp", 1: "fx-a", 2: "ss-n"}

    model = build_embedding_model(args, sample_rate=16000, id2label=id2label)
    with torch.no_grad():
        model.class_embedding_bank.weight.zero_()
        model.class_embedding_bank.weight[:, :3] = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )

    embedding_sequence, embedding_padding_mask = model(
        torch.ones((2, 8)),
        torch.zeros((2, 8), dtype=torch.bool),
        metadata=[
            {
                "metadata_class_probabilities": [
                    {"label": "m-sp", "probability": 3.0},
                    {"label": "fx-a", "probability": 1.0},
                ]
            },
            {"metadata_class_probabilities": None},
        ],
    )

    assert model.output_dim == 512
    assert embedding_sequence.shape == (2, 1, 512)
    assert torch.equal(embedding_padding_mask, torch.zeros((2, 1), dtype=torch.bool))
    assert torch.allclose(embedding_sequence[0, 0, :3], torch.tensor([0.75, 0.5, 0.0]))
    assert torch.allclose(embedding_sequence[1, 0], torch.zeros(512))


def test_build_embedding_model_llm_requires_labels() -> None:
    args = Namespace(
        embedding_model="llm",
        checkpoint_dir="/tmp/checkpoints",
        trust_checkpoint=True,
    )

    try:
        build_embedding_model(args, sample_rate=16000)
    except ValueError as exc:
        assert "id2label is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError for llm embedding model without id2label.")


def test_clap_metadata_text_uses_metadata_summary() -> None:
    texts = metadata_to_summary_texts(
        [
            {"metadata_summary": "A metal impact.", "tags": "ignored"},
            {"metadata_summary": "Rain and water outdoors."},
            {"title": "missing summary", "tags": "ignored"},
        ],
        batch_size=3,
    )

    assert texts == ["A metal impact.", "Rain and water outdoors.", ""]


def test_clap_keyword_text_uses_tags() -> None:
    texts = metadata_to_keyword_texts(
        [
            {"metadata_summary": "ignored", "tags": "metal; impact, hit"},
            {"tags": ["rain", "water", "outdoors"]},
            {"metadata_summary": "missing tags"},
        ],
        batch_size=3,
    )

    assert texts == ["metal impact hit", "rain water outdoors", ""]


def test_build_clap_embedding_model_loads_default_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "clap.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")

    class FakeRetrieval(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.kwargs = kwargs
            self.audio_projection = torch.nn.Linear(1, 3)
            self.text_projection = torch.nn.Linear(1, 2)
            self.loaded_state_dict = None

        def load_state_dict(self, state_dict, strict=True):
            self.loaded_state_dict = state_dict
            return [], []

    expected_state_dict = {"weight": torch.ones(1)}
    with (
        patch(
            "dcase2026_task1.models.clap._build_audio_retrieval_model",
            return_value=FakeRetrieval(),
        ),
        patch(
            "dcase2026_task1.models.clap.torch.load",
            return_value={"state_dict": expected_state_dict},
        ) as torch_load,
    ):
        model = build_clap_embedding_model(
            checkpoint_dir=tmp_path,
            trust_checkpoint=True,
            sample_rate=32000,
        )

    torch_load.assert_called_once_with(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    assert model.retrieval_model.loaded_state_dict is expected_state_dict
    assert model.checkpoint_cfg["checkpoint_alias"] == "clap"
    assert model.checkpoint_cfg["checkpoint_loaded"] is True
    assert model.checkpoint_cfg["metadata_text_key"] == "metadata_summary"


def test_build_clap_embedding_model_loads_compiled_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "clap.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")

    class FakeRetrieval(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.audio_projection = torch.nn.Linear(1, 3)
            self.text_projection = torch.nn.Linear(1, 2)
            self.loaded_state_dict = None

        def load_state_dict(self, state_dict, strict=True):
            self.loaded_state_dict = state_dict
            return [], []

    expected_weight = torch.ones(1)
    compiled_state_dict = {
        "_orig_mod.audio_projection.weight": expected_weight,
        "text_branch._orig_mod.projection.weight": torch.zeros(1),
    }
    retrieval_model = FakeRetrieval()
    with (
        patch(
            "dcase2026_task1.models.clap._build_audio_retrieval_model",
            return_value=retrieval_model,
        ),
        patch(
            "dcase2026_task1.models.clap.torch.load",
            return_value={"state_dict": compiled_state_dict},
        ),
    ):
        build_clap_embedding_model(
            checkpoint_dir=tmp_path,
            trust_checkpoint=True,
            sample_rate=32000,
        )

    assert set(retrieval_model.loaded_state_dict) == {
        "audio_projection.weight",
        "text_branch.projection.weight",
    }
    assert retrieval_model.loaded_state_dict["audio_projection.weight"] is expected_weight
    assert torch.equal(
        retrieval_model.loaded_state_dict["text_branch.projection.weight"],
        torch.zeros(1),
    )


def test_clap_embedding_model_concatenates_audio_and_text_embeddings() -> None:
    class FakeRetrieval(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.audio_projection = torch.nn.Linear(1, 3)
            self.text_projection = torch.nn.Linear(1, 2)
            self.batch = None

        def forward_audio(self, batch):
            self.batch = batch
            return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        def forward_text(self, batch):
            return torch.tensor([[7.0, 8.0], [9.0, 10.0]])

    retrieval_model = FakeRetrieval()
    model = CLAPEmbeddingModel(retrieval_model, sample_rate=32000)

    embeddings, padding_mask = model(
        torch.ones((2, 64000)),
        torch.tensor(
            [
                [False] * 64000,
                [False] * 32000 + [True] * 32000,
            ]
        ),
        metadata=[
            {"metadata_summary": "A metal hit."},
            {"metadata_summary": "Rain and water."},
        ],
    )

    assert model.output_dim == 5
    assert embeddings.shape == (2, 1, 5)
    assert torch.equal(
        embeddings[:, 0],
        torch.tensor([[1.0, 2.0, 3.0, 7.0, 8.0], [4.0, 5.0, 6.0, 9.0, 10.0]]),
    )
    assert torch.equal(padding_mask, torch.zeros((2, 1), dtype=torch.bool))
    assert retrieval_model.batch["captions"] == [
        ["A metal hit."],
        ["Rain and water."],
    ]
    assert torch.allclose(retrieval_model.batch["duration"], torch.tensor([2.0, 1.0]))


def test_build_lclap_embedding_model_loads_local_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "lclap.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    class FakeLAIONModule:
        def __init__(self) -> None:
            self.loaded_path = None

        def load_ckpt(self, path=None):
            self.loaded_path = path

    fake_module = FakeLAIONModule()
    with patch(
        "dcase2026_task1.models.lclap._build_laion_clap_module",
        return_value=fake_module,
    ):
        model = build_lclap_embedding_model(
            checkpoint_dir=tmp_path,
            trust_checkpoint=True,
            sample_rate=48000,
        )

    assert fake_module.loaded_path == str(checkpoint_path)
    assert model.checkpoint_cfg["arch"] == "lclap"
    assert model.checkpoint_cfg["checkpoint_alias"] == "lclap"
    assert model.checkpoint_cfg["checkpoint_path"] == str(checkpoint_path)


def test_build_lclap_embedding_model_loads_default_checkpoint(tmp_path: Path) -> None:
    class FakeLAIONModule:
        def __init__(self) -> None:
            self.loaded_path = "not-called"

        def load_ckpt(self, path=None):
            self.loaded_path = path

    fake_module = FakeLAIONModule()
    with patch(
        "dcase2026_task1.models.lclap._build_laion_clap_module",
        return_value=fake_module,
    ):
        model = build_lclap_embedding_model(
            checkpoint_dir=tmp_path,
            trust_checkpoint=True,
            sample_rate=48000,
        )

    assert fake_module.loaded_path is None
    assert model.checkpoint_cfg["checkpoint_path"] is None


def test_lclap_embedding_model_segments_audio_and_concatenates_text() -> None:
    class FakeLAIONModule:
        def __init__(self) -> None:
            self.audio_shape = None
            self.texts = None

        def get_audio_embedding_from_data(self, x, use_tensor=True):
            self.audio_shape = x.shape
            return torch.stack(
                [
                    torch.full((2,), float(index + 1), dtype=torch.float32)
                    for index in range(x.shape[0])
                ]
            )

        def get_text_embedding(self, texts, use_tensor=True):
            self.texts = texts
            return torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    clap_module = FakeLAIONModule()
    model = LAIONCLAPEmbeddingModel(
        clap_module,
        sample_rate=4,
        max_audio_seconds=2.5,
        audio_embedding_dim=2,
        text_embedding_dim=2,
        quantize_audio=False,
    )

    embeddings, padding_mask = model(
        torch.ones((2, 13)),
        torch.tensor(
            [
                [False] * 13,
                [False] * 6 + [True] * 7,
            ]
        ),
        metadata=[
            {"metadata_summary": "A metal hit."},
            {"metadata_summary": "Rain and water."},
        ],
    )

    assert clap_module.audio_shape == torch.Size([3, 10])
    assert clap_module.texts == ["A metal hit.", "Rain and water."]
    assert embeddings.shape == (2, 1, 4)
    assert torch.equal(
        embeddings[:, 0],
        torch.tensor(
            [
                [1.5, 1.5, 10.0, 20.0],
                [3.0, 3.0, 30.0, 40.0],
            ]
        ),
    )
    assert torch.equal(padding_mask, torch.zeros((2, 1), dtype=torch.bool))


def test_lclap_audio_encoder_segments_audio_only() -> None:
    class FakeLAIONModule:
        def __init__(self) -> None:
            self.audio_shape = None

        def get_audio_embedding_from_data(self, x, use_tensor=True):
            self.audio_shape = x.shape
            return torch.stack(
                [
                    torch.full((2,), float(index + 1), dtype=torch.float32)
                    for index in range(x.shape[0])
                ]
            )

    clap_module = FakeLAIONModule()
    model = LAIONCLAPAudioEncoder(
        clap_module,
        sample_rate=4,
        max_audio_seconds=2.5,
        audio_embedding_dim=2,
        quantize_audio=False,
    )

    embeddings, padding_mask = model(
        torch.ones((2, 13)),
        torch.tensor(
            [
                [False] * 13,
                [False] * 6 + [True] * 7,
            ]
        ),
        metadata=[{"metadata_summary": "ignored"}, {"metadata_summary": "ignored"}],
    )

    assert clap_module.audio_shape == torch.Size([3, 10])
    assert model.output_dim == 2
    assert torch.equal(
        embeddings[:, 0],
        torch.tensor(
            [
                [1.5, 1.5],
                [3.0, 3.0],
            ]
        ),
    )
    assert torch.equal(padding_mask, torch.zeros((2, 1), dtype=torch.bool))


def test_lclap_text_encoder_uses_metadata_only_without_l2_normalization() -> None:
    class FakeLAIONModule:
        def __init__(self) -> None:
            self.texts = None

        def get_text_embedding(self, texts, use_tensor=True):
            self.texts = texts
            return torch.tensor([[3.0, 4.0], [6.0, 8.0]])

    clap_module = FakeLAIONModule()
    model = LAIONCLAPTextEncoder(
        clap_module,
        text_embedding_dim=2,
    )

    embeddings, padding_mask = model(
        torch.ones((2, 13)),
        torch.tensor(
            [
                [False] * 13,
                [False] * 6 + [True] * 7,
            ]
        ),
        metadata=[
            {"metadata_summary": "A metal hit."},
            {"metadata_summary": "Rain and water."},
        ],
    )

    assert clap_module.texts == ["A metal hit.", "Rain and water."]
    assert model.output_dim == 2
    assert torch.equal(
        embeddings[:, 0],
        torch.tensor(
            [
                [3.0, 4.0],
                [6.0, 8.0],
            ]
        ),
    )
    assert torch.equal(padding_mask, torch.zeros((2, 1), dtype=torch.bool))


def test_lclap_embedding_model_skips_text_l2_normalization() -> None:
    class FakeLAIONModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.received_text = None

        def encode_text(self, text_input, device):
            self.received_text = text_input
            return torch.tensor([[3.0, 4.0]], device=device)

    class FakeLAIONModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = FakeLAIONModel()

        def tokenizer(self, texts):
            return {"input_ids": torch.tensor([[len(texts[0])]])}

        def get_audio_embedding_from_data(self, x, use_tensor=True):
            return torch.tensor([[1.0, 2.0]], device=x.device)

        def get_text_embedding(self, texts, use_tensor=True):
            return torch.tensor([[0.6, 0.8]])

    clap_module = FakeLAIONModule()
    model = LAIONCLAPEmbeddingModel(
        clap_module,
        sample_rate=4,
        max_audio_seconds=2.5,
        audio_embedding_dim=2,
        text_embedding_dim=2,
        quantize_audio=False,
    )

    embeddings, _ = model(
        torch.ones((1, 10)),
        torch.zeros((1, 10), dtype=torch.bool),
        metadata=[{"metadata_summary": "A sound."}],
    )

    assert torch.equal(embeddings[0, 0, 2:], torch.tensor([3.0, 4.0]))
    assert clap_module.model.received_text["input_ids"].shape == torch.Size([1, 1])


def test_lclap_embedding_model_skips_audio_l2_normalization() -> None:
    class FakeLAIONModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.received_audio = None

        def encode_audio(self, audio_input, device):
            self.received_audio = audio_input
            return {"embedding": torch.tensor([[3.0, 4.0]], device=device)}

        def audio_projection(self, audio_embeddings):
            return audio_embeddings * 2.0

    class FakeLAIONModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enable_fusion = False
            self.model = FakeLAIONModel()
            self.model_cfg = {"audio_cfg": {"sample_rate": 48000}}

        def tokenizer(self, texts):
            return {"input_ids": torch.tensor([[len(texts[0])]])}

        def get_audio_embedding_from_data(self, x, use_tensor=True):
            return torch.tensor([[0.6, 0.8]], device=x.device)

        def get_text_embedding(self, texts, use_tensor=True):
            return torch.tensor([[1.0, 2.0]])

    def fake_get_audio_features(
        temp_dict,
        audio_waveform,
        max_len,
        data_truncating,
        data_filling,
        audio_cfg,
        require_grad,
    ):
        temp_dict["waveform"] = audio_waveform
        temp_dict["max_len"] = torch.tensor(float(max_len))
        return temp_dict

    laion_clap = ModuleType("laion_clap")
    training = ModuleType("laion_clap.training")
    data = ModuleType("laion_clap.training.data")
    data.get_audio_features = fake_get_audio_features

    clap_module = FakeLAIONModule()
    model = LAIONCLAPEmbeddingModel(
        clap_module,
        sample_rate=4,
        max_audio_seconds=2.5,
        audio_embedding_dim=2,
        text_embedding_dim=2,
        quantize_audio=False,
    )

    with patch.dict(
        sys.modules,
        {
            "laion_clap": laion_clap,
            "laion_clap.training": training,
            "laion_clap.training.data": data,
        },
    ):
        embeddings, _ = model(
            torch.ones((1, 10)),
            torch.zeros((1, 10), dtype=torch.bool),
            metadata=[{"metadata_summary": "A sound."}],
        )

    assert torch.equal(embeddings[0, 0, :2], torch.tensor([6.0, 8.0]))
    assert torch.equal(
        clap_module.model.received_audio["waveform"],
        torch.ones((1, 10)),
    )


def test_lclap_embedding_model_wraps_spectrogram_extractor_only() -> None:
    class FakeSpectrogramExtractor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))
            self.received_dtype = None
            self.output_requires_grad = None

        def forward(self, x):
            self.received_dtype = x.dtype
            output = x * self.weight
            self.output_requires_grad = output.requires_grad
            return output

    class FakeAudioBranch(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spectrogram_extractor = FakeSpectrogramExtractor()
            self.linear = torch.nn.Linear(1, 1).half()
            self.reshape_received_dtype = None
            self.reshape_output_requires_grad = None

        def reshape_wav2img(self, x):
            self.reshape_received_dtype = x.dtype
            output = x * self.linear.weight.flatten()[0]
            self.reshape_output_requires_grad = output.requires_grad
            return output

    class FakeLAIONModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received_dtype = None
            self.model = torch.nn.Module()
            self.model.audio_branch = FakeAudioBranch()

        def get_audio_embedding_from_data(self, x, use_tensor=True):
            self.received_dtype = x.dtype
            self.model.audio_branch.spectrogram_extractor(x)
            self.model.audio_branch.reshape_wav2img(x)
            return torch.ones((x.shape[0], 2), dtype=torch.float32)

        def get_text_embedding(self, texts, use_tensor=True):
            return torch.ones((len(texts), 2), dtype=torch.float32)

    clap_module = FakeLAIONModule()
    model = LAIONCLAPEmbeddingModel(
        clap_module,
        sample_rate=4,
        max_audio_seconds=2.5,
        audio_embedding_dim=2,
        text_embedding_dim=2,
        quantize_audio=False,
    )

    embeddings, _ = model(
        torch.ones((1, 10), dtype=torch.float16),
        torch.zeros((1, 10), dtype=torch.bool),
        metadata=[{"metadata_summary": "A sound."}],
    )

    assert clap_module.received_dtype == torch.float32
    assert (
        clap_module.model.audio_branch.spectrogram_extractor.module.received_dtype
        == torch.float32
    )
    assert (
        next(clap_module.model.audio_branch.spectrogram_extractor.parameters()).dtype
        == torch.float32
    )
    assert (
        next(clap_module.model.audio_branch.spectrogram_extractor.parameters()).requires_grad
        is False
    )
    assert (
        clap_module.model.audio_branch.spectrogram_extractor.module.output_requires_grad
        is False
    )
    assert clap_module.model.audio_branch.reshape_received_dtype == torch.float32
    assert clap_module.model.audio_branch.reshape_output_requires_grad is False
    assert next(clap_module.model.audio_branch.linear.parameters()).dtype == torch.float16
    assert next(clap_module.model.audio_branch.linear.parameters()).requires_grad is True
    assert embeddings.dtype == torch.float32


def test_lclap_quantizes_audio_like_laion_example() -> None:
    quantized = int16_quantize_audio(torch.tensor([[-2.0, -0.5, 0.5, 2.0]]))

    assert torch.allclose(
        quantized,
        torch.tensor([[-1.0, -16383.0 / 32767.0, 16383.0 / 32767.0, 1.0]]),
    )


def test_clap_segment_wrapper_pads_short_inputs_to_minimum_length() -> None:
    class FakeAudioModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received_shape = None

        def forward(self, x):
            self.received_shape = x.shape
            return torch.ones((x.shape[0], 3))

    audio_model = FakeAudioModel()
    wrapper = CutInputIntoSegmentsWrapper(
        audio_model,
        max_input_length=10,
        segment_length=10,
        hop_size=10,
        min_input_length=4,
    )

    outputs = wrapper(torch.ones((2, 3)))

    assert audio_model.received_shape == torch.Size([2, 4])
    assert outputs.shape == (2, 1, 3)


def test_clap_segment_wrapper_pads_trailing_chunk_to_minimum_length() -> None:
    class FakeAudioModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received_shape = None

        def forward(self, x):
            self.received_shape = x.shape
            return torch.ones((x.shape[0], 3))

    audio_model = FakeAudioModel()
    wrapper = CutInputIntoSegmentsWrapper(
        audio_model,
        max_input_length=10,
        segment_length=10,
        hop_size=10,
        min_input_length=10,
    )

    outputs = wrapper(torch.ones((2, 13)))

    assert audio_model.received_shape == torch.Size([4, 10])
    assert outputs.shape == (2, 2, 3)


def test_m2d_aggregate_outputs_concatenates_segment_time_axis() -> None:
    segment_embeddings = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0]],
        ]
    )
    segment_masks = torch.tensor(
        [
            [False, False],
            [False, True],
            [False, False],
        ]
    )
    segment_batch_indices = torch.tensor([0, 1, 0], dtype=torch.long)

    embeddings, padding_mask = _concatenate_segment_outputs(
        (segment_embeddings, segment_masks),
        segment_batch_indices,
        batch_size=2,
    )

    assert embeddings.shape == (2, 4, 2)
    assert torch.equal(
        embeddings[0],
        torch.tensor([[1.0, 10.0], [2.0, 20.0], [5.0, 50.0], [6.0, 60.0]]),
    )
    assert torch.equal(
        embeddings[1],
        torch.tensor([[3.0, 30.0], [4.0, 40.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    assert torch.equal(
        padding_mask,
        torch.tensor(
            [
                [False, False, False, False],
                [False, True, True, True],
            ]
        ),
    )


def test_m2d_te_concatenates_pooled_m2d_and_raw_text_embeddings() -> None:
    class FakeM2D(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 2
            self.received_metadata = None

        def forward(self, waveforms, padding_mask=None, metadata=None):
            self.received_metadata = metadata
            return (
                torch.tensor(
                    [
                        [[1.0, 10.0], [3.0, 30.0], [100.0, 1000.0]],
                        [[5.0, 50.0], [7.0, 70.0], [0.0, 0.0]],
                    ],
                    device=waveforms.device,
                ),
                torch.tensor(
                    [
                        [False, False, True],
                        [False, False, True],
                    ],
                    device=waveforms.device,
                ),
            )

    class FakeTextEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 2
            self.received_metadata = None

        def forward(self, waveforms, padding_mask=None, metadata=None):
            self.received_metadata = metadata
            return (
                torch.tensor([[3.0, 4.0], [6.0, 8.0]], device=waveforms.device).unsqueeze(1),
                torch.zeros((waveforms.shape[0], 1), dtype=torch.bool, device=waveforms.device),
            )

    m2d = FakeM2D()
    text_encoder = FakeTextEncoder()
    model = M2DTextEncoderEmbeddingModel(m2d, text_encoder)

    embeddings, padding_mask = model(
        torch.ones((2, 8)),
        torch.zeros((2, 8), dtype=torch.bool),
        metadata=[
            {"metadata_summary": "A metal hit."},
            {"metadata_summary": "Rain and water."},
        ],
    )

    assert text_encoder.received_metadata == [
        {"metadata_summary": "A metal hit."},
        {"metadata_summary": "Rain and water."},
    ]
    assert m2d.received_metadata == [
        {"metadata_summary": "A metal hit."},
        {"metadata_summary": "Rain and water."},
    ]
    assert embeddings.shape == (2, 1, 4)
    assert torch.equal(
        embeddings[:, 0],
        torch.tensor(
            [
                [2.0, 20.0, 3.0, 4.0],
                [6.0, 60.0, 6.0, 8.0],
            ]
        ),
    )
    assert torch.equal(padding_mask, torch.zeros((2, 1), dtype=torch.bool))


def test_m2d_token_padding_mask_marks_only_valid_time_steps() -> None:
    fake_model = SimpleNamespace(
        cfg=SimpleNamespace(hop_size=160, input_size=[80, 1001]),
        backbone=SimpleNamespace(patch_size=lambda: torch.tensor([16, 16])),
    )
    segment_padding_mask = torch.ones((2, 160000), dtype=torch.bool)
    segment_padding_mask[0, :160000] = False
    segment_padding_mask[1, :32000] = False

    token_padding_mask = _build_m2d_token_padding_mask(
        model=fake_model,
        padding_mask=segment_padding_mask,
        batch_size=2,
        sequence_length=62,
        device=segment_padding_mask.device,
    )

    assert token_padding_mask.shape == (2, 62)
    assert int((~token_padding_mask[0]).sum().item()) == 62
    assert int((~token_padding_mask[1]).sum().item()) == 13
    assert token_padding_mask[1, 13:].all()


def test_extract_passt_embeddings_transposes_fbanks_for_model() -> None:
    captured = {}

    class FakePaSST(torch.nn.Module):
        def forward(self, x):
            captured["shape"] = x.shape
            return torch.zeros((x.shape[0], 2)), torch.ones((x.shape[0], 3, 4))

    fake_fbanks = torch.arange(2 * 5 * 7, dtype=torch.float32).reshape(2, 5, 7)
    with patch(
        "dcase2026_task1.models.passt._preprocess_waveforms",
        return_value=fake_fbanks,
    ):
        features = _extract_passt_embeddings(
            FakePaSST(),
            torch.ones((2, 16)),
            torch.zeros((2, 16), dtype=torch.bool),
        )

    assert captured["shape"] == (2, 1, 7, 5)
    assert features.shape == (2, 3, 4)


def test_load_initial_training_state_dict_reads_lightning_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    expected = {"classifier.weight": torch.ones((1, 1))}
    torch.save({"state_dict": expected}, checkpoint_path)

    state_dict = load_initial_training_state_dict(checkpoint_path)

    assert state_dict.keys() == expected.keys()
    assert torch.equal(state_dict["classifier.weight"], expected["classifier.weight"])


def test_pool_embedding_sequence_means_over_sequence_dimension() -> None:
    pooled = pool_embedding_sequence(
        torch.tensor(
            [
                [[1.0, 3.0], [5.0, 7.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ]
        )
    )

    assert torch.allclose(pooled, torch.tensor([[3.0, 5.0], [4.0, 6.0]]))


def test_masked_mean_embedding_sequence_ignores_padding() -> None:
    pooled = masked_mean_embedding_sequence(
        torch.tensor(
            [
                [[1.0, 3.0], [5.0, 7.0]],
                [[2.0, 4.0], [100.0, 100.0]],
            ]
        ),
        embedding_padding_mask=torch.tensor(
            [
                [False, False],
                [False, True],
            ]
        ),
    )

    assert torch.allclose(pooled, torch.tensor([[3.0, 5.0], [2.0, 4.0]]))

def test_compute_hierarchical_metrics_match_text_eval_behavior() -> None:
    metrics = compute_hierarchical_metrics(["m-sp", "fx-a"], ["m-sp", "m-si"])

    assert metrics["hierarchical_precision"] == 0.6875
    assert metrics["hierarchical_recall"] == 0.6875
    assert metrics["hierarchical_f1"] == 0.6666666666666666


def test_compute_hierarchical_metrics_can_include_class_wise_metrics() -> None:
    metrics = compute_hierarchical_metrics(
        ["m-sp", "fx-a"],
        ["m-sp", "fx-o"],
        class_names=["m-sp", "fx-a"],
        include_class_wise=True,
    )

    assert metrics["class_wise_hierarchical"]["m-sp"] == {
        "hierarchical_precision": 1.0,
        "hierarchical_recall": 1.0,
        "hierarchical_f1": 1.0,
    }
    assert metrics["class_wise_hierarchical"]["fx-a"] == {
        "hierarchical_precision": 0.0,
        "hierarchical_recall": 0.375,
        "hierarchical_f1": 0.0,
    }


def test_compute_classification_metrics_includes_hierarchical_precision() -> None:
    logits = torch.tensor([[3.0, 1.0], [2.0, 4.0]])
    labels = torch.tensor([0, 1])

    metrics = compute_classification_metrics(
        logits.numpy(),
        labels.numpy(),
        num_labels=2,
        id2label={0: "m-sp", 1: "fx-a"},
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["hierarchical_precision"] == 1.0
    assert metrics["hierarchical_recall"] == 1.0
    assert metrics["hierarchical_f1"] == 1.0


def test_compute_classification_metrics_can_include_class_wise_hierarchical_metrics() -> None:
    logits = torch.tensor([[3.0, 1.0], [4.0, 2.0]])
    labels = torch.tensor([0, 1])

    metrics = compute_classification_metrics(
        logits.numpy(),
        labels.numpy(),
        num_labels=2,
        id2label={0: "m-sp", 1: "fx-a"},
        include_class_wise_hierarchical=True,
    )

    assert set(metrics["class_wise_hierarchical"]) == {"m-sp", "fx-a"}
    assert metrics["class_wise_hierarchical"]["m-sp"]["hierarchical_precision"] == 0.5
    assert metrics["class_wise_hierarchical"]["fx-a"]["hierarchical_recall"] == 0.0


def test_apply_hard_llm_constraints_masks_to_allowed_labels() -> None:
    logits = torch.tensor([[3.0, 5.0, 1.0], [2.0, 1.0, 4.0]])
    id2label = {0: "m-sp", 1: "fx-a", 2: "ss-n"}

    constrained = apply_hard_llm_constraints(
        logits.numpy(),
        metadata=[
            {
                "metadata_class_probabilities": [
                    {"label": "m-sp", "probability": 0.6},
                    {"label": "other", "probability": 0.4},
                ]
            },
            {
                "metadata_class_probabilities": [
                    {"label": "fx-a", "probability": 0.3},
                    {"label": "ss-n", "probability": 0.7},
                ]
            },
        ],
        id2label=id2label,
    )

    assert constrained.argmax(axis=-1).tolist() == [0, 2]
    assert constrained[0, 1] < -1e100
    assert constrained[0, 2] < -1e100


def test_apply_soft_llm_constraints_adds_llm_log_prior() -> None:
    logits = torch.tensor([[4.0, 4.1, 0.0], [0.5, 0.2, 0.1]])
    id2label = {0: "m-sp", 1: "fx-a", 2: "ss-n"}

    constrained = apply_soft_llm_constraints(
        logits.numpy(),
        metadata=[
            {
                "metadata_class_probabilities": [
                    {"label": "m-sp", "probability": 0.9},
                    {"label": "fx-a", "probability": 0.1},
                    {"label": "other", "probability": 0.0},
                ]
            },
            {"metadata_class_probabilities": None},
        ],
        id2label=id2label,
    )

    assert constrained.argmax(axis=-1).tolist() == [0, 0]
    assert constrained[1].tolist() == logits.numpy()[1].tolist()


def test_build_llm_prior_weights_matches_filtered_normalized_metadata_probabilities() -> None:
    weights = build_llm_prior_weights(
        metadata=[
            {
                "metadata_class_probabilities": [
                    {"label": "m-sp", "probability": 3.0},
                    {"label": "fx-a", "probability": 1.0},
                    {"label": "other", "probability": 9.0},
                    {"label": "missing", "probability": 4.0},
                ]
            },
            {"metadata_class_probabilities": None},
        ],
        id2label={0: "m-sp", 1: "fx-a", 2: "ss-n"},
    )

    assert torch.allclose(weights[0], torch.tensor([0.75, 0.25, 0.0]))
    assert torch.allclose(weights[1], torch.tensor([0.0, 0.0, 0.0]))


def test_build_prediction_head_is_multilayer() -> None:
    head = build_prediction_head(input_dim=4, output_dim=3, dropout=0.1)
    outputs = head(torch.ones((2, 4)))

    assert isinstance(head, torch.nn.Sequential)
    assert isinstance(head[0], torch.nn.Linear)
    assert isinstance(head[1], torch.nn.GELU)
    assert isinstance(head[2], torch.nn.Dropout)
    assert isinstance(head[3], torch.nn.Linear)
    assert outputs.shape == (2, 3)


def test_build_prediction_head_supports_configurable_hidden_layers() -> None:
    head = build_prediction_head(
        input_dim=4,
        output_dim=3,
        dropout=0.1,
        hidden_layers=2,
        hidden_dim=5,
    )
    outputs = head(torch.ones((2, 4)))

    assert isinstance(head, torch.nn.Sequential)
    assert head[0].in_features == 4
    assert head[0].out_features == 5
    assert head[3].in_features == 5
    assert head[3].out_features == 5
    assert head[6].in_features == 5
    assert head[6].out_features == 3
    assert outputs.shape == (2, 3)


def test_build_prediction_head_supports_no_hidden_layers() -> None:
    head = build_prediction_head(
        input_dim=4,
        output_dim=3,
        dropout=0.1,
        hidden_layers=0,
        hidden_dim=5,
    )

    assert len(head) == 1
    assert head[0].in_features == 4
    assert head[0].out_features == 3


def test_epochs_to_update_steps_converts_fractional_epochs() -> None:
    assert epochs_to_update_steps(0.0, 10) == 0
    assert epochs_to_update_steps(1.5, 10) == 15
    assert epochs_to_update_steps(None, 10) is None


def test_build_lr_lambda_supports_warmup_constant_and_decay() -> None:
    lr_lambda = build_lr_lambda(
        warmup_steps=3,
        decay_start_step=5,
        total_steps=10,
        min_lr_scale=0.0,
    )

    assert lr_lambda(0) == 0.0
    assert lr_lambda(1) == 0.5
    assert lr_lambda(2) == 1.0
    assert lr_lambda(4) == 1.0
    assert lr_lambda(5) == 1.0
    assert lr_lambda(7) == 0.5
    assert lr_lambda(9) == 0.0
    assert lr_lambda(10) == 0.0


def test_build_lr_lambda_keeps_constant_lr_without_decay() -> None:
    lr_lambda = build_lr_lambda(
        warmup_steps=0,
        decay_start_step=None,
        total_steps=10,
        min_lr_scale=0.0,
    )

    assert lr_lambda(0) == 1.0
    assert lr_lambda(100) == 1.0


def test_build_lr_lambda_respects_min_learning_rate() -> None:
    lr_lambda = build_lr_lambda(
        warmup_steps=0,
        decay_start_step=2,
        total_steps=6,
        min_lr_scale=0.25,
    )

    assert lr_lambda(0) == 1.0
    assert lr_lambda(2) == 1.0
    assert lr_lambda(5) == 0.25
    assert lr_lambda(6) == 0.25


def test_run_experiment_tests_last_trained_parameters(tmp_path) -> None:
    records = [
        {
            "sound_id": index,
            "class_idx": index % 2,
            "class": f"class-{index % 2}",
            "source_dataset": "BSD10k",
            "audio_path": Path(f"/tmp/sample_{index}.wav"),
        }
        for index in range(6)
    ]
    initial_checkpoint_path = tmp_path / "initial.ckpt"
    initial_checkpoint_path.write_bytes(b"checkpoint")
    test_call: dict[str, object] = {}

    class FakeLightningModule(torch.nn.Module):
        def save_hyperparameters(self, *_args, **_kwargs) -> None:
            return None

        def log(self, *_args, **_kwargs) -> None:
            return None

        def log_dict(self, *_args, **_kwargs) -> None:
            return None

    class FakeTrainer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.strategy = SimpleNamespace(root_device=torch.device("cpu"))

        def fit(self, model, datamodule=None) -> None:
            test_call["fit_model"] = model
            test_call["fit_datamodule"] = datamodule

        def test(self, model=None, datamodule=None, ckpt_path=None):
            test_call["test_model"] = model
            test_call["test_datamodule"] = datamodule
            test_call["ckpt_path"] = ckpt_path
            return [{"test/accuracy": 0.5}]

    class FakeLightningNamespace:
        LightningDataModule = object
        LightningModule = FakeLightningModule
        Trainer = FakeTrainer

    class FakeModelCheckpoint:
        def __init__(self, **_kwargs) -> None:
            self.best_model_path = str(tmp_path / "best.ckpt")
            self.best_model_score = torch.tensor(0.25)

    class FakeEarlyStopping:
        instances: list["FakeEarlyStopping"] = []

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.__class__.instances.append(self)

    class FakeLearningRateMonitor:
        def __init__(self, **_kwargs) -> None:
            return None

    class FakeBeats(torch.nn.Module):
        def __init__(self, _config) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1)

        def load_state_dict(self, _state_dict, strict=False):
            return [], []

    args = Namespace(
        bsd10k_root=str(tmp_path / "bsd10k"),
        bsd35k_root=str(tmp_path / "bsd35k"),
        bsd2k_root=str(tmp_path / "bsd2k"),
        include_bsd35k_cs=False,
        only_bsd35k_cs=False,
        embedding_model="beats",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        init_checkpoint_path=str(initial_checkpoint_path),
        trust_checkpoint=True,
        fold=0,
        n_splits=5,
        validation_size=0.2,
        seed=42,
        max_train_items=2,
        max_val_items=2,
        max_test_items=2,
        batch_size=2,
        num_workers=0,
        learning_rate=3e-5,
        weight_decay=0.01,
        head_dropout=0.1,
        head_hidden_layers=1,
        head_hidden_dim=None,
        label_smoothing=0.2,
        use_class_frequency_loss=True,
        max_epochs=1,
        early_stopping_patience=10,
        warmup_epochs=0.0,
        lr_decay_start_epoch=None,
        min_learning_rate=0.0,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        freeze_encoder=False,
        save_checkpoints=True,
        precision="32-true",
        devices="1",
        accelerator="cpu",
        output_root=str(tmp_path / "outputs"),
        wandb_project="disabled-project",
        wandb_entity=None,
        wandb_mode="disabled",
        use_llm_prior_embedding_fusion=False,
    )

    with (
        patch(
            "dcase2026_task1.experiments.training._get_lightning_runtime",
            return_value=(
                FakeLightningNamespace,
                FakeModelCheckpoint,
                FakeEarlyStopping,
                FakeLearningRateMonitor,
                object,
            ),
        ),
        patch(
            "dcase2026_task1.experiments.training._get_progress_bar_callback",
            return_value=object(),
        ),
        patch(
            "dcase2026_task1.experiments.training.get_experiment_records",
            return_value=(records[:2], records[2:4], records[4:6]),
        ),
        patch(
            "dcase2026_task1.experiments.training.load_full_dataset_records",
            return_value=records[:2],
        ),
        patch(
            "dcase2026_task1.experiments.training.predict_logits_for_records",
            return_value=(["1", "2"], torch.zeros((2, 2)).numpy()),
        ),
        patch("dcase2026_task1.experiments.training.write_logits_npz"),
        patch(
            "dcase2026_task1.models.beats.resolve_checkpoint_path",
            return_value=tmp_path / "beats.pt",
        ),
        patch("dcase2026_task1.models.beats.validate_checkpoint_file"),
        patch(
            "torch.load",
            side_effect=[
                {"cfg": {"encoder_embed_dim": 1}, "model": {}},
                {"state_dict": {"fusion_head.3.weight": torch.ones((2, 2))}},
                {"state_dict": {}},
            ],
        ),
        patch("dcase2026_task1.models.beats.BEATsConfig", side_effect=lambda cfg: SimpleNamespace(**cfg)),
        patch("dcase2026_task1.models.beats.BEATs", FakeBeats),
        patch("pathlib.Path.exists", return_value=True),
        patch.object(FakeLightningModule, "load_state_dict", return_value=None),
    ):
        run_experiment(args)

    assert test_call["ckpt_path"] == str(tmp_path / "best.ckpt")
    assert test_call["test_model"] is test_call["fit_model"]
    assert test_call["test_datamodule"] is test_call["fit_datamodule"]
    assert isinstance(test_call["fit_model"].fusion_head, torch.nn.Sequential)
    assert test_call["fit_model"].fusion_head[0].in_features == 1
    assert test_call["fit_model"].fusion_head[-1].out_features == 2
    assert not hasattr(test_call["fit_model"], "classifier")
    assert test_call["fit_model"].loss_fn.label_smoothing == 0.2
    assert torch.allclose(test_call["fit_model"].loss_fn.weight, torch.ones(2))
    loaded_initial_state = FakeLightningModule.load_state_dict.call_args_list[0].args[0]
    assert torch.equal(loaded_initial_state["fusion_head.3.weight"], torch.ones((2, 2)))
    assert len(FakeEarlyStopping.instances) == 1
    assert FakeEarlyStopping.instances[0].kwargs == {
        "monitor": "val/hierarchical_f1",
        "mode": "max",
        "patience": 10,
    }
