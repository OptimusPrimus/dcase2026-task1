from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dcase2026_task1.data.splits import (
    DEFAULT_BSD_SPLIT_SEED,
    build_experiment_split,
    get_experiment_records,
)
from dcase2026_task1.experiments.beats_finetuning import (
    DEFAULT_CHECKPOINT_ALIAS,
    build_id2label,
    build_label_map,
    build_lr_lambda,
    build_label_specs,
    compute_classification_metrics,
    compute_hierarchical_metrics,
    epochs_to_update_steps,
    mean_segment_logits,
    maybe_limit,
    resolve_checkpoint_path,
    resolve_dataset_roots,
    split_waveforms_into_segments,
    run_experiment,
)
from dcase2026_task1.models.beats import BEATs, BEATsConfig


def test_resolve_dataset_roots() -> None:
    roots = resolve_dataset_roots("/tmp/bsd10k", "/tmp/bsd35k")
    assert roots["BSD10k"].as_posix() == "/tmp/bsd10k"
    assert roots["BSD35k-CS"].as_posix() == "/tmp/bsd35k"


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


def test_maybe_limit() -> None:
    assert maybe_limit([1, 2, 3], None) == [1, 2, 3]
    assert maybe_limit([1, 2, 3], 2) == [1, 2]


def test_vendored_beats_package_exports_model_classes() -> None:
    assert BEATs.__name__ == "BEATs"
    assert BEATsConfig.__name__ == "BEATsConfig"


def test_beats_patchout_u_drops_tokens_only_during_training() -> None:
    config = BEATsConfig(
        {
            "input_patch_size": 1,
            "embed_dim": 2,
            "encoder_embed_dim": 2,
            "encoder_layers": 1,
            "encoder_ffn_embed_dim": 8,
            "encoder_attention_heads": 1,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "encoder_layerdrop": 0.0,
            "dropout_input": 0.0,
            "conv_pos": 1,
            "conv_pos_groups": 1,
            "patchout_u": 0.5,
        }
    )
    model = BEATs(config)

    captured: dict[str, torch.Tensor | None] = {}

    class FakeEncoder(torch.nn.Module):
        def forward(self, x, padding_mask=None):
            captured["x"] = x
            captured["padding_mask"] = padding_mask
            return x, []

    model.encoder = FakeEncoder()
    source = torch.zeros(2, 160)
    fake_fbank = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)

    with patch.object(model, "preprocess", return_value=fake_fbank):
        model.train()
        torch.manual_seed(0)
        train_features, train_padding_mask = model.extract_features(source)

        assert captured["x"] is not None
        assert train_features.shape == (2, 3, 2)
        assert train_padding_mask is None

        model.eval()
        eval_features, eval_padding_mask = model.extract_features(source)

    assert eval_features.shape == (2, 6, 2)
    assert eval_padding_mask is None


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
            fold=0,
            n_splits=5,
            validation_size=0.2,
        )

    assert all(record["source_dataset"] == "BSD10k" for record in split.val_records)
    assert all(record["source_dataset"] == "BSD10k" for record in split.test_records)
    assert split.noisy_train_size == len(noisy_records)
    assert sum(record["source_dataset"] == "BSD35k-CS" for record in split.train_records) == len(noisy_records)


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
            fold=1,
            n_splits=5,
            validation_size=0.2,
        )
        split_b = build_experiment_split(
            bsd10k_root=Path("/tmp/bsd10k"),
            bsd35k_root=None,
            include_bsd35k_cs=False,
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


def test_mean_segment_logits_averages_segments_per_sample() -> None:
    segment_logits = torch.tensor(
        [
            [1.0, 3.0],
            [5.0, 7.0],
            [2.0, 4.0],
        ]
    )
    segment_batch_indices = torch.tensor([0, 0, 1])

    logits = mean_segment_logits(segment_logits, segment_batch_indices, batch_size=2)

    assert torch.allclose(logits, torch.tensor([[3.0, 5.0], [2.0, 4.0]]))


def test_compute_hierarchical_metrics_match_text_eval_behavior() -> None:
    metrics = compute_hierarchical_metrics(["m-sp", "fx-a"], ["m-sp", "m-si"])

    assert metrics["hierarchical_precision"] == 0.6875
    assert metrics["hierarchical_recall"] == 0.6875
    assert metrics["hierarchical_f1"] == 0.6666666666666666


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
    test_call: dict[str, object] = {}

    class FakeLightningModule(torch.nn.Module):
        def save_hyperparameters(self, values) -> None:
            self.hparams = values

        def log(self, *_args, **_kwargs) -> None:
            return None

        def log_dict(self, *_args, **_kwargs) -> None:
            return None

    class FakeTrainer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

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
            self.best_model_path = "best.ckpt"
            self.best_model_score = torch.tensor(0.25)

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
        include_bsd35k_cs=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_alias=DEFAULT_CHECKPOINT_ALIAS,
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
        patchout_u=0.25,
        max_epochs=1,
        warmup_epochs=0.0,
        lr_decay_start_epoch=None,
        min_learning_rate=0.0,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        freeze_encoder=False,
        precision="32-true",
        devices="1",
        accelerator="cpu",
        output_root=str(tmp_path / "outputs"),
        wandb_project="disabled-project",
        wandb_entity=None,
        wandb_mode="disabled",
    )

    with (
        patch(
            "dcase2026_task1.experiments.beats_finetuning._get_lightning_runtime",
            return_value=(FakeLightningNamespace, FakeModelCheckpoint, FakeLearningRateMonitor, object),
        ),
        patch(
            "dcase2026_task1.experiments.beats_finetuning._get_progress_bar_callback",
            return_value=object(),
        ),
        patch(
            "dcase2026_task1.experiments.beats_finetuning.get_experiment_records",
            return_value=(records[:2], records[2:4], records[4:6]),
        ),
        patch(
            "dcase2026_task1.experiments.beats_finetuning.resolve_checkpoint_path",
            return_value=tmp_path / "beats.pt",
        ),
        patch("dcase2026_task1.experiments.beats_finetuning.validate_checkpoint_file"),
        patch(
            "torch.load",
            return_value={"cfg": {"encoder_embed_dim": 1}, "model": {}},
        ),
        patch("dcase2026_task1.experiments.beats_finetuning.BEATsConfig", side_effect=lambda cfg: SimpleNamespace(**cfg)),
        patch("dcase2026_task1.experiments.beats_finetuning.BEATs", FakeBeats),
    ):
        experiment_dir = run_experiment(args)

    assert test_call["ckpt_path"] is None
    assert test_call["test_model"] is test_call["fit_model"]
    assert test_call["test_datamodule"] is test_call["fit_datamodule"]
    assert test_call["fit_model"].hparams["patchout_u"] == 0.25

    config_path = experiment_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["training"]["patchout_u"] == 0.25
