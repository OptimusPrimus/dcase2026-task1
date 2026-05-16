from __future__ import annotations

import torch

from dcase2026_task1.experiments.beats_finetuning import (
    DEFAULT_CHECKPOINT_ALIAS,
    DEFAULT_BEATS_REPO_ROOT,
    OFFICIAL_CHECKPOINT_URLS,
    build_id2label,
    build_label_map,
    build_label_specs,
    compute_classification_metrics,
    compute_hierarchical_metrics,
    mean_segment_logits,
    maybe_limit,
    resolve_checkpoint_path,
    resolve_beats_module_dir,
    resolve_dataset_roots,
    resolve_dataset_selection,
    split_waveforms_into_segments,
)


def test_resolve_dataset_selection_aliases() -> None:
    ten_k = resolve_dataset_selection("10k")
    assert ten_k.canonical_name == "BSD10k"
    assert ten_k.dataset_names == ("BSD10k",)

    twenty_five_k = resolve_dataset_selection("25k")
    assert twenty_five_k.canonical_name == "BSD35k-CS"
    assert twenty_five_k.dataset_names == ("BSD35k-CS",)

    combined = resolve_dataset_selection("combined")
    assert combined.canonical_name == "combined"
    assert combined.dataset_names == ("BSD10k", "BSD35k-CS")


def test_resolve_dataset_roots_for_combined() -> None:
    selection = resolve_dataset_selection("combined")
    roots = resolve_dataset_roots(selection, "/tmp/bsd10k", "/tmp/bsd25k")
    assert roots["BSD10k"].as_posix() == "/tmp/bsd10k"
    assert roots["BSD35k-CS"].as_posix() == "/tmp/bsd25k"


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


def test_resolve_beats_module_dir_from_repo_root(tmp_path) -> None:
    beats_dir = tmp_path / "beats"
    beats_dir.mkdir()
    (beats_dir / "BEATs.py").write_text("", encoding="utf-8")
    (beats_dir / "backbone.py").write_text("", encoding="utf-8")

    assert resolve_beats_module_dir(tmp_path) == beats_dir.resolve()


def test_default_beats_repo_root_points_to_vendored_model() -> None:
    resolved = resolve_beats_module_dir(DEFAULT_BEATS_REPO_ROOT)
    assert resolved == DEFAULT_BEATS_REPO_ROOT.resolve()


def test_resolve_checkpoint_path_from_explicit_path(tmp_path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_text("x", encoding="utf-8")

    resolved = resolve_checkpoint_path(
        checkpoint_path=str(checkpoint_path),
        checkpoint_dir=tmp_path / "cache",
        checkpoint_alias=DEFAULT_CHECKPOINT_ALIAS,
        checkpoint_url=None,
    )

    assert resolved == checkpoint_path.resolve()


def test_official_checkpoint_alias_is_available() -> None:
    assert DEFAULT_CHECKPOINT_ALIAS in OFFICIAL_CHECKPOINT_URLS


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
