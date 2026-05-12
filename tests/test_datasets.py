from __future__ import annotations

from pathlib import Path
from pprint import pprint

from dcase2026_task1.data.datasets import BSDDataset

DEFAULT_BSD35K_ROOT = str(Path.home() / "data" / "BSD35k-CS")
DEFAULT_BSD10K_ROOT = str(Path.home() / "data" / "BSD10k")


def _example_view(item: dict) -> dict:
    return {
        "source_dataset": item["source_dataset"],
        "sound_id": item["sound_id"],
        "class": item["class"],
        "class_idx": item["class_idx"],
        "title": item["title"],
        "audio_path": item["audio_path"],
        "class_description": item["class_description"],
    }


def test_bsd35k_dataset_example() -> None:
    dataset = BSDDataset(
        root=DEFAULT_BSD35K_ROOT,
        dataset_name="BSD35k-CS",
        load_audio=False,
    )

    assert len(dataset) > 0
    item = dataset[0]
    assert item["source_dataset"] == "BSD35k-CS"
    assert item["audio_path"].endswith(".wav")
    assert "metadata" in item
    assert "class_description" in item

    print("\nBSD35k-CS example:")
    pprint(_example_view(item))


def test_bsd10k_dataset_example() -> None:
    dataset = BSDDataset(
        root=DEFAULT_BSD10K_ROOT,
        dataset_name="BSD10k",
        load_audio=False,
    )

    assert len(dataset) > 0
    item = dataset[0]
    assert item["source_dataset"] == "BSD10k"
    assert item["audio_path"].endswith(".wav")
    assert "metadata" in item
    assert "class_description" in item

    print("\nBSD10k example:")
    pprint(_example_view(item))
