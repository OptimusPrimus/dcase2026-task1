from __future__ import annotations

import json
from pathlib import Path
from pprint import pprint

import pytest

from dcase2026_task1.data.datasets import BSDDataset

DEFAULT_BSD35K_ROOT = str(Path.home() / "data" / "BSD35k-CS")
DEFAULT_BSD10K_ROOT = str(Path.home() / "data" / "BSD10k")
DEFAULT_BSD2K_ROOT = str(Path.home() / "data" / "BSD2k")


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


def _write_dataset_fixture(root: Path, dataset_name: str) -> None:
    (root / "audio").mkdir(parents=True)
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True)

    (metadata_dir / f"{dataset_name}_metadata.csv").write_text(
        "\n".join(
            [
                "sound_id,class,class_idx,class_top,confidence,uploader,license,title,tags,description",
                "123,fx-o,401,fx,1.0,user,cc0,clip one,tag-a,desc one",
                "456,is-w,203,is,0.9,user,cc0,clip two,tag-b,desc two",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (metadata_dir / "BST_description.csv").write_text(
        "\n".join(
            [
                "class_idx,class_key,class_key_long,top_level,second_level,description,examples",
                "401,fx-o,Sound effects / Objects and household appliances,fx,objects,Object sounds,door close",
                "203,is-w,Instrument sample / Wind,is,wind,Wind instrument sample,flute note",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_bsd2k_fixture(root: Path) -> None:
    (root / "audio").mkdir(parents=True)
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True)

    (metadata_dir / "BSD2k_metadata.csv").write_text(
        "\n".join(
            [
                "anonymous_id,title,tags,description",
                "anon-001,clip one,tag-a,desc one",
                "anon-002.wav,clip two,tag-b,desc two",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("dataset_name", ["BSD10k", "BSD35k-CS", "BSD2k"])
def test_dataset_loads_class_probability_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dataset_name: str
) -> None:
    dataset_root = tmp_path / dataset_name
    if dataset_name == "BSD2k":
        _write_bsd2k_fixture(dataset_root)
    else:
        _write_dataset_fixture(dataset_root, dataset_name)

    class_probs_path = tmp_path / "class_probs.jsonl"
    class_probs_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset_index": 0,
                        "raw_response": (
                            '[{"label":"fx-o","probability":0.8},{"label":"other","probability":0.2}]'
                        ),
                    }
                ),
                json.dumps({"dataset_index": 1, "raw_response": "not json"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary_path = tmp_path / "summaries.jsonl"
    summary_path.write_text(
        "\n".join(
            [
                json.dumps({"dataset_index": 0, "raw_response": "A short object sound."}),
                json.dumps({"dataset_index": 1, "raw_response": "A second summary."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "dcase2026_task1.data.datasets.DEFAULT_METADATA_CLASS_PROBABILITIES_PATHS",
        {dataset_name: class_probs_path},
    )
    monkeypatch.setattr(
        "dcase2026_task1.data.datasets.DEFAULT_METADATA_SUMMARY_PATHS",
        {dataset_name: summary_path},
    )

    dataset = BSDDataset(root=dataset_root, dataset_name=dataset_name, load_audio=False)

    item0 = dataset[0]
    assert item0["dataset_index"] == 0
    assert item0["metadata_class_probabilities_raw"] == (
        '[{"label":"fx-o","probability":0.8},{"label":"other","probability":0.2}]'
    )
    assert item0["metadata_class_probabilities"] == [
        {"label": "fx-o", "probability": 0.8},
        {"label": "other", "probability": 0.2},
    ]
    assert item0["metadata"]["metadata_class_probabilities"] == [
        {"label": "fx-o", "probability": 0.8},
        {"label": "other", "probability": 0.2},
    ]
    assert item0["metadata_summary_raw"] == "A short object sound."
    assert item0["metadata_summary"] == "A short object sound."
    assert item0["metadata"]["metadata_summary"] == "A short object sound."

    item1 = dataset[1]
    assert item1["dataset_index"] == 1
    assert item1["metadata_class_probabilities_raw"] == "not json"
    assert item1["metadata_class_probabilities"] is None
    assert item1["metadata_summary_raw"] == "A second summary."
    assert item1["metadata_summary"] == "A second summary."


def test_dataset_without_class_probability_metadata_keeps_extra_metadata_empty(tmp_path: Path) -> None:
    dataset_root = tmp_path / "BSD35k-CS"
    _write_dataset_fixture(dataset_root, "BSD35k-CS")

    dataset = BSDDataset(root=dataset_root, dataset_name="BSD35k-CS", load_audio=False)

    item = dataset[0]
    assert item["dataset_index"] == 0
    assert item["metadata_class_probabilities_raw"] is None
    assert item["metadata_class_probabilities"] is None
    assert item["metadata_summary_raw"] is None
    assert item["metadata_summary"] is None


def test_bsd2k_dataset_loads_reduced_metadata_schema(tmp_path: Path) -> None:
    dataset_root = tmp_path / "BSD2k"
    _write_bsd2k_fixture(dataset_root)

    dataset = BSDDataset(root=dataset_root, dataset_name="BSD2k", load_audio=False)

    item0 = dataset[0]
    assert item0["source_dataset"] == "BSD2k"
    assert item0["anonymous_id"] == "anon-001"
    assert item0["sound_id"] is None
    assert item0["class"] is None
    assert item0["class_idx"] is None
    assert item0["uploader"] is None
    assert item0["audio_path"].endswith("/audio/anon-001.wav")
    assert item0["class_description"] is None
    assert item0["description_class_key"] is None
    assert item0["metadata"]["anonymous_id"] == "anon-001"

    item1 = dataset[1]
    assert item1["audio_path"].endswith("/audio/anon-002.wav")
