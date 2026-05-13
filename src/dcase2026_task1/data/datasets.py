from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, Dataset

DEFAULT_BSD35K_ROOT = Path.home() / "data" / "BSD35k-CS"
DEFAULT_BSD10K_ROOT = Path.home() / "data" / "BSD10k"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def metadata_csv(self) -> Path:
        return self.metadata_dir / f"{self.name}_metadata.csv"

    @property
    def description_csv(self) -> Path:
        preferred = self.metadata_dir / "BST_description.csv"
        fallback = self.metadata_dir / "BTS_description.csv"
        if preferred.exists():
            return preferred
        return fallback


class BSDDataset(Dataset[dict[str, Any]]):
    """PyTorch dataset for a single BSD dataset root."""

    def __init__(self, root: str | Path, dataset_name: str, load_audio: bool = True) -> None:
        self.spec = DatasetSpec(name=dataset_name, root=Path(root))
        self.load_audio = load_audio

        self._validate_layout()
        self.class_descriptions = self._load_description_index(self.spec.description_csv)
        self.records = self._load_records()

    def _validate_layout(self) -> None:
        expected_paths = [
            self.spec.root,
            self.spec.audio_dir,
            self.spec.metadata_dir,
            self.spec.metadata_csv,
            self.spec.description_csv,
        ]
        missing = [str(path) for path in expected_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing dataset files or folders for "
                f"{self.spec.name}: {', '.join(missing)}"
            )

    @staticmethod
    def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    def _load_description_index(self, csv_path: Path) -> dict[int, dict[str, Any]]:
        rows = self._read_csv_rows(csv_path)
        descriptions: dict[int, dict[str, Any]] = {}
        for row in rows:
            class_idx = int(row["class_idx"])
            descriptions[class_idx] = {
                "class_key": row["class_key"],
                "class_idx": class_idx,
                "class_key_long": row["class_key_long"],
                "top_level": row["top_level"],
                "second_level": row["second_level"],
                "description": row["description"],
                "examples": row["examples"],
            }
        return descriptions

    def _load_records(self) -> list[dict[str, Any]]:
        rows = self._read_csv_rows(self.spec.metadata_csv)
        records: list[dict[str, Any]] = []
        for row in rows:
            class_idx = int(row["class_idx"])
            sound_id = int(row["sound_id"])
            audio_path = self.spec.audio_dir / f"{sound_id}.wav"
            class_description = self.class_descriptions.get(class_idx)
            if class_description is None:
                raise KeyError(
                    f"Missing description entry for class_idx={class_idx} "
                    f"in {self.spec.description_csv}"
                )

            record = {
                "sound_id": sound_id,
                "class": row["class"],
                "class_idx": class_idx,
                "class_top": row["class_top"],
                "confidence": row["confidence"],
                "uploader": row["uploader"],
                "license": row["license"],
                "title": row["title"],
                "tags": row["tags"],
                "description": row["description"],
                "audio_path": str(audio_path),
                "source_dataset": self.spec.name,
                "metadata": {
                    "sound_id": sound_id,
                    "class": row["class"],
                    "class_idx": class_idx,
                    "class_top": row["class_top"],
                    "confidence": row["confidence"],
                    "uploader": row["uploader"],
                    "license": row["license"],
                    "title": row["title"],
                    "tags": row["tags"],
                    "description": row["description"],
                },
                "class_description": class_description,
                "description_class_key": class_description["class_key"],
                "description_class_key_long": class_description["class_key_long"],
                "description_top_level": class_description["top_level"],
                "description_second_level": class_description["second_level"],
                "description_text": class_description["description"],
                "description_examples": class_description["examples"],
            }
            records.append(record)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.records[index])
        audio_path = Path(item["audio_path"])

        if self.load_audio:
            try:
                import torchaudio
            except ImportError as exc:
                raise ImportError(
                    "BSDDataset with load_audio=True requires torchaudio to be installed."
                ) from exc
            waveform, sample_rate = torchaudio.load(audio_path)
            item["waveform"] = waveform
            item["sample_rate"] = sample_rate

        return item


class BSDCombinedDataset(ConcatDataset):
    """Combined dataset over BSD35k-CS and BSD10k."""

    def __init__(
        self,
        bsd35k_root: str | Path = DEFAULT_BSD35K_ROOT,
        bsd10k_root: str | Path = DEFAULT_BSD10K_ROOT,
        load_audio: bool = True,
    ) -> None:
        self.datasets_by_name = {
            "BSD35k-CS": BSDDataset(
                root=bsd35k_root,
                dataset_name="BSD35k-CS",
                load_audio=load_audio,
            ),
            "BSD10k": BSDDataset(
                root=bsd10k_root,
                dataset_name="BSD10k",
                load_audio=load_audio,
            ),
        }
        super().__init__(list(self.datasets_by_name.values()))

    @property
    def records(self) -> list[dict[str, Any]]:
        combined_records: list[dict[str, Any]] = []
        for dataset in self.datasets:
            combined_records.extend(dataset.records)
        return combined_records
