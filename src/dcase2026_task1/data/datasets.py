from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

DEFAULT_BSD35K_ROOT = Path.home() / "data" / "BSD35k-CS"
DEFAULT_BSD10K_ROOT = Path.home() / "data" / "BSD10k"
DEFAULT_BSD2K_ROOT = Path.home() / "data" / "BSD2k"
DEFAULT_BSD10K_METADATA_CLASS_PROBABILITIES_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "outputs"
    / "experiments"
    / "20260611_211801_BSD10k_gpt-5.4-mini_f5706ed2"
    / "predictions.jsonl"
)

DEFAULT_BSD2K_METADATA_CLASS_PROBABILITIES_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "outputs"
    / "experiments"
    / "20260611_211913_BSD2k_gpt-5.4-mini_b628ef79"
    / "predictions.jsonl"
)

DEFAULT_BSD35K_METADATA_CLASS_PROBABILITIES_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "outputs"
    / "experiments"
    / "20260611_220253_BSD35k-CS_gpt-5.4-mini_f51027cc"
    / "predictions.jsonl"
)

DEFAULT_METADATA_CLASS_PROBABILITIES_PATHS = {
    "BSD10k": DEFAULT_BSD10K_METADATA_CLASS_PROBABILITIES_PATH,
    "BSD2k": DEFAULT_BSD2K_METADATA_CLASS_PROBABILITIES_PATH,
    "BSD35k-CS": DEFAULT_BSD35K_METADATA_CLASS_PROBABILITIES_PATH,
}


def load_audio_waveform(audio_path: str | Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "Loading audio requires soundfile to be installed."
        ) from exc

    waveform, sample_rate = sf.read(str(audio_path), always_2d=False, dtype="float32")
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    elif waveform.ndim == 2:
        waveform = waveform.T
    else:
        raise RuntimeError(
            f"Expected mono or stereo audio at {audio_path}, got shape {waveform.shape!r}."
        )
    return waveform, int(sample_rate)


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
        self.class_descriptions = (
            self._load_description_index(self.spec.description_csv)
            if self._requires_class_descriptions()
            else {}
        )
        self.extra_metadata_by_index = self._load_extra_metadata_by_index()
        self.records = self._load_records()

    def _requires_class_descriptions(self) -> bool:
        return self.spec.name != "BSD2k"

    def _validate_layout(self) -> None:
        expected_paths = [
            self.spec.root,
            self.spec.audio_dir,
            self.spec.metadata_dir,
            self.spec.metadata_csv,
        ]
        if self._requires_class_descriptions():
            expected_paths.append(self.spec.description_csv)
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

    @staticmethod
    def _read_jsonl_rows(jsonl_path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, dict):
                    rows.append(data)
        return rows

    @staticmethod
    def _parse_json_string(value: Any) -> Any | None:
        if not isinstance(value, str):
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    def _load_extra_metadata_by_index(self) -> dict[int, dict[str, Any]]:
        return self._load_class_probability_metadata()

    def _load_class_probability_metadata(self) -> dict[int, dict[str, Any]]:
        jsonl_path = DEFAULT_METADATA_CLASS_PROBABILITIES_PATHS.get(self.spec.name)
        if jsonl_path is None or not jsonl_path.exists():
            return {}

        probabilities_by_index: dict[int, dict[str, Any]] = {}
        for row in self._read_jsonl_rows(jsonl_path):
            dataset_index = row.get("dataset_index")
            if not isinstance(dataset_index, int):
                continue
            raw_response = row.get("raw_response")
            probabilities_by_index[dataset_index] = {
                "metadata_class_probabilities_raw": (
                    raw_response if isinstance(raw_response, str) else None
                ),
                "metadata_class_probabilities": self._parse_json_string(raw_response),
            }
        return probabilities_by_index

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
        for dataset_index, row in enumerate(rows):
            anonymous_id = row.get("anonymous_id")
            sound_id_value = row.get("sound_id")
            sound_id = int(sound_id_value) if sound_id_value not in (None, "") else None
            class_idx_value = row.get("class_idx")
            class_idx = int(class_idx_value) if class_idx_value not in (None, "") else None
            audio_path = self._resolve_audio_path(sound_id=sound_id, anonymous_id=anonymous_id)
            class_description = self._resolve_class_description(class_idx)

            extra_metadata = self.extra_metadata_by_index.get(dataset_index, {})
            metadata_class_probabilities_raw = extra_metadata.get(
                "metadata_class_probabilities_raw"
            )
            metadata_class_probabilities = extra_metadata.get(
                "metadata_class_probabilities"
            )

            record = {
                "dataset_index": dataset_index,
                "anonymous_id": anonymous_id,
                "sound_id": sound_id,
                "class": row.get("class"),
                "class_idx": class_idx,
                "class_top": row.get("class_top"),
                "confidence": row.get("confidence"),
                "uploader": row.get("uploader"),
                "license": row.get("license"),
                "title": row.get("title"),
                "tags": row.get("tags"),
                "description": row.get("description"),
                "audio_path": str(audio_path),
                "source_dataset": self.spec.name,
                "metadata_class_probabilities_raw": metadata_class_probabilities_raw,
                "metadata_class_probabilities": metadata_class_probabilities,
                "metadata": {
                    "dataset_index": dataset_index,
                    "anonymous_id": anonymous_id,
                    "sound_id": sound_id,
                    "class": row.get("class"),
                    "class_idx": class_idx,
                    "class_top": row.get("class_top"),
                    "confidence": row.get("confidence"),
                    "uploader": row.get("uploader"),
                    "license": row.get("license"),
                    "title": row.get("title"),
                    "tags": row.get("tags"),
                    "description": row.get("description"),
                    "metadata_class_probabilities_raw": metadata_class_probabilities_raw,
                    "metadata_class_probabilities": metadata_class_probabilities,
                },
                "class_description": class_description,
                "description_class_key": (
                    class_description["class_key"] if class_description is not None else None
                ),
                "description_class_key_long": (
                    class_description["class_key_long"] if class_description is not None else None
                ),
                "description_top_level": (
                    class_description["top_level"] if class_description is not None else None
                ),
                "description_second_level": (
                    class_description["second_level"] if class_description is not None else None
                ),
                "description_text": (
                    class_description["description"] if class_description is not None else None
                ),
                "description_examples": (
                    class_description["examples"] if class_description is not None else None
                ),
            }
            records.append(record)
        return records

    def _resolve_audio_path(self, sound_id: int | None, anonymous_id: str | None) -> Path:
        if sound_id is not None:
            return self.spec.audio_dir / f"{sound_id}.wav"
        if anonymous_id in (None, ""):
            raise KeyError(
                f"Expected one of sound_id or anonymous_id in {self.spec.metadata_csv}"
            )

        anonymous_path = Path(anonymous_id)
        filename = (
            anonymous_path.name if anonymous_path.suffix else f"{anonymous_path.name}.wav"
        )
        return self.spec.audio_dir / filename

    def _resolve_class_description(self, class_idx: int | None) -> dict[str, Any] | None:
        if class_idx is None:
            return None

        class_description = self.class_descriptions.get(class_idx)
        if class_description is None:
            raise KeyError(
                f"Missing description entry for class_idx={class_idx} "
                f"in {self.spec.description_csv}"
            )
        return class_description

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.records[index])
        audio_path = Path(item["audio_path"])

        if self.load_audio:
            waveform, sample_rate = load_audio_waveform(audio_path)
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
