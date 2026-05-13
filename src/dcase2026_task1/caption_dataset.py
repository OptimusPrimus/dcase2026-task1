from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dcase2026_task1.data.datasets import DEFAULT_BSD10K_ROOT, BSDDataset
from dcase2026_task1.models import AudioFlamingo3AudioCaptioningSkill, AudioFlamingo3Model
from dcase2026_task1.tasks import AudioCaptioningTask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create audio captions for every BSD10k example and store them in a CSV file."
    )
    parser.add_argument(
        "--bsd10k-root",
        default=str(DEFAULT_BSD10K_ROOT),
        help="Root directory of the BSD10k dataset.",
    )
    parser.add_argument(
        "--model-id",
        default="nvidia/audio-flamingo-3-hf",
        help="Model identifier passed to the backend.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Transformers device_map value for the model backend.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on examples for smoke runs.",
    )
    parser.add_argument(
        "--output",
        default="outputs/audio_captions.csv",
        help="Path of the CSV file to write.",
    )
    return parser


def write_captions(
    dataset: BSDDataset,
    model: AudioFlamingo3Model,
    output_path: Path,
    max_items: int | None = None,
) -> None:
    task = AudioCaptioningTask()
    skill = AudioFlamingo3AudioCaptioningSkill(task)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limit = len(dataset) if max_items is None else min(len(dataset), max_items)
    fieldnames = [
        "dataset_index",
        "sound_id",
        "source_dataset",
        "audio_path",
        "title",
        "tags",
        "description",
        "target_class_idx",
        "target_class",
        "caption",
        "raw_response",
        "final_response",
        "reasoning",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index in range(limit):
            item = dataset[index]
            response = model.predict(item, skill)
            writer.writerow(
                {
                    "dataset_index": index,
                    "sound_id": item["sound_id"],
                    "source_dataset": item["source_dataset"],
                    "audio_path": item["audio_path"],
                    "title": item["title"],
                    "tags": item["tags"],
                    "description": item["description"],
                    "target_class_idx": int(item["class_idx"]),
                    "target_class": item["class"],
                    "caption": response.caption,
                    "raw_response": response.raw_response,
                    "final_response": response.final_response,
                    "reasoning": response.reasoning,
                }
            )
            handle.flush()

            if (index + 1) % 10 == 0 or index + 1 == limit:
                print(f"Captioned {index + 1}/{limit} items.")


def main() -> None:
    args = build_parser().parse_args()
    dataset = BSDDataset(
        root=args.bsd10k_root,
        dataset_name="BSD10k",
        load_audio=False,
    )
    model = AudioFlamingo3Model(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    output_path = Path(args.output)

    print(f"Loaded {len(dataset)} items from BSD10k.")
    write_captions(
        dataset=dataset,
        model=model,
        output_path=output_path,
        max_items=args.max_items,
    )
    print(f"Wrote captions to {output_path}")


if __name__ == "__main__":
    main()
