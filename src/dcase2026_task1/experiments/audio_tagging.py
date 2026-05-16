from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import uuid4

from tqdm import tqdm

from dcase2026_task1.data.datasets import DEFAULT_BSD10K_ROOT, DEFAULT_BSD35K_ROOT, BSDDataset
from dcase2026_task1.models import AudioSetTaggingModel, AudioTaggingInput


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an AudioSet audio-tagging model and store per-label probabilities."
    )
    parser.add_argument("--dataset", choices=["BSD10k", "BSD35k-CS"], default="BSD10k")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--model-id", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-root", default="outputs/audio_tagging")
    return parser


def resolve_dataset_root(dataset_name: str, explicit_root: str | None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root)
    if dataset_name == "BSD10k":
        return DEFAULT_BSD10K_ROOT
    return DEFAULT_BSD35K_ROOT


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size < 1:
        raise ValueError("batch_size must be >= 1.")
    iterator = iter(items)
    batches: list[list[Any]] = []
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return batches
        batches.append(batch)


def create_experiment_dir(output_root: Path, dataset_name: str) -> Path:
    experiment_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}_audioset_{uuid4().hex[:8]}"
    )
    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def run_experiment(args: argparse.Namespace) -> Path:
    dataset_root = resolve_dataset_root(args.dataset, args.dataset_root)
    dataset = BSDDataset(root=dataset_root, dataset_name=args.dataset, load_audio=False)
    model = AudioSetTaggingModel(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    output_root = Path(args.output_root)
    experiment_dir = create_experiment_dir(output_root, args.dataset)

    limit = len(dataset) if args.max_items is None else min(len(dataset), args.max_items)

    config = {
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "model_id": args.model_id,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "batch_size": args.batch_size,
        "num_items": limit,
        "top_k": args.top_k,
    }
    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    predictions_path = experiment_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for batch_indices in tqdm(chunked(list(range(limit)), args.batch_size), desc="Batches", unit="batch"):
            batch_items = [dataset[index] for index in batch_indices]
            batch_inputs = [
                AudioTaggingInput(audio_path=item["audio_path"])
                for item in batch_items
            ]
            batch_outputs = model.predict_batch_outputs(batch_inputs)

            for index, item, model_output in zip(
                batch_indices,
                batch_items,
                batch_outputs,
                strict=True,
            ):
                sorted_scores = sorted(model_output.scores, key=lambda score: score.score, reverse=True)
                row = {
                    "dataset_index": index,
                    "sound_id": item["sound_id"],
                    "source_dataset": item["source_dataset"],
                    "audio_path": item["audio_path"],
                    "title": item["title"],
                    "tags": item["tags"],
                    "description": item["description"],
                    "target_class_idx": int(item["class_idx"]),
                    "target_class": item["class"],
                    "top_tags": [asdict(score) for score in sorted_scores[: args.top_k]],
                    "class_probabilities": [asdict(score) for score in model_output.scores],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    return experiment_dir


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = run_experiment(args)
    print(f"Wrote experiment outputs to {experiment_dir}")


if __name__ == "__main__":
    main()
