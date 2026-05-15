from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import uuid4

from tqdm import tqdm

from dcase2026_task1.data.datasets import (
    DEFAULT_BSD10K_ROOT,
    DEFAULT_BSD35K_ROOT,
    BSDDataset,
)
from dcase2026_task1.models import GenerativeModel, ModelInput, OpenAIModel, QwenModel


@dataclass(frozen=True)
class CandidateClass:
    class_idx: int
    class_name: str
    description_top_level: str
    description_second_level: str
    description: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a metadata-only text classification experiment over one BSD dataset."
    )
    parser.add_argument("--dataset", choices=["BSD10k", "BSD35k-CS"], default="BSD10k")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--model", choices=["qwen", "openai"], default="openai")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Reserved dtype setting for the Qwen backend.",
    )
    parser.add_argument("--device", default="auto", help="Reserved device setting for the Qwen backend.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", default="outputs/experiments")
    return parser


def resolve_dataset_root(dataset_name: str, explicit_root: str | None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root)
    if dataset_name == "BSD10k":
        return DEFAULT_BSD10K_ROOT
    return DEFAULT_BSD35K_ROOT


def build_candidate_classes(dataset: BSDDataset) -> list[CandidateClass]:
    by_class_idx: dict[int, CandidateClass] = {}
    for record in dataset.records:
        class_idx = int(record["class_idx"])
        if class_idx in by_class_idx:
            continue
        by_class_idx[class_idx] = CandidateClass(
            class_idx=class_idx,
            class_name=str(record["class"]),
            description_top_level=str(record["description_top_level"]),
            description_second_level=str(record["description_second_level"]),
            description=str(record["description_text"]),
        )
    return [by_class_idx[idx] for idx in sorted(by_class_idx)]


def build_prompt(item: dict[str, Any], candidate_classes: list[CandidateClass]) -> str:
    class_lines = [
        (
            f"{index}. {candidate.class_name} "
            f"({candidate.description_top_level} -> {candidate.description_second_level}): "
            f"{candidate.description}"
        )
        for index, candidate in enumerate(candidate_classes, start=1)
    ]
    return (
        "You are classifying an audio event using metadata only.\n"
        "Choose the most liekly one option from the list below or 'unknown' if you have not enough information.\n"
        "The available classes are:\n"
        f"{chr(10).join(class_lines)}\n\n"
        "Clip metadata:\n"
        f'- title="{item.get("title", "")}"\n'
        f'- tags="{item.get("tags", "")}"\n'
        f'- description="{item.get("description", "")}"\n'
    )


def parse_prediction(
    raw_response: str,
    candidate_classes: list[CandidateClass],
) -> tuple[int | None, str | None]:
    import re

    option_match = re.search(r"\b(\d+)\b", raw_response)
    if option_match is None:
        return None, raw_response.strip() or None
    option_index = int(option_match.group(1))
    if not 1 <= option_index <= len(candidate_classes):
        return None, str(option_index)
    return candidate_classes[option_index - 1].class_idx, str(option_index)


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


def create_experiment_dir(output_root: Path, dataset_name: str, model_name: str) -> Path:
    experiment_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}_{model_name}_{uuid4().hex[:8]}"
    )
    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def load_text_model(args: argparse.Namespace) -> GenerativeModel:
    if args.model == "qwen":
        return QwenModel(
            model_id=args.model_id or "Qwen/Qwen3.6-27B",
            device=args.device,
            torch_dtype=args.torch_dtype,
            api_base=args.api_base,
            api_key=args.api_key,
            max_new_tokens=args.max_new_tokens,
        )
    if args.model == "openai":
        return OpenAIModel(
            model_id=args.model_id or "gpt-5.4-mini",
            api_key=args.api_key,
            base_url=args.api_base,
            max_new_tokens=args.max_new_tokens,
            temperature=0.2,
            top_p=0.85,
        )
    raise ValueError(f"Unsupported model backend: {args.model}")


def run_experiment(args: argparse.Namespace) -> Path:
    dataset_root = resolve_dataset_root(args.dataset, args.dataset_root)
    dataset = BSDDataset(root=dataset_root, dataset_name=args.dataset, load_audio=False)
    candidate_classes = build_candidate_classes(dataset)
    model = None if args.dry_run else load_text_model(args)
    output_root = Path(args.output_root)
    experiment_dir = create_experiment_dir(output_root, args.dataset, args.model)

    limit = len(dataset) if args.max_items is None else min(len(dataset), args.max_items)
    if args.dry_run and args.max_items is None:
        limit = min(limit, 5)

    config = {
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "model": args.model,
        "model_id": args.model_id,
        "api_base": args.api_base,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "dry_run": args.dry_run,
        "num_items": limit,
        "candidate_classes": [asdict(candidate) for candidate in candidate_classes],
    }
    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    predictions_path = experiment_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for batch_indices in tqdm(chunked(list(range(limit)), args.batch_size), desc="Batches", unit="batch"):
            batch_items = [dataset[index] for index in batch_indices]
            batch_inputs = [ModelInput(prompt=build_prompt(item, candidate_classes)) for item in batch_items]
            raw_responses = ["" for _ in batch_inputs] if args.dry_run else model.generate_batch(batch_inputs)

            for index, item, model_input, raw_response in zip(
                batch_indices,
                batch_items,
                batch_inputs,
                raw_responses,
                strict=True,
            ):
                predicted_class_idx, parsed_label = parse_prediction(raw_response, candidate_classes)
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
                    "prompt": model_input.prompt,
                    "raw_response": raw_response or None,
                    "parsed_label": parsed_label,
                    "predicted_class_idx": predicted_class_idx,
                    "dry_run": args.dry_run,
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
