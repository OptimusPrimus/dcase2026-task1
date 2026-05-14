from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from sklearn.metrics import accuracy_score
from torch.utils.data import Subset

from dcase2026_task1.data.datasets import (
    DEFAULT_BSD10K_ROOT,
    BSDDataset,
)
from dcase2026_task1.data.splits import build_stratified_folds
from dcase2026_task1.models import (
    AudioFlamingo3ClassificationSkill,
    AudioFlamingo3Model,
    AudioLanguageModel,
    ModelSkill,
    QwenClassificationSkill,
    QwenModel,
)
from dcase2026_task1.tasks import ClassificationTask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an audio-language model on stratified five-fold splits for BSD10k."
    )
    parser.add_argument(
        "--bsd10k-root",
        default=str(DEFAULT_BSD10K_ROOT),
        help="Root directory of the BSD10k dataset.",
    )
    parser.add_argument(
        "--model",
        choices=["audio-flamingo-3", "qwen", "qwen-text"],
        default="qwen",
        help="Audio-language model backend.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
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
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallelism degree for Qwen inference.",
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Pass disable_custom_all_reduce=True to the vLLM Qwen backend.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Pass enforce_eager=True to the vLLM Qwen backend.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Evaluate a single fold index. Default: run all folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fold generation.",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Number of stratified folds.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Validation split ratio taken from the development folds.",
    )
    parser.add_argument(
        "--max-test-items",
        type=int,
        default=None,
        help="Optional cap on test examples per fold for smoke runs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval",
        help="Directory for fold predictions and summary files.",
    )
    return parser


def load_model(args: argparse.Namespace) -> AudioLanguageModel:
    if args.model == "audio-flamingo-3":
        model_id = args.model_id or "nvidia/audio-flamingo-3-hf"
        return AudioFlamingo3Model(
            model_id=model_id,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )
    if args.model in {"qwen", "qwen-text"}:
        model_id = args.model_id or "Qwen/Qwen3.6-27B"
        return QwenModel(
            model_id=model_id,
            device=args.device,
            torch_dtype=args.torch_dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
            enforce_eager=args.enforce_eager,
        )
    raise ValueError(f"Unsupported model backend: {args.model}")


def load_skill(
    args: argparse.Namespace,
    task: ClassificationTask,
) -> ModelSkill:
    if args.model == "audio-flamingo-3":
        return AudioFlamingo3ClassificationSkill(task)
    if args.model in {"qwen", "qwen-text"}:
        return QwenClassificationSkill(task)
    raise ValueError(f"Unsupported model backend: {args.model}")


def build_candidate_classes(dataset: BSDDataset) -> list[dict[str, Any]]:
    by_class_idx: dict[int, dict[str, Any]] = {}
    for record in dataset.records:
        class_idx = int(record["class_idx"])
        if class_idx in by_class_idx:
            continue
        by_class_idx[class_idx] = {
            "class_idx": class_idx,
            "class_name": record["class"],
            "class_key": record["description_class_key"],
            "class_key_long": record["description_class_key_long"],
            "description_top_level": record["description_top_level"],
            "description_second_level": record["description_second_level"],
            "description": record["description_text"],
        }
    return [by_class_idx[idx] for idx in sorted(by_class_idx)]


def evaluate_fold(
    dataset: BSDDataset,
    model: AudioLanguageModel,
    skill: ModelSkill,
    fold_index: int,
    test_indices: list[int],
    output_dir: Path,
    max_test_items: int | None = None,
) -> dict[str, Any]:
    test_subset = Subset(dataset, test_indices)
    limit = len(test_subset) if max_test_items is None else min(len(test_subset), max_test_items)
    true_labels: list[int] = []
    pred_labels: list[int] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"fold_{fold_index:02d}_predictions.jsonl"

    with predictions_path.open("w", encoding="utf-8") as handle:
        for subset_index in range(limit):
            item = test_subset[subset_index]
            prediction = model.predict(item, skill)
            row = {
                "fold": fold_index,
                "subset_index": subset_index,
                "sound_id": item["sound_id"],
                "source_dataset": item["source_dataset"],
                "audio_path": item["audio_path"],
                "title": item["title"],
                "tags": item["tags"],
                "description": item["description"],
                "target_class_idx": int(item["class_idx"]),
                "target_class": item["class"],
                "predicted_class_idx": prediction.predicted_class_idx,
                "predicted_class": prediction.predicted_class_name,
                "parsed_label": prediction.parsed_label,
                "raw_response": prediction.raw_response,
                "final_response": prediction.final_response,
                "reasoning": prediction.reasoning,
                "correct": prediction.predicted_class_idx == int(item["class_idx"]),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

            true_labels.append(int(item["class_idx"]))
            pred_labels.append(
                -1
                if prediction.predicted_class_idx is None
                else int(prediction.predicted_class_idx)
            )

    accuracy = accuracy_score(true_labels, pred_labels) if true_labels else 0.0

    return {
        "fold": fold_index,
        "num_test_items": len(true_labels),
        "accuracy": accuracy,
        "predictions_path": str(predictions_path),
    }


def main() -> None:
    args = build_parser().parse_args()
    dataset = BSDDataset(
        root=args.bsd10k_root,
        dataset_name="BSD10k",
        load_audio=False,
    )
    candidate_classes = build_candidate_classes(dataset)
    task = ClassificationTask(candidate_classes)
    labels = [int(record["class_idx"]) for record in dataset.records]
    folds = build_stratified_folds(
        labels=labels,
        n_splits=args.k_folds,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    selected_folds = folds
    if args.fold is not None:
        selected_folds = [fold for fold in folds if fold.fold == args.fold]
        if not selected_folds:
            raise ValueError(f"Fold {args.fold} is out of range for {len(folds)} folds.")

    model = load_model(args)
    skill = load_skill(args, task)
    output_dir = Path(args.output_dir)
    results: list[dict[str, Any]] = []

    print(f"Loaded {len(dataset)} items across {len(candidate_classes)} classes.")
    for fold in selected_folds:
        print(
            f"Fold {fold.fold}: "
            f"train={len(fold.train_indices)} "
            f"val={len(fold.val_indices)} "
            f"test={len(fold.test_indices)}"
        )
        result = evaluate_fold(
            dataset=dataset,
            model=model,
            skill=skill,
            fold_index=fold.fold,
            test_indices=fold.test_indices,
            output_dir=output_dir,
            max_test_items=args.max_test_items,
        )
        results.append(result)
        print(
            f"Fold {result['fold']} accuracy={result['accuracy']:.4f} "
            f"items={result['num_test_items']} "
            f"predictions={result['predictions_path']}"
        )

    summary = {
        "model": args.model,
        "model_id": args.model_id
        or ("nvidia/audio-flamingo-3-hf" if args.model == "audio-flamingo-3" else "Qwen/Qwen3.6-27B"),
        "num_folds": len(results),
        "mean_accuracy": mean(result["accuracy"] for result in results) if results else 0.0,
        "folds": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
