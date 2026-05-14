from __future__ import annotations

import argparse
import json
from pathlib import Path

from dcase2026_task1.data.datasets import DEFAULT_BSD10K_ROOT, BSDDataset
from dcase2026_task1.models import (
    QwenMetadataSummarizationSkill,
    QwenModel,
)
from dcase2026_task1.tasks import MetadataSummarizationTask
from tqdm import tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create metadata summaries for every BSD10k example and store them in a JSONL file."
    )
    parser.add_argument(
        "--bsd10k-root",
        default=str(DEFAULT_BSD10K_ROOT),
        help="Root directory of the BSD10k dataset.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.6-27B",
        help="Model identifier passed to the backend.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Reserved backend device setting.",
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
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on examples for smoke runs.",
    )
    parser.add_argument(
        "--output",
        default="outputs/metadata_summaries.jsonl",
        help="Path of the JSONL file to write.",
    )
    return parser


def write_metadata_summaries(
    dataset: BSDDataset,
    model: QwenModel,
    output_path: Path,
    max_items: int | None = None,
) -> None:
    task = MetadataSummarizationTask()
    skill = QwenMetadataSummarizationSkill(task)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limit = len(dataset) if max_items is None else min(len(dataset), max_items)

    with output_path.open("w", encoding="utf-8") as handle:
        for index in tqdm(range(limit), total=limit, desc="Summarizing", unit="item"):
            item = dataset[index]
            response = model.predict(item, skill)
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
                "audio_content": response.audio_content,
                "metadata_details": response.metadata_details,
                "raw_response": response.raw_response,
                "final_response": response.final_response,
                "reasoning": response.reasoning,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def main() -> None:
    args = build_parser().parse_args()
    dataset = BSDDataset(
        root=args.bsd10k_root,
        dataset_name="BSD10k",
        load_audio=False,
    )
    model = QwenModel(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        enforce_eager=args.enforce_eager,
    )
    output_path = Path(args.output)

    print(f"Loaded {len(dataset)} items from BSD10k.")
    write_metadata_summaries(
        dataset=dataset,
        model=model,
        output_path=output_path,
        max_items=args.max_items,
    )
    print(f"Wrote metadata summaries to {output_path}")


if __name__ == "__main__":
    main()
