from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "experiments"
    / "20260611_004827_BSD10k_gpt-5.4-mini_a2869640"
)
PREDICTIONS_FILENAME = "predictions.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check how often the ground-truth class appears anywhere in the "
            "predicted class list from metadata-only class-probability outputs."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        default=str(DEFAULT_EXPERIMENT_DIR),
        help="Directory containing predictions.jsonl.",
    )
    return parser


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row at line {line_number} in {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at line {line_number} in {path}, got {type(row).__name__}.")
            rows.append(row)
    return rows


def parse_predicted_labels(raw_response: Any) -> list[str] | None:
    if not isinstance(raw_response, str):
        return None
    try:
        predictions = json.loads(raw_response)
    except json.JSONDecodeError:
        return None
    if not isinstance(predictions, list):
        return None

    labels: list[str] = []
    for item in predictions:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if isinstance(label, str):
            labels.append(label)
    return labels


def contains_label_within_top_k(predicted_labels: list[str], target_class: str, k: int) -> bool:
    if k <= 0:
        return False
    return target_class in predicted_labels[:k]


def evaluate_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parsed_count = 0
    hit_count = 0
    top_1_hit_count = 0
    top_2_hit_count = 0
    parse_failure_count = 0
    class_totals: Counter[str] = Counter()
    class_hits: Counter[str] = Counter()
    class_top_1_hits: Counter[str] = Counter()
    class_top_2_hits: Counter[str] = Counter()
    missed_examples_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        target_class = row.get("target_class")
        if not isinstance(target_class, str):
            raise ValueError(f"Row missing string target_class: {row}")
        class_totals[target_class] += 1

        predicted_labels = parse_predicted_labels(row.get("raw_response"))
        if predicted_labels is None:
            parse_failure_count += 1
            missed_examples_by_class[target_class].append(
                {
                    "dataset_index": row.get("dataset_index"),
                    "sound_id": row.get("sound_id"),
                    "target_class": target_class,
                    "predicted_labels": None,
                }
            )
            continue

        parsed_count += 1
        if target_class in predicted_labels:
            hit_count += 1
            class_hits[target_class] += 1
            if contains_label_within_top_k(predicted_labels, target_class, 1):
                top_1_hit_count += 1
                class_top_1_hits[target_class] += 1
            if contains_label_within_top_k(predicted_labels, target_class, 2):
                top_2_hit_count += 1
                class_top_2_hits[target_class] += 1
        else:
            missed_examples_by_class[target_class].append(
                {
                    "dataset_index": row.get("dataset_index"),
                    "sound_id": row.get("sound_id"),
                    "target_class": target_class,
                    "predicted_labels": predicted_labels,
                }
            )

    per_class = []
    for label in sorted(class_totals):
        total_for_class = class_totals[label]
        hits_for_class = class_hits[label]
        per_class.append(
            {
                "label": label,
                "total": total_for_class,
                "hits": hits_for_class,
                "coverage": hits_for_class / total_for_class if total_for_class else 0.0,
                "top_1_hits": class_top_1_hits[label],
                "top_1_coverage": class_top_1_hits[label] / total_for_class if total_for_class else 0.0,
                "top_2_hits": class_top_2_hits[label],
                "top_2_coverage": class_top_2_hits[label] / total_for_class if total_for_class else 0.0,
                "misses": total_for_class - hits_for_class,
            }
        )

    return {
        "num_rows": total,
        "num_parsed_predictions": parsed_count,
        "num_parse_failures": parse_failure_count,
        "num_hits": hit_count,
        "coverage": hit_count / total if total else 0.0,
        "num_top_1_hits": top_1_hit_count,
        "top_1_coverage": top_1_hit_count / total if total else 0.0,
        "num_top_2_hits": top_2_hit_count,
        "top_2_coverage": top_2_hit_count / total if total else 0.0,
        "per_class": per_class,
        "missed_examples_by_class": dict(missed_examples_by_class),
    }


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = Path(args.experiment_dir)
    predictions_path = experiment_dir / PREDICTIONS_FILENAME
    rows = load_prediction_rows(predictions_path)
    results = evaluate_coverage(rows)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
