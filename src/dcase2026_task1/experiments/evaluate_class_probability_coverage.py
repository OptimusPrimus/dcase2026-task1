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
            "predicted class list from class-probability outputs."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        default=str(DEFAULT_EXPERIMENT_DIR),
        help="Directory containing predictions.jsonl.",
    )
    parser.add_argument(
        "--predictions-filename",
        default=PREDICTIONS_FILENAME,
        help="JSONL filename to evaluate inside --experiment-dir.",
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


def parse_metadata_probability_labels(raw_response: Any) -> list[str] | None:
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


def parse_audio_classifier_predictions(predictions: Any) -> list[str] | None:
    if not isinstance(predictions, list):
        return None

    labels: list[str] = []
    for item in predictions:
        if not isinstance(item, dict):
            return None
        label = item.get("label")
        probability = item.get("probability")
        if not isinstance(label, str) or not isinstance(probability, (int, float)):
            return None
        labels.append(label)
    return labels


def parse_audio_classifier_correction_labels(
    raw_response: Any,
    *,
    allowed_labels: set[str] | None,
) -> list[str] | None:
    if not isinstance(raw_response, str):
        return None
    try:
        predictions = json.loads(raw_response)
    except json.JSONDecodeError:
        return None
    if not isinstance(predictions, dict):
        return None

    scored_labels: list[tuple[str, float, int]] = []
    for order, (label, confidence) in enumerate(predictions.items()):
        if not isinstance(label, str):
            return None
        if allowed_labels is not None and label not in allowed_labels:
            return None
        if not isinstance(confidence, (int, float)):
            return None
        confidence_value = float(confidence)
        if confidence_value < 0.0 or confidence_value > 1.0:
            return None
        scored_labels.append((label, confidence_value, order))

    if not scored_labels:
        return None
    return [label for label, _confidence, _order in sorted(scored_labels, key=lambda item: (-item[1], item[2]))]


def resolve_predicted_labels(row: dict[str, Any]) -> tuple[list[str] | None, str, str | None]:
    audio_classifier_labels = parse_audio_classifier_predictions(row.get("audio_classifier_predictions"))
    if audio_classifier_labels is not None:
        corrected_labels = parse_audio_classifier_correction_labels(
            row.get("raw_response"),
            allowed_labels=set(audio_classifier_labels),
        )
        if corrected_labels is not None:
            return corrected_labels, "llm_correction", None
        return audio_classifier_labels, "audio_classifier_fallback", "invalid_llm_correction"

    metadata_labels = parse_metadata_probability_labels(row.get("raw_response"))
    if metadata_labels is not None:
        return metadata_labels, "metadata_probability", None
    return None, "parse_failure", "invalid_metadata_probability"


def contains_label_within_top_k(predicted_labels: list[str], target_class: str, k: int) -> bool:
    if k <= 0:
        return False
    return target_class in predicted_labels[:k]


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_from_precision_recall(precision: float, recall: float) -> float:
    return safe_divide(2.0 * precision * recall, precision + recall)


def evaluate_prediction_set_coverage(rows: list[tuple[dict[str, Any], list[str] | None]]) -> dict[str, Any]:
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

    for row, predicted_labels in rows:
        target_class = row.get("target_class")
        if not isinstance(target_class, str):
            raise ValueError(f"Row missing string target_class: {row}")
        class_totals[target_class] += 1

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


def evaluate_f1_metrics(rows: list[tuple[dict[str, Any], list[str] | None]]) -> dict[str, Any]:
    set_true_positives = 0
    set_false_positives = 0
    set_false_negatives = 0
    top_1_true_positives = 0
    top_1_false_positives = 0
    top_1_false_negatives = 0
    per_class_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row, predicted_labels in rows:
        target_class = row.get("target_class")
        if not isinstance(target_class, str):
            raise ValueError(f"Row missing string target_class: {row}")

        predicted_set = set(predicted_labels or [])
        if target_class in predicted_set:
            set_true_positives += 1
            per_class_counts[target_class]["set_tp"] += 1
        else:
            set_false_negatives += 1
            per_class_counts[target_class]["set_fn"] += 1

        false_positive_labels = predicted_set - {target_class}
        set_false_positives += len(false_positive_labels)
        for label in false_positive_labels:
            per_class_counts[label]["set_fp"] += 1

        top_1_label = predicted_labels[0] if predicted_labels else None
        if top_1_label == target_class:
            top_1_true_positives += 1
            per_class_counts[target_class]["top_1_tp"] += 1
        else:
            top_1_false_negatives += 1
            per_class_counts[target_class]["top_1_fn"] += 1
            if top_1_label is not None:
                top_1_false_positives += 1
                per_class_counts[top_1_label]["top_1_fp"] += 1

    set_precision = safe_divide(set_true_positives, set_true_positives + set_false_positives)
    set_recall = safe_divide(set_true_positives, set_true_positives + set_false_negatives)
    top_1_precision = safe_divide(top_1_true_positives, top_1_true_positives + top_1_false_positives)
    top_1_recall = safe_divide(top_1_true_positives, top_1_true_positives + top_1_false_negatives)

    per_class: list[dict[str, Any]] = []
    set_f1_values: list[float] = []
    top_1_f1_values: list[float] = []
    for label in sorted(per_class_counts):
        counts = per_class_counts[label]
        class_set_precision = safe_divide(counts["set_tp"], counts["set_tp"] + counts["set_fp"])
        class_set_recall = safe_divide(counts["set_tp"], counts["set_tp"] + counts["set_fn"])
        class_set_f1 = f1_from_precision_recall(class_set_precision, class_set_recall)
        class_top_1_precision = safe_divide(counts["top_1_tp"], counts["top_1_tp"] + counts["top_1_fp"])
        class_top_1_recall = safe_divide(counts["top_1_tp"], counts["top_1_tp"] + counts["top_1_fn"])
        class_top_1_f1 = f1_from_precision_recall(class_top_1_precision, class_top_1_recall)
        set_f1_values.append(class_set_f1)
        top_1_f1_values.append(class_top_1_f1)
        per_class.append(
            {
                "label": label,
                "set_precision": class_set_precision,
                "set_recall": class_set_recall,
                "set_f1": class_set_f1,
                "top_1_precision": class_top_1_precision,
                "top_1_recall": class_top_1_recall,
                "top_1_f1": class_top_1_f1,
            }
        )

    return {
        "set_precision": set_precision,
        "set_recall": set_recall,
        "set_micro_f1": f1_from_precision_recall(set_precision, set_recall),
        "set_macro_f1": safe_divide(sum(set_f1_values), len(set_f1_values)),
        "top_1_precision": top_1_precision,
        "top_1_recall": top_1_recall,
        "top_1_micro_f1": f1_from_precision_recall(top_1_precision, top_1_recall),
        "top_1_macro_f1": safe_divide(sum(top_1_f1_values), len(top_1_f1_values)),
        "per_class": per_class,
    }


def compare_to_audio_classifier(
    resolved_rows: list[tuple[dict[str, Any], list[str] | None, str, str | None]],
) -> dict[str, Any] | None:
    comparable_count = 0
    exact_match_count = 0
    top_1_match_count = 0
    changed_count = 0
    total_overlap = 0.0
    total_jaccard = 0.0
    examples: list[dict[str, Any]] = []

    for row, resolved_labels, source, _fallback_reason in resolved_rows:
        audio_classifier_labels = parse_audio_classifier_predictions(row.get("audio_classifier_predictions"))
        if audio_classifier_labels is None or resolved_labels is None:
            continue

        comparable_count += 1
        if resolved_labels == audio_classifier_labels:
            exact_match_count += 1
        else:
            changed_count += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "dataset_index": row.get("dataset_index"),
                        "sound_id": row.get("sound_id"),
                        "target_class": row.get("target_class"),
                        "source": source,
                        "resolved_labels": resolved_labels,
                        "audio_classifier_labels": audio_classifier_labels,
                    }
                )

        if resolved_labels[:1] == audio_classifier_labels[:1]:
            top_1_match_count += 1

        resolved_set = set(resolved_labels)
        audio_classifier_set = set(audio_classifier_labels)
        intersection_size = len(resolved_set & audio_classifier_set)
        union_size = len(resolved_set | audio_classifier_set)
        total_overlap += intersection_size / len(resolved_set) if resolved_set else 0.0
        total_jaccard += intersection_size / union_size if union_size else 0.0

    if comparable_count == 0:
        return None

    return {
        "num_comparable_rows": comparable_count,
        "num_exact_matches": exact_match_count,
        "exact_match_rate": exact_match_count / comparable_count,
        "num_top_1_matches": top_1_match_count,
        "top_1_match_rate": top_1_match_count / comparable_count,
        "num_changed_predictions": changed_count,
        "changed_prediction_rate": changed_count / comparable_count,
        "mean_resolved_label_overlap_with_audio_classifier": total_overlap / comparable_count,
        "mean_jaccard_with_audio_classifier": total_jaccard / comparable_count,
        "changed_examples": examples,
    }


def evaluate_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resolved_rows: list[tuple[dict[str, Any], list[str] | None, str, str | None]] = []
    source_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    for row in rows:
        predicted_labels, source, fallback_reason = resolve_predicted_labels(row)
        resolved_rows.append((row, predicted_labels, source, fallback_reason))
        source_counts[source] += 1
        if fallback_reason is not None:
            fallback_reason_counts[fallback_reason] += 1

    resolved_prediction_rows = [(row, predicted_labels) for row, predicted_labels, _source, _reason in resolved_rows]
    audio_classifier_prediction_rows = [
        (row, parse_audio_classifier_predictions(row.get("audio_classifier_predictions"))) for row in rows
    ]
    resolved_coverage = evaluate_prediction_set_coverage(resolved_prediction_rows)
    audio_classifier_coverage = evaluate_prediction_set_coverage(audio_classifier_prediction_rows)
    f1_with_prediction_correction = evaluate_f1_metrics(resolved_prediction_rows)
    f1_without_prediction_correction = evaluate_f1_metrics(audio_classifier_prediction_rows)
    results = {
        **resolved_coverage,
        "prediction_source_counts": dict(source_counts),
        "fallback_reason_counts": dict(fallback_reason_counts),
        "audio_classifier_coverage": audio_classifier_coverage,
        "f1_with_prediction_correction": f1_with_prediction_correction,
        "f1_without_prediction_correction": f1_without_prediction_correction,
        "f1_delta_with_minus_without_prediction_correction": {
            "set_micro_f1": (
                f1_with_prediction_correction["set_micro_f1"]
                - f1_without_prediction_correction["set_micro_f1"]
            ),
            "set_macro_f1": (
                f1_with_prediction_correction["set_macro_f1"]
                - f1_without_prediction_correction["set_macro_f1"]
            ),
            "top_1_micro_f1": (
                f1_with_prediction_correction["top_1_micro_f1"]
                - f1_without_prediction_correction["top_1_micro_f1"]
            ),
            "top_1_macro_f1": (
                f1_with_prediction_correction["top_1_macro_f1"]
                - f1_without_prediction_correction["top_1_macro_f1"]
            ),
        },
    }
    comparison = compare_to_audio_classifier(resolved_rows)
    if comparison is not None:
        results["audio_classifier_comparison"] = comparison
    return results


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = Path(args.experiment_dir)
    predictions_path = experiment_dir / args.predictions_filename
    rows = load_prediction_rows(predictions_path)
    results = evaluate_coverage(rows)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
