from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean, median, stdev
from typing import Any, Iterable

import numpy as np

DEFAULT_BSD_SPLIT_SEED = 566182
DEFAULT_BSD10K_ROOT = Path.home() / "data" / "BSD10k"


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "analysis"
    / "bsd10k_validation_pseudo_labels"
)
DEFAULT_BSD10K_METADATA_CLASS_PROBABILITIES_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "outputs"
    / "experiments"
    / "20260611_211801_BSD10k_gpt-5.4-mini_f5706ed2"
    / "predictions.jsonl"
)


@dataclass(frozen=True)
class PseudoLabel:
    label: str
    confidence: float
    probabilities: dict[str, float]
    probability_sum: float
    entropy: float
    margin: float | None


@dataclass(frozen=True)
class AnalysisRow:
    dataset_index: int
    sound_id: int | None
    true_label: str
    pseudo_label: str
    ranked_pseudo_labels: tuple[str, ...]
    confidence: float
    probability_sum: float
    entropy: float
    margin: float | None
    agreement: bool
    title: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the BSD10k validation split against soft pseudo labels. "
            "The pseudo label JSONL must contain dataset_index and raw_response rows "
            "where raw_response is a JSON array of {label, probability} entries."
        )
    )
    parser.add_argument("--bsd10k-root", type=Path, default=DEFAULT_BSD10K_ROOT)
    parser.add_argument(
        "--pseudo-labels",
        type=Path,
        default=DEFAULT_BSD10K_METADATA_CLASS_PROBABILITIES_PATH,
        help="Path to predictions.jsonl with soft pseudo labels.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_BSD_SPLIT_SEED)
    parser.add_argument("--top-confusions", type=int, default=30)
    parser.add_argument(
        "--confidence-filter-fractions",
        default=None,
        help=(
            "Comma-separated retained fractions for per-class confidence filtering. "
            "Example: 0.1,0.2,0.5,1.0. Defaults to 0.05, 0.10, ..., 1.00."
        ),
    )
    parser.add_argument(
        "--top-n-values",
        default="1,2,3,5",
        help="Comma-separated N values for ranked-prefix top-N accuracy curves.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip optional PNG plots. CSV and JSON outputs are always written.",
    )
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row at {path}:{line_number}.")
            rows.append(row)
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_bsd10k_records(root: Path) -> list[dict[str, Any]]:
    metadata_csv = root / "metadata" / "BSD10k_metadata.csv"
    description_csv = root / "metadata" / "BST_description.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing BSD10k metadata CSV: {metadata_csv}")
    if not description_csv.exists():
        raise FileNotFoundError(f"Missing BSD10k description CSV: {description_csv}")

    descriptions = {
        int(row["class_idx"]): row
        for row in read_csv_rows(description_csv)
        if row.get("class_idx") not in (None, "")
    }
    records: list[dict[str, Any]] = []
    for dataset_index, row in enumerate(read_csv_rows(metadata_csv)):
        class_idx = int(row["class_idx"])
        sound_id_value = row.get("sound_id")
        records.append(
            {
                "dataset_index": dataset_index,
                "sound_id": int(sound_id_value) if sound_id_value else None,
                "class": row.get("class"),
                "class_idx": class_idx,
                "title": row.get("title"),
                "class_description": descriptions.get(class_idx),
            }
        )
    return records


def fallback_stratified_validation_indices(
    labels: list[int],
    *,
    fold: int,
    n_splits: int,
    validation_size: float,
    seed: int,
) -> list[int]:
    if not 0 <= fold < n_splits:
        raise ValueError(f"--fold must be in [0, {n_splits - 1}], got {fold}.")

    rng = Random(seed)
    indices_by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indices_by_label[label].append(index)

    validation_indices: list[int] = []
    for label_indices in indices_by_label.values():
        shuffled = list(label_indices)
        rng.shuffle(shuffled)
        test_folds = np.array_split(np.asarray(shuffled), n_splits)
        trainval_indices = [
            int(index)
            for split_index, split in enumerate(test_folds)
            if split_index != fold
            for index in split.tolist()
        ]
        rng.shuffle(trainval_indices)
        val_count = max(1, round(len(trainval_indices) * validation_size))
        validation_indices.extend(trainval_indices[:val_count])
    return sorted(validation_indices)


def validation_indices(
    labels: list[int],
    *,
    fold: int,
    n_splits: int,
    validation_size: float,
    seed: int,
) -> tuple[list[int], str]:
    try:
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    except ImportError:
        print(
            "Warning: scikit-learn is not installed; using a local stratified "
            "fallback split. Install project dependencies to reproduce the exact "
            "training validation split.",
            file=sys.stderr,
        )
        return (
            fallback_stratified_validation_indices(
                labels,
                fold=fold,
                n_splits=n_splits,
                validation_size=validation_size,
                seed=seed,
            ),
            "fallback",
        )

    label_array = np.asarray(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(label_array)), label_array))
    if not 0 <= fold < len(folds):
        raise ValueError(f"--fold must be in [0, {len(folds) - 1}], got {fold}.")

    trainval_idx, _test_idx = folds[fold]
    trainval_labels = label_array[trainval_idx]
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=seed + 11,
    )
    _, val_idx_rel = next(sss.split(np.zeros(len(trainval_labels)), trainval_labels))
    return trainval_idx[val_idx_rel].tolist(), "sklearn"


def parse_probability_items(raw_response: Any) -> list[dict[str, Any]]:
    if isinstance(raw_response, str):
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return []
    else:
        parsed = raw_response

    if not isinstance(parsed, list):
        return []

    items: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            items.append(item)
    return items


def parse_pseudo_label(row: dict[str, Any]) -> PseudoLabel | None:
    probability_items = parse_probability_items(
        row.get("raw_response", row.get("metadata_class_probabilities"))
    )
    probabilities: dict[str, float] = {}
    for item in probability_items:
        label = item.get("label")
        probability = item.get("probability")
        if not isinstance(label, str):
            continue
        try:
            probability_float = float(probability)
        except (TypeError, ValueError):
            continue
        if math.isfinite(probability_float):
            probabilities[label] = probability_float

    if not probabilities:
        return None

    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    top_label, top_probability = sorted_probs[0]
    second_probability = sorted_probs[1][1] if len(sorted_probs) > 1 else None
    probability_sum = sum(probabilities.values())
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities.values()
        if probability > 0.0
    )
    return PseudoLabel(
        label=top_label,
        confidence=top_probability,
        probabilities=probabilities,
        probability_sum=probability_sum,
        entropy=entropy,
        margin=(
            top_probability - second_probability
            if second_probability is not None
            else None
        ),
    )


def load_pseudo_labels(path: Path) -> dict[int, PseudoLabel]:
    pseudo_labels: dict[int, PseudoLabel] = {}
    for row in read_jsonl(path):
        dataset_index = row.get("dataset_index")
        if not isinstance(dataset_index, int):
            continue
        pseudo_label = parse_pseudo_label(row)
        if pseudo_label is None:
            continue
        pseudo_labels[dataset_index] = pseudo_label
    return pseudo_labels


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def distribution_rows(counter: Counter[str], total: int) -> list[dict[str, Any]]:
    return [
        {
            "class": label,
            "count": count,
            "fraction": count / total if total else 0.0,
        }
        for label, count in sorted(counter.items())
    ]


def describe_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def confidence_filter_fractions(raw_fractions: str | None) -> list[float]:
    if raw_fractions is None:
        return [round(float(value), 4) for value in np.linspace(0.05, 1.0, 20)]

    fractions: list[float] = []
    for raw_value in raw_fractions.split(","):
        value = float(raw_value.strip())
        if not 0.0 < value <= 1.0:
            raise ValueError(
                "--confidence-filter-fractions values must be in the interval (0, 1]."
            )
        fractions.append(value)
    return sorted(set(fractions))


def parse_top_n_values(raw_values: str) -> list[int]:
    values: list[int] = []
    for raw_value in raw_values.split(","):
        value = int(raw_value.strip())
        if value < 1:
            raise ValueError("--top-n-values must contain positive integers.")
        values.append(value)
    return sorted(set(values))


def confidence_filtered_accuracy_rows(
    rows: list[AnalysisRow],
    fractions: list[float],
) -> list[dict[str, Any]]:
    rows_by_pseudo_label: dict[str, list[AnalysisRow]] = defaultdict(list)
    for row in rows:
        rows_by_pseudo_label[row.pseudo_label].append(row)

    sorted_groups = {
        label: sorted(group, key=lambda row: row.confidence, reverse=True)
        for label, group in rows_by_pseudo_label.items()
    }

    output_rows: list[dict[str, Any]] = []
    for fraction in fractions:
        retained_rows: list[AnalysisRow] = []
        class_accuracy_values: list[float] = []
        class_thresholds: list[float] = []
        for group in sorted_groups.values():
            retained_count = max(1, math.ceil(len(group) * fraction))
            retained_group = group[:retained_count]
            retained_rows.extend(retained_group)
            class_accuracy_values.append(
                sum(row.agreement for row in retained_group) / len(retained_group)
            )
            class_thresholds.append(retained_group[-1].confidence)

        agreement_count = sum(row.agreement for row in retained_rows)
        output_rows.append(
            {
                "retained_fraction_per_class": fraction,
                "retained_count": len(retained_rows),
                "total_count": len(rows),
                "actual_retained_fraction": len(retained_rows) / len(rows),
                "agreement_count": agreement_count,
                "agreement_rate": agreement_count / len(retained_rows),
                "macro_agreement_rate_by_pseudo_label": mean(class_accuracy_values),
                "mean_confidence": mean([row.confidence for row in retained_rows]),
                "mean_class_confidence_threshold": mean(class_thresholds),
                "min_class_confidence_threshold": min(class_thresholds),
                "max_class_confidence_threshold": max(class_thresholds),
            }
        )
    return output_rows


def top_n_accuracy_rows(
    rows: list[AnalysisRow],
    top_n_values: list[int],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []

    for top_n in top_n_values:
        correct_count = sum(
            row.true_label in row.ranked_pseudo_labels[:top_n]
            for row in rows
        )
        output_rows.append(
            {
                "top_n": top_n,
                "accuracy": correct_count / len(rows),
                "correct_count": correct_count,
                "total_count": len(rows),
            }
        )
    return output_rows


def classwise_confidence_filtered_accuracy_rows(
    rows: list[AnalysisRow],
    fractions: list[float],
) -> list[dict[str, Any]]:
    rows_by_pseudo_label: dict[str, list[AnalysisRow]] = defaultdict(list)
    for row in rows:
        rows_by_pseudo_label[row.pseudo_label].append(row)

    output_rows: list[dict[str, Any]] = []
    for label, group in sorted(rows_by_pseudo_label.items()):
        sorted_group = sorted(group, key=lambda row: row.confidence, reverse=True)
        for fraction in fractions:
            retained_count = max(1, math.ceil(len(sorted_group) * fraction))
            retained_group = sorted_group[:retained_count]
            agreement_count = sum(row.agreement for row in retained_group)
            output_rows.append(
                {
                    "pseudo_label": label,
                    "retained_fraction_per_class": fraction,
                    "retained_count": retained_count,
                    "class_total_count": len(sorted_group),
                    "actual_retained_fraction": retained_count / len(sorted_group),
                    "agreement_count": agreement_count,
                    "agreement_rate": agreement_count / retained_count,
                    "mean_confidence": mean([row.confidence for row in retained_group]),
                    "confidence_threshold": retained_group[-1].confidence,
                }
            )
    return output_rows


def grouped_confidence_rows(
    rows: list[AnalysisRow],
    *,
    group_attr: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[AnalysisRow]] = defaultdict(list)
    for row in rows:
        grouped[str(getattr(row, group_attr))].append(row)

    output_rows: list[dict[str, Any]] = []
    for label, group in sorted(grouped.items()):
        confidence_summary = describe_values([row.confidence for row in group])
        margin_values = [row.margin for row in group if row.margin is not None]
        margin_summary = describe_values(margin_values)
        agreement_count = sum(row.agreement for row in group)
        output_rows.append(
            {
                "class": label,
                "count": len(group),
                "agreement_count": agreement_count,
                "agreement_rate": agreement_count / len(group),
                "confidence_mean": confidence_summary["mean"],
                "confidence_median": confidence_summary["median"],
                "confidence_std": confidence_summary["std"],
                "confidence_min": confidence_summary["min"],
                "confidence_max": confidence_summary["max"],
                "margin_mean": margin_summary["mean"],
                "entropy_mean": mean([row.entropy for row in group]),
            }
        )
    return output_rows


def confusion_matrix(
    rows: list[AnalysisRow],
    labels: list[str],
) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for row in rows:
        matrix[label_to_index[row.true_label], label_to_index[row.pseudo_label]] += 1
    return matrix


def write_confusion_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    fieldnames = ["annotation"] + labels
    table_rows = []
    for label, values in zip(labels, matrix, strict=True):
        table_rows.append({"annotation": label, **dict(zip(labels, values.tolist(), strict=True))})
    write_csv(path, table_rows, fieldnames)


def top_confusion_rows(rows: list[AnalysisRow], limit: int) -> list[dict[str, Any]]:
    counts = Counter(
        (row.true_label, row.pseudo_label)
        for row in rows
        if row.true_label != row.pseudo_label
    )
    return [
        {
            "annotation": true_label,
            "pseudo_label": pseudo_label,
            "count": count,
        }
        for (true_label, pseudo_label), count in counts.most_common(limit)
    ]


def maybe_write_plots(
    output_dir: Path,
    rows: list[AnalysisRow],
    matrix: np.ndarray,
    labels: list[str],
    filtered_accuracy_rows: list[dict[str, Any]],
    classwise_filtered_accuracy_rows: list[dict[str, Any]],
    top_n_accuracy_rows: list[dict[str, Any]],
    skip_plots: bool,
) -> list[str]:
    if skip_plots:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written: list[str] = []

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Pseudo label")
    ax.set_ylabel("Annotation")
    ax.set_title("BSD10k validation pseudo-label confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    confusion_path = output_dir / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=180)
    plt.close(fig)
    written.append(str(confusion_path))

    annotation_counts = Counter(row.true_label for row in rows)
    pseudo_label_counts = Counter(row.pseudo_label for row in rows)
    x = np.arange(len(labels))
    width = 0.42
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        x - width / 2,
        [annotation_counts.get(label, 0) for label in labels],
        width,
        label="Annotations",
    )
    ax.bar(
        x + width / 2,
        [pseudo_label_counts.get(label, 0) for label in labels],
        width,
        label="Hard pseudo labels",
    )
    ax.set_xticks(x, labels=labels, rotation=90)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("BSD10k validation class distribution")
    ax.legend()
    fig.tight_layout()
    distribution_path = output_dir / "class_distribution_annotation_vs_pseudo_label.png"
    fig.savefig(distribution_path, dpi=180)
    plt.close(fig)
    written.append(str(distribution_path))

    confidence_by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        confidence_by_label[row.pseudo_label].append(row.confidence)
    plot_labels = sorted(confidence_by_label)
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        ax.boxplot(
            [confidence_by_label[label] for label in plot_labels],
            tick_labels=plot_labels,
        )
    except TypeError:
        ax.boxplot(
            [confidence_by_label[label] for label in plot_labels],
            labels=plot_labels,
        )
    ax.set_xlabel("Pseudo label")
    ax.set_ylabel("Top pseudo-label probability")
    ax.set_title("Pseudo-label confidence by predicted class")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    confidence_path = output_dir / "confidence_by_pseudo_label.png"
    fig.savefig(confidence_path, dpi=180)
    plt.close(fig)
    written.append(str(confidence_path))

    fig, ax = plt.subplots(figsize=(9, 5))
    x_values = [
        row["retained_fraction_per_class"]
        for row in filtered_accuracy_rows
    ]
    ax.plot(
        x_values,
        [row["agreement_rate"] for row in filtered_accuracy_rows],
        marker="o",
        label="Micro accuracy",
    )
    ax.plot(
        x_values,
        [
            row["macro_agreement_rate_by_pseudo_label"]
            for row in filtered_accuracy_rows
        ],
        marker="s",
        label="Macro accuracy by predicted class",
    )
    ax.set_xlabel("Top confidence fraction retained within each predicted class")
    ax.set_ylabel("Pseudo-label agreement with annotation")
    ax.set_title("Pseudo-label accuracy under per-class confidence filtering")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    filtered_accuracy_path = output_dir / "confidence_filtered_accuracy.png"
    fig.savefig(filtered_accuracy_path, dpi=180)
    plt.close(fig)
    written.append(str(filtered_accuracy_path))

    classwise_rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in classwise_filtered_accuracy_rows:
        classwise_rows_by_label[str(row["pseudo_label"])].append(row)

    fig, ax = plt.subplots(figsize=(12, 7))
    for label in sorted(classwise_rows_by_label):
        class_rows = sorted(
            classwise_rows_by_label[label],
            key=lambda row: row["retained_fraction_per_class"],
        )
        ax.plot(
            [row["retained_fraction_per_class"] for row in class_rows],
            [row["agreement_rate"] for row in class_rows],
            linewidth=1.3,
            alpha=0.75,
            label=label,
        )
    ax.set_xlabel("Top confidence fraction retained within each predicted class")
    ax.set_ylabel("Class-wise pseudo-label agreement with annotation")
    ax.set_title("Class-wise pseudo-label accuracy under confidence filtering")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize="small")
    fig.tight_layout()
    classwise_filtered_accuracy_path = (
        output_dir / "classwise_confidence_filtered_accuracy.png"
    )
    fig.savefig(classwise_filtered_accuracy_path, dpi=180)
    plt.close(fig)
    written.append(str(classwise_filtered_accuracy_path))

    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_top_n_rows = sorted(top_n_accuracy_rows, key=lambda row: row["top_n"])
    ax.plot(
        [row["top_n"] for row in sorted_top_n_rows],
        [row["accuracy"] for row in sorted_top_n_rows],
        marker="o",
    )
    ax.set_xlabel("N in top-N predictions")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-N pseudo-label accuracy")
    ax.set_xticks([row["top_n"] for row in sorted_top_n_rows])
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    top_n_accuracy_path = output_dir / "top_n_accuracy.png"
    fig.savefig(top_n_accuracy_path, dpi=180)
    plt.close(fig)
    written.append(str(top_n_accuracy_path))

    return written


def analysis_rows(
    records: list[dict[str, Any]],
    pseudo_labels: dict[int, PseudoLabel],
    val_indices: list[int],
) -> tuple[list[AnalysisRow], list[int]]:
    rows: list[AnalysisRow] = []
    missing_indices: list[int] = []
    for dataset_index in val_indices:
        record = records[dataset_index]
        pseudo_label = pseudo_labels.get(dataset_index)
        if pseudo_label is None:
            missing_indices.append(dataset_index)
            continue
        true_label = str(record["class"])
        rows.append(
            AnalysisRow(
                dataset_index=dataset_index,
                sound_id=record.get("sound_id"),
                true_label=true_label,
                pseudo_label=pseudo_label.label,
                ranked_pseudo_labels=tuple(
                    label
                    for label, _probability in sorted(
                        pseudo_label.probabilities.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ),
                confidence=pseudo_label.confidence,
                probability_sum=pseudo_label.probability_sum,
                entropy=pseudo_label.entropy,
                margin=pseudo_label.margin,
                agreement=true_label == pseudo_label.label,
                title=record.get("title"),
            )
        )
    return rows, missing_indices


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_bsd10k_records(args.bsd10k_root)
    labels_by_index = [int(record["class_idx"]) for record in records]
    val_indices, split_implementation = validation_indices(
        labels=labels_by_index,
        fold=args.fold,
        n_splits=args.n_splits,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    pseudo_labels = load_pseudo_labels(args.pseudo_labels)
    rows, missing_indices = analysis_rows(records, pseudo_labels, val_indices)
    if not rows:
        raise RuntimeError(
            "No validation records had parseable pseudo labels. "
            f"Checked {len(val_indices)} validation indices against {args.pseudo_labels}."
        )

    true_distribution = Counter(row.true_label for row in rows)
    pseudo_distribution = Counter(row.pseudo_label for row in rows)
    labels = sorted(set(true_distribution) | set(pseudo_distribution))
    matrix = confusion_matrix(rows, labels)
    agreement_count = sum(row.agreement for row in rows)
    accuracy = agreement_count / len(rows)
    filtered_accuracy_rows = confidence_filtered_accuracy_rows(
        rows,
        confidence_filter_fractions(args.confidence_filter_fractions),
    )
    classwise_filtered_accuracy_rows = classwise_confidence_filtered_accuracy_rows(
        rows,
        confidence_filter_fractions(args.confidence_filter_fractions),
    )
    top_n_rows = top_n_accuracy_rows(
        rows,
        parse_top_n_values(args.top_n_values),
    )
    plot_paths = maybe_write_plots(
        args.output_dir,
        rows,
        matrix,
        labels,
        filtered_accuracy_rows,
        classwise_filtered_accuracy_rows,
        top_n_rows,
        args.no_plots,
    )

    write_csv(
        args.output_dir / "validation_records.csv",
        [
            {
                "dataset_index": row.dataset_index,
                "sound_id": row.sound_id,
                "annotation": row.true_label,
                "pseudo_label": row.pseudo_label,
                "ranked_pseudo_labels": " ".join(row.ranked_pseudo_labels),
                "confidence": row.confidence,
                "probability_sum": row.probability_sum,
                "entropy": row.entropy,
                "margin": row.margin,
                "agreement": row.agreement,
                "title": row.title,
            }
            for row in rows
        ],
        [
            "dataset_index",
            "sound_id",
            "annotation",
            "pseudo_label",
            "ranked_pseudo_labels",
            "confidence",
            "probability_sum",
            "entropy",
            "margin",
            "agreement",
            "title",
        ],
    )
    write_csv(
        args.output_dir / "annotation_distribution.csv",
        distribution_rows(true_distribution, len(rows)),
        ["class", "count", "fraction"],
    )
    write_csv(
        args.output_dir / "pseudo_label_distribution.csv",
        distribution_rows(pseudo_distribution, len(rows)),
        ["class", "count", "fraction"],
    )
    write_confusion_csv(args.output_dir / "confusion_matrix.csv", matrix, labels)
    write_csv(
        args.output_dir / "top_confusions.csv",
        top_confusion_rows(rows, args.top_confusions),
        ["annotation", "pseudo_label", "count"],
    )
    write_csv(
        args.output_dir / "confidence_by_pseudo_label.csv",
        grouped_confidence_rows(rows, group_attr="pseudo_label"),
        [
            "class",
            "count",
            "agreement_count",
            "agreement_rate",
            "confidence_mean",
            "confidence_median",
            "confidence_std",
            "confidence_min",
            "confidence_max",
            "margin_mean",
            "entropy_mean",
        ],
    )
    write_csv(
        args.output_dir / "confidence_by_annotation.csv",
        grouped_confidence_rows(rows, group_attr="true_label"),
        [
            "class",
            "count",
            "agreement_count",
            "agreement_rate",
            "confidence_mean",
            "confidence_median",
            "confidence_std",
            "confidence_min",
            "confidence_max",
            "margin_mean",
            "entropy_mean",
        ],
    )
    write_csv(
        args.output_dir / "confidence_filtered_accuracy.csv",
        filtered_accuracy_rows,
        [
            "retained_fraction_per_class",
            "retained_count",
            "total_count",
            "actual_retained_fraction",
            "agreement_count",
            "agreement_rate",
            "macro_agreement_rate_by_pseudo_label",
            "mean_confidence",
            "mean_class_confidence_threshold",
            "min_class_confidence_threshold",
            "max_class_confidence_threshold",
        ],
    )
    write_csv(
        args.output_dir / "classwise_confidence_filtered_accuracy.csv",
        classwise_filtered_accuracy_rows,
        [
            "pseudo_label",
            "retained_fraction_per_class",
            "retained_count",
            "class_total_count",
            "actual_retained_fraction",
            "agreement_count",
            "agreement_rate",
            "mean_confidence",
            "confidence_threshold",
        ],
    )
    write_csv(
        args.output_dir / "top_n_accuracy.csv",
        top_n_rows,
        [
            "top_n",
            "accuracy",
            "correct_count",
            "total_count",
        ],
    )

    summary = {
        "bsd10k_root": str(args.bsd10k_root),
        "pseudo_labels": str(args.pseudo_labels),
        "output_dir": str(args.output_dir),
        "fold": args.fold,
        "n_splits": args.n_splits,
        "validation_size": args.validation_size,
        "seed": args.seed,
        "split_implementation": split_implementation,
        "validation_records": len(val_indices),
        "records_with_pseudo_labels": len(rows),
        "missing_or_unparseable_pseudo_labels": len(missing_indices),
        "agreement_count": agreement_count,
        "agreement_rate": accuracy,
        "confidence": describe_values([row.confidence for row in rows]),
        "confidence_when_agree": describe_values(
            [row.confidence for row in rows if row.agreement]
        ),
        "confidence_when_disagree": describe_values(
            [row.confidence for row in rows if not row.agreement]
        ),
        "plot_paths": plot_paths,
        "outputs": [
            "validation_records.csv",
            "annotation_distribution.csv",
            "pseudo_label_distribution.csv",
            "confusion_matrix.csv",
            "top_confusions.csv",
            "confidence_by_pseudo_label.csv",
            "confidence_by_annotation.csv",
            "confidence_filtered_accuracy.csv",
            "classwise_confidence_filtered_accuracy.csv",
            "top_n_accuracy.csv",
            "summary.json",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
