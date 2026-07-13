from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dcase2026_task1.data.datasets import DEFAULT_BSD10K_ROOT
from dcase2026_task1.data.splits import (
    DEFAULT_BSD_SPLIT_SEED,
    build_stratified_folds,
    load_records_by_dataset_name,
)

DEFAULT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parent
    / "20260706_010318_BSD10k_lclap_d8a36922"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "analysis_output"
)
DEFAULT_FOCUSED_CONFUSION_CLASSES = ("ss-u", "ss-i", "fx-a", "sp-c", "ss-n", "fx-n")


@dataclass(frozen=True)
class PredictionRow:
    dataset_index: int
    sound_id: int
    true_label: str
    predicted_label: str
    ranked_labels: tuple[str, ...]
    probabilities: np.ndarray
    top1_confidence: float
    title: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze BSD10k test-split predictions exported as bsd10k_logits.npz. "
            "By default this uses the 20260706_010318_BSD10k_lclap_d8a36922 run."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--bsd10k-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--validation-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-confusions", type=int, default=30)
    parser.add_argument(
        "--focused-confusion-classes",
        default=",".join(DEFAULT_FOCUSED_CONFUSION_CLASSES),
        help="Comma-separated classes to include in the focused confusion matrix.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def read_config(experiment_dir: Path) -> dict[str, Any]:
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_bsd10k_root(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.bsd10k_root is not None:
        return args.bsd10k_root
    config_root = config.get("dataset_roots", {}).get("BSD10k")
    if isinstance(config_root, str) and Path(config_root).exists():
        return Path(config_root)
    prediction_root = config.get("prediction_datasets", {}).get("bsd10k_root")
    if isinstance(prediction_root, str) and Path(prediction_root).exists():
        return Path(prediction_root)
    return DEFAULT_BSD10K_ROOT


def split_value(
    args_value: Any,
    config: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    if args_value is not None:
        return args_value
    return config.get("split", {}).get(key, default)


def parse_class_list(raw_classes: str) -> list[str]:
    classes: list[str] = []
    seen: set[str] = set()
    for raw_class in raw_classes.split(","):
        class_name = raw_class.strip()
        if not class_name or class_name in seen:
            continue
        classes.append(class_name)
        seen.add(class_name)
    return classes


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def load_npz_predictions(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing logits NPZ: {path}")

    predictions: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        if "label_names" not in data.files:
            raise ValueError(f"{path} does not contain a label_names entry.")
        label_names = [str(label) for label in data["label_names"].tolist()]
        for file_id in data.files:
            if file_id == "label_names":
                continue
            logits = np.asarray(data[file_id], dtype=np.float64)
            if logits.shape != (len(label_names),):
                raise ValueError(
                    f"Unexpected logits shape for {file_id!r}: {logits.shape}; "
                    f"expected ({len(label_names)},)."
                )
            predictions[file_id] = logits
    return label_names, predictions


def prediction_rows(
    records: list[dict[str, Any]],
    test_indices: list[int],
    label_names: list[str],
    logits_by_file_id: dict[str, np.ndarray],
) -> tuple[list[PredictionRow], list[int]]:
    rows: list[PredictionRow] = []
    missing_indices: list[int] = []

    for dataset_index in test_indices:
        record = records[dataset_index]
        sound_id = int(record["sound_id"])
        logits = logits_by_file_id.get(str(sound_id))
        if logits is None:
            missing_indices.append(dataset_index)
            continue

        probabilities = softmax(logits)
        ranked_label_ids = np.argsort(probabilities)[::-1]
        predicted_label_id = int(ranked_label_ids[0])
        rows.append(
            PredictionRow(
                dataset_index=dataset_index,
                sound_id=sound_id,
                true_label=str(record["class"]),
                predicted_label=label_names[predicted_label_id],
                ranked_labels=tuple(label_names[int(index)] for index in ranked_label_ids),
                probabilities=probabilities.astype(np.float64, copy=False),
                top1_confidence=float(probabilities[predicted_label_id]),
                title=record.get("title"),
            )
        )
    return rows, missing_indices


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def confusion_matrix(rows: list[PredictionRow], labels: list[str]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for row in rows:
        matrix[label_to_index[row.true_label], label_to_index[row.predicted_label]] += 1
    return matrix


def probability_mass_matrix(rows: list[PredictionRow], labels: list[str]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    counts = np.zeros(len(labels), dtype=np.int64)
    for row in rows:
        predicted_index = label_to_index[row.predicted_label]
        counts[predicted_index] += 1
        matrix[predicted_index] += row.probabilities

    nonzero = counts > 0
    matrix[nonzero] = matrix[nonzero] / counts[nonzero, np.newaxis]
    return matrix


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    row_header: str,
    row_labels: list[str],
    column_labels: list[str],
) -> None:
    rows = [
        {
            row_header: label,
            **{
                column_label: value
                for column_label, value in zip(column_labels, matrix_row.tolist(), strict=True)
            },
        }
        for label, matrix_row in zip(row_labels, matrix, strict=True)
    ]
    write_csv(path, rows, [row_header] + column_labels)


def prediction_csv_row(row: PredictionRow, labels: list[str]) -> dict[str, Any]:
    return {
        "dataset_index": row.dataset_index,
        "sound_id": row.sound_id,
        "actual_class": row.true_label,
        "predicted_class": row.predicted_label,
        "ranked_labels": " ".join(row.ranked_labels),
        "top1_confidence": row.top1_confidence,
        "title": row.title,
        **{
            f"probability_{label}": probability
            for label, probability in zip(labels, row.probabilities.tolist(), strict=True)
        },
    }


def prediction_csv_fieldnames(labels: list[str]) -> list[str]:
    return [
        "dataset_index",
        "sound_id",
        "actual_class",
        "predicted_class",
        "ranked_labels",
        "top1_confidence",
        "title",
        *[f"probability_{label}" for label in labels],
    ]


def top_n_rows(rows: list[PredictionRow], top_n_values: list[int]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for top_n in top_n_values:
        correct_count = sum(row.true_label in row.ranked_labels[:top_n] for row in rows)
        output_rows.append(
            {
                "top_n": top_n,
                "correct_count": correct_count,
                "total_count": len(rows),
                "accuracy": correct_count / len(rows) if rows else 0.0,
            }
        )
    return output_rows


def top_confusion_rows(rows: list[PredictionRow], limit: int) -> list[dict[str, Any]]:
    counts = Counter(
        (row.true_label, row.predicted_label)
        for row in rows
        if row.true_label != row.predicted_label
    )
    output_rows: list[dict[str, Any]] = []
    for (true_label, predicted_label), count in counts.most_common(limit):
        class_rows = [
            row
            for row in rows
            if row.true_label == true_label and row.predicted_label == predicted_label
        ]
        output_rows.append(
            {
                "actual_class": true_label,
                "predicted_class": predicted_label,
                "count": count,
                "fraction_of_all_predictions": count / len(rows) if rows else 0.0,
                "mean_top1_confidence": float(
                    np.mean([row.top1_confidence for row in class_rows])
                ),
            }
        )
    return output_rows


def compute_hierarchical_metrics(
    y_true: list[str],
    y_pred: list[str],
    class_names: list[str],
) -> dict[str, Any]:
    def safe_mean(values: list[float]) -> float:
        return float(np.mean(values).item()) if values else 0.0

    def safe_f1(precision: float, recall: float) -> float:
        denominator = precision + recall
        if denominator == 0.0:
            return 0.0
        return (2 * precision * recall) / denominator

    def partial_match(true_label: str, predicted_label: str, d: float = 0.75) -> float:
        if true_label == predicted_label:
            return 1.0
        if true_label.split("-")[0] == predicted_label.split("-")[0]:
            return d / 2
        return 0.0

    class_hierarchical_precision = {
        class_name: safe_mean(
            [
                partial_match(target_class, predicted_class)
                for target_class, predicted_class in zip(y_true, y_pred, strict=True)
                if predicted_class == class_name
            ]
        )
        for class_name in class_names
    }
    class_hierarchical_recall = {
        class_name: safe_mean(
            [
                partial_match(target_class, predicted_class)
                for target_class, predicted_class in zip(y_true, y_pred, strict=True)
                if target_class == class_name
            ]
        )
        for class_name in class_names
    }
    class_hierarchical_f1 = {
        class_name: safe_f1(
            class_hierarchical_precision[class_name],
            class_hierarchical_recall[class_name],
        )
        for class_name in class_names
    }
    return {
        "hierarchical_precision": safe_mean(list(class_hierarchical_precision.values())),
        "hierarchical_recall": safe_mean(list(class_hierarchical_recall.values())),
        "hierarchical_f1": safe_mean(list(class_hierarchical_f1.values())),
        "class_wise_hierarchical": {
            class_name: {
                "hierarchical_precision": class_hierarchical_precision[class_name],
                "hierarchical_recall": class_hierarchical_recall[class_name],
                "hierarchical_f1": class_hierarchical_f1[class_name],
            }
            for class_name in class_names
        },
    }


def confidence_summary_by_predicted_class(
    rows: list[PredictionRow],
    labels: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.predicted_label].append(row.top1_confidence)

    summary_rows: list[dict[str, Any]] = []
    for label in labels:
        values = grouped.get(label, [])
        summary_rows.append(
            {
                "predicted_class": label,
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "min": float(np.min(values)) if values else None,
                "max": float(np.max(values)) if values else None,
            }
        )
    return summary_rows


def write_plots(
    output_dir: Path,
    rows: list[PredictionRow],
    labels: list[str],
    confusion: np.ndarray,
    probability_mass: np.ndarray,
    focused_confusion_labels: list[str],
    focused_confusion: np.ndarray,
    skip_plots: bool,
) -> list[str]:
    if skip_plots:
        return []
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/dcase2026_task1_matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written: list[str] = []

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("BSD10k test confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    if focused_confusion_labels:
        fig_width = max(6.0, 1.1 * len(focused_confusion_labels))
        fig_height = max(5.0, 1.0 * len(focused_confusion_labels))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(focused_confusion, cmap="Reds")
        ax.set_xticks(
            range(len(focused_confusion_labels)),
            labels=focused_confusion_labels,
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(len(focused_confusion_labels)), labels=focused_confusion_labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title("BSD10k test focused confusion matrix")
        for row_index in range(focused_confusion.shape[0]):
            for column_index in range(focused_confusion.shape[1]):
                value = int(focused_confusion[row_index, column_index])
                if value:
                    ax.text(
                        column_index,
                        row_index,
                        str(value),
                        ha="center",
                        va="center",
                        color="black",
                    )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = output_dir / "focused_confusion_matrix.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path))

    grouped_confidences: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped_confidences[row.predicted_label].append(row.top1_confidence)
    plot_labels = [label for label in labels if grouped_confidences.get(label)]
    fig, ax = plt.subplots(figsize=(12, 6))
    boxplot_kwargs = {
        "x": [grouped_confidences[label] for label in plot_labels],
    }
    try:
        ax.boxplot(boxplot_kwargs["x"], tick_labels=plot_labels)
    except TypeError:
        ax.boxplot(boxplot_kwargs["x"], labels=plot_labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Top-1 probability")
    ax.set_title("Top-1 confidence distribution by predicted class")
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    path = output_dir / "top1_confidence_by_predicted_class.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(probability_mass, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Class receiving probability mass")
    ax.set_ylabel("Top-1 predicted class")
    ax.set_title("Mean class probability mass grouped by predicted class")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = output_dir / "probability_mass_by_predicted_class.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    return written


def main() -> None:
    args = build_parser().parse_args()
    config = read_config(args.experiment_dir)
    bsd10k_root = resolve_bsd10k_root(args, config)
    fold = int(split_value(args.fold, config, "fold", 0))
    n_splits = int(split_value(args.n_splits, config, "n_splits", 5))
    validation_size = float(
        split_value(args.validation_size, config, "validation_size", 0.2)
    )
    seed = int(split_value(args.seed, config, "split_seed", DEFAULT_BSD_SPLIT_SEED))

    records = load_records_by_dataset_name("BSD10k", bsd10k_root)
    splits = build_stratified_folds(
        [int(record["class_idx"]) for record in records],
        n_splits=n_splits,
        validation_size=validation_size,
        seed=seed,
    )
    if not 0 <= fold < len(splits):
        raise ValueError(f"fold must be in [0, {len(splits) - 1}], got {fold}.")
    test_indices = splits[fold].test_indices

    label_names, logits_by_file_id = load_npz_predictions(args.experiment_dir / "bsd10k_logits.npz")
    rows, missing_indices = prediction_rows(records, test_indices, label_names, logits_by_file_id)
    if not rows:
        raise RuntimeError("No test records could be matched to exported BSD10k logits.")

    labels = list(label_names)
    focused_labels = [
        label for label in parse_class_list(args.focused_confusion_classes) if label in labels
    ]
    confusion = confusion_matrix(rows, labels)
    focused_confusion = confusion_matrix(
        [
            row
            for row in rows
            if row.true_label in focused_labels and row.predicted_label in focused_labels
        ],
        focused_labels,
    )
    probability_mass = probability_mass_matrix(rows, labels)
    top_confusions = top_confusion_rows(rows, args.top_confusions)
    topn = top_n_rows(rows, [1, 2, 3])
    hierarchical = compute_hierarchical_metrics(
        [row.true_label for row in rows],
        [row.predicted_label for row in rows],
        class_names=labels,
    )
    accuracy = topn[0]["accuracy"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(
        args.output_dir / "confusion_matrix.csv",
        confusion,
        "actual_class",
        labels,
        labels,
    )
    write_matrix_csv(
        args.output_dir / "focused_confusion_matrix.csv",
        focused_confusion,
        "actual_class",
        focused_labels,
        focused_labels,
    )
    write_matrix_csv(
        args.output_dir / "probability_mass_by_predicted_class.csv",
        probability_mass,
        "predicted_class",
        labels,
        labels,
    )
    write_csv(
        args.output_dir / "test_predictions.csv",
        [prediction_csv_row(row, labels) for row in rows],
        prediction_csv_fieldnames(labels),
    )
    ssu_ssi_confusion_rows = [
        row
        for row in rows
        if row.true_label != row.predicted_label
        and {row.true_label, row.predicted_label} == {"ss-u", "ss-i"}
    ]
    write_csv(
        args.output_dir / "ss-u_ss-i_confusions.csv",
        [prediction_csv_row(row, labels) for row in ssu_ssi_confusion_rows],
        prediction_csv_fieldnames(labels),
    )
    write_csv(
        args.output_dir / "top_n_accuracy.csv",
        topn,
        ["top_n", "correct_count", "total_count", "accuracy"],
    )
    write_csv(
        args.output_dir / "top1_confidence_by_predicted_class.csv",
        confidence_summary_by_predicted_class(rows, labels),
        ["predicted_class", "count", "mean", "median", "min", "max"],
    )
    write_csv(
        args.output_dir / "most_frequent_confusions.csv",
        top_confusions,
        [
            "actual_class",
            "predicted_class",
            "count",
            "fraction_of_all_predictions",
            "mean_top1_confidence",
        ],
    )
    (args.output_dir / "most_frequent_confusions.json").write_text(
        json.dumps(top_confusions, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_paths = write_plots(
        args.output_dir,
        rows,
        labels,
        confusion,
        probability_mass,
        focused_labels,
        focused_confusion,
        args.no_plots,
    )

    summary = {
        "experiment_dir": str(args.experiment_dir),
        "bsd10k_root": str(bsd10k_root),
        "output_dir": str(args.output_dir),
        "fold": fold,
        "n_splits": n_splits,
        "validation_size": validation_size,
        "seed": seed,
        "test_records": len(test_indices),
        "matched_test_predictions": len(rows),
        "missing_test_predictions": len(missing_indices),
        "accuracy": accuracy,
        "top_n_accuracy": topn,
        "hierarchical_metrics": hierarchical,
        "focused_confusion_classes": focused_labels,
        "ss-u_ss-i_confusions": len(ssu_ssi_confusion_rows),
        "plot_paths": plot_paths,
        "outputs": [
            "summary.json",
            "test_predictions.csv",
            "top_n_accuracy.csv",
            "confusion_matrix.csv",
            "confusion_matrix.png",
            "focused_confusion_matrix.csv",
            "focused_confusion_matrix.png",
            "top1_confidence_by_predicted_class.csv",
            "top1_confidence_by_predicted_class.png",
            "probability_mass_by_predicted_class.csv",
            "probability_mass_by_predicted_class.png",
            "most_frequent_confusions.csv",
            "most_frequent_confusions.json",
            "ss-u_ss-i_confusions.csv",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
