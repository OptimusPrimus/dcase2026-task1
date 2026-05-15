from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


PARSE_FAILURE_LABEL = "__PARSE_FAILURE__"
HIERARCHICAL_PARENT_MATCH_SCORE = 0.375
AUDIOSET_RUN_ID = "20260515_184526_BSD10k_audioset_2794f53e"
SPEECH_LABEL = "Speech"
SPEECH_THRESHOLD = 0.5
SPEECH_FALLBACK_CLASS = "fx-h"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions from a text_metadata_classification experiment."
    )
    parser.add_argument(
        "predictions",
        help="Path to predictions.jsonl written by text_metadata_classification.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the evaluation summary as JSON.",
    )
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def load_audioset_speech_scores() -> dict[str, float]:
    predictions_path = (
        Path(__file__).resolve().parent
        / "outputs"
        / "audio_tagging"
        / AUDIOSET_RUN_ID
        / "predictions.jsonl"
    )
    rows = load_jsonl(predictions_path)
    speech_scores: dict[str, float] = {}
    for row in rows:
        audio_path = str(row.get("audio_path") or "")
        if not audio_path:
            continue
        score = next(
            (
                float(class_probability["score"])
                for class_probability in row.get("class_probabilities", [])
                if class_probability.get("label") == SPEECH_LABEL
            ),
            0.0,
        )
        speech_scores[audio_path] = score
    return speech_scores


def load_candidate_map(predictions_path: Path) -> dict[str, str]:
    config_path = predictions_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Expected sibling config.json next to {predictions_path}, but none was found."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = config.get("candidate_classes", [])
    candidate_map: dict[str, str] = {}
    for index, candidate in enumerate(candidates, start=1):
        class_name = str(candidate["class_name"])
        candidate_map[str(index)] = class_name
        candidate_map[class_name] = class_name

    return candidate_map


def parsed_label_to_class_name(parsed_label: str | None, candidate_map: dict[str, str]) -> str:
    if parsed_label is None:
        return PARSE_FAILURE_LABEL
    return candidate_map.get(str(parsed_label), PARSE_FAILURE_LABEL)


def recover_prediction_from_response(
    row: dict[str, Any],
    candidate_map: dict[str, str],
) -> str:
    raw_response = str(row.get("raw_response") or "")
    raw_response_lower = raw_response.lower()
    class_names = sorted(
        {
            class_name
            for key, class_name in candidate_map.items()
            if key == class_name
        },
        key=len,
        reverse=True,
    )
    for class_name in class_names:
        if class_name.lower() in raw_response_lower:
            return class_name
    return PARSE_FAILURE_LABEL


def collect_labels(
    rows: list[dict[str, Any]],
    candidate_map: dict[str, str],
    audioset_speech_scores: dict[str, float] | None = None,
) -> tuple[list[str], list[str], int]:
    y_true: list[str] = []
    y_pred: list[str] = []
    parse_failures = 0
    audioset_speech_scores = audioset_speech_scores or {}

    for row in rows:
        target_class = str(row["target_class"])
        predicted_class = parsed_label_to_class_name(row.get("parsed_label"), candidate_map)

        if predicted_class == PARSE_FAILURE_LABEL:
            predicted_class = recover_prediction_from_response(row, candidate_map)

        if predicted_class == PARSE_FAILURE_LABEL:
            parse_failures += 1

        if predicted_class.startswith("sp-"):
            audio_path = str(row.get("audio_path") or "")
            speech_score = audioset_speech_scores.get(audio_path, 0.0)
            if speech_score <= SPEECH_THRESHOLD:
                predicted_class = SPEECH_FALLBACK_CLASS

        y_true.append(target_class)
        y_pred.append(predicted_class)

    return y_true, y_pred, parse_failures


def plot_class_precision_recall(
    class_hierarchical_precision: dict[str, float],
    class_hierarchical_recall: dict[str, float],
    output_dir: Path,
) -> None:
    labels = list(class_hierarchical_precision.keys())
    precision_values = [class_hierarchical_precision[label] for label in labels]
    recall_values = [class_hierarchical_recall[label] for label in labels]
    positions = list(range(len(labels)))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 6))
    ax.bar(
        [position - width / 2 for position in positions],
        precision_values,
        width=width,
        label="Hierarchical precision",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        recall_values,
        width=width,
        label="Hierarchical recall",
    )
    ax.set_title("Class-wise hierarchical precision and recall")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "class_precision_recall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    candidate_map: dict[str, str],
    output_dir: Path,
) -> None:
    labels = []
    seen: set[str] = set()
    for key, class_name in candidate_map.items():
        if key != class_name or class_name in seen:
            continue
        seen.add(class_name)
        labels.append(class_name)
    if PARSE_FAILURE_LABEL in y_pred:
        labels.append(PARSE_FAILURE_LABEL)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig_width = max(10, len(labels) * 0.6)
    fig_height = max(8, len(labels) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Exact-match confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_exact.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_sample_predictions_csv(
    rows: list[dict[str, Any]],
    candidate_map: dict[str, str],
    output_path: Path,
    audioset_speech_scores: dict[str, float] | None = None,
) -> None:
    fieldnames = [
        "audio_path",
        "filename",
        "title",
        "tags",
        "description",
        "target_class",
        "predicted_class",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            _, predicted_classes, _ = collect_labels([row], candidate_map, audioset_speech_scores)
            predicted_class = predicted_classes[0]

            audio_path = str(row.get("audio_path") or "")
            writer.writerow(
                {
                    "audio_path": audio_path,
                    "filename": Path(audio_path).name if audio_path else "",
                    "title": str(row.get("title") or ""),
                    "tags": str(row.get("tags") or ""),
                    "description": str(row.get("description") or ""),
                    "target_class": str(row.get("target_class") or ""),
                    "predicted_class": predicted_class,
                }
            )




def evaluate_predictions(
    rows: list[dict[str, Any]],
    candidate_map: dict[str, str],
    audioset_speech_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Predictions file is empty.")

    y_true, y_pred, parse_failures = collect_labels(rows, candidate_map, audioset_speech_scores)

    import numpy as np

    def partial_match(y_t, y_p, d=0.75):
        if y_t == y_p:
            return 1
        if y_t.split('-')[0] == y_t.split('-')[0]:
            return d / 2
        return 0

    hP = {c: np.mean([partial_match(y_t, y_p) for y_t, y_p in zip(y_true, y_pred) if y_p == c]).item() for c in set(y_true)}
    hR = {c: np.mean([partial_match(y_t, y_p) for y_t, y_p in zip(y_true, y_pred) if y_t == c]).item() for c in set(y_true)}
    hF = {c: (2*hP[c]*hR[c])/ (hP[c] + hR[c]) for c in set(y_true)}

    return {
        "num_items": len(rows),
        "num_parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(rows),
        "class_hierarchical_precision": hP,
        "class_hierarchical_recall": hR,
        "class_hierarchical_f1": hF,
        "hierarchical_precision": np.mean(list(hP.values())).item(),
        "hierarchical_recall": np.mean(list(hR.values())).item(),
        "hierarchical_f1": np.mean(list(hF.values())).item(),
    }


def main() -> None:
    args = build_parser().parse_args()
    predictions_path = Path(args.predictions)
    rows = load_jsonl(predictions_path)
    candidate_map = load_candidate_map(predictions_path)
    audioset_speech_scores = load_audioset_speech_scores()
    y_true, y_pred, _ = collect_labels(rows, candidate_map, audioset_speech_scores)
    summary = evaluate_predictions(rows, candidate_map, audioset_speech_scores)

    output_dir = predictions_path.parent / "evaluation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_class_precision_recall(
        summary["class_hierarchical_precision"],
        summary["class_hierarchical_recall"],
        output_dir,
    )
    plot_confusion_matrix(y_true, y_pred, candidate_map, output_dir)
    write_sample_predictions_csv(
        rows,
        candidate_map,
        output_dir / "sample_predictions.csv",
        audioset_speech_scores,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
