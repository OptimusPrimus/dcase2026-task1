from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


PARSE_FAILURE_LABEL = "__PARSE_FAILURE__"
HIERARCHICAL_PARENT_MATCH_SCORE = 0.375


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




def evaluate_predictions(
    rows: list[dict[str, Any]],
    candidate_map: dict[str, str],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Predictions file is empty.")

    y_true: list[str] = []
    y_pred: list[str] = []
    parse_failures = 0

    for row in rows:
        target_class = str(row["target_class"])
        predicted_class = parsed_label_to_class_name(row.get("parsed_label"), candidate_map)

        if predicted_class == PARSE_FAILURE_LABEL:
            predicted_class = recover_prediction_from_response(row, candidate_map)

        if predicted_class == PARSE_FAILURE_LABEL:
            parse_failures += 1

        y_true.append(target_class)
        y_pred.append(predicted_class)

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
    summary = evaluate_predictions(rows, candidate_map)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
