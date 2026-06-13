from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from dcase2026_task1.data.splits import (
    DEFAULT_BSD_SPLIT_SEED,
    build_stratified_folds,
    load_records_by_dataset_name,
)
from dcase2026_task1.experiments.training import (
    DEFAULT_BSD10K_ROOT,
    DEFAULT_OUTPUT_ROOT,
    compute_classification_metrics,
    resolve_record_file_id,
)

BSD10K_LOGITS_FILENAME = "bsd10k_logits.npz"
PREDICTION_FILENAMES_BY_DATASET = {
    "BSD10k": "bsd10k_logits.npz",
    "BSD35k-CS": "bsd35k_cs_logits.npz",
    "BSD2k": "bsd2k_hidden_test_logits.npz",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Average exported logits from one or more training runs, write ensembled "
            "prediction files, and evaluate BSD10k on the same validation/test split "
            "used during training."
        )
    )
    parser.add_argument(
        "models",
        nargs="+",
        help=(
            "Training run directory names or paths. Directory names are resolved "
            "relative to --output-root."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory containing training run directories.",
    )
    parser.add_argument(
        "--bsd10k-root",
        default=None,
        help="Override BSD10k root. Defaults to the first model config, then the training default.",
    )
    parser.add_argument(
        "--prediction-filename",
        default=BSD10K_LOGITS_FILENAME,
        help="BSD10k prediction npz filename inside each model directory.",
    )
    parser.add_argument(
        "--ensemble-root",
        default=None,
        help="Root where the ensemble output folder is created. Defaults to --output-root.",
    )
    parser.add_argument(
        "--ensemble-dir",
        default=None,
        help=(
            "Override ensemble output directory. Defaults to "
            "ensemble_{sorted model directory names} under --ensemble-root."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path where the ensemble metrics JSON should be written.",
    )
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_model_dir(model: str, output_root: Path) -> Path:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return model_path.resolve()

    candidate = output_root / model
    if candidate.exists():
        return candidate.resolve()

    matches = sorted(path for path in output_root.glob(f"*{model}*") if path.is_dir())
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        match_names = ", ".join(path.name for path in matches[:10])
        raise ValueError(
            f"Model name {model!r} is ambiguous under {output_root}: {match_names}."
        )
    raise FileNotFoundError(f"Could not resolve model {model!r} under {output_root}.")


def sanitize_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "model"


def default_ensemble_dir(model_dirs: list[Path], ensemble_root: Path) -> Path:
    model_names = sorted(sanitize_path_part(model_dir.name) for model_dir in model_dirs)
    return ensemble_root / f"ensemble_{'__'.join(model_names)}"


def labels_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    labels = config.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("Model config does not contain a non-empty labels list.")
    return labels


def label_names_from_config(config: dict[str, Any]) -> list[str]:
    labels = labels_from_config(config)
    return [str(item["class_name"]) for item in sorted(labels, key=lambda item: int(item["label_id"]))]


def label_map_from_config(config: dict[str, Any]) -> dict[int, int]:
    labels = labels_from_config(config)
    return {
        int(item["dataset_class_idx"]): int(item["label_id"])
        for item in labels
    }


def id2label_from_config(config: dict[str, Any]) -> dict[int, str]:
    labels = labels_from_config(config)
    return {
        int(item["label_id"]): str(item["class_name"])
        for item in labels
    }


def require_matching_model_configs(configs: list[dict[str, Any]]) -> None:
    if not configs:
        raise ValueError("At least one model config is required.")

    reference_labels = label_names_from_config(configs[0])
    reference_split = configs[0].get("split", {})
    for index, config in enumerate(configs[1:], start=1):
        labels = label_names_from_config(config)
        if labels != reference_labels:
            raise ValueError(
                f"Model config at position {index} has different labels from the first model."
            )

        split = config.get("split", {})
        for key in ("fold", "n_splits", "validation_size", "split_seed"):
            if split.get(key) != reference_split.get(key):
                raise ValueError(
                    f"Model config at position {index} uses a different split {key!r}: "
                    f"{split.get(key)!r} != {reference_split.get(key)!r}."
                )


def load_logits_npz(path: Path, expected_label_names: list[str]) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    with np.load(path, allow_pickle=False) as data:
        if "label_names" not in data:
            raise ValueError(f"Prediction file {path} does not contain label_names.")
        label_names = [str(label) for label in data["label_names"].tolist()]
        if label_names != expected_label_names:
            raise ValueError(f"Prediction file {path} has labels that do not match the model config.")

        return {
            key: np.asarray(data[key], dtype=np.float64)
            for key in data.files
            if key != "label_names"
        }


def write_logits_npz(
    path: Path,
    logits_by_file_id: dict[str, np.ndarray],
    label_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        file_id: np.asarray(logits, dtype=np.float32)
        for file_id, logits in sorted(logits_by_file_id.items())
    }
    payload["label_names"] = np.asarray(label_names, dtype=np.str_)
    np.savez(path, **payload)


def average_logits_by_file_id(
    logits_per_model: list[dict[str, np.ndarray]],
    dataset_name: str,
) -> dict[str, np.ndarray]:
    if not logits_per_model:
        raise ValueError("At least one model prediction file is required.")

    reference_file_ids = set(logits_per_model[0])
    for model_index, logits_by_file_id in enumerate(logits_per_model[1:], start=1):
        file_ids = set(logits_by_file_id)
        missing = sorted(reference_file_ids - file_ids)
        extra = sorted(file_ids - reference_file_ids)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing {len(missing)} ids, first={missing[0]!r}")
            if extra:
                parts.append(f"extra {len(extra)} ids, first={extra[0]!r}")
            raise ValueError(
                f"Model at position {model_index} has different {dataset_name} file IDs: "
                + "; ".join(parts)
            )

    averaged: dict[str, np.ndarray] = {}
    for file_id in sorted(reference_file_ids):
        rows = [logits_by_file_id[file_id] for logits_by_file_id in logits_per_model]
        averaged[file_id] = np.mean(np.stack(rows, axis=0), axis=0)
    return averaged


def write_ensembled_prediction_files(
    model_dirs: list[Path],
    output_dir: Path,
    label_names: list[str],
    prediction_filenames_by_dataset: dict[str, str] | None = None,
) -> dict[str, str]:
    filenames_by_dataset = prediction_filenames_by_dataset or PREDICTION_FILENAMES_BY_DATASET
    output_paths: dict[str, str] = {}
    for dataset_name, filename in filenames_by_dataset.items():
        logits_per_model = [
            load_logits_npz(model_dir / filename, label_names)
            for model_dir in model_dirs
        ]
        averaged = average_logits_by_file_id(logits_per_model, dataset_name)
        output_path = output_dir / filename
        write_logits_npz(output_path, averaged, label_names)
        output_paths[dataset_name] = str(output_path)
    return output_paths


def average_logits_for_records(
    records: list[dict[str, Any]],
    logits_per_model: list[dict[str, np.ndarray]],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for record in records:
        file_id = resolve_record_file_id(record)
        model_rows: list[np.ndarray] = []
        for model_index, logits_by_file_id in enumerate(logits_per_model):
            try:
                model_rows.append(logits_by_file_id[file_id])
            except KeyError as exc:
                raise KeyError(
                    f"Model at position {model_index} is missing BSD10k logits for file_id={file_id!r}."
                ) from exc
        rows.append(np.mean(np.stack(model_rows, axis=0), axis=0))
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(rows, axis=0)


def labels_for_records(records: list[dict[str, Any]], label_map: dict[int, int]) -> np.ndarray:
    labels: list[int] = []
    for record in records:
        class_idx = int(record["class_idx"])
        try:
            labels.append(label_map[class_idx])
        except KeyError as exc:
            raise KeyError(f"No label mapping for dataset class_idx={class_idx}.") from exc
    return np.asarray(labels, dtype=np.int64)


def build_bsd10k_eval_records(
    bsd10k_root: Path,
    split_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = load_records_by_dataset_name("BSD10k", bsd10k_root)
    split_seed = int(split_config.get("split_seed", DEFAULT_BSD_SPLIT_SEED))
    if split_seed != DEFAULT_BSD_SPLIT_SEED:
        raise ValueError(
            f"Unsupported split_seed={split_seed}. Training currently uses {DEFAULT_BSD_SPLIT_SEED}."
        )

    folds = build_stratified_folds(
        labels=[int(record["class_idx"]) for record in records],
        n_splits=int(split_config["n_splits"]),
        validation_size=float(split_config["validation_size"]),
        seed=split_seed,
    )
    fold_index = int(split_config["fold"])
    if not 0 <= fold_index < len(folds):
        raise ValueError(f"fold must be in [0, {len(folds) - 1}], got {fold_index}.")

    split = folds[fold_index]
    val_records = [records[index] for index in split.val_indices]
    test_records = [records[index] for index in split.test_indices]

    val_size = split_config.get("val_size")
    test_size = split_config.get("test_size")
    if val_size is not None:
        val_records = val_records[: int(val_size)]
    if test_size is not None:
        test_records = test_records[: int(test_size)]
    return val_records, test_records


def evaluate_ensemble(
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    logits_per_model: list[dict[str, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    label_map = label_map_from_config(config)
    id2label = id2label_from_config(config)
    num_labels = len(id2label)

    val_logits = average_logits_for_records(val_records, logits_per_model)
    test_logits = average_logits_for_records(test_records, logits_per_model)
    val_labels = labels_for_records(val_records, label_map)
    test_labels = labels_for_records(test_records, label_map)

    return {
        "val": compute_classification_metrics(
            val_logits,
            val_labels,
            num_labels,
            id2label=id2label,
        ),
        "test": compute_classification_metrics(
            test_logits,
            test_labels,
            num_labels,
            id2label=id2label,
        ),
        "counts": {
            "val": len(val_records),
            "test": len(test_records),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser()
    model_dirs = [resolve_model_dir(model, output_root) for model in args.models]
    configs = [read_json(model_dir / "config.json") for model_dir in model_dirs]
    require_matching_model_configs(configs)

    reference_config = configs[0]
    label_names = label_names_from_config(reference_config)
    ensemble_root = Path(args.ensemble_root or output_root).expanduser()
    ensemble_dir = (
        Path(args.ensemble_dir).expanduser()
        if args.ensemble_dir is not None
        else default_ensemble_dir(model_dirs, ensemble_root)
    )
    ensemble_prediction_paths = write_ensembled_prediction_files(
        model_dirs=model_dirs,
        output_dir=ensemble_dir,
        label_names=label_names,
        prediction_filenames_by_dataset={
            **PREDICTION_FILENAMES_BY_DATASET,
            "BSD10k": args.prediction_filename,
        },
    )
    logits_per_model = [
        load_logits_npz(model_dir / args.prediction_filename, label_names)
        for model_dir in model_dirs
    ]

    config_bsd10k_root = (
        reference_config.get("dataset_roots", {}).get("BSD10k")
        if isinstance(reference_config.get("dataset_roots"), dict)
        else None
    )
    bsd10k_root = Path(args.bsd10k_root or config_bsd10k_root or DEFAULT_BSD10K_ROOT).expanduser()
    split_config = reference_config["split"]
    val_records, test_records = build_bsd10k_eval_records(bsd10k_root, split_config)
    results = evaluate_ensemble(
        val_records=val_records,
        test_records=test_records,
        logits_per_model=logits_per_model,
        config=reference_config,
    )
    results.update(
        {
            "models": [str(model_dir) for model_dir in model_dirs],
            "prediction_filename": args.prediction_filename,
            "ensemble_dir": str(ensemble_dir),
            "ensemble_prediction_paths": ensemble_prediction_paths,
            "bsd10k_root": str(bsd10k_root),
            "split": {
                "fold": split_config["fold"],
                "n_splits": split_config["n_splits"],
                "validation_size": split_config["validation_size"],
                "split_seed": split_config.get("split_seed", DEFAULT_BSD_SPLIT_SEED),
            },
        }
    )
    output_json = (
        Path(args.output_json).expanduser()
        if args.output_json is not None
        else ensemble_dir / "ensemble_metrics.json"
    )
    write_json(output_json, results)
    return results


def main() -> None:
    args = build_parser().parse_args()
    results = run(args)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
