#!/usr/bin/env python3
"""Verify DCASE Task 1 submission outputs against the BSD2k hidden-test set."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_COLUMNS = ("id", "predicted_bst_second_level_class")
SCORE_COLUMN = "prediction_score"
DEFAULT_BSD2K_ROOT = Path.home() / "data" / "BSD2k"
DEFAULT_BSD10K_ROOT = Path.home() / "data" / "BSD10k"


@dataclass
class CheckResult:
    path: Path
    row_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    class_counts: Counter[str] = field(default_factory=Counter)
    score_min: float | None = None
    score_max: float | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def fail(message: str) -> None:
    raise SystemExit(message)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("file has no CSV header")
        return list(reader)


def load_expected_ids(bsd2k_root: Path) -> tuple[set[str], set[str], list[str]]:
    metadata_path = bsd2k_root / "metadata" / "BSD2k_metadata.csv"
    audio_dir = bsd2k_root / "audio"
    missing = [path for path in (metadata_path, audio_dir) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        fail(f"Missing BSD2k files or directories: {missing_text}")

    rows = read_csv_rows(metadata_path)
    metadata_ids = {
        row["anonymous_id"].strip()
        for row in rows
        if row.get("anonymous_id", "").strip()
    }
    audio_ids = {path.stem for path in audio_dir.glob("*.wav")}

    layout_errors = []
    missing_audio = sorted(metadata_ids - audio_ids)
    extra_audio = sorted(audio_ids - metadata_ids)
    if missing_audio:
        layout_errors.append(
            f"BSD2k metadata has {len(missing_audio)} id(s) without .wav audio: "
            f"{format_examples(missing_audio)}"
        )
    if extra_audio:
        layout_errors.append(
            f"BSD2k audio has {len(extra_audio)} .wav file(s) not in metadata: "
            f"{format_examples(extra_audio)}"
        )
    return metadata_ids, audio_ids, layout_errors


def load_valid_labels(description_csv: Path, allow_other: bool) -> set[str]:
    if not description_csv.exists():
        fail(f"Missing class description CSV: {description_csv}")

    rows = read_csv_rows(description_csv)
    labels = set()
    for row in rows:
        class_key = row.get("class_key", "").strip()
        if "-" not in class_key:
            continue
        if not allow_other and class_key.endswith("-other"):
            continue
        labels.add(class_key)
    if not labels:
        fail(f"No second-level labels found in {description_csv}")
    return labels


def default_description_csv(bsd2k_root: Path) -> Path:
    bsd10k_description = DEFAULT_BSD10K_ROOT / "metadata" / "BST_description.csv"
    if bsd10k_description.exists():
        return bsd10k_description
    return bsd2k_root / "metadata" / "BST_description.csv"


def discover_output_csvs(path: Path) -> tuple[list[Path], tempfile.TemporaryDirectory[str] | None]:
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    target = path
    if path.suffix == ".zip":
        temp_dir = tempfile.TemporaryDirectory(prefix="dcase_task1_submission_")
        with zipfile.ZipFile(path) as archive:
            extract_zip_safely(archive, Path(temp_dir.name))
        target = Path(temp_dir.name)

    if target.is_file():
        if target.suffix != ".csv":
            fail(f"Expected a CSV file, package directory, or zip file, got: {path}")
        return [target], temp_dir

    if not target.exists():
        fail(f"Submission path does not exist: {path}")

    outputs = sorted(target.rglob("*.output.csv"))
    if not outputs:
        outputs = sorted(target.rglob("bsd2k_predictions.csv"))
    if not outputs:
        fail(f"No submission output CSVs found under {path}")
    return outputs, temp_dir


def extract_zip_safely(archive: zipfile.ZipFile, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    for member in archive.infolist():
        member_path = (target_root / member.filename).resolve()
        if not member_path.is_relative_to(target_root):
            fail(f"Refusing to extract zip member outside target directory: {member.filename}")
    archive.extractall(target_root)


def format_examples(values: list[str], limit: int = 5) -> str:
    examples = values[:limit]
    suffix = "" if len(values) <= limit else ", ..."
    return ", ".join(examples) + suffix


def check_package_siblings(output_path: Path, result: CheckResult) -> None:
    if output_path.name == "bsd2k_predictions.csv":
        return

    label = output_path.name.removesuffix(".output.csv")
    expected_meta = output_path.with_name(f"{label}.meta.yaml")
    if not expected_meta.exists():
        result.errors.append(f"missing sibling meta file: {expected_meta.name}")
    if output_path.parent.name != label:
        result.warnings.append(
            f"output file label {label!r} does not match directory {output_path.parent.name!r}"
        )


def verify_output_csv(
    path: Path,
    expected_ids: set[str],
    valid_labels: set[str],
    check_package: bool,
) -> CheckResult:
    result = CheckResult(path=path)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
            if missing_columns:
                result.errors.append(
                    "missing required column(s): " + ", ".join(missing_columns)
                )
                return result

            ids: list[str] = []
            scores: list[float] = []
            for row_index, row in enumerate(reader, start=2):
                prediction_id = row.get("id", "").strip()
                predicted_class = row.get("predicted_bst_second_level_class", "").strip()
                result.row_count += 1

                if not prediction_id:
                    result.errors.append(f"row {row_index}: empty id")
                else:
                    ids.append(prediction_id)

                if not predicted_class:
                    result.errors.append(f"row {row_index}: empty predicted class")
                elif predicted_class not in valid_labels:
                    result.errors.append(
                        f"row {row_index}: invalid predicted class {predicted_class!r}"
                    )
                else:
                    result.class_counts[predicted_class] += 1

                if SCORE_COLUMN in fieldnames:
                    raw_score = row.get(SCORE_COLUMN, "").strip()
                    if not raw_score:
                        result.errors.append(f"row {row_index}: empty prediction score")
                        continue
                    try:
                        score = float(raw_score)
                    except ValueError:
                        result.errors.append(
                            f"row {row_index}: non-numeric prediction score {raw_score!r}"
                        )
                        continue
                    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                        result.errors.append(
                            f"row {row_index}: prediction score outside [0, 1]: {raw_score!r}"
                        )
                    else:
                        scores.append(score)

        id_counts = Counter(ids)
        duplicate_ids = sorted(prediction_id for prediction_id, count in id_counts.items() if count > 1)
        if duplicate_ids:
            result.errors.append(
                f"{len(duplicate_ids)} duplicate id(s): {format_examples(duplicate_ids)}"
            )

        prediction_ids = set(ids)
        missing_predictions = sorted(expected_ids - prediction_ids)
        extra_predictions = sorted(prediction_ids - expected_ids)
        if missing_predictions:
            result.errors.append(
                f"missing predictions for {len(missing_predictions)} BSD2k audio file(s): "
                f"{format_examples(missing_predictions)}"
            )
        if extra_predictions:
            result.errors.append(
                f"contains {len(extra_predictions)} id(s) not in BSD2k hidden test: "
                f"{format_examples(extra_predictions)}"
            )

        if SCORE_COLUMN not in fieldnames:
            result.warnings.append("prediction_score column is absent")
        elif scores:
            result.score_min = min(scores)
            result.score_max = max(scores)
            if result.score_min == result.score_max:
                result.warnings.append(
                    f"all prediction scores are identical ({result.score_min:.10g})"
                )

        if result.row_count == 0:
            result.errors.append("no prediction rows found")
        elif len(result.class_counts) == 1:
            predicted_class = next(iter(result.class_counts))
            result.warnings.append(f"all rows predict the same class ({predicted_class})")

        if check_package:
            check_package_siblings(path, result)

    except csv.Error as exc:
        result.errors.append(f"CSV parse error: {exc}")
    except OSError as exc:
        result.errors.append(f"could not read file: {exc}")

    return result


def print_result(result: CheckResult) -> None:
    status = "OK" if result.ok else "FAIL"
    print(f"[{status}] {result.path}")
    print(f"  rows: {result.row_count}")
    if result.score_min is not None and result.score_max is not None:
        print(f"  score range: {result.score_min:.10g} .. {result.score_max:.10g}")
    if result.class_counts:
        top_classes = result.class_counts.most_common(5)
        formatted = ", ".join(f"{label}={count}" for label, count in top_classes)
        print(f"  top classes: {formatted}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    for error in result.errors:
        print(f"  error: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "package" / "task1",
        help=(
            "Submission output CSV, package directory, or zip file. "
            "Defaults to submission/package/task1."
        ),
    )
    parser.add_argument(
        "--bsd2k-root",
        type=Path,
        default=DEFAULT_BSD2K_ROOT,
        help=f"BSD2k dataset root. Defaults to {DEFAULT_BSD2K_ROOT}.",
    )
    parser.add_argument(
        "--description-csv",
        type=Path,
        default=None,
        help=(
            "BST_description.csv containing valid class_key values. "
            "Defaults to BSD10k metadata, then BSD2k metadata."
        ),
    )
    parser.add_argument(
        "--allow-other",
        action="store_true",
        help="Allow second-level '*-other' labels as plausible predictions.",
    )
    parser.add_argument(
        "--no-package-checks",
        action="store_true",
        help="Skip package label/meta sibling checks for *.output.csv files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bsd2k_root = args.bsd2k_root.expanduser()
    description_csv = (
        args.description_csv.expanduser()
        if args.description_csv is not None
        else default_description_csv(bsd2k_root)
    )

    metadata_ids, audio_ids, layout_errors = load_expected_ids(bsd2k_root)
    expected_ids = metadata_ids & audio_ids
    if not expected_ids:
        fail(f"No BSD2k hidden-test ids found in {bsd2k_root}")

    valid_labels = load_valid_labels(description_csv, allow_other=args.allow_other)
    output_paths, temp_dir = discover_output_csvs(args.submission.expanduser())

    try:
        print(f"BSD2k expected audio files: {len(expected_ids)}")
        print(f"Valid prediction labels: {len(valid_labels)} from {description_csv}")
        for layout_error in layout_errors:
            print(f"Dataset error: {layout_error}")

        results = [
            verify_output_csv(
                path=path,
                expected_ids=expected_ids,
                valid_labels=valid_labels,
                check_package=not args.no_package_checks,
            )
            for path in output_paths
        ]
        for result in results:
            print_result(result)

        if layout_errors or any(not result.ok for result in results):
            return 1
        print(f"Verified {len(results)} submission output file(s).")
        return 0
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    sys.exit(main())
