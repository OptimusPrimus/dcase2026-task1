#!/usr/bin/env python3
"""Create the DCASE Task 1 submission package."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TEAM = "Primus_CPJKU"
TASK = "task1"
META_SUFFIX = ".meta.yaml"
PREDICTION_COLUMN = "predicted_bst_second_level_class"


@dataclass(frozen=True)
class Submission:
    index: int
    source_dir: str | None
    system_summary: str
    audio_representation: str
    text_representation: str
    total_parameters: str

    @property
    def label(self) -> str:
        return f"{TEAM}_{TASK}_{self.index}"

    @property
    def name(self) -> str:
        return f"Ensemble {self.index}"

    @property
    def abbreviation(self) -> str:
        return f"ens_{self.index}"


SUBMISSIONS = [
    Submission(
        index=1,
        source_dir="ensemble_beats_6b71b13c__clap_30005d33__clap_8dd65654__lclap_b43b5985",
        system_summary="Ensemble of BEATs, 2x CP-CLAP (RoBERTa + PaSST), and LAION-CLAP (RoBERTa + tiny-HTSAT).",
        audio_representation="BEATs, CP-CLAP PaSST, LAION-CLAP tiny-HTSAT",
        text_representation="LLM prediction embeddings (title, tags, description), 2x CP-CLAP RoBERTa, LAION-CLAP RoBERTa",
        total_parameters="543 M",
    ),
    Submission(
        index=2,
        source_dir="ensemble_clap_30005d33__clap_8dd65654__m2d_543ed2a5",
        system_summary="Ensemble of 2x CP-CLAP (RoBERTa + PaSST) and M2D.",
        audio_representation="CP-CLAP PaSST, M2D",
        text_representation="LLM prediction embeddings (title, tags, description), 2x CP-CLAP RoBERTa",
        total_parameters="383 M",
    ),
    # Placeholder: set source_dir once submission 3 is available.
    Submission(
        index=3,
        source_dir="ensemble_clap_2f736a37__clap_50074c26__m2d_7268a87e",
        system_summary="Ensemble of 2x CP-CLAP (RoBERTa + PaSST) and M2D.",
        audio_representation="CP-CLAP PaSST, M2D",
        text_representation="LLM prediction embeddings (title, tags, description), 2x CP-CLAP RoBERTa",
        total_parameters="383 M",
    ),
    # Placeholder: set source_dir once submission 4 is available.
    Submission(
        index=4,
        source_dir="ensemble_clap_8dd65654__clap_e6413040__lclap_b43b5985__m2d_7268a87e",
        system_summary=(
            "Ensemble of 2x CP-CLAP (RoBERTa + PaSST), LAION-CLAP (RoBERTa + tiny-HTSAT), and M2D."
        ),
        audio_representation="CP-CLAP PaSST, LAION-CLAP tiny-HTSAT, M2D",
        text_representation="LLM prediction embeddings (title, tags, description), 2x CP-CLAP RoBERTa, LAION-CLAP RoBERTa",
        total_parameters="539 M",
    ),
]


def meta_header(submission: Submission) -> str:
    return f"""submission:

  label: {submission.label}


  name: {submission.name}

  abbreviation: {submission.abbreviation}

  authors:
    - lastname: Primus
      firstname: Paul
      email: paul.primus@jku.at           # Contact email address
      corresponding: true                    # Mark true for one of the authors
      affiliation:
        abbreviation: CP-JKU
        institute: Johannes Kepler University
        department: Institute of Computational Perception
        location: Linz, Austria

    # Second author
    - lastname: Widmer
      firstname: Gerhard
      email: gerhard.widmer@jku.at
      affiliation:
        abbreviation: CP-JKU
        institute: Johannes Kepler University
        department: Institute of Computational Perception
        location: Linz, Austria

system:
  source_code: https://github.com/OptimusPrimus/dcase2026_task1

  description:

    input_sampling_rate: variable

    summary: {submission.system_summary}

    audio_representation: {submission.audio_representation}

    text_representation: {submission.text_representation}

    data_augmentation: !!null

    machine_learning_method: transformer, CLAP, LLM

    external_data_usage: pretrained audio model embeddings, CLAP embeddings, GPT-5.4-mini
    hierarchical_setting: !!null

  complexity:
    total_parameters: {submission.total_parameters}

"""


def read_results_yaml(source_dir: Path) -> str:
    evaluation_path = source_dir / "system_evaluation.yml"
    text = evaluation_path.read_text(encoding="utf-8")
    results_start = text.find("results:")
    if results_start == -1:
        raise ValueError(f"{evaluation_path} does not contain a 'results:' section")
    return text[results_start:].rstrip() + "\n"


def create_submission(source_root: Path, output_root: Path, submission: Submission) -> Path | None:
    if submission.source_dir is None:
        return None

    source_dir = source_root / submission.source_dir
    predictions_path = source_dir / "bsd2k_predictions.csv"
    evaluation_path = source_dir / "system_evaluation.yml"
    missing = [path for path in (predictions_path, evaluation_path) if not path.exists()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required file(s) for {submission.label}: {missing_paths}")

    submission_dir = output_root / submission.label
    submission_dir.mkdir(parents=True, exist_ok=True)

    output_predictions_path = submission_dir / f"{submission.label}.output.csv"
    shutil.copy2(predictions_path, output_predictions_path)

    meta_text = meta_header(submission) + read_results_yaml(source_dir)
    (submission_dir / f"{submission.label}{META_SUFFIX}").write_text(meta_text, encoding="utf-8")

    return output_predictions_path


def read_predictions(predictions_path: Path) -> dict[str, str]:
    with predictions_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"id", PREDICTION_COLUMN}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{predictions_path} is missing required column(s): {missing}")
        return {row["id"]: row[PREDICTION_COLUMN] for row in reader}


def load_default_font(size: int = 12) -> ImageFont.ImageFont:
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2
    draw.text((x, y), text, font=font, fill=fill)


def agreement_rate(left: dict[str, str], right: dict[str, str], left_label: str, right_label: str) -> float:
    left_ids = set(left)
    right_ids = set(right)
    if left_ids != right_ids:
        only_left = len(left_ids - right_ids)
        only_right = len(right_ids - left_ids)
        raise ValueError(
            f"Prediction IDs differ for {left_label} and {right_label}: "
            f"{only_left} only in {left_label}, {only_right} only in {right_label}"
        )
    if not left:
        raise ValueError(f"No predictions found for {left_label}")
    matches = sum(left[prediction_id] == right[prediction_id] for prediction_id in left)
    return matches / len(left)


def plot_agreement(predictions: dict[str, dict[str, str]], output_path: Path) -> None:
    labels = list(predictions)
    cell_size = 92
    left_margin = 170
    top_margin = 110
    right_margin = 28
    bottom_margin = 42
    width = left_margin + cell_size * len(labels) + right_margin
    height = top_margin + cell_size * len(labels) + bottom_margin

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_default_font(18)
    label_font = load_default_font(12)
    value_font = load_default_font(15)

    draw.text((24, 20), "Prediction agreement", font=title_font, fill=(20, 28, 36))
    draw.text((24, 48), "Share of equal predicted classes after aligning by id", font=label_font, fill=(87, 96, 106))

    for index, label in enumerate(labels):
        short_label = label.replace(f"{TEAM}_{TASK}_", "sub ")
        x0 = left_margin + index * cell_size
        draw_centered_text(draw, (x0, 72, x0 + cell_size, top_margin), short_label, label_font, (45, 52, 60))
        y0 = top_margin + index * cell_size
        draw.text((24, y0 + cell_size / 2 - 8), short_label, font=label_font, fill=(45, 52, 60))

    for row_index, row_label in enumerate(labels):
        for column_index, column_label in enumerate(labels):
            rate = agreement_rate(
                predictions[row_label],
                predictions[column_label],
                row_label,
                column_label,
            )
            intensity = int(255 - 155 * rate)
            fill = (intensity, 210, 235)
            x0 = left_margin + column_index * cell_size
            y0 = top_margin + row_index * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(220, 226, 232))
            draw_centered_text(draw, (x0, y0, x1, y1), f"{rate:.1%}", value_font, (18, 32, 44))

    image.save(output_path)


def plot_class_distribution(predictions: dict[str, dict[str, str]], output_path: Path) -> None:
    labels = list(predictions)
    counts_by_submission = {
        label: Counter(submission_predictions.values()) for label, submission_predictions in predictions.items()
    }
    classes = sorted({class_name for counts in counts_by_submission.values() for class_name in counts})
    max_count = max((max(counts.values()) for counts in counts_by_submission.values() if counts), default=1)

    bar_width = 16
    group_gap = 10
    left_margin = 74
    right_margin = 24
    top_margin = 72
    bottom_margin = 122
    plot_height = 320
    group_width = bar_width * len(labels) + group_gap
    width = max(900, left_margin + group_width * len(classes) + right_margin)
    height = top_margin + plot_height + bottom_margin

    colors = [(40, 115, 190), (230, 138, 50), (80, 155, 85), (170, 85, 150)]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_default_font(18)
    label_font = load_default_font(11)
    small_font = load_default_font(9)

    draw.text((24, 20), "Predicted class distribution", font=title_font, fill=(20, 28, 36))

    plot_left = left_margin
    plot_top = top_margin
    plot_bottom = top_margin + plot_height
    plot_right = width - right_margin
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(170, 178, 186), width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(170, 178, 186), width=1)

    for tick_index in range(5):
        value = round(max_count * tick_index / 4)
        y = plot_bottom - int((value / max_count) * plot_height)
        draw.line((plot_left - 4, y, plot_right, y), fill=(232, 236, 240), width=1)
        draw.text((20, y - 7), str(value), font=small_font, fill=(87, 96, 106))

    for class_index, class_name in enumerate(classes):
        group_x = left_margin + class_index * group_width
        for submission_index, label in enumerate(labels):
            count = counts_by_submission[label][class_name]
            bar_height = int((count / max_count) * (plot_height - 1))
            x0 = group_x + submission_index * bar_width
            y0 = plot_bottom - bar_height
            x1 = x0 + bar_width - 2
            draw.rectangle((x0, y0, x1, plot_bottom), fill=colors[submission_index % len(colors)])

        text_x = group_x + (bar_width * len(labels)) / 2 - 12
        draw.text((text_x, plot_bottom + 10), class_name, font=small_font, fill=(45, 52, 60))

    legend_x = left_margin
    legend_y = height - 44
    for submission_index, label in enumerate(labels):
        x = legend_x + submission_index * 190
        draw.rectangle((x, legend_y, x + 12, legend_y + 12), fill=colors[submission_index % len(colors)])
        draw.text((x + 18, legend_y - 1), label, font=label_font, fill=(45, 52, 60))

    image.save(output_path)


def create_analysis_plots(prediction_paths: dict[str, Path], submission_dir: Path) -> None:
    predictions = {label: read_predictions(path) for label, path in prediction_paths.items()}
    if len(predictions) < 2:
        return

    plot_agreement(predictions, submission_dir / "prediction_agreement.png")
    plot_class_distribution(predictions, submission_dir / "predicted_class_distribution.png")


def create_package(submission_dir: Path, output_dir: Path, clean: bool) -> Path:
    output_root = output_dir / TASK
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    prediction_paths = {}
    for submission in SUBMISSIONS:
        prediction_path = create_submission(submission_dir, output_root, submission)
        if prediction_path is not None:
            prediction_paths[submission.label] = prediction_path

    create_analysis_plots(prediction_paths, submission_dir)

    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory that will contain the task1 package directory. Defaults to submission/package.",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove an existing task1 package directory before recreating it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir or submission_dir / "package"

    output_root = create_package(submission_dir=submission_dir, output_dir=output_dir, clean=args.clean)
    print(f"Created submission package at {output_root}")


if __name__ == "__main__":
    main()
