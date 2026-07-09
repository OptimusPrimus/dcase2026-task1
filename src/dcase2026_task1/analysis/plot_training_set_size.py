from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "analysis" / "plot"

CONFIDENCE_SUBSET_FH1 = {
    0.1: [0.77903, 0.77786, 0.77494],
    0.2: [0.77936, 0.7921, 0.79227],
    0.4: [0.80584, 0.80514, 0.7993],
    0.6: [0.81287, 0.81085, 0.81102],
    0.8: [0.80709, 0.81251, 0.81325],
    1.0: [0.81103, 0.81803, 0.80766],
}

# Placeholder until the random-subset runs are available.
RANDOM_SUBSET_FH1 = {
    0.1: [0.78981, 0.79652, 0.79549],
    0.2: [0.80281, 0.80652, 0.79549],
    0.4: [0.80605, 0.80894, 0.81829],
    0.6: [0.81426, 0.81588, 0.80764],
    0.8: [0.81164, 0.81159, 0.80753],
    1.0: [0.81103, 0.81803, 0.80766],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the impact of BSD35k-CS training-set size on hierarchical F1."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where training_set_size_fh1.{pdf,png} are written.",
    )
    return parser


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (3.35, 1.5),
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def series_stats(values_by_size: dict[float, list[float]]) -> tuple[list[float], list[float], list[float | None]]:
    sizes = sorted(values_by_size)
    means = [mean(values_by_size[size]) for size in sizes]
    stds = [
        stdev(values_by_size[size]) if len(values_by_size[size]) > 1 else None
        for size in sizes
    ]
    return sizes, means, stds


def plot_series(
    ax: plt.Axes,
    sizes: list[float],
    means: list[float],
    stds: list[float | None],
    *,
    color: str,
    marker: str,
    label: str,
    linestyle: str = "-",
) -> None:
    ax.plot(
        sizes,
        means,
        color=color,
        marker=marker,
        markersize=3.5,
        linewidth=1.4,
        linestyle=linestyle,
        label=label,
    )

    errorbar_sizes = [size for size, std in zip(sizes, stds, strict=True) if std is not None]
    errorbar_means = [
        value for value, std in zip(means, stds, strict=True) if std is not None
    ]
    errorbar_stds = [std for std in stds if std is not None]
    ax.errorbar(
        errorbar_sizes,
        errorbar_means,
        yerr=errorbar_stds,
        fmt="none",
        ecolor=color,
        elinewidth=0.9,
        capsize=2.2,
        capthick=0.9,
        alpha=0.8,
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()

    confidence_sizes, confidence_means, confidence_stds = series_stats(CONFIDENCE_SUBSET_FH1)
    random_sizes, random_means, random_stds = series_stats(RANDOM_SUBSET_FH1)

    fig, ax = plt.subplots()
    plot_series(
        ax,
        random_sizes,
        random_means,
        random_stds,
        color="#7f7f7f",
        marker="s",
        label="random subset",
        linestyle="--",
    )
    plot_series(
        ax,
        confidence_sizes,
        confidence_means,
        confidence_stds,
        color="#1f77b4",
        marker="o",
        label="most confident subset",
    )

    ax.set_xlabel("Relative training-set size")
    ax.set_ylabel("Hierarchical F1")
    ax.set_xlim(0.07, 1.03)
    ax.set_ylim(0.745, 0.825)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.2", "0.4", "0.6", "0.8", "1"])
    ax.grid(True, axis="y", color="0.88", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)

    fig.tight_layout(pad=0.25)
    fig.savefig(output_dir / "training_set_size_fh1.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "training_set_size_fh1.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
