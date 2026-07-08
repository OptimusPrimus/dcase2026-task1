from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "analysis" / "plot"

PSEUDO_LABEL_WEIGHT_FH1 = {
    0.0: [0.61299, 0.62193, 0.59463],
    0.2: [0.71107, 0.70233, 0.69781],
    0.4: [0.75052, 0.74295],
    0.6: [0.77514, 0.76758],
    0.8: [0.80585, 0.79836],
    1.0: [0.80797, 0.80331],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the impact of pseudo-label weight on hierarchical F1."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where pseudo_label_weight_fh1.{pdf,png} are written.",
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


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()

    weights = sorted(PSEUDO_LABEL_WEIGHT_FH1)
    means = [mean(PSEUDO_LABEL_WEIGHT_FH1[weight]) for weight in weights]
    stds = [
        stdev(PSEUDO_LABEL_WEIGHT_FH1[weight])
        if len(PSEUDO_LABEL_WEIGHT_FH1[weight]) > 1
        else None
        for weight in weights
    ]

    fig, ax = plt.subplots()
    ax.plot(
        weights,
        means,
        color="#1f77b4",
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        label="mean hF1",
    )

    errorbar_weights = [
        weight for weight, std in zip(weights, stds, strict=True) if std is not None
    ]
    errorbar_means = [
        value for value, std in zip(means, stds, strict=True) if std is not None
    ]
    errorbar_stds = [std for std in stds if std is not None]
    ax.errorbar(
        errorbar_weights,
        errorbar_means,
        yerr=errorbar_stds,
        fmt="none",
        ecolor="#1f77b4",
        elinewidth=0.9,
        capsize=2.2,
        capthick=0.9,
        alpha=0.8,
        # label="standard deviation",
    )

    ax.set_xlabel(r"Pseudo-label loss weight $\lambda$")
    ax.set_ylabel("Hierarchical F1")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.58, 0.83)
    ax.set_xticks(weights)
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.grid(True, axis="y", color="0.88", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)

    fig.tight_layout(pad=0.25)
    fig.savefig(output_dir / "pseudo_label_weight_fh1.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "pseudo_label_weight_fh1.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
