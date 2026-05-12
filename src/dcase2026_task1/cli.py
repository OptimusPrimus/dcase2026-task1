from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from dcase2026_task1.data.datasets import BSDCombinedDataset


def build_parser() -> argparse.ArgumentParser:
    default_bsd35k_root = str(Path.home() / "data" / "BSD35k-CS")
    default_bsd10k_root = str(Path.home() / "data" / "BSD10k")

    parser = argparse.ArgumentParser(
        description="Inspect the combined BSD35k-CS and BSD10k datasets."
    )
    parser.add_argument(
        "--bsd35k-root",
        default=default_bsd35k_root,
        help="Root directory of the BSD35k-CS dataset.",
    )
    parser.add_argument(
        "--bsd10k-root",
        default=default_bsd10k_root,
        help="Root directory of the BSD10k dataset.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip waveform loading and return metadata only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of items to print.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = BSDCombinedDataset(
        bsd35k_root=args.bsd35k_root,
        bsd10k_root=args.bsd10k_root,
        load_audio=not args.no_audio,
    )

    print(f"Dataset size: {len(dataset)}")
    for index in range(min(args.limit, len(dataset))):
        item = dataset[index]
        if "waveform" in item:
            item = {
                **item,
                "waveform_shape": tuple(item["waveform"].shape),
                "waveform": "<tensor>",
            }
        pprint(item)


if __name__ == "__main__":
    main()
