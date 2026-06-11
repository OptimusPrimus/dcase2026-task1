"""Vendored Microsoft BEATs source tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .BEATs import BEATs, BEATsConfig
from .chunked import ChunkedBEATs

DEFAULT_CHECKPOINT_ALIAS = "beats_iter3plus_as2m"


def resolve_checkpoint_path(
    checkpoint_dir: str | Path,
    checkpoint_alias: str = DEFAULT_CHECKPOINT_ALIAS,
) -> Path:
    resolved_checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    destination = resolved_checkpoint_dir / f"{checkpoint_alias}.pt"
    if not destination.exists():
        raise FileNotFoundError(f"Checkpoint not found at {destination}.")
    return destination


def validate_checkpoint_file(path: Path) -> None:
    header = path.read_bytes()[:512]
    lowered = header.lower().lstrip()
    if (
        lowered.startswith(b"<!doctype html")
        or lowered.startswith(b"<html")
        or lowered.startswith(b"<?xml")
    ):
        raise ValueError(
            f"{path} does not look like a PyTorch checkpoint. It looks like an HTML/XML response instead, "
            "which usually means the OneDrive share page was downloaded instead of the checkpoint file. "
            f"Delete {path} and re-run with a real checkpoint file via --checkpoint-path, or provide a direct "
            "download URL via --checkpoint-url."
        )
    if b"<title>" in lowered[:256] or b"onedrive" in lowered[:256]:
        raise ValueError(
            f"{path} appears to contain a web page instead of checkpoint bytes. "
            f"Delete {path} and re-run with --checkpoint-path pointing at a valid .pt file."
        )


def load_embedding_checkpoint(
    checkpoint_dir: str | Path,
    checkpoint_alias: str = DEFAULT_CHECKPOINT_ALIAS,
    *,
    trust_checkpoint: bool,
) -> dict[str, Any]:
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_dir=checkpoint_dir,
        checkpoint_alias=checkpoint_alias,
    )
    validate_checkpoint_file(checkpoint_path)
    if not trust_checkpoint:
        raise ValueError(
            "Original BEATs checkpoints require torch.load(..., weights_only=False). "
            "Re-run with --trust-checkpoint only if the checkpoint source is trusted."
        )
    return torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )


def build_beats_embedding_model(
    *,
    checkpoint_dir: str | Path,
    checkpoint_alias: str = DEFAULT_CHECKPOINT_ALIAS,
    trust_checkpoint: bool,
    sample_rate: int,
) -> ChunkedBEATs:
    checkpoint = load_embedding_checkpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_alias=checkpoint_alias,
        trust_checkpoint=trust_checkpoint,
    )
    config = BEATsConfig(checkpoint["cfg"])
    config.finetuned_model = False
    beats_model = BEATs(config)
    state_dict = {
        key: value
        for key, value in checkpoint["model"].items()
        if not key.startswith("predictor.")
    }
    missing_keys, unexpected_keys = beats_model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected BEATs checkpoint keys: {unexpected_keys}")
    non_predictor_missing = [
        key for key in missing_keys if not key.startswith("predictor")
    ]
    if non_predictor_missing:
        raise RuntimeError(f"Missing BEATs checkpoint keys: {non_predictor_missing}")

    model = ChunkedBEATs(
        beats_model,
        sample_rate=sample_rate,
        max_audio_seconds=10,
    )
    model.output_dim = int(config.encoder_embed_dim)
    model.checkpoint_cfg = checkpoint["cfg"]
    return model


__all__ = [
    "DEFAULT_CHECKPOINT_ALIAS",
    "BEATs",
    "BEATsConfig",
    "ChunkedBEATs",
    "build_beats_embedding_model",
    "load_embedding_checkpoint",
    "resolve_checkpoint_path",
    "validate_checkpoint_file",
]
