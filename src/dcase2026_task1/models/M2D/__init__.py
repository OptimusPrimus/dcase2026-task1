from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import torch

from dcase2026_task1.models.audio_wrappers import (
    ArbitraryLengthAudioWrapper,
    pack_segment_outputs,
)

DEFAULT_CHECKPOINT_ARCHIVE_ALIAS = "m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly.zip"
DEFAULT_CHECKPOINT_FILENAME = "weights_ep67it3124-0.48558.pth"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 10.01
_STAGING_ROOT = Path("/tmp/dcase2026_task1_m2d")


def _archive_stem(checkpoint_archive_alias: str) -> str:
    return Path(checkpoint_archive_alias).stem


def _stage_checkpoint_file(
    source_path: Path,
    *,
    checkpoint_archive_alias: str,
    checkpoint_filename: str,
) -> Path:
    staged_path = _STAGING_ROOT / _archive_stem(checkpoint_archive_alias) / checkpoint_filename
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    if not staged_path.exists() or source_path.stat().st_mtime > staged_path.stat().st_mtime:
        shutil.copy2(source_path, staged_path)
    return staged_path


def _extract_checkpoint_from_zip(
    archive_path: Path,
    *,
    checkpoint_archive_alias: str,
    checkpoint_filename: str,
) -> Path:
    staged_path = _STAGING_ROOT / _archive_stem(checkpoint_archive_alias) / checkpoint_filename
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    if staged_path.exists() and staged_path.stat().st_mtime >= archive_path.stat().st_mtime:
        return staged_path

    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if Path(name).name == checkpoint_filename]
        if not members:
            raise FileNotFoundError(
                f"{archive_path} does not contain {checkpoint_filename!r}."
            )
        with archive.open(members[0]) as source, staged_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    return staged_path


def resolve_checkpoint_path(
    checkpoint_dir: str | Path,
    checkpoint_archive_alias: str = DEFAULT_CHECKPOINT_ARCHIVE_ALIAS,
    checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
) -> Path:
    resolved_checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    archive_stem = _archive_stem(checkpoint_archive_alias)

    extracted_candidate = resolved_checkpoint_dir / archive_stem / checkpoint_filename
    if extracted_candidate.exists():
        return extracted_candidate

    direct_candidate = resolved_checkpoint_dir / checkpoint_filename
    if direct_candidate.exists():
        return _stage_checkpoint_file(
            direct_candidate,
            checkpoint_archive_alias=checkpoint_archive_alias,
            checkpoint_filename=checkpoint_filename,
        )

    archive_candidate = resolved_checkpoint_dir / checkpoint_archive_alias
    if archive_candidate.exists():
        return _extract_checkpoint_from_zip(
            archive_candidate,
            checkpoint_archive_alias=checkpoint_archive_alias,
            checkpoint_filename=checkpoint_filename,
        )

    recursive_matches = list(resolved_checkpoint_dir.rglob(checkpoint_filename))
    if recursive_matches:
        match = recursive_matches[0]
        if match.parent.name == archive_stem:
            return match
        return _stage_checkpoint_file(
            match,
            checkpoint_archive_alias=checkpoint_archive_alias,
            checkpoint_filename=checkpoint_filename,
        )

    raise FileNotFoundError(
        f"Checkpoint {checkpoint_filename!r} not found under {resolved_checkpoint_dir}."
    )


def validate_checkpoint_file(path: Path) -> None:
    header = path.read_bytes()[:512]
    lowered = header.lower().lstrip()
    if (
        lowered.startswith(b"<!doctype html")
        or lowered.startswith(b"<html")
        or lowered.startswith(b"<?xml")
    ):
        raise ValueError(
            f"{path} does not look like a PyTorch checkpoint. It looks like an HTML/XML response instead."
        )


def _extract_m2d_embeddings(
    model: torch.nn.Module,
    waveforms: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    metadata: list[dict[str, object]] | None = None,
) -> torch.Tensor:
    del padding_mask, metadata
    return model.encode(waveforms)


class ChunkedM2D(ArbitraryLengthAudioWrapper):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        sample_rate: int,
        max_audio_seconds: float,
    ) -> None:
        super().__init__(
            model,
            sample_rate=sample_rate,
            max_audio_seconds=max_audio_seconds,
            segment_forward=_extract_m2d_embeddings,
            aggregate_outputs=pack_segment_outputs,
        )


def build_m2d_embedding_model(
    *,
    checkpoint_dir: str | Path,
    trust_checkpoint: bool,
    sample_rate: int,
    checkpoint_archive_alias: str = DEFAULT_CHECKPOINT_ARCHIVE_ALIAS,
    checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
) -> ChunkedM2D:
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(
            f"M2D expects sample_rate={DEFAULT_SAMPLE_RATE}, got {sample_rate}."
        )
    if not trust_checkpoint:
        raise ValueError(
            "Original M2D checkpoints require torch.load(..., weights_only=False). "
            "Re-run with --trust-checkpoint only if the checkpoint source is trusted."
        )

    checkpoint_path = resolve_checkpoint_path(
        checkpoint_dir=checkpoint_dir,
        checkpoint_archive_alias=checkpoint_archive_alias,
        checkpoint_filename=checkpoint_filename,
    )
    validate_checkpoint_file(checkpoint_path)

    from .portable_m2d import PortableM2D

    m2d_model = PortableM2D(weight_file=str(checkpoint_path))
    model = ChunkedM2D(
        m2d_model,
        sample_rate=sample_rate,
        max_audio_seconds=DEFAULT_MAX_AUDIO_SECONDS,
    )
    model.output_dim = int(m2d_model.cfg.feature_d)
    model.checkpoint_cfg = {
        "archive_alias": checkpoint_archive_alias,
        "checkpoint_filename": checkpoint_filename,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "input_size": list(m2d_model.cfg.input_size),
        "patch_size": list(m2d_model.cfg.patch_size),
        "feature_dim": int(m2d_model.cfg.feature_d),
    }
    return model


__all__ = [
    "ChunkedM2D",
    "DEFAULT_CHECKPOINT_ARCHIVE_ALIAS",
    "DEFAULT_CHECKPOINT_FILENAME",
    "DEFAULT_MAX_AUDIO_SECONDS",
    "DEFAULT_SAMPLE_RATE",
    "build_m2d_embedding_model",
    "resolve_checkpoint_path",
    "validate_checkpoint_file",
]
