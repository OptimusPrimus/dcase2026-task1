from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import torch

from dcase2026_task1.models.audio_wrappers import (
    ArbitraryLengthAudioWrapper,
)

DEFAULT_CHECKPOINT_ARCHIVE_ALIAS = "m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly.zip"
DEFAULT_CHECKPOINT_FILENAME = "weights_ep67it3124-0.48558.pth"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 10.0
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
) -> tuple[torch.Tensor, torch.Tensor]:
    del metadata
    embeddings = model.encode(waveforms)
    token_padding_mask = _build_m2d_token_padding_mask(
        model=model,
        padding_mask=padding_mask,
        batch_size=waveforms.shape[0],
        sequence_length=embeddings.shape[1],
        device=embeddings.device,
    )
    return embeddings, token_padding_mask


def _build_m2d_token_padding_mask(
    *,
    model: torch.nn.Module,
    padding_mask: torch.Tensor | None,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if padding_mask is None:
        return torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)

    token_padding_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=device)
    hop_size = int(model.cfg.hop_size)
    patch_frames = int(model.backbone.patch_size()[1])
    max_input_frames = int(model.cfg.input_size[1])

    for batch_index in range(batch_size):
        valid_length = int((~padding_mask[batch_index]).sum().item())
        if valid_length <= 0:
            valid_token_count = 1
        else:
            valid_mel_frames = min(max_input_frames, 1 + (valid_length // hop_size))
            valid_token_count = min(
                sequence_length,
                max(1, (valid_mel_frames + patch_frames - 1) // patch_frames),
            )
        token_padding_mask[batch_index, :valid_token_count] = False

    return token_padding_mask


def _concatenate_segment_outputs(
    segment_outputs: tuple[torch.Tensor, torch.Tensor],
    segment_batch_indices: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    segment_embeddings, segment_padding_masks = segment_outputs
    feature_dim = segment_embeddings.shape[-1]
    per_batch_embeddings: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    per_batch_masks: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

    for segment_index, batch_index in enumerate(segment_batch_indices.tolist()):
        per_batch_embeddings[batch_index].append(segment_embeddings[segment_index])
        per_batch_masks[batch_index].append(segment_padding_masks[segment_index])

    concatenated_embeddings: list[torch.Tensor] = []
    concatenated_masks: list[torch.Tensor] = []
    max_sequence_length = 1
    for batch_index in range(batch_size):
        if per_batch_embeddings[batch_index]:
            batch_embeddings = torch.cat(per_batch_embeddings[batch_index], dim=0)
            batch_mask = torch.cat(per_batch_masks[batch_index], dim=0)
        else:
            batch_embeddings = segment_embeddings.new_zeros((1, feature_dim))
            batch_mask = torch.ones((1,), dtype=torch.bool, device=segment_padding_masks.device)
        concatenated_embeddings.append(batch_embeddings)
        concatenated_masks.append(batch_mask)
        max_sequence_length = max(max_sequence_length, int(batch_embeddings.shape[0]))

    packed_embeddings = segment_embeddings.new_zeros((batch_size, max_sequence_length, feature_dim))
    packed_masks = torch.ones((batch_size, max_sequence_length), dtype=torch.bool, device=segment_padding_masks.device)
    for batch_index, (batch_embeddings, batch_mask) in enumerate(zip(concatenated_embeddings, concatenated_masks, strict=False)):
        sequence_length = batch_embeddings.shape[0]
        packed_embeddings[batch_index, :sequence_length] = batch_embeddings
        packed_masks[batch_index, :sequence_length] = batch_mask

    return packed_embeddings, packed_masks


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
            aggregate_outputs=_concatenate_segment_outputs,
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
