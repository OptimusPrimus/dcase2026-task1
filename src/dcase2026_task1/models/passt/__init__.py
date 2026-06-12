from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchaudio.compliance.kaldi as ta_kaldi

from dcase2026_task1.models.audio_wrappers import (
    ArbitraryLengthAudioWrapper,
    pack_segment_outputs,
)

from .passt import checkpoint_filter_fn, passt_s_swa_p16_128_ap476, passt_s_kd_p16_128_ap486, passt_s_swa_p16_s16_128_ap473

DEFAULT_CHECKPOINT_ALIAS = "passt_s_swa_p16_s16_128_ap473"
DEFAULT_SAMPLE_RATE = 32000
DEFAULT_NUM_CLASSES = 527
DEFAULT_NUM_MEL_BINS = 128
DEFAULT_FRAME_LENGTH_MS = 25.0
DEFAULT_FRAME_SHIFT_MS = 10.0
DEFAULT_INPUT_TDIM = 998


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
            f"{path} does not look like a PyTorch checkpoint. It looks like an HTML/XML response instead."
        )
    if b"<title>" in lowered[:256] or b"github" in lowered[:256]:
        raise ValueError(
            f"{path} appears to contain a web page instead of checkpoint bytes. "
            f"Delete {path} and re-run with --checkpoint-dir pointing at a valid .pt file."
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
            "Official PaSST checkpoints require torch.load(..., weights_only=False). "
            "Re-run with --trust-checkpoint only if the checkpoint source is trusted."
        )
    return torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )


def _extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state_dict: dict[str, Any]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = dict(checkpoint["state_dict"])
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = dict(checkpoint["model"])
    else:
        state_dict = dict(checkpoint)

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    if any(key.startswith("net.") for key in state_dict):
        state_dict = {
            key.removeprefix("net."): value
            for key, value in state_dict.items()
            if key.startswith("net.")
        }

    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("mel.")
    }


def _preprocess_waveforms(
    waveforms: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    fbanks: list[torch.Tensor] = []
    lengths: list[int] = []

    for batch_index in range(waveforms.shape[0]):
        if padding_mask is None:
            valid_length = waveforms.shape[1]
        else:
            valid_length = int((~padding_mask[batch_index]).sum().item())
        valid_length = max(valid_length, 800)

        waveform = waveforms[batch_index, :valid_length].detach().cpu().unsqueeze(0)
        fbank = ta_kaldi.fbank(
            waveform,
            num_mel_bins=DEFAULT_NUM_MEL_BINS,
            sample_frequency=DEFAULT_SAMPLE_RATE,
            frame_length=DEFAULT_FRAME_LENGTH_MS,
            frame_shift=DEFAULT_FRAME_SHIFT_MS,
        )
        fbanks.append(fbank)
        lengths.append(int(fbank.shape[0]))

    max_frames = 998 #max(lengths, default=1)
    batched_fbanks = waveforms.new_zeros((waveforms.shape[0], max_frames, DEFAULT_NUM_MEL_BINS))
    for batch_index, fbank in enumerate(fbanks):
        batched_fbanks[batch_index, : fbank.shape[0]] = fbank.to(
            device=waveforms.device,
            dtype=waveforms.dtype,
        )
    return batched_fbanks


def _extract_passt_embeddings(
    model: torch.nn.Module,
    waveforms: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    metadata: list[dict[str, object]] | None = None,
) -> torch.Tensor:
    del metadata
    fbanks = _preprocess_waveforms(waveforms, padding_mask)
    passt_inputs = fbanks.transpose(1, 2).unsqueeze(1)
    _logits, features = model(passt_inputs)
    return features


class ChunkedPaSST(ArbitraryLengthAudioWrapper):
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
            segment_forward=_extract_passt_embeddings,
            aggregate_outputs=pack_segment_outputs,
        )


def build_passt_embedding_model(
    *,
    checkpoint_dir: str | Path,
    checkpoint_alias: str = DEFAULT_CHECKPOINT_ALIAS,
    trust_checkpoint: bool,
    sample_rate: int,
) -> ChunkedPaSST:
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(
            f"PaSST expects sample_rate={DEFAULT_SAMPLE_RATE}, got {sample_rate}."
        )

#    checkpoint = load_embedding_checkpoint(
#        checkpoint_dir=checkpoint_dir,
#        checkpoint_alias=checkpoint_alias,
#        trust_checkpoint=trust_checkpoint,
#    )
    passt_model = passt_s_swa_p16_s16_128_ap473(
        pretrained=True,
        num_classes=DEFAULT_NUM_CLASSES,
        in_chans=1,
        img_size=(DEFAULT_NUM_MEL_BINS, DEFAULT_INPUT_TDIM),
        stride=(16, 16),
        u_patchout=0,
        s_patchout_t=15,
        s_patchout_f=2,
    )
    #state_dict = checkpoint_filter_fn(_extract_state_dict(checkpoint), passt_model)
    #missing_keys, unexpected_keys = passt_model.load_state_dict(state_dict, strict=False)
    #if unexpected_keys:
    #    raise RuntimeError(f"Unexpected PaSST checkpoint keys: {unexpected_keys}")
    #if missing_keys:
    #    raise RuntimeError(f"Missing PaSST checkpoint keys: {missing_keys}")

    model = ChunkedPaSST(
        passt_model,
        sample_rate=sample_rate,
        max_audio_seconds=10,
    )
    model.output_dim = int(passt_model.embed_dim)
    model.checkpoint_cfg = {
        "arch": DEFAULT_CHECKPOINT_ALIAS,
        "num_classes": DEFAULT_NUM_CLASSES,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "num_mel_bins": DEFAULT_NUM_MEL_BINS,
        "frame_length_ms": DEFAULT_FRAME_LENGTH_MS,
        "frame_shift_ms": DEFAULT_FRAME_SHIFT_MS,
        "input_tdim": DEFAULT_INPUT_TDIM,
    }
    return model


__all__ = [
    "ChunkedPaSST",
    "DEFAULT_CHECKPOINT_ALIAS",
    "DEFAULT_FRAME_LENGTH_MS",
    "DEFAULT_FRAME_SHIFT_MS",
    "DEFAULT_INPUT_TDIM",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_NUM_MEL_BINS",
    "DEFAULT_SAMPLE_RATE",
    "build_passt_embedding_model",
    "checkpoint_filter_fn",
    "load_embedding_checkpoint",
    "passt_s_swa_p16_128_ap476",
    "passt_s_kd_p16_128_ap486",
    "passt_s_swa_p16_s16_128_ap473",
    "resolve_checkpoint_path",
    "validate_checkpoint_file",
]
