from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch

from dcase2026_task1.models.audio_wrappers import (
    mean_segment_outputs,
    split_waveforms_into_segments,
)
from dcase2026_task1.models.clap import metadata_to_summary_texts

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_MAX_AUDIO_SECONDS = 10.0
DEFAULT_CHECKPOINT_ALIAS = "lclap"
DEFAULT_AUDIO_EMBEDDING_DIM = 512
DEFAULT_TEXT_EMBEDDING_DIM = 512
DEFAULT_ENABLE_FUSION = False
DEFAULT_QUANTIZE_AUDIO = True
SPECTROGRAM_MODULE_NAMES = ("spectrogram_extractor", "logmel_extractor")
AUDIO_PREPROCESSING_METHOD_NAMES = ("reshape_wav2img",)


def int16_quantize_audio(waveforms: torch.Tensor) -> torch.Tensor:
    return (
        waveforms.clamp(min=-1.0, max=1.0)
        .mul(32767.0)
        .to(dtype=torch.int16)
        .to(dtype=torch.float32)
        .div(32767.0)
    )


def resolve_checkpoint_path(
    checkpoint_dir: str | Path,
    checkpoint_alias: str = DEFAULT_CHECKPOINT_ALIAS,
) -> Path | None:
    resolved_checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    for suffix in (".ckpt", ".pt", ".pth"):
        candidate = resolved_checkpoint_dir / f"{checkpoint_alias}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _build_laion_clap_module(
    *,
    enable_fusion: bool = DEFAULT_ENABLE_FUSION,
    amodel: str | None = None,
) -> Any:
    import laion_clap

    kwargs: dict[str, Any] = {"enable_fusion": enable_fusion}
    if amodel is not None:
        kwargs["amodel"] = amodel
    return laion_clap.CLAP_Module(**kwargs)


class Float32AutocastDisabledModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module.float()
        for parameter in self.module.parameters():
            parameter.requires_grad = False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        device = _first_tensor_device([*args, *kwargs.values()])
        with autocast_disabled(device):
            self.module.float()
            args = tuple(_floating_tensors_to_float32(arg) for arg in args)
            kwargs = {
                key: _floating_tensors_to_float32(value)
                for key, value in kwargs.items()
            }
            with torch.no_grad():
                return self.module(*args, **kwargs)


def _wrap_float32_no_grad_method(module: torch.nn.Module, name: str) -> None:
    method = getattr(module, name, None)
    if method is None or getattr(method, "_lclap_float32_no_grad", False):
        return

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        device = _first_tensor_device([*args, *kwargs.values()])
        with autocast_disabled(device):
            args = tuple(_floating_tensors_to_float32(arg) for arg in args)
            kwargs = {
                key: _floating_tensors_to_float32(value)
                for key, value in kwargs.items()
            }
            with torch.no_grad():
                return method(*args, **kwargs)

    wrapped._lclap_float32_no_grad = True  # type: ignore[attr-defined]
    setattr(module, name, wrapped)


class LAIONCLAPEmbeddingModel(torch.nn.Module):
    def __init__(
        self,
        clap_module: Any,
        *,
        sample_rate: int,
        max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
        audio_embedding_dim: int = DEFAULT_AUDIO_EMBEDDING_DIM,
        text_embedding_dim: int = DEFAULT_TEXT_EMBEDDING_DIM,
        quantize_audio: bool = DEFAULT_QUANTIZE_AUDIO,
    ) -> None:
        super().__init__()
        self.clap_module = clap_module
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.audio_embedding_dim = audio_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.quantize_audio = quantize_audio
        self.output_dim = audio_embedding_dim + text_embedding_dim
        wrap_spectrogram_modules_float32(self.clap_module)

    def forward(
        self,
        waveforms: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summary_texts = metadata_to_summary_texts(
            metadata,
            batch_size=waveforms.shape[0],
        )
        max_segment_samples = int(self.max_audio_seconds * self.sample_rate)
        segment_waveforms, segment_padding_mask, segment_batch_indices = split_waveforms_into_segments(
            waveforms,
            padding_mask,
            max_segment_samples=max_segment_samples,
        )
        del segment_padding_mask

        audio_embeddings = self._audio_embeddings_from_segments(segment_waveforms)
        audio_embeddings = mean_segment_outputs(
            audio_embeddings,
            segment_batch_indices,
            batch_size=waveforms.shape[0],
        )
        text_embeddings = self._as_tensor(
            self.clap_module.get_text_embedding(summary_texts, use_tensor=True),
            device=waveforms.device,
        )

        embeddings = torch.cat([audio_embeddings, text_embeddings], dim=-1).unsqueeze(1)
        embedding_padding_mask = torch.zeros(
            (waveforms.shape[0], 1),
            dtype=torch.bool,
            device=waveforms.device,
        )
        return embeddings, embedding_padding_mask

    def _audio_embeddings_from_segments(self, segment_waveforms: torch.Tensor) -> torch.Tensor:
        segment_waveforms = segment_waveforms.to(dtype=torch.float32)
        if self.quantize_audio:
            segment_waveforms = int16_quantize_audio(segment_waveforms)
        audio_embeddings = self.clap_module.get_audio_embedding_from_data(
            x=segment_waveforms,
            use_tensor=True,
        )
        return self._as_tensor(audio_embeddings, device=segment_waveforms.device)

    @staticmethod
    def _as_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=device)


def build_lclap_embedding_model(
    *,
    checkpoint_dir: str | Path,
    trust_checkpoint: bool,
    sample_rate: int,
    checkpoint_alias: str = DEFAULT_CHECKPOINT_ALIAS,
    load_checkpoint: bool = True,
    enable_fusion: bool = DEFAULT_ENABLE_FUSION,
    amodel: str | None = None,
) -> LAIONCLAPEmbeddingModel:
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(
            f"LAION-CLAP expects sample_rate={DEFAULT_SAMPLE_RATE}, got {sample_rate}."
        )

    clap_module = _build_laion_clap_module(enable_fusion=enable_fusion, amodel=amodel)
    checkpoint_path = None
    if load_checkpoint:
        if not trust_checkpoint:
            raise ValueError(
                "LAION-CLAP checkpoints require checkpoint deserialization. "
                "Re-run with --trust-checkpoint only if the checkpoint source is trusted."
            )
        checkpoint_path = resolve_checkpoint_path(
            checkpoint_dir=checkpoint_dir,
            checkpoint_alias=checkpoint_alias,
        )
        if checkpoint_path is None:
            clap_module.load_ckpt()
        else:
            clap_module.load_ckpt(str(checkpoint_path))

    model = LAIONCLAPEmbeddingModel(clap_module, sample_rate=sample_rate)
    model.checkpoint_cfg = {
        "arch": "lclap",
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "audio_projection_dim": DEFAULT_AUDIO_EMBEDDING_DIM,
        "text_projection_dim": DEFAULT_TEXT_EMBEDDING_DIM,
        "checkpoint_alias": checkpoint_alias,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_loaded": load_checkpoint,
        "enable_fusion": enable_fusion,
        "amodel": amodel,
    }
    return model


def autocast_disabled(device: torch.device | None) -> Any:
    if device is not None and device.type in {"cuda", "cpu", "xpu", "mps"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def _first_tensor_device(values: Iterable[Any]) -> torch.device | None:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, dict):
            device = _first_tensor_device(value.values())
            if device is not None:
                return device
        if isinstance(value, (list, tuple)):
            device = _first_tensor_device(value)
            if device is not None:
                return device
    return None


def _floating_tensors_to_float32(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.to(dtype=torch.float32)
    if isinstance(value, dict):
        return {
            key: _floating_tensors_to_float32(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_floating_tensors_to_float32(item) for item in value)
    if isinstance(value, list):
        return [_floating_tensors_to_float32(item) for item in value]
    return value


def wrap_spectrogram_modules_float32(module: Any) -> None:
    if not isinstance(module, torch.nn.Module):
        return
    for child in module.modules():
        for name in AUDIO_PREPROCESSING_METHOD_NAMES:
            _wrap_float32_no_grad_method(child, name)
        for name in SPECTROGRAM_MODULE_NAMES:
            maybe_spectrogram_module = getattr(child, name, None)
            if (
                isinstance(maybe_spectrogram_module, torch.nn.Module)
                and not isinstance(maybe_spectrogram_module, Float32AutocastDisabledModule)
            ):
                setattr(
                    child,
                    name,
                    Float32AutocastDisabledModule(maybe_spectrogram_module),
                )


__all__ = [
    "DEFAULT_CHECKPOINT_ALIAS",
    "DEFAULT_MAX_AUDIO_SECONDS",
    "DEFAULT_SAMPLE_RATE",
    "Float32AutocastDisabledModule",
    "LAIONCLAPEmbeddingModel",
    "build_lclap_embedding_model",
    "int16_quantize_audio",
    "resolve_checkpoint_path",
    "wrap_spectrogram_modules_float32",
]
