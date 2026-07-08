from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

DEFAULT_SAMPLE_RATE = 32000
DEFAULT_MAX_AUDIO_SECONDS = 10.0
DEFAULT_CHECKPOINT_ALIAS = "clap"
DEFAULT_ROBERTA_BASE = False
DEFAULT_S_PATCHOUT_T = 15
DEFAULT_S_PATCHOUT_F = 2
DEFAULT_INITIAL_TAU = 0.07
SUMMARY_METADATA_KEY = "metadata_summary"
KEYWORD_METADATA_KEY = "tags"
TITLE_METADATA_KEY = "title"


def _metadata_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(
            part.strip()
            for part in value.replace(";", ",").split(",")
            if part.strip()
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _metadata_item_to_text(
    metadata_item: dict[str, Any] | None,
    metadata_text_key: str,
) -> str:
    if metadata_item is None:
        return ""
    return _metadata_value_to_text(metadata_item.get(metadata_text_key))


def metadata_to_summary_texts(
    metadata: list[dict[str, Any]] | None,
    batch_size: int,
) -> list[str]:
    return metadata_to_texts(
        metadata,
        batch_size=batch_size,
        metadata_text_key=SUMMARY_METADATA_KEY,
    )


def metadata_to_keyword_texts(
    metadata: list[dict[str, Any]] | None,
    batch_size: int,
) -> list[str]:
    return metadata_to_texts(
        metadata,
        batch_size=batch_size,
        metadata_text_key=KEYWORD_METADATA_KEY,
    )


def metadata_to_texts(
    metadata: list[dict[str, Any]] | None,
    *,
    batch_size: int,
    metadata_text_key: str,
) -> list[str]:
    if metadata is None:
        return [""] * batch_size
    if len(metadata) != batch_size:
        raise ValueError(
            f"Expected metadata for {batch_size} items, got {len(metadata)}."
        )
    return [_metadata_item_to_text(item, metadata_text_key) for item in metadata]


class CLAPEmbeddingModel(torch.nn.Module):
    def __init__(
        self,
        retrieval_model: torch.nn.Module,
        *,
        sample_rate: int,
        max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
        metadata_text_key: str = SUMMARY_METADATA_KEY,
    ) -> None:
        super().__init__()
        self.retrieval_model = retrieval_model
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.metadata_text_key = metadata_text_key
        self.output_dim = int(retrieval_model.audio_projection.out_features) + int(
            retrieval_model.text_projection.out_features
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata_texts = metadata_to_texts(
            metadata,
            batch_size=waveforms.shape[0],
            metadata_text_key=self.metadata_text_key,
        )
        durations = self._durations_from_padding_mask(waveforms, padding_mask)
        batch = {
            "audio": waveforms.unsqueeze(1),
            "duration": durations,
            "captions": [[text] for text in metadata_texts],
        }
        audio_embeddings = self.retrieval_model.forward_audio(batch)
        text_embeddings = self.retrieval_model.forward_text(batch)
        embeddings = torch.cat(
            [audio_embeddings, text_embeddings],
            dim=-1,
        ).unsqueeze(1)
        embedding_padding_mask = torch.zeros(
            (waveforms.shape[0], 1),
            dtype=torch.bool,
            device=waveforms.device,
        )
        return embeddings, embedding_padding_mask

    def _durations_from_padding_mask(
        self,
        waveforms: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if padding_mask is None:
            lengths = torch.full(
                (waveforms.shape[0],),
                waveforms.shape[1],
                dtype=torch.float32,
                device=waveforms.device,
            )
        else:
            lengths = (~padding_mask).sum(dim=1).to(dtype=torch.float32)
        return lengths / float(self.sample_rate)


def resolve_checkpoint_path(
    checkpoint_dir: str | Path,
    checkpoint_alias: str = DEFAULT_CHECKPOINT_ALIAS,
) -> Path:
    resolved_checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    for suffix in (".ckpt", ".pt", ".pth"):
        candidate = resolved_checkpoint_dir / f"{checkpoint_alias}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"CLAP checkpoint {checkpoint_alias!r} not found under {resolved_checkpoint_dir}."
    )


def _build_audio_retrieval_model() -> torch.nn.Module:
    from .retrieval_module import AudioRetrievalModel

    return AudioRetrievalModel(
        s_patchout_t=DEFAULT_S_PATCHOUT_T,
        s_patchout_f=DEFAULT_S_PATCHOUT_F,
        roberta_base=DEFAULT_ROBERTA_BASE,
        initial_tau=DEFAULT_INITIAL_TAU,
        tau_trainable=False,
        compile=False,
    )


def _normalize_compiled_state_dict_keys(state_dict: Any) -> Any:
    if not isinstance(state_dict, dict):
        return state_dict

    return {
        ".".join(part for part in key.split(".") if part != "_orig_mod")
        if isinstance(key, str)
        else key: value
        for key, value in state_dict.items()
    }


def build_clap_embedding_model(
    *,
    checkpoint_dir: str | Path,
    trust_checkpoint: bool,
    sample_rate: int,
    checkpoint_alias: str = DEFAULT_CHECKPOINT_ALIAS,
    load_checkpoint: bool = True,
    metadata_text_key: str = SUMMARY_METADATA_KEY,
    arch: str = "clap",
) -> CLAPEmbeddingModel:
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(
            f"CLAP expects sample_rate={DEFAULT_SAMPLE_RATE}, got {sample_rate}."
        )

    retrieval_model = _build_audio_retrieval_model()
    if load_checkpoint:
        if not trust_checkpoint:
            raise ValueError(
                "CLAP checkpoints require torch.load(..., weights_only=False). "
                "Re-run with --trust-checkpoint only if the checkpoint source is trusted."
            )
        checkpoint_path = resolve_checkpoint_path(
            checkpoint_dir=checkpoint_dir,
            checkpoint_alias=checkpoint_alias,
        )
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        state_dict = (
            checkpoint.get("state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        state_dict = _normalize_compiled_state_dict_keys(state_dict)
        retrieval_model.load_state_dict(state_dict, strict=False)

    model = CLAPEmbeddingModel(
        retrieval_model,
        sample_rate=sample_rate,
        metadata_text_key=metadata_text_key,
    )
    model.checkpoint_cfg = {
        "arch": arch,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "audio_projection_dim": int(retrieval_model.audio_projection.out_features),
        "text_projection_dim": int(retrieval_model.text_projection.out_features),
        "checkpoint_alias": checkpoint_alias,
        "checkpoint_loaded": load_checkpoint,
        "metadata_text_key": metadata_text_key,
    }
    return model


__all__ = [
    "CLAPEmbeddingModel",
    "DEFAULT_CHECKPOINT_ALIAS",
    "DEFAULT_MAX_AUDIO_SECONDS",
    "DEFAULT_SAMPLE_RATE",
    "KEYWORD_METADATA_KEY",
    "SUMMARY_METADATA_KEY",
    "build_clap_embedding_model",
    "metadata_to_keyword_texts",
    "metadata_to_summary_texts",
    "metadata_to_texts",
    "resolve_checkpoint_path",
]
