from __future__ import annotations

import torch

from dcase2026_task1.models.audio_wrappers import (
    ArbitraryLengthAudioWrapper,
    mean_segment_outputs,
)

from .BEATs import BEATs


def _extract_beats_sequence_embeddings(
    model: BEATs,
    waveforms: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    metadata: list[dict[str, object]] | None = None,
) -> torch.Tensor:
    del metadata
    features, feature_padding_mask = model.extract_features(
        waveforms,
        padding_mask=padding_mask,
    )
    if feature_padding_mask is not None:
        valid = (~feature_padding_mask).unsqueeze(-1)
        pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return pooled.unsqueeze(1)
    return features.mean(dim=1, keepdim=True)


class ChunkedBEATs(ArbitraryLengthAudioWrapper):
    def __init__(
        self,
        model: BEATs,
        *,
        sample_rate: int,
        max_audio_seconds: float,
    ) -> None:
        super().__init__(
            model,
            sample_rate=sample_rate,
            max_audio_seconds=max_audio_seconds,
            segment_forward=_extract_beats_sequence_embeddings,
            aggregate_outputs=mean_segment_outputs,
        )
