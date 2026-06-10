from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn


def split_waveforms_into_segments(
    waveforms: torch.Tensor,
    padding_mask: torch.Tensor | None,
    max_segment_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, waveform_length = waveforms.shape
    segment_waveforms: list[torch.Tensor] = []
    segment_padding_masks: list[torch.Tensor] = []
    segment_batch_indices: list[int] = []

    for batch_index in range(batch_size):
        if padding_mask is None:
            valid_length = waveform_length
        else:
            valid_length = int((~padding_mask[batch_index]).sum().item())
        valid_length = max(valid_length, 1)

        sample_waveform = waveforms[batch_index]
        for start in range(0, valid_length, max_segment_samples):
            stop = min(start + max_segment_samples, valid_length)
            segment_length = stop - start

            segment = torch.zeros(
                max_segment_samples,
                dtype=waveforms.dtype,
                device=waveforms.device,
            )
            segment[:segment_length] = sample_waveform[start:stop]
            segment_waveforms.append(segment)

            segment_mask = torch.ones(
                max_segment_samples,
                dtype=torch.bool,
                device=waveforms.device,
            )
            segment_mask[:segment_length] = False
            segment_padding_masks.append(segment_mask)
            segment_batch_indices.append(batch_index)

    return (
        torch.stack(segment_waveforms, dim=0),
        torch.stack(segment_padding_masks, dim=0),
        torch.tensor(segment_batch_indices, dtype=torch.long, device=waveforms.device),
    )


def mean_segment_outputs(
    segment_outputs: torch.Tensor,
    segment_batch_indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    outputs = torch.zeros(
        (batch_size, *segment_outputs.shape[1:]),
        dtype=segment_outputs.dtype,
        device=segment_outputs.device,
    )
    counts = torch.zeros(batch_size, dtype=segment_outputs.dtype, device=segment_outputs.device)

    outputs.index_add_(0, segment_batch_indices, segment_outputs)
    counts.index_add_(
        0,
        segment_batch_indices,
        torch.ones(segment_batch_indices.shape[0], dtype=segment_outputs.dtype, device=segment_outputs.device),
    )
    count_shape = (batch_size, *([1] * (segment_outputs.ndim - 1)))
    return outputs / counts.view(count_shape).clamp_min(1.0)


class ArbitraryLengthAudioWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        sample_rate: int,
        max_audio_seconds: float,
        segment_forward: Callable[..., Any],
        aggregate_outputs: Callable[[Any, torch.Tensor, int], Any],
    ) -> None:
        super().__init__()
        if max_audio_seconds <= 0:
            raise ValueError(f"max_audio_seconds must be positive, got {max_audio_seconds}.")
        self.model = model
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.segment_forward = segment_forward
        self.aggregate_outputs = aggregate_outputs

    def forward(
        self,
        waveforms: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        max_segment_samples = int(self.max_audio_seconds * self.sample_rate)
        segmented_waveforms, segmented_padding_mask, segment_batch_indices = split_waveforms_into_segments(
            waveforms,
            padding_mask,
            max_segment_samples=max_segment_samples,
        )
        segment_outputs = self.segment_forward(
            self.model,
            segmented_waveforms,
            segmented_padding_mask,
            *args,
            **kwargs,
        )
        return self.aggregate_outputs(
            segment_outputs,
            segment_batch_indices,
            batch_size=waveforms.shape[0],
        )
