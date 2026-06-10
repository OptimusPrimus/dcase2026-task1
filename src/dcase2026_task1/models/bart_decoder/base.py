from __future__ import annotations

from typing import Any

import torch
from torch import nn


class BartMetadataDecoder(nn.Module):
    def __init__(
        self,
        audio_embedding_dim: int,
        model_id: str = "facebook/bart-base",
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoTokenizer, BartModel
        except ImportError as exc:
            raise ImportError(
                "BartMetadataDecoder requires transformers>=4.57.0."
            ) from exc

        self.model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = BartModel.from_pretrained(model_id)
        self.output_dim = int(self._model.config.d_model)
        self.audio_projection = nn.Linear(audio_embedding_dim, self.output_dim)

        placeholder_token_id = self._tokenizer.eos_token_id
        if placeholder_token_id is None:
            raise ValueError(f"{model_id} tokenizer does not define eos_token_id.")
        self.register_buffer(
            "_decoder_placeholder_token_id",
            torch.tensor([[placeholder_token_id]], dtype=torch.long),
            persistent=False,
        )

    def _build_tag_texts(
        self,
        metadata: list[dict[str, Any]] | None,
        batch_size: int,
    ) -> list[str]:
        if metadata is None:
            return [""] * batch_size
        tag_texts: list[str] = []
        for item in metadata:
            tags = item.get("tags", "") if item is not None else ""
            tag_texts.append(str(" ".join(tags.split(","))))
        if len(tag_texts) != batch_size:
            raise ValueError(
                f"Expected metadata for {batch_size} items, got {len(tag_texts)}."
            )
        return tag_texts

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        metadata: list[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        batch_size = audio_embeddings.shape[0]
        tag_texts = self._build_tag_texts(metadata, batch_size)
        tokenized = self._tokenizer(
            tag_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        model_device = audio_embeddings.device
        input_ids = tokenized["input_ids"].to(model_device)
        attention_mask = tokenized["attention_mask"].to(model_device)
        projected_audio = self.audio_projection(audio_embeddings)
        audio_attention_mask = torch.ones(
            projected_audio.shape[:2],
            dtype=attention_mask.dtype,
            device=model_device,
        )
        placeholder_token_ids = self._decoder_placeholder_token_id.expand(batch_size, -1).to(model_device)
        decoder_input_ids = torch.cat([input_ids, placeholder_token_ids], dim=1)
        decoder_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=model_device),
            ],
            dim=1,
        )
        decoder_outputs = self._model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=projected_audio,
            encoder_attention_mask=audio_attention_mask,
            return_dict=True,
        )
        return decoder_outputs.last_hidden_state[:, -1, :]
