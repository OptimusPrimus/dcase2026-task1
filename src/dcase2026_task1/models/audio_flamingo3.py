from __future__ import annotations

import re
from typing import Any

import torch

from dcase2026_task1.models.base import AudioLanguageModel, PredictionResult


class AudioFlamingo3Classifier(AudioLanguageModel):
    def __init__(
        self,
        model_id: str = "nvidia/audio-flamingo-3-hf",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 128,
    ) -> None:
        try:
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "AudioFlamingo3Classifier requires transformers>=4.57.0."
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._processor_cls = AutoProcessor
        self._model_cls = AudioFlamingo3ForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(model_id)
        resolved_dtype = self._resolve_dtype(torch_dtype)
        self._model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=resolved_dtype,
            low_cpu_mem_usage=True,
        )

    @staticmethod
    def _resolve_dtype(torch_dtype: str) -> torch.dtype | str:
        if torch_dtype == "auto":
            return "auto"
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(
                f"Unsupported torch_dtype={torch_dtype}. "
                f"Expected one of: {', '.join(dtype_map)} or auto."
            )
        return dtype_map[torch_dtype]

    def predict(
        self,
        item: dict[str, Any],
        candidate_classes: list[dict[str, Any]],
    ) -> PredictionResult:
        prompt = self._build_prompt(item, candidate_classes)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": item["audio_path"]},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = outputs[:, inputs.input_ids.shape[1] :]
        raw_response = self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
        predicted_class_idx, parsed_label = self._parse_prediction(
            raw_response,
            candidate_classes,
        )
        predicted_class_name = None
        if predicted_class_idx is not None:
            for candidate in candidate_classes:
                if candidate["class_idx"] == predicted_class_idx:
                    predicted_class_name = candidate["class_name"]
                    break
        return PredictionResult(
            predicted_class_idx=predicted_class_idx,
            predicted_class_name=predicted_class_name,
            raw_response=raw_response,
            parsed_label=parsed_label,
        )

    def _build_prompt(
        self,
        item: dict[str, Any],
        candidate_classes: list[dict[str, Any]],
    ) -> str:
        class_lines = []
        for option_index, candidate in enumerate(candidate_classes, start=1):
            description_text = candidate["description"] or ""
            class_lines.append(
                f"{option_index}. "
                f'{candidate["description_top_level"]} -> '
                f'{candidate["description_second_level"]}: '
                f"{description_text}"
            )

        metadata_description = (item.get("description") or "")
        title = item.get("title") or ""
        tags = item.get("tags") or ""

        return (
            "You are classifying an audio clip into one of the known dataset classes.\n"
            "Use the audio together with the provided metadata.\n"
            "Choose exactly one option from the list below.\n"
            "Output only the option index.\n"
            "Choose only from these classes:\n"
            f"{chr(10).join(class_lines)}\n\n"
            "Clip metadata:\n"
            f'- title="{title}"\n'
            f'- tags="{tags}"\n'
            f'- description="{metadata_description}"\n'
        )

    @staticmethod
    def _parse_prediction(
        raw_response: str,
        candidate_classes: list[dict[str, Any]],
    ) -> tuple[int | None, str | None]:
        option_match = re.search(r"\b(\d+)\b", raw_response)
        if option_match:
            option_index = int(option_match.group(1))
            if 1 <= option_index <= len(candidate_classes):
                candidate = candidate_classes[option_index - 1]
                return int(candidate["class_idx"]), str(option_index)

        return None, raw_response.strip() or None
