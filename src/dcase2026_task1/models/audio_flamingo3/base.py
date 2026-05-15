from __future__ import annotations

from typing import Any

from dcase2026_task1.models.base import GenerativeModel, ModelInput


class AudioFlamingo3Model(GenerativeModel):
    def __init__(
        self,
        model_id: str = "nvidia/audio-flamingo-3-hf",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 1024,
    ) -> None:
        try:
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "AudioFlamingo3Model requires transformers>=4.57.0."
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._processor = AutoProcessor.from_pretrained(model_id)
        resolved_dtype = self._resolve_dtype(torch_dtype)
        self._model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=resolved_dtype,
            low_cpu_mem_usage=True,
        )

    @staticmethod
    def _resolve_dtype(torch_dtype: str) -> Any:
        if torch_dtype == "auto":
            return "auto"
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "AudioFlamingo3Model requires torch to resolve torch_dtype."
            ) from exc
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

    def generate_batch(self, model_inputs: list[ModelInput]) -> list[str]:
        return [self._generate_raw_response(model_input) for model_input in model_inputs]

    def _generate_raw_response(self, model_input: ModelInput) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": model_input.prompt},
                    {"type": "audio", "path": model_input.audio_path},
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
        return self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
