from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dcase2026_task1.models.base import AudioLanguageModel, ModelInput, ModelSkill


class QwenModel(AudioLanguageModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.6-27B",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 1024,
    ) -> None:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "QwenModel requires transformers>=4.57.0."
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        resolved_dtype = self._resolve_dtype(torch_dtype)
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
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
            raise ImportError("QwenModel requires torch to resolve torch_dtype.") from exc
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

    def predict_batch(
        self,
        items: list[Mapping[str, object]],
        skill: ModelSkill,
    ) -> list[object]:
        model_inputs = skill.build_inputs(items)
        raw_responses = [
            self._generate_raw_response(model_input)
            for model_input in model_inputs
        ]
        return skill.parse_outputs(raw_responses, items)

    def _generate_raw_response(self, model_input: ModelInput) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": model_input.prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=True,
        ).to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            do_sample=True,
        )
        generated = outputs[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
