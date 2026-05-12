from __future__ import annotations

import re
from typing import Any

import torch

from dcase2026_task1.models.base import AudioLanguageModel, PredictionResult


class QwenTextClassifier(AudioLanguageModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-9B",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 32,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "QwenTextClassifier requires transformers>=4.57.0."
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        resolved_dtype = self._resolve_dtype(torch_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=resolved_dtype,
            low_cpu_mem_usage=True,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

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
        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=True
        ).to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=1.0,  # good for reasoning mode
            top_p=0.95,
            top_k=20,
            do_sample=True,
        )
        generated = outputs[:, inputs.input_ids.shape[1] :]
        raw_response = self._tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
        reasoning, final_response = self._split_reasoning(raw_response)
        predicted_class_idx, parsed_label = self._parse_prediction(
            final_response,
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
            final_response=final_response,
            reasoning=reasoning,
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
            "You are classifying an audio event using metadata only.\n"
            "Do not rely on any audio input.\n"
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
    def _split_reasoning(text: str) -> tuple[str | None, str]:
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else None
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return reasoning, answer

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
