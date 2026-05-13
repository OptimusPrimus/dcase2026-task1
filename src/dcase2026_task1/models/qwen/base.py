from __future__ import annotations

from collections.abc import Mapping

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
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError("QwenModel requires vllm>=0.19.0.") from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._llm = LLM(
            model=model_id,
            dtype=self._resolve_dtype(torch_dtype),
        )
        self._sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
        )

    @staticmethod
    def _resolve_dtype(torch_dtype: str) -> str:
        supported = {"auto", "float16", "bfloat16", "float32"}
        if torch_dtype not in supported:
            raise ValueError(
                f"Unsupported torch_dtype={torch_dtype}. "
                f"Expected one of: {', '.join(sorted(supported))}."
            )
        return torch_dtype

    def predict_batch(
        self,
        items: list[Mapping[str, object]],
        skill: ModelSkill,
    ) -> list[object]:
        model_inputs = skill.build_inputs(items)
        raw_responses = self._generate_raw_responses(model_inputs)
        return skill.parse_outputs(raw_responses, items)

    @staticmethod
    def _build_messages(model_input: ModelInput) -> list[dict[str, str]]:
        return [{"role": "user", "content": model_input.prompt}]

    def _generate_raw_responses(self, model_inputs: list[ModelInput]) -> list[str]:
        if not hasattr(self, "_llm"):
            return [self._generate_raw_response(model_input) for model_input in model_inputs]

        conversations = [self._build_messages(model_input) for model_input in model_inputs]
        outputs = self._llm.chat(
            messages=conversations,
            sampling_params=self._sampling_params,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": True},
        )
        return [output.outputs[0].text.strip() for output in outputs]

    def _generate_raw_response(self, model_input: ModelInput) -> str:
        return self._generate_raw_responses([model_input])[0]
