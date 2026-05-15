from __future__ import annotations

import os

from dcase2026_task1.models.base import GenerativeModel, ModelInput, ModelOutput


class OpenAIModel(GenerativeModel):
    def __init__(
        self,
        model_id: str = "gpt-5.4-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.85,
        enable_reasoning: bool = False,
        reasoning_effort: str = "medium",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAIModel requires openai>=1.0.0.") from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_reasoning = enable_reasoning
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def generate_batch(self, model_inputs: list[ModelInput]) -> list[str]:
        return [output.text for output in self.generate_batch_outputs(model_inputs)]

    def generate_batch_outputs(self, model_inputs: list[ModelInput]) -> list[ModelOutput]:
        return [self._generate_output(model_input) for model_input in model_inputs]

    @staticmethod
    def _build_input(model_input: ModelInput) -> str:
        return model_input.prompt

    @staticmethod
    def _extract_text(response: object) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text.strip()
        return ""

    @staticmethod
    def _extract_reasoning_summary(response: object) -> str | None:
        output_items = getattr(response, "output", None)
        if not isinstance(output_items, list):
            return None

        summary_texts: list[str] = []
        for item in output_items:
            if getattr(item, "type", None) != "reasoning":
                continue
            summary = getattr(item, "summary", None)
            if not isinstance(summary, list):
                continue
            for part in summary:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    summary_texts.append(text.strip())

        if not summary_texts:
            return None
        return "\n\n".join(summary_texts)

    def _generate_output(self, model_input: ModelInput) -> ModelOutput:
        request: dict[str, object] = {
            "model": self.model_id,
            "input": self._build_input(model_input),
            "max_output_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.enable_reasoning:
            request["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": "auto",
            }

        response = self._client.responses.create(**request)
        return ModelOutput(
            text=self._extract_text(response),
            reasoning_summary=self._extract_reasoning_summary(response),
        )
