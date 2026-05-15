from __future__ import annotations

import os

from dcase2026_task1.models.base import GenerativeModel, ModelInput


class OpenAIModel(GenerativeModel):
    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.85,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAIModel requires openai>=1.0.0.") from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def generate_batch(self, model_inputs: list[ModelInput]) -> list[str]:
        return [self._generate_raw_response(model_input) for model_input in model_inputs]

    @staticmethod
    def _build_messages(model_input: ModelInput) -> list[dict[str, str]]:
        return [{"role": "user", "content": model_input.prompt}]

    def _generate_raw_response(self, model_input: ModelInput) -> str:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=self._build_messages(model_input),
            max_completion_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        message = response.choices[0].message.content
        if message is None:
            return ""
        return message.strip()
