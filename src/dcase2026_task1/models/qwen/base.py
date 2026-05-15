from __future__ import annotations

import json
import os
from urllib import error, request

from dcase2026_task1.models.base import GenerativeModel, ModelInput


class QwenModel(GenerativeModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.6-27B",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 1024,
        api_base: str | None = None,
        api_key: str | None = None,
        tensor_parallel_size: int = 1,
        disable_custom_all_reduce: bool = False,
        enforce_eager: bool = False,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.api_base = (
            api_base or os.environ.get("VLLM_API_BASE") or "http://127.0.0.1:8000/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self._resolve_dtype(torch_dtype)
        self._ignored_runtime_options = {
            "device": device,
            "tensor_parallel_size": tensor_parallel_size,
            "disable_custom_all_reduce": disable_custom_all_reduce,
            "enforce_eager": enforce_eager,
        }

    @staticmethod
    def _resolve_dtype(torch_dtype: str) -> str:
        supported = {"auto", "float16", "bfloat16", "float32"}
        if torch_dtype not in supported:
            raise ValueError(
                f"Unsupported torch_dtype={torch_dtype}. "
                f"Expected one of: {', '.join(sorted(supported))}."
            )
        return torch_dtype

    def generate_batch(self, model_inputs: list[ModelInput]) -> list[str]:
        return [self._generate_raw_response(model_input) for model_input in model_inputs]

    @staticmethod
    def _build_messages(model_input: ModelInput) -> list[dict[str, str]]:
        return [{"role": "user", "content": model_input.prompt}]

    def _generate_raw_response(self, model_input: ModelInput) -> str:
        payload = {
            "model": self.model_id,
            "messages": self._build_messages(model_input),
            "max_tokens": self.max_new_tokens,
            "temperature": 0.2,
            "top_p": 0.85,
            "repetition_penalty": 1.1,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = request.Request(
            url=f"{self.api_base}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(http_request) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"QwenModel API request failed with HTTP {exc.code}: {details}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"QwenModel could not reach vLLM API at {self.api_base}: {exc.reason}"
            ) from exc

        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise RuntimeError(
                f"QwenModel received an unexpected API response: {payload!r}"
            ) from exc
