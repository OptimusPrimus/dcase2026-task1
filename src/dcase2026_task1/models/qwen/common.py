from __future__ import annotations

import re


def split_reasoning(text: str) -> tuple[str | None, str]:
    if "</think>" in text and "<think>" not in text:
        text = f"<think>{text}"
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else None
    answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return reasoning, answer
