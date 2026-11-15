"""Minimal LLM client for any OpenAI-compatible endpoint (Gemini, Groq, OpenAI, ...)."""
from __future__ import annotations

import json
import re
import time

from . import config


class LLMError(RuntimeError):
    pass


def get_client():
    api_key, base_url, _ = config.llm_settings()
    if not api_key:
        raise LLMError(
            "LLM_API_KEY is not set. Put it in the .env file (local) or in the Space secrets (Hugging Face)."
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def chat(messages: list[dict], temperature: float = 0.2, json_mode: bool = False, retries: int = 3) -> str:
    client = get_client()
    _, _, model = config.llm_settings()
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, **extra
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_error = exc
            msg = str(exc).lower()
            if extra and "response_format" in msg:  # provider does not support JSON mode
                extra = {}
                continue
            if any(s in msg for s in ("429", "rate", "quota", "resource_exhausted", "503", "overloaded")):
                time.sleep(6 * (attempt + 1))
                continue
            break
    raise LLMError(f"LLM call failed ({model}): {last_error}")


def extract_json(text: str):
    """Parse JSON from a model reply, tolerating ```json fences and surrounding prose."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("No JSON found in model output")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[start : end + 1])
