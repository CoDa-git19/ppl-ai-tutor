"""Check the LLM connection: list available models and send one test message.

Usage:  python -m scripts.check_llm
        python -m scripts.check_llm --all      (list every model, not only chat-like ones)
"""
from __future__ import annotations

import argparse

from ppl_tutor import config, llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    api_key, base_url, model = config.llm_settings()
    print(f"Base URL : {base_url}")
    print(f"Model    : {model}")
    print(f"API key  : {'set (' + api_key[:4] + '...)' if api_key else 'MISSING'}\n")

    try:
        client = llm.get_client()
        ids = sorted(m.id.replace("models/", "") for m in client.models.list())
        keep = ids if args.all else [i for i in ids if any(w in i for w in ("flash", "pro", "llama", "gpt"))]
        print("Available models:")
        for i in keep:
            print("  ", i)
    except Exception as exc:
        print(f"Could not list models: {exc}")

    print("\nTest message ...")
    try:
        reply = llm.chat([{"role": "user", "content": "Reply with exactly: OK"}], temperature=0)
        print(f"Reply: {reply!r}\nLLM connection works.")
    except Exception as exc:
        print(f"FAILED: {exc}")
        print("\nFixes: 401/403 -> wrong key | 404 -> set LLM_MODEL to a name from the list above"
              " | 429 -> free quota hit, wait a minute or pick another model")


if __name__ == "__main__":
    main()
