"""Central configuration. Values come from environment variables or a local .env file."""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:  # python-dotenv is optional (not installed in CI)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# Print Vietnamese text safely in the Windows console.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "sample"
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(ROOT / "index")))

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.8-flash"


def llm_settings() -> tuple[str, str, str]:
    """Read LLM settings at call time so Hugging Face Space secrets are always picked up."""
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).strip()
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip()
    return api_key, base_url, model
