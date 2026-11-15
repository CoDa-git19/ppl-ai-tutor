"""Build, save and load the retrieval index (chunks + dense embeddings)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config
from .chunking import Chunk, make_chunks
from .loaders import load_folder


class Embedder:
    """Thin wrapper around sentence-transformers, loaded lazily (heavy import)."""

    def __init__(self, model_name: str = config.EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        if "e5" in self.model_name.lower():  # E5 models expect these prefixes
            texts = [f"{kind}: {t}" for t in texts]
        vectors = self._load().encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 64
        )
        return np.asarray(vectors, dtype=np.float32)


def build_index(data_dir: Path, out_dir: Path, embedder: Embedder | None = None) -> dict:
    print(f"Loading notes from {data_dir}")
    sections = load_folder(data_dir)
    if not sections:
        raise SystemExit(f"No readable .tex/.md/.txt/.pdf files found in {data_dir}")
    chunks = make_chunks(sections, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"{len(sections)} sections -> {len(chunks)} chunks")

    embedder = embedder or Embedder()
    t0 = time.time()
    vectors = embedder.encode([c.search_text for c in chunks], kind="passage")
    print(f"Embedded {len(chunks)} chunks with {embedder.model_name} in {time.time() - t0:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    np.save(out_dir / "embeddings.npy", vectors)
    meta = {
        "embed_model": embedder.model_name,
        "num_chunks": len(chunks),
        "num_sections": len(sections),
        "sources": sorted({c.source for c in chunks}),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_index(index_dir: Path = config.INDEX_DIR) -> tuple[list[Chunk], np.ndarray | None, dict]:
    chunks_path = index_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Index not found in {index_dir}. Run: python -m scripts.build_index"
        )
    chunks = [Chunk(**d) for d in json.loads(chunks_path.read_text(encoding="utf-8"))]
    emb_path = index_dir / "embeddings.npy"
    embeddings = np.load(emb_path) if emb_path.exists() else None
    meta_path = index_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return chunks, embeddings, meta
