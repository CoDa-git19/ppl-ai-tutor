"""Split sections into overlapping chunks that fit comfortably in an LLM prompt."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .loaders import Section


@dataclass
class Chunk:
    id: int
    source: str
    title: str
    text: str
    page: int | None = None

    @property
    def search_text(self) -> str:
        """Text used for retrieval: the section title adds useful keywords."""
        return f"{self.title}\n{self.text}"

    @property
    def label(self) -> str:
        page = f", p.{self.page}" if self.page else ""
        return f"{self.source} › {self.title}{page}"

    def to_dict(self) -> dict:
        return asdict(self)


def _hard_split(paragraph: str, size: int) -> list[str]:
    pieces, start = [], 0
    while start < len(paragraph):
        end = min(start + size, len(paragraph))
        if end < len(paragraph):
            cut = paragraph.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        pieces.append(paragraph[start:end].strip())
        start = end
    return [p for p in pieces if p]


def split_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Paragraph-aware splitter; consecutive chunks share about `overlap` characters."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if para:
            pieces.extend([para] if len(para) <= size else _hard_split(para, size))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            space = tail.find(" ")
            tail = tail[space + 1 :] if space != -1 else tail
            current = f"{tail}\n\n{piece}" if tail else piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def make_chunks(sections: list[Section], size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in sections:
        for piece in split_text(sec.text, size, overlap):
            chunks.append(Chunk(len(chunks), sec.source, sec.title, piece, sec.page))
    return chunks
