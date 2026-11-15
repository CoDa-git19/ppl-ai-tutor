"""Fast tests: no network, no API key, no embedding model download."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ppl_tutor.chunking import Chunk, make_chunks, split_text
from ppl_tutor.engine import format_context, render_quiz_md, validate_quiz
from ppl_tutor.llm import extract_json
from ppl_tutor.loaders import Section, load_markdown, load_tex
from ppl_tutor.retrieval import Hit, Retriever, rrf, tokenize


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder, enough to exercise the dense path."""

    model_name = "fake"

    def encode(self, texts, kind="passage"):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                out[row, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


CHUNKS = [
    Chunk(0, "notes.md", "Scope", "Static scope resolves names using the program text and enclosing blocks."),
    Chunk(1, "notes.md", "Parsing", "LL(1) parsers use FIRST and FOLLOW sets to fill the parse table."),
    Chunk(2, "notes.md", "Closures", "A closure is a function together with its defining environment."),
    Chunk(3, "notes.md", "Types", "Python is dynamically typed but strongly typed."),
]


def test_split_text_respects_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    chunks = split_text(text, size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c) <= 300 + 60 + 10 for c in chunks)
    assert chunks[0][-20:].split()[-1] in chunks[1]  # consecutive chunks share text


def test_split_text_short_and_empty():
    assert split_text("", 100, 10) == []
    assert split_text("short text", 100, 10) == ["short text"]


def test_make_chunks_keeps_metadata():
    secs = [Section("a.tex", "Intro", "x " * 600, page=3)]
    chunks = make_chunks(secs, size=200, overlap=20)
    assert len(chunks) > 1
    assert [c.id for c in chunks] == list(range(len(chunks)))
    assert chunks[0].label == "a.tex › Intro, p.3"


def test_load_tex_sections_code_and_comments(tmp_path):
    tex = r"""\documentclass{article}
\begin{document}
\section{Scope and \textbf{Binding}}
Static scope uses the program text. % this comment must disappear
\subsection{Example}
Here is code:
\begin{lstlisting}
x = 1  % kept inside code
def f(): return x
\end{lstlisting}
And inline \verb|a_b{c}| code.
\end{document}"""
    path = tmp_path / "notes.tex"
    path.write_text(tex, encoding="utf-8")
    sections = load_tex(path)
    titles = [s.title for s in sections]
    assert "Scope and Binding" in titles
    assert "Scope and Binding › Example" in titles
    full = "\n".join(s.text for s in sections)
    assert "comment must disappear" not in full
    assert "x = 1  % kept inside code" in full
    assert "`a_b{c}`" in full


def test_load_markdown_headings(tmp_path):
    md = "# Course\nintro text that is long enough\n## Scope\nstatic scope text\n```\n# not a heading\n```\n"
    path = tmp_path / "n.md"
    path.write_text(md, encoding="utf-8")
    sections = load_markdown(path)
    assert [s.title for s in sections] == ["Course", "Course › Scope"]
    assert "# not a heading" in sections[1].text


def test_tokenize_vietnamese():
    assert tokenize("Phạm vi TĨNH và động") == ["phạm", "vi", "tĩnh", "và", "động"]


def test_rrf_prefers_documents_ranked_high_in_both_lists():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert {fused[0][0], fused[1][0]} == {1, 2}
    assert fused[-1][0] in (3, 4)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_retriever_finds_relevant_chunk(mode):
    emb = FakeEmbedder()
    vectors = emb.encode([c.search_text for c in CHUNKS])
    retriever = Retriever(CHUNKS, vectors, emb)
    hits = retriever.search("how do FIRST and FOLLOW sets work in LL(1)?", k=2, mode=mode)
    assert hits[0].chunk.id == 1


def test_hybrid_falls_back_to_bm25_without_embeddings():
    retriever = Retriever(CHUNKS)
    assert not retriever.has_dense
    assert retriever.search("closure environment", k=1)[0].chunk.id == 2


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Sure! Here it is: {"a": 3} hope it helps') == {"a": 3}


def test_validate_quiz_drops_bad_items():
    data = {
        "questions": [
            {"question": "Q1?", "options": ["a", "b", "c", "d"], "answer": 2, "explanation": "e", "source": 1},
            {"question": "bad", "options": ["a", "b"], "answer": 0},
            {"question": "bad2", "options": ["a", "b", "c", "d"], "answer": 7},
        ]
    }
    valid = validate_quiz(data)
    assert len(valid) == 1 and valid[0]["answer"] == 2
    md = render_quiz_md(valid, [Hit(CHUNKS[0], 1.0, 1)])
    assert "Question 1" in md and "**C**" in md and "notes.md › Scope" in md


def test_format_context_numbers_sources():
    hits = [Hit(CHUNKS[0], 1.0, 1), Hit(CHUNKS[2], 0.5, 2)]
    ctx = format_context(hits)
    assert ctx.startswith("[1] (notes.md › Scope)")
    assert "[2] (notes.md › Closures)" in ctx
