"""The tutor: retrieval + grounded answers + quiz generation."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from . import config, llm
from .index import Embedder, load_index
from .prompts import QUIZ_SYSTEM, STYLES, TUTOR_SYSTEM
from .retrieval import Hit, Retriever


@dataclass
class Answer:
    text: str
    hits: list[Hit]


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.label})\n{h.chunk.text}" for i, h in enumerate(hits, start=1))


def format_sources_md(hits: list[Hit]) -> str:
    if not hits:
        return ""
    lines = ["**Sources**"]
    for i, h in enumerate(hits, start=1):
        preview = " ".join(h.chunk.text.split())[:160]
        lines.append(f"{i}. `{h.chunk.label}` — {preview}…")
    return "\n".join(lines)


def validate_quiz(data) -> list[dict]:
    items = data.get("questions", []) if isinstance(data, dict) else data
    valid = []
    for q in items if isinstance(items, list) else []:
        try:
            options = [str(o) for o in q["options"]]
            answer = int(q["answer"])
            if len(options) == 4 and 0 <= answer < 4 and str(q["question"]).strip():
                valid.append(
                    {
                        "question": str(q["question"]).strip(),
                        "options": options,
                        "answer": answer,
                        "explanation": str(q.get("explanation", "")).strip(),
                        "source": q.get("source"),
                    }
                )
        except (KeyError, TypeError, ValueError):
            continue
    return valid


def render_quiz_md(questions: list[dict], hits: list[Hit]) -> str:
    if not questions:
        return "The model did not return valid questions. Try again or change the topic."
    letters = "ABCD"
    blocks = []
    for n, q in enumerate(questions, start=1):
        opts = "\n".join(f"- **{letters[i]}.** {o}" for i, o in enumerate(q["options"]))
        src = ""
        if isinstance(q.get("source"), int) and 1 <= q["source"] <= len(hits):
            src = f"\n\n_Source: {hits[q['source'] - 1].chunk.label}_"
        blocks.append(
            f"### Question {n}\n{q['question']}\n\n{opts}\n\n"
            f"<details><summary>Show answer</summary>\n\n"
            f"**{letters[q['answer']]}** — {q['explanation']}{src}\n\n</details>"
        )
    return "\n\n---\n\n".join(blocks)


class Tutor:
    def __init__(self, index_dir: Path = config.INDEX_DIR, mode: str = "hybrid"):
        chunks, embeddings, meta = load_index(index_dir)
        embedder = Embedder(meta.get("embed_model", config.EMBED_MODEL)) if embeddings is not None else None
        self.retriever = Retriever(chunks, embeddings, embedder)
        self.mode = mode
        self.meta = meta

    def retrieve(self, query: str, k: int = config.TOP_K) -> list[Hit]:
        return self.retriever.search(query, k=k, mode=self.mode)

    def ask(self, question: str, k: int = config.TOP_K, style: str = "step-by-step") -> Answer:
        hits = self.retrieve(question, k)
        user = (
            f"Course excerpts:\n\n{format_context(hits)}\n\n"
            f"Student question: {question}\n\n{STYLES.get(style, STYLES['step-by-step'])}"
        )
        text = llm.chat([{"role": "system", "content": TUTOR_SYSTEM}, {"role": "user", "content": user}])
        return Answer(text, hits)

    def quiz(self, topic: str = "", n: int = 5, difficulty: str = "medium", language: str = "Vietnamese"):
        if topic.strip():
            hits = self.retrieve(topic, k=6)
        else:  # no topic: quiz on random parts of the notes
            sample = random.sample(self.retriever.chunks, k=min(6, len(self.retriever.chunks)))
            hits = [Hit(c, 0.0, i) for i, c in enumerate(sample, start=1)]
        system = QUIZ_SYSTEM.replace("{language}", language).replace("{difficulty}", difficulty)
        user = f"Course excerpts:\n\n{format_context(hits)}\n\nWrite {n} questions."
        raw = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.5,
            json_mode=True,
        )
        return validate_quiz(llm.extract_json(raw)), hits
