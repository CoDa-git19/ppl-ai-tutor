"""Compare BM25 vs dense vs hybrid retrieval on eval/qa.jsonl (no API calls).

Usage:  python -m scripts.eval_retrieval
Output: docs/retrieval_results.md and docs/retrieval_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

from ppl_tutor import config
from ppl_tutor.index import Embedder, load_index
from ppl_tutor.retrieval import Retriever


def evaluate(retriever: Retriever, qa: list[dict], mode: str, depth: int = 20) -> dict:
    hit1 = hit3 = hit5 = 0
    rr = 0.0
    for item in qa:
        ranked = [i for i, _ in retriever.rank(item["question"], k=depth, mode=mode)]
        rank = ranked.index(item["gold_id"]) + 1 if item["gold_id"] in ranked else None
        if rank:
            hit1 += rank <= 1
            hit3 += rank <= 3
            hit5 += rank <= 5
            rr += 1.0 / rank
    n = len(qa)
    return {"mode": mode, "n": n, "hit@1": hit1 / n, "hit@3": hit3 / n, "hit@5": hit5 / n, "mrr": rr / n}


def to_markdown(rows: list[dict], meta: dict) -> str:
    lines = [
        (
            f"Retrieval evaluation on {rows[0]['n']} synthetic questions "
            f"({meta.get('num_chunks', '?')} chunks, embeddings: `{meta.get('embed_model', '?')}`)"
        ),
        "",
        "| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |",
        "|---|---|---|---|---|",
    ]
    names = {"bm25": "BM25 (lexical)", "dense": "Dense (multilingual-e5)", "hybrid": "Hybrid (RRF)"}
    best = max(r["mrr"] for r in rows)
    for r in rows:
        name = names[r["mode"]]
        name = f"**{name}**" if r["mrr"] == best else name
        lines.append(f"| {name} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['hit@5']:.2f} | {r['mrr']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    qa_path = config.ROOT / "eval" / "qa.jsonl"
    if not qa_path.exists():
        raise SystemExit("eval/qa.jsonl not found. Run: python -m scripts.make_eval --n 20")
    qa = [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    chunks, embeddings, meta = load_index()
    embedder = Embedder(meta.get("embed_model", config.EMBED_MODEL)) if embeddings is not None else None
    retriever = Retriever(chunks, embeddings, embedder)
    modes = ["bm25", "dense", "hybrid"] if retriever.has_dense else ["bm25"]
    rows = [evaluate(retriever, qa, m) for m in modes]

    md = to_markdown(rows, meta)
    out = config.ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "retrieval_results.md").write_text(md, encoding="utf-8")
    (out / "retrieval_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(md)
    print(f"Saved to {Path('docs') / 'retrieval_results.md'}")


if __name__ == "__main__":
    main()
