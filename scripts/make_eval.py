"""Create a synthetic retrieval test set: one LLM-written question per sampled chunk.

Usage:  python -m scripts.make_eval --n 20
Output: eval/qa.jsonl  (question + id of the chunk that answers it)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from ppl_tutor import config, llm
from ppl_tutor.index import load_index
from ppl_tutor.prompts import EVAL_QUESTION_SYSTEM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=4.0, help="pause between calls (free-tier rate limit)")
    ap.add_argument("--out", type=Path, default=config.ROOT / "eval" / "qa.jsonl")
    args = ap.parse_args()

    chunks, _, _ = load_index()
    pool = [c for c in chunks if len(c.text) >= 200]
    random.seed(args.seed)
    picked = random.sample(pool, k=min(args.n, len(pool)))
    print(f"Writing {len(picked)} questions from {len(pool)} candidate chunks")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(picked, start=1):
            try:
                raw = llm.chat(
                    [
                        {"role": "system", "content": EVAL_QUESTION_SYSTEM},
                        {"role": "user", "content": f"Excerpt ({chunk.label}):\n{chunk.text}"},
                    ],
                    temperature=0.3,
                    json_mode=True,
                )
                question = str(llm.extract_json(raw)["question"]).strip()
            except Exception as exc:
                print(f"  {i:>2}. skipped chunk {chunk.id}: {exc}")
                continue
            f.write(json.dumps({"question": question, "gold_id": chunk.id, "source": chunk.label},
                               ensure_ascii=False) + "\n")
            written += 1
            print(f"  {i:>2}. {question[:90]}")
            time.sleep(args.sleep)
    print(f"\nSaved {written} questions to {args.out}")


if __name__ == "__main__":
    main()
