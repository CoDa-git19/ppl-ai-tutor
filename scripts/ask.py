"""Ask the tutor from the terminal.

Usage:  python -m scripts.ask "What is static scoping?"
        python -m scripts.ask "Phạm vi tĩnh là gì?" --retrieval-only     (no API key needed)
"""
from __future__ import annotations

import argparse

from ppl_tutor.engine import Tutor, format_sources_md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ap.add_argument("--style", choices=["concise", "step-by-step", "beginner"], default="step-by-step")
    ap.add_argument("--retrieval-only", action="store_true", help="only show retrieved chunks")
    args = ap.parse_args()

    tutor = Tutor(mode=args.mode)
    if args.retrieval_only:
        for h in tutor.retrieve(args.question, args.k):
            print(f"[{h.rank}] score={h.score:.4f}  {h.chunk.label}")
            print("    " + " ".join(h.chunk.text.split())[:200] + "\n")
        return
    answer = tutor.ask(args.question, k=args.k, style=args.style)
    print(answer.text)
    print()
    print(format_sources_md(answer.hits))


if __name__ == "__main__":
    main()
