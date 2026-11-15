"""Build the retrieval index from your course notes.

Usage:  python -m scripts.build_index            (reads data/raw, falls back to data/sample)
        python -m scripts.build_index --data data/sample
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ppl_tutor import config
from ppl_tutor.index import build_index
from ppl_tutor.loaders import SUPPORTED


def has_notes(folder: Path) -> bool:
    return folder.exists() and any(p.suffix.lower() in SUPPORTED for p in folder.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None, help="folder with .tex/.md/.txt/.pdf notes")
    ap.add_argument("--out", type=Path, default=config.INDEX_DIR)
    args = ap.parse_args()

    data = args.data
    if data is None:
        data = config.RAW_DIR if has_notes(config.RAW_DIR) else config.SAMPLE_DIR
        if data == config.SAMPLE_DIR:
            print("data/raw is empty -> using the bundled sample notes in data/sample")
    meta = build_index(data, args.out)
    print(f"\nIndex saved to {args.out}")
    print(f"  chunks : {meta['num_chunks']}")
    print(f"  sources: {', '.join(meta['sources'])}")


if __name__ == "__main__":
    main()
