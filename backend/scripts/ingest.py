#!/usr/bin/env python3
"""Build (or rebuild) the hybrid index from the corpus directory.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --corpus /path/to/specs --stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.rag.vector_store import HybridIndex  # noqa: E402
from app.services.ingestion import build_index  # noqa: E402
from config import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest 3GPP specifications into the RAG index.")
    parser.add_argument("--corpus", type=Path, default=settings.corpus_dir)
    parser.add_argument("--index", type=Path, default=settings.index_dir)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_target_chars)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap_chars)
    parser.add_argument("--stats", action="store_true", help="Print per-document chunk statistics.")
    args = parser.parse_args()

    print(f"Corpus : {args.corpus}")
    print(f"Index  : {args.index}\n")

    index = HybridIndex(args.index)
    print(f"Embedder: {index.embedder.name}\n")

    try:
        report = build_index(
            args.corpus,
            index,
            target_chars=args.chunk_size,
            overlap_chars=args.overlap,
            min_chars=settings.chunk_min_chars,
            progress=lambda msg: print(f"  {msg}"),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print(f"Indexed {report['chunk_count']} chunks from {report['document_count']} documents "
          f"in {report['elapsed_seconds']}s")
    if report["duplicates_removed"]:
        print(f"Removed {report['duplicates_removed']} duplicate chunks")
    print("=" * 72)

    if args.stats:
        print(f"\n{'Document':<14} {'Chunks':>7} {'Clauses':>8} {'Chars':>10}  Title")
        print("-" * 90)
        for doc in report["documents"]:
            print(
                f"{doc['doc_id']:<14} {doc['chunks']:>7} {doc['clauses']:>8} "
                f"{doc['characters']:>10}  {doc['title'][:40]}"
            )

    if report["failures"]:
        print("\nFailures:")
        for failure in report["failures"]:
            print(f"  {failure['file']}: {failure['error']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
