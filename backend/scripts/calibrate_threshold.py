#!/usr/bin/env python3
"""Calibrate MIN_RETRIEVAL_SCORE against the golden set.

The retrieval gate's threshold is not a universal constant: it depends on the
reranker in use (the lexical-semantic fallback and a cross-encoder produce
completely different score distributions). Hard-coding a number that was tuned
on one setup and shipping it with another is how a gate silently stops gating.

This script measures the actual score distributions for answerable and
unanswerable questions and recommends the highest threshold that rejects no
answerable question.

    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --margin 0.03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.engine import get_engine  # noqa: E402

GOLDEN_SET = BACKEND_DIR / "eval" / "golden_set.json"


def _histogram(scores: list[float], width: int = 40) -> str:
    if not scores:
        return "(none)"
    buckets = [0] * 10
    for score in scores:
        buckets[min(9, int(score * 10))] += 1
    peak = max(buckets) or 1
    lines = []
    for i, count in enumerate(buckets):
        bar = "#" * int(width * count / peak)
        lines.append(f"    {i/10:.1f}-{(i+1)/10:.1f}  {bar:<{width}} {count}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate the retrieval abstention gate.")
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.04,
        help="Safety margin below the lowest answerable score (default 0.04).",
    )
    args = parser.parse_args()

    engine = get_engine()
    if not engine.index.is_ready:
        print("Index is not built. Run `python scripts/ingest.py` first.", file=sys.stderr)
        return 1

    cases = json.loads(args.golden_set.read_text(encoding="utf-8"))["cases"]

    answerable: list[tuple[float, str]] = []
    unanswerable: list[tuple[float, str]] = []
    for case in cases:
        result = engine.retriever.retrieve(case["question"])
        bucket = answerable if case.get("answerable") else unanswerable
        bucket.append((round(result.top_score, 4), case["id"]))

    answerable.sort()
    unanswerable.sort()
    a_scores = [s for s, _ in answerable]
    u_scores = [s for s, _ in unanswerable]

    print(f"Reranker : {engine.reranker.name}")
    print(f"Embedder : {engine.index.embedder.name}")
    print(f"Chunks   : {engine.index.size}\n")

    print(f"ANSWERABLE ({len(a_scores)} cases) - want these ABOVE the threshold")
    print(_histogram(a_scores))
    print(f"    min={a_scores[0]}  ({answerable[0][1]})   max={a_scores[-1]}\n")

    print(f"UNANSWERABLE ({len(u_scores)} cases) - want these BELOW the threshold")
    print(_histogram(u_scores))
    print(f"    min={u_scores[0]}   max={u_scores[-1]}  ({unanswerable[-1][1]})\n")

    recommended = max(0.0, round(a_scores[0] - args.margin, 2))
    caught = sum(1 for s in u_scores if s < recommended)
    rejected = sum(1 for s in a_scores if s < recommended)

    print("=" * 70)
    print(f"RECOMMENDED  MIN_RETRIEVAL_SCORE={recommended}")
    print(f"  Gate catches {caught}/{len(u_scores)} unanswerable questions before any LLM call")
    print(f"  Gate wrongly rejects {rejected}/{len(a_scores)} answerable questions")

    overlap = [(s, cid) for s, cid in unanswerable if s >= a_scores[0]]
    if overlap:
        print(f"\n  {len(overlap)} unanswerable question(s) score above the lowest answerable one:")
        for score, case_id in overlap:
            print(f"    {case_id:<12} {score}")
        print(
            "\n  These are NOT separable by retrieval score, and no threshold will\n"
            "  catch them. They are typically false-premise questions whose topic IS\n"
            "  in the corpus (a non-existent timer, a non-existent 5QI value) - the\n"
            "  right clause is retrieved, it simply does not contain the asserted\n"
            "  fact. Gate 2 (the generator declaring insufficient context) and\n"
            "  Gate 3 (entailment verification) exist precisely for this class."
        )
    print("=" * 70)
    print(f"\nSet it in backend/.env:\n    MIN_RETRIEVAL_SCORE={recommended}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
