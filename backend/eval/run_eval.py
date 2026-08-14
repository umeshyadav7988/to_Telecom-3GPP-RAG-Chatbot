#!/usr/bin/env python3
"""CLI evaluation runner.

    python eval/run_eval.py                       # full golden set
    python eval/run_eval.py --retrieval-only      # no LLM calls, no cost
    python eval/run_eval.py --category false_premise
    python eval/run_eval.py --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.engine import get_engine  # noqa: E402
from eval.metrics import aggregate, score_case  # noqa: E402

GOLDEN_SET = BACKEND_DIR / "eval" / "golden_set.json"

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the golden set.")
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET)
    parser.add_argument("--category", action="append", help="Filter by category (repeatable).")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retrieval-only", action="store_true", help="Skip generation (free, fast).")
    parser.add_argument("--out", type=Path, help="Write the full JSON report here.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.golden_set.exists():
        print(f"Golden set not found: {args.golden_set}", file=sys.stderr)
        return 1

    cases = json.loads(args.golden_set.read_text(encoding="utf-8"))["cases"]
    if args.category:
        wanted = set(args.category)
        cases = [c for c in cases if c.get("category") in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No cases matched the filters.", file=sys.stderr)
        return 1

    engine = get_engine()
    if not engine.index.is_ready:
        print("Index is not built. Run `python scripts/ingest.py` first.", file=sys.stderr)
        return 1

    mode = "retrieval-only" if args.retrieval_only else (
        "generative" if engine.client else "extractive"
    )
    print(f"{BOLD}Evaluating {len(cases)} cases | mode={mode} | "
          f"chunks={engine.index.size} | embedder={engine.index.embedder.name}{RESET}\n")

    results = []
    for i, case in enumerate(cases, start=1):
        try:
            scored = score_case(engine, case, retrieval_only=args.retrieval_only)
        except Exception as exc:
            scored = {
                "id": case.get("id"),
                "category": case.get("category"),
                "question": case.get("question"),
                "expected_answerable": bool(case.get("answerable")),
                "error": str(exc),
                "passed": False,
            }
        results.append(scored)

        if not args.quiet:
            mark = f"{GREEN}PASS{RESET}" if scored.get("passed") else f"{RED}FAIL{RESET}"
            note = ""
            if scored.get("abstained"):
                note = f" {YELLOW}[abstained]{RESET}"
            elif scored.get("claims_flagged"):
                note = f" {YELLOW}[{scored['claims_flagged']} flagged]{RESET}"
            print(f"[{i:>2}/{len(cases)}] {mark} {scored['id']:<12} {scored['question'][:60]}{note}")
            if not scored.get("passed"):
                detail = scored.get("failure_mode") or scored.get("error") or ""
                if scored.get("missing_required"):
                    detail += f" missing={scored['missing_required']}"
                print(f"          {DIM}{detail}{RESET}")

    summary = aggregate(results)

    print(f"\n{BOLD}{'=' * 74}{RESET}")
    head = summary["headline"]
    print(f"{BOLD}HEADLINE{RESET}")
    if head["hallucination_rate"] is None:
        print(f"  Hallucination rate        {DIM}n/a (retrieval-only run: nothing was generated){RESET}")
    else:
        colour = GREEN if head["hallucination_rate"] == 0 else RED
        print(f"  Hallucination rate        {colour}{head['hallucination_rate']:.1%}{RESET}"
              f"  ({head['hallucination_count']}/{head['unanswerable_cases']} unanswerable "
              f"questions answered anyway)")
    print(f"  Overall pass rate         {head['overall_pass_rate']:.1%}")

    if not args.retrieval_only:
        ans, abst, gnd, cal = (
            summary["answering"], summary["abstention"],
            summary["groundedness"], summary["calibration"],
        )
        print(f"\n{BOLD}ANSWERING{RESET}")
        print(f"  Answerable pass rate      {ans['answerable_pass_rate']:.1%}"
              f"  ({ans['answerable_cases']} cases)")
        print(f"  Over-abstention rate      {ans['over_abstention_rate']:.1%}"
              f"  (refused {ans['over_abstention_count']} answerable questions)")
        print(f"\n{BOLD}ABSTENTION{RESET}")
        print(f"  Precision                 {abst['precision']:.1%}   (abstained and was right to)")
        print(f"  Recall                    {abst['recall']:.1%}   (of questions it should refuse)")
        print(f"\n{BOLD}GROUNDEDNESS{RESET}")
        print(f"  Fully grounded answers    {gnd['fully_grounded_rate']:.1%}")
        print(f"  With flagged claims       {gnd['answers_with_flagged_claims']}")
        print(f"  With removed claims       {gnd['answers_with_removed_claims']}")
        print(f"  Uncited claims            {gnd['uncited_claims_total']}")
        print(f"\n{BOLD}CALIBRATION{RESET}")
        print(f"  Mean confidence, correct  {cal['mean_confidence_when_correct']}")
        print(f"  Mean confidence, wrong    {cal['mean_confidence_when_wrong']}")
        print(f"  Separation                {cal['separation']}"
              f"   {DIM}(higher is better; negative means miscalibrated){RESET}")

    ret = summary["retrieval"]
    print(f"\n{BOLD}RETRIEVAL{RESET}")
    print(f"  Clause hit rate           {ret['clause_hit_rate']:.1%}  ({ret['cases_measured']} cases)")
    print(f"  Document hit rate         {ret['document_hit_rate']:.1%}")
    print(f"  MRR                       {ret['mrr']}")

    lat = summary["latency_ms"]
    print(f"\n{BOLD}LATENCY{RESET}  mean {lat['mean']} ms | p50 {lat['p50']} ms | p95 {lat['p95']} ms")

    print(f"\n{BOLD}BY CATEGORY{RESET}")
    for category, stats in sorted(summary["by_category"].items()):
        print(f"  {category:<18} {stats['passed']}/{stats['total']}  {stats['pass_rate']:.0%}")

    if summary["failures"]:
        print(f"\n{BOLD}{RED}FAILURES{RESET}")
        for failure in summary["failures"]:
            print(f"  {failure['id']:<12} [{failure['failure_mode']}] {failure['question'][:56]}")

    print(f"{BOLD}{'=' * 74}{RESET}")

    if args.out:
        args.out.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8"
        )
        print(f"\nFull report written to {args.out}")

    # Non-zero exit on any hallucination makes this usable as a CI gate.
    # A retrieval-only run gates on retrieval instead, since nothing was generated.
    if head["hallucination_count"] is None:
        return 0 if summary["retrieval"]["clause_hit_rate"] >= 0.8 else 1
    return 1 if head["hallucination_count"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
