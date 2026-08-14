"""Evaluation metrics for the RAG pipeline.

The headline number is **hallucination rate**: the fraction of unanswerable
questions that received a substantive answer anyway. Everything else —
accuracy, groundedness, retrieval recall — is diagnostic detail explaining
where that number comes from.

Abstention is scored as a first-class outcome, not as a failure. A system that
answers everything scores 100% "coverage" and is useless in a domain where a
wrong timer value causes a field incident.
"""

from __future__ import annotations

import re
import time


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(\d)\s+([a-z%])", r"\1\2", text)   # "100 ms" -> "100ms"
    text = re.sub(r"[\s,]+", " ", text)
    return text


def _contains(haystack: str, needle: str) -> bool:
    return _normalise(needle) in _normalise(haystack)


def _retrieval_hit(sources: list[dict], case: dict) -> dict:
    """Did retrieval surface at least one expected document/clause?"""
    expected_docs = set(case.get("expected_docs") or [])
    expected_clauses = set(case.get("expected_clauses") or [])
    if not expected_docs and not expected_clauses:
        return {"applicable": False, "doc_hit": None, "clause_hit": None, "rank": None}

    doc_hit = False
    clause_hit = False
    rank = None

    for i, source in enumerate(sources, start=1):
        doc_ok = (not expected_docs) or (source.get("doc_id") in expected_docs)
        clause_id = source.get("clause_id") or ""
        clause_ok = (not expected_clauses) or any(
            clause_id == c or clause_id.startswith(c + ".") or c.startswith(clause_id + ".")
            for c in expected_clauses
        )
        if doc_ok:
            doc_hit = True
        if clause_ok and doc_ok:
            clause_hit = True
            if rank is None:
                rank = i

    return {
        "applicable": True,
        "doc_hit": doc_hit,
        "clause_hit": clause_hit,
        "rank": rank,
    }


def score_case(engine, case: dict, retrieval_only: bool = False) -> dict:
    """Run one golden-set case through the pipeline and grade the outcome."""
    question = case["question"]
    started = time.perf_counter()

    retrieval = engine.retriever.retrieve(question)
    sources = [item.to_dict(include_text=False) for item in retrieval.chunks]
    hit = _retrieval_hit(sources, case)

    base = {
        "id": case.get("id"),
        "category": case.get("category"),
        "question": question,
        "expected_answerable": bool(case.get("answerable")),
        "retrieval": {
            "top_score": round(retrieval.top_score, 4),
            "passed_gate": retrieval.passed_gate,
            **hit,
            "top_sources": [s["citation_label"] for s in sources[:3]],
        },
    }

    if retrieval_only:
        # Grade retrieval alone: an answerable case should pass the gate and
        # surface an expected clause; an unanswerable one should be gated out
        # or at least not retrieve a false match.
        if case.get("answerable"):
            base["passed"] = bool(retrieval.passed_gate and hit.get("clause_hit"))
        else:
            base["passed"] = True  # judged at answer time, not retrieval time
        base["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return base

    result = engine.pipeline.run(question)
    answer = result.get("answer") or ""
    status = result.get("status", "")
    abstained = status == "abstained"
    confidence = (result.get("confidence") or {}).get("score", 0.0)
    verification = result.get("verification") or {}
    claims = result.get("claims") or []

    # --- content checks -------------------------------------------------
    missing_required = [t for t in (case.get("must_include") or []) if not _contains(answer, t)]
    present_forbidden = [t for t in (case.get("must_not_include") or []) if _contains(answer, t)]

    # --- pass/fail ------------------------------------------------------
    if case.get("answerable"):
        passed = (not abstained) and not missing_required and not present_forbidden
        failure_mode = (
            "abstained_on_answerable" if abstained
            else "missing_required_content" if missing_required
            else "contained_forbidden_content" if present_forbidden
            else None
        )
    else:
        passed = abstained
        failure_mode = None if abstained else "hallucinated_on_unanswerable"

    # --- groundedness ---------------------------------------------------
    uncited = sum(1 for c in claims if not c.get("citations"))
    flagged = sum(1 for c in claims if c.get("status") == "flagged")
    removed = verification.get("removed", 0)
    grounded = (uncited == 0) and (removed == 0) and (flagged == 0)

    base.update(
        {
            "passed": passed,
            "failure_mode": failure_mode,
            "abstained": abstained,
            "status": status,
            "confidence": confidence,
            "answer": answer[:600],
            "missing_required": missing_required,
            "present_forbidden": present_forbidden,
            "grounded": grounded,
            "claims_total": len(claims),
            "claims_uncited": uncited,
            "claims_flagged": flagged,
            "claims_removed": removed,
            "verifier_ran": verification.get("verifier_ran", False),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    )
    return base


def aggregate(results: list[dict]) -> dict:
    """Roll individual case results into headline metrics."""
    total = len(results) or 1
    answerable = [r for r in results if r.get("expected_answerable")]
    unanswerable = [r for r in results if not r.get("expected_answerable")]

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    # --- the headline ---------------------------------------------------
    # Retrieval-only runs never invoke the generator, so nothing can have
    # hallucinated. Report the metric as N/A rather than as a perfect 0% (or,
    # worse, a spurious 100% because no result carries an `abstained` flag).
    graded_end_to_end = any("abstained" in r for r in results)
    if graded_end_to_end:
        hallucinations = [
            r for r in unanswerable if not r.get("abstained") and not r.get("error")
        ]
        hallucination_rate = ratio(len(hallucinations), len(unanswerable))
        hallucination_count = len(hallucinations)
    else:
        hallucinations = []
        hallucination_rate = None
        hallucination_count = None

    # --- abstention behaviour -------------------------------------------
    correct_abstentions = sum(1 for r in unanswerable if r.get("abstained"))
    over_abstentions = sum(1 for r in answerable if r.get("abstained"))
    total_abstentions = correct_abstentions + over_abstentions

    # --- retrieval -------------------------------------------------------
    with_expectations = [
        r for r in answerable if (r.get("retrieval") or {}).get("applicable")
    ]
    clause_hits = sum(1 for r in with_expectations if r["retrieval"].get("clause_hit"))
    doc_hits = sum(1 for r in with_expectations if r["retrieval"].get("doc_hit"))
    ranks = [
        r["retrieval"]["rank"] for r in with_expectations if r["retrieval"].get("rank")
    ]
    mrr = round(sum(1 / rank for rank in ranks) / len(with_expectations), 4) if with_expectations else 0.0

    # --- confidence calibration -----------------------------------------
    answered = [r for r in results if not r.get("abstained") and "confidence" in r]
    correct_conf = [r["confidence"] for r in answered if r.get("passed")]
    wrong_conf = [r["confidence"] for r in answered if not r.get("passed")]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    latencies = sorted(r.get("latency_ms", 0) for r in results)
    grounded_answers = [r for r in answered if r.get("grounded")]

    by_category: dict[str, dict] = {}
    for r in results:
        entry = by_category.setdefault(
            r.get("category", "uncategorised"), {"total": 0, "passed": 0}
        )
        entry["total"] += 1
        entry["passed"] += 1 if r.get("passed") else 0
    for entry in by_category.values():
        entry["pass_rate"] = ratio(entry["passed"], entry["total"])

    return {
        "headline": {
            "hallucination_rate": hallucination_rate,
            "hallucination_count": hallucination_count,
            "unanswerable_cases": len(unanswerable),
            "end_to_end": graded_end_to_end,
            "overall_pass_rate": ratio(sum(1 for r in results if r.get("passed")), total),
        },
        "answering": {
            "answerable_cases": len(answerable),
            "answerable_pass_rate": ratio(
                sum(1 for r in answerable if r.get("passed")), len(answerable)
            ),
            "over_abstention_count": over_abstentions,
            "over_abstention_rate": ratio(over_abstentions, len(answerable)),
        },
        "abstention": {
            "total_abstentions": total_abstentions,
            "precision": ratio(correct_abstentions, total_abstentions),
            "recall": ratio(correct_abstentions, len(unanswerable)),
        },
        "retrieval": {
            "clause_hit_rate": ratio(clause_hits, len(with_expectations)),
            "document_hit_rate": ratio(doc_hits, len(with_expectations)),
            "mrr": mrr,
            "cases_measured": len(with_expectations),
        },
        "groundedness": {
            "fully_grounded_rate": ratio(len(grounded_answers), len(answered)),
            "answers_with_flagged_claims": sum(1 for r in answered if r.get("claims_flagged")),
            "answers_with_removed_claims": sum(1 for r in answered if r.get("claims_removed")),
            "uncited_claims_total": sum(r.get("claims_uncited", 0) for r in answered),
        },
        "calibration": {
            "mean_confidence_when_correct": mean(correct_conf),
            "mean_confidence_when_wrong": mean(wrong_conf),
            "separation": (
                round(mean(correct_conf) - mean(wrong_conf), 4)
                if correct_conf and wrong_conf
                else None
            ),
        },
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "p50": latencies[len(latencies) // 2] if latencies else 0,
            "p95": latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0,
        },
        "by_category": by_category,
        "failures": [
            {
                "id": r.get("id"),
                "category": r.get("category"),
                "failure_mode": r.get("failure_mode") or ("error" if r.get("error") else None),
                "question": r.get("question"),
                "detail": r.get("error")
                or r.get("missing_required")
                or r.get("present_forbidden")
                or (r.get("answer") or "")[:200],
            }
            for r in results
            if not r.get("passed")
        ],
    }
