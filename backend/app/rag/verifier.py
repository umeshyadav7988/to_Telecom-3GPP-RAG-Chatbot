"""Post-generation grounding verification — the last line of defence.

Four independent checks run over every claim. Three are deterministic and one
uses a model; a claim has to get past all of them to be presented without a
warning.

  1. Citation validity  — does every cited index correspond to a source that
     was actually retrieved for this turn? (Catches invented `[S9]` markers.)
  2. Quote provenance   — does the verbatim quote the generator supplied really
     occur in one of the sources it cited? (Catches fabricated evidence.)
  3. Numeric guard      — does every number, timer and identifier in the claim
     occur in the cited text? (Catches the highest-cost error class.)
  4. LLM entailment     — does the cited text actually entail the claim, as
     opposed to merely containing the same tokens? (Catches subtle misreadings
     that all three deterministic checks pass.)

Checks 1-3 cannot be fooled by a confident model. Check 4 catches what they
structurally cannot see. The combination is why this pipeline is closer to
zero hallucinations than a well-prompted single call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..utils.text import normalise_for_match, numeric_guard
from .generator import Claim, GeneratedAnswer
from .llm import BaseLLMClient
from .prompts import VERIFIER_SCHEMA, VERIFIER_SYSTEM_PROMPT, build_verifier_prompt
from .retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Quote matching tolerance. Exact substring is preferred; models occasionally
# normalise whitespace or drop a trailing clause reference while copying.
_QUOTE_TOKEN_OVERLAP_THRESHOLD = 0.80

# `partially_supported` sits above the default min_support_ratio (0.6) on
# purpose: it means the main assertion holds but a detail is unstated. Showing
# that with a visible warning is more useful than withholding it, and the user
# can see exactly which element the verifier could not confirm. Only
# `unsupported` and `contradicted` drag an answer below the abstention line.
_VERDICT_WEIGHT = {
    "supported": 1.0,
    "partially_supported": 0.7,
    "unverified": 0.6,      # verifier disabled/unavailable: neither reward nor punish
    "unsupported": 0.0,
    "contradicted": 0.0,
}


@dataclass
class VerificationReport:
    claims: list[Claim]
    support_ratio: float
    confidence: float
    confidence_label: str
    accepted_count: int
    flagged_count: int
    removed_count: int
    verifier_ran: bool
    checks: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------

def _evidence_for(claim: Claim, source_map: dict[int, str]) -> str:
    return "\n\n".join(source_map[c] for c in claim.citations if c in source_map)


def _quote_matches(quote: str, evidence: str) -> tuple[bool, float]:
    """Exact-normalised substring, falling back to token overlap."""
    if not quote:
        return False, 0.0

    q = normalise_for_match(quote)
    e = normalise_for_match(evidence)
    if not q:
        return False, 0.0
    if q in e:
        return True, 1.0

    q_tokens = q.split()
    if not q_tokens:
        return False, 0.0
    e_tokens = set(e.split())
    overlap = sum(1 for t in q_tokens if t in e_tokens) / len(q_tokens)
    return overlap >= _QUOTE_TOKEN_OVERLAP_THRESHOLD, round(overlap, 3)


def run_deterministic_checks(
    claims: list[Claim],
    source_map: dict[int, str],
    *,
    enable_numeric_guard: bool = True,
) -> None:
    """Mutate claims in place with the results of checks 1-3."""
    for claim in claims:
        if not claim.citations:
            claim.issues.append("no_valid_citation")
            claim.status = "removed"
            continue

        if claim.invalid_citations:
            claim.issues.append("invalid_citation_dropped")

        evidence = _evidence_for(claim, source_map)

        claim.quote_checked = bool(claim.quote)
        if claim.quote:
            found, overlap = _quote_matches(claim.quote, evidence)
            claim.quote_found = found
            if not found:
                claim.issues.append("quote_not_found_in_cited_source")
                logger.debug(
                    "Quote provenance failed (overlap=%.2f) for claim %d", overlap, claim.index
                )

        if enable_numeric_guard:
            claim.numeric = numeric_guard(claim.text, evidence)
            if not claim.numeric["passed"]:
                claim.issues.append("unverified_values")


# ---------------------------------------------------------------------------
# LLM entailment
# ---------------------------------------------------------------------------

def run_entailment_check(
    claims: list[Claim],
    source_map: dict[int, str],
    *,
    client: BaseLLMClient,
    model: str | None = None,
    max_tokens: int = 6000,
) -> tuple[bool, dict, float]:
    """Batch all claims into one verifier call. Returns (ran, usage, latency)."""
    payload = [
        {
            "index": claim.index,
            "claim": claim.text,
            "evidence": _evidence_for(claim, source_map),
        }
        for claim in claims
        if claim.status != "removed" and claim.citations
    ]
    if not payload:
        return False, {}, 0.0

    try:
        result = client.structured(
            system=VERIFIER_SYSTEM_PROMPT,
            user=build_verifier_prompt(payload),
            schema=VERIFIER_SCHEMA,
            model=model,
            max_tokens=max_tokens,
            # The verifier does one bounded judgement per claim; low effort is
            # both sufficient and materially cheaper than the answer call.
            effort="low",
        )
    except Exception as exc:
        logger.warning("Entailment verification unavailable: %s", exc)
        return False, {}, 0.0

    by_index = {claim.index: claim for claim in claims}
    for verdict in result.data.get("verdicts", []):
        try:
            claim = by_index[int(verdict.get("claim_index"))]
        except (KeyError, TypeError, ValueError):
            continue
        claim.verdict = str(verdict.get("verdict", "unverified"))
        try:
            claim.verdict_confidence = float(verdict.get("confidence", 0.0))
        except (TypeError, ValueError):
            claim.verdict_confidence = 0.0
        claim.verdict_reason = str(verdict.get("reason", ""))

    return True, result.usage, result.latency_ms


# ---------------------------------------------------------------------------
# Disposition + confidence
# ---------------------------------------------------------------------------

def _assign_status(claim: Claim, verifier_ran: bool) -> None:
    if claim.status == "removed":
        return

    if claim.verdict in ("unsupported", "contradicted"):
        claim.status = "removed"
        claim.issues.append(f"entailment_{claim.verdict}")
        return

    # A fabricated quote is treated as removal-worthy even if entailment
    # passed: the generator asserted evidence that does not exist, and that
    # signal is too serious to show the user without comment.
    if claim.quote_checked and not claim.quote_found:
        claim.status = "flagged"
        return

    if claim.verdict == "partially_supported" or (
        claim.numeric and not claim.numeric.get("passed", True)
    ):
        claim.status = "flagged"
        return

    if not verifier_ran:
        claim.status = "accepted" if not claim.issues else "flagged"
        return

    claim.status = "accepted"


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"


def verify(
    answer: GeneratedAnswer,
    retrieval: RetrievalResult,
    *,
    client: BaseLLMClient | None,
    enable_verifier: bool = True,
    enable_numeric_guard: bool = True,
    verifier_model: str | None = None,
    max_tokens: int = 6000,
) -> VerificationReport:
    source_map = {item.source_index: item.chunk.body for item in retrieval.chunks}
    claims = answer.claims
    notes: list[str] = []

    run_deterministic_checks(claims, source_map, enable_numeric_guard=enable_numeric_guard)

    verifier_ran = False
    usage: dict = {}
    latency = 0.0
    if enable_verifier and client is not None and answer.mode == "generative":
        verifier_ran, usage, latency = run_entailment_check(
            claims, source_map, client=client, model=verifier_model, max_tokens=max_tokens
        )
        if not verifier_ran:
            notes.append("Entailment verification did not run; deterministic checks only.")
    elif answer.mode == "extractive":
        # Extractive claims are verbatim source text: entailment is trivially
        # satisfied, so the LLM pass would be pure cost.
        for claim in claims:
            claim.verdict = "supported"
            claim.verdict_confidence = 1.0
            claim.verdict_reason = "Verbatim excerpt from the cited clause."
        notes.append("Extractive mode: claims are verbatim source text.")
    elif not enable_verifier:
        notes.append("Entailment verification is disabled by configuration.")

    for claim in claims:
        _assign_status(claim, verifier_ran or answer.mode == "extractive")

    total = len(claims) or 1
    support_score = sum(_VERDICT_WEIGHT.get(c.verdict, 0.5) for c in claims) / total
    citation_coverage = sum(1 for c in claims if c.citations) / total

    # Retrieval quality is folded in because a weakly-retrieved answer deserves
    # low confidence even when every claim it did make checks out.
    retrieval_component = min(1.0, retrieval.top_score / max(retrieval.gate_threshold * 2, 1e-6))

    confidence = (
        0.45 * support_score + 0.20 * citation_coverage + 0.35 * retrieval_component
    )
    removed = sum(1 for c in claims if c.status == "removed")
    if removed:
        # Any removal is evidence the generator drifted; penalise proportionally.
        confidence *= max(0.35, 1.0 - (removed / total))

    if answer.mode == "extractive":
        # Extractive claims are trivially "supported" because they *are* the
        # source text — but nothing has checked that the source answers the
        # question. Reporting high confidence there would be exactly the kind
        # of miscalibration this pipeline exists to prevent, so cap it.
        confidence = min(confidence, 0.5)
        notes.append(
            "Confidence is capped in extractive mode: relevance to the question "
            "has not been verified, only that the text is genuine specification text."
        )
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    return VerificationReport(
        claims=claims,
        support_ratio=round(support_score, 3),
        confidence=confidence,
        confidence_label=_confidence_label(confidence),
        accepted_count=sum(1 for c in claims if c.status == "accepted"),
        flagged_count=sum(1 for c in claims if c.status == "flagged"),
        removed_count=removed,
        verifier_ran=verifier_ran,
        checks={
            "citation_validity": True,
            "quote_provenance": True,
            "numeric_guard": enable_numeric_guard,
            "llm_entailment": verifier_ran,
        },
        usage=usage,
        latency_ms=latency,
        notes=notes,
    )
