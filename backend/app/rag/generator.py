"""Grounded answer generation.

Produces a list of atomic, individually cited claims rather than a paragraph.
Claims are the unit the verifier operates on: a paragraph can be 90% correct
and 10% invented, and there is no way to act on that. A claim is either
supported by its cited clause or it is not, and an unsupported one can be
removed without destroying the rest of the answer.

If no API key is configured the module degrades to `extractive` mode: it
returns the retrieved clauses verbatim as claims. The answer is less fluent,
but it is 100% grounded by construction — which is the right thing for a
demo machine with no credentials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .extractive import extract_units, render_unit
from .llm import BaseLLMClient, LLMRefusal
from .prompts import (
    ANSWER_SCHEMA,
    ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
    format_sources_block,
)
from .retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class Claim:
    """One assertion plus every signal gathered about its groundedness."""

    index: int
    text: str
    citations: list[int]
    quote: str = ""
    modality: str = "descriptive"

    # Deterministic checks (utils.text)
    citations_valid: bool = True
    invalid_citations: list[int] = field(default_factory=list)
    quote_found: bool = False
    quote_checked: bool = False
    numeric: dict = field(default_factory=dict)

    # LLM entailment pass
    verdict: str = "unverified"
    verdict_confidence: float = 0.0
    verdict_reason: str = ""

    # Final disposition
    status: str = "accepted"          # accepted | flagged | removed
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "citations": self.citations,
            "quote": self.quote,
            "modality": self.modality,
            "status": self.status,
            "issues": self.issues,
            "verification": {
                "verdict": self.verdict,
                "confidence": round(self.verdict_confidence, 3),
                "reason": self.verdict_reason,
                "citations_valid": self.citations_valid,
                "invalid_citations": self.invalid_citations,
                "quote_found_in_source": self.quote_found,
                "quote_checked": self.quote_checked,
                "numeric_guard": self.numeric,
            },
        }


@dataclass
class GeneratedAnswer:
    answerable: bool
    claims: list[Claim]
    refusal_reason: str = ""
    caveats: list[str] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    mode: str = "generative"          # generative | extractive
    model: str = ""
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Extractive fallback
# ---------------------------------------------------------------------------

def _extractive_answer(
    question: str, retrieval: RetrievalResult, max_claims: int = 5
) -> GeneratedAnswer:
    """Answer by selecting the sentences and table rows that address the question.

    Returning whole chunks would be simpler, but a 1400-character clause buries
    the one sentence that answers a lookup — and truncating to fit loses it.
    `extractive.extract_units` does the last mile deterministically, which is
    what makes no-LLM mode usable for factual lookups rather than merely
    "here is roughly the right page".
    """
    units = extract_units(question, retrieval.chunks, max_units=max_claims)

    # Nothing inside the retrieved clauses actually addressed the question,
    # even though the clauses themselves scored well.
    if not units:
        return GeneratedAnswer(
            answerable=False,
            claims=[],
            refusal_reason=(
                "The retrieved clauses did not contain any sentence or table row "
                "matching the question."
            ),
            mode="extractive",
        )

    by_source = {item.source_index: item for item in retrieval.chunks}
    claims: list[Claim] = []
    for i, unit in enumerate(units, start=1):
        chunk = by_source[unit.source_index].chunk
        claims.append(
            Claim(
                index=i,
                text=render_unit(unit),
                citations=[unit.source_index],
                quote=unit.text,
                modality="mandatory" if chunk.is_normative else "descriptive",
            )
        )

    return GeneratedAnswer(
        answerable=True,
        claims=claims,
        refusal_reason="",
        caveats=[
            "Extractive mode: no LLM key is configured, so these are verbatim "
            "specification excerpts selected by lexical match, not a synthesised "
            "answer. Relevance has not been verified by entailment."
        ],
        followups=[],
        mode="extractive",
    )


# ---------------------------------------------------------------------------
# Generative path
# ---------------------------------------------------------------------------

def _coerce_claims(raw_claims, valid_indices: set[int]) -> list[Claim]:
    claims: list[Claim] = []
    for i, raw in enumerate(raw_claims or [], start=1):
        if not isinstance(raw, dict):
            continue
        text = (raw.get("text") or "").strip()
        if not text:
            continue

        citations, invalid = [], []
        for c in raw.get("citations") or []:
            try:
                value = int(c)
            except (TypeError, ValueError):
                continue
            if value in valid_indices:
                if value not in citations:
                    citations.append(value)
            elif value not in invalid:
                invalid.append(value)

        claims.append(
            Claim(
                index=i,
                text=text,
                citations=citations,
                quote=(raw.get("quote") or "").strip(),
                modality=(raw.get("modality") or "descriptive").strip(),
                citations_valid=not invalid,
                invalid_citations=invalid,
            )
        )
    return claims


def generate_answer(
    question: str,
    retrieval: RetrievalResult,
    *,
    client: BaseLLMClient | None,
    history_block: str = "",
    model: str | None = None,
    max_tokens: int = 8000,
) -> GeneratedAnswer:
    """Produce a cited, claim-structured answer from the retrieved clauses."""
    if not retrieval.chunks:
        return GeneratedAnswer(
            answerable=False,
            claims=[],
            refusal_reason="No relevant clauses were retrieved from the indexed specifications.",
            mode="extractive" if client is None else "generative",
        )

    if client is None:
        return _extractive_answer(question, retrieval)

    sources_block = format_sources_block(retrieval.chunks)
    prompt = build_answer_prompt(question, sources_block, history_block)
    valid_indices = {item.source_index for item in retrieval.chunks}

    try:
        result = client.structured(
            system=ANSWER_SYSTEM_PROMPT,
            user=prompt,
            schema=ANSWER_SCHEMA,
            model=model,
            max_tokens=max_tokens,
            effort="medium",
        )
    except LLMRefusal as exc:
        logger.warning("Generation refused by safety classifiers: %s", exc)
        return GeneratedAnswer(
            answerable=False,
            claims=[],
            refusal_reason=(
                "The model declined to answer this request "
                f"({exc.category or 'unspecified category'})."
            ),
            mode="generative",
        )
    except Exception as exc:
        logger.exception("Generation failed")
        return GeneratedAnswer(
            answerable=False,
            claims=[],
            refusal_reason=f"Answer generation failed: {exc}",
            mode="generative",
        )

    data = result.data
    claims = _coerce_claims(data.get("claims"), valid_indices)
    answerable = bool(data.get("answerable")) and bool(claims)

    return GeneratedAnswer(
        answerable=answerable,
        claims=claims,
        refusal_reason=(data.get("refusal_reason") or "").strip(),
        caveats=[c for c in (data.get("caveats") or []) if isinstance(c, str) and c.strip()],
        followups=[f for f in (data.get("followups") or []) if isinstance(f, str) and f.strip()][:3],
        mode="generative",
        model=result.model,
        usage=result.usage,
        latency_ms=result.latency_ms,
    )
