"""End-to-end RAG orchestration.

    question
       -> contextualise (resolve follow-up references)
       -> hybrid retrieve (dense + BM25 -> RRF -> rerank)
       -> GATE 1: retrieval score below threshold?  -> abstain, no LLM call
       -> grounded generation (claims + citations + verbatim quotes)
       -> GATE 2: model itself says the context is insufficient? -> abstain
       -> verification (citations, quotes, numerics, entailment)
       -> GATE 3: too little of the answer survived? -> abstain
       -> assemble cited answer + confidence

Three independent abstention gates, each cheaper than the one after it. The
first one is free.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

from ..utils.text import split_sentences
from .generator import GeneratedAnswer, generate_answer
from .llm import BaseLLMClient
from .premise_guard import check_premise
from .prompts import REWRITE_SCHEMA, REWRITE_SYSTEM_PROMPT
from .retriever import HybridRetriever, RetrievalResult
from .verifier import VerificationReport, verify

logger = logging.getLogger(__name__)


ABSTENTION_TEMPLATES = {
    "empty_index": (
        "No specifications are indexed yet. Ingest 3GPP documents into the corpus "
        "directory and rebuild the index before asking questions."
    ),
    "no_relevant_context": (
        "I could not find this in the indexed 3GPP specifications, so I am not going "
        "to answer from memory."
    ),
    "false_premise": (
        "Your question assumes something the specifications do not contain, so there "
        "is no correct answer for me to give."
    ),
    "insufficient_context": (
        "The clauses I retrieved are related to your question but do not actually "
        "contain the answer, so I am not going to infer one."
    ),
    "verification_failed": (
        "I drafted an answer, but too little of it survived verification against the "
        "cited clauses to be trustworthy, so I am withholding it."
    ),
}


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        client: BaseLLMClient | None = None,
        answer_model: str = "claude-opus-5",
        verifier_model: str = "claude-opus-5",
        rewrite_model: str = "claude-opus-5",
        enable_verifier: bool = True,
        enable_numeric_guard: bool = True,
        enable_premise_guard: bool = True,
        min_support_ratio: float = 0.6,
        max_answer_tokens: int = 8000,
        max_verifier_tokens: int = 6000,
    ):
        self.retriever = retriever
        self.client = client
        self.answer_model = answer_model
        self.verifier_model = verifier_model
        self.rewrite_model = rewrite_model
        self.enable_verifier = enable_verifier
        self.enable_numeric_guard = enable_numeric_guard
        self.enable_premise_guard = enable_premise_guard
        self.min_support_ratio = min_support_ratio
        self.max_answer_tokens = max_answer_tokens
        self.max_verifier_tokens = max_verifier_tokens

    # ------------------------------------------------------------------
    # Query contextualisation
    # ------------------------------------------------------------------

    @staticmethod
    def _history_block(history: list[dict], max_turns: int = 6) -> str:
        if not history:
            return ""
        recent = history[-max_turns:]
        lines = []
        for turn in recent:
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = (turn.get("content") or "").strip()
            if len(content) > 600:
                content = content[:600] + " ..."
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def contextualise(self, question: str, history: list[dict]) -> tuple[str, bool]:
        """Rewrite a follow-up into a standalone retrieval query.

        "What about for URLLC?" retrieves nothing useful on its own. Without
        this step, follow-up turns fall through the retrieval gate and the
        assistant abstains on questions it can answer perfectly well.
        """
        if not history or self.client is None:
            return question, False

        # Cheap heuristic first: a question that already contains a domain noun
        # and no dangling reference is almost certainly standalone.
        lowered = question.lower().strip()
        has_reference = any(
            token in lowered
            for token in (" it", "it ", "its ", "that", "those", "these", "them",
                          "the same", "what about", "and for", "instead", "he ", "they ")
        ) or lowered.startswith(("what about", "and ", "also ", "why", "how about"))
        if not has_reference and len(lowered.split()) >= 5:
            return question, False

        try:
            result = self.client.structured(
                system=REWRITE_SYSTEM_PROMPT,
                user=(
                    f"Conversation so far:\n{self._history_block(history)}\n\n"
                    f"Latest user message:\n{question}"
                ),
                schema=REWRITE_SCHEMA,
                model=self.rewrite_model,
                max_tokens=1500,
                effort="low",
            )
            rewritten = (result.data.get("standalone_query") or "").strip()
            if rewritten and rewritten.lower() != question.lower():
                logger.info("Query rewritten: %r -> %r", question, rewritten)
                return rewritten, True
        except Exception as exc:
            logger.warning("Query rewrite failed, using original question: %s", exc)

        return question, False

    # ------------------------------------------------------------------
    # Response assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _render_answer(report: VerificationReport) -> str:
        """Concatenate surviving claims into cited prose."""
        parts: list[str] = []
        for claim in report.claims:
            if claim.status == "removed":
                continue
            markers = "".join(f"[S{c}]" for c in claim.citations)
            text = claim.text.rstrip()
            if text and text[-1] not in ".!?:;":
                text += "."
            parts.append(f"{text} {markers}".strip())
        return " ".join(parts)

    def _abstention_payload(
        self,
        kind: str,
        retrieval: RetrievalResult,
        *,
        detail: str = "",
    ) -> dict:
        message = ABSTENTION_TEMPLATES.get(kind, ABSTENTION_TEMPLATES["no_relevant_context"])
        if detail:
            message = f"{message}\n\n{detail}"

        nearest = [
            {
                "citation_label": item.chunk.citation_label,
                "score": round(item.score, 4),
            }
            for item in retrieval.chunks[:3]
        ]
        if nearest:
            listed = "; ".join(f"{n['citation_label']} (score {n['score']})" for n in nearest)
            message += (
                f"\n\nClosest clauses considered: {listed}. "
                "Try naming the specification or clause you have in mind, or rephrase "
                "using the standard's terminology."
            )
        else:
            message += (
                "\n\nNothing in the index came close. Check that the relevant "
                "specification has been ingested."
            )

        return {
            "type": kind,
            "message": message,
            "nearest_clauses": nearest,
            "top_score": round(retrieval.top_score, 4),
            "threshold": round(retrieval.gate_threshold, 4),
        }

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run(self, question: str, history: list[dict] | None = None) -> dict:
        """Blocking variant. Returns the same payload the stream ends with."""
        final: dict = {}
        for event in self.run_stream(question, history or []):
            if event["event"] == "result":
                final = event["data"]
        return final

    def run_stream(self, question: str, history: list[dict] | None = None) -> Iterator[dict]:
        """Yield pipeline stage events, ending with a `result` event.

        Stages are streamed rather than just the final text because the
        pipeline's trustworthiness *is* the product. A user who watches
        "retrieving -> 6 clauses -> verifying -> 1 claim dropped" understands
        why they should believe the answer.
        """
        history = history or []
        started = time.perf_counter()
        timings: dict[str, float] = {}

        # --- Stage 1: contextualise ------------------------------------
        yield {"event": "stage", "data": {"stage": "contextualising"}}
        t0 = time.perf_counter()
        search_query, rewritten = self.contextualise(question, history)
        timings["rewrite_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if rewritten:
            yield {
                "event": "query_rewritten",
                "data": {"original": question, "standalone": search_query},
            }

        # --- Stage 2: retrieve -----------------------------------------
        yield {"event": "stage", "data": {"stage": "retrieving"}}
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(search_query)
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        timings.update(retrieval.timings_ms)

        retrieval_payload = {
            "query": question,
            "search_query": search_query,
            "was_rewritten": rewritten,
            "candidate_count": retrieval.candidate_count,
            "top_score": round(retrieval.top_score, 4),
            "gate_threshold": round(retrieval.gate_threshold, 4),
            "passed_gate": retrieval.passed_gate,
            "filters_applied": retrieval.filters_applied,
            "retriever": {
                "reranker": self.retriever.reranker.name,
                "top_k": self.retriever.top_k,
                "rerank_top_n": self.retriever.rerank_top_n,
            },
        }
        sources = [item.to_dict() for item in retrieval.chunks]
        yield {"event": "sources", "data": {"sources": sources, "retrieval": retrieval_payload}}

        # --- GATE 1: nothing relevant ----------------------------------
        if not self.retriever.index.is_ready:
            payload = self._build_payload(
                question, "abstained", "",
                abstention=self._abstention_payload("empty_index", retrieval),
                sources=sources, retrieval=retrieval_payload, timings=timings, started=started,
            )
            yield {"event": "result", "data": payload}
            return

        if not retrieval.passed_gate:
            yield {"event": "stage", "data": {"stage": "abstaining", "gate": "retrieval"}}
            payload = self._build_payload(
                question, "abstained", "",
                abstention=self._abstention_payload("no_relevant_context", retrieval),
                sources=sources, retrieval=retrieval_payload, timings=timings, started=started,
            )
            yield {"event": "result", "data": payload}
            return

        # --- GATE 1.5: false premise (deterministic) -------------------
        # Catches questions whose topic is in the corpus but whose asserted
        # entity is not ("timer T3599", "5QI 91"). Retrieval scores these
        # highly by construction, so no threshold separates them — but a
        # string comparison does, without spending a generation token.
        if self.enable_premise_guard:
            evidence = "\n".join(item.chunk.body for item in retrieval.chunks)
            premise = check_premise(search_query, evidence)
            if not premise.passed:
                yield {"event": "stage", "data": {"stage": "abstaining", "gate": "premise"}}
                retrieval_payload["premise_check"] = {
                    "passed": False,
                    "missing": premise.missing,
                    "checked": premise.checked,
                }
                payload = self._build_payload(
                    question, "abstained", "",
                    abstention=self._abstention_payload(
                        "false_premise", retrieval, detail=premise.describe()
                    ),
                    sources=sources, retrieval=retrieval_payload, timings=timings,
                    started=started,
                )
                yield {"event": "result", "data": payload}
                return
            retrieval_payload["premise_check"] = {
                "passed": True,
                "checked": premise.checked,
            }

        # --- Stage 3: generate -----------------------------------------
        yield {"event": "stage", "data": {"stage": "generating"}}
        t0 = time.perf_counter()
        answer: GeneratedAnswer = generate_answer(
            question,
            retrieval,
            client=self.client,
            history_block=self._history_block(history),
            model=self.answer_model,
            max_tokens=self.max_answer_tokens,
        )
        timings["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # --- GATE 2: model declares insufficient context ----------------
        if not answer.answerable:
            yield {"event": "stage", "data": {"stage": "abstaining", "gate": "generation"}}
            payload = self._build_payload(
                question, "abstained", "",
                abstention=self._abstention_payload(
                    "insufficient_context", retrieval, detail=answer.refusal_reason
                ),
                sources=sources, retrieval=retrieval_payload, timings=timings, started=started,
                mode=answer.mode, usage={"generation": answer.usage},
            )
            yield {"event": "result", "data": payload}
            return

        yield {
            "event": "draft",
            "data": {
                "claims": [
                    {"index": c.index, "text": c.text, "citations": c.citations}
                    for c in answer.claims
                ]
            },
        }

        # --- Stage 4: verify -------------------------------------------
        yield {"event": "stage", "data": {"stage": "verifying"}}
        t0 = time.perf_counter()
        report = verify(
            answer,
            retrieval,
            client=self.client,
            enable_verifier=self.enable_verifier,
            enable_numeric_guard=self.enable_numeric_guard,
            verifier_model=self.verifier_model,
            max_tokens=self.max_verifier_tokens,
        )
        timings["verification_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # --- GATE 3: too little survived -------------------------------
        surviving = [c for c in report.claims if c.status != "removed"]
        if not surviving or report.support_ratio < self.min_support_ratio:
            yield {"event": "stage", "data": {"stage": "abstaining", "gate": "verification"}}
            detail = (
                f"{report.removed_count} of {len(report.claims)} drafted claims failed "
                f"verification against the cited clauses (support ratio "
                f"{report.support_ratio:.2f}, minimum {self.min_support_ratio:.2f})."
            )
            payload = self._build_payload(
                question, "abstained", "",
                abstention=self._abstention_payload(
                    "verification_failed", retrieval, detail=detail
                ),
                sources=sources, retrieval=retrieval_payload, timings=timings, started=started,
                report=report, mode=answer.mode,
                usage={"generation": answer.usage, "verification": report.usage},
                caveats=answer.caveats,
            )
            yield {"event": "result", "data": payload}
            return

        # --- Success ----------------------------------------------------
        answer_text = self._render_answer(report)
        status = "answered" if report.flagged_count == 0 else "answered_with_flags"

        payload = self._build_payload(
            question, status, answer_text,
            sources=sources, retrieval=retrieval_payload, timings=timings, started=started,
            report=report, mode=answer.mode,
            usage={"generation": answer.usage, "verification": report.usage},
            caveats=answer.caveats, followups=answer.followups,
        )
        yield {"event": "result", "data": payload}

    # ------------------------------------------------------------------

    def _build_payload(
        self,
        question: str,
        status: str,
        answer_text: str,
        *,
        sources: list,
        retrieval: dict,
        timings: dict,
        started: float,
        abstention: dict | None = None,
        report: VerificationReport | None = None,
        mode: str = "generative",
        usage: dict | None = None,
        caveats: list | None = None,
        followups: list | None = None,
    ) -> dict:
        timings = dict(timings)
        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)

        # Only sources actually cited by a surviving claim are "used"; the rest
        # were considered and are shown separately so retrieval stays auditable.
        cited: set[int] = set()
        if report:
            for claim in report.claims:
                if claim.status != "removed":
                    cited.update(claim.citations)
        for source in sources:
            source["was_cited"] = source["source_index"] in cited

        payload = {
            "question": question,
            "status": status,
            "answer": answer_text,
            "mode": mode,
            "sources": sources,
            "retrieval": retrieval,
            "timings_ms": timings,
            "usage": usage or {},
            "caveats": caveats or [],
            "followups": followups or [],
            "sentences": split_sentences(answer_text) if answer_text else [],
        }

        if abstention:
            payload["abstention"] = abstention
            payload["answer"] = abstention["message"]
            payload["confidence"] = {"score": 0.0, "label": "abstained"}
            payload["claims"] = [c.to_dict() for c in (report.claims if report else [])]
            payload["verification"] = (
                {
                    "support_ratio": report.support_ratio,
                    "accepted": report.accepted_count,
                    "flagged": report.flagged_count,
                    "removed": report.removed_count,
                    "verifier_ran": report.verifier_ran,
                    "checks": report.checks,
                    "notes": report.notes,
                }
                if report
                else {"checks": {}, "notes": []}
            )
            return payload

        assert report is not None
        payload["abstention"] = None
        payload["confidence"] = {
            "score": report.confidence,
            "label": report.confidence_label,
        }
        payload["claims"] = [c.to_dict() for c in report.claims]
        payload["verification"] = {
            "support_ratio": report.support_ratio,
            "accepted": report.accepted_count,
            "flagged": report.flagged_count,
            "removed": report.removed_count,
            "verifier_ran": report.verifier_ran,
            "checks": report.checks,
            "notes": report.notes,
        }
        return payload
