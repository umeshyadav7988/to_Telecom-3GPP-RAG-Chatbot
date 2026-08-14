"""End-to-end guardrail tests using a scripted (deliberately misbehaving) LLM.

These are the tests that matter. The pipeline's claim is "near-zero
hallucinations", and the only way to demonstrate that without an API key is to
substitute a model that hallucinates *on purpose* and assert that each defence
catches its specific failure mode:

    fabricated number      -> numeric guard          -> claim flagged
    fabricated quote       -> quote provenance       -> claim flagged
    invented citation [S99]-> citation validation    -> claim removed
    unentailed claim       -> LLM entailment         -> claim removed
    everything unsupported -> gate 3                 -> whole answer withheld
    model admits ignorance -> gate 2                 -> abstain before verify

A real model fails these ways occasionally. A scripted one fails them every
time, which is exactly what a regression test needs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.rag.chunking import chunk_document
from app.rag.llm import LLMResult
from app.rag.loaders import LoadedDocument
from app.rag.pipeline import RAGPipeline
from app.rag.reranker import LexicalSemanticReranker
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import HybridIndex

SPEC = """3GPP TS 24.501 V17.9.0
Non-Access-Stratum (NAS) protocol for 5G System

10 Timer handling

10.2 Timers of 5GS mobility management

10.2.1 Timers of 5GS mobility management in the UE
The timer T3512 has a default value of 54 minutes. It is started in
5GMM-REGISTERED when the UE enters 5GMM-IDLE mode, and on expiry the UE
initiates the periodic registration update procedure.

10.2.2 Timers of 5GS mobility management in the network
The timer T3550 has a default value of 6 seconds. It is started when the
REGISTRATION ACCEPT message containing a 5G-GUTI is sent, and is stopped on
receipt of the REGISTRATION COMPLETE message.
"""


# ---------------------------------------------------------------------------
# Scripted LLM
# ---------------------------------------------------------------------------

class ScriptedClient:
    """Stands in for AnthropicClient, returning whatever the test dictates."""

    def __init__(self, answer: dict, verdicts: list[dict] | None = None):
        self._answer = answer
        self._verdicts = verdicts
        self.calls: list[str] = []

    def structured(self, *, system, user, schema, model=None, max_tokens=0, effort="medium", **_):
        properties = schema.get("properties", {})

        if "standalone_query" in properties:
            self.calls.append("rewrite")
            data = {"standalone_query": "", "changed": False}
        elif "verdicts" in properties:
            self.calls.append("verify")
            if self._verdicts is None:
                # Default: agree with whatever the generator said.
                data = {
                    "verdicts": [
                        {
                            "claim_index": i + 1,
                            "verdict": "supported",
                            "confidence": 0.95,
                            "reason": "Entailed by the cited clause.",
                        }
                        for i in range(len(self._answer.get("claims", [])))
                    ]
                }
            else:
                data = {"verdicts": self._verdicts}
        else:
            self.calls.append("answer")
            data = self._answer

        return LLMResult(data=data, raw_text="", model=model or "scripted", usage={})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def index(tmp_path_factory) -> HybridIndex:
    doc = LoadedDocument(
        source_path="/tmp/24501.txt",
        filename="24501.txt",
        text=SPEC,
        doc_id="TS 24.501",
        title="NAS protocol for 5G System",
        version="17.9.0",
        release="17",
    )
    store = HybridIndex(tmp_path_factory.mktemp("index"))
    store.build(chunk_document(doc, min_chars=10))
    return store


def make_pipeline(index: HybridIndex, client, **kwargs) -> RAGPipeline:
    retriever = HybridRetriever(
        index,
        LexicalSemanticReranker(),
        top_k=10,
        rerank_top_n=3,
        min_score=kwargs.pop("min_score", 0.2),
    )
    return RAGPipeline(retriever, client=client, **kwargs)


QUESTION = "What is the default value of timer T3512?"


def answer_with(text: str, quote: str, citations=(1,)) -> dict:
    return {
        "answerable": True,
        "refusal_reason": "",
        "claims": [
            {
                "text": text,
                "citations": list(citations),
                "quote": quote,
                "modality": "descriptive",
            }
        ],
        "caveats": [],
        "followups": [],
    }


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def test_faithful_answer_is_accepted(index):
    client = ScriptedClient(
        answer_with(
            "The timer T3512 has a default value of 54 minutes.",
            "The timer T3512 has a default value of 54 minutes",
        )
    )
    result = make_pipeline(index, client).run(QUESTION)

    assert result["status"] == "answered"
    assert result["verification"]["removed"] == 0
    assert result["verification"]["flagged"] == 0
    assert "[S1]" in result["answer"]
    assert result["confidence"]["score"] > 0.6


# ---------------------------------------------------------------------------
# Guard 1: numeric fabrication
# ---------------------------------------------------------------------------

def test_fabricated_number_is_flagged(index):
    """The model copies a real quote but states a number that is not in it."""
    client = ScriptedClient(
        answer_with(
            "The timer T3512 has a default value of 42 minutes.",
            "The timer T3512 has a default value of 54 minutes",
        )
    )
    result = make_pipeline(index, client).run(QUESTION)

    claim = result["claims"][0]
    assert claim["status"] == "flagged"
    assert "unverified_values" in claim["issues"]
    unsupported = claim["verification"]["numeric_guard"]["unsupported_tokens"]
    assert any(u["token"] == "42minutes" for u in unsupported)
    assert result["status"] == "answered_with_flags"


def test_numeric_guard_can_be_disabled(index):
    client = ScriptedClient(
        answer_with(
            "The timer T3512 has a default value of 42 minutes.",
            "The timer T3512 has a default value of 54 minutes",
        )
    )
    result = make_pipeline(index, client, enable_numeric_guard=False).run(QUESTION)
    assert result["claims"][0]["verification"]["numeric_guard"] == {}


# ---------------------------------------------------------------------------
# Guard 2: quote provenance
# ---------------------------------------------------------------------------

def test_fabricated_quote_is_flagged(index):
    """A quote that does not occur in the cited clause is invented evidence."""
    client = ScriptedClient(
        answer_with(
            "The timer T3512 has a default value of 54 minutes.",
            "T3512 is defined in ITU-T Recommendation Q.700 as a supervision timer",
        )
    )
    result = make_pipeline(index, client).run(QUESTION)

    claim = result["claims"][0]
    assert claim["verification"]["quote_found_in_source"] is False
    assert "quote_not_found_in_cited_source" in claim["issues"]
    assert claim["status"] == "flagged"


# ---------------------------------------------------------------------------
# Guard 3: citation validity
# ---------------------------------------------------------------------------

def test_citation_to_a_nonexistent_source_is_dropped(index):
    """[S99] was never retrieved; a claim resting only on it cannot stand."""
    client = ScriptedClient(
        answer_with(
            "The timer T3512 has a default value of 54 minutes.",
            "The timer T3512 has a default value of 54 minutes",
            citations=(99,),
        )
    )
    result = make_pipeline(index, client).run(QUESTION)

    assert result["status"] == "abstained"
    assert result["abstention"]["type"] == "verification_failed"
    claim = result["claims"][0]
    assert claim["citations"] == []
    assert 99 in claim["verification"]["invalid_citations"]
    assert claim["status"] == "removed"


# ---------------------------------------------------------------------------
# Guard 4: LLM entailment
# ---------------------------------------------------------------------------

def test_unentailed_claim_is_removed_by_the_verifier(index):
    """Every deterministic check passes; only entailment catches this one.

    The claim reuses real tokens from the source ("T3512", "54 minutes") but
    asserts something the clause never says, so the numeric guard and quote
    check both pass. This is the case that justifies the extra LLM call.
    """
    client = ScriptedClient(
        answer_with(
            "The timer T3512 must be set to 54 minutes by every operator worldwide.",
            "The timer T3512 has a default value of 54 minutes",
        ),
        verdicts=[
            {
                "claim_index": 1,
                "verdict": "unsupported",
                "confidence": 0.9,
                "reason": "The clause gives a default value; it does not mandate it globally.",
            }
        ],
    )
    result = make_pipeline(index, client).run(QUESTION)

    claim = result["claims"][0]
    assert claim["status"] == "removed"
    assert "entailment_unsupported" in claim["issues"]
    assert result["status"] == "abstained"


def test_partial_support_flags_rather_than_removes(index):
    client = ScriptedClient(
        answer_with(
            "The timer T3512 has a default value of 54 minutes.",
            "The timer T3512 has a default value of 54 minutes",
        ),
        verdicts=[
            {
                "claim_index": 1,
                "verdict": "partially_supported",
                "confidence": 0.6,
                "reason": "The value is stated but the applicable state is not.",
            }
        ],
    )
    result = make_pipeline(index, client).run(QUESTION)
    assert result["claims"][0]["status"] == "flagged"
    assert result["status"] == "answered_with_flags"


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def test_gate_1_abstains_before_calling_the_llm(index):
    """An off-corpus question must cost zero generation tokens."""
    client = ScriptedClient(answer_with("Anything at all.", "Anything at all."))
    pipeline = make_pipeline(index, client, min_score=0.99)  # nothing can pass

    result = pipeline.run("What is the E2 service model in the O-RAN architecture?")

    assert result["status"] == "abstained"
    assert result["abstention"]["type"] == "no_relevant_context"
    assert "answer" not in client.calls, "The generator must not run after gate 1"


def test_gate_1_5_catches_a_false_premise_without_any_model(index):
    """`T3599` does not exist, but the timer table retrieves strongly for it.

    Retrieval score cannot separate this from a real timer question — measured
    on the golden set it scores above four genuinely answerable questions. The
    deterministic entity check is what catches it, and it does so before the
    generator runs.
    """
    client = ScriptedClient(answer_with("T3599 defaults to 54 minutes.", "54 minutes"))
    result = make_pipeline(index, client).run("What is the default value of timer T3599?")

    assert result["status"] == "abstained"
    assert result["abstention"]["type"] == "false_premise"
    assert "T3599" in result["answer"]
    assert "answer" not in client.calls, "The generator must not run after gate 1.5"


def test_premise_guard_lets_a_real_identifier_through(index):
    """The guard must not fire on T3512, which is genuinely in the corpus."""
    client = ScriptedClient(
        answer_with(
            "The timer T3512 has a default value of 54 minutes.",
            "The timer T3512 has a default value of 54 minutes",
        )
    )
    result = make_pipeline(index, client).run("What is the default value of timer T3512?")

    assert result["status"] == "answered"
    assert result["retrieval"]["premise_check"]["passed"] is True


def test_gate_2_abstains_when_the_model_reports_insufficient_context(index):
    """Gate 2 in isolation: premise guard off, so the model's own call decides."""
    client = ScriptedClient(
        {
            "answerable": False,
            "refusal_reason": "The excerpts describe T3512 and T3550 but never mention T3599.",
            "claims": [],
            "caveats": [],
            "followups": [],
        }
    )
    pipeline = make_pipeline(index, client, enable_premise_guard=False)
    result = pipeline.run("What is the default value of timer T3599?")

    assert result["status"] == "abstained"
    assert result["abstention"]["type"] == "insufficient_context"
    assert "T3599" in result["answer"]
    assert "verify" not in client.calls, "Verification must not run on an abstention"


def test_gate_3_withholds_when_too_little_survives(index):
    """Two of three claims fail entailment -> below min_support_ratio -> withhold."""
    client = ScriptedClient(
        {
            "answerable": True,
            "refusal_reason": "",
            "claims": [
                {
                    "text": "The timer T3512 has a default value of 54 minutes.",
                    "citations": [1],
                    "quote": "The timer T3512 has a default value of 54 minutes",
                    "modality": "descriptive",
                },
                {
                    "text": "The timer T3512 applies only to roaming subscribers.",
                    "citations": [1],
                    "quote": "The timer T3512 has a default value of 54 minutes",
                    "modality": "descriptive",
                },
                {
                    "text": "The timer T3512 was introduced in Release 15.",
                    "citations": [1],
                    "quote": "The timer T3512 has a default value of 54 minutes",
                    "modality": "descriptive",
                },
            ],
            "caveats": [],
            "followups": [],
        },
        verdicts=[
            {"claim_index": 1, "verdict": "supported", "confidence": 0.95, "reason": "Stated."},
            {"claim_index": 2, "verdict": "unsupported", "confidence": 0.9, "reason": "Not stated."},
            {"claim_index": 3, "verdict": "unsupported", "confidence": 0.9, "reason": "Not stated."},
        ],
    )
    result = make_pipeline(index, client, min_support_ratio=0.6).run(QUESTION)

    assert result["status"] == "abstained"
    assert result["abstention"]["type"] == "verification_failed"
    assert result["verification"]["removed"] == 2


def test_confidence_falls_when_claims_are_removed(index):
    """Confidence must track what survived, not what was drafted."""
    faithful = ScriptedClient(
        answer_with(
            "The timer T3512 has a default value of 54 minutes.",
            "The timer T3512 has a default value of 54 minutes",
        )
    )
    clean = make_pipeline(index, faithful).run(QUESTION)

    degraded = ScriptedClient(
        {
            "answerable": True,
            "refusal_reason": "",
            "claims": [
                {
                    "text": "The timer T3512 has a default value of 54 minutes.",
                    "citations": [1],
                    "quote": "The timer T3512 has a default value of 54 minutes",
                    "modality": "descriptive",
                },
                {
                    "text": "The timer T3512 is negotiated per PDU session.",
                    "citations": [1],
                    "quote": "The timer T3512 has a default value of 54 minutes",
                    "modality": "descriptive",
                },
            ],
            "caveats": [],
            "followups": [],
        },
        verdicts=[
            {"claim_index": 1, "verdict": "supported", "confidence": 0.95, "reason": "Stated."},
            {"claim_index": 2, "verdict": "unsupported", "confidence": 0.9, "reason": "Not stated."},
        ],
    )
    partial = make_pipeline(index, degraded, min_support_ratio=0.4).run(QUESTION)

    assert partial["status"] != "abstained"
    assert partial["confidence"]["score"] < clean["confidence"]["score"]


# ---------------------------------------------------------------------------
# Sources reporting
# ---------------------------------------------------------------------------

def test_retrieved_but_uncited_sources_are_marked(index):
    """Retrieval stays auditable: what was considered vs what was used."""
    client = ScriptedClient(
        answer_with(
            "The timer T3512 has a default value of 54 minutes.",
            "The timer T3512 has a default value of 54 minutes",
        )
    )
    result = make_pipeline(index, client).run(QUESTION)

    assert any(s["was_cited"] for s in result["sources"])
    assert all("citation_label" in s for s in result["sources"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
