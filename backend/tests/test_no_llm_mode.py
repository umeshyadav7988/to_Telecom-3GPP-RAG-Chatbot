"""The no-LLM path: deterministic false-premise detection and answer extraction.

These two components are what make the system usable with no API key at all.
Measured on the golden set they take the no-model configuration from 64% to 84%
overall, and abstention recall from 43% to 86%.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunking import chunk_document
from app.rag.extractive import extract_units, render_unit
from app.rag.loaders import LoadedDocument
from app.rag.premise_guard import check_premise, extract_asserted_entities
from app.rag.reranker import LexicalSemanticReranker
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import HybridIndex

SPEC = """3GPP TS 23.501 V17.9.0
System architecture for the 5G System

5.7 QoS model

5.7.4 Standardized 5QI to QoS characteristics mapping
The following table specifies the standardised 5QI values.

5QI Value | Resource Type | Default Priority Level | Packet Delay Budget | Packet Error Rate
1 | GBR | 20 | 100 ms | 10^-2
2 | GBR | 40 | 150 ms | 10^-3
82 | Delay-critical GBR | 19 | 10 ms | 10^-4

5.15 Network slicing

5.15.2.2 Standardised SST values
The following SST values are standardised:
SST value 1: eMBB.
SST value 2: URLLC.
SST value 3: MIoT.

Annex C (normative): Protection schemes

C.3.4 Profile A
Profile A uses Curve25519 for the elliptic curve Diffie-Hellman primitive.

C.3.4a Profile B
Profile B uses the elliptic curve secp256r1 for the Diffie-Hellman primitive.
"""


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    doc = LoadedDocument(
        source_path="/tmp/23501.txt",
        filename="23501.txt",
        text=SPEC,
        doc_id="TS 23.501",
        title="System architecture for the 5G System",
        version="17.9.0",
        release="17",
    )
    index = HybridIndex(tmp_path_factory.mktemp("idx"))
    index.build(chunk_document(doc, min_chars=10))
    return HybridRetriever(index, LexicalSemanticReranker(), top_k=20, rerank_top_n=6, min_score=0.1)


def _answer(retriever, question: str) -> str:
    result = retriever.retrieve(question)
    return "\n".join(render_unit(u) for u in extract_units(question, result.chunks))


# --- chunking regression ----------------------------------------------------

def test_table_rows_are_not_parsed_as_clause_headings(retriever):
    """Regression: `1 | GBR | 20 | 100 ms` once parsed as clause "1".

    Every numeric table row matched the clause-heading grammar, so each became
    its own tiny section and was then dropped by the minimum-length filter.
    The effect was silent: retrieval still returned the table's *header* chunk,
    so clause-level metrics looked perfect while the actual values had been
    deleted from the index.
    """
    bodies = "\n".join(c.body for c in retriever.index.chunks)
    assert "1 | GBR | 20 | 100 ms" in bodies
    assert "82 | Delay-critical GBR" in bodies
    assert not any(c.clause_id == "82" for c in retriever.index.chunks)


# --- false-premise guard ----------------------------------------------------

def test_extracts_checkable_entities_from_a_question():
    entities = extract_asserted_entities("What is the default value of timer T3599?")
    assert any(e["token"] == "T3599" for e in entities)


def test_ignores_pure_acronyms_and_single_digits():
    """`AMF` may be spelled out in the clause; `1` is far too common."""
    tokens = {e["token"].upper() for e in extract_asserted_entities("Does the AMF handle 5QI 1?")}
    assert "AMF" not in tokens
    assert "1" not in tokens


def test_flags_a_nonexistent_timer():
    verdict = check_premise(
        "What is the default value of timer T3599?",
        "The timer T3512 has a default value of 54 minutes.",
    )
    assert verdict.passed is False
    assert verdict.missing[0]["token"] == "T3599"
    assert "T3599" in verdict.describe()


def test_flags_a_nonexistent_qos_index():
    verdict = check_premise(
        "What is the Packet Delay Budget for 5QI 91?",
        "5QI Value | Resource Type\n1 | GBR | 100 ms\n82 | Delay-critical GBR | 10 ms",
    )
    assert verdict.passed is False
    assert any(m["token"] == "91" for m in verdict.missing)


def test_flags_an_uningested_specification():
    verdict = check_premise(
        "What does TS 24.601 say about registration?",
        "See TS 24.501 clause 5.5.1 for the registration procedure.",
    )
    assert verdict.passed is False
    assert any("24.601" in m["token"] for m in verdict.missing)


def test_passes_a_real_identifier():
    verdict = check_premise(
        "What is the default value of timer T3512?",
        "The timer T3512 has a default value of 54 minutes.",
    )
    assert verdict.passed is True


def test_passes_a_question_with_no_checkable_entities():
    verdict = check_premise("What are the RRC states?", "RRC_IDLE, RRC_INACTIVE and RRC_CONNECTED.")
    assert verdict.passed is True


def test_cannot_catch_a_false_causal_premise():
    """The honest limit of a deterministic guard.

    Every entity in this question is real; only the asserted relationship
    between them is false. Catching it needs entailment, which is exactly the
    residue that justifies keeping a model in the pipeline.
    """
    verdict = check_premise(
        "Why did 3GPP remove network slicing in Release 17?",
        "A Network Slice is defined within a PLMN. Release 17 adds SST value 5.",
    )
    assert verdict.passed is True


# --- extraction -------------------------------------------------------------

def test_table_lookup_returns_the_row_and_its_header(retriever):
    """The row alone (`1 | GBR | 20 | 100 ms`) is unreadable without the header."""
    answer = _answer(retriever, "What is the Packet Delay Budget for 5QI 1?")
    assert "100 ms" in answer
    assert "Packet Delay Budget" in answer   # the header came along
    assert "150 ms" not in answer            # the 5QI 2 row did not


def test_list_continuation_pulls_in_the_enumerated_values(retriever):
    """"The following SST values are standardised:" is only half an answer."""
    answer = _answer(retriever, "What are the standardised SST values?")
    assert "eMBB" in answer
    assert "URLLC" in answer


def test_qualifier_conflict_keeps_the_sibling_clause_out(retriever):
    """Profile A and Profile B differ by one character that tokenizers discard."""
    answer = _answer(retriever, "Which elliptic curve does ECIES Profile A use?")
    assert "Curve25519" in answer
    assert "secp256r1" not in answer


def test_extraction_is_far_smaller_than_the_raw_chunks(retriever):
    """The point of extraction: return the fact, not the page."""
    result = retriever.retrieve("What is the Packet Delay Budget for 5QI 1?")
    raw = sum(len(item.chunk.body) for item in result.chunks)
    extracted = sum(len(u.text) for u in extract_units("What is the Packet Delay Budget for 5QI 1?", result.chunks))
    assert extracted < raw


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
