"""Clause-aware chunking tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunking import chunk_document
from app.rag.loaders import LoadedDocument

SPEC = """3GPP TS 23.501 V17.9.0
System architecture for the 5G System

5 Overall description

5.15 Network slicing

5.15.1 General concepts
A Network Slice is defined within a PLMN and shall include the Core Network
Control Plane and User Plane Network Functions.

5.15.2 Identification of a Network Slice

5.15.2.1 S-NSSAI
An S-NSSAI identifies a Network Slice. The SST field has a length of 8 bits and
the SD field has a length of 24 bits.

Annex C (normative): Protection schemes

C.3 Elliptic Curve Integrated Encryption Scheme

C.3.4 Profile A
Profile A uses Curve25519 for the elliptic curve Diffie-Hellman primitive.
"""


def _doc(text: str = SPEC) -> LoadedDocument:
    return LoadedDocument(
        source_path="/tmp/23501.txt",
        filename="23501.txt",
        text=text,
        doc_id="TS 23.501",
        title="System architecture for the 5G System",
        version="17.9.0",
        release="17",
    )


def test_detects_nested_clause_numbers():
    chunks = chunk_document(_doc(), min_chars=10)
    clause_ids = {c.clause_id for c in chunks}
    assert "5.15.1" in clause_ids
    assert "5.15.2.1" in clause_ids


def test_annex_subclauses_are_detected():
    """Regression: TS 33.501 keeps SUPI protection entirely inside Annex C.

    Without lettered-clause support the whole annex collapses into one chunk
    and citations degrade from '§C.3.4' to '§Annex C'.
    """
    chunks = chunk_document(_doc(), min_chars=10)
    clause_ids = {c.clause_id for c in chunks}
    assert "C.3.4" in clause_ids

    profile_a = next(c for c in chunks if c.clause_id == "C.3.4")
    assert "Curve25519" in profile_a.body
    assert profile_a.clause_title == "Profile A"


def test_breadcrumb_carries_ancestors():
    chunks = chunk_document(_doc(), min_chars=10)
    snssai = next(c for c in chunks if c.clause_id == "5.15.2.1")
    assert "5 Overall description" in snssai.breadcrumb
    assert "5.15 Network slicing" in snssai.breadcrumb


def test_citation_label_is_engineer_checkable():
    chunks = chunk_document(_doc(), min_chars=10)
    snssai = next(c for c in chunks if c.clause_id == "5.15.2.1")
    assert snssai.citation_label == "TS 23.501 §5.15.2.1 — S-NSSAI"


def test_breadcrumb_is_prefixed_into_indexed_text_but_not_body():
    """The embedder needs clause context; the quote shown to the user must not."""
    chunks = chunk_document(_doc(), min_chars=10)
    snssai = next(c for c in chunks if c.clause_id == "5.15.2.1")
    assert "Network slicing" in snssai.text
    assert "Network slicing" not in snssai.body


def test_normative_language_is_flagged():
    chunks = chunk_document(_doc(), min_chars=10)
    general = next(c for c in chunks if c.clause_id == "5.15.1")
    assert general.is_normative is True


def test_prose_lines_are_not_mistaken_for_headings():
    text = SPEC + "\n5 UEs shall be able to register with the network at any time.\n"
    chunks = chunk_document(_doc(text), min_chars=10)
    # "5 UEs shall be able..." is a sentence, not a clause heading.
    assert not any(c.clause_title.startswith("UEs shall be able") for c in chunks)


def test_long_clause_is_split_with_overlap():
    body = "\n\n".join(f"Paragraph {i} about session management." * 12 for i in range(12))
    text = f"6 Session management\n\n6.1 Overview\n{body}\n"
    chunks = chunk_document(_doc(text), target_chars=600, overlap_chars=120, min_chars=10)
    overview = [c for c in chunks if c.clause_id == "6.1"]
    assert len(overview) > 1
    assert all(c.part_count == len(overview) for c in overview)
    assert overview[0].citation_label.endswith(f"(part 1/{len(overview)})")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
