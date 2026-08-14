"""Deterministic grounding guard tests (app/utils/text.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.text import (
    extract_citations,
    extract_verifiable_tokens,
    numeric_guard,
    split_sentences,
    strip_citations,
)

SOURCE = (
    "The following SST values are standardised: SST value 1: eMBB. "
    "The default value of timer T3512 is 54 minutes. "
    "For 5QI 1 the Packet Delay Budget is 100 ms and the Packet Error Rate is 10^-2."
)


# --- sentence splitting -----------------------------------------------------

def test_does_not_split_on_3gpp_abbreviations():
    text = "The UE may register, e.g. after a power cycle. It then starts T3512."
    assert len(split_sentences(text)) == 2


def test_does_not_split_on_decimals_or_clause_numbers():
    text = "See TS 23.501 clause 5.15.2 for details. The SST is 8 bits."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "23.501" in sentences[0]
    assert "5.15.2" in sentences[0]


# --- citation parsing -------------------------------------------------------

def test_extracts_all_citation_marker_forms():
    assert extract_citations("Foo [S1] bar [S2][S3] baz [S4, S5].") == [1, 2, 3, 4, 5]


def test_citation_extraction_deduplicates():
    assert extract_citations("[S2] and again [S2]") == [2]


def test_strip_citations_leaves_clean_prose():
    assert strip_citations("The PDB is 100 ms [S1][S2].") == "The PDB is 100 ms ."


# --- verifiable token extraction --------------------------------------------

def test_extracts_quantities_identifiers_and_spec_refs():
    tokens = extract_verifiable_tokens("Per TS 23.501, 5QI 1 has a PDB of 100 ms [S1].")
    assert "100ms" in tokens["quantities"]
    assert "TS 23.501" in tokens["spec_refs"]
    assert any(t.upper() == "5QI" for t in tokens["identifiers"])


def test_citation_markers_are_not_treated_as_facts():
    """[S1] must never be mistaken for an identifier or a number to verify."""
    tokens = extract_verifiable_tokens("The AMF terminates N1 [S1][S12].")
    assert not any(t.upper().startswith("S1") and t.upper() != "S1" for t in tokens["identifiers"])
    assert "12" not in tokens["numbers"]


# --- the numeric guard ------------------------------------------------------

def test_guard_passes_when_values_are_copied_from_source():
    result = numeric_guard("For 5QI 1 the Packet Delay Budget is 100 ms.", SOURCE)
    assert result["passed"] is True
    assert result["support_ratio"] == 1.0


def test_guard_catches_a_fabricated_timer_value():
    """The single highest-cost failure mode: a plausible but invented number."""
    result = numeric_guard("The default value of timer T3512 is 42 minutes.", SOURCE)
    assert result["passed"] is False
    assert any(u["token"] == "42minutes" for u in result["unsupported_tokens"])


def test_guard_catches_a_fabricated_identifier():
    result = numeric_guard("Timer T3599 controls periodic registration.", SOURCE)
    assert result["passed"] is False
    assert any(u["token"] == "T3599" for u in result["unsupported_tokens"])


def test_guard_normalises_unit_spacing():
    """'100ms' in the claim must match '100 ms' in the source, and vice versa."""
    assert numeric_guard("The PDB is 100ms.", "the Packet Delay Budget is 100 ms")["passed"]
    assert numeric_guard("The PDB is 100 ms.", "a PDB of 100ms")["passed"]


def test_guard_ignores_citation_markers_when_checking():
    result = numeric_guard("The PDB for 5QI 1 is 100 ms [S1][S3].", SOURCE)
    assert result["passed"] is True


def test_guard_is_vacuously_true_for_claims_with_no_facts():
    result = numeric_guard("This clause describes the overall approach.", SOURCE)
    assert result["passed"] is True
    assert result["checked_tokens"] == 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
