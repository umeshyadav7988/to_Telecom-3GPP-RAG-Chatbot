"""Deterministic false-premise detection — abstention without a model.

The retrieval gate catches questions whose *topic* is absent from the corpus.
It cannot catch a question whose topic is present but whose premise is false:

    "What is the default value of timer T3599?"      (T3599 does not exist)
    "What is the Packet Delay Budget for 5QI 91?"    (5QI 91 does not exist)

Both retrieve the correct table with high confidence, because everything about
them *except the asserted entity* is a genuine match. Measured on the golden
set, they score 0.542 and 0.478 — above four of the eighteen answerable
questions, so no threshold separates them.

The insight here is that the numeric guard already used on answers works just
as well on questions. If a question asserts a specific alphanumeric identifier
and that identifier appears nowhere in the clauses the retriever just returned,
the question is asking about something that does not exist. That is a string
comparison, not an inference — no model required.

What it deliberately does not check
-----------------------------------
* Pure-alphabetic acronyms (AMF, PDB, GBR). A question may legitimately
  abbreviate a term the clause spells out, so a literal miss proves nothing.
* Single digits. "5QI 1" and "numerology 3" are answerable; "1" and "3" are far
  too common to discriminate.
* Anything semantic. "Why did 3GPP remove network slicing in Release 17?" uses
  only real entities and states a false relationship between them. Catching
  that needs entailment — it is the residue this guard cannot reach, and the
  reason a model still earns its place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..utils.text import normalise_for_match

# Identifiers worth checking are the ones that encode a *value*: a timer
# number, a QoS index, an algorithm generation, a specification number.
_IDENTIFIER_RE = re.compile(
    r"\b("
    r"[A-Z]\d{3,5}[a-z]?"                    # T3512, T3599
    r"|[0-9]?[A-Z]{2,}[-_]?[A-Z0-9]*\d+"     # NEA3, 5G-AKA, N3IWF
    r")\b"
)
_SPEC_REF_RE = re.compile(r"\b(?:TS|TR)\s?(\d{2}\.\d{3})\b", re.IGNORECASE)
# A bare number of two or more digits, ignoring decimals and clause references.
_MULTIDIGIT_RE = re.compile(r"(?<![\w.\-^])(\d{2,})(?![\w.])")

# Words that match the identifier shapes above but carry no factual assertion.
_ALLOWLIST = {
    "3GPP", "5G", "4G", "6G", "2G", "3G", "5GS", "5GC", "4GS", "NG", "NR",
    "LTE", "IPV4", "IPV6", "IPV4V6", "V2X", "QOS", "RAN", "UE",
}


@dataclass
class PremiseVerdict:
    """Outcome of checking a question's asserted entities against the evidence."""

    passed: bool
    missing: list = field(default_factory=list)   # [{"token": ..., "kind": ...}]
    checked: list = field(default_factory=list)

    def describe(self) -> str:
        if self.passed:
            return ""
        tokens = ", ".join(f"`{m['token']}`" for m in self.missing)
        plural = "do not appear" if len(self.missing) > 1 else "does not appear"
        return (
            f"Your question refers to {tokens}, which {plural} anywhere in the "
            f"clauses retrieved for it."
        )


def extract_asserted_entities(question: str) -> list[dict]:
    """Pull the specific, checkable entities a question asserts the existence of."""
    entities: list[dict] = []
    seen: set[str] = set()

    def add(token: str, kind: str) -> None:
        key = normalise_for_match(token)
        if key and key not in seen:
            seen.add(key)
            entities.append({"token": token, "kind": kind})

    for match in _SPEC_REF_RE.finditer(question):
        add(match.group(0), "spec_ref")

    # Remove spec references before the other passes so "24.501" is not also
    # reported as the bare number "24".
    remainder = _SPEC_REF_RE.sub(" ", question)

    for match in _IDENTIFIER_RE.finditer(remainder):
        token = match.group(1)
        if token.upper() in _ALLOWLIST:
            continue
        add(token, "identifier")

    # Multi-digit numbers only count when the question also names an identifier
    # they could belong to ("5QI 91", "Release 17"). A bare "How many of the 24
    # ..." should not trip the guard.
    has_identifier_context = bool(
        _IDENTIFIER_RE.search(remainder)
        or re.search(r"\b(5QI|QFI|SST|SSC|release|rel|timer|profile|numerology|FR)\b",
                     remainder, re.IGNORECASE)
    )
    if has_identifier_context:
        for match in _MULTIDIGIT_RE.finditer(remainder):
            add(match.group(1), "value")

    return entities


def check_premise(question: str, evidence: str) -> PremiseVerdict:
    """Verify every entity the question asserts occurs in the retrieved text."""
    entities = extract_asserted_entities(question)
    if not entities:
        return PremiseVerdict(passed=True)

    haystack = normalise_for_match(evidence)
    missing = [e for e in entities if normalise_for_match(e["token"]) not in haystack]

    return PremiseVerdict(passed=not missing, missing=missing, checked=entities)
