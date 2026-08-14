"""Deterministic text checks used by the grounding verifier.

These run before (and independently of) any LLM verification. An LLM judge can
itself be wrong; a regex that asks "does the number 54 actually appear in the
cited clause?" cannot. Numeric and identifier fabrication is the highest-cost
failure mode in a telecom assistant — a wrong timer value or a wrong 5QI is
worse than no answer — so it gets a check that does not involve a model.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Abbreviations that must not terminate a sentence. 3GPP prose is dense with
# them, and splitting on "e.g." produces claim fragments that verify as
# unsupported for purely cosmetic reasons.
_ABBREVIATIONS = (
    "e.g", "i.e", "etc", "cf", "vs", "approx", "Fig", "No", "Rel", "vol",
    "TS", "TR", "Sec", "cl",
)
_ABBREV_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.\s*$",
    re.IGNORECASE,
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def split_sentences(text: str) -> list[str]:
    """Split into sentences without breaking on 3GPP abbreviations or decimals."""
    if not text:
        return []

    # Protect decimals and clause numbers ("23.501", "5.15.2") from the splitter.
    protected = re.sub(r"(?<=\d)\.(?=\d)", "\x00", text)

    raw = _SENT_SPLIT_RE.split(protected)
    sentences: list[str] = []
    buffer = ""
    for part in raw:
        candidate = (buffer + " " + part).strip() if buffer else part.strip()
        if _ABBREV_RE.search(candidate):
            buffer = candidate
            continue
        buffer = ""
        if candidate:
            sentences.append(candidate.replace("\x00", "."))
    if buffer:
        sentences.append(buffer.replace("\x00", "."))

    return [s for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Citation markers
# ---------------------------------------------------------------------------

_CITATION_BLOCK_RE = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]", re.IGNORECASE)
_CITATION_ID_RE = re.compile(r"S(\d+)", re.IGNORECASE)


def extract_citations(text: str) -> list[int]:
    """Pull the 1-based source indices out of `[S1]`, `[S2, S4]`, `[S1][S3]`."""
    found: list[int] = []
    for block in _CITATION_BLOCK_RE.finditer(text):
        for m in _CITATION_ID_RE.finditer(block.group(1)):
            value = int(m.group(1))
            if value not in found:
                found.append(value)
    return found


def strip_citations(text: str) -> str:
    return _CITATION_BLOCK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Verifiable-token extraction (the numeric / identifier guard)
# ---------------------------------------------------------------------------

# Values that carry a unit, e.g. "100 ms", "54 minutes", "3.5 GHz", "10^-6".
_QUANTITY_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*"
    r"(ms|msec|milliseconds?|s|sec|seconds?|min|minutes?|h|hours?|"
    r"hz|khz|mhz|ghz|bps|kbps|mbps|gbps|bits?|bytes?|kb|mb|gb|%|db|dbm)\b",
    re.IGNORECASE,
)
# Bare integers/decimals that are not part of a citation marker or a clause ref.
_NUMBER_RE = re.compile(r"(?<![\w.\-])(\d+(?:\.\d+)?)(?![\w%])")
# Telecom identifiers: 5QI, N3IWF, T3512, NEA2, AMF, SUPI, 5G-AKA, RRC_IDLE...
_IDENTIFIER_RE = re.compile(
    r"\b("
    r"[0-9]?[A-Z]{2,}(?:[-_][A-Z0-9]{1,})*"      # AMF, RRC_IDLE, 5G-AKA, N3IWF
    r"|[A-Z]\d{3,4}"                              # T3512, T3502
    r"|\d[A-Z]{2,3}\b"                            # 5QI, 5GS, 5GC
    r")\b"
)
_SPEC_REF_RE = re.compile(r"\b(?:TS|TR)\s?\d{2}\.\d{3}\b", re.IGNORECASE)
_CLAUSE_REF_RE = re.compile(r"§\s?\d+(?:\.\d+)*|\bclause\s+\d+(?:\.\d+)*", re.IGNORECASE)

# Words that look like identifiers but are ordinary prose or our own scaffolding.
_IDENTIFIER_ALLOWLIST = {
    "THE", "AND", "FOR", "NOT", "MAY", "CAN", "ALL", "ANY", "ONE", "TWO",
    "USE", "SEE", "PER", "VIA", "NOTE", "WHEN", "THIS", "THAT", "WITH", "FROM",
    "SHALL", "SHOULD", "MUST", "IF", "IS", "ARE", "IT", "AS", "AN", "OR",
}


def normalise_for_match(text: str) -> str:
    """Canonical form for substring comparison against source text."""
    lowered = text.lower()
    lowered = lowered.replace("−", "-").replace("–", "-").replace("—", "-")
    # "100 ms" == "100ms";  "5 QI" == "5QI"
    lowered = re.sub(r"(\d)\s+([a-z%])", r"\1\2", lowered)
    lowered = re.sub(r"[\s,]+", " ", lowered)
    return lowered


def extract_verifiable_tokens(text: str) -> dict[str, list[str]]:
    """Extract the factual atoms a claim asserts, grouped by kind."""
    clean = strip_citations(text)

    quantities = [f"{m.group(1)}{m.group(2).lower()}" for m in _QUANTITY_RE.finditer(clean)]

    # Remove quantities before hunting bare numbers so "100 ms" is not also
    # reported as the number "100".
    without_quantities = _QUANTITY_RE.sub(" ", clean)
    numbers = [
        m.group(1)
        for m in _NUMBER_RE.finditer(without_quantities)
        # Single digits are almost always list ordinals ("1.", "step 2") and
        # generate noise, so require two significant characters.
        if len(m.group(1)) > 1
    ]

    # Identifiers are split by fabrication risk:
    #   * alphanumeric  (5QI, T3512, NEA3, N3IWF) - hard-checked. These are
    #     values in disguise; inventing one is as costly as inventing a number.
    #   * alphabetic    (AMF, PDB, SUPI)          - soft-checked. A claim may
    #     legitimately abbreviate a term the clause spells out ("PDB" for
    #     "Packet Delay Budget"), so a literal miss is weak evidence. The
    #     entailment pass judges these instead.
    identifiers, acronyms = [], []
    for m in _IDENTIFIER_RE.finditer(clean):
        token = m.group(1)
        if token.upper() in _IDENTIFIER_ALLOWLIST or len(token) <= 1:
            continue
        (identifiers if any(ch.isdigit() for ch in token) else acronyms).append(token)

    spec_refs = [m.group(0) for m in _SPEC_REF_RE.finditer(clean)]
    clause_refs = [m.group(0) for m in _CLAUSE_REF_RE.finditer(clean)]

    def dedupe(items: list[str]) -> list[str]:
        seen, out = set(), []
        for i in items:
            key = i.lower()
            if key not in seen:
                seen.add(key)
                out.append(i)
        return out

    return {
        "quantities": dedupe(quantities),
        "numbers": dedupe(numbers),
        "identifiers": dedupe(identifiers),
        "acronyms": dedupe(acronyms),
        "spec_refs": dedupe(spec_refs),
        "clause_refs": dedupe(clause_refs),
    }


def numeric_guard(claim: str, cited_text: str) -> dict:
    """Check that every quantity/number/identifier in `claim` occurs in `cited_text`.

    Returns the unsupported tokens plus a 0-1 support ratio. Note this is a
    *necessary*, not sufficient, condition for groundedness: it proves the
    tokens were copied from the source, not that the sentence around them is a
    faithful reading of it. The LLM entailment pass covers that second half.
    """
    tokens = extract_verifiable_tokens(claim)
    haystack = normalise_for_match(cited_text)

    unsupported: list[dict] = []
    soft_unsupported: list[dict] = []
    checked = 0

    for kind in ("quantities", "numbers", "spec_refs", "identifiers"):
        for token in tokens[kind]:
            checked += 1
            needle = normalise_for_match(token)
            if needle and needle not in haystack:
                unsupported.append({"token": token, "kind": kind})

    # Soft signals: reported for transparency, but they do not fail the guard.
    # Clause references are here too — a claim may cite a clause the source
    # merely cross-references rather than one whose text we retrieved.
    for kind in ("acronyms", "clause_refs"):
        for token in tokens[kind]:
            if normalise_for_match(token) not in haystack:
                soft_unsupported.append({"token": token, "kind": kind})

    supported = checked - len(unsupported)
    ratio = (supported / checked) if checked else 1.0

    return {
        "checked_tokens": checked,
        "unsupported_tokens": unsupported,
        "soft_unsupported_tokens": soft_unsupported,
        "support_ratio": round(ratio, 4),
        "passed": not unsupported,
    }
