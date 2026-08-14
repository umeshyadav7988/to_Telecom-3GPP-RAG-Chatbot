"""Answer extraction without a language model.

Dumping the top retrieved chunks is a poor no-LLM answer. Chunks run to ~1400
characters, so the sentence that actually answers the question is buried, and
truncating to fit loses it. This module does the last mile deterministically:
score the individual sentences and table rows inside the retrieved clauses and
return the ones that answer the question.

Four spec-specific behaviours carry almost all of the value. Each was added in
response to a measured failure on the golden set, not on principle:

* **Row-key matching.** A 3GPP spec answers "what is the PDB for 5QI 1" with a
  table row, and that row reads `1 | GBR | 20 | 100 ms | ...`. It shares *no
  words* with the question — the column names live in the header and the entity
  is the bare first cell. Matching the query's numeric/identifier tokens
  against a row's first cell is what makes table lookups work at all.

* **Header carry-along.** `1 | GBR | 20 | 100 ms` is meaningless without
  `5QI Value | Resource Type | Priority | Packet Delay Budget`.

* **List continuation.** Specs enumerate constantly: "The following SST values
  are standardised:" followed by the actual values. Selecting the colon line
  and dropping its list is the wrong half of the answer.

* **Phrase matching.** "Profile A" and "Profile B" are near-identical
  sentences; the only distinguishing token is a single letter that every
  reasonable tokenizer discards. A verbatim two-token phrase check separates
  them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils.text import normalise_for_match, split_sentences
from .bm25 import _DOMAIN_STOPWORDS
from .embeddings import tokenize

_TABLE_ROW_RE = re.compile(r"\|")
_HEADER_HINTS = re.compile(
    r"\b(value|type|level|budget|rate|name|description|parameter|timer|cause|"
    r"default|example|services|resource|priority|packet|subcarrier|spacing|"
    r"cyclic|prefix|supported|start|stop|expiry|normal)\b",
    re.IGNORECASE,
)
# Interrogatives carry no retrieval signal but would otherwise be scored.
_QUESTION_WORDS = {
    "what", "which", "how", "why", "when", "where", "who", "does", "do", "did",
    "is", "are", "was", "were", "list", "describe", "explain", "give", "tell",
    "many", "much", "value", "used",
}
_STOPWORDS = _DOMAIN_STOPWORDS | _QUESTION_WORDS

# Tokens that could name a table row: a bare number, or an alphanumeric id.
_ROW_KEY_RE = re.compile(r"\b(\d{1,4}|[A-Z]\d{3,5}[a-z]?|[0-9]?[A-Z]{2,}\d*)\b")


@dataclass
class ExtractedUnit:
    text: str
    source_index: int
    kind: str          # "sentence" | "table_row"
    score: float
    header: str = ""   # column header, for table rows
    position: int = 0  # order within its clause, for list continuation


def _stem(token: str) -> str:
    """Crude plural folding so `values` matches `value`."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _query_weights(query: str) -> dict[str, float]:
    """Weight query terms by how discriminative they are."""
    weights: dict[str, float] = {}
    for raw in tokenize(query):
        if raw in _STOPWORDS or len(raw) <= 1:
            continue
        term = _stem(raw)
        if any(c.isdigit() for c in term):
            weight = 3.0        # identifiers: 5qi, t3512, nea3
        elif len(term) <= 4:
            weight = 1.5        # short acronyms: amf, gbr, pdb
        else:
            weight = 1.0
        weights[term] = max(weights.get(term, 0.0), weight)
    return weights


def _query_row_keys(query: str) -> set[str]:
    """Tokens from the question that could identify a table row."""
    return {m.group(1).lower() for m in _ROW_KEY_RE.finditer(query)}


def _query_phrases(query: str) -> list[str]:
    """Adjacent token pairs, kept verbatim — including single characters.

    `profile a` is the only thing separating the Profile A clause from the
    Profile B clause, and both tokens survive here where the weighted-term pass
    discards the `a`.
    """
    tokens = [t for t in tokenize(query) if t not in _QUESTION_WORDS]
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def _is_header_row(line: str) -> bool:
    cells = [c.strip() for c in line.split("|")]
    if len(cells) < 2:
        return False
    numeric_cells = sum(1 for c in cells if re.search(r"\d", c))
    return numeric_cells <= 1 and bool(_HEADER_HINTS.search(line))


def _split_units(body: str) -> tuple[list[tuple[str, str]], str]:
    """Split a clause body into (kind, text) units; return the table header too."""
    units: list[tuple[str, str]] = []
    header = ""

    for block in body.split("\n"):
        line = block.strip()
        if not line:
            continue
        if _TABLE_ROW_RE.search(line):
            if _is_header_row(line) and not header:
                header = line
                continue
            units.append(("table_row", line))
        else:
            for sentence in split_sentences(line):
                if len(sentence.split()) >= 2:
                    units.append(("sentence", sentence))

    return units, header


def _has_qualifier_conflict(text: str, phrases: list[str]) -> bool:
    """True if the unit describes a *sibling* of what the question asked about.

    Specs are built from near-identical parallel clauses distinguished by a
    single short qualifier: "Profile A" / "Profile B", "SST value 1" / "SST
    value 2", "Event A3" / "Event A4". The sentences around them share almost
    every word, so term overlap cannot separate them, and the qualifier itself
    is a one-character token that every tokenizer discards.

    Asking about Profile A and being handed the Profile B sentence is not a
    near-miss — it is a wrong answer with a real citation attached, which is
    the most dangerous shape an error can take here.
    """
    lowered = normalise_for_match(text)
    for phrase in phrases:
        head, _, qualifier = phrase.partition(" ")
        # Only short qualifiers behave this way; "network slicing" is not a
        # head+qualifier pair.
        if not qualifier or len(qualifier) > 2 or len(head) < 4:
            continue
        if head not in lowered or normalise_for_match(phrase) in lowered:
            continue
        rival = re.search(rf"\b{re.escape(head)}\s+(\w{{1,2}})\b", lowered)
        if rival and rival.group(1) != qualifier:
            return True
    return False


def _score_unit(kind: str, text: str, weights: dict[str, float],
                row_keys: set[str], phrases: list[str]) -> float:
    if not weights and not row_keys:
        return 0.0

    tokens = tokenize(text)
    token_set = {_stem(t) for t in tokens}
    total = sum(weights.values()) or 1.0
    covered = sum(w for term, w in weights.items() if term in token_set)
    score = covered / total

    normalised = normalise_for_match(text)
    for phrase in phrases:
        if normalise_for_match(phrase) in normalised:
            score += 0.30

    if kind == "table_row":
        first_cell = text.split("|")[0].strip().lower()
        # The dominant signal: this row is *about* the thing being asked about.
        if first_cell in row_keys:
            score += 1.0
        elif any(t in row_keys for t in tokenize(first_cell)):
            score += 0.6
        if re.search(r"\d", text):
            score += 0.05
    elif (
        len(tokens) < 8
        and not text.rstrip().endswith(":")   # a list intro, not a stub
        and not any(w >= 3.0 and term in token_set for term, w in weights.items())
    ):
        # Short sentences are usually cross-references ("See clause 5.7.4."),
        # unless they carry an identifier the question asked about or introduce
        # the list that holds the answer.
        score *= 0.7

    if _has_qualifier_conflict(text, phrases):
        score *= 0.45

    return score


def extract_units(
    query: str,
    retrieved,
    *,
    max_units: int = 8,
    max_chars: int = 2200,
    min_score: float = 0.16,
    relative_floor: float = 0.45,
) -> list[ExtractedUnit]:
    """Select the sentences and rows within retrieved clauses that answer `query`."""
    weights = _query_weights(query)
    row_keys = _query_row_keys(query)
    phrases = _query_phrases(query)

    # Keep every clause's units in document order so a selected "…:" line can
    # pull its list items along afterwards.
    per_source: dict[int, list[ExtractedUnit]] = {}
    candidates: list[ExtractedUnit] = []

    for item in retrieved:
        units, header = _split_units(item.chunk.body)
        ordered: list[ExtractedUnit] = []
        for position, (kind, text) in enumerate(units):
            raw = _score_unit(kind, text, weights, row_keys, phrases)
            unit = ExtractedUnit(
                text=text,
                source_index=item.source_index,
                kind=kind,
                # Blend in the clause's own relevance so a strong unit in a weak
                # clause does not outrank a strong unit in the best one.
                score=round(0.75 * raw + 0.25 * item.score, 4),
                header=header if kind == "table_row" else "",
                position=position,
            )
            ordered.append(unit)
            if raw >= min_score:
                candidates.append(unit)
        per_source[item.source_index] = ordered

    candidates.sort(key=lambda u: u.score, reverse=True)

    # Relative floor: anything far below the best unit is topically related
    # noise rather than an answer. Swept against the golden set — 0.40-0.50 is
    # a flat optimum (84% vs 80% overall), so 0.45 sits in the middle of the
    # stable band rather than on its edge. Continuation units are exempt: they
    # are pulled in by their parent and inherit its relevance.
    floor = candidates[0].score * relative_floor if candidates else 0.0

    selected: dict[tuple[int, int], ExtractedUnit] = {}
    budget = 0

    def take(unit: ExtractedUnit) -> bool:
        key = (unit.source_index, unit.position)
        if key in selected:
            return False
        nonlocal budget
        if len(selected) >= max_units or budget + len(unit.text) > max_chars:
            return False
        selected[key] = unit
        budget += len(unit.text)
        return True

    for unit in candidates:
        if unit.score < floor:
            break        # candidates are score-sorted
        if not take(unit):
            continue
        # List continuation: "The following SST values are standardised:" is
        # only half an answer without the values that follow it.
        if unit.text.rstrip().endswith(":"):
            siblings = per_source.get(unit.source_index, [])
            for follower in siblings[unit.position + 1 : unit.position + 9]:
                if follower.kind != unit.kind and follower.kind == "sentence":
                    pass  # allow either kind; ordering below stops at a break
                if not take(follower):
                    break

    ordered_selection = sorted(selected.values(), key=lambda u: (u.source_index, u.position))
    return ordered_selection


def render_unit(unit: ExtractedUnit) -> str:
    """Human-readable text for a unit, including a table header where relevant."""
    if unit.kind == "table_row" and unit.header:
        return f"{unit.header}\n{unit.text}"
    return unit.text
