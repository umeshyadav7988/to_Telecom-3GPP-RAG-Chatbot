"""Clause-aware chunking for 3GPP specifications.

Why not a generic recursive character splitter?
-----------------------------------------------
Generic splitters cut on whitespace at N characters. For 3GPP that produces
chunks whose provenance is "page 412" — useless for a system whose whole
premise is verifiable citations. Worse, they routinely sever a requirement
from its scoping clause, which is the single richest source of confident
wrong answers ("the UE shall..." — under which condition? in which state?).

This splitter instead recovers the document's clause tree:

    5                       Overall description
    5.15                    Network slicing
    5.15.2                  Identification of a network slice
    5.15.2.1                S-NSSAI

and emits chunks that (a) never cross a clause boundary and (b) carry the full
breadcrumb of ancestor headings. A citation therefore reads
`TS 23.501 §5.15.2.1 — S-NSSAI`, which a telecom engineer can look up in the
actual spec in seconds. That verifiability is the anti-hallucination property:
a fabricated claim has nowhere to hide behind a vague page reference.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

from .loaders import LoadedDocument

# ---------------------------------------------------------------------------
# Heading grammar
# ---------------------------------------------------------------------------

# "5.15.2.1 S-NSSAI"  /  "5.15.2.1\tS-NSSAI"  — number then title, no sentence
# punctuation at the end (that would make it a numbered list item, not a head).
# The optional trailing letter covers 3GPP's inserted clauses ("6.4.1a").
_CLAUSE_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,3}){0,6}[a-z]?)\s+(?P<title>[^\n]{2,140}?)\s*$"
)

# "Annex A (normative): Foo"  /  "Annex B: Bar"
_ANNEX_RE = re.compile(
    r"^(?P<num>Annex\s+[A-Z])\s*(?:\((?P<kind>normative|informative)\))?\s*[:.]?\s*"
    r"(?P<title>[^\n]{0,140})\s*$",
    re.IGNORECASE,
)

# Clauses *inside* an annex are lettered: "C.3.4 Profile A", "A.1 Causes...",
# "C.3.4a Profile B". Without this the whole annex collapses into one chunk and
# its citations degrade from "§C.3.4" to "§Annex C" — a real loss of precision
# in TS 33.501, where the SUPI protection schemes live entirely in Annex C.
_ANNEX_CLAUSE_RE = re.compile(
    r"^(?P<num>[A-Z](?:\.\d{1,3}){1,5}[a-z]?)\s+(?P<title>[^\n]{2,140}?)\s*$"
)

_FRONT_MATTER = {
    "foreword", "scope", "references", "definitions", "abbreviations",
    "definitions and abbreviations", "definitions, symbols and abbreviations",
}

# A heading title should not look like prose.
_PROSE_HINTS = re.compile(r"[.;]\s|\bshall\b|\bmay\b|\bis\b|\bare\b|\bthe\b\s+\w+\s+\b(is|are|shall)\b", re.I)

_REQUIREMENT_RE = re.compile(r"\b(shall|shall not|must|should|may)\b", re.I)
_TABLE_LINE_RE = re.compile(r"^[^\n|]{0,80}\|.+\|", re.M)
_TABLE_DELIMITER = "|"


@dataclass
class Chunk:
    """One retrievable unit of specification text."""

    chunk_id: str
    doc_id: str                 # "TS 23.501"
    doc_title: str
    version: str
    release: str
    clause_id: str              # "5.15.2.1" or "Annex A"
    clause_title: str           # "S-NSSAI"
    breadcrumb: str             # "5 Overall description > 5.15 Network slicing > ..."
    text: str                   # heading line + body, as indexed
    body: str                   # body only, without the heading line
    part: int = 0               # index when one clause is split across chunks
    part_count: int = 1
    page: int | None = None
    char_start: int = 0
    char_end: int = 0
    source_path: str = ""
    is_normative: bool = False  # contains shall/must/should/may
    has_table: bool = False
    token_estimate: int = 0

    @property
    def citation_label(self) -> str:
        """Human-readable citation, e.g. `TS 23.501 §5.15.2.1 — S-NSSAI`."""
        head = f"{self.doc_id} §{self.clause_id}" if self.clause_id else self.doc_id
        if self.clause_title:
            head = f"{head} — {self.clause_title}"
        if self.part_count > 1:
            head = f"{head} (part {self.part + 1}/{self.part_count})"
        return head

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation_label"] = self.citation_label
        return d


@dataclass
class _Section:
    number: str
    title: str
    level: int
    start: int
    lines: list = field(default_factory=list)
    ancestors: list = field(default_factory=list)  # [(number, title), ...]


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

def _looks_like_heading(line: str) -> tuple[str, str, int] | None:
    """Return (number, title, level) if the line is a clause heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 160:
        return None

    # A table row is never a heading. Without this, a 5QI row like
    #   "1 | GBR | 20 | 100 ms | 10^-2 | Conversational Voice"
    # parses as clause "1" titled "| GBR | 20 | ...", which splits the table
    # into one bogus section per row. Each is then below `min_chars` and gets
    # dropped — silently deleting the entire QoS characteristics table from the
    # index while retrieval still happily returns the table's *header* chunk.
    # Numeric tables are where a spec keeps its most citable facts, so this is
    # the worst possible thing to lose. Regression test in test_chunking.py.
    if _TABLE_DELIMITER in stripped:
        return None

    m = _ANNEX_RE.match(stripped)
    if m:
        number = re.sub(r"\s+", " ", m.group("num")).title()
        return number, (m.group("title") or "").strip(), 1

    m = _CLAUSE_RE.match(stripped) or _ANNEX_CLAUSE_RE.match(stripped)
    if not m:
        return None

    number, title = m.group("num"), m.group("title").strip()

    # Reject numbered list items and cross-references masquerading as headings.
    if title.endswith((".", ";", ",", ":")) and not title.lower() in _FRONT_MATTER:
        return None
    if _PROSE_HINTS.search(title) and len(title.split()) > 8:
        return None
    if re.match(r"^\d", title):          # "5.1 2.3 GHz band" style false hit
        return None
    if number.count(".") > 6:
        return None
    # A heading is title-ish: it should not be a full sentence.
    if len(title.split()) > 14:
        return None

    return number, title, number.count(".") + 1


# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------

def _split_sections(text: str) -> list[_Section]:
    """Walk the document once, building a stack-based clause tree."""
    sections: list[_Section] = []
    stack: list[tuple[str, str, int]] = []  # (number, title, level)
    current: _Section | None = None
    offset = 0

    for line in text.split("\n"):
        head = _looks_like_heading(line)
        if head:
            number, title, level = head
            while stack and stack[-1][2] >= level:
                stack.pop()
            ancestors = [(n, t) for n, t, _ in stack]
            current = _Section(
                number=number,
                title=title,
                level=level,
                start=offset,
                ancestors=ancestors,
            )
            sections.append(current)
            stack.append((number, title, level))
        else:
            if current is None:
                # Text before the first heading (cover page, ToC).
                current = _Section(number="", title="Front matter", level=0, start=0)
                sections.append(current)
            current.lines.append(line)
        offset += len(line) + 1

    return sections


# ---------------------------------------------------------------------------
# Windowing inside a clause
# ---------------------------------------------------------------------------

def _paragraphs(body: str) -> list[str]:
    """Split a clause body into atomic units we will never cut through."""
    blocks: list[str] = []
    buffer: list[str] = []
    for line in body.split("\n"):
        if not line.strip():
            if buffer:
                blocks.append("\n".join(buffer))
                buffer = []
        else:
            buffer.append(line)
    if buffer:
        blocks.append("\n".join(buffer))
    return [b for b in blocks if b.strip()]


def _window(blocks: list[str], target: int, overlap: int) -> list[str]:
    """Pack paragraphs into ~`target`-char windows with `overlap` carry-over."""
    if not blocks:
        return []

    windows: list[str] = []
    current: list[str] = []
    size = 0

    for block in blocks:
        block_len = len(block)
        # An oversized single paragraph (a big table) becomes its own window,
        # hard-split only if it is truly enormous.
        if block_len > target * 2 and not current:
            for i in range(0, block_len, target):
                windows.append(block[i : i + target])
            continue

        if size + block_len > target and current:
            windows.append("\n\n".join(current))
            # Carry the tail of the previous window forward so a requirement
            # split across the boundary still has its scoping sentence.
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev)
            current = carry
            size = carried

        current.append(block)
        size += block_len

    if current:
        windows.append("\n\n".join(current))

    return windows


def _chunk_id(doc_id: str, clause: str, part: int, text: str) -> str:
    digest = hashlib.sha1(f"{doc_id}|{clause}|{part}|{text[:256]}".encode()).hexdigest()
    return digest[:16]


def _estimate_tokens(text: str) -> int:
    # ~4 chars/token is close enough for budgeting; exact counts come from the
    # API's count_tokens endpoint when it matters.
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    doc: LoadedDocument,
    target_chars: int = 1400,
    overlap_chars: int = 200,
    min_chars: int = 120,
) -> list[Chunk]:
    """Turn a loaded specification into clause-scoped, citable chunks."""
    chunks: list[Chunk] = []

    for section in _split_sections(doc.text):
        body = "\n".join(section.lines).strip()
        if not body:
            continue

        breadcrumb_parts = [
            f"{n} {t}".strip() for n, t in section.ancestors if (n or t)
        ]
        if section.number or section.title:
            breadcrumb_parts.append(f"{section.number} {section.title}".strip())
        breadcrumb = " > ".join(breadcrumb_parts)

        heading_line = f"{section.number} {section.title}".strip()
        windows = _window(_paragraphs(body), target_chars, overlap_chars)

        # Drop stubs like "5.1 General" whose body is one cross-reference,
        # unless they carry a requirement.
        windows = [
            w
            for w in windows
            if len(w.strip()) >= min_chars or _REQUIREMENT_RE.search(w)
        ]
        if not windows:
            continue

        for i, window in enumerate(windows):
            # Prefixing the breadcrumb into the indexed text is deliberate: it
            # gives the embedder and BM25 the clause context, so a query like
            # "network slicing identifier" matches §5.15.2 even when the body
            # only ever says "S-NSSAI".
            indexed_text = f"{breadcrumb}\n\n{window}" if breadcrumb else window

            chunk = Chunk(
                chunk_id=_chunk_id(doc.doc_id, section.number, i, window),
                doc_id=doc.doc_id,
                doc_title=doc.title,
                version=doc.version,
                release=doc.release,
                clause_id=section.number,
                clause_title=section.title,
                breadcrumb=breadcrumb,
                text=indexed_text,
                body=window,
                part=i,
                part_count=len(windows),
                page=doc.page_for_offset(section.start),
                char_start=section.start,
                char_end=section.start + len(window),
                source_path=doc.source_path,
                is_normative=bool(_REQUIREMENT_RE.search(window)),
                has_table=bool(_TABLE_LINE_RE.search(window)),
                token_estimate=_estimate_tokens(indexed_text),
            )
            _ = heading_line  # retained for readability of the structure above
            chunks.append(chunk)

    return chunks
