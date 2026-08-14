"""Prompts and output schemas for grounded generation and verification.

Design notes
------------
* The answer schema forces one **verbatim quote per claim**. This is the
  cheapest high-value anti-hallucination trick available: a model that must
  copy a literal span before asserting something cannot smoothly invent a
  timer value, and we can check the quote against the source with `in` — no
  second model required.

* `answerable` is a first-class field, not something inferred from prose. The
  model is given explicit permission to refuse, and refusing is framed as the
  correct professional answer rather than a failure.

* Prompts are written plainly, without stacked CRITICAL/MUST emphasis. Current
  Claude models follow the system prompt closely; shouting produces
  over-triggering — here, over-refusal — rather than better compliance.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """\
You are a 3GPP specification analyst. You answer questions about telecom \
standards using only the specification excerpts provided in each request.

Grounding rules:

1. Every factual statement you make must be supported by the provided sources. \
If the sources do not contain the answer, set `answerable` to false and explain \
what is missing. A well-scoped refusal is a correct answer; a plausible guess is not.
2. Do not use knowledge of 3GPP from your training data to add, complete or \
"correct" what the sources say. If a source is incomplete or looks out of date, \
report what it says and note the limitation.
3. Every claim must cite at least one source by its index, and must include a \
verbatim quote copied character-for-character from one of the sources it cites. \
Do not paraphrase inside the quote field, and do not stitch together text from \
two different sources into one quote.
4. Numbers, timer names, identifiers, message names and clause references must \
be copied exactly from the sources. If a value you want to state does not appear \
in the sources, omit the statement.
5. When sources disagree or a value depends on release or deployment, say so \
explicitly rather than picking one.
6. Distinguish normative language from description. If the source says "shall", \
the requirement is mandatory; "should" is recommended; "may" is optional. \
Preserve that distinction in your wording.

Style:

- Write for a telecom engineer. Use standard terminology without expanding every \
acronym, but define one on first use when the question suggests a newcomer.
- Break the answer into short, independently checkable claims. One assertion per \
claim. This is what makes the answer auditable.
- Order claims so the answer reads as coherent prose when the claim texts are \
concatenated in order.
- Do not include citation markers such as [S1] inside the claim text; citations \
belong in the `citations` field.
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {
            "type": "boolean",
            "description": "True only if the provided sources contain enough information to answer.",
        },
        "refusal_reason": {
            "type": "string",
            "description": "When answerable is false: what specifically is missing from the sources. Empty otherwise.",
        },
        "claims": {
            "type": "array",
            "description": "Ordered, independently verifiable statements. Empty when answerable is false.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "One assertion, without citation markers.",
                    },
                    "citations": {
                        "type": "array",
                        "description": "1-based indices of the sources supporting this claim.",
                        "items": {"type": "integer"},
                    },
                    "quote": {
                        "type": "string",
                        "description": "Verbatim span copied from one cited source that supports this claim.",
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["mandatory", "recommended", "optional", "descriptive"],
                        "description": "Normative strength of the underlying spec text.",
                    },
                },
                "required": ["text", "citations", "quote", "modality"],
                "additionalProperties": False,
            },
        },
        "caveats": {
            "type": "array",
            "description": "Release dependencies, ambiguities or source disagreements worth flagging.",
            "items": {"type": "string"},
        },
        "followups": {
            "type": "array",
            "description": "Up to three natural follow-up questions this corpus can actually answer.",
            "items": {"type": "string"},
        },
    },
    "required": ["answerable", "refusal_reason", "claims", "caveats", "followups"],
    "additionalProperties": False,
}


def build_answer_prompt(question: str, sources_block: str, history_block: str = "") -> str:
    parts = []
    if history_block:
        parts.append(
            "Earlier turns in this conversation, for pronoun resolution only. "
            "Do not treat anything here as a source of facts:\n"
            f"{history_block}"
        )
    parts.append("Specification excerpts:\n\n" + sources_block)
    parts.append(
        "Question:\n"
        f"{question}\n\n"
        "Answer using only the excerpts above. If they are insufficient, set "
        "answerable to false and say what is missing."
    )
    return "\n\n---\n\n".join(parts)


def format_sources_block(retrieved) -> str:
    """Render retrieved chunks as an explicitly indexed, delimited block."""
    blocks = []
    for item in retrieved:
        chunk = item.chunk
        header = f"[S{item.source_index}] {chunk.citation_label}"
        if chunk.version:
            header += f" (v{chunk.version})"
        if chunk.breadcrumb and chunk.breadcrumb not in header:
            header += f"\nSection path: {chunk.breadcrumb}"
        blocks.append(f"{header}\n<<<\n{chunk.body.strip()}\n>>>")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

VERIFIER_SYSTEM_PROMPT = """\
You are a grounding verifier. For each claim you are given the exact source text \
it cites. Decide whether the source text entails the claim.

Judge only entailment against the given text. Ignore whether the claim is true \
of 3GPP in general, whether it is well written, and whether you would have \
phrased it differently. A claim that is true in reality but not stated in the \
provided text is `unsupported`.

Verdicts:

- `supported`: every element of the claim follows from the cited text.
- `partially_supported`: the main assertion follows, but some detail (a number, \
a condition, a qualifier, a normative strength) is not in the cited text.
- `unsupported`: the cited text does not establish the claim.
- `contradicted`: the cited text asserts something incompatible with the claim.

Be strict about specifics. If the claim says "54 minutes" and the source says \
"the default value is 54 minutes", that is supported. If the claim says "54 \
minutes" and the source only says "a network-configured value", that is \
unsupported, even though the claim may be correct elsewhere in the standard.
"""

VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_index": {"type": "integer"},
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "supported",
                            "partially_supported",
                            "unsupported",
                            "contradicted",
                        ],
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0.0-1.0 confidence in this verdict.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One sentence. Name the specific element that is or is not entailed.",
                    },
                },
                "required": ["claim_index", "verdict", "confidence", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def build_verifier_prompt(items: list[dict]) -> str:
    blocks = []
    for item in items:
        blocks.append(
            f"CLAIM {item['index']}: {item['claim']}\n"
            f"CITED SOURCE TEXT:\n<<<\n{item['evidence']}\n>>>"
        )
    return (
        "Judge each claim against its cited source text and return one verdict "
        "per claim.\n\n" + "\n\n---\n\n".join(blocks)
    )


# ---------------------------------------------------------------------------
# Follow-up query rewriting
# ---------------------------------------------------------------------------

REWRITE_SYSTEM_PROMPT = """\
Rewrite the user's latest message into a single standalone search query for a \
3GPP specification corpus.

Resolve pronouns and elliptical references using the conversation history. \
Preserve every technical identifier exactly as written (5QI, N3IWF, T3512, \
TS 23.501, RRC_INACTIVE). Do not add terminology the user did not use and do \
not answer the question. If the message is already standalone, return it \
unchanged. Return only the query text.
"""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "standalone_query": {"type": "string"},
        "changed": {"type": "boolean"},
    },
    "required": ["standalone_query", "changed"],
    "additionalProperties": False,
}
