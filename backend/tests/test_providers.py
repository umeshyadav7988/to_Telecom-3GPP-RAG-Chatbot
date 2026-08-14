"""Provider abstraction: schema translation and client selection.

The schema adapter is the sharp edge of multi-provider support. Claude's strict
structured output *requires* `additionalProperties: false`; Gemini's
`response_schema` *rejects* it. Every schema in `prompts.py` therefore has to be
translated, and a silent failure here means the generator falls back to
free-text JSON — losing the guarantee that a claim always carries citations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.llm import build_client, to_gemini_schema
from app.rag.prompts import ANSWER_SCHEMA, REWRITE_SCHEMA, VERIFIER_SCHEMA

ALL_SCHEMAS = [ANSWER_SCHEMA, VERIFIER_SCHEMA, REWRITE_SCHEMA]


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


# --- schema translation -----------------------------------------------------

def test_additional_properties_is_stripped_everywhere():
    """Gemini rejects the whole request if this survives at any depth."""
    for schema in ALL_SCHEMAS:
        converted = to_gemini_schema(schema)
        assert all("additionalProperties" not in node for node in _walk(converted))


def test_real_schemas_still_contain_additional_properties():
    """Guards the test above from silently passing if the schemas change."""
    assert any("additionalProperties" in node for node in _walk(ANSWER_SCHEMA))


def test_translation_preserves_the_contract():
    """Types, properties, enums and required fields must all survive."""
    converted = to_gemini_schema(ANSWER_SCHEMA)

    assert converted["type"] == "object"
    assert set(converted["required"]) == set(ANSWER_SCHEMA["required"])
    assert set(converted["properties"]) == set(ANSWER_SCHEMA["properties"])

    claim = converted["properties"]["claims"]["items"]
    assert set(claim["required"]) == {"text", "citations", "quote", "modality"}
    assert claim["properties"]["modality"]["enum"] == [
        "mandatory", "recommended", "optional", "descriptive",
    ]
    assert claim["properties"]["citations"]["items"]["type"] == "integer"


def test_verifier_enum_survives():
    verdict = to_gemini_schema(VERIFIER_SCHEMA)["properties"]["verdicts"]["items"]
    assert set(verdict["properties"]["verdict"]["enum"]) == {
        "supported", "partially_supported", "unsupported", "contradicted",
    }


def test_property_ordering_is_pinned_on_objects():
    """`answerable` must be emitted before `claims`.

    Deciding answerability first and then filling claims is a materially
    different task from writing claims and retrofitting the flag.
    """
    converted = to_gemini_schema(ANSWER_SCHEMA)
    ordering = converted["propertyOrdering"]
    assert ordering == list(ANSWER_SCHEMA["properties"].keys())
    assert ordering.index("answerable") < ordering.index("claims")


def test_translation_does_not_mutate_the_original():
    """The same schema object is reused by the Anthropic path in-process."""
    before = ANSWER_SCHEMA["additionalProperties"]
    to_gemini_schema(ANSWER_SCHEMA)
    assert ANSWER_SCHEMA["additionalProperties"] == before


def test_scalar_and_list_nodes_pass_through():
    assert to_gemini_schema({"type": "string"}) == {"type": "string"}
    assert to_gemini_schema(["a", "b"]) == ["a", "b"]
    assert to_gemini_schema(7) == 7


def test_adapted_schemas_validate_against_the_real_sdk():
    """Round-trip every schema through google-genai's own Schema validator.

    Catches drift between our adapter and the SDK's accepted dialect without
    needing an API key or a network call.
    """
    types = pytest.importorskip(
        "google.genai.types", reason="google-genai not installed"
    )
    for schema in ALL_SCHEMAS:
        validated = types.Schema(**to_gemini_schema(schema))
        assert validated.type is not None
        assert validated.required == schema["required"]
        assert list(validated.properties) == list(schema["properties"])


# --- client selection -------------------------------------------------------

def test_no_key_means_extractive_mode_not_a_crash():
    """A missing key is a supported configuration, not an error."""
    assert build_client("none", "", "gemini-2.5-pro") is None
    assert build_client("gemini", "", "gemini-2.5-pro") is None
    assert build_client("anthropic", "", "claude-opus-5") is None


def test_unknown_provider_degrades_instead_of_raising():
    assert build_client("llama", "key", "whatever") is None


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
