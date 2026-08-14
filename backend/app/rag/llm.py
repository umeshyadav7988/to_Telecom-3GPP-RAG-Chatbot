"""LLM clients behind one provider-agnostic interface.

The RAG pipeline needs exactly one capability from a model: *give me JSON that
matches this schema*. Everything else — retrieval, citation validation, the
numeric guard, abstention gates 1 and 3 — is provider-independent, so the
provider surface is deliberately kept to a single method.

Two implementations:

* `AnthropicClient` — Claude, via `output_config.format` structured outputs.
* `GeminiClient`    — Gemini, via `response_schema` + JSON response MIME type.

Neither is required. With no key configured the pipeline runs in extractive
mode: it retrieves and cites verbatim clauses but performs no generation, and
`build_client()` returns None to say so.

Both classes handle the same three realities:
  * structured JSON output, since free-text parsing is a hallucination source
    in its own right
  * graceful degradation when the installed SDK predates an optional parameter
  * safety refusals, which are a normal response on both providers rather than
    an exception, and must not be mistaken for an answer
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """No API key configured, or the provider SDK is not installed."""


class LLMRefusal(RuntimeError):
    """The provider's safety systems declined the request."""

    def __init__(self, category: str | None, explanation: str | None):
        self.category = category
        self.explanation = explanation
        super().__init__(f"Model declined the request (category={category!r})")


@dataclass
class LLMResult:
    data: dict
    raw_text: str
    model: str
    provider: str = ""
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0


def _parse_json(text: str, model: str) -> dict:
    if not text or not text.strip():
        raise RuntimeError(f"{model} returned an empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # Structured output makes this near-impossible, but a truncated or
        # degraded response must not surface as a 500 with a stack trace.
        raise RuntimeError(f"{model} returned non-JSON output: {text[:400]!r}") from exc


class BaseLLMClient(ABC):
    """One structured JSON call. That is the entire provider contract."""

    provider: str = "base"

    @abstractmethod
    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        model: str | None = None,
        max_tokens: int = 8000,
        effort: str = "medium",
        cache_system: bool = True,
    ) -> LLMResult:
        ...


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, api_key: str, *, default_model: str = "claude-opus-5"):
        if not api_key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("The `anthropic` package is not installed") from exc

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.default_model = default_model

    @staticmethod
    def _first_text(response) -> str:
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""

    @staticmethod
    def _usage(response) -> dict:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        return {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        }

    def _create(self, **kwargs):
        """Call the API, progressively dropping optional params on 400s.

        Structured outputs, adaptive thinking and `effort` have all moved
        between SDK versions. Rather than pin an exact release, degrade: answer
        quality drops slightly without `effort`, but the service stays up. Each
        removal is logged so an operator can see what happened.
        """
        attempts = [
            kwargs,
            {k: v for k, v in kwargs.items() if k != "thinking"},
            {
                **{k: v for k, v in kwargs.items() if k != "thinking"},
                "output_config": {
                    k: v for k, v in kwargs.get("output_config", {}).items() if k == "format"
                },
            },
            {k: v for k, v in kwargs.items() if k not in ("thinking", "output_config")},
        ]

        last_error: Exception | None = None
        for attempt_no, params in enumerate(attempts):
            params = {k: v for k, v in params.items() if v not in (None, {}, [])}
            try:
                return self._client.messages.create(**params)
            except self._anthropic.BadRequestError as exc:
                last_error = exc
                logger.warning(
                    "Request rejected (attempt %d/%d): %s - retrying with fewer optional params",
                    attempt_no + 1,
                    len(attempts),
                    getattr(exc, "message", exc),
                )
        raise last_error  # type: ignore[misc]

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        model: str | None = None,
        max_tokens: int = 8000,
        effort: str = "medium",
        cache_system: bool = True,
    ) -> LLMResult:
        model = model or self.default_model
        system_param = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if cache_system
            else system
        )

        started = time.perf_counter()
        response = self._create(
            model=model,
            max_tokens=max_tokens,
            system=system_param,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(
                getattr(details, "category", None), getattr(details, "explanation", None)
            )

        text = self._first_text(response)
        if stop_reason == "max_tokens" and not text.rstrip().endswith("}"):
            raise RuntimeError(
                "Model output was truncated by max_tokens before the JSON closed; "
                "raise MAX_ANSWER_TOKENS."
            )

        return LLMResult(
            data=_parse_json(text, model),
            raw_text=text,
            model=model,
            provider=self.provider,
            usage=self._usage(response),
            latency_ms=round(latency_ms, 1),
        )


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

# Gemini's `response_schema` accepts an OpenAPI 3.0 subset, not full JSON
# Schema. `additionalProperties` in particular is rejected — and our schemas
# set it to false everywhere, because Claude's strict mode requires it.
_UNSUPPORTED_SCHEMA_KEYS = {
    "additionalProperties",
    "$schema",
    "$id",
    "definitions",
    "$defs",
    "propertyNames",
    "patternProperties",
}


def to_gemini_schema(schema):
    """Convert a JSON Schema to the dialect Gemini accepts.

    Drops unsupported keywords and pins property order, so the model emits
    fields in the order the schema declares them. That matters here: `claims`
    is easier for the model to fill correctly after it has committed to
    `answerable`, rather than deciding answerability retrospectively.
    """
    if isinstance(schema, dict):
        converted = {
            key: to_gemini_schema(value)
            for key, value in schema.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS
        }
        if converted.get("type") == "object" and isinstance(converted.get("properties"), dict):
            converted.setdefault("propertyOrdering", list(converted["properties"].keys()))
        return converted
    if isinstance(schema, list):
        return [to_gemini_schema(item) for item in schema]
    return schema


class GeminiClient(BaseLLMClient):
    provider = "gemini"

    # Roles map onto Gemini's thinking budget. The verifier and the rewriter
    # make bounded judgements where extra deliberation buys nothing, so we turn
    # thinking off there — it is the single biggest cost lever on 2.5 models.
    _EFFORT_THINKING_BUDGET = {"low": 0, "medium": -1, "high": -1, "max": -1}

    def __init__(self, api_key: str, *, default_model: str = "gemini-2.5-pro"):
        if not api_key:
            raise LLMUnavailable("GEMINI_API_KEY is not set")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable(
                "The `google-genai` package is not installed (pip install google-genai)"
            ) from exc

        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.default_model = default_model

    @staticmethod
    def _usage(response) -> dict:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {}
        return {
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            "thinking_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
            "cache_read_input_tokens": getattr(usage, "cached_content_token_count", 0) or 0,
        }

    def _check_refusal(self, response, model: str) -> None:
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason:
            raise LLMRefusal(str(block_reason), getattr(feedback, "block_reason_message", None))

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise LLMRefusal(None, "The model returned no candidates.")

        finish_reason = str(getattr(candidates[0], "finish_reason", "") or "")
        if "SAFETY" in finish_reason or "PROHIBITED" in finish_reason or "BLOCK" in finish_reason:
            raise LLMRefusal(finish_reason, "Blocked by the provider's safety filters.")
        if "MAX_TOKENS" in finish_reason:
            raise RuntimeError(
                f"{model} hit max_output_tokens before closing the JSON; raise MAX_ANSWER_TOKENS."
            )

    def _generate(self, *, model: str, system: str, user: str, schema: dict,
                  max_tokens: int, effort: str):
        """Call Gemini, dropping optional config on rejection.

        Same degradation ladder as the Anthropic path: `thinking_config` is
        model-dependent (Pro cannot disable thinking) and newer than some
        installed SDKs, so a rejection retries without it rather than failing
        the request.
        """
        types = self._types
        base = dict(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=to_gemini_schema(schema),
            max_output_tokens=max_tokens,
            # Deterministic decoding: this is extraction and judgement, not
            # creative writing. Sampling here would only add variance to
            # claims that must match the cited clause exactly.
            temperature=0.0,
        )

        budget = self._EFFORT_THINKING_BUDGET.get(effort, -1)
        attempts = []
        if budget == 0 and "flash" in model:
            # Only flash models accept a zero budget; Pro enforces a minimum.
            attempts.append({**base, "thinking_config": types.ThinkingConfig(thinking_budget=0)})
        attempts.append(base)
        attempts.append({k: v for k, v in base.items() if k != "response_schema"})

        last_error: Exception | None = None
        for attempt_no, config in enumerate(attempts):
            try:
                return self._client.models.generate_content(
                    model=model,
                    contents=user,
                    config=types.GenerateContentConfig(**config),
                )
            except Exception as exc:
                # Only retry on request-shape rejections; auth, quota and
                # network errors will not improve by dropping a parameter.
                message = str(exc).lower()
                retryable = any(
                    token in message
                    for token in ("invalid", "unsupported", "unknown field", "400", "not supported")
                )
                last_error = exc
                if not retryable or attempt_no == len(attempts) - 1:
                    raise
                logger.warning(
                    "Gemini rejected the request config (attempt %d/%d): %s - retrying simpler",
                    attempt_no + 1,
                    len(attempts),
                    exc,
                )
        raise last_error  # type: ignore[misc]

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        model: str | None = None,
        max_tokens: int = 8000,
        effort: str = "medium",
        cache_system: bool = True,
    ) -> LLMResult:
        model = model or self.default_model

        started = time.perf_counter()
        response = self._generate(
            model=model,
            system=system,
            user=user,
            schema=schema,
            max_tokens=max_tokens,
            effort=effort,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        self._check_refusal(response, model)
        text = getattr(response, "text", "") or ""

        return LLMResult(
            data=_parse_json(text, model),
            raw_text=text,
            model=model,
            provider=self.provider,
            usage=self._usage(response),
            latency_ms=round(latency_ms, 1),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_client(provider: str, api_key: str, default_model: str) -> BaseLLMClient | None:
    """Return a client, or None to signal extractive-only operation.

    A missing key is a supported configuration, not an error: retrieval,
    citations, the numeric guard and abstention gate 1 all work without one.
    """
    if provider == "none" or not api_key:
        logger.warning(
            "No LLM provider configured - running in extractive mode (retrieval and "
            "citation only, no generation). Set GEMINI_API_KEY or ANTHROPIC_API_KEY."
        )
        return None

    try:
        if provider == "gemini":
            client = GeminiClient(api_key, default_model=default_model)
        elif provider == "anthropic":
            client = AnthropicClient(api_key, default_model=default_model)
        else:
            logger.error("Unknown LLM provider %r; falling back to extractive mode.", provider)
            return None
    except LLMUnavailable as exc:
        logger.warning("LLM disabled: %s", exc)
        return None

    logger.info("LLM provider: %s (default model %s)", client.provider, default_model)
    return client
