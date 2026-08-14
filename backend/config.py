"""Central configuration, loaded once from the environment.

Everything tunable about the RAG pipeline lives here so that behaviour can be
changed without touching code — which matters for a system whose selling point
is calibrated, measurable abstention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _path(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    p = Path(raw)
    return p if p.is_absolute() else BASE_DIR / p


# Per-provider model defaults. Roles differ in what they actually need:
#   answer   — reading comprehension over technical prose. Worth the strong model.
#   verifier — a bounded yes/no entailment judgement per claim. A fast model is
#              both sufficient and much cheaper, and it runs on every turn.
#   rewrite  — resolve a pronoun. Trivial.
PROVIDER_DEFAULTS = {
    "anthropic": {
        "answer": "claude-opus-5",
        "verifier": "claude-opus-5",
        "rewrite": "claude-opus-5",
    },
    "gemini": {
        "answer": "gemini-2.5-pro",
        "verifier": "gemini-2.5-flash",
        "rewrite": "gemini-2.5-flash",
    },
}


@dataclass(frozen=True)
class Settings:
    # --- LLM provider ------------------------------------------------------
    # "auto" picks whichever key is present (Gemini wins if both are set, since
    # configuring it is a deliberate act). Force with LLM_PROVIDER.
    llm_provider_setting: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "auto").strip().lower()
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    # GOOGLE_API_KEY is the variable the google-genai SDK reads by default, so
    # accept it as an alias rather than making people set the same value twice.
    gemini_api_key: str = field(
        default_factory=lambda: (
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        ).strip()
    )

    # Empty => fall back to the provider default (see PROVIDER_DEFAULTS).
    answer_model_setting: str = field(default_factory=lambda: os.getenv("ANSWER_MODEL", "").strip())
    verifier_model_setting: str = field(
        default_factory=lambda: os.getenv("VERIFIER_MODEL", "").strip()
    )
    rewrite_model_setting: str = field(
        default_factory=lambda: os.getenv("REWRITE_MODEL", "").strip()
    )

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = field(default_factory=lambda: _int("RETRIEVAL_TOP_K", 24))
    rerank_top_n: int = field(default_factory=lambda: _int("RERANK_TOP_N", 6))
    dense_weight: float = field(default_factory=lambda: _float("DENSE_WEIGHT", 1.0))
    sparse_weight: float = field(default_factory=lambda: _float("SPARSE_WEIGHT", 1.0))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))

    # --- Chunking ----------------------------------------------------------
    chunk_target_chars: int = field(default_factory=lambda: _int("CHUNK_TARGET_CHARS", 1400))
    chunk_overlap_chars: int = field(default_factory=lambda: _int("CHUNK_OVERLAP_CHARS", 200))
    chunk_min_chars: int = field(default_factory=lambda: _int("CHUNK_MIN_CHARS", 120))

    # --- Anti-hallucination ------------------------------------------------
    # Calibrated on the bundled golden set with the lexical-semantic reranker
    # (scripts/calibrate_threshold.py). Re-run that script after switching
    # reranker or corpus — the two scorers produce different distributions.
    min_retrieval_score: float = field(default_factory=lambda: _float("MIN_RETRIEVAL_SCORE", 0.42))
    min_support_ratio: float = field(default_factory=lambda: _float("MIN_SUPPORT_RATIO", 0.6))
    enable_verifier: bool = field(default_factory=lambda: _bool("ENABLE_VERIFIER", True))
    enable_numeric_guard: bool = field(default_factory=lambda: _bool("ENABLE_NUMERIC_GUARD", True))
    # Deterministic false-premise detection. Free, and the only defence
    # against "timer T3599" that works with no model configured.
    enable_premise_guard: bool = field(default_factory=lambda: _bool("ENABLE_PREMISE_GUARD", True))

    # --- Generation --------------------------------------------------------
    max_answer_tokens: int = field(default_factory=lambda: _int("MAX_ANSWER_TOKENS", 8000))
    max_verifier_tokens: int = field(default_factory=lambda: _int("MAX_VERIFIER_TOKENS", 6000))
    max_history_turns: int = field(default_factory=lambda: _int("MAX_HISTORY_TURNS", 6))

    # --- Paths -------------------------------------------------------------
    corpus_dir: Path = field(default_factory=lambda: _path("CORPUS_DIR", "data/corpus"))
    index_dir: Path = field(default_factory=lambda: _path("INDEX_DIR", "data/index"))
    db_path: Path = field(default_factory=lambda: _path("DB_PATH", "data/chat.db"))

    # --- Server ------------------------------------------------------------
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("PORT", 5001))
    debug: bool = field(default_factory=lambda: _bool("FLASK_DEBUG", False))
    cors_origins: tuple = field(
        default_factory=lambda: tuple(
            o.strip()
            for o in os.getenv(
                "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",")
            if o.strip()
        )
    )

    # --- Deployment ---------------------------------------------------------
    # Build the index at boot when none is found on disk. Essential on
    # serverless platforms, where the filesystem is read-only and reset on
    # every cold start, so a prebuilt index cannot be persisted between
    # invocations. The bundled corpus indexes in well under a second.
    auto_index_on_boot: bool = field(default_factory=lambda: _bool("AUTO_INDEX_ON_BOOT", False))

    # --- Provider resolution ------------------------------------------------

    @property
    def llm_provider(self) -> str:
        """Resolved provider: "anthropic", "gemini", or "none"."""
        forced = self.llm_provider_setting
        if forced == "anthropic":
            return "anthropic" if self.anthropic_api_key else "none"
        if forced == "gemini":
            return "gemini" if self.gemini_api_key else "none"
        if self.gemini_api_key:
            return "gemini"
        if self.anthropic_api_key:
            return "anthropic"
        return "none"

    @property
    def api_key(self) -> str:
        return {
            "gemini": self.gemini_api_key,
            "anthropic": self.anthropic_api_key,
        }.get(self.llm_provider, "")

    def _model_for(self, role: str, override: str) -> str:
        if override:
            return override
        provider = self.llm_provider
        if provider == "none":
            # Nothing will be called, but keep a sensible label for /api/status.
            provider = "gemini" if self.llm_provider_setting == "gemini" else "anthropic"
        return PROVIDER_DEFAULTS[provider][role]

    @property
    def answer_model(self) -> str:
        return self._model_for("answer", self.answer_model_setting)

    @property
    def verifier_model(self) -> str:
        return self._model_for("verifier", self.verifier_model_setting)

    @property
    def rewrite_model(self) -> str:
        return self._model_for("rewrite", self.rewrite_model_setting)

    @property
    def llm_enabled(self) -> bool:
        """False => extractive-only mode (retrieve + cite, never generate)."""
        return self.llm_provider != "none"

    @property
    def is_serverless(self) -> bool:
        """True on Vercel / AWS Lambda style platforms."""
        return bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

    @property
    def supports_streaming(self) -> bool:
        """Whether Server-Sent Events survive the hosting layer.

        Vercel's Python runtime serves WSGI apps through a buffering proxy, so
        an SSE response is delivered as one blob after the handler returns.
        The frontend reads this flag and falls back to the blocking endpoint
        rather than appearing to hang for the length of the answer.
        """
        if os.getenv("FORCE_STREAMING"):
            return True
        return not self.is_serverless

    @property
    def read_only_filesystem(self) -> bool:
        """Corpus uploads and reindex-from-disk are unavailable when True."""
        return self.is_serverless


settings = Settings()


def _ensure_dir(path: Path) -> None:
    """Create a directory, tolerating a read-only filesystem.

    On serverless platforms the deployment bundle is read-only. Failing at
    import time would take the whole function down before it could report a
    useful error, so a missing writable directory is downgraded to a runtime
    concern (the index lives in /tmp there, and uploads are rejected with a
    clear message).
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


_ensure_dir(settings.corpus_dir)
_ensure_dir(settings.index_dir)
_ensure_dir(settings.db_path.parent)
