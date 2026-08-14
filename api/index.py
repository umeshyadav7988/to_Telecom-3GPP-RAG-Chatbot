"""Vercel serverless entrypoint.

Vercel's Python runtime discovers a WSGI callable named `app` in this module
and routes every request rewritten to `/api/index` into it (see vercel.json).
Flask then dispatches on the *original* request path, so the blueprints keep
their `/api/...` prefixes unchanged — no route duplication between local and
deployed builds.

Environment differences from a normal server are handled by configuration
rather than by branching in application code:

    INDEX_DIR=/tmp/rag-index    the deployment bundle is read-only; /tmp is not
    DB_PATH=/tmp/rag-chat.db    ephemeral, per-instance conversation history
    AUTO_INDEX_ON_BOOT=true     rebuild from the bundled corpus on cold start

All three are set in vercel.json so a deploy needs no manual configuration
beyond ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"

# `config` and `app` are imported as top-level modules by the backend package,
# which assumes backend/ is the working directory. It never is on Vercel.
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Defaults, not overrides: anything set in the Vercel dashboard still wins.
os.environ.setdefault("INDEX_DIR", "/tmp/rag-index")
os.environ.setdefault("DB_PATH", "/tmp/rag-chat.db")
os.environ.setdefault("AUTO_INDEX_ON_BOOT", "true")
os.environ.setdefault("CORS_ORIGINS", "*")

from app import create_app  # noqa: E402  (must follow the sys.path setup)

app = create_app()

# Some Vercel Python builds look for `handler` instead of `app`.
handler = app
