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

# --- Serve the built SPA from Flask ------------------------------------------
#
# Vercel classifies this repo as a Python backend-framework project and routes
# *every* path into this function, ignoring the `outputDirectory` in
# vercel.json -- so frontend/dist is never published as static assets and the
# edge has nothing to serve. Flask therefore serves the build itself; the dist
# tree reaches the bundle via `includeFiles`.
#
# Locally this block is inert: `npm run dev` serves the frontend from Vite and
# proxies /api to this app, and frontend/dist need not exist.

DIST = ROOT / "frontend" / "dist"

if DIST.is_dir():
    from flask import abort, send_from_directory  # noqa: E402

    def _serve_spa(path: str = ""):
        # An unmatched /api/* path is a broken API call, not a client route.
        # Without this it would fall through to index.html and hand the caller
        # HTML with a 200, turning a 404 into a silent success.
        if path.startswith("api/"):
            abort(404)
        # A real file (hashed asset, favicon) wins; anything else is a client
        # route and gets index.html so the SPA router can resolve it.
        if path and (DIST / path).is_file():
            return send_from_directory(DIST, path)
        return send_from_directory(DIST, "index.html")

    # create_app() already owns "/" with the JSON API banner. Deployed, that
    # path belongs to the SPA, so swap the view rather than adding a second
    # rule for "/" (Flask rejects duplicate rules). The banner stays reachable
    # through the /api/* blueprints.
    app.view_functions["root"] = _serve_spa

    # Static rules like /api/health still win over this converter rule, so the
    # API keeps precedence regardless of registration order.
    app.add_url_rule("/<path:path>", endpoint="spa", view_func=_serve_spa)

# Some Vercel Python builds look for `handler` instead of `app`.
handler = app
