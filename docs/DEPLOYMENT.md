# Deploying to Vercel

The repository is ready to deploy as-is: the React frontend builds to a static
bundle and the Flask backend runs as a Python serverless function.

```bash
npm i -g vercel
cd telecom-rag-chatbot
vercel                       # preview deployment
vercel --prod                # production
```

Or import the Git repository at [vercel.com/new](https://vercel.com/new). Set
**Framework Preset → Other**; everything else comes from `vercel.json`.

### The one thing you must configure

Add **one** of these in **Project → Settings → Environment Variables** (not in
`vercel.json`, which is committed):

| Name | Provider |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio — used by default if both are set |
| `ANTHROPIC_API_KEY` | Anthropic |

Without either the deployment still works, in extractive mode — it retrieves and cites
verbatim clauses but generates nothing, and the false-premise abstention gates are
inactive. See *Which LLM, and why a key is needed at all* in the README.

---

## What the deployment looks like

```
        ┌───────────────────────────────┐
 /*  →  │  Static CDN                   │   frontend/dist  (Vite build)
        └───────────────────────────────┘
        ┌───────────────────────────────┐
/api/*→ │  Python serverless function   │   api/index.py → Flask app
        │  1024 MB · 60 s max duration  │
        │  /tmp: index + SQLite         │
        └───────────────────────────────┘
```

`vercel.json` rewrites every `/api/*` request to `api/index.py`. Vercel forwards the
original path, so the Flask blueprints keep their `/api/...` prefixes and the code is
identical locally and deployed.

---

## The four serverless constraints, and how each is handled

Serverless is not just "a server that scales". Four properties break a naive port, and
each needed a real fix rather than a config flag.

### 1. The filesystem is read-only

The deployment bundle cannot be written to. Two consequences:

- **`config.py`** creates its data directories through `_ensure_dir()`, which swallows
  `OSError`. Failing at import time would take the function down before it could report
  anything useful.
- **Uploads and reindex-from-disk return HTTP 409** with an explanation instead of a
  500 stack trace. The UI reads `capabilities.corpus_writable` and disables both
  buttons, showing a note that says how to change the corpus (commit and redeploy).

### 2. There is nowhere to persist the index

`/tmp` is writable but per-instance and wiped between cold starts, so a prebuilt index
can't be carried across invocations.

Rather than fight this, the index is **rebuilt from the bundled corpus on every cold
start** (`AUTO_INDEX_ON_BOOT=true`, `INDEX_DIR=/tmp/rag-index`). Measured cold-start
build for the bundled specs:

```
Indexing 144 chunks from 5 documents
Index built in 0.18s
```

That is cheap enough to be invisible next to a single LLM call, and it makes the
deployment genuinely stateless — no volume, no external vector database, no drift
between the committed corpus and what's actually being searched.

This only holds while the corpus is small. See *Scaling past the bundled corpus* below.

### 3. SSE streaming is buffered

Vercel serves WSGI apps through a proxy that buffers the response body, so an SSE
stream arrives as one blob after the handler returns. Streaming would still *work*, but
the user would watch a spinner for the full answer and then get everything at once —
strictly worse than not streaming.

So `/api/status` reports `capabilities.streaming: false` (from
`settings.supports_streaming`, which checks the `VERCEL` env var), and the frontend
switches to the blocking `POST /api/chat` endpoint. Same payload, same result, honest
progress indication.

Set `FORCE_STREAMING=1` to override if you move to a runtime that streams properly.

> The SSE **parser** in `client.js` handles buffered delivery fine — it splits on frame
> boundaries whenever bytes arrive. That's why the evaluation endpoint still works on
> Vercel: a retrieval-only run finishes in about a second, so buffering is unnoticeable.

### 4. Functions have a hard timeout

`maxDuration: 60` in `vercel.json` (the Hobby-plan ceiling; Pro allows 300).

The chat pipeline makes up to three model calls — rewrite, generate, verify — so a
complex question can approach that. If you see timeouts:

| Fix | Cost |
|---|---|
| `ENABLE_VERIFIER=false` | Removes ~40% of latency and the entailment check. Deterministic guards (citations, quotes, numerics) still run. |
| Upgrade to Pro, raise `maxDuration` to 300 | None, besides the plan |
| `RERANK_TOP_N=4` | Smaller prompts, slightly lower recall |

A **full generative evaluation run is 25 cases × 2–3 calls** and will always exceed the
limit. The Evaluation tab detects a serverless deployment and pre-ticks *retrieval only*
(fast, free, and the retrieval metrics are the ones that don't need a model). Run the
full evaluation locally.

---

## Bundle size

`requirements.txt` at the repo root is deliberately smaller than
`backend/requirements.txt`:

```
Flask · flask-cors · python-dotenv · numpy · google-genai · anthropic
```

Both LLM SDKs are listed because each is imported lazily inside its own client class —
the unused one costs bundle size but never import time. **Delete whichever provider you
are not deploying with** to shrink the bundle.

`pypdf` and `python-docx` are omitted because their imports live *inside* the loader
functions in `loaders.py`, so they're only needed to ingest `.pdf`/`.docx`. The bundled
corpus is `.txt`. **Add them back if you commit PDF or DOCX specifications** — they are
re-ingested on every cold start, so the readers must be present.

`.vercelignore` keeps `.venv/`, `tests/`, `scripts/` and the local index out of the
bundle. Note that `backend/eval/` is deliberately *not* ignored: `/api/evaluation`
imports `eval.metrics` and reads `eval/golden_set.json` at request time.

`sentence-transformers` and `faiss` (`requirements-ml.txt`) will **not** fit in a
serverless bundle — PyTorch alone is far past the 250 MB unzipped limit. Vercel
deployments always use the zero-dependency embedder and lexical reranker. Recalibrate
`MIN_RETRIEVAL_SCORE` for that pairing (the committed default, `0.42`, already is).

---

## Configuration reference

Set in `vercel.json` (no action needed):

| Variable | Value | Why |
|---|---|---|
| `INDEX_DIR` | `/tmp/rag-index` | Only writable path |
| `DB_PATH` | `/tmp/rag-chat.db` | Ephemeral, per-instance |
| `AUTO_INDEX_ON_BOOT` | `true` | Rebuild on cold start |
| `CORS_ORIGINS` | `*` | Same-origin anyway; harmless |

Set in the dashboard as needed:

| Variable | Default | Use |
|---|---|---|
| `GEMINI_API_KEY` | — | **One key required** for generative answers |
| `ANTHROPIC_API_KEY` | — | Alternative provider |
| `LLM_PROVIDER` | `auto` | `auto` prefers Gemini; force with `gemini` / `anthropic` |
| `ANSWER_MODEL` | per provider | `gemini-2.5-flash` is a cheaper, faster choice |
| `ENABLE_VERIFIER` | `true` | Set `false` if you hit timeouts |
| `MIN_RETRIEVAL_SCORE` | `0.42` | Raise for stricter abstention |
| `MIN_SUPPORT_RATIO` | `0.6` | Raise for stricter verification |
| `FORCE_STREAMING` | unset | Re-enable SSE on a streaming runtime |

> `CORPUS_DIR` is resolved **relative to `backend/`**, so leave it unset (default
> `data/corpus`). Setting it to `backend/data/corpus` resolves to
> `backend/backend/data/corpus` and finds nothing.

---

## Known limitations of the Vercel deployment

Worth stating before a reviewer finds them:

1. **Conversation history is ephemeral.** SQLite lives in `/tmp`, which is per-instance
   and wiped on cold start. Follow-ups within one warm instance work; the sidebar may
   look empty after idling. For durable history, point `DB_PATH` at a hosted Postgres
   and swap the `Store` implementation — the interface is small and deliberately
   isolated in `services/store.py`.
2. **Feedback is lost with it.** Same cause. The thumbs-up/down signal is genuinely
   valuable (it flags confidence miscalibration), so a real deployment should persist it
   externally.
3. **No incremental progress in the UI.** Covered above; a deliberate trade, not a bug.
4. **Cold starts pay the index build** (~0.2 s bundled) plus Python/numpy import (~1 s).
5. **The corpus is fixed at deploy time.** Uploading is disabled by design.

---

## Scaling past the bundled corpus

Rebuild-on-cold-start is right for ~150 chunks and wrong for a full 3GPP corpus. The
crossover is roughly where the build exceeds a second or two — a few thousand chunks.

Past that, pick one:

| Option | Change |
|---|---|
| **Commit a prebuilt index** | Un-ignore `backend/data/index/`, run `scripts/ingest.py` before committing, set `AUTO_INDEX_ON_BOOT=false`, point `INDEX_DIR` at the committed directory (read-only reads are fine). Bundle-size capped. |
| **Object storage** | Fetch `vectors.npy` + `chunks.json.gz` from S3/R2 into `/tmp` on cold start. |
| **Managed vector DB** | Replace `HybridIndex` with Pinecone/Qdrant/pgvector. Keep BM25 server-side or use the provider's hybrid search — **do not drop the lexical half**, it's what makes `5QI 82` retrieve the right row. |
| **Move off serverless** | A container on Fly.io / Render / Cloud Run restores streaming, persistent SQLite, uploads, and the ML extras in one step. |

For a production 3GPP assistant I'd choose the last option. The serverless deployment is
excellent for demonstrating the system; a long-lived process with a real disk suits a
retrieval workload with a large static index much better.

---

## Verifying a deployment

```bash
BASE=https://your-app.vercel.app

curl -s $BASE/api/health
curl -s $BASE/api/status | jq '{index_ready, chunk_count, mode, capabilities}'

# Should answer, with citations
curl -s -X POST $BASE/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"What is the Packet Delay Budget for 5QI 1?"}' \
  | jq '{status, confidence, answer, sources: [.sources[].citation_label]}'

# Should abstain
curl -s -X POST $BASE/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"What is the default value of timer T3599?"}' \
  | jq '{status, abstention: .abstention.type}'
```

Locally, the exact serverless code path can be exercised without deploying:

```bash
cd telecom-rag-chatbot
VERCEL=1 ./backend/.venv/bin/python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('e', 'api/index.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
c = m.app.test_client()
print(c.get('/api/status').get_json()['capabilities'])
print(c.post('/api/chat', json={'message':'What are the standardised SST values?'}).get_json()['status'])
"
```
