# Telecom 3GPP RAG Chatbot

A Retrieval-Augmented Generation assistant for 3GPP telecom specifications, engineered
around one goal: **minimal to near-zero hallucinations**.

React frontend · Flask backend · hybrid retrieval · multi-stage grounding verification ·
measurable evaluation harness.

> ### Runs with no API key
>
> Clone, install, ingest, go. No key, no GPU, no network, no signup — and it is not a
> stub: **84% pass rate and 14.3% hallucination rate** on the golden set with zero
> models involved. Retrieval, clause-level citations, answer extraction and two of the
> four abstention gates are entirely deterministic.
>
> Adding a key ([Gemini](https://aistudio.google.com/apikey) or
> [Anthropic](https://console.anthropic.com/settings/keys)) buys synthesis across
> clauses and the two defences that need reading comprehension. See
> [**Running with and without an LLM**](#running-with-and-without-an-llm).

---

## The core idea

Most RAG systems treat hallucination as a prompting problem. It isn't. It's a
*system* problem, and it has three distinct causes that need three distinct defences:

| Cause | Example | Defence in this system |
|---|---|---|
| The corpus doesn't contain the answer, but the model answers anyway | "What's the max transmit power of an LTE-M Cat-M1 device?" | **Gate 1** — retrieval-score threshold, abstains *before* any LLM call |
| The corpus contains the *topic* but not the *fact* | "What is the default value of timer T3599?" (T3599 doesn't exist) | **Gate 1.5** — deterministic entity check (no model), then **Gate 2** the generator setting `answerable: false`, then **Gate 3** entailment |
| The answer is mostly right but a value drifted | "T3512 defaults to 42 minutes" (it's 54) | **Numeric guard** — every number/identifier must literally occur in the cited clause |

The system also refuses to be graded only on hallucination rate. Refusing every
question would score a perfect 0%, so the evaluation reports **over-abstention rate**
directly beside it. Both numbers only mean something together.

> **[`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md)** traces three real queries end to end —
> one answered, one caught by Gate 1, and one false-premise question that Gate 1
> provably *cannot* catch — with the actual scores from a live run.

---

## Quick start

Requirements: **Python 3.10+**, **Node 18+**. No GPU, no API key, no network access
(a condensed 3GPP corpus is bundled).

### 1. Backend

```bash
cd backend

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/ingest.py --stats     # build the index  (~0.2 s)
python run.py                        # serves on http://localhost:5001
```

That's it — no `.env` needed. The app boots in **extractive mode** and answers real
questions with real citations.

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev                          # http://localhost:5173
```

Open **http://localhost:5173**. Vite proxies `/api` to the backend, so there is no
CORS setup and no API base URL to configure.

### 3. Verify

```bash
cd backend
python -m pytest tests/ -q           # 58 tests, ~0.4 s
python eval/run_eval.py              # golden-set evaluation (no key needed)
```

### 4. Optional — add a model

```bash
cp .env.example .env
# set ONE of:
#   GEMINI_API_KEY=...      https://aistudio.google.com/apikey
#   ANTHROPIC_API_KEY=...   https://console.anthropic.com/settings/keys
```

Restart the backend. The sidebar switches from `extractive mode` to
`gemini · gemini-2.5-pro` (or the Anthropic equivalent).

---

## Running with and without an LLM

The system has two modes. Both are real; the difference is *which* defences are
available, not whether it works.

| | **Extractive** (no key) | **Generative** (one key) |
|---|---|---|
| Setup | nothing | one env var |
| Answer style | verbatim sentences and table rows | synthesised prose |
| Cost / latency | free · ~3 ms | per-token · ~4–8 s |
| Clause-level citations | ✅ | ✅ |
| Gate 1 — off-corpus | ✅ | ✅ |
| Gate 1.5 — false premise | ✅ deterministic | ✅ + entailment |
| Gate 2 — insufficient context | ❌ | ✅ |
| Check 1–3 — citations, quotes, numbers | ✅ | ✅ |
| Check 4 — entailment | ❌ | ✅ |
| Cross-clause synthesis | ❌ | ✅ |
| **Golden-set pass rate** | **84.0%** | *see note below* |
| **Hallucination rate** | **14.3%** | *see note below* |

### Extractive mode — what makes it work

**Most of the system needs no model at all.** Four deterministic components carry it:

| | What it does |
|---|---|
| **Hybrid retrieval + Gate 1** | Finds the clause; refuses off-corpus questions |
| **Premise guard** (Gate 1.5) | Refuses false premises — [`premise_guard.py`](backend/app/rag/premise_guard.py) |
| **Answer extraction** | Returns the *fact*, not the page — [`extractive.py`](backend/app/rag/extractive.py) |
| **Checks 1–3** | Citation validity, quote provenance, numeric guard |

**The premise guard is the numeric guard turned around** — applied to the *question*
instead of the answer. If a question asserts a specific alphanumeric identifier
(`T3599`, `5QI 91`, `TS 24.601`) that appears nowhere in the clauses just retrieved,
then the thing being asked about does not exist. That is a string comparison, not an
inference. It lifted abstention recall from 42.9% → 85.7% at zero cost and **zero**
over-abstention.

**Answer extraction** scores individual sentences and table rows instead of dumping
1400-character chunks. Four spec-specific behaviours do the work, each added in
response to a measured failure:

| Behaviour | Why a spec needs it |
|---|---|
| **Row-key matching** | `1 \| GBR \| 20 \| 100 ms` shares *no words* with "what is the PDB for 5QI 1?" — the columns are in the header and the entity is the bare first cell |
| **Header carry-along** | That row is meaningless without `5QI Value \| Resource Type \| … \| Packet Delay Budget` |
| **List continuation** | "The following SST values are standardised:" is the wrong half of the answer without the values |
| **Qualifier conflict** | Profile A and Profile B are near-identical sentences differing by one character every tokenizer discards |

Measured effect of the three no-LLM changes together:

| Metric (extractive) | Before | After |
|---|---:|---:|
| Overall pass rate | 64.0% | **84.0%** |
| Answerable pass rate | 72.2% | **83.3%** |
| Hallucination rate | 57.1% | **14.3%** |
| Abstention recall | 42.9% | **85.7%** |
| Over-abstention | 0.0% | **0.0%** |
| MRR | 0.889 | **0.917** |

Extractive mode caps confidence at **0.5** by design: the text is genuine spec text,
but nothing verified that it *answers the question*. Reporting high confidence there
would be exactly the miscalibration this project exists to prevent.

### Where extractive mode stops

One golden-set case still fails without a model, and it is the instructive one:

> *"Why did 3GPP remove network slicing support in Release 17?"*

Every entity is real — `3GPP`, `network slicing` and `Release 17` all appear in the
corpus. Only the asserted **relationship** is false. No string comparison reaches that.
There is a regression test asserting the guard *cannot* catch it
(`test_cannot_catch_a_false_causal_premise`), so the limitation stays visible in CI
instead of quietly drifting.

That residue is precisely what a model buys:

| Needs a model | Why it can't be done without one |
|---|---|
| Cross-clause synthesis | Extraction returns passages, not an argument |
| **Gate 2** — "the sources don't contain this" | Judging *sufficiency* against retrieved text |
| **Check 4** — per-claim entailment | Judging whether text *entails* a claim, vs merely sharing tokens |

### Choosing a provider

Set **one** of these in `backend/.env`:

```bash
GEMINI_API_KEY=...      # Google AI Studio — https://aistudio.google.com/apikey
ANTHROPIC_API_KEY=...   # Anthropic Console — https://console.anthropic.com/settings/keys
```

Gemini is used when both are present; force either with `LLM_PROVIDER=gemini|anthropic`.
Model defaults are per-provider and **role-aware** — the verifier runs on every turn and
only makes bounded yes/no judgements, so it gets the fast model:

| Role | Gemini | Anthropic |
|---|---|---|
| Answer | `gemini-2.5-pro` | `claude-opus-5` |
| Verifier | `gemini-2.5-flash` | `claude-opus-5` |
| Rewrite | `gemini-2.5-flash` | `claude-opus-5` |

Override with `ANSWER_MODEL` / `VERIFIER_MODEL` / `REWRITE_MODEL`. `GOOGLE_API_KEY` is
accepted as an alias for `GEMINI_API_KEY` (it is what the Google SDK reads natively).
The active mode and provider are shown in the UI sidebar.

### Better quality, still keyless

```bash
pip install -r requirements-ml.txt   # ~2 GB (PyTorch); no API, no key, no per-token cost
python scripts/ingest.py             # rebuild — a different embedder needs new vectors
python scripts/calibrate_threshold.py
```

Swaps in `BAAI/bge-small-en-v1.5` embeddings and a `ms-marco-MiniLM` cross-encoder
reranker, auto-detected. A local NLI model (e.g.
`MoritzLaurer/DeBERTa-v3-base-mnli`, ~400 MB) would slot into Check 4 the same way and
close most of the remaining gap — the verifier interface is one method wide. Not wired
up here because it cannot ship in a serverless bundle, but it is the natural next step
for a self-hosted, fully-offline deployment.

---

## Architecture

```
                    ┌──────────────────────── React (Vite) ────────────────────────┐
                    │  Chat  ·  Citation inspector  ·  Live pipeline trace         │
                    │  Corpus manager  ·  Evaluation dashboard                     │
                    └───────────────────────────┬──────────────────────────────────┘
                                                │  SSE  (stage-by-stage)
                    ┌───────────────────────────▼──────────────────────────────────┐
                    │                      Flask API                               │
                    └───────────────────────────┬──────────────────────────────────┘
                                                │
   INGEST (offline)                             │  QUERY (per turn)
   ─────────────────                            │  ───────────────
   .pdf/.docx/.txt                              ▼
        │                              ┌── contextualise ──┐  resolve follow-ups
        ▼                              │                   │
   clause-aware chunker                ▼                   │
   TS 23.501 §5.15.2.1                 hybrid retrieval    │
        │                              ├─ dense (cosine)   │
        ├──► dense vectors             ├─ BM25 (lexical)   │
        └──► BM25 statistics           └─ RRF fusion       │
                                              │            │
                                            rerank         │
                                              │            │
                              ╔═══════════════▼═══════════════╗
                              ║ GATE 1    score < threshold?  ║──► abstain   ⚙ no model
                              ╚═══════════════┬═══════════════╝
                              ╔═══════════════▼═══════════════╗
                              ║ GATE 1.5  asserted entity     ║──► abstain   ⚙ no model
                              ║           missing from text?  ║
                              ╚═══════════════┬═══════════════╝
                                              │
                        ┌─────────────────────┴─────────────────────┐
                 no key │                                           │ key set
                        ▼                                           ▼
              ⚙ answer extraction                          grounded generation
                sentences + table rows                     claims + citations + quotes
                        │                                           │
                        │                          ╔════════════════▼═══════════════╗
                        │                          ║ GATE 2  answerable == false?   ║──► abstain
                        │                          ╚════════════════┬═══════════════╝
                        └─────────────────────┬─────────────────────┘
                                              ▼
                                    verification (4 checks)
                              ⚙ ① citation validity   ⚙ ② quote provenance
                              ⚙ ③ numeric guard          ④ LLM entailment
                                              │
                              ╔═══════════════▼═══════════════╗
                              ║ GATE 3  support < minimum?    ║──► abstain
                              ╚═══════════════┬═══════════════╝
                                              ▼
                                cited answer + calibrated confidence

                              ⚙ = deterministic, runs with no API key
```

---

## Design decisions

### 1. Clause-aware chunking, not character splitting

A generic recursive splitter cuts every N characters. For 3GPP that produces chunks
whose provenance is *"page 412"* — useless for a system whose premise is verifiable
citations. Worse, it routinely severs a requirement from its scoping clause, which is
the single richest source of confident wrong answers (*"the UE shall…"* — under which
condition? in which state?).

`app/rag/chunking.py` instead recovers the document's clause tree and emits chunks that
never cross a clause boundary and carry the full ancestor breadcrumb:

```
TS 23.501 §5.15.2.1 — S-NSSAI
  5 Overall description > 5.15 Network slicing > 5.15.2 Identification > 5.15.2.1 S-NSSAI
```

An engineer can open the real spec and check that in seconds. **That verifiability is
the anti-hallucination property** — a fabricated claim has nowhere to hide behind a
vague page reference.

It also handles lettered annex clauses (`C.3.4`), which matters because TS 33.501 keeps
the entire SUPI protection scheme in Annex C. Without that, citations degrade from
`§C.3.4` to `§Annex C`. There's a regression test for exactly this.

### 2. Hybrid retrieval — BM25 is not optional here

3GPP queries are full of exact identifiers: `5QI 82`, `N3IWF`, `T3512`, `RRC_INACTIVE`.
A dense embedder's nearest neighbours are semantically adjacent but factually wrong —
returning the *5QI 83* row instead of *5QI 82* is a hallucination waiting to happen.
BM25 nails exact-term matching; dense covers paraphrase.

They're fused with **Reciprocal Rank Fusion** rather than score interpolation, because
cosine similarity and BM25 scores live on incomparable scales and per-query
normalisation is unstable when one retriever returns nothing. Ranks are always
comparable.

### 3. The reranker exists to make abstention possible

RRF gives a good *ordering* but rank-derived scores are uncalibrated — the top hit
always scores ~`1/(k+1)` whether or not the corpus answers the question. **You cannot
threshold on that, and a system that cannot threshold cannot abstain.**

So the reranker emits a score in `[0,1]` that means something absolute. The fallback
scorer is deliberately interpretable and its components are exposed in the API:

```
score = 0.45 · dense_cosine + 0.40 · query_term_coverage + 0.15 · clause_title_match
```

Coverage is weighted heavily on purpose: in a specification corpus, a passage that
doesn't mention the query's identifiers is almost never the right passage, however
close it sits in embedding space.

### 4. Answers are claims, not paragraphs

The generator returns structured JSON — a list of atomic claims, each with citations, a
**verbatim quote**, and a normative modality (`shall`/`should`/`may`/descriptive).

A paragraph can be 90% correct and 10% invented, and there's no way to act on that. A
claim is either supported by its cited clause or it isn't, and an unsupported one can
be removed without destroying the rest of the answer.

Forcing a verbatim quote per claim is the cheapest high-value trick available: a model
that must copy a literal span before asserting something can't smoothly invent a timer
value, and the quote is checkable with `in` — no second model required.

### 5. Four verification checks, three of them deterministic

| # | Check | Catches | Needs an LLM? |
|---|---|---|---|
| 1 | Citation validity | Invented `[S9]` markers pointing at sources never retrieved | No |
| 2 | Quote provenance | Fabricated supporting evidence | No |
| 3 | Numeric guard | Drifted numbers, timers, identifiers — the costliest error class | No |
| 4 | Entailment | Subtle misreadings that pass all three above | Yes |

Checks 1–3 cannot be fooled by a confident model. Check 4 catches what they
structurally cannot see. `tests/test_pipeline_guardrails.py` proves each one fires by
substituting a model that hallucinates *on purpose*.

The numeric guard distinguishes token classes by fabrication risk. Alphanumeric
identifiers (`5QI`, `T3512`, `NEA3`) are hard-checked — they're values in disguise.
Pure acronyms (`AMF`, `PDB`) are soft-checked, because a claim may legitimately
abbreviate a term the clause spells out; entailment judges those instead.

### 6. The retrieval gate cannot do this alone — and we measured it

Running `scripts/calibrate_threshold.py` against the golden set gives:

```
ANSWERABLE   (18 cases)  min 0.456   max 0.732
UNANSWERABLE ( 7 cases)  min 0.307   max 0.542   ← overlaps!

RECOMMENDED  MIN_RETRIEVAL_SCORE=0.42
  catches 3/7 unanswerable questions before any LLM call
  wrongly rejects 0/18 answerable questions

  2 unanswerable questions score ABOVE the lowest answerable one:
    trap-002  0.478   "PDB for 5QI 91?"        (5QI 91 doesn't exist)
    trap-001  0.542   "default value of T3599?" (T3599 doesn't exist)
```

**The distributions are not separable.** Those two questions retrieve the correct table
with high confidence — the topic *is* in the corpus, the asserted fact simply isn't.
No threshold anywhere will catch them.

This is the empirical justification for the whole multi-gate architecture: a
retrieval-threshold-only system is structurally incapable of catching false-premise
questions, which is precisely the class a telecom engineer is most likely to ask by
accident.

### 7. …but a *deterministic* check can catch most of them

The overlap above says a **score** cannot separate those cases. It does not say nothing
can. The two that slip through both assert an entity that does not exist — and that is
checkable without judgement.

The numeric guard already verifies that every number and identifier in an *answer*
occurs in the cited clause. Point the same check at the *question*:

```
"What is the default value of timer T3599?"
    asserted entities  → T3599
    retrieved evidence → T3502, T3510, T3511, T3512, T3517, T3550, T3560 …
    T3599 ∉ evidence   → abstain, before the generator runs
```

Deliberately narrow, because a false positive here means refusing a real question:

| Checked | Not checked | Why |
|---|---|---|
| `T3599`, `NEA3`, `N3IWF` | `AMF`, `PDB`, `GBR` | A claim may abbreviate a term the clause spells out |
| `TS 24.601` | — | A spec number is either indexed or it isn't |
| `91` (2+ digits, with identifier context) | `1`, `3` | Single digits are far too common to discriminate |

Result: abstention recall **42.9% → 85.7%**, over-abstention **0.0% → 0.0%**, cost
zero. It is the only false-premise defence that exists at all when no key is configured.
[`premise_guard.py`](backend/app/rag/premise_guard.py)

### 8. Without a model, extract the fact — don't dump the page

The obvious no-LLM answer is "return the top chunks". It measures badly: a clause runs
to ~1400 characters, so the sentence that answers a lookup is buried, and truncating to
fit loses it. Every table lookup failed this way even though retrieval put the right
table first every time.

[`extractive.py`](backend/app/rag/extractive.py) scores individual sentences and table
rows instead. The row-key rule is the one that matters most — a 3GPP spec answers
"what is the PDB for 5QI 1?" with

```
1 | GBR | 20 | 100 ms | 10^-2 | Conversational Voice
```

which shares **no words at all** with the question. The column names live in the header
and the entity is the bare first cell, so term overlap scores it zero. Matching the
question's numeric tokens against a row's *first cell* is what makes table lookups work
at all; the header is then carried along, because the row alone is unreadable.

The relative-score floor that trims near-miss noise was chosen by sweeping it against
the golden set, not by intuition — 0.40–0.50 is a flat optimum (84% vs 80% overall), so
the default sits at 0.45, mid-band rather than on an edge.

### 9. A silent bug this work surfaced

While debugging table lookups: rows like `1 | GBR | 20 | 100 ms | 10^-2` matched the
clause-heading grammar. Each parsed as clause `"1"` titled `"| GBR | 20 | …"`, became
its own tiny section, and was then dropped by the minimum-length filter. **The entire
5QI QoS characteristics table was absent from the index.**

It hid because retrieval still returned the table's *header* chunk, so clause-level
metrics read 100% while the values themselves were gone. Numeric tables are where a
spec keeps its most citable facts, which makes this the worst possible thing to lose
quietly. One line in `_looks_like_heading`, plus a regression test that asserts the row
survives ingestion and that no clause is ever numbered `"82"`.

### 10. Confidence tracks what survived, not what was drafted

```
confidence = 0.45 · entailment_support + 0.20 · citation_coverage + 0.35 · retrieval_quality
```

then penalised proportionally to how many claims verification removed, and hard-capped
at 0.5 in extractive mode (where relevance was never checked). The evaluation reports
**calibration separation** — mean confidence when correct minus mean confidence when
wrong. A negative number means the system is confidently wrong, which is worse than
being wrong.

---

## Anti-hallucination summary

| Layer | Mechanism | File |
|---|---|---|
| Chunking | Clause-scoped, breadcrumbed, citable | `app/rag/chunking.py` |
| Retrieval | Dense + BM25 → RRF → rerank | `app/rag/retriever.py` |
| **Gate 1** | Score threshold → abstain before any LLM call | `app/rag/pipeline.py` |
| **Gate 1.5** | Deterministic false-premise check → abstain, still no LLM call | `app/rag/premise_guard.py` |
| No-LLM answers | Sentence / table-row extraction instead of chunk dumps | `app/rag/extractive.py` |
| Prompting | Explicit permission to refuse; no parametric knowledge | `app/rag/prompts.py` |
| Structure | Claims + citations + verbatim quotes + modality | `app/rag/generator.py` |
| **Gate 2** | Model declares `answerable: false` | `app/rag/pipeline.py` |
| Check 1 | Citation validity | `app/rag/verifier.py` |
| Check 2 | Quote provenance | `app/rag/verifier.py` |
| Check 3 | Numeric / identifier guard | `app/utils/text.py` |
| Check 4 | LLM entailment, per claim | `app/rag/verifier.py` |
| **Gate 3** | Support ratio below minimum → withhold | `app/rag/pipeline.py` |
| Calibration | Confidence penalised by removals | `app/rag/verifier.py` |
| Transparency | Every score and verdict surfaced in the UI | `frontend/src/components/` |

---

## Evaluation

```bash
cd backend
python eval/run_eval.py                        # full golden set
python eval/run_eval.py --retrieval-only       # no LLM calls, free and instant
python eval/run_eval.py --category false_premise
python eval/run_eval.py --out report.json
```

Also runnable from the **Evaluation** tab in the UI, streaming case-by-case.

The golden set (`eval/golden_set.json`) has 25 cases; **7 are unanswerable on purpose**:

- `out_of_scope` — LTE-M, O-RAN, an un-ingested spec (TS 38.211)
- `false_premise` — a timer that doesn't exist (`T3599`), a 5QI that doesn't exist (`5QI 91`),
  a false historical claim ("why did 3GPP remove network slicing?"), a fictional spec (`TS 24.601`)

Metrics reported:

| Metric | Meaning |
|---|---|
| **Hallucination rate** | Unanswerable questions answered anyway. The headline. |
| **Over-abstention rate** | Answerable questions refused. Stops the headline being gamed. |
| Abstention precision / recall | Quality of the refusal decision |
| Clause hit rate, MRR | Retrieval recall against expected clauses |
| Fully-grounded rate | Answers with zero flagged, removed or uncited claims |
| Calibration separation | Confidence when correct − confidence when wrong |
| Latency p50 / p95 | Per stage and total |

`run_eval.py` exits non-zero on any hallucination, so it works as a CI gate.

### Measured results in this environment

**Extractive mode — no API key, full end-to-end run:**

```
Overall pass rate      84.0%     Clause hit rate    100.0%   (18 answerable cases)
Answerable pass rate   83.3%     Document hit rate  100.0%
Hallucination rate     14.3%     MRR                  0.917
Over-abstention rate    0.0%     Latency        ~3 ms mean, 3.8 ms p95
Abstention precision  100.0%     recall              85.7%

by category:  factual_lookup 9/10 · table_lookup 2/3 · procedure 2/3
              multi_hop 2/2 · out_of_scope 3/3 · false_premise 3/4
```

The one `false_premise` miss is the false-*causal* premise described above; the other
three are caught deterministically.

**Guardrails**, via `tests/test_pipeline_guardrails.py` — a scripted LLM that
hallucinates deliberately, so each defence is verified in isolation:

```
58 passed in 0.38s
```

**Honest caveat on the generative path.** No API key was available in the environment
where this was built, so the *generative* pipeline has not been exercised against a
live API. What *has* been verified:

- every guardrail branch, against a scripted client that fabricates numbers, quotes and
  citations on purpose (`test_pipeline_guardrails.py`)
- Gemini schema translation, round-tripped through the real `google-genai` validator
  (`test_providers.py`)
- the full extractive pipeline end-to-end, including both deterministic gates

Run `python eval/run_eval.py` with a key set to produce the generative numbers. Expect
the `false_premise` and `procedure` categories to improve, since Gate 2 and Check 4
target exactly those.

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness (never touches the index) |
| `GET` | `/api/status` | Index, models, and every active guardrail setting |
| `POST` | `/api/chat` | Blocking answer |
| `POST` | `/api/chat/stream` | **SSE** — stage-by-stage pipeline trace |
| `POST` | `/api/search` | Retrieval only, no generation |
| `GET` | `/api/documents` | Corpus files + indexed specs |
| `POST` | `/api/documents/upload` | Add a spec (`.pdf`, `.docx`, `.txt`, `.md`) |
| `POST` | `/api/documents/reindex` | Rebuild the index |
| `GET` | `/api/documents/chunk/<id>` | Full chunk text (citation drill-down) |
| `GET` | `/api/conversations` | History |
| `POST` | `/api/feedback` | Thumbs up/down, stored with the pipeline trace |
| `GET` | `/api/evaluation/golden-set` | Inspect the eval cases |
| `POST` | `/api/evaluation/run` | **SSE** — run the evaluation |

### `POST /api/chat`

```jsonc
// request
{ "message": "What is the Packet Delay Budget for 5QI 1?", "conversation_id": null }
```

```jsonc
// response (trimmed)
{
  "status": "answered",                  // answered | answered_with_flags | abstained
  "answer": "For 5QI 1 the Packet Delay Budget is 100 ms. [S1][S2]",
  "confidence": { "score": 0.87, "label": "high" },
  "claims": [{
    "text": "For 5QI 1 the Packet Delay Budget is 100 ms.",
    "citations": [1, 2],
    "quote": "5QI Value 1 | GBR | 20 | 100 ms | 10^-2",
    "modality": "descriptive",
    "status": "accepted",
    "verification": {
      "verdict": "supported", "confidence": 0.96,
      "citations_valid": true,
      "quote_found_in_source": true,
      "numeric_guard": { "checked_tokens": 2, "unsupported_tokens": [], "passed": true }
    }
  }],
  "sources": [{
    "source_index": 1, "citation_key": "S1",
    "citation_label": "TS 23.501 §5.7.3.3 — Packet Delay Budget",
    "clause_id": "5.7.3.3",
    "breadcrumb": "5 Overall description > 5.7 QoS model > 5.7.3 QoS characteristics > …",
    "was_cited": true, "is_normative": true,
    "scores": { "final": 0.79, "fusion": 0.032, "dense": 0.61, "sparse": 8.4,
                "dense_rank": 1, "sparse_rank": 2 },
    "text": "The Packet Delay Budget defines an upper bound …"
  }],
  "verification": { "support_ratio": 1.0, "accepted": 3, "flagged": 0, "removed": 0,
                    "verifier_ran": true,
                    "checks": { "citation_validity": true, "quote_provenance": true,
                                "numeric_guard": true, "llm_entailment": true } },
  "retrieval": { "top_score": 0.79, "gate_threshold": 0.42, "passed_gate": true },
  "timings_ms": { "retrieval_ms": 3.2, "generation_ms": 4100, "verification_ms": 1800,
                  "total_ms": 5910 },
  "abstention": null
}
```

When the system abstains, `status` is `"abstained"`, `answer` explains *why*, and
`abstention.nearest_clauses` lists what was considered and how it scored — so a refusal
is still actionable.

### SSE event sequence (`/api/chat/stream`)

```
open → stage(contextualising) → [query_rewritten] → stage(retrieving) → sources
     → stage(generating) → draft → stage(verifying) → result → done
```

Abstention short-circuits to `stage(abstaining)` with the gate name
(`retrieval` | `premise` | `generation` | `verification`), then `result`.

---

## Using real 3GPP specifications

The bundled corpus is a **condensed excerpt** (clearly marked in each file) so the
project runs offline immediately. For the genuine article:

```bash
cd backend
python scripts/download_3gpp.py --list          # see the default set
python scripts/download_3gpp.py                 # TS 23.501, 23.502, 24.501, 33.501, 38.300, 38.331, …
python scripts/download_3gpp.py 38.331 38.321   # or pick specific ones

# Some older specs ship as legacy Word .doc, which python-docx can't read:
soffice --headless --convert-to docx --outdir data/corpus data/corpus/*.doc

python scripts/ingest.py --stats
python scripts/calibrate_threshold.py           # re-calibrate the gate for the new corpus
```

Or drag files into the **Corpus** tab in the UI and hit *Rebuild index*.

> **Recalibrate after changing the corpus or the reranker.** The gate threshold is not a
> universal constant — the lexical fallback and a cross-encoder produce completely
> different score distributions, and shipping a threshold tuned on one with the other is
> how a gate silently stops gating.

---

## Configuration

All in `backend/.env` (see `.env.example`):

| Variable | Default | Effect |
|---|---|---|
| `GEMINI_API_KEY` | — | Enables generative mode (also accepts `GOOGLE_API_KEY`) |
| `ANTHROPIC_API_KEY` | — | Alternative provider. Both empty ⇒ extractive mode |
| `LLM_PROVIDER` | `auto` | `auto` prefers Gemini; force with `gemini` / `anthropic` |
| `ANSWER_MODEL` | per provider | Generation |
| `VERIFIER_MODEL` | per provider | Entailment (bounded per-claim judgements — runs at low effort / zero thinking budget) |
| `MIN_RETRIEVAL_SCORE` | `0.42` | **Gate 1.** Calibrated; raise to be stricter |
| `MIN_SUPPORT_RATIO` | `0.6` | **Gate 3.** Fraction of claims that must survive |
| `ENABLE_VERIFIER` | `true` | Check 4 (one extra LLM call) |
| `ENABLE_NUMERIC_GUARD` | `true` | Check 3 (free) |
| `ENABLE_PREMISE_GUARD` | `true` | Gate 1.5 (free) |
| `RETRIEVAL_TOP_K` | `24` | Candidates per retriever |
| `RERANK_TOP_N` | `6` | Clauses shown to the LLM |
| `CHUNK_TARGET_CHARS` | `1400` | Window size within a clause |

---

## Project layout

```
telecom-rag-chatbot/
├── vercel.json             build, routing, function limits, serverless env
├── package.json            root build entrypoint for Vercel
├── requirements.txt        slim dependency set for the serverless bundle
├── .vercelignore
├── api/index.py            Vercel entrypoint → wraps the Flask app
├── start.sh                local one-command launcher
├── backend/
│   ├── app/
│   │   ├── api/                chat · documents · conversations · evaluation · health
│   │   ├── rag/
│   │   │   ├── loaders.py      PDF/DOCX/TXT → normalised text
│   │   │   ├── chunking.py     clause-aware splitter  ★
│   │   │   ├── embeddings.py   neural + zero-dependency fallback
│   │   │   ├── bm25.py         BM25 Okapi, hand-rolled
│   │   │   ├── vector_store.py persisted hybrid index
│   │   │   ├── retriever.py    RRF fusion + gate  ★
│   │   │   ├── premise_guard.py deterministic false-premise check  ★
│   │   │   ├── extractive.py    no-LLM answer extraction  ★
│   │   │   ├── reranker.py     precision + calibrated score  ★
│   │   │   ├── prompts.py      grounded prompts + JSON schemas  ★
│   │   │   ├── generator.py    claim-structured generation  ★
│   │   │   ├── verifier.py     4-check verification  ★
│   │   │   ├── llm.py          provider abstraction: Gemini + Anthropic
│   │   │   └── pipeline.py     orchestration + 3 gates  ★
│   │   ├── services/           engine singleton · ingestion · SQLite store
│   │   └── utils/text.py       numeric guard, citation parsing  ★
│   ├── data/corpus/            5 condensed 3GPP specs (bundled)
│   ├── eval/                   golden set · metrics · CLI runner
│   ├── scripts/                ingest · download_3gpp · calibrate_threshold
│   └── tests/                  58 tests
└── frontend/src/
    ├── api/client.js           SSE-aware fetch client
    └── components/             ChatView · SourcePanel · PipelineTrace · Corpus · Evaluation
```

★ = anti-hallucination critical path

---

## Frontend

Four things the UI does that a plain chat window doesn't:

1. **Clickable citations.** `[S1]` chips scroll the inspector to the actual clause text,
   with its breadcrumb and every retrieval score.
2. **Per-claim verification.** Each claim shows all four checks with pass/fail marks,
   the supporting quote, and any value the numeric guard couldn't find.
3. **Live pipeline trace.** Stages light up as they run; when a gate fires, the skipped
   stages are struck through. In a domain where a wrong timer causes a field incident,
   an opaque chat bubble *asks* for trust — this *earns* it.
4. **Retrieved-but-uncited sources are shown too**, dimmed. Retrieval stays auditable:
   you can see what was considered, not just what was used.

Plus the corpus manager (upload/reindex) and the evaluation dashboard.

---

## Testing

```bash
cd backend && python -m pytest tests/ -v
```

| File | Covers |
|---|---|
| `test_chunking.py` | Clause detection, annex sub-clauses, breadcrumbs, false-positive heading rejection, overlap |
| `test_guards.py` | Sentence splitting on 3GPP abbreviations, citation parsing, numeric guard (incl. fabricated values) |
| `test_pipeline_guardrails.py` | **End-to-end with a deliberately hallucinating LLM** — one test per defence, one per gate |
| `test_providers.py` | Gemini schema translation (validated against the real SDK), provider selection, no-key degradation |
| `test_no_llm_mode.py` | Premise guard, table-row extraction, list continuation, qualifier conflicts, and the table-heading chunking regression |

That last file is the important one. Sample:

```python
def test_fabricated_number_is_flagged(index):
    """The model copies a real quote but states a number that is not in it."""
    client = ScriptedClient(answer_with(
        "The timer T3512 has a default value of 42 minutes.",   # fabricated
        "The timer T3512 has a default value of 54 minutes",    # real quote
    ))
    result = make_pipeline(index, client).run(QUESTION)

    assert result["claims"][0]["status"] == "flagged"
    assert "unverified_values" in result["claims"][0]["issues"]
```

---

## Known limitations

Stated plainly rather than buried:

1. **Extractive mode cannot catch a false *causal* premise.** It catches false *entity*
   premises deterministically (`T3599`, `5QI 91`, `TS 24.601`), but "why did 3GPP remove
   network slicing?" uses only real entities and asserts a false relationship between
   them. That needs entailment. A regression test pins the limitation so it stays
   visible.
2. **Extractive mode does not synthesise.** It returns the sentences and rows that
   answer the question, not an argument built across clauses. Multi-hop questions get
   the right passages side by side rather than a combined answer.
3. **The default embedder is weak at paraphrase.** Mitigated by hybrid retrieval and
   surfaced as abstentions rather than errors; install `requirements-ml.txt` for real
   semantic recall — still no API key required.
4. **Verification costs a second LLM call** (~40% latency, low effort so modest tokens).
   Disable with `ENABLE_VERIFIER=false` and accept the reduced guarantee.
5. **Thresholds are corpus- and reranker-specific.** Recalibrate `MIN_RETRIEVAL_SCORE`
   after either changes; the extraction floor was swept on the bundled golden set and
   would want re-sweeping on a substantially different corpus.
6. **The premise guard is tuned for 3GPP identifier shapes** (`T####`, `5QI`, `TS NN.NNN`).
   A different standards body would need its patterns extended.
7. **Table parsing assumes a `|` delimiter.** The bundled corpus and Word-table
   extraction both produce it; a PDF whose tables come out space-aligned would not
   benefit from row-key matching.
8. **Single-process index.** Rebuilds take a lock; horizontal scaling would need a shared
   vector store.
9. **The bundled corpus is condensed** and labelled as such in every file. Use
   `scripts/download_3gpp.py` for normative text.

## Possible next steps

- Local NLI verifier (`DeBERTa-v3-base-mnli`) to close the last keyless gap — Check 4 without an API
- Cross-encoder listwise reranking, and query decomposition for multi-hop questions
- Cross-document consistency checking (flag when TS 23.501 and TS 24.501 disagree)
- Human-verified answer cache keyed by normalised question
- Release-aware filtering (`Rel-16` vs `Rel-17` answers to the same question)

---

## Deployment

### Vercel

Ready to deploy as-is — full guide in **[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)**.

```bash
npm i -g vercel
vercel --prod
```

**No environment variables are required** — the deployment boots in extractive mode and
answers with citations immediately. To enable generation, set **one** of
`GEMINI_API_KEY` or `ANTHROPIC_API_KEY` in Project → Settings → Environment Variables.
Everything else is handled by `vercel.json`.

The frontend builds to a static CDN bundle and the Flask app runs as a Python
serverless function (`api/index.py`). Serverless imposes four constraints, each handled
in code rather than papered over:

| Constraint | Handling |
|---|---|
| Read-only filesystem | Directory creation tolerates `OSError`; uploads/reindex return **409** with an explanation instead of a 500 |
| No persistent index | Rebuilt from the bundled corpus into `/tmp` on each cold start — **0.18 s** for 144 chunks, so the deployment stays stateless |
| SSE responses are buffered | `/api/status` reports `capabilities.streaming: false`; the frontend switches to the blocking endpoint rather than showing a dead spinner |
| 60 s function timeout | Documented mitigations (`ENABLE_VERIFIER=false`, Pro plan); the Evaluation tab pre-selects retrieval-only mode |

The UI adapts to these automatically by reading the `capabilities` block — it isn't
told which environment it's in.

> **Ephemeral history:** SQLite lives in `/tmp`, so conversations and feedback do not
> survive a cold start. `services/store.py` is deliberately isolated behind a small
> interface so it can be swapped for Postgres. See the limitations section in the
> deployment guide.

Verify the serverless path locally, without deploying:

```bash
VERCEL=1 ./backend/.venv/bin/python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('e', 'api/index.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
c = m.app.test_client()
print(c.get('/api/status').get_json()['capabilities'])
"
```

### Self-hosted

```bash
gunicorn --workers 2 --threads 4 --timeout 180 "run:app"
```

Threads matter: SSE responses hold a worker for the duration of an answer.
`npm run build` emits a static `frontend/dist/` for any CDN or reverse proxy.

A long-lived container (Fly.io, Render, Cloud Run) is the better fit for a production
3GPP assistant: it restores real streaming, persistent history, corpus uploads and the
`requirements-ml.txt` models — none of which fit a serverless bundle.

---

## License

Provided as an engineering assessment submission. 3GPP specifications are © 3GPP;
the bundled excerpts are condensed for demonstration and are not normative text.
# Telecom-3GPP-RAG-Chatbot
# Telecom-3GPP-RAG-Chatbot_01
