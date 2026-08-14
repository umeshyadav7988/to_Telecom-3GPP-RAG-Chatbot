# Pipeline walkthrough — three real queries

Traces captured from the running system against the bundled corpus
(144 chunks, 5 specifications, lexical-semantic reranker, `MIN_RETRIEVAL_SCORE=0.42`).

They are chosen to show the three distinct outcomes the architecture produces, and
in particular why one abstention gate would not be enough.

---

## Query 1 — answerable → answered

> **"What is the Packet Delay Budget for 5QI 1?"**

### Stage 1 · Contextualise

No conversation history and no dangling reference, so the heuristic short-circuits and
the query is used as-is. No LLM call.

### Stage 2 · Hybrid retrieval

```
gate=PASS  top=0.664  threshold=0.42  candidates=34

S1  0.664   dense#1  bm25#2   TS 23.501 §5.7.3.3 — Packet Delay Budget
       coverage=0.75  dense=0.642  title_match=0.50
S2  0.616   dense#2  bm25#1   TS 23.501 §5.7.4 — Standardized 5QI to QoS characteristics mapping
       coverage=0.75  dense=0.618  title_match=0.25
S3  0.554   dense#4  bm25#3   TS 23.501 §5.7.3 — QoS characteristics
       coverage=0.75  dense=0.565  title_match=0.00
```

Note what hybrid retrieval bought here. The **definition** of PDB (§5.7.3.3) ranks first
on the dense retriever; the **table containing the actual value** (§5.7.4) ranks first on
BM25, because `5QI` and `1` are exact lexical hits. A dense-only system would have put
the table second or lower and might have dropped it from the top-*n* window entirely —
answering "what a PDB is" instead of "what the value is".

`title_match=0.50` on S1 is the clause-title signal: the breadcrumb is prefixed into the
indexed text, so "Packet Delay Budget" matches the heading, not just the body.

### Stage 3 · Gate 1

`0.664 ≥ 0.42` → pass. Generation proceeds.

### Stage 4 · Grounded generation

The model returns structured JSON — one claim per assertion, each with citations and a
verbatim quote it must have copied from a cited source:

```jsonc
{
  "answerable": true,
  "claims": [{
    "text": "For 5QI 1 the Packet Delay Budget is 100 ms.",
    "citations": [2],
    "quote": "1 | GBR | 20 | 100 ms | 10^-2 | Conversational Voice",
    "modality": "descriptive"
  }, {
    "text": "5QI 1 is a GBR resource type with a default priority level of 20 and a Packet Error Rate of 10^-2.",
    "citations": [2],
    "quote": "1 | GBR | 20 | 100 ms | 10^-2 | Conversational Voice",
    "modality": "descriptive"
  }, {
    "text": "The Packet Delay Budget defines an upper bound for the time a packet may be delayed between the UE and the N6 termination point at the UPF, and is the same in uplink and downlink.",
    "citations": [1],
    "quote": "defines an upper bound for the time that a packet may be delayed between the UE and the N6 termination point at the UPF",
    "modality": "descriptive"
  }]
}
```

### Stage 5 · Verification

| Check | Claim 1 | Claim 2 | Claim 3 |
|---|---|---|---|
| Citation validity | ✓ `[2]` retrieved | ✓ | ✓ |
| Quote provenance | ✓ found in S2 | ✓ | ✓ |
| Numeric guard | ✓ `100ms`, `5QI` present | ✓ `20`, `10^-2` present | ✓ no values |
| Entailment | supported | supported | supported |

### Result

```
status      answered
confidence  0.87 (high)
answer      For 5QI 1 the Packet Delay Budget is 100 ms. [S2] 5QI 1 is a GBR resource
            type with a default priority level of 20 and a Packet Error Rate of
            10^-2. [S2] The Packet Delay Budget defines an upper bound for the time a
            packet may be delayed between the UE and the N6 termination point at the
            UPF, and is the same in uplink and downlink. [S1]
```

S3 was retrieved but never cited — the UI shows it dimmed under "retrieved, not cited",
so retrieval stays auditable.

---

## Query 2 — off-corpus → **Gate 1** abstention, zero LLM cost

> **"Which service models does the O-RAN Alliance define for the E2 interface?"**

### Stage 2 · Hybrid retrieval

```
gate=BLOCK  top=0.368  threshold=0.42  candidates=38

S1  0.368   dense#8   bm25#5   TS 23.501 §5.6.9 — Session and Service Continuity
       coverage=0.273  dense=0.545  title_match=0.091
S2  0.329   dense#13  bm25#4   TS 33.501 §13.5 — Protection across PLMN interconnect
       coverage=0.182  dense=0.539  title_match=0.091
S3  0.318   dense#7   bm25#8   TS 23.501 §3.1 — Definitions
       coverage=0.182  dense=0.545  title_match=0.000
```

This is what a genuine miss looks like: retrieval matched on the generic words
("service", "interface", "define") and nothing on the specific ones. Term coverage
collapses to **0.27**, dragging the composite score below the gate.

Notice that `dense≈0.545` for all three — the dense retriever alone is nearly
uninformative here, which is precisely why coverage carries 40% of the weight.

### Stage 3 · Gate 1 → abstain

```
status  abstained  (type: no_relevant_context)

I could not find this in the indexed 3GPP specifications, so I am not going to
answer from memory.

Closest clauses considered: TS 23.501 §5.6.9 — Session and Service Continuity
(score 0.368); TS 33.501 §13.5 — Protection across PLMN interconnect (score 0.329);
TS 23.501 §3.1 — Definitions (score 0.318). Try naming the specification or clause
you have in mind, or rephrase using the standard's terminology.
```

**Zero generation tokens were spent.** The refusal is also actionable — it says what it
looked at and how close it got, rather than a bare "I don't know".

---

## Query 3 — false premise → Gate 1 **passes**, later gates catch it

> **"What is the default value of timer T3599 in 5GMM?"**
>
> T3599 does not exist. T3502, T3510, T3511, T3512, T3517… do.

This is the case that justifies the whole multi-gate design.

### Stage 2 · Hybrid retrieval

```
gate=PASS  top=0.542  threshold=0.42  candidates=30

S1  0.542   dense#1   bm25#2   TS 24.501 §10.2.1 — Timers of 5GS mobility management
                                                    in the UE (part 2/2)
       coverage=0.636  dense=0.578  title_match=0.182
S2  0.457   dense#10  bm25#1   TS 24.501 §5.5.1.2.5 — Initial registration not accepted
S3  0.448   dense#9   bm25#5   TS 24.501 §10.2.2 — Timers ... in the network
```

**Retrieval works perfectly and that is the problem.** The question is about a 5GMM
timer's default value; the corpus contains exactly that table; the retriever finds it
and scores it 0.542 — comfortably above the gate, and *higher than 4 of the 18
genuinely answerable questions in the golden set*.

The topic is present. The fact is not. **No retrieval threshold can distinguish those
two situations**, because they produce identical retrieval behaviour.

Empirically, from `scripts/calibrate_threshold.py`:

```
ANSWERABLE    min 0.456  ─┐
UNANSWERABLE  max 0.542  ─┴─ the distributions overlap
```

Any threshold high enough to block T3599 (>0.542) would also block a quarter of the
answerable questions. This is not a tuning failure; it is a structural limit of
threshold-based abstention.

### Stage 4 · Gate 2 — the generator refuses

The model is handed the real timer table and asked about a timer that isn't in it. The
prompt gives it explicit permission to refuse, and the schema makes refusal a
first-class field rather than something it must express in prose:

```jsonc
{
  "answerable": false,
  "refusal_reason": "The retrieved clauses list T3502, T3510, T3511, T3512, T3516, T3517, T3519, T3520, T3521, T3525 and T3540 for the UE, and T3513, T3550, T3555, T3560 and T3570 for the network. None of them is T3599, and no default value for T3599 appears anywhere in the provided text.",
  "claims": []
}
```

```
status  abstained  (type: insufficient_context)

The clauses I retrieved are related to your question but do not actually contain the
answer, so I am not going to infer one.

The retrieved clauses list T3502, T3510, T3511, T3512, … None of them is T3599 …
```

Verification never runs — there is nothing to verify — so Gate 2 costs one call, not two.

### If the model had answered anyway

Suppose it pattern-matched the table and produced *"T3599 has a default value of 54
minutes"*. Two later defences fire independently:

1. **Numeric guard** (deterministic, free): tokenises the claim to `T3599`, `54minutes`.
   `54minutes` occurs in the evidence (it's T3512's value) but **`T3599` does not** →
   `unsupported_tokens: [{token: "T3599", kind: "identifiers"}]` → claim flagged, and
   the UI names the exact token it could not find.
2. **Entailment check**: the cited text describes T3512, not T3599 → `unsupported` →
   claim removed → support ratio `0.0 < 0.6` → **Gate 3** withholds the whole answer.

This exact scenario is a regression test:
`tests/test_pipeline_guardrails.py::test_fabricated_number_is_flagged`.

---

## What the three queries show

| | Query 1 | Query 2 | Query 3 |
|---|---|---|---|
| Retrieval score | 0.664 | 0.368 | **0.542** |
| Gate 1 | pass | **blocks** | pass |
| Gate 2 | pass | — | **blocks** |
| Gate 3 | pass | — | — |
| LLM calls | 3 (rewrite skipped) | **0** | 1 |
| Outcome | answered, cited | abstained | abstained |

Query 2 is cheap to catch and Query 3 is not. A single-gate system has to choose which
one it handles; this one handles both, and each gate is cheaper than the one after it —
so the common case (a genuine miss) never pays for the expensive defence.

---

## Reproducing these traces

```bash
cd backend
source .venv/bin/activate

# Retrieval traces (no API key needed)
python - <<'PY'
import sys; sys.path.insert(0, '.')
from app.services.engine import get_engine
e = get_engine()
for q in ["What is the Packet Delay Budget for 5QI 1?",
          "Which service models does the O-RAN Alliance define for the E2 interface?",
          "What is the default value of timer T3599 in 5GMM?"]:
    r = e.retriever.retrieve(q)
    print(f"\nQ: {q}")
    print(f"   gate={'PASS' if r.passed_gate else 'BLOCK'} top={r.top_score:.3f} "
          f"thr={r.gate_threshold} candidates={r.candidate_count}")
    for c in r.chunks[:3]:
        print(f"   S{c.source_index} {c.score:.3f}  dense#{c.dense_rank} "
              f"bm25#{c.sparse_rank}  {c.chunk.citation_label}")
        print(f"        {c.explain}")
PY

# Threshold separation analysis
python scripts/calibrate_threshold.py

# Full generative traces (needs ANTHROPIC_API_KEY)
curl -sN -X POST localhost:5001/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the default value of timer T3599 in 5GMM?"}'
```

The generation and verification payloads above are illustrative of the schema the
pipeline enforces; the retrieval numbers are captured verbatim from a real run.
