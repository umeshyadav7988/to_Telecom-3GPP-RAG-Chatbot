import React from 'react'

/**
 * Live view of the retrieve -> generate -> verify pipeline.
 *
 * This is deliberately prominent rather than hidden behind a debug toggle. In
 * a domain where a wrong timer value causes a field incident, a user needs to
 * see *why* an answer should be trusted — which clauses were retrieved, how
 * well they scored, and what verification did to the draft. An opaque chat
 * bubble asks for trust; this earns it.
 */

const STAGES = [
  { key: 'contextualising', label: 'Resolve question', hint: 'Rewrite follow-ups into a standalone query' },
  { key: 'retrieving', label: 'Hybrid retrieval', hint: 'Dense + BM25, fused and reranked' },
  { key: 'generating', label: 'Grounded generation', hint: 'Claims with citations and verbatim quotes' },
  { key: 'verifying', label: 'Verification', hint: 'Citations, quotes, numerics, entailment' },
]

const STAGE_ORDER = STAGES.map((s) => s.key)

export default function PipelineTrace({ stage, gate, retrieval, verification, timings, done }) {
  const currentIndex = STAGE_ORDER.indexOf(stage)

  return (
    <div className="pipeline-trace">
      <div className="pipeline-stages">
        {STAGES.map((s, i) => {
          let state = 'pending'
          if (done) state = 'complete'
          else if (i < currentIndex) state = 'complete'
          else if (i === currentIndex) state = 'active'
          if (gate && i > currentIndex) state = 'skipped'

          return (
            <div key={s.key} className={`pipeline-stage ${state}`} title={s.hint}>
              <span className="pipeline-dot" />
              <span className="pipeline-label">{s.label}</span>
            </div>
          )
        })}
      </div>

      {gate && (
        <div className="pipeline-gate">
          Abstention gate fired at <strong>{gate}</strong> — the remaining stages were skipped.
        </div>
      )}

      {retrieval && (
        <div className="pipeline-facts">
          <span>
            top score <strong>{retrieval.top_score}</strong> / threshold {retrieval.gate_threshold}
          </span>
          <span>{retrieval.candidate_count} candidates</span>
          {retrieval.was_rewritten && <span className="tag">query rewritten</span>}
          {retrieval.filters_applied?.doc_numbers && (
            <span className="tag">filtered to {retrieval.filters_applied.doc_numbers.join(', ')}</span>
          )}
        </div>
      )}

      {verification && (
        <div className="pipeline-facts">
          <span className="ok">{verification.accepted} accepted</span>
          {verification.flagged > 0 && <span className="warn">{verification.flagged} flagged</span>}
          {verification.removed > 0 && <span className="bad">{verification.removed} removed</span>}
          <span>support {Math.round((verification.support_ratio ?? 0) * 100)}%</span>
          {verification.verifier_ran === false && <span className="tag">entailment skipped</span>}
        </div>
      )}

      {done && timings && (
        <div className="pipeline-timings">
          {Object.entries(timings)
            .filter(([k]) => k.endsWith('_ms'))
            .map(([k, v]) => (
              <span key={k}>
                {k.replace(/_ms$/, '').replace(/_/g, ' ')} {v}ms
              </span>
            ))}
        </div>
      )}
    </div>
  )
}
