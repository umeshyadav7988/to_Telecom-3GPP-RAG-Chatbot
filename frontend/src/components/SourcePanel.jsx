import React, { useState } from 'react'
import { ScoreBar } from './common.jsx'

function SourceCard({ source, expanded, onToggle, highlighted }) {
  const scores = source.scores || {}
  return (
    <div
      id={`source-${source.source_index}`}
      className={`source-card${highlighted ? ' highlighted' : ''}${source.was_cited ? '' : ' uncited'}`}
    >
      <button type="button" className="source-head" onClick={onToggle}>
        <span className="source-key">S{source.source_index}</span>
        <span className="source-label">{source.citation_label}</span>
        <span className="source-score">{scores.final}</span>
      </button>

      <ScoreBar value={scores.final ?? 0} />

      <div className="source-meta">
        {source.was_cited ? (
          <span className="tag tag-cited">cited</span>
        ) : (
          <span className="tag">retrieved, not cited</span>
        )}
        {source.is_normative && <span className="tag tag-normative">normative</span>}
        {source.has_table && <span className="tag">table</span>}
        {source.version && <span className="tag">v{source.version}</span>}
        {source.page && <span className="tag">p.{source.page}</span>}
      </div>

      {expanded && (
        <div className="source-body">
          {source.breadcrumb && <div className="source-breadcrumb">{source.breadcrumb}</div>}
          <pre>{source.text}</pre>
          <div className="source-scores">
            <span>dense #{scores.dense_rank ?? '—'} ({scores.dense ?? '—'})</span>
            <span>bm25 #{scores.sparse_rank ?? '—'} ({scores.sparse ?? '—'})</span>
            <span>fusion {scores.fusion}</span>
            {scores.explain?.term_coverage !== undefined && (
              <span>coverage {scores.explain.term_coverage}</span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function ClaimCard({ claim, onCite }) {
  const v = claim.verification || {}
  const statusLabel =
    claim.status === 'removed'
      ? 'removed by verification'
      : claim.status === 'flagged'
        ? 'needs review'
        : 'verified'

  return (
    <div className={`claim-card claim-${claim.status}`}>
      <div className="claim-head">
        <span className={`claim-status claim-status-${claim.status}`}>{statusLabel}</span>
        {claim.modality && claim.modality !== 'descriptive' && (
          <span className="tag tag-modality">{claim.modality}</span>
        )}
        <span className="claim-cites">
          {claim.citations.map((c) => (
            <button key={c} type="button" className="cite-chip small" onClick={() => onCite?.(c)}>
              S{c}
            </button>
          ))}
        </span>
      </div>

      <p className="claim-text">{claim.text}</p>

      <div className="claim-checks">
        <Check ok={v.citations_valid} label="citations valid" />
        <Check
          ok={v.quote_checked ? v.quote_found_in_source : null}
          label="quote found in source"
        />
        <Check
          ok={v.numeric_guard?.checked_tokens ? v.numeric_guard.passed : null}
          label={
            v.numeric_guard?.checked_tokens
              ? `values verified (${v.numeric_guard.checked_tokens})`
              : 'no values to verify'
          }
        />
        <Check
          ok={
            v.verdict === 'supported'
              ? true
              : v.verdict === 'partially_supported'
                ? null
                : v.verdict === 'unverified'
                  ? null
                  : false
          }
          label={`entailment: ${v.verdict}`}
        />
      </div>

      {v.numeric_guard?.unsupported_tokens?.length > 0 && (
        <div className="claim-warning">
          Not found in the cited clause:{' '}
          {v.numeric_guard.unsupported_tokens.map((t) => t.token).join(', ')}
        </div>
      )}
      {v.reason && <div className="claim-reason">{v.reason}</div>}
      {claim.quote && (
        <details className="claim-quote">
          <summary>Supporting quote</summary>
          <blockquote>{claim.quote}</blockquote>
        </details>
      )}
    </div>
  )
}

function Check({ ok, label }) {
  const cls = ok === true ? 'pass' : ok === false ? 'fail' : 'neutral'
  const mark = ok === true ? '✓' : ok === false ? '✕' : '–'
  return (
    <span className={`check check-${cls}`}>
      <span className="check-mark">{mark}</span> {label}
    </span>
  )
}

export default function SourcePanel({ message, activeSource, onCite }) {
  const [tab, setTab] = useState('sources')
  const [expanded, setExpanded] = useState(() => new Set())

  if (!message) {
    return (
      <aside className="inspector">
        <div className="inspector-empty">
          <h3>Inspector</h3>
          <p>
            Ask a question, then select an answer to inspect the clauses it was built from and the
            verification result for every individual claim.
          </p>
        </div>
      </aside>
    )
  }

  const sources = message.sources || []
  const claims = message.claims || []

  const toggle = (index) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      next.has(index) ? next.delete(index) : next.add(index)
      return next
    })
  }

  return (
    <aside className="inspector">
      <div className="inspector-tabs">
        <button className={tab === 'sources' ? 'active' : ''} onClick={() => setTab('sources')}>
          Sources ({sources.length})
        </button>
        <button className={tab === 'claims' ? 'active' : ''} onClick={() => setTab('claims')}>
          Claims ({claims.length})
        </button>
        <button className={tab === 'trace' ? 'active' : ''} onClick={() => setTab('trace')}>
          Trace
        </button>
      </div>

      <div className="inspector-body">
        {tab === 'sources' && (
          <>
            {sources.length === 0 && <p className="muted">No clauses were retrieved.</p>}
            {sources.map((s) => (
              <SourceCard
                key={s.source_index}
                source={s}
                expanded={expanded.has(s.source_index)}
                highlighted={activeSource === s.source_index}
                onToggle={() => toggle(s.source_index)}
              />
            ))}
          </>
        )}

        {tab === 'claims' && (
          <>
            {claims.length === 0 && (
              <p className="muted">
                No claims were produced — the pipeline abstained before generation.
              </p>
            )}
            {claims.map((c) => (
              <ClaimCard key={c.index} claim={c} onCite={onCite} />
            ))}
          </>
        )}

        {tab === 'trace' && (
          <div className="trace-detail">
            <Section title="Retrieval" data={message.retrieval} />
            <Section title="Verification" data={message.verification} />
            <Section title="Timings (ms)" data={message.timings_ms} />
            {message.usage && Object.keys(message.usage).length > 0 && (
              <Section title="Token usage" data={message.usage} />
            )}
            {message.abstention && <Section title="Abstention" data={message.abstention} />}
          </div>
        )}
      </div>
    </aside>
  )
}

function Section({ title, data }) {
  if (!data) return null
  return (
    <div className="trace-section">
      <h4>{title}</h4>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  )
}
