import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api/client.js'
import { EmptyState, Spinner } from './common.jsx'

/**
 * Runs the golden set from the browser and shows the metrics that matter.
 *
 * Hallucination rate leads because it is the claim the project is making. The
 * pair "over-abstention rate" sits directly beside it on purpose: driving
 * hallucinations to zero is trivial if you are allowed to refuse everything,
 * so the two numbers are only meaningful together.
 */
export default function EvaluationView({ status }) {
  const [cases, setCases] = useState([])
  const [results, setResults] = useState([])
  const [summary, setSummary] = useState(null)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState({ done: 0, total: 0 })
  const [error, setError] = useState(null)
  const abortRef = useRef(null)

  // A full generative run is 25 cases x 2-3 LLM calls, which comfortably
  // exceeds a serverless function timeout. Default to the retrieval-only run
  // there so the first click succeeds instead of dying at the platform limit.
  const serverless = status?.capabilities?.streaming === false
  const [retrievalOnly, setRetrievalOnly] = useState(false)
  useEffect(() => {
    if (serverless) setRetrievalOnly(true)
  }, [serverless])

  useEffect(() => {
    api
      .goldenSet()
      .then((data) => setCases(data.cases || []))
      .catch((exc) => setError(exc.message))
  }, [])

  const run = async () => {
    setRunning(true)
    setResults([])
    setSummary(null)
    setError(null)
    abortRef.current = new AbortController()

    try {
      await api.runEvaluation(
        { retrieval_only: retrievalOnly },
        (event, data) => {
          if (event === 'start') setProgress({ done: 0, total: data.total })
          else if (event === 'case') {
            setResults((prev) => [...prev, data.result])
            setProgress({ done: data.index, total: data.total })
          } else if (event === 'summary') setSummary(data)
        },
        abortRef.current.signal,
      )
    } catch (exc) {
      if (exc.name !== 'AbortError') setError(exc.message)
    } finally {
      setRunning(false)
    }
  }

  const stop = () => abortRef.current?.abort()

  return (
    <section className="panel">
      <header className="panel-header">
        <div>
          <h2>Evaluation</h2>
          <p className="muted">
            {cases.length} golden-set cases. {cases.filter((c) => !c.answerable).length} of them are
            unanswerable on purpose — those measure whether the system refuses instead of inventing.
          </p>
        </div>
        <div className="panel-actions">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={retrievalOnly}
              onChange={(e) => setRetrievalOnly(e.target.checked)}
              disabled={running}
            />
            retrieval only (no LLM cost)
          </label>
          {running ? (
            <button className="stop" onClick={stop}>
              Stop
            </button>
          ) : (
            <button className="primary" onClick={run}>
              Run evaluation
            </button>
          )}
        </div>
      </header>

      {error && <div className="notice notice-error">{error}</div>}

      {serverless && !retrievalOnly && (
        <div className="notice">
          A full generative run makes roughly 50–75 model calls and will exceed this
          deployment&rsquo;s function timeout. Run it locally, or keep &ldquo;retrieval only&rdquo;
          ticked here.
        </div>
      )}

      {running && (
        <div className="notice">
          <Spinner label={`Case ${progress.done} of ${progress.total}`} />
        </div>
      )}

      {summary && (
        <>
          <div className="stat-row headline-row">
            <Metric
              label="Hallucination rate"
              value={
                summary.headline.hallucination_rate === null
                  ? 'n/a'
                  : pct(summary.headline.hallucination_rate)
              }
              tone={
                summary.headline.hallucination_rate === null
                  ? 'neutral'
                  : summary.headline.hallucination_rate === 0
                    ? 'good'
                    : 'bad'
              }
              hint={
                summary.headline.hallucination_rate === null
                  ? 'Retrieval-only run: nothing was generated'
                  : `${summary.headline.hallucination_count}/${summary.headline.unanswerable_cases} unanswerable questions answered anyway`
              }
            />
            <Metric
              label="Over-abstention"
              value={pct(summary.answering.over_abstention_rate)}
              tone={summary.answering.over_abstention_rate > 0.15 ? 'warn' : 'good'}
              hint={`Refused ${summary.answering.over_abstention_count} answerable question(s)`}
            />
            <Metric
              label="Overall pass rate"
              value={pct(summary.headline.overall_pass_rate)}
              tone={summary.headline.overall_pass_rate > 0.85 ? 'good' : 'warn'}
            />
          </div>

          <div className="metric-grid">
            <MetricGroup
              title="Abstention"
              rows={[
                ['Precision', pct(summary.abstention.precision)],
                ['Recall', pct(summary.abstention.recall)],
                ['Total abstentions', summary.abstention.total_abstentions],
              ]}
            />
            <MetricGroup
              title="Retrieval"
              rows={[
                ['Clause hit rate', pct(summary.retrieval.clause_hit_rate)],
                ['Document hit rate', pct(summary.retrieval.document_hit_rate)],
                ['MRR', summary.retrieval.mrr],
              ]}
            />
            <MetricGroup
              title="Groundedness"
              rows={[
                ['Fully grounded', pct(summary.groundedness.fully_grounded_rate)],
                ['With flagged claims', summary.groundedness.answers_with_flagged_claims],
                ['With removed claims', summary.groundedness.answers_with_removed_claims],
                ['Uncited claims', summary.groundedness.uncited_claims_total],
              ]}
            />
            <MetricGroup
              title="Calibration"
              rows={[
                ['Confidence when correct', summary.calibration.mean_confidence_when_correct ?? '—'],
                ['Confidence when wrong', summary.calibration.mean_confidence_when_wrong ?? '—'],
                ['Separation', summary.calibration.separation ?? '—'],
              ]}
            />
            <MetricGroup
              title="Latency (ms)"
              rows={[
                ['Mean', summary.latency_ms.mean],
                ['p50', summary.latency_ms.p50],
                ['p95', summary.latency_ms.p95],
              ]}
            />
            <MetricGroup
              title="By category"
              rows={Object.entries(summary.by_category).map(([k, v]) => [
                k.replace(/_/g, ' '),
                `${v.passed}/${v.total}`,
              ])}
            />
          </div>
        </>
      )}

      {results.length > 0 ? (
        <>
          <h3>Cases</h3>
          <table className="table">
            <thead>
              <tr>
                <th>Result</th>
                <th>ID</th>
                <th>Category</th>
                <th>Question</th>
                <th>Outcome</th>
                <th className="num">Conf.</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.id} className={r.passed ? '' : 'row-fail'}>
                  <td>
                    <span className={`badge ${r.passed ? 'badge-high' : 'badge-removed'}`}>
                      {r.passed ? 'pass' : 'fail'}
                    </span>
                  </td>
                  <td className="mono">{r.id}</td>
                  <td className="muted">{r.category?.replace(/_/g, ' ')}</td>
                  <td>{r.question}</td>
                  <td className="muted">
                    {r.error
                      ? `error: ${r.error}`
                      : r.abstained
                        ? 'abstained'
                        : (r.failure_mode || 'answered').replace(/_/g, ' ')}
                  </td>
                  <td className="num mono">
                    {r.confidence !== undefined ? r.confidence.toFixed?.(2) ?? r.confidence : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        !running && (
          <EmptyState title="No run yet">
            Run the evaluation to measure hallucination rate, abstention quality, retrieval recall
            and confidence calibration against the golden set.
          </EmptyState>
        )
      )}
    </section>
  )
}

const pct = (value) => (value === null || value === undefined ? '—' : `${Math.round(value * 100)}%`)

function Metric({ label, value, tone = 'neutral', hint }) {
  return (
    <div className={`metric metric-${tone}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
      {hint && <span className="metric-hint">{hint}</span>}
    </div>
  )
}

function MetricGroup({ title, rows }) {
  return (
    <div className="metric-group">
      <h4>{title}</h4>
      <table>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <td>{label}</td>
              <td className="mono num">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
