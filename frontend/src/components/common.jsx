import React from 'react'

export function ConfidenceBadge({ confidence, status }) {
  if (status === 'abstained') {
    return <span className="badge badge-abstained">abstained</span>
  }
  if (!confidence) return null
  const { score, label } = confidence
  return (
    <span className={`badge badge-${label}`} title={`Confidence score ${score}`}>
      {label} confidence · {Math.round(score * 100)}%
    </span>
  )
}

/**
 * Renders answer text, turning `[S1]` markers into clickable chips.
 * The marker is the whole point of the answer format, so it gets to be
 * interactive rather than decorative punctuation.
 */
export function CitedText({ text, onCite, activeSource }) {
  if (!text) return null
  const parts = text.split(/(\[S\d+\])/g)
  return (
    <p className="cited-text">
      {parts.map((part, i) => {
        const match = /^\[S(\d+)\]$/.exec(part)
        if (!match) return <React.Fragment key={i}>{part}</React.Fragment>
        const index = Number(match[1])
        return (
          <button
            key={i}
            type="button"
            className={`cite-chip${activeSource === index ? ' active' : ''}`}
            onClick={() => onCite?.(index)}
            title={`Jump to source S${index}`}
          >
            S{index}
          </button>
        )
      })}
    </p>
  )
}

export function ScoreBar({ value, max = 1 }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  return (
    <div className="score-bar" aria-hidden="true">
      <div className="score-bar-fill" style={{ width: `${pct}%` }} />
    </div>
  )
}

export function Spinner({ label }) {
  return (
    <span className="spinner-wrap">
      <span className="spinner" />
      {label ? <span className="spinner-label">{label}</span> : null}
    </span>
  )
}

export function EmptyState({ title, children }) {
  return (
    <div className="empty-state">
      <h3>{title}</h3>
      <div>{children}</div>
    </div>
  )
}
