import React, { useEffect, useRef, useState } from 'react'
import { CitedText, ConfidenceBadge, Spinner } from './common.jsx'
import PipelineTrace from './PipelineTrace.jsx'
import { api } from '../api/client.js'

const SUGGESTIONS = [
  'What are the standardised SST values for network slicing?',
  'What is the Packet Delay Budget for 5QI 1?',
  'What is the default value of timer T3512?',
  'What is the difference between RRC_INACTIVE and RRC_IDLE?',
  // Deliberately unanswerable — the honest refusal is the demo.
  'What is the default value of timer T3599?',
]

function Feedback({ message }) {
  const [sent, setSent] = useState(null)
  if (message.role !== 'assistant' || !message.turn_id) return null

  const send = async (rating) => {
    setSent(rating)
    try {
      await api.feedback({
        turn_id: message.turn_id,
        rating,
        confidence: message.confidence?.score,
        status: message.status,
      })
    } catch {
      setSent(null)
    }
  }

  if (sent) return <span className="feedback-sent">Thanks — recorded.</span>
  return (
    <span className="feedback">
      <button onClick={() => send('up')} title="This answer was correct">
        Helpful
      </button>
      <button onClick={() => send('down')} title="This answer was wrong or unhelpful">
        Not helpful
      </button>
    </span>
  )
}

function AssistantMessage({ message, isSelected, onSelect, onCite, activeSource }) {
  const abstained = message.status === 'abstained'
  return (
    <div
      className={`message assistant${isSelected ? ' selected' : ''}${abstained ? ' abstained' : ''}`}
      onClick={onSelect}
    >
      <div className="message-header">
        <ConfidenceBadge confidence={message.confidence} status={message.status} />
        {message.mode === 'extractive' && (
          <span className="badge badge-mode" title="No API key configured: verbatim excerpts only">
            extractive mode
          </span>
        )}
        {message.verification?.flagged > 0 && (
          <span className="badge badge-flagged">{message.verification.flagged} claim(s) flagged</span>
        )}
        {message.verification?.removed > 0 && (
          <span className="badge badge-removed">
            {message.verification.removed} claim(s) removed
          </span>
        )}
      </div>

      {abstained ? (
        <div className="abstention">
          {message.answer.split('\n\n').map((para, i) => (
            <p key={i}>{para}</p>
          ))}
        </div>
      ) : (
        <CitedText text={message.answer} onCite={onCite} activeSource={activeSource} />
      )}

      {message.caveats?.length > 0 && (
        <ul className="caveats">
          {message.caveats.map((c, i) => (
            <li key={i}>{c}</li>
          ))}
        </ul>
      )}

      {message.trace && (
        <PipelineTrace
          stage={message.trace.stage}
          gate={message.trace.gate}
          retrieval={message.retrieval}
          verification={message.verification}
          timings={message.timings_ms}
          done
        />
      )}

      <div className="message-footer">
        <span className="muted">
          {message.sources?.length || 0} clause(s) retrieved
          {message.sources ? ` · ${message.sources.filter((s) => s.was_cited).length} cited` : ''}
        </span>
        <Feedback message={message} />
      </div>
    </div>
  )
}

export default function ChatView({
  messages,
  streaming,
  liveStage,
  liveGate,
  liveRetrieval,
  selectedId,
  onSelect,
  onSend,
  onCite,
  activeSource,
  onStop,
  disabled,
}) {
  const [input, setInput] = useState('')
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, liveStage])

  const submit = (event) => {
    event.preventDefault()
    const text = input.trim()
    if (!text || streaming) return
    setInput('')
    onSend(text)
  }

  return (
    <section className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="welcome">
            <h2>3GPP Specification Assistant</h2>
            <p>
              Answers are built only from the indexed specifications. Every statement carries a
              clause-level citation you can open and check, and when the corpus does not contain the
              answer the assistant says so instead of guessing.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => onSend(s)} disabled={disabled}>
                  {s}
                </button>
              ))}
            </div>
            <p className="hint">
              The last suggestion asks about a timer that does not exist — a good way to see the
              abstention behaviour.
            </p>
          </div>
        )}

        {messages.map((m) =>
          m.role === 'user' ? (
            <div key={m.id} className="message user">
              <p>{m.content}</p>
            </div>
          ) : (
            <AssistantMessage
              key={m.id}
              message={m}
              isSelected={selectedId === m.id}
              onSelect={() => onSelect(m.id)}
              onCite={onCite}
              activeSource={selectedId === m.id ? activeSource : null}
            />
          ),
        )}

        {streaming && (
          <div className="message assistant streaming">
            <Spinner label={liveStage ? liveStage.replace(/ing$/, 'ing') : 'working'} />
            <PipelineTrace
              stage={liveStage}
              gate={liveGate}
              retrieval={liveRetrieval}
              done={false}
            />
          </div>
        )}

        <div ref={endRef} />
      </div>

      <form className="composer" onSubmit={submit}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) submit(e)
          }}
          placeholder={
            disabled
              ? 'Build the index first (Corpus tab) before asking questions'
              : 'Ask about 5G architecture, NAS procedures, security or NR…  (Enter to send)'
          }
          rows={2}
          disabled={disabled}
        />
        {streaming ? (
          <button type="button" className="stop" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!input.trim() || disabled}>
            Send
          </button>
        )}
      </form>
    </section>
  )
}
