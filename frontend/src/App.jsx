import React, { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api/client.js'
import ChatView from './components/ChatView.jsx'
import CorpusView from './components/CorpusView.jsx'
import EvaluationView from './components/EvaluationView.jsx'
import SourcePanel from './components/SourcePanel.jsx'

let messageCounter = 0
const nextId = () => `m${++messageCounter}`

export default function App() {
  const [tab, setTab] = useState('chat')
  const [status, setStatus] = useState(null)
  const [statusError, setStatusError] = useState(null)

  const [messages, setMessages] = useState([])
  const [conversationId, setConversationId] = useState(null)
  const [conversations, setConversations] = useState([])

  const [streaming, setStreaming] = useState(false)
  const [liveStage, setLiveStage] = useState(null)
  const [liveGate, setLiveGate] = useState(null)
  const [liveRetrieval, setLiveRetrieval] = useState(null)

  const [selectedId, setSelectedId] = useState(null)
  const [activeSource, setActiveSource] = useState(null)

  const abortRef = useRef(null)

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await api.status())
      setStatusError(null)
    } catch (exc) {
      setStatusError(exc.message)
    }
  }, [])

  const refreshConversations = useCallback(async () => {
    try {
      const data = await api.conversations()
      setConversations(data.conversations || [])
    } catch {
      /* conversation history is non-critical */
    }
  }, [])

  useEffect(() => {
    refreshStatus()
    refreshConversations()
  }, [refreshStatus, refreshConversations])

  const send = useCallback(
    async (text) => {
      const userMessage = { id: nextId(), role: 'user', content: text }
      const assistantId = nextId()

      setMessages((prev) => [...prev, userMessage])
      setStreaming(true)
      setLiveStage('contextualising')
      setLiveGate(null)
      setLiveRetrieval(null)
      abortRef.current = new AbortController()

      let partial = { sources: [], retrieval: null }
      let gate = null

      // Serverless hosts (Vercel) buffer WSGI responses, so an SSE stream
      // arrives in one blob after the answer is finished. Streaming there
      // costs the same time but shows no progress, so use the blocking
      // endpoint and label the wait honestly instead.
      if (status && status.capabilities?.streaming === false) {
        try {
          setLiveStage('retrieving')
          const data = await api.chat({ message: text, conversation_id: conversationId })
          setMessages((prev) => [
            ...prev,
            { id: assistantId, role: 'assistant', ...data, trace: { stage: 'verifying' } },
          ])
          setSelectedId(assistantId)
          setActiveSource(null)
          setConversationId(data.conversation_id)
        } catch (exc) {
          setMessages((prev) => [
            ...prev,
            {
              id: assistantId,
              role: 'assistant',
              status: 'abstained',
              answer: `Could not reach the backend: ${exc.message}`,
              sources: [],
              claims: [],
            },
          ])
        } finally {
          setStreaming(false)
          setLiveStage(null)
          refreshConversations()
        }
        return
      }

      try {
        await api.chatStream(
          { message: text, conversation_id: conversationId },
          (event, data) => {
            if (event === 'open') {
              setConversationId(data.conversation_id)
            } else if (event === 'stage') {
              setLiveStage(data.stage)
              if (data.gate) {
                gate = data.gate
                setLiveGate(data.gate)
              }
            } else if (event === 'sources') {
              partial = { ...partial, sources: data.sources, retrieval: data.retrieval }
              setLiveRetrieval(data.retrieval)
            } else if (event === 'result') {
              const assistantMessage = {
                id: assistantId,
                role: 'assistant',
                ...data,
                trace: { stage: 'verifying', gate },
              }
              setMessages((prev) => [...prev, assistantMessage])
              setSelectedId(assistantId)
              setActiveSource(null)
              setConversationId(data.conversation_id)
            } else if (event === 'error') {
              setMessages((prev) => [
                ...prev,
                {
                  id: assistantId,
                  role: 'assistant',
                  status: 'abstained',
                  answer: `The request failed: ${data.detail || data.message}`,
                  sources: [],
                  claims: [],
                },
              ])
            }
          },
          abortRef.current.signal,
        )
      } catch (exc) {
        if (exc.name !== 'AbortError') {
          setMessages((prev) => [
            ...prev,
            {
              id: assistantId,
              role: 'assistant',
              status: 'abstained',
              answer: `Could not reach the backend: ${exc.message}`,
              sources: [],
              claims: [],
            },
          ])
        }
      } finally {
        setStreaming(false)
        setLiveStage(null)
        refreshConversations()
      }
    },
    [conversationId, refreshConversations, status],
  )

  const stop = () => abortRef.current?.abort()

  const selectSource = (index) => {
    setActiveSource(index)
    // Nudge the inspector to the referenced clause so a citation click lands
    // on the actual text rather than just highlighting a chip.
    requestAnimationFrame(() => {
      document
        .getElementById(`source-${index}`)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    })
  }

  const openConversation = async (id) => {
    try {
      const data = await api.conversation(id)
      const restored = data.turns.map((turn) => ({
        id: nextId(),
        role: turn.role,
        content: turn.content,
        ...(turn.payload || {}),
        ...(turn.role === 'assistant' ? { turn_id: turn.id } : {}),
      }))
      setMessages(restored)
      setConversationId(id)
      setSelectedId(null)
      setTab('chat')
    } catch {
      /* ignore */
    }
  }

  const newConversation = () => {
    setMessages([])
    setConversationId(null)
    setSelectedId(null)
    setActiveSource(null)
    setTab('chat')
  }

  const selectedMessage = messages.find((m) => m.id === selectedId) || null
  const indexReady = status?.index_ready

  return (
    <div className="app">
      <nav className="sidebar">
        <div className="brand">
          <h1>3GPP Assistant</h1>
          <p>Grounded retrieval over telecom standards</p>
        </div>

        <div className="tabs">
          {['chat', 'corpus', 'evaluation'].map((t) => (
            <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>
              {t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>

        <div className="sidebar-section">
          <div className="sidebar-heading">
            <span>Conversations</span>
            <button className="link" onClick={newConversation}>
              New
            </button>
          </div>
          <ul className="conversation-list">
            {conversations.length === 0 && <li className="muted">No history yet</li>}
            {conversations.slice(0, 12).map((c) => (
              <li key={c.id}>
                <button
                  className={c.id === conversationId ? 'active' : ''}
                  onClick={() => openConversation(c.id)}
                  title={c.title}
                >
                  {c.title}
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="sidebar-footer">
          {statusError ? (
            <div className="status status-bad">
              Backend unreachable
              <span>{statusError}</span>
            </div>
          ) : !status ? (
            <div className="status">Connecting…</div>
          ) : (
            <div className={`status ${indexReady ? 'status-ok' : 'status-warn'}`}>
              <strong>{indexReady ? 'Index ready' : 'No index'}</strong>
              <span>
                {status.chunk_count} chunks · {status.documents.length} specs
              </span>
              <span>
                {status.mode === 'generative'
                  ? `${status.provider} · ${status.models.answer}`
                  : 'extractive mode'}
              </span>
              {status.mode === 'extractive' && (
                <span className="status-note">
                  Set GEMINI_API_KEY or ANTHROPIC_API_KEY to enable synthesised answers
                  and entailment verification.
                </span>
              )}
              {status.capabilities?.streaming === false && (
                <span className="status-note">
                  Serverless deployment — answers arrive complete rather than
                  stage-by-stage, and history is per-instance.
                </span>
              )}
            </div>
          )}
        </div>
      </nav>

      <main className="main">
        {tab === 'chat' && (
          <ChatView
            messages={messages}
            streaming={streaming}
            liveStage={liveStage}
            liveGate={liveGate}
            liveRetrieval={liveRetrieval}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onSend={send}
            onCite={selectSource}
            activeSource={activeSource}
            onStop={stop}
            disabled={!indexReady}
          />
        )}
        {tab === 'corpus' && <CorpusView status={status} onRefresh={refreshStatus} />}
        {tab === 'evaluation' && <EvaluationView status={status} />}
      </main>

      {tab === 'chat' && (
        <SourcePanel
          message={selectedMessage}
          activeSource={activeSource}
          onCite={selectSource}
        />
      )}
    </div>
  )
}
