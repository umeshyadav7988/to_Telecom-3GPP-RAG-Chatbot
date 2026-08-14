import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api/client.js'
import { EmptyState, Spinner } from './common.jsx'

export default function CorpusView({ status, onRefresh }) {
  const [documents, setDocuments] = useState(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState(null)
  const [error, setError] = useState(null)
  const fileInput = useRef(null)
  const readOnly = status?.capabilities?.corpus_writable === false

  const load = async () => {
    try {
      setDocuments(await api.documents())
      setError(null)
    } catch (exc) {
      setError(exc.message)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const reindex = async () => {
    setBusy(true)
    setMessage(null)
    setError(null)
    try {
      const report = await api.reindex()
      setMessage(
        `Indexed ${report.chunk_count} chunks from ${report.document_count} document(s) in ${report.elapsed_seconds}s.`,
      )
      await load()
      await onRefresh?.()
    } catch (exc) {
      setError(exc.message)
    } finally {
      setBusy(false)
    }
  }

  const upload = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      await api.upload(file)
      setMessage(`Uploaded ${file.name}. Rebuild the index to make it searchable.`)
      await load()
    } catch (exc) {
      setError(exc.message)
    } finally {
      setBusy(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  return (
    <section className="panel">
      <header className="panel-header">
        <div>
          <h2>Corpus</h2>
          <p className="muted">
            Specifications available to the retriever. Uploads are only searchable after a rebuild.
          </p>
        </div>
        <div className="panel-actions">
          <input
            ref={fileInput}
            type="file"
            accept=".pdf,.docx,.txt,.md"
            onChange={upload}
            style={{ display: 'none' }}
          />
          <button onClick={() => fileInput.current?.click()} disabled={busy || readOnly}>
            Upload spec
          </button>
          <button className="primary" onClick={reindex} disabled={busy || readOnly}>
            {busy ? <Spinner label="Rebuilding" /> : 'Rebuild index'}
          </button>
        </div>
      </header>

      {readOnly && (
        <div className="notice">
          This deployment has a read-only filesystem, so uploads and on-demand reindexing are
          disabled. The index is rebuilt from the bundled corpus on every cold start — to change
          the corpus, commit files to <code>backend/data/corpus/</code> and redeploy.
        </div>
      )}

      {message && <div className="notice notice-ok">{message}</div>}
      {error && <div className="notice notice-error">{error}</div>}

      <div className="stat-row">
        <Stat label="Documents" value={status?.documents?.length ?? 0} />
        <Stat label="Chunks" value={status?.chunk_count ?? 0} />
        <Stat label="Embedder" value={status?.embedder ?? '—'} small />
        <Stat
          label="LLM"
          value={
            status?.mode === 'generative'
              ? `${status.provider} · ${status.models?.answer}`
              : 'extractive (no key)'
          }
          small
        />
      </div>

      <h3>Indexed specifications</h3>
      {!status?.documents?.length ? (
        <EmptyState title="Nothing indexed">
          Put .pdf/.docx/.txt files in <code>backend/data/corpus/</code> (or use{' '}
          <code>scripts/download_3gpp.py</code>), then rebuild the index.
        </EmptyState>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Spec</th>
              <th>Title</th>
              <th>Ver</th>
              <th className="num">Clauses</th>
              <th className="num">Chunks</th>
              <th className="num">Normative</th>
            </tr>
          </thead>
          <tbody>
            {status.documents.map((doc) => (
              <tr key={doc.doc_id}>
                <td className="mono">{doc.doc_id}</td>
                <td>{doc.title}</td>
                <td className="mono">{doc.version || '—'}</td>
                <td className="num">{doc.clauses}</td>
                <td className="num">{doc.chunks}</td>
                <td className="num">{doc.normative_chunks}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h3>Files on disk</h3>
      {!documents ? (
        <Spinner label="Loading" />
      ) : documents.files.length === 0 ? (
        <p className="muted">No files found in {documents.corpus_dir}</p>
      ) : (
        <ul className="file-list">
          {documents.files.map((f) => (
            <li key={f.filename}>
              <span className="mono">{f.filename}</span>
              <span className="muted">{Math.round(f.size_bytes / 1024)} KB</span>
              <span className={`tag ${f.indexed ? 'tag-cited' : ''}`}>
                {f.indexed ? 'indexed' : 'not indexed'}
              </span>
            </li>
          ))}
        </ul>
      )}

      <h3>Active guardrails</h3>
      <table className="table">
        <tbody>
          {Object.entries(status?.guardrails || {}).map(([key, value]) => (
            <tr key={key}>
              <td>{key.replace(/_/g, ' ')}</td>
              <td className="mono">{String(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

function Stat({ label, value, small }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className={`stat-value${small ? ' small' : ''}`}>{value}</span>
    </div>
  )
}
