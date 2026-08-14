/**
 * Backend client.
 *
 * SSE is consumed with fetch + ReadableStream rather than EventSource, because
 * EventSource cannot issue POST requests and the chat endpoint needs a JSON
 * body. The parser below handles the one thing hand-rolled SSE readers usually
 * get wrong: a chunk boundary can land in the middle of an event, so partial
 * frames must be buffered rather than parsed eagerly.
 */

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  const text = await response.text()
  let body = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = { error: text }
  }
  if (!response.ok) {
    throw new Error(body?.error || body?.detail || `Request failed (${response.status})`)
  }
  return body
}

/** Read an SSE endpoint, invoking `onEvent(name, data)` per frame. */
async function streamSSE(path, payload, onEvent, signal) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Request failed (${response.status})`)
  }
  if (!response.body) throw new Error('Streaming is not supported by this browser.')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // Frames are separated by a blank line. Keep the trailing partial frame.
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      if (!frame.trim()) continue
      let name = 'message'
      const dataLines = []
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) name = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
      }
      if (!dataLines.length) continue
      try {
        onEvent(name, JSON.parse(dataLines.join('\n')))
      } catch {
        // A malformed frame should not tear down an otherwise healthy stream.
      }
    }
  }
}

export const api = {
  status: () => request('/api/status'),
  health: () => request('/api/health'),

  chatStream: (payload, onEvent, signal) =>
    streamSSE('/api/chat/stream', payload, onEvent, signal),
  chat: (payload) => request('/api/chat', { method: 'POST', body: JSON.stringify(payload) }),
  search: (query, topN = 8) =>
    request('/api/search', { method: 'POST', body: JSON.stringify({ query, top_n: topN }) }),

  documents: () => request('/api/documents'),
  reindex: () => request('/api/documents/reindex', { method: 'POST' }),
  chunk: (chunkId) => request(`/api/documents/chunk/${chunkId}`),
  upload: async (file) => {
    const form = new FormData()
    form.append('file', file)
    const response = await fetch('/api/documents/upload', { method: 'POST', body: form })
    const body = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(body?.error || 'Upload failed')
    return body
  },

  conversations: () => request('/api/conversations'),
  conversation: (id) => request(`/api/conversations/${id}`),
  deleteConversation: (id) => request(`/api/conversations/${id}`, { method: 'DELETE' }),

  feedback: (payload) => request('/api/feedback', { method: 'POST', body: JSON.stringify(payload) }),

  goldenSet: () => request('/api/evaluation/golden-set'),
  runEvaluation: (payload, onEvent, signal) =>
    streamSSE('/api/evaluation/run', payload, onEvent, signal),
}
