import { type FormEvent, useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, api, askQuestion } from '../lib/api'
import type { ChatMessage, ChatSession, Citation, DocumentItem } from '../types'

/** In-flight exchange: live-token draft until the authoritative `done` event. */
interface Draft {
  key: string
  question: string
  text: string
  error: string | null
}

const BADGE_STYLES: Record<string, string> = {
  High: 'border-emerald-600/50 bg-emerald-500/10 text-emerald-300',
  Medium: 'border-amber-600/50 bg-amber-500/10 text-amber-300',
  Low: 'border-orange-600/50 bg-orange-500/10 text-orange-300',
  Refused: 'border-rose-600/50 bg-rose-500/10 text-rose-300',
}

function ConfidenceBadge({
  label,
  score,
}: {
  label: string
  score: number
}) {
  return (
    <span
      title={`Heuristic (not calibrated). Composite similarity: ${score.toFixed(2)}`}
      className={`inline-block rounded-full border px-2 py-0.5 text-[11px] font-medium ${
        BADGE_STYLES[label] ?? BADGE_STYLES.Medium
      }`}
    >
      {label}
    </span>
  )
}

/** Render answer text with clickable `[n]` citation chips. */
function AnswerBody({
  text,
  citations,
  onOpen,
}: {
  text: string
  citations: Citation[]
  onOpen: (c: Citation) => void
}) {
  const parts = text.split(/(\[\d+\])/g)
  return (
    <p className="whitespace-pre-wrap text-sm leading-6 text-slate-200">
      {parts.map((part, i) => {
        const m = /^\[(\d+)\]$/.exec(part)
        if (!m) return <span key={i}>{part}</span>
        const cite = citations.find((c) => c.index === Number(m[1]))
        if (!cite) return <span key={i}>{part}</span>
        return (
          <button
            key={i}
            onClick={() => onOpen(cite)}
            className="mx-0.5 inline-block rounded border border-sky-700/60 bg-sky-500/15 px-1 text-xs font-medium text-sky-300 transition hover:bg-sky-500/30"
            title={`Page ${cite.page_number} — click to view source`}
          >
            {part}
          </button>
        )
      })}
    </p>
  )
}

export default function ChatPanel() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [docs, setDocs] = useState<DocumentItem[]>([])
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])
  const [question, setQuestion] = useState('')
  const [sending, setSending] = useState(false)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // --- Initial loads ---
  useEffect(() => {
    void (async () => {
      try {
        const [s, d] = await Promise.all([api.listSessions(), api.listDocuments()])
        setSessions(s)
        setDocs(d.filter((doc) => doc.status === 'ready'))
      } catch (err) {
        setError(err instanceof ApiError ? err.message : 'Failed to load chat data.')
      }
    })()
  }, [])

  // --- Auto-scroll to newest content ---
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, draft])

  const loadMessages = useCallback(async (id: string) => {
    setError(null)
    setDraft(null)
    try {
      setMessages(await api.listMessages(id))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to load messages.')
    }
  }, [])

  async function newChat() {
    setSessionId(null)
    setMessages([])
    setDraft(null)
    setError(null)
  }

  async function selectSession(id: string) {
    setSessionId(id)
    await loadMessages(id)
  }

  function toggleDoc(id: string) {
    setSelectedDocs((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    )
  }

  async function send(e: FormEvent) {
    e.preventDefault()
    const q = question.trim()
    if (!q || sending) return

    setSending(true)
    setError(null)
    setQuestion('')
    setDraft({ key: `d-${Date.now()}`, question: q, text: '', error: null })

    try {
      let sid = sessionId
      if (!sid) {
        const session = await api.createSession('New chat')
        sid = session.id
        setSessionId(sid)
        setSessions((prev) => [session, ...prev])
      }

      await askQuestion(sid, q, selectedDocs, {
        onToken: (text) =>
          setDraft((prev) => (prev ? { ...prev, text: prev.text + text } : prev)),
        onError: (detail) =>
          setDraft((prev) => (prev ? { ...prev, error: detail } : prev)),
        onDone: (message) => {
          setMessages((prev) => [...prev, message])
          setDraft(null)
        },
      })
    } catch (err) {
      const detail = err instanceof ApiError ? err.message : 'Request failed.'
      setDraft((prev) => (prev ? { ...prev, error: detail } : null))
      setError(detail)
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="grid gap-4 md:grid-cols-[210px_1fr]">
      {/* ---------- Sidebar: sessions ---------- */}
      <aside className="space-y-2">
        <button
          onClick={() => void newChat()}
          className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-200 transition hover:bg-slate-700"
        >
          + New chat
        </button>
        <ul className="space-y-1">
          {sessions.map((s) => (
            <li key={s.id}>
              <button
                onClick={() => void selectSession(s.id)}
                className={`w-full truncate rounded-md px-3 py-2 text-left text-xs transition ${
                  sessionId === s.id
                    ? 'bg-sky-500/15 text-sky-300'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
                title={s.title}
              >
                {s.title}
              </button>
            </li>
          ))}
        </ul>
        {docs.length > 0 && (
          <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
            <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-slate-500">
              Search within
            </p>
            <div className="flex flex-wrap gap-1.5">
              {docs.map((d) => {
                const active = selectedDocs.includes(d.id)
                return (
                  <button
                    key={d.id}
                    onClick={() => toggleDoc(d.id)}
                    className={`max-w-[140px] truncate rounded-full border px-2 py-0.5 text-[11px] transition ${
                      active
                        ? 'border-sky-600 bg-sky-500/20 text-sky-300'
                        : 'border-slate-700 text-slate-400 hover:border-slate-500'
                    }`}
                    title={d.filename}
                  >
                    {d.filename}
                  </button>
                )
              })}
            </div>
            <p className="mt-2 text-[10px] text-slate-600">
              None selected = all your documents.
            </p>
          </div>
        )}
      </aside>

      {/* ---------- Main: conversation ---------- */}
      <section className="flex h-[70vh] flex-col rounded-xl border border-slate-800 bg-slate-900/50">
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.length === 0 && !draft && (
            <div className="pt-10 text-center">
              <p className="text-sm text-slate-400">
                Ask a question about your documents.
              </p>
              <p className="mt-1 text-xs text-slate-600">
                Answers carry page-level citations and are refused when evidence is
                insufficient.
              </p>
            </div>
          )}

          {messages.map((m) => (
            <div key={m.id} className="space-y-2">
              <div className="flex justify-end">
                <p className="max-w-[85%] rounded-2xl rounded-br-sm bg-sky-600/90 px-4 py-2 text-sm text-white">
                  {m.question}
                </p>
              </div>
              <div className="max-w-[92%] rounded-2xl rounded-bl-sm border border-slate-700 bg-slate-800/70 px-4 py-3">
                <div className="mb-1.5 flex items-center gap-2">
                  <ConfidenceBadge label={m.confidence_label} score={m.confidence_score} />
                  {m.refused && m.refusal_gate && (
                    <span className="text-[11px] text-slate-500">
                      gate: {m.refusal_gate}
                    </span>
                  )}
                </div>
                <AnswerBody
                  text={m.answer}
                  citations={m.citations}
                  onOpen={setActiveCitation}
                />
                {m.citations.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1.5 border-t border-slate-700/70 pt-2">
                    {m.citations.map((c) => (
                      <button
                        key={c.index}
                        onClick={() => setActiveCitation(c)}
                        className="max-w-[220px] truncate rounded border border-slate-600 bg-slate-700/50 px-1.5 py-0.5 text-[11px] text-slate-300 hover:bg-slate-600"
                        title={c.text}
                      >
                        [{c.index}] {c.filename} · p.{c.page_number}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Live draft (streaming) */}
          {draft && (
            <div className="space-y-2">
              <div className="flex justify-end">
                <p className="max-w-[85%] rounded-2xl rounded-br-sm bg-sky-600/90 px-4 py-2 text-sm text-white">
                  {draft.question}
                </p>
              </div>
              <div className="max-w-[92%] rounded-2xl rounded-bl-sm border border-slate-700 bg-slate-800/70 px-4 py-3">
                <p className="whitespace-pre-wrap text-sm leading-6 text-slate-200">
                  {draft.text || (
                    <span className="text-slate-500">Thinking…</span>
                  )}
                  <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-sky-500 align-text-bottom" />
                </p>
                {draft.error && (
                  <p className="mt-2 text-xs text-rose-400">{draft.error}</p>
                )}
              </div>
            </div>
          )}

          {error && !draft && (
            <p className="rounded-lg border border-rose-800/60 bg-rose-950/60 px-3 py-2 text-xs text-rose-300">
              {error}
            </p>
          )}
          <div ref={bottomRef} />
        </div>

        {/* ---------- Composer ---------- */}
        <form
          onSubmit={(e) => void send(e)}
          className="flex gap-2 border-t border-slate-800 p-3"
        >
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask your documents…"
            maxLength={2000}
            className="flex-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-sky-600"
          />
          <button
            type="submit"
            disabled={sending || !question.trim()}
            className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-sky-500 disabled:opacity-40"
          >
            {sending ? '…' : 'Send'}
          </button>
        </form>
      </section>

      {/* ---------- Source panel modal ---------- */}
      {activeCitation && (
        <div
          className="fixed inset-0 z-20 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setActiveCitation(null)}
        >
          <div
            className="w-full max-w-lg rounded-xl border border-slate-700 bg-slate-900 p-5 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-3 flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-white">
                  {activeCitation.filename}
                </p>
                <p className="text-xs text-slate-400">
                  Page {activeCitation.page_number} · similarity{' '}
                  {(activeCitation.similarity * 100).toFixed(1)}% · citation [
                  {activeCitation.index}]
                </p>
              </div>
              <button
                onClick={() => setActiveCitation(null)}
                className="rounded-md px-2 py-1 text-slate-400 hover:bg-slate-800 hover:text-white"
              >
                ✕
              </button>
            </div>
            <div className="max-h-[50vh] overflow-y-auto rounded-lg border border-slate-800 bg-slate-950 p-3">
              <p className="whitespace-pre-wrap text-sm leading-6 text-slate-300">
                {activeCitation.text}
              </p>
            </div>
            <p className="mt-2 text-[11px] text-slate-600">
              Exact retrieved chunk — the only evidence the answer was allowed to
              cite.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
