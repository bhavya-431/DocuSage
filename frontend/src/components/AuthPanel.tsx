import { type FormEvent, useState } from 'react'
import { ApiError, api, auth } from '../lib/api'
import type { User } from '../types'

interface Props {
  onAuthed: (user: User) => void
}

export default function AuthPanel({ onAuthed }: Props) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const res =
        mode === 'login'
          ? await api.login(email, password)
          : await api.register(email, password)
      auth.store(res)
      onAuthed(res.user)
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : 'Could not reach the API. Is the backend running on :8000?',
      )
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-sm rounded-xl border border-slate-800 bg-slate-900 p-6 shadow-2xl">
        <h1 className="text-xl font-semibold text-white">DocuSage</h1>
        <p className="mb-5 mt-1 text-sm text-slate-400">
          Upload PDFs, ask questions, get answers with page-level citations.
        </p>

        <div className="mb-4 grid grid-cols-2 gap-1 rounded-lg bg-slate-800 p-1">
          {(['login', 'register'] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => {
                setMode(m)
                setError(null)
              }}
              className={`rounded-md py-1.5 text-sm capitalize transition ${
                mode === m ? 'bg-slate-600 text-white' : 'text-slate-400'
              }`}
            >
              {m === 'login' ? 'Sign in' : 'Create account'}
            </button>
          ))}
        </div>

        <form onSubmit={submit} className="space-y-3">
          <input
            type="email"
            required
            autoComplete="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-sky-600"
          />
          <input
            type="password"
            required
            minLength={6}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            placeholder="Password (min 6 characters)"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-sky-600"
          />

          {error && (
            <p className="rounded-lg border border-rose-800/60 bg-rose-950/60 px-3 py-2 text-xs text-rose-300">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-lg bg-sky-600 py-2 text-sm font-medium text-white transition hover:bg-sky-500 disabled:opacity-50"
          >
            {busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}
          </button>
        </form>
      </div>
    </div>
  )
}
