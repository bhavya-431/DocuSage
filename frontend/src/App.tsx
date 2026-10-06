import { useState } from 'react'
import AuthPanel from './components/AuthPanel'
import ChatPanel from './components/ChatPanel'
import DocumentsPanel from './components/DocumentsPanel'
import { auth } from './lib/api'
import type { User } from './types'

type Tab = 'documents' | 'chat'

const TABS: { id: Tab; label: string }[] = [
  { id: 'documents', label: 'Documents' },
  { id: 'chat', label: 'Chat' },
]

export default function App() {
  const [user, setUser] = useState<User | null>(() => auth.user())
  const [tab, setTab] = useState<Tab>('documents')

  if (!user) return <AuthPanel onAuthed={setUser} />

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b border-slate-800 bg-slate-900/70 backdrop-blur">
        <div className="mx-auto flex h-14 w-full max-w-6xl items-center gap-3 px-4">
          <span className="text-lg font-semibold tracking-tight text-white">
            DocuSage
          </span>
          <span className="hidden text-xs text-slate-400 sm:inline">
            Enterprise Document Intelligence
          </span>
          <nav className="ml-auto flex gap-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`rounded-md px-3 py-1.5 text-sm transition ${
                  tab === t.id
                    ? 'bg-slate-700 text-white'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <span className="hidden text-xs text-slate-500 md:inline">{user.email}</span>
          <button
            onClick={() => {
              auth.clear()
              setUser(null)
            }}
            className="rounded-md px-2 py-1 text-xs text-slate-400 hover:bg-slate-800 hover:text-slate-200"
          >
            Logout
          </button>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
        {tab === 'documents' ? <DocumentsPanel /> : <ChatPanel />}
      </main>

      <footer className="border-t border-slate-900 py-3 text-center text-[11px] text-slate-600">
        Confidence is a heuristic based on retrieval similarity — not a calibrated
        probability.
      </footer>
    </div>
  )
}
