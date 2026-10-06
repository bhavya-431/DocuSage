/**
 * API client for the DocuSage backend.
 *
 * Includes a fetch-based SSE reader for the streaming chat endpoint
 * (EventSource cannot POST, so we parse `data: {...}\n\n` frames manually).
 */
import type { AuthResponse, ChatMessage, ChatSession, DocumentItem, User } from '../types'

const API_BASE =
  (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

const TOKEN_KEY = 'docusage_token'
const USER_KEY = 'docusage_user'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export const auth = {
  token(): string | null {
    return localStorage.getItem(TOKEN_KEY)
  },
  user(): User | null {
    const raw = localStorage.getItem(USER_KEY)
    return raw ? (JSON.parse(raw) as User) : null
  },
  store(res: AuthResponse): void {
    localStorage.setItem(TOKEN_KEY, res.access_token)
    localStorage.setItem(USER_KEY, JSON.stringify(res.user))
  },
  clear(): void {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  },
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    ...((init.headers as Record<string, string>) ?? {}),
  }
  const token = auth.token()
  if (token) headers.Authorization = `Bearer ${token}`
  if (init.body && !(init.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }

  const res = await fetch(`${API_BASE}${path}`, { ...init, headers })
  if (!res.ok) {
    throw new ApiError(res.status, await extractDetail(res))
  }
  return (await res.json()) as T
}

async function extractDetail(res: Response): Promise<string> {
  try {
    const data = (await res.json()) as { detail?: unknown }
    if (typeof data.detail === 'string') return data.detail
    if (data.detail) return JSON.stringify(data.detail)
  } catch {
    /* non-JSON error body */
  }
  return res.statusText || `Request failed (${res.status})`
}

export const api = {
  register: (email: string, password: string) =>
    request<AuthResponse>('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  login: (email: string, password: string) =>
    request<AuthResponse>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  health: () => request<{ status: string }>('/health'),

  listDocuments: () => request<DocumentItem[]>('/api/documents'),

  uploadDocument: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return request<DocumentItem>('/api/documents/upload', { method: 'POST', body: fd })
  },

  deleteDocument: (id: string) =>
    request<{ deleted: boolean }>(`/api/documents/${id}`, { method: 'DELETE' }),

  listSessions: () => request<ChatSession[]>('/api/chat/sessions'),

  createSession: (title = 'New chat') =>
    request<ChatSession>('/api/chat/sessions', {
      method: 'POST',
      body: JSON.stringify({ title }),
    }),

  listMessages: (sessionId: string) =>
    request<ChatMessage[]>(`/api/chat/sessions/${sessionId}/messages`),
}

export interface AskHandlers {
  onToken: (text: string) => void
  onDone: (message: ChatMessage) => void
  onError: (detail: string) => void
}

/**
 * POST one question and consume the SSE stream.
 * Events: `token`* → `done` (authoritative validated message), or `error`.
 */
export async function askQuestion(
  sessionId: string,
  question: string,
  documentIds: string[] | null,
  handlers: AskHandlers,
): Promise<void> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const token = auth.token()
  if (token) headers.Authorization = `Bearer ${token}`

  const res = await fetch(`${API_BASE}/api/chat/sessions/${sessionId}/messages`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      question,
      document_ids: documentIds && documentIds.length > 0 ? documentIds : null,
    }),
  })

  if (!res.ok || !res.body) {
    throw new ApiError(res.status, await extractDetail(res))
  }

  const dispatch = (frame: string): void => {
    if (!frame.startsWith('data: ')) return
    let evt: {
      type?: string
      text?: string
      message?: ChatMessage
      detail?: string
    }
    try {
      evt = JSON.parse(frame.slice(6)) as typeof evt
    } catch {
      return
    }
    if (evt.type === 'token' && evt.text) handlers.onToken(evt.text)
    else if (evt.type === 'done' && evt.message) handlers.onDone(evt.message)
    else if (evt.type === 'error') handlers.onError(evt.detail ?? 'Generation failed.')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx = buffer.indexOf('\n\n')
    while (idx !== -1) {
      dispatch(buffer.slice(0, idx))
      buffer = buffer.slice(idx + 2)
      idx = buffer.indexOf('\n\n')
    }
  }
  if (buffer.trim()) dispatch(buffer)
}
