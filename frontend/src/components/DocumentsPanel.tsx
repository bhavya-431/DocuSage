import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, api } from '../lib/api'
import type { DocumentItem } from '../types'

const STATUS_STYLES: Record<string, string> = {
  uploaded: 'border-slate-600 bg-slate-700/40 text-slate-300',
  processing: 'border-amber-600/50 bg-amber-500/10 text-amber-300',
  ready: 'border-emerald-600/50 bg-emerald-500/10 text-emerald-300',
  failed: 'border-rose-600/50 bg-rose-500/10 text-rose-300',
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function DocumentsPanel() {
  const [docs, setDocs] = useState<DocumentItem[]>([])
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const refresh = useCallback(async () => {
    try {
      setDocs(await api.listDocuments())
      setLoaded(true)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to load documents.')
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  // Poll while any document is still moving through the pipeline
  const pending = docs.some((d) => d.status === 'uploaded' || d.status === 'processing')
  useEffect(() => {
    if (!pending) return
    const id = setInterval(() => void refresh(), 2500)
    return () => clearInterval(id)
  }, [pending, refresh])

  async function upload(file: File) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are supported in v1.')
      return
    }
    setUploading(true)
    setError(null)
    setNotice(null)
    try {
      await api.uploadDocument(file)
      setNotice(`"${file.name}" uploaded — processing started.`)
      await refresh()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Upload failed.')
    } finally {
      setUploading(false)
    }
  }

  async function remove(id: string) {
    setError(null)
    try {
      await api.deleteDocument(id)
      await refresh()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Delete failed.')
    }
  }

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-semibold text-white">Your documents</h2>
        <p className="text-sm text-slate-400">
          PDFs are chunked, embedded, and stored per account (max 20 docs / 25 MB each).
        </p>
      </div>

      {/* Drag & drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          const file = e.dataTransfer.files[0]
          if (file) void upload(file)
        }}
        onClick={() => fileInput.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed px-6 py-10 text-center transition ${
          dragOver
            ? 'border-sky-500 bg-sky-500/10'
            : 'border-slate-700 bg-slate-900/60 hover:border-slate-500'
        }`}
      >
        <input
          ref={fileInput}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) void upload(file)
            e.target.value = ''
          }}
        />
        <p className="text-sm text-slate-300">
          {uploading ? 'Uploading…' : 'Drag & drop a PDF here, or click to browse'}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          Encrypted, corrupted, scanned, oversized, and duplicate files are rejected
          gracefully.
        </p>
      </div>

      {notice && (
        <p className="rounded-lg border border-emerald-800/60 bg-emerald-950/50 px-3 py-2 text-xs text-emerald-300">
          {notice}
        </p>
      )}
      {error && (
        <p className="rounded-lg border border-rose-800/60 bg-rose-950/60 px-3 py-2 text-xs text-rose-300">
          {error}
        </p>
      )}

      {/* Document list */}
      {loaded && docs.length === 0 ? (
        <p className="rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-8 text-center text-sm text-slate-500">
          No documents yet — upload a PDF to start asking questions.
        </p>
      ) : (
        <ul className="space-y-2">
          {docs.map((doc) => (
            <li
              key={doc.id}
              className="flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3"
            >
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-slate-200">
                  {doc.filename}
                </p>
                <p className="text-xs text-slate-500">
                  {formatBytes(doc.file_size_bytes)}
                  {doc.page_count > 0 && ` · ${doc.page_count} pages`}
                  {typeof doc.chunk_count === 'number' && doc.chunk_count > 0 && ` · ${doc.chunk_count} chunks`}
                </p>
                {doc.status === 'failed' && doc.error_message && (
                  <p className="mt-1 text-xs text-rose-400">{doc.error_message}</p>
                )}
              </div>
              <span
                className={`rounded-full border px-2.5 py-0.5 text-xs capitalize ${STATUS_STYLES[doc.status] ?? STATUS_STYLES.uploaded} ${
                  doc.status === 'processing' ? 'animate-pulse' : ''
                }`}
              >
                {doc.status}
              </span>
              <button
                onClick={() => void remove(doc.id)}
                className="rounded-md px-2 py-1 text-xs text-slate-500 transition hover:bg-rose-500/10 hover:text-rose-300"
                title="Delete document"
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
