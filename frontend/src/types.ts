/** Shared API types — mirrors the FastAPI Pydantic response models. */

export interface User {
  id: string
  email: string
  created_at: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: User
}

export type DocumentStatus = 'uploaded' | 'processing' | 'ready' | 'failed'

export interface DocumentItem {
  id: string
  filename: string
  file_size_bytes: number
  page_count: number
  status: DocumentStatus
  error_message: string | null
  created_at: string
  updated_at: string
  chunk_count?: number
}

export interface Citation {
  index: number
  chunk_id: string
  document_id: string
  filename: string
  page_number: number
  text: string
  similarity: number
}

export type ConfidenceLabel = 'High' | 'Medium' | 'Low' | 'Refused'

export interface ChatSession {
  id: string
  title: string
  created_at: string
}

export interface ChatMessage {
  id: string
  session_id: string
  question: string
  answer: string
  document_ids: string[]
  citations: Citation[]
  refused: boolean
  refusal_gate: string | null
  confidence_label: ConfidenceLabel
  confidence_score: number
  created_at: string
}
