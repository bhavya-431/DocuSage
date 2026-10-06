"""Chat API — streaming Q&A over the user's selected documents.

Streaming safety contract: token events only ever contain text that already
passed citation validation (via ``StreamingCitationSanitizer``), so a
fabricated citation can never render live. The final ``done`` event carries
the authoritative ``GenerationResult`` produced by ``GenerationService.finalize``
(the exact same code path as the non-streaming service), plus the persisted
message with its verified citations.

PRD notes:
- Chat over *selected* documents (``document_ids`` filter, still user-scoped).
- NO conversation memory (§5): the prompt is built from retrieval + the
  current question only; history is persisted for display, never injected.
"""
import json
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models.chat import ChatMessage, ChatSession
from app.models.user import User
from app.providers.embeddings import get_embedding_provider
from app.providers.llm import get_llm_provider
from app.repositories.chat_repo import ChatRepository
from app.repositories.document_repo import DocumentRepository
from app.services.generation import (
    SYSTEM_PROMPT,
    GenerationResult,
    GenerationService,
    StreamingCitationSanitizer,
)
from app.services.retrieval import RetrievalResult, RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=255)


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    question: str
    answer: str
    document_ids: list[str]
    citations: list[dict]
    refused: bool
    refusal_gate: str | None
    confidence_label: str
    confidence_score: float
    created_at: datetime


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[uuid.UUID] | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Error polish: a whitespace-only question is a clear 422, not an
        empty retrieval against the user's documents."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question must not be blank.")
        return stripped


def _sse(payload: dict) -> str:
    """Format one server-sent-event frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _citation_payload(
    retrieval_result: RetrievalResult,
    used_indexes: list[int],
    filenames: dict[uuid.UUID, str],
) -> list[dict]:
    """Build the source-panel payload from *actually retrieved* chunks only."""
    by_index = {c.citation_index: c for c in retrieval_result.chunks}
    payload: list[dict] = []
    for idx in used_indexes:
        chunk = by_index.get(idx)
        if chunk is None:
            continue  # defensive: never render a citation that was not retrieved
        payload.append(
            {
                "index": idx,
                "chunk_id": str(chunk.chunk_id),
                "document_id": str(chunk.document_id),
                "filename": filenames.get(chunk.document_id, "unknown.pdf"),
                "page_number": chunk.page_number,
                "text": chunk.text,
                "similarity": round(chunk.similarity, 4),
            }
        )
    return payload


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post(
    "/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_session(
    req: CreateSessionRequest = CreateSessionRequest(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatSession:
    """Create an empty chat session owned by the authenticated user."""
    return await ChatRepository.create_session(db, current_user.id, title=req.title)


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ChatSession]:
    """List all chat sessions belonging to the authenticated user."""
    return await ChatRepository.list_sessions(db, current_user.id)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def list_messages(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ChatMessage]:
    """List a session's messages — 404 unless the session belongs to the caller."""
    chat_session = await ChatRepository.get_session(db, current_user.id, session_id)
    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found or access denied.",
        )
    return await ChatRepository.list_messages(db, current_user.id, session_id)


@router.post("/sessions/{session_id}/messages")
async def ask(
    session_id: uuid.UUID,
    req: AskRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Ask one question and stream the validated answer as SSE.

    Events: ``token``* → ``done`` (carries the persisted message), or a lone
    ``done`` for retrieval-gate refusals (no LLM call is ever made then).
    """
    # --- Session ownership (hard isolation) ---
    chat_session = await ChatRepository.get_session(db, current_user.id, session_id)
    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found or access denied.",
        )

    # --- Document selection: every id must belong to the caller ---
    document_ids: list[uuid.UUID] | None = req.document_ids or None
    if document_ids:
        owned_docs = await DocumentRepository.get_by_ids(
            db, current_user.id, document_ids
        )
        if len(owned_docs) != len(set(document_ids)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="One or more selected documents were not found in your account.",
            )
        filenames = {d.id: d.filename for d in owned_docs}
    else:
        all_docs = await DocumentRepository.list_documents(db, current_user.id)
        filenames = {d.id: d.filename for d in all_docs}

    # --- Services (same wiring pattern as the documents API) ---
    retrieval_service = RetrievalService(get_embedding_provider())
    generation_service = GenerationService(get_llm_provider())

    # --- Retrieval runs before the stream starts so failures surface as HTTP ---
    retrieval = await retrieval_service.retrieve(
        db, current_user.id, req.question, document_ids
    )

    async def _persist(result: GenerationResult) -> ChatMessage:
        message = await ChatRepository.create_message(
            session=db,
            user_id=current_user.id,
            session_id=session_id,
            question=req.question,
            answer=result.answer,
            document_ids=[str(d) for d in document_ids] if document_ids else [],
            citations=_citation_payload(retrieval, result.citations_used, filenames),
            refused=result.refused,
            refusal_gate=result.refusal_gate,
            confidence_label=result.confidence.value,
            confidence_score=result.confidence_score,
        )
        # Title the session from its first question for the sidebar.
        if chat_session.title == "New chat":
            await ChatRepository.update_title(
                db, current_user.id, session_id, req.question[:60]
            )
        await db.commit()
        return message

    def _message_event(message: ChatMessage) -> str:
        payload = MessageResponse.model_validate(message).model_dump(mode="json")
        return _sse({"type": "done", "message": payload})

    async def event_stream():
        # ---- Gate 1: retrieval refused → no LLM call at all ----
        if not retrieval.gate_passed:
            result = generation_service.refusal(
                "retrieval", retrieval.refusal_reason or "insufficient evidence", retrieval
            )
            yield _message_event(await _persist(result))
            return

        prompt = generation_service.build_prompt(req.question, retrieval)
        sanitizer = StreamingCitationSanitizer(
            GenerationService.owned_indexes(retrieval, current_user.id)
        )
        raw_parts: list[str] = []

        try:
            async for token in generation_service.llm_provider.stream_response(
                prompt, system_prompt=SYSTEM_PROMPT
            ):
                raw_parts.append(token)
                safe_text = sanitizer.feed(token)
                if safe_text:
                    yield _sse({"type": "token", "text": safe_text})
            tail = sanitizer.flush()
            if tail:
                yield _sse({"type": "token", "text": tail})
        except Exception as exc:
            logger.error("LLM streaming failed: %s", exc, exc_info=True)
            result = generation_service.refusal(
                "generation",
                "the language model failed mid-answer, so the partial response "
                "was discarded",
                retrieval,
            )
            yield _sse({"type": "error", "detail": "Generation failed. Please retry."})
            yield _message_event(await _persist(result))
            return

        # ---- Gate 2 + citation integrity + confidence (authoritative result) ----
        result = generation_service.finalize(
            "".join(raw_parts), retrieval, current_user.id
        )
        yield _message_event(await _persist(result))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
