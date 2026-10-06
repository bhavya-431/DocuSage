"""Chat session and message entities.

Note (PRD §5): v1 has NO conversation memory — each question is answered from
retrieval alone. Messages are persisted only as a per-user history record with
the evidence that backed each answer.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ChatSession(Base):
    """A chat conversation container owned by exactly one user."""

    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), default="New chat", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ChatMessage(Base):
    """One question/answer exchange with its verified evidence payload."""

    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    # Document selection at ask time ([] = search all of the user's documents)
    document_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    # Verified citations: [{index, chunk_id, document_id, filename, page_number,
    #                       text, similarity}] — only ever written post-validation.
    citations: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    refused: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    refusal_gate: Mapped[str | None] = mapped_column(String(16), nullable=True)
    confidence_label: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
