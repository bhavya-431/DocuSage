import uuid
from sqlalchemy import ForeignKey, Integer, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.core.config import get_settings
from app.models.base import Base

settings = get_settings()


class DocumentChunk(Base):
    """Chunked document text with vector embeddings and page-level metadata."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), index=True, nullable=False
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding = mapped_column(Vector(settings.EMBEDDING_DIMENSION), nullable=False)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")  # type: ignore # noqa: F821
