import uuid
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.document import Document


class DocumentRepository:
    """Repository strictly enforcing tenant user_id isolation on all document operations."""

    @staticmethod
    async def create_document(
        session: AsyncSession,
        user_id: uuid.UUID,
        filename: str,
        file_hash: str,
        file_size_bytes: int,
        page_count: int = 0,
        status: str = "uploaded",
    ) -> Document:
        doc = Document(
            user_id=user_id,
            filename=filename,
            file_hash=file_hash,
            file_size_bytes=file_size_bytes,
            page_count=page_count,
            status=status,
        )
        session.add(doc)
        await session.flush()
        return doc

    @staticmethod
    async def get_by_id(
        session: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> Document | None:
        """Fetch document by ID, strictly scoped to user_id."""
        stmt = select(Document).where(
            Document.id == document_id, Document.user_id == user_id
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_ids(
        session: AsyncSession, user_id: uuid.UUID, document_ids: list[uuid.UUID]
    ) -> list[Document]:
        """Fetch multiple documents by ID, strictly scoped to user_id."""
        stmt = select(Document).where(
            Document.user_id == user_id, Document.id.in_(document_ids)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_by_hash(
        session: AsyncSession, user_id: uuid.UUID, file_hash: str
    ) -> Document | None:
        """Find existing document with matching hash, scoped to the current user."""
        stmt = select(Document).where(
            Document.user_id == user_id, Document.file_hash == file_hash
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def list_documents(
        session: AsyncSession, user_id: uuid.UUID
    ) -> list[Document]:
        """List all documents belonging to user_id in reverse chronological order."""
        stmt = (
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def count_by_user(session: AsyncSession, user_id: uuid.UUID) -> int:
        """Count total documents uploaded by user to enforce quota (e.g. max 20)."""
        stmt = (
            select(func.count(Document.id))
            .where(Document.user_id == user_id)
        )
        result = await session.execute(stmt)
        return result.scalar() or 0

    @staticmethod
    async def update_status(
        session: AsyncSession,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        status: str,
        page_count: int | None = None,
        error_message: str | None = None,
    ) -> Document | None:
        """Update processing status strictly for the document owned by user_id."""
        doc = await DocumentRepository.get_by_id(session, user_id, document_id)
        if not doc:
            return None
        doc.status = status
        if page_count is not None:
            doc.page_count = page_count
        if error_message is not None:
            doc.error_message = error_message
        await session.flush()
        return doc

    @staticmethod
    async def delete_document(
        session: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> bool:
        """Delete document and cascade chunks, strictly scoped to user_id."""
        doc = await DocumentRepository.get_by_id(session, user_id, document_id)
        if not doc:
            return False
        await session.delete(doc)
        await session.flush()
        return True
