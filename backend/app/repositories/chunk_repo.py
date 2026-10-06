import uuid
from typing import Any
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.chunk import DocumentChunk


class ChunkRepository:
    """Repository strictly enforcing tenant user_id isolation on all chunk and vector search operations."""

    @staticmethod
    async def create_chunks(
        session: AsyncSession,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        chunks_data: list[dict[str, Any]],
    ) -> list[DocumentChunk]:
        """Bulk insert document chunks with vector embeddings and user_id tags."""
        chunks: list[DocumentChunk] = []
        for item in chunks_data:
            chunk = DocumentChunk(
                user_id=user_id,
                document_id=document_id,
                chunk_index=item["chunk_index"],
                text=item["text"],
                page_number=item["page_number"],
                char_count=item.get("char_count", len(item["text"])),
                embedding=item["embedding"],
            )
            chunks.append(chunk)
            session.add(chunk)
        await session.flush()
        return chunks

    @staticmethod
    async def similarity_search(
        session: AsyncSession,
        user_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int = 5,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Vector similarity search strictly scoped to user_id.

        Calculates cosine distance via pgvector:
        cosine distance <=> is in range [0, 2]
        cosine similarity = 1 - distance
        """
        distance_col = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
        similarity_col = (1.0 - distance_col).label("similarity")

        stmt = (
            select(DocumentChunk, similarity_col)
            .where(DocumentChunk.user_id == user_id)
        )

        if document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))

        stmt = stmt.order_by(distance_col.asc()).limit(top_k)

        result = await session.execute(stmt)
        return [(row[0], float(row[1])) for row in result.all()]

    @staticmethod
    async def count_by_document(
        session: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> int:
        """Count total chunks for a document owned by user_id."""
        stmt = (
            select(func.count(DocumentChunk.id))
            .where(
                DocumentChunk.user_id == user_id,
                DocumentChunk.document_id == document_id,
            )
        )
        result = await session.execute(stmt)
        return result.scalar() or 0
