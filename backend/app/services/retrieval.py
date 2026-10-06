"""Retrieval service — user-scoped semantic search and the retrieval gate (Gate 1).

Gate 1 (retrieval gate): if the highest similarity among the retrieved chunks is
below the configured threshold, the request is refused WITHOUT ever calling the
LLM. No generation happens on insufficient evidence.

Every search is scoped by ``user_id`` inside the SQL statement itself (see
``ChunkRepository.similarity_search``), so no caller can retrieve another
tenant's chunks.
"""
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.providers.base import EmbeddingProvider
from app.repositories.chunk_repo import ChunkRepository

settings = get_settings()


@dataclass(frozen=True)
class RetrievedChunk:
    """A single chunk retrieved from the requesting user's own documents.

    ``citation_index`` is the 1-based index rendered as ``[n]`` in answers.
    ``user_id`` is carried through so the generation gate can re-verify
    ownership before any citation is rendered.
    """

    citation_index: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    user_id: uuid.UUID
    page_number: int
    text: str
    similarity: float


@dataclass(frozen=True)
class RetrievalResult:
    """Outcome of one retrieval round, including the Gate 1 verdict."""

    query: str
    chunks: tuple[RetrievedChunk, ...]
    threshold: float
    gate_passed: bool
    refusal_reason: str | None

    @property
    def top_similarity(self) -> float | None:
        """Highest similarity score in the result set, or None if nothing was retrieved."""
        return self.chunks[0].similarity if self.chunks else None

    @property
    def similarities(self) -> list[float]:
        """Similarity scores in descending rank order (top-1 first)."""
        return [chunk.similarity for chunk in self.chunks]


class RetrievalService:
    """Embeds the query and performs user-scoped vector search with an explicit Gate 1.

    The gate decision is intentionally plain and inspectable — it is core
    interview-defensible logic and must not hide behind abstractions.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        threshold: float | None = None,
        top_k: int | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.threshold = (
            threshold if threshold is not None else settings.RETRIEVAL_SIMILARITY_THRESHOLD
        )
        self.top_k = top_k if top_k is not None else settings.TOP_K_CHUNKS

    async def retrieve(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        query: str,
        document_ids: list[uuid.UUID] | None = None,
    ) -> RetrievalResult:
        """Run user-scoped semantic search and evaluate the retrieval gate.

        Args:
            db: Async database session.
            user_id: Authenticated user — applied inside the vector SQL query.
            query: Natural-language question.
            document_ids: Optional subset of documents to search ("chat over
                selected documents"); still user-scoped regardless.
        """
        query_embedding = await self.embedding_provider.embed_query(query)

        rows = await ChunkRepository.similarity_search(
            session=db,
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=self.top_k,
            document_ids=document_ids,
        )

        chunks = tuple(
            RetrievedChunk(
                citation_index=index + 1,
                chunk_id=row[0].id,
                document_id=row[0].document_id,
                user_id=row[0].user_id,
                page_number=row[0].page_number,
                text=row[0].text,
                similarity=row[1],
            )
            for index, row in enumerate(rows)
        )

        # ---- Gate 1: retrieval gate (explicit, no LLM involved) ----
        top_similarity = chunks[0].similarity if chunks else None

        if not chunks:
            gate_passed = False
            refusal_reason = (
                "no passages in your documents are related to this question, "
                "so there is no evidence to answer from"
            )
        elif top_similarity < self.threshold:
            gate_passed = False
            refusal_reason = (
                f"the most relevant passage scored {top_similarity:.2f} similarity, "
                f"below the required threshold of {self.threshold:.2f}, "
                "so the available evidence is too weak to support a reliable answer"
            )
        else:
            gate_passed = True
            refusal_reason = None

        return RetrievalResult(
            query=query,
            chunks=chunks,
            threshold=self.threshold,
            gate_passed=gate_passed,
            refusal_reason=refusal_reason,
        )
