import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage, ChatSession


class ChatRepository:
    """Repository strictly enforcing tenant user_id isolation on all chat operations."""

    @staticmethod
    async def create_session(
        session: AsyncSession, user_id: uuid.UUID, title: str = "New chat"
    ) -> ChatSession:
        """Create a chat session owned by user_id."""
        chat_session = ChatSession(user_id=user_id, title=title)
        session.add(chat_session)
        await session.flush()
        return chat_session

    @staticmethod
    async def list_sessions(
        session: AsyncSession, user_id: uuid.UUID
    ) -> list[ChatSession]:
        """List all chat sessions belonging to user_id, newest first."""
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_session(
        session: AsyncSession, user_id: uuid.UUID, session_id: uuid.UUID
    ) -> ChatSession | None:
        """Fetch a chat session by ID, strictly scoped to user_id."""
        stmt = select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def update_title(
        session: AsyncSession,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        title: str,
    ) -> bool:
        """Rename a session, strictly scoped to user_id. Returns True if updated."""
        stmt = (
            update(ChatSession)
            .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .values(title=title)
        )
        result = await session.execute(stmt)
        return result.rowcount > 0

    @staticmethod
    async def create_message(
        session: AsyncSession,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        *,
        question: str,
        answer: str,
        document_ids: list[str],
        citations: list[dict],
        refused: bool,
        refusal_gate: str | None,
        confidence_label: str,
        confidence_score: float,
    ) -> ChatMessage:
        """Persist one validated question/answer exchange for user_id."""
        message = ChatMessage(
            user_id=user_id,
            session_id=session_id,
            question=question,
            answer=answer,
            document_ids=document_ids,
            citations=citations,
            refused=refused,
            refusal_gate=refusal_gate,
            confidence_label=confidence_label,
            confidence_score=confidence_score,
        )
        session.add(message)
        await session.flush()
        return message

    @staticmethod
    async def list_messages(
        session: AsyncSession, user_id: uuid.UUID, session_id: uuid.UUID
    ) -> list[ChatMessage]:
        """List messages of a session, strictly scoped to user_id, oldest first."""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.user_id == user_id, ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())
