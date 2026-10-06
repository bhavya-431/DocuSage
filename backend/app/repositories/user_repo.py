import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User


class UserRepository:
    """Repository managing user account database operations."""

    @staticmethod
    async def create_user(
        session: AsyncSession, email: str, hashed_password: str
    ) -> User:
        user = User(email=email.lower().strip(), hashed_password=hashed_password)
        session.add(user)
        await session.flush()
        return user

    @staticmethod
    async def get_by_email(session: AsyncSession, email: str) -> User | None:
        stmt = select(User).where(User.email == email.lower().strip())
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_id(session: AsyncSession, user_id: uuid.UUID) -> User | None:
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
