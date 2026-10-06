from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings driven by environment variables."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    APP_NAME: str = "DocuSage: Enterprise Document Intelligence Agent"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Database & Storage
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://docusage:docusage_secret@localhost:5432/docusage_db",
        description="Async PostgreSQL connection string with pgvector support",
    )

    # Security & Authentication
    JWT_SECRET_KEY: str = Field(
        default="dev-secret-key-change-in-production-must-be-at-least-32-chars",
        description="Secret key for signing JWT tokens",
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Provider Selection
    EMBEDDING_PROVIDER: Literal["local", "gemini"] = Field(
        default="local",
        description="Embedding provider to use: 'local' (MiniLM) or 'gemini'",
    )
    LLM_PROVIDER: Literal["gemini", "groq", "openrouter"] = Field(
        default="gemini",
        description="LLM provider to use: 'gemini', 'groq', or 'openrouter'",
    )

    # Provider Credentials
    GEMINI_API_KEY: str | None = Field(default=None, description="Google Gemini API key")
    GROQ_API_KEY: str | None = Field(default=None, description="Groq API key")
    OPENROUTER_API_KEY: str | None = Field(default=None, description="OpenRouter API key")

    # Provider Models & Vector Dimensions
    LLM_MODEL: str = Field(
        default="gemini-1.5-flash",
        description="Model name for generation (e.g., gemini-1.5-flash, llama-3.3-70b-versatile)",
    )
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name for embeddings (e.g., all-MiniLM-L6-v2, text-embedding-004)",
    )
    EMBEDDING_DIMENSION: int = Field(
        default=384,
        description="Vector dimension matching EMBEDDING_MODEL (384 for MiniLM, 768 for Gemini)",
    )

    # Ingestion & Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    MAX_DOCS_PER_USER: int = 20
    MAX_FILE_SIZE_MB: int = 25
    MAX_PAGES_PER_DOC: int = 150

    # Retrieval & Dual-Gate Abstention Parameters
    RETRIEVAL_SIMILARITY_THRESHOLD: float = Field(
        default=0.65,
        description="Cosine similarity threshold for Gate 1 abstention (refusal)",
    )
    TOP_K_CHUNKS: int = 5

    # Confidence Heuristic Bands (documented as a heuristic, NOT a calibrated probability)
    CONFIDENCE_HIGH_THRESHOLD: float = Field(
        default=0.80,
        description="Composite similarity at or above which confidence is labelled High",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
