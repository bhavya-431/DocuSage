import asyncio
from typing import Any
import httpx
from app.core.config import Settings, get_settings
from app.providers.base import EmbeddingProvider


class LocalMiniLMEmbeddingProvider:
    """Zero-cost local embedding provider using sentence-transformers (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self._model_name = model_name
        self._dimension = dimension
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        # Run synchronous model.encode in executor to prevent blocking the async loop
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._get_model().encode(texts, normalize_embeddings=True).tolist(),
        )
        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0] if results else [0.0] * self._dimension

    async def validate_credentials(self) -> tuple[bool, str]:
        """Local provider requires no network credentials."""
        return True, f"Local provider ready ({self._model_name}, dim={self._dimension})"


class GeminiEmbeddingProvider:
    """Google Gemini embedding provider using text-embedding-004 (768 dimensions)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "text-embedding-004",
        dimension: int = 768,
    ):
        settings = get_settings()
        self._api_key = api_key or settings.GEMINI_API_KEY
        self._model_name = model_name
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def validate_credentials(self) -> tuple[bool, str]:
        """Validate Gemini API key exists and is non-empty without making inference calls."""
        if not self._api_key:
            return False, "GEMINI_API_KEY is not configured"
        if len(self._api_key.strip()) < 10:
            return False, "GEMINI_API_KEY appears invalid (too short)"
        return True, f"Gemini credentials configured for {self._model_name}"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:batchEmbedContents?key={self._api_key}"
        requests_payload = [
            {
                "model": f"models/{self._model_name}",
                "content": {"parts": [{"text": text}]},
            }
            for text in texts
        ]

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json={"requests": requests_payload})
            response.raise_for_status()
            data = response.json()
            return [item["values"] for item in data.get("embeddings", [])]

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0] if results else [0.0] * self._dimension


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Factory to retrieve configured embedding provider instance."""
    settings = settings or get_settings()
    if settings.EMBEDDING_PROVIDER == "gemini":
        return GeminiEmbeddingProvider(
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.EMBEDDING_MODEL or "text-embedding-004",
            dimension=settings.EMBEDDING_DIMENSION if settings.EMBEDDING_DIMENSION != 384 else 768,
        )
    return LocalMiniLMEmbeddingProvider(
        model_name=settings.EMBEDDING_MODEL or "all-MiniLM-L6-v2",
        dimension=settings.EMBEDDING_DIMENSION,
    )
