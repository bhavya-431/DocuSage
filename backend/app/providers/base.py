from typing import AsyncGenerator, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol interface for embedding generation providers."""

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality of this embedding provider."""
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of text chunks."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a search query string."""
        ...

    async def validate_credentials(self) -> tuple[bool, str]:
        """Validate credentials and configuration without making billable inference calls."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol interface for Large Language Model generation providers."""

    async def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a complete text response from the model."""
        ...

    async def stream_response(
        self, prompt: str, system_prompt: str = ""
    ) -> AsyncGenerator[str, None]:
        """Yield streaming response tokens/chunks from the model."""
        ...

    async def validate_credentials(self) -> tuple[bool, str]:
        """Validate credentials and configuration without making billable inference calls."""
        ...
