import math
import pytest
from httpx import ASGITransport, AsyncClient
from app.core.config import Settings
from app.main import app
from app.providers.base import EmbeddingProvider, LLMProvider
from app.providers.embeddings import (
    GeminiEmbeddingProvider,
    LocalMiniLMEmbeddingProvider,
    get_embedding_provider,
)
from app.providers.llm import (
    GeminiLLMProvider,
    GroqLLMProvider,
    OpenRouterLLMProvider,
    get_llm_provider,
)


def test_settings_load_and_defaults():
    settings = Settings()
    assert settings.APP_NAME == "DocuSage: Enterprise Document Intelligence Agent"
    assert settings.EMBEDDING_PROVIDER in ["local", "gemini"]
    # Calibrated by evals/dataset.json (Phase 6): refusals score 0.04-0.09,
    # answerable evidence 0.27+ for the default local provider.
    assert settings.RETRIEVAL_SIMILARITY_THRESHOLD == 0.20
    assert settings.MAX_DOCS_PER_USER == 20
    assert settings.MAX_FILE_SIZE_MB == 25
    assert settings.MAX_PAGES_PER_DOC == 150


def test_protocols_runtime_checkable():
    local_emb = LocalMiniLMEmbeddingProvider()
    assert isinstance(local_emb, EmbeddingProvider)

    gemini_emb = GeminiEmbeddingProvider(api_key="test_key")
    assert isinstance(gemini_emb, EmbeddingProvider)

    gemini_llm = GeminiLLMProvider(api_key="test_key")
    assert isinstance(gemini_llm, LLMProvider)

    groq_llm = GroqLLMProvider(api_key="gsk_test")
    assert isinstance(groq_llm, LLMProvider)

    openrouter_llm = OpenRouterLLMProvider(api_key="sk-or-test")
    assert isinstance(openrouter_llm, LLMProvider)


@pytest.mark.asyncio
async def test_local_minilm_real_embedding_generation():
    provider = LocalMiniLMEmbeddingProvider()
    assert provider.dimension == 384

    # Test credential check (zero external dependency)
    valid, msg = await provider.validate_credentials()
    assert valid is True
    assert "ready" in msg.lower()

    # Real inference test on local CPU
    texts = ["Enterprise Document Intelligence Agent", "Dual-gate abstention RAG"]
    embeddings = await provider.embed_texts(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
    assert all(isinstance(v, float) for v in embeddings[0])

    # Check vector normalization (L2 norm should be approximately 1.0)
    norm = math.sqrt(sum(x * x for x in embeddings[0]))
    assert pytest.approx(norm, rel=1e-3) == 1.0

    # Query embedding test
    query_emb = await provider.embed_query("Enterprise Document")
    assert len(query_emb) == 384


@pytest.mark.asyncio
async def test_credentials_validation_without_inference_calls():
    # 1. Missing Gemini key
    gemini_no_key = GeminiEmbeddingProvider(api_key="")
    valid, msg = await gemini_no_key.validate_credentials()
    assert valid is False
    assert "not configured" in msg

    # 2. Configured Gemini key
    gemini_with_key = GeminiEmbeddingProvider(api_key="AIzaSyDummyKeyForTestingPurposes12345")
    valid, msg = await gemini_with_key.validate_credentials()
    assert valid is True

    # 3. Groq validation
    groq_invalid = GroqLLMProvider(api_key="invalid_format_key")
    valid, msg = await groq_invalid.validate_credentials()
    assert valid is False
    assert "gsk_" in msg

    groq_valid = GroqLLMProvider(api_key="gsk_valid_looking_key_1234567890")
    valid, msg = await groq_valid.validate_credentials()
    assert valid is True

    # 4. OpenRouter validation
    or_no_key = OpenRouterLLMProvider(api_key="")
    valid, msg = await or_no_key.validate_credentials()
    assert valid is False


def test_provider_factories():
    local_settings = Settings(EMBEDDING_PROVIDER="local", LLM_PROVIDER="groq")
    emb = get_embedding_provider(local_settings)
    assert isinstance(emb, LocalMiniLMEmbeddingProvider)
    assert emb.dimension == 384

    llm = get_llm_provider(local_settings)
    assert isinstance(llm, GroqLLMProvider)

    gemini_settings = Settings(EMBEDDING_PROVIDER="gemini", LLM_PROVIDER="gemini")
    emb_gemini = get_embedding_provider(gemini_settings)
    assert isinstance(emb_gemini, GeminiEmbeddingProvider)
    assert emb_gemini.dimension == 768

    llm_gemini = get_llm_provider(gemini_settings)
    assert isinstance(llm_gemini, GeminiLLMProvider)


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "database" in data
        assert "providers" in data
        assert "embedding" in data["providers"]
        assert "llm" in data["providers"]
        assert data["providers"]["embedding"]["provider"] in ["local", "gemini"]
