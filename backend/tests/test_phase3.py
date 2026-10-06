import io
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pymupdf
import pytest
from httpx import ASGITransport, AsyncClient
from app.api.deps import get_current_user, get_db
from app.core.config import get_settings
from app.main import app
from app.models.document import Document
from app.models.user import User
from app.repositories.document_repo import DocumentRepository
from app.services.ingestion import IngestionError, IngestionService

settings = get_settings()


def create_sample_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to generate an in-memory PDF with provided text on each page."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_encrypted_pdf_bytes(text: str, password: str = "secret123") -> bytes:
    """Helper to generate an encrypted PDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text, fontsize=11)
    # Save with encryption
    pdf_bytes = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw=password,
        owner_pw=password,
    )
    doc.close()
    return pdf_bytes


def test_sha256_hash_computation():
    data = b"DocuSage Enterprise RAG Test Data"
    hash1 = IngestionService.compute_sha256(data)
    hash2 = IngestionService.compute_sha256(data)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_pre_parse_validations():
    # 1. Non-pdf extension
    with pytest.raises(IngestionError, match="Only PDF"):
        IngestionService.validate_pre_parse(b"dummy", "report.docx", 5)

    # 2. Oversized file (>25 MB)
    oversized = b"x" * (26 * 1024 * 1024)
    with pytest.raises(IngestionError, match="exceeds maximum"):
        IngestionService.validate_pre_parse(oversized, "huge.pdf", 5)

    # 3. Quota reached (>=20)
    with pytest.raises(IngestionError, match="Upload quota reached"):
        IngestionService.validate_pre_parse(b"valid", "doc.pdf", 20)


def test_corrupted_pdf_handling():
    corrupted_bytes = b"%PDF-1.4\nCORRUPTED_GARBAGE_DATA_1234567890\n%%EOF"
    with pytest.raises(IngestionError, match="(?i)corrupted.*pdf"):
        IngestionService.extract_text_with_metadata(corrupted_bytes)


def test_scanned_pdf_no_text_layer():
    # Create empty PDF page with zero text characters
    empty_pdf_bytes = create_sample_pdf_bytes(["", ""])
    with pytest.raises(IngestionError, match="OCR not supported in v1"):
        IngestionService.extract_text_with_metadata(empty_pdf_bytes)


def test_password_protected_pdf_detection():
    encrypted_bytes = create_encrypted_pdf_bytes("Secret classified contents")
    with pytest.raises(IngestionError, match="Password-protected PDFs are not supported"):
        IngestionService.extract_text_with_metadata(encrypted_bytes)


def test_valid_pdf_extraction_and_chunking():
    page1 = (
        "DocuSage is an enterprise document intelligence agent designed for portfolio-grade RAG.\n\n"
        "It features strict user isolation and provider-agnostic abstractions."
    )
    page2 = (
        "Dual-gate abstention prevents hallucinations by refusing without an LLM call when evidence is insufficient.\n\n"
        "Inline page-level citations are verified against retrieved chunks."
    )
    pdf_bytes = create_sample_pdf_bytes([page1, page2])

    pages_content = IngestionService.extract_text_with_metadata(pdf_bytes)
    assert len(pages_content) == 2
    assert pages_content[0]["page_number"] == 1
    assert "DocuSage" in pages_content[0]["text"]
    assert pages_content[1]["page_number"] == 2
    assert "Dual-gate abstention" in pages_content[1]["text"]

    chunks = IngestionService.chunk_document(pages_content, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 2
    assert all("chunk_index" in c for c in chunks)
    assert all("page_number" in c for c in chunks)
    assert all("text" in c for c in chunks)
    assert all("char_count" in c for c in chunks)


@pytest.mark.asyncio
async def test_embedding_backoff_on_429():
    mock_provider = AsyncMock()
    # Fail first call with 429, succeed on second call
    call_count = 0

    async def mock_embed(texts):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("HTTP 429 Too Many Requests: Rate limit exceeded")
        return [[0.1] * 384 for _ in texts]

    mock_provider.embed_texts.side_effect = mock_embed

    service = IngestionService(mock_provider)
    chunks = [{"chunk_index": 0, "page_number": 1, "text": "Sample chunk", "char_count": 12}]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        embedded_chunks = await service.embed_chunks_with_backoff(chunks, batch_size=1)
        assert mock_sleep.called
        assert len(embedded_chunks[0]["embedding"]) == 384
        assert call_count == 2


@pytest.mark.asyncio
async def test_upload_api_duplicate_and_quota_rejections():
    test_user = User(
        id=uuid.uuid4(),
        email="tenant@example.com",
        hashed_password="hash",
    )
    mock_db = AsyncMock()

    app.dependency_overrides[get_current_user] = lambda: test_user
    app.dependency_overrides[get_db] = lambda: mock_db

    pdf_content = create_sample_pdf_bytes(["Valid test PDF page 1"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Test duplicate upload detection (returns 409)
        existing_doc = Document(
            id=uuid.uuid4(),
            user_id=test_user.id,
            filename="test.pdf",
            file_hash=IngestionService.compute_sha256(pdf_content),
            file_size_bytes=len(pdf_content),
            status="ready",
        )
        with patch.object(DocumentRepository, "count_by_user", return_value=1), \
             patch.object(DocumentRepository, "get_by_hash", return_value=existing_doc):

            resp = await client.post(
                "/api/documents/upload",
                files={"file": ("test.pdf", pdf_content, "application/pdf")},
            )
            assert resp.status_code == 409
            assert "duplicate" in resp.json()["detail"].lower()

        # 2. Test quota limit reached (returns 400)
        with patch.object(DocumentRepository, "count_by_user", return_value=20):
            resp = await client.post(
                "/api/documents/upload",
                files={"file": ("test_limit.pdf", pdf_content, "application/pdf")},
            )
            assert resp.status_code == 400
            assert "quota reached" in resp.json()["detail"].lower()

        # 3. Test non-PDF file rejection (returns 400)
        resp = await client.post(
            "/api/documents/upload",
            files={"file": ("malicious.exe", b"binary", "application/octet-stream")},
        )
        assert resp.status_code == 400

    app.dependency_overrides.clear()
