"""Phase 7 — demo seeding script tests (PRD Days 13–14: seed 3–4 demo PDFs)."""
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.repositories.document_repo import DocumentRepository
from app.repositories.user_repo import UserRepository
from app.services.ingestion import IngestionService
from scripts.seed_demo import collect_demo_pdfs, seed


def _mock_db() -> MagicMock:
    db = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


def test_collect_demo_pdfs_yields_ingestible_document_set():
    """3–4 demo PDFs, each passing the real pre-parse validations and extraction."""
    pdfs = collect_demo_pdfs()
    assert 3 <= len(pdfs) <= 4, "PRD asks to seed 3–4 demo PDFs"

    for filename, data in pdfs:
        assert filename.endswith(".pdf")
        assert data.startswith(b"%PDF")
        # Raises IngestionError on any failure: size, quota, extension checks
        IngestionService.validate_pre_parse(data, filename, user_doc_count=0)
        pages = IngestionService.extract_text_with_metadata(data)
        assert len(pages) >= 2, f"{filename}: expected multi-page fixture"
        assert sum(len(p["text"]) for p in pages) > 100
        chunks = IngestionService.chunk_document(pages)
        assert len(chunks) >= 1, f"{filename}: chunking produced nothing"


@pytest.mark.asyncio
async def test_seed_is_idempotent_and_never_reingests_existing_documents():
    pdfs = collect_demo_pdfs()
    db = _mock_db()
    user = SimpleNamespace(id=uuid.uuid4(), email="demo@docusage.local")

    with patch.object(UserRepository, "get_by_email", return_value=user), \
         patch.object(
             DocumentRepository, "get_by_hash",
             return_value=SimpleNamespace(id=uuid.uuid4()),
         ) as mock_lookup, \
         patch.object(
             IngestionService, "process_document", new_callable=AsyncMock
         ) as mock_process:
        report = await seed(db, email="demo@docusage.local", password="x")

    assert len(report.skipped) == len(pdfs)
    assert report.created == [] and report.failed == []
    assert mock_lookup.call_count == len(pdfs)
    mock_process.assert_not_called()  # re-runs never re-embed


@pytest.mark.asyncio
async def test_seed_creates_user_and_ingests_every_document():
    pdfs = collect_demo_pdfs()
    db = _mock_db()
    new_user = SimpleNamespace(id=uuid.uuid4(), email="new@demo.local")

    with patch.object(UserRepository, "get_by_email", return_value=None), \
         patch.object(
             UserRepository, "create_user", return_value=new_user
         ) as mock_create_user, \
         patch.object(DocumentRepository, "get_by_hash", return_value=None), \
         patch.object(
             DocumentRepository, "create_document",
             side_effect=lambda **kwargs: SimpleNamespace(
                 id=uuid.uuid4(), status="uploaded"
             ),
         ) as mock_create_doc, \
         patch.object(
             IngestionService, "process_document", new_callable=AsyncMock,
             return_value=SimpleNamespace(status="ready", error_message=None),
         ) as mock_process:
        report = await seed(db)

    mock_create_user.assert_awaited_once()
    assert mock_create_doc.call_count == len(pdfs)
    assert mock_process.call_count == len(pdfs)
    assert len(report.created) == len(pdfs)
    assert report.skipped == [] and report.failed == []


@pytest.mark.asyncio
async def test_seed_reports_failed_documents_without_crashing_the_run():
    pdfs = collect_demo_pdfs()
    db = _mock_db()
    user = SimpleNamespace(id=uuid.uuid4(), email="demo@docusage.local")

    with patch.object(UserRepository, "get_by_email", return_value=user), \
         patch.object(DocumentRepository, "get_by_hash", return_value=None), \
         patch.object(
             DocumentRepository, "create_document",
             side_effect=lambda **kwargs: SimpleNamespace(
                 id=uuid.uuid4(), status="uploaded"
             ),
         ), \
         patch.object(
             IngestionService, "process_document", new_callable=AsyncMock,
             return_value=SimpleNamespace(
                 status="failed", error_message="Corrupted PDF file: test"
             ),
         ):
        report = await seed(db)

    # One failing document is reported, the rest still process
    assert len(report.failed) == len(pdfs)
    assert "Corrupted PDF file" in report.failed[0]
    assert report.created == []
    # The summary surfaces the failure reason for the operator
    assert "Corrupted PDF file" in report.summary()
