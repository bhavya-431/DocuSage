import asyncio
import hashlib
import io
import logging
import random
import uuid
from typing import Any
import pdfplumber
import pymupdf
import pypdf
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import get_settings
from app.models.document import Document
from app.providers.base import EmbeddingProvider
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository

logger = logging.getLogger(__name__)
settings = get_settings()


class IngestionError(Exception):
    """Base exception for PDF ingestion pipeline errors."""
    pass


class IngestionService:
    """Service orchestrating PDF validation, text extraction, chunking, and embedding."""

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider

    @staticmethod
    def compute_sha256(content: bytes) -> str:
        """Compute SHA-256 hash of file binary for duplicate upload detection."""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def validate_pre_parse(
        file_bytes: bytes,
        filename: str,
        user_doc_count: int,
    ) -> None:
        """Run pre-parse validations (file size and quota caps)."""
        # 1. Size cap check (25 MB)
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise IngestionError(
                f"File size ({len(file_bytes) / (1024*1024):.1f} MB) exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB} MB."
            )

        # 2. Quota check (20 docs max)
        if user_doc_count >= settings.MAX_DOCS_PER_USER:
            raise IngestionError(
                f"Upload quota reached. Maximum {settings.MAX_DOCS_PER_USER} documents allowed per user."
            )

        # 3. Filename check
        if not filename.lower().endswith(".pdf"):
            raise IngestionError("Only PDF files are supported in v1.")

    @staticmethod
    def extract_text_with_metadata(file_bytes: bytes) -> list[dict[str, Any]]:
        """Extract text page-by-page preserving reading order and handling edge cases.

        Returns a list of dicts: [{"page_number": int, "text": str}]
        """
        # 1. Check for password encryption via pypdf first
        try:
            pypdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            if pypdf_reader.is_encrypted:
                raise IngestionError("Password-protected PDFs are not supported in v1.")
        except pypdf.errors.PdfReadError as e:
            if "encrypted" in str(e).lower() or "password" in str(e).lower():
                raise IngestionError("Password-protected PDFs are not supported in v1.")
            raise IngestionError(f"Corrupted PDF file: {str(e)}")
        except IngestionError:
            raise
        except Exception as e:
            logger.warning(f"pypdf pre-check encountered non-fatal error: {e}")

        # 2. Open via PyMuPDF (fitz)
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise IngestionError(f"Corrupted or invalid PDF format: {str(e)}")

        try:
            if doc.is_encrypted:
                raise IngestionError("Password-protected PDFs are not supported in v1.")

            page_count = len(doc)
            if page_count == 0:
                raise IngestionError("Document contains zero pages.")

            if page_count > settings.MAX_PAGES_PER_DOC:
                raise IngestionError(
                    f"Document has {page_count} pages, exceeding the limit of {settings.MAX_PAGES_PER_DOC} pages."
                )

            pages_content: list[dict[str, Any]] = []
            total_chars = 0

            for page_idx in range(page_count):
                page = doc[page_idx]
                page_num = page_idx + 1

                # Extract reading-order text blocks (PyMuPDF blocks)
                blocks = page.get_text("blocks")
                page_text_blocks = []
                for b in blocks:
                    # b: (x0, y0, x1, y1, text, block_no, block_type)
                    if b[6] == 0:  # Text block
                        text_val = b[4].strip()
                        if text_val:
                            page_text_blocks.append(text_val)

                page_text = "\n\n".join(page_text_blocks).strip()

                # Fallback to pdfplumber if PyMuPDF returned little or no text on this page
                if len(page_text) < 20:
                    try:
                        with pdfplumber.open(io.BytesIO(file_bytes)) as plumber_pdf:
                            if page_idx < len(plumber_pdf.pages):
                                plumber_text = plumber_pdf.pages[page_idx].extract_text()
                                if plumber_text and len(plumber_text.strip()) > len(page_text):
                                    page_text = plumber_text.strip()
                    except Exception:
                        pass

                total_chars += len(page_text)
                if page_text:
                    pages_content.append({"page_number": page_num, "text": page_text})

            # 3. Detect scanned / image-only PDFs without text layer
            if total_chars < 50:
                raise IngestionError(
                    "OCR not supported in v1: Scanned document without selectable text layer detected."
                )

            return pages_content
        finally:
            doc.close()

    @staticmethod
    def chunk_document(
        pages_content: list[dict[str, Any]],
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> list[dict[str, Any]]:
        """Split extracted pages into overlapping chunks retaining page-level metadata."""
        all_chunks: list[dict[str, Any]] = []
        global_chunk_idx = 0

        for page_data in pages_content:
            page_num = page_data["page_number"]
            page_text = page_data["text"]

            if not page_text:
                continue

            # Split by paragraphs / double newlines first, then assemble
            paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
            current_chunk = ""

            for para in paragraphs:
                if not current_chunk:
                    current_chunk = para
                elif len(current_chunk) + len(para) + 2 <= chunk_size:
                    current_chunk += "\n\n" + para
                else:
                    # Current chunk is full, record it
                    all_chunks.append({
                        "chunk_index": global_chunk_idx,
                        "page_number": page_num,
                        "text": current_chunk,
                        "char_count": len(current_chunk),
                    })
                    global_chunk_idx += 1

                    # Retain overlap from end of current chunk
                    if len(current_chunk) > chunk_overlap:
                        overlap_text = current_chunk[-chunk_overlap:]
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        current_chunk = para

            if current_chunk.strip():
                all_chunks.append({
                    "chunk_index": global_chunk_idx,
                    "page_number": page_num,
                    "text": current_chunk.strip(),
                    "char_count": len(current_chunk.strip()),
                })
                global_chunk_idx += 1

        return all_chunks

    async def embed_chunks_with_backoff(
        self, chunks: list[dict[str, Any]], batch_size: int = 16
    ) -> list[dict[str, Any]]:
        """Generate vector embeddings for all chunks with exponential backoff on 429 errors."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_texts = [c["text"] for c in batch]

            max_retries = 5
            attempt = 0
            while True:
                try:
                    embeddings = await self.embedding_provider.embed_texts(batch_texts)
                    for chunk_dict, emb in zip(batch, embeddings):
                        chunk_dict["embedding"] = emb
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    attempt += 1
                    # Detect rate limits (429) or quota exhaustion
                    if ("429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg) and attempt <= max_retries:
                        delay = (2 ** attempt) * 0.5 + random.uniform(0.1, 0.4)
                        logger.warning(
                            f"Encountered 429 rate limit during embedding batch. Backing off for {delay:.2f}s (attempt {attempt}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise e

        return chunks

    async def process_document(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        file_bytes: bytes,
    ) -> Document:
        """Execute full ingestion pipeline for an uploaded document."""
        try:
            # 1. Update status to processing
            await DocumentRepository.update_status(
                db, user_id=user_id, document_id=document_id, status="processing"
            )

            # 2. Extract text & pages
            pages_content = self.extract_text_with_metadata(file_bytes)
            page_count = max(p["page_number"] for p in pages_content) if pages_content else 0

            # 3. Chunk text with page metadata
            chunks_data = self.chunk_document(
                pages_content,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )

            if not chunks_data:
                raise IngestionError("Document produced no parseable text chunks.")

            # 4. Generate embeddings with backoff
            embedded_chunks = await self.embed_chunks_with_backoff(chunks_data)

            # 5. Store chunks in database scoped to user_id
            await ChunkRepository.create_chunks(
                db, user_id=user_id, document_id=document_id, chunks_data=embedded_chunks
            )

            # 6. Mark document as ready
            doc = await DocumentRepository.update_status(
                db,
                user_id=user_id,
                document_id=document_id,
                status="ready",
                page_count=page_count,
            )
            return doc  # type: ignore

        except Exception as e:
            logger.error(f"Ingestion failed for document {document_id}: {e}", exc_info=True)
            doc = await DocumentRepository.update_status(
                db,
                user_id=user_id,
                document_id=document_id,
                status="failed",
                error_message=str(e),
            )
            return doc  # type: ignore
