import uuid
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_current_user, get_db
from app.core.config import get_settings
from app.core.database import async_session_factory
from app.models.user import User
from app.providers.embeddings import get_embedding_provider
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.services.ingestion import IngestionError, IngestionService

router = APIRouter(prefix="/api/documents", tags=["Documents"])
settings = get_settings()


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_size_bytes: int
    page_count: int
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentDetailResponse(DocumentResponse):
    chunk_count: int = 0


async def _run_background_ingestion(
    user_id: uuid.UUID,
    document_id: uuid.UUID,
    file_bytes: bytes,
) -> None:
    """Async task running ingestion in the background with dedicated DB session."""
    provider = get_embedding_provider()
    service = IngestionService(provider)
    async with async_session_factory() as session:
        try:
            await service.process_document(
                db=session,
                user_id=user_id,
                document_id=document_id,
                file_bytes=file_bytes,
            )
            await session.commit()
        except Exception:
            await session.rollback()


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Upload PDF document, validate edge cases and quotas, and trigger async ingestion."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported in v1.",
        )

    # Read binary bytes
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # 1. Quota check (max 20 docs per user)
    user_doc_count = await DocumentRepository.count_by_user(db, current_user.id)
    if user_doc_count >= settings.MAX_DOCS_PER_USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Upload quota reached. Maximum {settings.MAX_DOCS_PER_USER} documents allowed per user.",
        )

    # 2. File size check (25 MB cap)
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB} MB.",
        )

    # 3. Duplicate check via SHA-256 hash scoped to current user
    file_hash = IngestionService.compute_sha256(file_bytes)
    existing_doc = await DocumentRepository.get_by_hash(db, current_user.id, file_hash)
    if existing_doc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Duplicate upload: This exact PDF document has already been uploaded to your account.",
        )

    # Create initial document record with status "uploaded"
    doc = await DocumentRepository.create_document(
        session=db,
        user_id=current_user.id,
        filename=file.filename,
        file_hash=file_hash,
        file_size_bytes=len(file_bytes),
        page_count=0,
        status="uploaded",
    )
    await db.commit()

    # Enqueue background ingestion pipeline
    background_tasks.add_task(
        _run_background_ingestion,
        user_id=current_user.id,
        document_id=doc.id,
        file_bytes=file_bytes,
    )

    return DocumentResponse.model_validate(doc)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[DocumentResponse]:
    """List all documents belonging to the authenticated user."""
    docs = await DocumentRepository.list_documents(db, current_user.id)
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    """Get single document metadata and chunk count strictly scoped to current user."""
    doc = await DocumentRepository.get_by_id(db, current_user.id, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or access denied.",
        )

    chunk_count = await ChunkRepository.count_by_document(db, current_user.id, document_id)
    detail = DocumentDetailResponse.model_validate(doc)
    detail.chunk_count = chunk_count
    return detail


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a document and cascade delete all its chunks, strictly scoped to current user."""
    deleted = await DocumentRepository.delete_document(db, current_user.id, document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or access denied.",
        )
    return {"deleted": True, "id": str(document_id)}
