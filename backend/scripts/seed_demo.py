"""Seed the demo corpus (PRD Days 13–14: seed 3–4 demo PDFs).

Ingests the four deterministic fixture PDFs (shared with the eval golden set)
through the **real** ingestion pipeline — validation, extraction, chunking,
embedding, pgvector storage — for a demo account.

Usage (from backend/, Postgres reachable):

    python -m scripts.seed_demo

Configuration (env, all optional):
    SEED_EMAIL      demo account email   (default: demo@docusage.local)
    SEED_PASSWORD   demo account password (default: docusage-demo — local only!)

Idempotent: documents whose SHA-256 already exists for the demo user are
skipped, so re-running is always safe. A failed document is reported, never
crashes the run.
"""
import asyncio
import os
from dataclasses import dataclass, field

from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import async_session_factory, engine
from app.core.security import hash_password
from app.models import Base  # noqa: F401 — registers all tables
from app.providers.embeddings import get_embedding_provider
from app.repositories.document_repo import DocumentRepository
from app.repositories.user_repo import UserRepository
from app.services.ingestion import IngestionService
from evals.corpus import CORPUS, render_pdf

settings = get_settings()

DEFAULT_EMAIL = os.environ.get("SEED_EMAIL", "demo@docusage.local")
DEFAULT_PASSWORD = os.environ.get("SEED_PASSWORD", "docusage-demo")


@dataclass
class SeedReport:
    created: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Seed complete — created: {len(self.created)}, "
            f"skipped (already present): {len(self.skipped)}, "
            f"failed: {len(self.failed)}"
            + (f" → {self.failed}" if self.failed else "")
        )


def collect_demo_pdfs() -> list[tuple[str, bytes]]:
    """Render the 3–4 fixture documents to PDF bytes (sync, side-effect free)."""
    return [(filename, render_pdf(pages)) for filename, pages in CORPUS.items()]


async def seed(db, email: str = DEFAULT_EMAIL, password: str = DEFAULT_PASSWORD) -> SeedReport:
    """Create the demo user (if needed) and ingest any missing demo PDFs.

    Args:
        db: async session (commit handled per document).
        email/password: demo credentials; existing account is reused as-is.
    """
    report = SeedReport()
    service = IngestionService(get_embedding_provider())

    user = await UserRepository.get_by_email(db, email)
    if user is None:
        user = await UserRepository.create_user(db, email, hash_password(password))
        await db.commit()
        print(f"Created demo user: {email}")
    else:
        print(f"Reusing existing user: {email}")

    for filename, data in collect_demo_pdfs():
        file_hash = IngestionService.compute_sha256(data)
        existing = await DocumentRepository.get_by_hash(db, user.id, file_hash)
        if existing is not None:
            report.skipped.append(filename)
            continue

        try:
            doc = await DocumentRepository.create_document(
                session=db,
                user_id=user.id,
                filename=filename,
                file_hash=file_hash,
                file_size_bytes=len(data),
                page_count=0,
                status="uploaded",
            )
            await db.commit()

            result = await service.process_document(
                db=db,
                user_id=user.id,
                document_id=doc.id,
                file_bytes=data,
            )
            await db.commit()

            if result is not None and result.status == "failed":
                report.failed.append(f"{filename}: {result.error_message}")
            else:
                report.created.append(filename)
        except Exception as exc:  # one bad doc must not abort the whole run
            await db.rollback()
            report.failed.append(f"{filename}: {exc}")

    return report


async def _amain() -> None:
    # Standalone script: bootstrap schema the same way the app lifespan does.
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        raise SystemExit(
            f"Cannot reach PostgreSQL at the configured DATABASE_URL: {exc}\n"
            "Start the local DB first: docker compose up -d"
        ) from exc

    async with async_session_factory() as session:
        report = await seed(session)
    print(report.summary())
    if report.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(_amain())
