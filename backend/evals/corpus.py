"""Deterministic fixture corpus for the DocuSage eval golden set.

Four small, digitally-generated PDFs (PRD §10: demo corpus = digital PDFs,
< 150 pages). Documents run through the REAL ingestion pipeline — PyMuPDF
extraction, pdfplumber fallback, chunking with page metadata — so the eval
measures production behaviour rather than a reimplementation.

Each paragraph stays well under the chunk size, so a paragraph always lives
inside a single chunk; `expect_phrase` values in dataset.json are substrings
of exactly one paragraph.
"""
import io

import pymupdf

from app.core.config import get_settings
from app.services.ingestion import IngestionService

settings = get_settings()

# filename -> list of pages; each page is a list of paragraphs.
CORPUS: dict[str, list[list[str]]] = {
    "employee_handbook.pdf": [
        [
            "Northwind Employee Handbook — Time Off. Full-time employees receive "
            "25 days of paid time off per year, in addition to 10 public holidays. "
            "PTO accrues at a rate of 2.08 days per month of service.",
            "Unused PTO may be carried over into the next calendar year up to a "
            "maximum of 5 days. Requests for PTO longer than one week require "
            "approval from your line manager at least 30 days in advance.",
        ],
        [
            "Northwind Employee Handbook — Remote Work. Employees may work remotely "
            "up to three days per week, with at least two days per week on site. "
            "A home office equipment allowance of 500 euros is provided once every "
            "three years.",
            "Northwind Employee Handbook — Code of Conduct. The code of conduct "
            "prohibits harassment and requires employees to disclose any conflict "
            "of interest within 14 days of becoming aware of it. Business travel "
            "expenses must be submitted within 30 days of return.",
        ],
    ],
    "security_policy.pdf": [
        [
            "Acme Security Policy — Access Control. All accounts must use passwords "
            "of at least 14 characters with two factors of authentication enabled "
            "through the company authenticator app. Passwords must be changed every "
            "12 months, or immediately after any suspected compromise.",
            "Access to production systems requires just-in-time approval and all "
            "sessions are logged for 400 days for audit purposes.",
        ],
        [
            "Acme Security Policy — Incident Reporting. Suspected security incidents "
            "must be reported to the security operations centre within one hour of "
            "discovery using the dedicated sec-incident channel.",
            "Acme Security Policy — Data Classification. Data is classified as "
            "public, internal, confidential, or restricted. Customer personal data "
            "is classified as restricted and may only be stored in approved systems. "
            "Lost devices must be reported within 2 hours to enable remote wipe.",
        ],
    ],
    "benefits_guide.pdf": [
        [
            "Northwind Benefits Guide — Health and Retirement. The company provides "
            "private health insurance covering the employee and dependants, with "
            "cover effective from the first day of employment.",
            "The retirement plan includes an employer match of 6 percent of base "
            "salary, vesting after three years of service. Annual eye and dental "
            "check-ups are reimbursed up to 200 euros per year.",
        ],
        [
            "Northwind Benefits Guide — Leave. Parental leave provides 20 weeks of "
            "fully paid leave for primary caregivers and 6 weeks of fully paid leave "
            "for secondary caregivers.",
            "Northwind Benefits Guide — Learning and Wellness. A learning budget of "
            "1500 euros per employee per year covers courses, books, and conferences. "
            "The wellness stipend of 40 euros per month can be used for gym "
            "memberships or sport activities.",
        ],
    ],
    "platform_runbook.pdf": [
        [
            "Platform Runbook — Rate Limits. The public API enforces a rate limit of "
            "600 requests per minute per organisation, returning HTTP status 429 "
            "with a Retry-After header when exceeded.",
            "Clients must use exponential backoff with jitter for retries, capped at "
            "5 attempts. The maximum request payload size is 25 megabytes.",
        ],
        [
            "Platform Runbook — Webhooks. Webhook deliveries are retried for up to "
            "24 hours with increasing intervals, and endpoints must respond within "
            "10 seconds or the delivery is considered failed.",
            "Platform Runbook — Deploys. Deployments to production require two "
            "approving reviews and are rolled out progressively to 10 percent, "
            "50 percent, then 100 percent of traffic. Database migrations must be "
            "backward compatible with the previous two releases.",
        ],
    ],
}

_LINE_WIDTH = 90  # keep rendered lines inside the PDF page box


def _wrap(text: str, width: int = _LINE_WIDTH) -> str:
    """Hard-wrap a paragraph so inserted text stays within the page mediabox
    (PyMuPDF does not auto-wrap, and off-page text can be dropped on extract)."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def render_pdf(pages: list[list[str]]) -> bytes:
    """Render one fixture document to PDF bytes."""
    doc = pymupdf.open()
    for paragraphs in pages:
        page = doc.new_page()
        text = "\n\n".join(_wrap(p) for p in paragraphs)
        page.insert_text((50, 72), text, fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def build_corpus_chunks() -> list[dict]:
    """Run every fixture PDF through the production ingestion pipeline.

    Returns chunk dicts (chunk_index, page_number, text, char_count) plus the
    source ``document`` filename.
    """
    all_chunks: list[dict] = []
    for filename, pages in CORPUS.items():
        pdf_bytes = render_pdf(pages)
        pages_content = IngestionService.extract_text_with_metadata(pdf_bytes)
        chunks = IngestionService.chunk_document(
            pages_content,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        for chunk in chunks:
            all_chunks.append({**chunk, "document": filename})
    return all_chunks
