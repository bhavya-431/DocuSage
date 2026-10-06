"""Phase 6 — eval golden set + quality gates (PRD Days 11–12)."""
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_user, get_db
from app.core.config import get_settings
from app.main import app
from app.models.chat import ChatSession
from app.repositories.chat_repo import ChatRepository
from app.repositories.chunk_repo import ChunkRepository
from app.services.retrieval import RetrievalService, evaluate_gate
from evals.corpus import build_corpus_chunks
from evals.runner import DEFAULT_DATASET_PATH, _normalise, run_eval

settings = get_settings()


def load_dataset() -> dict:
    return json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Golden-set shape (PRD §8 Days 11–12: 20–30 Q/A incl. 5 forced-refusal)
# ---------------------------------------------------------------------------
def test_dataset_meets_prd_shape():
    dataset = load_dataset()
    cases = dataset["cases"]

    assert 20 <= len(cases) <= 30, "PRD requires 20–30 Q/A cases"

    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"

    refusals = [c for c in cases if c["category"] == "refusal"]
    answerable = [c for c in cases if c["category"] == "answerable"]
    assert len(refusals) == 5, "PRD requires exactly 5 forced-refusal cases"
    assert len(answerable) >= 20

    for case in answerable:
        assert case["question"].strip(), f"{case['id']}: missing question"
        assert case["expect_phrase"], f"{case['id']}: missing expected phrase"
        assert case["document"], f"{case['id']}: missing source document"

    for case in refusals:
        assert case["document"] is None
        assert case["expect_phrase"] is None
        assert case.get("rationale"), f"{case['id']}: refusal needs a rationale"


def test_corpus_phrases_exist_in_ingested_chunks():
    """Every expect_phrase must survive the real ingestion pipeline — this
    catches dataset/corpus drift (line wrapping, extraction differences)."""
    chunks = build_corpus_chunks()
    assert len(chunks) >= 8, "fixture corpus should yield multiple chunks"
    assert len({c["document"] for c in chunks}) == 4, "all 4 fixture docs required"
    assert all(c["page_number"] >= 1 for c in chunks)

    normalised_corpus = [_normalise(c["text"]) for c in chunks]
    for case in load_dataset()["cases"]:
        if case["category"] != "answerable":
            continue
        phrase = _normalise(case["expect_phrase"])
        assert any(phrase in text for text in normalised_corpus), (
            f"{case['id']}: phrase {case['expect_phrase']!r} not found in any "
            "ingested chunk — dataset and corpus have drifted apart"
        )


# ---------------------------------------------------------------------------
# Full eval against production defaults (real local embeddings, no LLM)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_eval_meets_quality_bars_at_production_threshold():
    report = await run_eval()  # production threshold/top-k, local MiniLM

    assert len(report.results) == 25

    # SAFETY: a forced-refusal case must NEVER pass the retrieval gate.
    assert report.false_answer_count == 0, (
        "forced-refusal cases failed to refuse: "
        + ", ".join(r.id for r in report.refusals if r.gate_passed)
    )
    for r in report.refusals:
        assert r.gate_passed is False
        assert r.confidence == "Refused"
        assert r.passed is True  # passed == (not gate_passed) for refusals

    # Coverage: answerable cases answered with evidence in top-k.
    assert report.false_refusal_count == 0, (
        "false refusals: "
        + ", ".join(r.id for r in report.answerable if not r.gate_passed)
    )
    assert report.coverage_rate >= 0.95, (
        f"coverage {report.coverage_rate:.0%} below the 95% bar; failures: "
        + "; ".join(r.detail for r in report.answerable if not r.passed)
    )

    # Rank quality: the expected fact should usually be the top-1 chunk.
    assert report.top1_relevance_rate >= 0.90

    # Sanity: mean top-1 clears the gate threshold (separation exists).
    assert report.mean_top_similarity > report.threshold

    # Answerable cases that answered carry a real confidence label.
    for r in report.answerable:
        if r.gate_passed:
            assert r.confidence in ("High", "Medium", "Low")
            assert r.top_similarity is not None

    # The summary renders (used by `python -m evals.runner`).
    summary = report.summary()
    assert "false answers" in summary and "coverage" in summary


# ---------------------------------------------------------------------------
# Shared gate logic — eval measures the production gate, not a copy
# ---------------------------------------------------------------------------
def test_evaluate_gate_boundaries_and_messages():
    # Nothing retrieved
    passed, reason = evaluate_gate([], settings.RETRIEVAL_SIMILARITY_THRESHOLD)
    assert passed is False
    assert "no passages" in reason

    # Below threshold: reason explains the scores honestly
    passed, reason = evaluate_gate([0.08], settings.RETRIEVAL_SIMILARITY_THRESHOLD)
    assert passed is False
    assert "0.08" in reason and "0.20" in reason

    # Exactly at threshold → passes (>= is a pass)
    passed, reason = evaluate_gate([0.20], settings.RETRIEVAL_SIMILARITY_THRESHOLD)
    assert passed is True
    assert reason is None


@pytest.mark.asyncio
async def test_eval_gate_identical_to_production_retrieval_gate():
    """The eval's gate decision must equal RetrievalService.retrieve's output."""
    user_id = uuid.uuid4()
    rows = [
        (
            SimpleNamespace(
                id=uuid.uuid4(), document_id=uuid.uuid4(), user_id=user_id,
                page_number=1, text="evidence",
            ),
            0.5,
        ),
        (
            SimpleNamespace(
                id=uuid.uuid4(), document_id=uuid.uuid4(), user_id=user_id,
                page_number=2, text="weak",
            ),
            0.1,
        ),
    ]

    provider = AsyncMock()
    provider.embed_query.return_value = [0.1] * 384

    with patch.object(ChunkRepository, "similarity_search", return_value=rows):
        result = await RetrievalService(provider).retrieve(
            db=AsyncMock(), user_id=user_id, query="q"
        )

    eval_passed, eval_reason = evaluate_gate(
        [0.5, 0.1], settings.RETRIEVAL_SIMILARITY_THRESHOLD
    )
    assert result.gate_passed is eval_passed
    assert result.refusal_reason == eval_reason


# ---------------------------------------------------------------------------
# Error polish: blank questions rejected before touching retrieval
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ask_rejects_whitespace_only_question():
    user_id = uuid.uuid4()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    app.dependency_overrides[get_db] = lambda: AsyncMock()

    with patch.object(
        ChatRepository,
        "get_session",
        return_value=ChatSession(id=uuid.uuid4(), user_id=user_id),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/chat/sessions/00000000-0000-0000-0000-000000000001/messages",
                json={"question": "   \n\t "},
            )
    assert resp.status_code == 422
    assert "blank" in str(resp.json()).lower()

    # A padded-but-real question still passes validation (reaches ownership check)
    with patch.object(ChatRepository, "get_session", return_value=None):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/chat/sessions/00000000-0000-0000-0000-000000000001/messages",
                json={"question": "  real question  "},
            )
    assert resp.status_code == 404  # past validation, stopped at ownership

    app.dependency_overrides.clear()
