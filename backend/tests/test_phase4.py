import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import get_settings
from app.repositories.chunk_repo import ChunkRepository
from app.services.generation import (
    ConfidenceLabel,
    GenerationService,
    compute_confidence,
    validate_citations,
)
from app.services.retrieval import RetrievalService

settings = get_settings()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_chunk_row(
    user_id: uuid.UUID,
    text: str = "DocuSage enforces user isolation at the repository layer.",
    page_number: int = 1,
) -> SimpleNamespace:
    """Fake DocumentChunk-like row as returned by the ORM."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        user_id=user_id,
        page_number=page_number,
        text=text,
        char_count=len(text),
        chunk_index=0,
    )


def fake_embedding_provider() -> AsyncMock:
    provider = AsyncMock()
    provider.embed_query.return_value = [0.1] * 384
    provider.dimension = 384
    return provider


# ---------------------------------------------------------------------------
# Gate 1 — retrieval threshold & refusal
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_retrieval_gate_refuses_below_threshold_without_llm_call():
    user_id = uuid.uuid4()
    low_row = (make_chunk_row(user_id), 0.42)  # below 0.65 threshold

    retrieval = RetrievalService(fake_embedding_provider(), threshold=0.65, top_k=5)
    with patch.object(ChunkRepository, "similarity_search", return_value=[low_row]):
        result = await retrieval.retrieve(
            db=AsyncMock(), user_id=user_id, query="what is X?"
        )

    # Gate 1 verdict
    assert result.gate_passed is False
    assert result.top_similarity == pytest.approx(0.42)
    assert "0.42" in result.refusal_reason and "0.65" in result.refusal_reason

    # Refusal happens WITHOUT an LLM call
    llm = AsyncMock()
    llm.generate_response.return_value = "This must never be rendered [1]."
    outcome = await GenerationService(llm).generate(
        user_id=user_id, query="what is X?", retrieval_result=result
    )
    assert outcome.refused is True
    assert outcome.refusal_gate == "retrieval"
    assert outcome.confidence is ConfidenceLabel.REFUSED
    assert outcome.confidence_score == 0.0
    assert "[1]" not in outcome.answer
    llm.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_retrieval_gate_passes_with_high_similarity():
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    rows = [(make_chunk_row(user_a), 0.91), (make_chunk_row(user_b, page_number=2), 0.85)]

    retrieval = RetrievalService(fake_embedding_provider(), threshold=0.65, top_k=5)
    with patch.object(ChunkRepository, "similarity_search", return_value=rows):
        result = await retrieval.retrieve(
            db=AsyncMock(), user_id=user_a, query="what is X?"
        )

    assert result.gate_passed is True
    assert result.refusal_reason is None
    assert result.top_similarity == pytest.approx(0.91)
    assert [c.citation_index for c in result.chunks] == [1, 2]
    assert result.similarities == [0.91, 0.85]


@pytest.mark.asyncio
async def test_retrieval_gate_refuses_when_nothing_retrieved():
    retrieval = RetrievalService(fake_embedding_provider(), threshold=0.65, top_k=5)
    with patch.object(ChunkRepository, "similarity_search", return_value=[]):
        result = await retrieval.retrieve(
            db=AsyncMock(), user_id=uuid.uuid4(), query="anything?"
        )

    assert result.gate_passed is False
    assert result.top_similarity is None
    assert "no passages" in result.refusal_reason


# ---------------------------------------------------------------------------
# Vector search user isolation (SQL-level scoping)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_vector_search_sql_is_user_scoped():
    """The ANN query itself must filter by user_id — no route can bypass it."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_session.execute.return_value = mock_result

    user_a_id = uuid.uuid4()
    await ChunkRepository.similarity_search(
        session=mock_session,
        user_id=user_a_id,
        query_embedding=[0.0] * 384,
        top_k=5,
    )
    stmt = mock_session.execute.call_args[0][0]
    sql = str(stmt)
    assert "user_id" in sql
    assert "LIMIT" in sql.upper()

    # Same statement shape with a document subset still carries user_id
    mock_session.execute.reset_mock()
    await ChunkRepository.similarity_search(
        session=mock_session,
        user_id=user_a_id,
        query_embedding=[0.0] * 384,
        top_k=5,
        document_ids=[uuid.uuid4()],
    )
    stmt2 = mock_session.execute.call_args[0][0]
    assert "user_id" in str(stmt2)
    assert "document_id" in str(stmt2)


# ---------------------------------------------------------------------------
# Citation integrity — fabricated citations can never render
# ---------------------------------------------------------------------------
def test_citation_validation_strips_fabricated_and_drops_unbacked_claims():
    valid = {1, 2}
    raw = (
        "DocuSage uses repository-level isolation [1]. "
        "The moon is made of cheese [9]. "
        "Citations can be combined [2][7]. "
        "This sentence has no marker at all."
    )

    answer, used, dropped = validate_citations(raw, valid)

    assert "[9]" not in answer
    assert "[7]" not in answer
    assert "[1]" in answer and "[2]" in answer
    assert used == [1, 2]
    # The fully-unbacked claim was dropped whole
    assert len(dropped) == 1
    assert "moon is made of cheese" in dropped[0]
    # Uncited framing text is preserved
    assert "no marker at all" in answer


def test_citation_validation_empty_valid_set_drops_all_cited_sentences():
    answer, used, dropped = validate_citations("Fabricated [3]. Also fake [4].", set())
    assert used == []
    assert answer == ""
    assert len(dropped) == 2


def test_confidence_heuristic_bands_are_documented_not_calibrated():
    # High: strong top-1 and strong tail
    label, score = compute_confidence([0.95, 0.93], gate_passed=True)
    assert label is ConfidenceLabel.HIGH
    assert score == pytest.approx(0.7 * 0.95 + 0.3 * 0.94)

    # Medium: comfortably above threshold, below the High band
    label, score = compute_confidence([0.70, 0.68], gate_passed=True)
    assert label is ConfidenceLabel.MEDIUM
    assert score == pytest.approx(0.7 * 0.70 + 0.3 * 0.69)

    # Low: thin top-1 margin with a weak tail pulls the composite below threshold
    label, score = compute_confidence([0.66, 0.10], gate_passed=True)
    assert label is ConfidenceLabel.LOW
    assert score < settings.RETRIEVAL_SIMILARITY_THRESHOLD

    # Refused: gate failed or nothing retrieved
    assert compute_confidence([0.40], gate_passed=False) == (ConfidenceLabel.REFUSED, 0.0)
    assert compute_confidence([], gate_passed=True) == (ConfidenceLabel.REFUSED, 0.0)

    # Scores are always within [0, 1]
    for sims in ([0.99, 0.98], [0.70, 0.68], [0.66, 0.10]):
        _, s = compute_confidence(sims, gate_passed=True)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Gate 2 — generation gate & end-to-end answer
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generation_gate_refuses_when_all_citations_fabricated():
    user_id = uuid.uuid4()
    row = (make_chunk_row(user_id), 0.90)

    retrieval = RetrievalService(fake_embedding_provider(), threshold=0.65)
    with patch.object(ChunkRepository, "similarity_search", return_value=[row]):
        result = await retrieval.retrieve(
            db=AsyncMock(), user_id=user_id, query="q"
        )
    assert result.gate_passed is True  # Gate 1 passed

    llm = AsyncMock()
    llm.generate_response.return_value = "Completely invented claim [9]."
    outcome = await GenerationService(llm).generate(
        user_id=user_id, query="q", retrieval_result=result
    )

    assert outcome.refused is True
    assert outcome.refusal_gate == "generation"
    assert outcome.confidence is ConfidenceLabel.REFUSED
    assert "[9]" not in outcome.answer
    assert "discarded" in outcome.refusal_reason


@pytest.mark.asyncio
async def test_generation_gate_drops_citations_not_owned_by_user():
    """Defense in depth: a chunk from another tenant can never be cited."""
    owner_id = uuid.uuid4()
    attacker_id = uuid.uuid4()
    owned_row = (make_chunk_row(owner_id, text="Owned evidence."), 0.92)
    foreign_row = (make_chunk_row(owner_id, text="Foreign evidence."), 0.88)
    # Simulate a retrieval result built for the owner, replayed by another user
    retrieval = RetrievalService(fake_embedding_provider(), threshold=0.65)
    with patch.object(
        ChunkRepository, "similarity_search", return_value=[owned_row, foreign_row]
    ):
        result = await retrieval.retrieve(
            db=AsyncMock(), user_id=owner_id, query="q"
        )

    llm = AsyncMock()
    llm.generate_response.return_value = "Owned fact [1]. Foreign fact [2]."

    # Owner keeps both citations
    owner_outcome = await GenerationService(llm).generate(
        user_id=owner_id, query="q", retrieval_result=result
    )
    assert owner_outcome.refused is False
    assert owner_outcome.citations_used == [1, 2]

    # Non-owner: neither chunk is theirs → full Gate 2 refusal
    thief_outcome = await GenerationService(llm).generate(
        user_id=attacker_id, query="q", retrieval_result=result
    )
    assert thief_outcome.refused is True
    assert thief_outcome.refusal_gate == "generation"
    assert "[1]" not in thief_outcome.answer and "[2]" not in thief_outcome.answer


@pytest.mark.asyncio
async def test_end_to_end_valid_answer_with_verified_citations():
    user_id = uuid.uuid4()
    rows = [
        (make_chunk_row(user_id, text="Isolation happens at the repository layer."), 0.92),
        (make_chunk_row(user_id, text="Vector search filters by user_id in SQL.", page_number=2), 0.87),
    ]

    retrieval = RetrievalService(fake_embedding_provider(), threshold=0.65)
    with patch.object(ChunkRepository, "similarity_search", return_value=rows):
        result = await retrieval.retrieve(
            db=AsyncMock(), user_id=user_id, query="How is isolation enforced?"
        )
    assert result.gate_passed is True

    llm = AsyncMock()
    llm.generate_response.return_value = (
        "Isolation is enforced in the repository layer [1]. "
        "The vector query itself filters by user_id [2]."
    )
    outcome = await GenerationService(llm).generate(
        user_id=user_id,
        query="How is isolation enforced?",
        retrieval_result=result,
    )

    assert outcome.refused is False
    assert outcome.citations_used == [1, 2]
    assert "[1]" in outcome.answer and "[2]" in outcome.answer
    assert outcome.confidence is ConfidenceLabel.HIGH
    assert 0.0 < outcome.confidence_score <= 1.0
    assert outcome.dropped_claims == []

    # The prompt must ground the model: numbered chunks, page tags, the question
    prompt = llm.generate_response.call_args[0][0]
    assert "[1]" in prompt and "[2]" in prompt
    assert "page 2" in prompt
    assert "How is isolation enforced?" in prompt
    # And the system prompt must demand citations
    system_prompt = llm.generate_response.call_args.kwargs["system_prompt"]
    assert "Cite every factual sentence" in system_prompt
