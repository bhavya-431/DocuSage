import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_user, get_db
from app.main import app
from app.models.chat import ChatMessage, ChatSession
from app.models.user import User
from app.repositories.chat_repo import ChatRepository
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.services.generation import (
    StreamingCitationSanitizer,
    validate_citations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_sse(text: str) -> list[dict]:
    """Parse `data: {...}\n\n` SSE frames into dicts."""
    events = []
    for frame in text.split("\n\n"):
        frame = frame.strip()
        if frame.startswith("data: "):
            events.append(json.loads(frame[6:]))
    return events


def make_chunk_row(user_id, text="Evidence text.", page_number=1, document_id=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        user_id=user_id,
        page_number=page_number,
        text=text,
        char_count=len(text),
        chunk_index=0,
    )


def make_user() -> User:
    return User(
        id=uuid.uuid4(),
        email="chat_user@example.com",
        hashed_password="hash",
        created_at=datetime.now(timezone.utc),
    )


def make_session(user_id, title="New chat") -> ChatSession:
    return ChatSession(id=uuid.uuid4(), user_id=user_id, title=title)


def llm_mock(tokens: list[str]) -> tuple[MagicMock, list[str]]:
    """LLM mock whose stream_response is a recording async generator."""
    prompts: list[str] = []

    async def fake_stream(prompt: str, system_prompt: str = ""):
        prompts.append(prompt)
        for token in tokens:
            yield token

    llm = MagicMock()
    llm.stream_response = MagicMock(side_effect=fake_stream)
    return llm, prompts


def emb_mock() -> AsyncMock:
    provider = AsyncMock()
    provider.embed_query.return_value = [0.1] * 384
    return provider


# ---------------------------------------------------------------------------
# Repository-level user isolation (SQL scoping)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_chat_repository_sql_is_user_scoped():
    mock_session = MagicMock()
    mock_session.flush = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_result.rowcount = 5  # for update_title's rowcount check

    user_id = uuid.uuid4()
    session_id = uuid.uuid4()

    created = await ChatRepository.create_session(mock_session, user_id, title="T")
    # create_session writes via add/flush (no SELECT), and stamps the owner
    assert created.user_id == user_id
    mock_session.add.assert_called_once_with(created)

    await ChatRepository.list_sessions(mock_session, user_id)
    assert "user_id" in str(mock_session.execute.call_args[0][0])

    await ChatRepository.get_session(mock_session, user_id, session_id)
    assert "user_id" in str(mock_session.execute.call_args[0][0])

    await ChatRepository.list_messages(mock_session, user_id, session_id)
    sql = str(mock_session.execute.call_args[0][0])
    assert "user_id" in sql and "session_id" in sql

    await ChatRepository.update_title(mock_session, user_id, session_id, "New title")
    sql = str(mock_session.execute.call_args[0][0])
    assert "user_id" in sql and "UPDATE" in sql.upper()


@pytest.mark.asyncio
async def test_create_message_stamps_owner_and_session():
    mock_session = MagicMock()
    mock_session.flush = AsyncMock()

    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    message = await ChatRepository.create_message(
        mock_session,
        user_id,
        session_id,
        question="Q?",
        answer="A [1].",
        document_ids=["doc-1"],
        citations=[{"index": 1}],
        refused=False,
        refusal_gate=None,
        confidence_label="High",
        confidence_score=0.91,
    )
    assert message.user_id == user_id
    assert message.session_id == session_id
    assert message.citations == [{"index": 1}]
    assert message.confidence_label == "High"
    mock_session.add.assert_called_once_with(message)


# ---------------------------------------------------------------------------
# API-level isolation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_messages_returns_404_for_foreign_session():
    user = make_user()
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: AsyncMock()

    with patch.object(ChatRepository, "get_session", return_value=None):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/chat/sessions/{uuid.uuid4()}/messages")
    assert resp.status_code == 404

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ask_returns_404_for_foreign_session():
    user = make_user()
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: AsyncMock()

    with patch.object(ChatRepository, "get_session", return_value=None):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/api/chat/sessions/{uuid.uuid4()}/messages",
                json={"question": "hi"},
            )
    assert resp.status_code == 404

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ask_rejects_document_ids_not_owned_by_user():
    user = make_user()
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: AsyncMock()

    with patch.object(
        ChatRepository, "get_session", return_value=make_session(user.id)
    ), patch.object(DocumentRepository, "get_by_ids", return_value=[]):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/api/chat/sessions/{uuid.uuid4()}/messages",
                json={"question": "hi", "document_ids": [str(uuid.uuid4())]},
            )
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Gate 1 over SSE — refusal without an LLM call
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ask_gate1_refusal_streams_done_without_llm_call():
    user = make_user()
    chat_session = make_session(user.id)
    low_row = (make_chunk_row(user.id), 0.08)  # below the 0.20 threshold

    llm, prompts = llm_mock(["should never stream"])

    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: AsyncMock()

    async def fake_create(session=None, **kwargs):
        return ChatMessage(
            id=uuid.uuid4(), created_at=datetime.now(timezone.utc), **kwargs
        )

    with patch.object(
        ChatRepository, "get_session", return_value=chat_session
    ), patch.object(
        ChatRepository, "create_message", side_effect=fake_create
    ), patch.object(
        ChatRepository, "update_title", return_value=True
    ), patch(
        "app.api.chat.get_embedding_provider", return_value=emb_mock()
    ), patch(
        "app.api.chat.get_llm_provider", return_value=llm
    ), patch.object(
        ChunkRepository, "similarity_search", return_value=[low_row]
    ), patch.object(
        DocumentRepository, "list_documents", return_value=[]
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/api/chat/sessions/{chat_session.id}/messages",
                json={"question": "What is the policy?"},
            )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(resp.text)
    # Lone `done` event — no tokens were streamed
    assert len(events) == 1
    assert events[0]["type"] == "done"
    message = events[0]["message"]
    assert message["refused"] is True
    assert message["refusal_gate"] == "retrieval"
    assert message["confidence_label"] == "Refused"
    assert message["confidence_score"] == 0.0
    assert "I don't know" in message["answer"]
    assert "0.08" in message["answer"]

    # The LLM was never invoked
    llm.stream_response.assert_not_called()
    assert prompts == []

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Streaming path — only validated tokens ever reach the client
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ask_streams_only_validated_tokens_and_persists_verified_citations():
    user = make_user()
    chat_session = make_session(user.id)  # default title "New chat" → auto-rename
    doc_id = uuid.uuid4()
    rows = [
        (
            make_chunk_row(
                user.id,
                text="Isolation lives in the repository layer.",
                document_id=doc_id,
            ),
            0.93,
        ),
        (
            make_chunk_row(
                user.id,
                text="Vector SQL filters by user_id.",
                page_number=2,
                document_id=doc_id,
            ),
            0.88,
        ),
    ]
    prior_sentinel = "SENTINEL_PRIOR_ANSWER_SHOULD_NEVER_APPEAR"

    llm, prompts = llm_mock(
        [
            "Isolation lives in the repository layer [1]. ",
            "Fabricated claim [9]. ",
            "Vector SQL filters by user_id [2][7].",
        ]
    )

    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: AsyncMock()

    async def fake_create(session=None, **kwargs):
        return ChatMessage(
            id=uuid.uuid4(), created_at=datetime.now(timezone.utc), **kwargs
        )

    async def fake_history(session=None, *args, **kwargs):
        return [
            ChatMessage(
                id=uuid.uuid4(),
                user_id=user.id,
                session_id=chat_session.id,
                question="old q",
                answer=prior_sentinel,
                document_ids=[],
                citations=[],
                refused=False,
                refusal_gate=None,
                confidence_label="High",
                confidence_score=0.9,
                created_at=datetime.now(timezone.utc),
            )
        ]

    with patch.object(
        ChatRepository, "get_session", return_value=chat_session
    ), patch.object(
        ChatRepository, "create_message", side_effect=fake_create
    ), patch.object(
        ChatRepository, "list_messages", side_effect=fake_history
    ), patch.object(
        ChatRepository, "update_title", return_value=True
    ) as mock_rename, patch(
        "app.api.chat.get_embedding_provider", return_value=emb_mock()
    ), patch(
        "app.api.chat.get_llm_provider", return_value=llm
    ), patch.object(
        ChunkRepository, "similarity_search", return_value=rows
    ), patch.object(
        DocumentRepository,
        "list_documents",
        return_value=[SimpleNamespace(id=doc_id, filename="handbook.pdf")],
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/api/chat/sessions/{chat_session.id}/messages",
                json={"question": "How is isolation enforced?"},
            )

    assert resp.status_code == 200
    raw_stream = resp.text

    # --- No fabricated citation or unbacked claim ever rendered live ---
    assert "[9]" not in raw_stream
    assert "[7]" not in raw_stream
    assert "Fabricated claim" not in raw_stream

    events = parse_sse(raw_stream)
    types = [e["type"] for e in events]
    assert types[0] == "token"
    assert types[-1] == "done"
    assert types.count("done") == 1

    streamed_text = "".join(e.get("text", "") for e in events if e["type"] == "token")

    # --- Authoritative final message ---
    message = events[-1]["message"]
    assert message["refused"] is False
    assert message["confidence_label"] in ("High", "Medium", "Low")
    assert message["question"] == "How is isolation enforced?"
    # Live draft and final answer agree exactly
    assert message["answer"] == streamed_text
    assert "[9]" not in message["answer"] and "[7]" not in message["answer"]

    # --- Verified citations with source-panel payload ---
    assert message["citations"][0]["index"] == 1
    assert message["citations"][0]["filename"] == "handbook.pdf"
    assert message["citations"][0]["page_number"] == 1
    assert message["citations"][1]["page_number"] == 2
    assert all("text" in c and "similarity" in c for c in message["citations"])

    # --- No conversation memory: prompt = retrieval + question only ---
    assert len(prompts) == 1
    assert prior_sentinel not in prompts[0]
    assert "How is isolation enforced?" in prompts[0]
    assert "[1]" in prompts[0] and "[2]" in prompts[0]

    # --- Session auto-titled from first question ---
    mock_rename.assert_awaited_once()

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Sanitizer == batch validator (streaming agrees with final answer)
# ---------------------------------------------------------------------------
def test_streaming_sanitizer_matches_batch_validator_char_by_char():
    valid = {1, 2}
    raw = (
        "Grounded claim [1] with detail. "
        "Entirely fabricated sentence [9] here. "
        "Mixed claim [2] plus fake [7]. "
        "Framing sentence without markers. "
        "Trailing claim [1]."
    )

    sanitizer = StreamingCitationSanitizer(valid)
    streamed = ""
    for ch in raw:  # worst case: one char per token
        streamed += sanitizer.feed(ch)
    streamed += sanitizer.flush()

    batch_answer, batch_used, batch_dropped = validate_citations(raw, valid)

    assert streamed == batch_answer
    assert sanitizer.used_indexes == batch_used
    assert sanitizer.dropped_claims == batch_dropped
    assert "[9]" not in streamed and "[7]" not in streamed
    assert "entirely fabricated" not in streamed


def test_streaming_sanitizer_flush_emits_unterminated_final_sentence():
    sanitizer = StreamingCitationSanitizer({1})
    out = sanitizer.feed("Only claim [1]")  # no terminal punctuation
    assert out == ""
    assert sanitizer.flush() == "Only claim [1]"
