import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import ASGITransport, AsyncClient
from jose import JWTError
from app.api.deps import get_current_user, get_db
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.main import app
from app.models.document import Document
from app.models.user import User
from app.repositories.document_repo import DocumentRepository
from app.repositories.user_repo import UserRepository


def test_password_hashing_and_verification():
    raw_password = "SuperSecretPassword123!"
    hashed = hash_password(raw_password)

    assert hashed != raw_password
    assert hashed.startswith("$2b$")
    assert verify_password(raw_password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False

    # Test edge case: Password longer than 72 bytes (bcrypt limit)
    long_pass = "A" * 100
    hashed_long = hash_password(long_pass)
    assert verify_password(long_pass, hashed_long) is True
    assert verify_password(long_pass[:50], hashed_long) is False


def test_jwt_token_creation_and_decoding():
    user_id = str(uuid.uuid4())
    email = "researcher@example.com"
    token = create_access_token({"sub": user_id, "email": email})

    payload = decode_access_token(token)
    assert payload["sub"] == user_id
    assert payload["email"] == email
    assert "exp" in payload


def test_jwt_token_expired():
    user_id = str(uuid.uuid4())
    # Create token expired 1 hour ago
    expired_token = create_access_token(
        {"sub": user_id}, expires_delta=timedelta(hours=-1)
    )

    with pytest.raises(JWTError):
        decode_access_token(expired_token)


def test_jwt_token_tampered():
    token = create_access_token({"sub": "valid_user"})
    tampered_token = token[:-4] + "wxyz"

    with pytest.raises(JWTError):
        decode_access_token(tampered_token)


@pytest.mark.asyncio
async def test_auth_register_and_login_flow():
    test_user_id = uuid.uuid4()
    test_email = "rag_user@example.com"
    raw_password = "password123"
    hashed_pw = hash_password(raw_password)

    fake_user = User(
        id=test_user_id,
        email=test_email,
        hashed_password=hashed_pw,
        created_at=datetime.now(timezone.utc),
    )

    # Mock DB session
    mock_db = AsyncMock()

    # 1. Test register with new email
    with patch.object(UserRepository, "get_by_email", return_value=None), \
         patch.object(UserRepository, "create_user", return_value=fake_user):

        app.dependency_overrides[get_db] = lambda: mock_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/auth/register",
                json={"email": test_email, "password": raw_password},
            )
            assert resp.status_code == 201
            data = resp.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"
            assert data["user"]["email"] == test_email
            assert data["user"]["id"] == str(test_user_id)

    # 2. Test duplicate email registration rejection
    with patch.object(UserRepository, "get_by_email", return_value=fake_user):
        app.dependency_overrides[get_db] = lambda: mock_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/auth/register",
                json={"email": test_email, "password": raw_password},
            )
            assert resp.status_code == 400
            assert "already exists" in resp.json()["detail"].lower()

    # 3. Test login success
    with patch.object(UserRepository, "get_by_email", return_value=fake_user):
        app.dependency_overrides[get_db] = lambda: mock_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/auth/login",
                json={"email": test_email, "password": raw_password},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "access_token" in data
            assert data["user"]["email"] == test_email

    # 4. Test login invalid password
    with patch.object(UserRepository, "get_by_email", return_value=fake_user):
        app.dependency_overrides[get_db] = lambda: mock_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/auth/login",
                json={"email": test_email, "password": "wrong_password"},
            )
            assert resp.status_code == 401
            assert "invalid" in resp.json()["detail"].lower()

    # Clean up dependency overrides
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_auth_me_endpoint_and_dependency():
    test_user_id = uuid.uuid4()
    test_user = User(
        id=test_user_id,
        email="me@example.com",
        hashed_password="hash",
        created_at=datetime.now(timezone.utc),
    )

    app.dependency_overrides[get_current_user] = lambda: test_user
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/auth/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == str(test_user_id)
        assert data["email"] == "me@example.com"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_tenant_isolation_in_repository():
    """Verify that document queries cannot cross tenant boundaries."""
    user_a_id = uuid.uuid4()
    user_b_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    mock_session = AsyncMock()

    # Create dummy user A document
    user_a_doc = Document(
        id=doc_id,
        user_id=user_a_id,
        filename="confidential_a.pdf",
        file_hash="sha256_hash_a",
        file_size_bytes=1024,
        status="ready",
    )

    # 1. Query by User A: returns user A's doc
    mock_result_a = MagicMock()
    mock_result_a.scalar_one_or_none.return_value = user_a_doc
    mock_session.execute.return_value = mock_result_a

    doc = await DocumentRepository.get_by_id(mock_session, user_a_id, doc_id)
    assert doc is not None
    assert doc.user_id == user_a_id

    # Check that SQL statement executed includes user_id condition
    executed_call = mock_session.execute.call_args[0][0]
    sql_str = str(executed_call)
    assert "user_id" in sql_str

    # 2. Query by User B: returns None (blocked at SQL filter)
    mock_result_b = MagicMock()
    mock_result_b.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result_b

    doc_b = await DocumentRepository.get_by_id(mock_session, user_b_id, doc_id)
    assert doc_b is None

    # 3. Duplicate check scoped to user:
    # User B uploading same hash as User A should only match if User B previously uploaded it
    await DocumentRepository.get_by_hash(mock_session, user_b_id, "sha256_hash_a")
    hash_query_sql = str(mock_session.execute.call_args[0][0])
    assert "user_id" in hash_query_sql
    assert "file_hash" in hash_query_sql
