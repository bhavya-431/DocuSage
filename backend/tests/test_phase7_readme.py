"""Phase 7 — README acceptance tests (PRD §9: first-class deliverable)."""
from pathlib import Path

README_PATH = Path(__file__).resolve().parents[2] / "README.md"


def _readme() -> str:
    assert README_PATH.exists(), "README.md at the project root is required by PRD §9"
    return README_PATH.read_text(encoding="utf-8").lower()


def test_readme_has_architecture_and_module_walkthrough():
    """PRD §9.1: architecture diagram + module walkthrough."""
    text = _readme()
    assert "## 1. architecture" in text
    for token in ("fastapi", "pgvector", "providers", "repositories",
                  "ingestion", "retrieval", "generation", "sse"):
        assert token in text, f"architecture section must cover {token!r}"


def test_readme_setup_covers_local_and_deployed_modes():
    """PRD §9.2: local dev (docker + MiniLM, no keys) AND deployed mode (env swap)."""
    text = _readme()
    assert "## 2. setup" in text
    # Local, keyless
    assert "docker compose up" in text
    assert "minilm" in text
    assert "zero keys" in text
    # Deployed, env-driven
    assert "deployed mode" in text
    assert "database_url" in text
    assert "vite_api_url" in text
    assert "uvicorn app.main:app" in text
    assert "/health" in text


def test_readme_explains_abstention_gates_and_threshold_tradeoff():
    """PRD §9.3: how abstention works — both gates, thresholds, tradeoff."""
    text = _readme()
    assert "## 3. how abstention works" in text
    assert "retrieval gate" in text
    assert "generation gate" in text
    assert "0.20" in text and "threshold" in text
    assert "tradeoff" in text
    assert "calibrated probability" in text  # confidence honesty
    assert "fabricated" in text and "ownership" in text


def test_readme_documents_evaluation_methodology():
    """PRD §9.4: golden set construction, refusal-case design, metrics."""
    text = _readme()
    assert "## 4. evaluation methodology" in text
    assert "evals/dataset.json" in text
    assert "25" in text
    assert "forced-refusal" in text
    assert "expect_phrase" in text
    assert "coverage" in text
    assert "false answers" in text
    assert "python -m evals.runner" in text


def test_readme_lists_honest_limitations():
    """PRD §9.5: limitations — no OCR, heuristic confidence, English-only, tables."""
    text = _readme()
    assert "## 6. limitations" in text
    assert "no ocr" in text
    assert "heuristic" in text
    assert "english-only" in text
    assert "text-block tables" in text
    assert "memory" in text  # no-conversation-memory non-goal
