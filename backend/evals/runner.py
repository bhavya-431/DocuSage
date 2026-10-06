"""Eval harness for DocuSage's dual-gate abstention (PRD §9 methodology).

Scope — stated honestly, as the README will need to:

- The harness measures **Gate 1 (the retrieval gate)** and the confidence
  heuristic end-to-end through the real ingestion pipeline (PDF → extract →
  chunk with page metadata) and real embeddings, using an in-memory cosine
  search that mirrors pgvector similarity (`1 - (embedding <=> query)` for
  normalised vectors equals cosine similarity, which is what
  ``ChunkRepository.similarity_search`` returns).

- The gate decision itself is NOT reimplemented: the harness calls the exact
  ``evaluate_gate()`` used by ``RetrievalService.retrieve`` in production.

- The harness **never calls an LLM** — that is the point of the retrieval
  gate (refuse before spending a token). Gate 2 (generation/citation
  integrity) needs an LLM provider and is covered by unit tests in
  tests/test_phase4.py and tests/test_phase5.py.

Metrics tracked per PRD §9:

- coverage: answerable cases where the gate answered AND the expected phrase
  appeared in the top-k retrieved chunks (citations may cite any retrieved
  chunk, so top-k is the honest relevance criterion; top-1 rank quality is
  tracked separately as a metric)
- false-refusal rate: answerable cases the gate wrongly refused
- false-answer rate: forced-refusal cases the gate failed to refuse
  (the safety metric — must be 0)
- mean top-1 similarity across all cases
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import get_settings
from app.providers.base import EmbeddingProvider
from app.providers.embeddings import get_embedding_provider
from app.services.generation import compute_confidence
from app.services.retrieval import evaluate_gate

from evals.corpus import build_corpus_chunks

settings = get_settings()

DEFAULT_DATASET_PATH = Path(__file__).parent / "dataset.json"


def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace so phrase matching survives PDF
    extraction newlines and line-wrapping."""
    return " ".join(text.lower().split())


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity — provider-agnostic (works with or without
    pre-normalised embeddings, matching pgvector's cosine distance)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class EvalCaseResult:
    id: str
    category: str
    question: str
    gate_passed: bool
    top_similarity: float | None
    confidence: str
    phrase_found: bool | None
    phrase_in_top1: bool | None
    passed: bool
    detail: str


@dataclass
class EvalReport:
    results: list[EvalCaseResult] = field(default_factory=list)
    threshold: float = 0.0

    @property
    def answerable(self) -> list[EvalCaseResult]:
        return [r for r in self.results if r.category == "answerable"]

    @property
    def refusals(self) -> list[EvalCaseResult]:
        return [r for r in self.results if r.category == "refusal"]

    @property
    def coverage_rate(self) -> float:
        """Answerable cases correctly answered with evidence in top-1."""
        if not self.answerable:
            return 0.0
        return sum(1 for r in self.answerable if r.passed) / len(self.answerable)

    @property
    def top1_relevance_rate(self) -> float:
        """Answerable cases whose expected phrase was the top-1 chunk
        (pure rank quality, independent of the gate)."""
        answerable = [r for r in self.answerable if r.phrase_in_top1 is not None]
        if not answerable:
            return 0.0
        return sum(1 for r in answerable if r.phrase_in_top1) / len(answerable)

    @property
    def false_refusal_count(self) -> int:
        """Answerable cases the gate refused (wrongly)."""
        return sum(1 for r in self.answerable if not r.gate_passed)

    @property
    def false_answer_count(self) -> int:
        """Forced-refusal cases the gate failed to refuse — MUST be 0."""
        return sum(1 for r in self.refusals if r.gate_passed)

    @property
    def mean_top_similarity(self) -> float:
        scores = [r.top_similarity for r in self.results if r.top_similarity is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def summary(self) -> str:
        lines = [
            f"DocuSage eval — {len(self.results)} cases "
            f"(threshold {self.threshold:.2f})",
            f"  answerable:           {len(self.answerable)}",
            f"  forced refusals:      {len(self.refusals)}",
            f"  coverage:             {self.coverage_rate:.0%} "
            f"({sum(1 for r in self.answerable if r.passed)}/{len(self.answerable)} answered "
            f"with evidence in top-k)",
            f"  top-1 rank quality:   {self.top1_relevance_rate:.0%}",
            f"  false refusals:       {self.false_refusal_count}",
            f"  false answers (safety, want 0): {self.false_answer_count}",
            f"  mean top-1 similarity: {self.mean_top_similarity:.3f}",
        ]
        failures = [r for r in self.results if not r.passed]
        if failures:
            lines.append("  failures:")
            lines.extend(
                f"    - {r.id} [{r.category}] {r.detail}" for r in failures
            )
        return "\n".join(lines)


async def run_eval(
    dataset_path: Path | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    threshold: float | None = None,
    top_k: int | None = None,
) -> EvalReport:
    """Run the golden set through corpus ingestion → embedding → retrieval gate.

    Args:
        dataset_path: defaults to evals/dataset.json.
        embedding_provider: defaults to the env-configured provider (local
            MiniLM in development — zero API keys required).
        threshold: defaults to RETRIEVAL_SIMILARITY_THRESHOLD (production value).
        top_k: defaults to TOP_K_CHUNKS (production value).
    """
    threshold = threshold if threshold is not None else settings.RETRIEVAL_SIMILARITY_THRESHOLD
    top_k = top_k if top_k is not None else settings.TOP_K_CHUNKS
    provider = embedding_provider or get_embedding_provider()

    dataset = json.loads(
        (dataset_path or DEFAULT_DATASET_PATH).read_text(encoding="utf-8")
    )
    cases: list[dict] = dataset["cases"]

    # Ingest the fixture corpus through the production pipeline, embed once.
    chunks = build_corpus_chunks()
    if not chunks:
        raise RuntimeError("Fixture corpus produced no chunks — eval cannot run.")
    chunk_vectors = await provider.embed_texts([c["text"] for c in chunks])

    report = EvalReport(threshold=threshold)

    for case in cases:
        query_vector = await provider.embed_query(case["question"])
        scores = [cosine_similarity(query_vector, vec) for vec in chunk_vectors]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_idx = ranked[:top_k]
        top_scores = [scores[i] for i in top_idx]

        gate_passed, reason = evaluate_gate(top_scores, threshold)
        label, _score = compute_confidence(top_scores, gate_passed)

        # Relevance: expected fact must be among retrieved (top-k) chunks;
        # top-1 rank quality is tracked as a separate metric.
        if case["category"] == "answerable":
            phrase = _normalise(case["expect_phrase"])
            found = any(phrase in _normalise(chunks[i]["text"]) for i in top_idx)
            found_top1 = phrase in _normalise(chunks[top_idx[0]]["text"])
        else:
            found = None
            found_top1 = None

        if case["category"] == "answerable":
            passed = gate_passed and found
            if not gate_passed:
                detail = f"false refusal: {reason}"
            elif not found:
                detail = "answered but expected phrase not in top-k chunks"
            else:
                detail = ""
        else:  # refusal
            passed = not gate_passed
            detail = "" if passed else "SAFETY: gate passed on out-of-corpus question"

        report.results.append(
            EvalCaseResult(
                id=case["id"],
                category=case["category"],
                question=case["question"],
                gate_passed=gate_passed,
                top_similarity=top_scores[0] if top_scores else None,
                confidence=label.value,
                phrase_found=found,
                phrase_in_top1=found_top1,
                passed=passed,
                detail=detail,
            )
        )

    return report


if __name__ == "__main__":
    import asyncio

    _report = asyncio.run(run_eval())
    print(_report.summary())
