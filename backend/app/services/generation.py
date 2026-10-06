"""Generation service — the generation gate (Gate 2), citation-integrity
validation, and the confidence heuristic.

Guarantees provided here:

1. Gate 1 short-circuit: if retrieval refused, no LLM call is ever made.
2. Citation integrity: every ``[n]`` marker in a rendered answer is verified
   against the chunks that were actually retrieved AND owned by the requesting
   user. Fabricated indexes are stripped; sentences whose citations are
   entirely fabricated are dropped (their claims were unbacked).
3. Gate 2 (generation gate): if no verifiable citation survives validation,
   the answer is refused instead of shown.
4. Confidence is a documented HEURISTIC composite of top-k retrieval
   similarities. It is NOT a calibrated probability and must never be
   described as one.
"""
import re
import uuid
from dataclasses import dataclass
from enum import Enum

from app.core.config import get_settings
from app.providers.base import LLMProvider
from app.services.retrieval import RetrievalResult

settings = get_settings()

SYSTEM_PROMPT = (
    "You are DocuSage, an enterprise document intelligence assistant. "
    "Answer ONLY using the numbered context chunks provided. "
    "Cite every factual sentence with the bracketed index of the chunk that "
    "supports it, for example [1] or [2][3]. "
    "Never use knowledge from outside the context, and never invent indexes. "
    "If the context does not contain the answer, say so explicitly."
)

# Matches citation markers like [1] or [12] (pure integers only).
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# Splits an answer into sentences, keeping the terminating punctuation.
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

# Finds a completed sentence boundary inside a streaming buffer: a terminator
# immediately followed by whitespace (same condition SENTENCE_PATTERN splits on).
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s")


class ConfidenceLabel(str, Enum):
    """Confidence badge values shown in the UI. Heuristic, not calibrated."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    REFUSED = "Refused"


def compute_confidence(
    similarities: list[float], gate_passed: bool
) -> tuple[ConfidenceLabel, float]:
    """Composite confidence heuristic over top-k retrieval similarities.

    composite = 0.7 * top-1 + 0.3 * mean(top-k)

    Bands:
      - composite >= CONFIDENCE_HIGH_THRESHOLD        -> High
      - composite >= RETRIEVAL_SIMILARITY_THRESHOLD   -> Medium
      - otherwise                                     -> Low
        (gate passed on a thin top-1 margin with a weak tail)
      - gate not passed (or nothing retrieved)        -> Refused

    This is a heuristic, NOT a calibrated probability.
    """
    if not gate_passed or not similarities:
        return ConfidenceLabel.REFUSED, 0.0

    top = similarities[0]
    mean_top_k = sum(similarities) / len(similarities)
    composite = round(0.7 * top + 0.3 * mean_top_k, 4)

    if composite >= settings.CONFIDENCE_HIGH_THRESHOLD:
        return ConfidenceLabel.HIGH, composite
    if composite >= settings.RETRIEVAL_SIMILARITY_THRESHOLD:
        return ConfidenceLabel.MEDIUM, composite
    return ConfidenceLabel.LOW, composite


def _validate_sentence(sentence: str, valid_indexes: set[int]) -> str | None:
    """Validate one complete sentence against the retrieved evidence.

    Returns the sentence with fabricated markers stripped, or ``None`` when
    *every* marker in it is fabricated (the claim is unbacked and must never
    render). Sentences without markers are framing text and are kept.
    """
    markers = [int(m) for m in CITATION_PATTERN.findall(sentence)]
    if not markers:
        return sentence
    if not any(m in valid_indexes for m in markers):
        return None
    return CITATION_PATTERN.sub(
        lambda m: m.group(0) if int(m.group(1)) in valid_indexes else "",
        sentence,
    )


def validate_citations(
    raw_answer: str, valid_indexes: set[int]
) -> tuple[str, list[int], list[str]]:
    """Apply citation-integrity validation to a raw model answer.

    Rules (explicit by design — never let a fabricated citation render):

    - A marker pointing outside ``valid_indexes`` is fabricated and stripped.
    - A sentence whose markers are *entirely* fabricated is dropped whole,
      because its claims are unbacked by retrieved evidence.
    - Sentences with no marker at all are kept as framing text; the system
      prompt requires factual sentences to carry markers.

    Returns:
        (sanitized_answer, used_indexes_in_order, dropped_claims)
    """
    sentences = [
        s for s in SENTENCE_PATTERN.split(raw_answer.strip()) if s.strip()
    ]

    kept: list[str] = []
    used: list[int] = []
    dropped: list[str] = []

    for sentence in sentences:
        cleaned = _validate_sentence(sentence, valid_indexes)
        if cleaned is None:
            dropped.append(sentence.strip())
            continue
        kept.append(cleaned)
        for idx in (int(m) for m in CITATION_PATTERN.findall(cleaned)):
            if idx not in used:
                used.append(idx)

    return " ".join(kept), used, dropped


class StreamingCitationSanitizer:
    """Validates citations incrementally so unsafe text never reaches a client.

    Buffers the in-progress sentence and emits only complete sentences that
    pass the exact same rules as ``validate_citations``:

    - fabricated markers are stripped *before* the sentence is emitted,
    - a sentence whose markers are entirely fabricated is dropped whole,

    so a fabricated citation can never render live. Concatenating all emitted
    text equals ``validate_citations(raw)[0]`` for the full raw stream, which
    keeps the live draft and the final validated answer identical.
    """

    def __init__(self, valid_indexes: set[int]):
        self.valid_indexes = valid_indexes
        self.buffer = ""
        self.used_indexes: list[int] = []
        self.dropped_claims: list[str] = []
        self._emitted_any = False

    def feed(self, token: str) -> str:
        """Consume a raw model token; return text that is safe to render now."""
        self.buffer += token
        return self._drain(final=False)

    def flush(self) -> str:
        """Emit any trailing buffered sentence once the stream ends."""
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        emitted: list[str] = []
        while True:
            match = SENTENCE_BOUNDARY_PATTERN.search(self.buffer)
            if match:
                sentence = self.buffer[: match.start()]
                self.buffer = self.buffer[match.end() :]
            elif final and self.buffer.strip():
                sentence = self.buffer
                self.buffer = ""
            else:
                break
            self._handle_sentence(sentence, emitted)

        if not emitted:
            return ""
        prefix = " " if self._emitted_any else ""
        self._emitted_any = True
        return prefix + " ".join(emitted)

    def _handle_sentence(self, sentence: str, emitted: list[str]) -> None:
        sentence = sentence.strip()
        if not sentence:
            return
        cleaned = _validate_sentence(sentence, self.valid_indexes)
        if cleaned is None:
            self.dropped_claims.append(sentence)
            return
        emitted.append(cleaned)
        for idx in (int(m) for m in CITATION_PATTERN.findall(cleaned)):
            if idx not in self.used_indexes:
                self.used_indexes.append(idx)


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of one generation round: answer or honest refusal + confidence."""

    answer: str
    refused: bool
    refusal_gate: str | None  # "retrieval" | "generation" | None
    refusal_reason: str | None
    confidence: ConfidenceLabel
    confidence_score: float
    citations_used: list[int]
    dropped_claims: list[str]


class GenerationService:
    """Generates answers strictly from retrieved evidence with dual-gate abstention."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    @staticmethod
    def build_prompt(query: str, retrieval_result: RetrievalResult) -> str:
        """Build the grounded prompt with numbered, page-tagged context chunks."""
        blocks = [
            f"[{c.citation_index}] (document {c.document_id}, page {c.page_number})\n{c.text}"
            for c in retrieval_result.chunks
        ]
        context = "\n\n".join(blocks)
        return (
            "Context chunks from the user's documents:\n\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            "Answer the question using ONLY the context chunks above. "
            "Cite every factual sentence with its bracketed chunk index, "
            "for example [1] or [2][3]. "
            "If the context does not contain the answer, say so explicitly."
        )

    async def generate(
        self,
        user_id: uuid.UUID,
        query: str,
        retrieval_result: RetrievalResult,
    ) -> GenerationResult:
        """Produce a validated answer (or an honest refusal) for one question.

        Args:
            user_id: The authenticated user. Only chunks owned by this user
                may be cited — re-verified here as defense in depth.
            query: The user's question.
            retrieval_result: Output of ``RetrievalService.retrieve``.
        """
        # ---- Gate 1 short-circuit: refusal without an LLM call ----
        if not retrieval_result.gate_passed:
            reason = retrieval_result.refusal_reason or (
                "the available evidence is insufficient to answer this question"
            )
            return self.refusal("retrieval", reason, retrieval_result)

        prompt = self.build_prompt(query, retrieval_result)
        raw_answer = await self.llm_provider.generate_response(
            prompt, system_prompt=SYSTEM_PROMPT
        )
        return self.finalize(raw_answer, retrieval_result, user_id)

    @staticmethod
    def owned_indexes(retrieval_result: RetrievalResult, user_id: uuid.UUID) -> set[int]:
        """Citation indexes whose retrieved chunk is owned by ``user_id``.

        Ownership re-check — citations must map to chunks owned by this user,
        independent of how retrieval was invoked.
        """
        return {
            c.citation_index
            for c in retrieval_result.chunks
            if c.user_id == user_id
        }

    def finalize(
        self,
        raw_answer: str,
        retrieval_result: RetrievalResult,
        user_id: uuid.UUID,
    ) -> GenerationResult:
        """Apply Gate 2 + citation integrity + confidence to a complete answer.

        Shared by the non-streaming path (``generate``) and the streaming chat
        endpoint (called once its stream completes), so both behave identically.
        """
        owned = self.owned_indexes(retrieval_result, user_id)
        answer, used, dropped = validate_citations(raw_answer, owned)

        # ---- Gate 2: generation gate — no verifiable evidence link ----
        if not used:
            reason = (
                "the generated response could not be verified against any "
                "retrieved passage, so it was discarded rather than shown unverified"
            )
            return self.refusal("generation", reason, retrieval_result)

        label, score = compute_confidence(
            retrieval_result.similarities, gate_passed=True
        )
        return GenerationResult(
            answer=answer,
            refused=False,
            refusal_gate=None,
            refusal_reason=None,
            confidence=label,
            confidence_score=score,
            citations_used=used,
            dropped_claims=dropped,
        )

    @staticmethod
    def refusal(
        gate: str,
        reason: str,
        retrieval_result: RetrievalResult,
    ) -> GenerationResult:
        """Build an honest refusal that explains why evidence was insufficient.

        Refusals always carry the ``Refused`` badge with score 0.0, regardless
        of what the similarity heuristic would have suggested.
        """
        return GenerationResult(
            answer=f"I don't know: {reason}.",
            refused=True,
            refusal_gate=gate,
            refusal_reason=reason,
            confidence=ConfidenceLabel.REFUSED,
            confidence_score=0.0,
            citations_used=[],
            dropped_claims=[],
        )
