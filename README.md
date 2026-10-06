# DocuSage — Enterprise Document Intelligence Agent (RAG)

Upload PDFs → ask questions → get streaming answers with **inline page-level
citations**, **confidence badges**, and honest **"I don't know" refusals** when
evidence is insufficient. Multi-user, provider-agnostic, runnable locally with
zero API keys.

> Confidence is a **heuristic** based on retrieval similarity. It is **not** a
> calibrated probability, and the UI never presents it as one.

---

## 1. Architecture

```
React (Vite + TS + Tailwind)  ──SSE/JSON──►  FastAPI  ──►  PostgreSQL + pgvector
   frontend/                                 backend/app/
                                             ├── api/          HTTP + SSE routes
                                             ├── core/         config (env-driven), db, security (JWT)
                                             ├── models/       User, Document, DocumentChunk, ChatSession, ChatMessage
                                             ├── providers/    LLMProvider / EmbeddingProvider protocols
                                             ├── repositories/ every query user_id-scoped (incl. vector search)
                                             └── services/     ingestion · retrieval · generation
                                             backend/evals/    golden set + harness (no DB, no LLM)
```

### Module walkthrough

| Module | Responsibility |
|---|---|
| `core/config.py` | All settings via env (`pydantic-settings`); defaults suit local dev |
| `core/security.py` | bcrypt password hashing, JWT issue/verify |
| `providers/base.py` | `EmbeddingProvider` and `LLMProvider` **protocols** — no vendor names in business logic |
| `providers/embeddings.py` | `LocalMiniLMEmbeddingProvider` (offline, 384-d) \| `GeminiEmbeddingProvider` (768-d), selected by `EMBEDDING_PROVIDER` |
| `providers/llm.py` | Gemini \| Groq \| OpenRouter implementations, selected by `LLM_PROVIDER` |
| `services/ingestion.py` | PDF validation → extraction (PyMuPDF blocks, pdfplumber fallback) → chunking (800 chars / 150 overlap, page metadata) → batched embedding with exponential backoff on 429s |
| `services/retrieval.py` | User-scoped vector search + **`evaluate_gate()`** (Gate 1) — the exact function the eval harness measures |
| `services/generation.py` | **Gate 2**, citation-integrity validation, streaming sentence sanitizer, confidence heuristic |
| `repositories/*` | Every read/write filtered by `user_id` **inside the SQL**, including `WHERE user_id = …` alongside the ANN search — no route can bypass isolation |
| `evals/` | 25-case golden set (20 answerable + 5 forced-refusal) + runner |

### API surface

```
POST /api/auth/register | /api/auth/login        GET /api/auth/me
GET|POST /api/documents            POST /api/documents/upload
GET|DELETE /api/documents/{id}
POST /api/chat/sessions            GET  /api/chat/sessions
GET  /api/chat/sessions/{id}/messages
POST /api/chat/sessions/{id}/messages   ← SSE stream: token* → done | error
GET  /health                       ← liveness + DB + pgvector + provider checks
```

---

## 2. Setup

### Local development — no API keys required

```bash
# 1. Postgres + pgvector (one command)
docker compose up -d

# 2. Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload            # http://localhost:8000  (schema auto-bootstraps)

# 3. Frontend (separate terminal)
cd frontend
npm install
npm run dev                              # http://localhost:5173
```

- Embeddings default to **local MiniLM** (`EMBEDDING_PROVIDER=local`) — offline, zero keys.
- Copy `.env.example` → `.env` to override anything; `docker compose` DB credentials
  match its defaults.
- **Honest note:** the retrieval gate, citations, upload pipeline and the whole
  test suite run keyless. Generating *answers* needs one free-tier LLM key
  (`GEMINI_API_KEY` / `GROQ_API_KEY` / `OPENROUTER_API_KEY`). Without a key the
  system stays safe: generation fails closed into a Gate-2 refusal, never a
  hallucination. `GET /health` reports per-provider status.

### Deployed mode — same codebase, env swap

1. **Database:** any Postgres with the `pgvector` extension (Neon, Supabase,
   Aiven — free tiers work; this is *your deployment choice*, never baked into
   code). Set `DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db`.
   The app bootstraps `CREATE EXTENSION IF NOT EXISTS vector` + tables on boot.
2. **Secrets/providers:** set a real `JWT_SECRET_KEY`
   (`openssl rand -hex 32`), choose `LLM_PROVIDER` / `EMBEDDING_PROVIDER` and
   their keys. If you change the embedding model, set `EMBEDDING_MODEL`,
   `EMBEDDING_DIMENSION`, and **re-tune `RETRIEVAL_SIMILARITY_THRESHOLD`**
   (see §3–4) — similarity scales differ per provider.
3. **API:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   (set `ENVIRONMENT=production`, `DEBUG=false`).
4. **Frontend:** `VITE_API_URL=https://your-api.example.com npm run build`,
   then serve `frontend/dist/` from any static host.
5. **Verify:** `GET /health` must show `"status": "ok"`.

No free-tier assumptions exist in the code — free models are *example
configurations* documented here, swappable via one env var.

---

## 3. How abstention works (both gates)

**Gate 1 — retrieval gate (no LLM call, ever):**

```
top-1 cosine similarity  <  RETRIEVAL_SIMILARITY_THRESHOLD (default 0.20)
    → refuse immediately, with the measured scores in the reason
```

Runs inside `evaluate_gate()` (`services/retrieval.py`), shared verbatim by the
API and the eval harness. An empty result set also refuses.

**Gate 2 — generation gate:** every answer is validated against the chunks that
were *actually retrieved and owned by the asking user*:

- `[n]` markers outside the retrieved set are **stripped** (fabricated citations
  can never render — including mid-stream, sentence-by-sentence);
- a sentence whose citations are *entirely* fabricated is **dropped whole**;
- if no verifiable citation survives, the answer is **refused** (gate:
  `generation`);
- citations are re-checked for `user_id` ownership as defense in depth.

**Confidence heuristic:** `composite = 0.7·top-1 + 0.3·mean(top-k)`, labelled
High (≥ `CONFIDENCE_HIGH_THRESHOLD`, default 0.50) / Medium (≥ 0.20) / Low
(below, i.e. a thin top-1 margin with a weak tail) / Refused.

**Threshold ↔ refusal tradeoff (measured, not guessed):** on the golden set,
out-of-corpus questions score **0.04–0.09** while answerable evidence scores
**0.27–0.69**. At 0.20 the eval shows **100% coverage with 0% false answers**.
Raising the threshold toward 0.65 (an earlier guess) cut coverage to **10%**
(17/20 false refusals) while adding **zero** extra safety, because refusals
never approached it. Lowering below ~0.15 would erode the margin against
out-of-corpus questions. The threshold is the knob: higher → more refusals,
fewer answers; the eval is how you re-tune it after changing embedding models.

---

## 4. Evaluation methodology

**Golden set** — `backend/evals/dataset.json`, 25 cases:

- **4 fixture PDFs** (handbook, security policy, benefits, runbook) rendered
  deterministically and pushed through the *real* ingestion pipeline
  (extract → chunk → page metadata);
- **20 answerable** questions, each with an `expect_phrase` that must appear in
  the retrieved chunks (guards against dataset/corpus drift);
- **5 forced-refusal** cases (recipes, sports, medical, physics, translation) —
  topics with no corpus evidence, where an ungated LLM would confidently
  hallucinate. Each carries a written rationale.

**Metrics tracked** (`python -m evals.runner`, also asserted in
`tests/test_phase6_eval.py`):

| Metric | Current | Meaning |
|---|---|---|
| coverage | **100% (20/20)** | answerable ∧ gate passed ∧ evidence in top-k |
| top-1 rank quality | 95% | expected fact ranked first |
| false refusals | 0 | answerable cases wrongly refused |
| **false answers (safety)** | **0/5** | forced-refusal cases that passed the gate — must stay 0 |
| mean top-1 similarity | 0.409 | separation vs threshold 0.20 |

**Running it:** `pytest backend/tests` (full suite incl. the eval) or
`cd backend && python -m evals.runner` for the printable report. The harness
uses in-memory cosine mirroring pgvector similarity — deliberately **no DB and
no LLM involved**, because Gate 1's whole point is to refuse before spending a
token. Gate 2/citation integrity is covered by unit tests
(`test_phase4.py`, `test_phase5.py`).

---

## 5. Demo walkthrough

```bash
cd backend
python -m scripts.seed_demo       # creates demo user + ingests 4 PDFs (idempotent)
```

Then open the frontend, sign in as the seeded account, and show:

1. Documents tab — 4 ready PDFs (re-run the script: everything is skipped → idempotency).
2. Chat — an answerable question: streaming tokens, `[n]` chips → source panel
   with exact chunk text + page, confidence badge.
3. A refusal question ("What's a banana bread recipe?"): immediate refusal,
   `Refused` badge, gate `retrieval`, **no LLM call**.
4. A fabricated-citation demo (via unit tests) if recording CI evidence.

Demo credentials come from `SEED_EMAIL` / `SEED_PASSWORD` env
(defaults documented in the script) — change them for anything non-local.

---

## 6. Limitations (honest)

- **No OCR** — scanned/image-only PDFs are detected and refused with a clear
  message, never silently ingested.
- **Heuristic confidence** — a composite of similarity scores, *not* a
  calibrated probability; bands are tuned for local MiniLM and re-tunable per
  provider.
- **English-only** in v1; multi-language support is out of scope.
- **Text-block tables** — reading-order extraction (PyMuPDF blocks with
  pdfplumber fallback), not pixel-perfect table reconstruction.
- **No conversation memory** (by PRD): each question is answered from retrieval
  alone; history is stored for display only.
- No hybrid/BM25 search (pure pgvector v1), no OAuth (email/password JWT), no
  admin panel, original PDFs are not stored (only chunks + a SHA-256 for
  duplicate detection).
- Live free-tier deployment and a recorded demo video are operator steps; the
  code path is env-driven as described in §2.

---

## 7. Repository status

- Backend: FastAPI + SQLAlchemy(async) + asyncpg + pgvector; **46+ tests**
  (`cd backend && python -m pytest tests/`).
- Frontend: Vite + React + TypeScript + Tailwind (`npm run build` green).
- Checkpoints: `bc480de` (Phase 5), `bc3b15c` (Phase 6).
- PRD: see `PRD.md` expectations summarized above; config knobs live in
  `.env.example` with inline calibration notes.
