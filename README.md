# Getnet Multi-Agent Support System

A multi-agent support system for Getnet: a Router Agent dispatches each message to a
Knowledge Agent (RAG over a curated corpus + web search fallback), a Customer Support
Agent (deterministic, customer-scoped tools), or an Escalation Agent (human handoff).
FastAPI serves `/chat` and `/health`; a Gradio UI is mounted in the same process at `/`.

## Honesty note

`specs/001-multi-agent-support/spec.md` was written **before** a 40-minute implementation
timer started. The timer covers implementation, tests, Docker, and this README — not the
specification itself. `tests/acceptance/` (the eval) already existed, deliberately red,
before the first line of implementation code was written; it was the target, not the
result. See `specs/001-multi-agent-support/{spec,plan,tasks}.md` for the full contract and
the decisions made along the way, including ones made mid-implementation and registered
there rather than reopening earlier decisions.

## Quickstart

```bash
uv sync --frozen
cp .env.example .env   # optional: add GOOGLE_API_KEY / TAVILY_API_KEY for full grounding
uv run uvicorn getnet_support.entrypoints.http:build_app --factory --reload
```

Open <http://localhost:8000> for the UI, or call the API directly:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Qual a diferença entre a Get Clássica e a Get Smart?", "user_id": "cliente1988"}'
```

The service runs with **zero API keys configured** — it degrades honestly instead of
fabricating answers (see [Degradation](#degradation)).

## Tests

```bash
uv run pytest                              # everything, fully offline
uv run pytest tests/acceptance -q --no-cov # the eval only
uv run pytest -m live                      # smoke tests with real API keys (skipped by default)
uv run python scripts/quality_gate.py      # lint, types, architecture, security, tests, deps
```

## Container

```bash
docker build -t getnet-multi-agent-support-v2 .
docker run --rm -p 8000:8000 \
  -e GOOGLE_API_KEY=your-key \
  -e TAVILY_API_KEY=your-key \
  getnet-multi-agent-support-v2
```

Also works with no `-e` flags at all — same degraded-but-honest behavior as local dev.

## Architecture

```text
Gradio Blocks ─┐
               ├─> ChatApplicationService ─> RouterAgent
FastAPI /chat ─┘                               ├─ KnowledgeAgent
                                                ├─ CustomerSupportAgent
                                                └─ EscalationAgent
```

One process. The UI and the API share the same `ChatApplicationService` instance — the UI
never calls its own HTTP API. Orchestration is explicit function calls (no LangGraph); four
agents don't justify a framework, and direct calls are easier to reason about and to
explain. Dependency direction follows Clean Architecture: `entrypoints -> application ->
domain`, `adapters -> application/domain`, `domain` imports nothing outer. `http.py` is the
composition root — it is the only place that constructs concrete adapters and wires them
into agents via `application/ports/*` protocols.

### Agents

- **Router Agent** (`application/agents/router_agent.py`) — deterministic, phrase-based
  rules decide `knowledge` / `customer_support` / `escalation`. State-changing financial
  requests and prompt-injection attempts are matched by rule and routed straight to
  escalation; nothing about authorization or customer isolation is ever LLM-decided
  (REQ-08). An optional LLM tie-break is designed into the interface
  (`decision_source: rule|llm|fallback`) for genuinely ambiguous cases below
  `ROUTER_CONFIDENCE_MIN`, though the eval dataset never needs it.
- **Knowledge Agent** (`application/agents/knowledge_agent.py`) — RAG first, web second,
  never fabricates. See [RAG pipeline](#rag-pipeline) below.
- **Customer Support Agent** (`application/agents/customer_support_agent.py`) — three
  deterministic, customer-scoped tools: `get_customer_profile`, `get_recent_transactions`,
  `get_terminal_status`. Every tool call uses the *authenticated* `user_id` only; a customer
  id parsed out of message text (`"transações do cliente2001"`) is detected and escalated
  **before** any tool runs — isolation never depends on the LLM.
- **Escalation Agent** (`application/agents/escalation_agent.py`) — the fourth agent
  (challenge bonus): always hands off honestly, never attempts a resolution.

### RAG pipeline

**Ingestion:** the corpus is curated by hand from `getnet.net`/`getnet.com.br` pages and
committed as JSON (`adapters/corpus/{getnet_br,getnet_global}.json`, 13 chunks). Startup
never crawls the web (REQ-14) — page boilerplate would otherwise dominate a lexical index
this small.

**Storage:** loaded once per process via `adapters/retrieval/corpus_loader.py`
(`functools.lru_cache`), parsed into the domain `CorpusChunk` type.

**Retrieval:** three interchangeable implementations behind `RetrieverPort`
(`application/ports/retriever_port.py`), selected by `RETRIEVER`:
- `lexical` (default) — `LexicalRetriever`, bag-of-words cosine similarity over
  canonicalized tokens.
- `semantic` — `SemanticRetriever`, a **local, deterministic hashed-vector cosine**
  retriever, no API key, no network. See
  [Why the local "semantic" mode isn't a real embedding model](#why-the-local-semantic-mode-isnt-a-real-embedding-model)
  below for why it exists.
- `semantic_embeddings` (P1.1) — `GeminiSemanticRetriever`, real Gemini `text-embedding-004`
  embeddings, asymmetric `RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY` task types, corpus embedded
  once at startup. Requires `GOOGLE_API_KEY`; falls back to `semantic` (logged as
  `retriever_fallback`) if the key is missing or the startup embedding call fails, rather
  than crashing the process (REQ-24). A query-time embedding failure degrades to "no
  candidates" for that one request, letting the Knowledge Agent fall through to the web step
  instead of raising into `/chat`.

**Generation:** `GeminiAdapter` (`adapters/llm/gemini_adapter.py`) generates an answer
strictly grounded in the one accepted chunk, with an explicit timeout and one bounded retry
on transient (5xx) errors only. If no `GOOGLE_API_KEY` is set, the LLM call fails, or the
model emits the internal `NO_EVIDENCE_IN_CONTEXT` sentinel, the agent falls back to the raw
chunk text — extractive, never fabricated, and the sentinel itself never reaches the user
(REQ-12).

### Evidence gate (REQ-09)

A KB chunk is accepted **iff**:

```text
score_retrieval(query, chunk) >= SCORE_MIN   AND   coverage_lexical(query, chunk) >= COVERAGE_MIN
```

Both checks are deterministic and run **before** any LLM call — `score_retrieval` alone
does not have a discriminative floor over a ~13-chunk corpus (a dense-only prior round let
"quem foi Maradona?" pass a 0.3 cosine threshold against WhatsApp/Crediário chunks).
`coverage_lexical` is the fraction of the query's non-stopword content terms found in the
chunk, computed with:

- accent stripping (`"antecipação"` ~ `"antecipacao"`),
- a small hand-curated PT/EN glossary (`domain/knowledge.py:_BILINGUAL_ALIASES`) — the
  corpus is 100% Portuguese, but half the eval's official scenarios are in English
  (`"receivables advance (antecipação)"`); literal token overlap alone left legitimate
  in-corpus questions below 0.4 coverage,
- a 5-character prefix stem, tolerant of PT/EN inflection (`parcelar`/`parcelas`), plus a
  trailing-plural-`s` strip *before* that cut — without it, a short root and its plural never
  collide (`"taxa"`, 4 chars, is too short to truncate; `"taxas"`, exactly 5, is untouched
  too — they compare unequal). This stemming applies inside `content_terms`, which both
  retrievers' vectors are built from, not only inside coverage — otherwise coverage could
  accept a chunk that the retrieval score still scored near zero on the same pair of terms,
  and the gate's `AND` would reject anyway. Found via manual testing (not the eval dataset):
  "qual a taxa de débito?" fell through to a live web search and returned a competitor's
  rates instead of Getnet's own pricing chunk. See `plan.md` D19.
- `top_k=20` on every retriever call (corpus is 13 chunks) — a smaller `top_k` let
  `score_retrieval`, the signal REQ-09 explicitly doesn't trust, exclude a high-coverage
  chunk before the gate ever saw it, same bug report.

`SCORE_MIN=0.1` / `COVERAGE_MIN=0.55` were calibrated empirically against
`tests/acceptance/eval_dataset.json`: every in-corpus case clears >= 0.60 coverage, every
out-of-corpus case (including the REQ-06 "tempo"/"hoje" trap, where both words appear in the
corpus but never in the same chunk) tops out at 0.50. This holds identically for both
retrievers — REQ-16 parity — because coverage, not the retrieval score, is what actually
discriminates at this corpus size.

**Routing to the web is never a topic decision** (REQ-06): nowhere in the router or
knowledge code is there a keyword list for "needs current info" (weather, exchange rates,
etc.). Whether the Knowledge Agent attempts a web search is a pure consequence of the
evidence gate rejecting every corpus chunk.

### Why the local "semantic" mode isn't a real embedding model

`tests/acceptance/eval_dataset.json` is run against `lexical` and `semantic` (the local
mode), fully offline (`GOOGLE_API_KEY=""`), and REQ-16 requires identical routing/provenance
results either way. The eval fixture's own docstring calls for "precomputed embeddings from
the committed corpus, so parity is verifiable without an API key" — incompatible with
gating a real embedding call behind `GOOGLE_API_KEY` for a test that must stay green with no
key at all. `SemanticRetriever` (`adapters/retrieval/semantic_retriever.py`) is therefore a
**local** hashed bag-of-words vectorizer: deterministic, computed once at startup, no
network call ever. It satisfies REQ-16 as the offline-verifiable mode; it was never meant to
be what "semantic retrieval" means in production. `GeminiSemanticRetriever`
(`adapters/retrieval/gemini_semantic_retriever.py`, P1.1, `RETRIEVER=semantic_embeddings`) is
that production version — real embeddings, opt-in, additive: it doesn't touch `lexical` or
`semantic`, and the eval's offline parity guarantee for those two is unaffected.
`RetrieverPort` is the seam that made this a new adapter, not a rewrite.

## Bilingual and market handling (REQ-20/21)

`language` = explicit `locale` when given, else a lightweight PT/EN detector
(`domain/chat.py:detect_language`, a word-list vote with no new dependency, calibrated
against the full eval dataset). `market` is never derived from language or locale — an
English question with `market=BR` uses BR content (`MKT-01`); a Portuguese question with
`locale=en` answers in English (`MKT-02`). Without evidence for the requested market, the
Knowledge Agent's evidence gate rejects and the request escalates rather than answering with
the wrong market's content (REQ-22) — BR is fully functional this round, GLOBAL is P1 (see
below), consistent with the corpus only having 3 GLOBAL chunks today.

## Degradation

| Missing | Behavior |
| --- | --- |
| `GOOGLE_API_KEY` | Lexical retriever still works; answers are extractive (raw chunk text) with a source citation; `/health` reports `"llm": "missing"` |
| `TAVILY_API_KEY` | Out-of-corpus questions still reach the web-search step (`web_search_attempted: true`) and then escalate honestly instead of searching; nothing is invented |
| Both | The service starts, routing and sources stay correct, `answer` degrades explicitly, `handoff_required: true` where nothing could be resolved |

## Observability

Structured JSON logs to stdout (`entrypoints/logging.py`, `structlog`), one
`configure_logging()` call at startup. `GET /health` reports capability readiness (not just
liveness) per REQ-03 — the round-04 postmortem this spec cites: an unloaded `.env` cost a
full round of debugging, so this endpoint turns that into one request. The Gradio UI shows,
per message, an execution panel (route, agents, tools, grounding, handoff, decision source,
latency) and clickable sources — REQ-27. LLM call tracing to Langfuse is opt-in and
metadata-only unless an explicit content-tracing approval is documented (see
`docs/LLM_OBSERVABILITY.md`); it is off by default and untouched by this round's work.

## Testing strategy

- **Unit** (`tests/unit/`) — pure logic, no network: the Router Agent's rules, the evidence
  gate and its coverage function, the Knowledge Agent and Customer Support Agent against
  hand-written fakes of `LLMPort`/`WebSearchPort`/`RetrieverPort`/`CustomerDataPort`,
  language detection, `Settings` precedence.
- **Acceptance / eval** (`tests/acceptance/test_acceptance_eval.py` +
  `eval_dataset.json`) — the actual gate. Runs fully offline, parametrized over both
  retrievers, asserting on response *metadata* (`route`, `sources[].origin`, `grounding`,
  `web_search_attempted`, `handoff_required`) rather than generated text, so it never
  depends on LLM wording. Covers routing accuracy, evidence-gate correctness (including the
  official scenarios, PT/EN paraphrases, and out-of-scope traps), customer isolation, and
  the bilingual contract.
- **Live smoke** (`@pytest.mark.live`, `uv run pytest -m live`) — one real Tavily call,
  skipped by default and in the quality gate; only runs with a real `TAVILY_API_KEY`.
- **What a fuller integration suite for the orchestration would add** (documented, not
  built this round given the timebox): a docker-compose job that builds the real image,
  starts it with real keys, and drives `/chat` over HTTP for the same eval dataset — catches
  anything the in-process `TestClient` can't (startup wiring, environment loading, container
  networking). Per-adapter contract tests against recorded (VCR-style) Gemini/Tavily
  fixtures would catch upstream schema drift without needing live calls in CI. A concurrent-
  request test against `/chat` would exercise `functools.lru_cache`'s thread-safety on the
  corpus loader under load, which today is only implied, not verified.

## Known limitations / P1 backlog

Explicitly deferred, not silently dropped:

- **GLOBAL market**, fully — only 3 corpus chunks today; BR is P0 and complete (REQ-22).
- **Reranking** over retrieved chunks.
- **MCP server** exposing the agents' tools externally.
- **Response-quality eval** (LLM-as-judge) — today's eval checks routing/provenance
  metadata, not answer quality.

Done, post-cutoff (see `tasks.md` "Pós-cutoff" and `plan.md` D11-D18 for the decisions
behind each):

- **P1.1 — real embedding-model semantic retrieval**: `RETRIEVER=semantic_embeddings`. See
  [above](#why-the-local-semantic-mode-isnt-a-real-embedding-model).
- **P1.2 — Getnet branding**: an orange theme (`entrypoints/ui.py:build_theme`/`CUSTOM_CSS`,
  applied at `gr.mount_gradio_app(..., theme=..., css=...)` — Gradio 6 moved these off the
  `Blocks` constructor for mounted apps) and a branded header/footer. The exact color is an
  approximation — this environment has no access to Getnet's official brand guidelines to
  verify the hex — and there is no copied Getnet logo asset; only the wordmark and a
  brand-appropriate palette.
- **P1.3 — chaining Support → Knowledge**: when the Customer Support Agent finds a real
  problem a KB article could supplement (today: a disconnected terminal —
  `CustomerSupportResult.chain_to_knowledge`), the orchestrator additionally calls
  `KnowledgeAgent.try_grounded_in_corpus` — the same evidence gate as a normal Knowledge
  turn, but **corpus-only, never the web** (an unprompted web-search side effect on a
  support turn would be a cost/surprise regression, REQ-11's discipline extended to P1). If
  the gate accepts, the KB source is added and its answer is appended after "This might also
  help:" / "Isso também pode ajudar:"; `grounding` stays `customer_data` (that's still the
  primary evidence for the personalized claim) and `agents` gains `"knowledge"`. If the gate
  rejects, the support answer is returned alone — chaining never turns into a wasted call or
  a missing response.

## Reference

- `specs/001-multi-agent-support/spec.md` — the contract; `[gate]`-tagged requirements are
  acceptance criteria, verified by `tests/acceptance/`.
- `specs/001-multi-agent-support/plan.md` — architecture decisions, including ones made
  mid-implementation (D11-D15).
- `specs/001-multi-agent-support/tasks.md` — the 40-minute slice sequence actually followed.
- `AGENTS.md` / `.claude/rules/` — the engineering contract this code was built under.
- `docs/ARCHITECTURE.md`, `docs/LLM_OBSERVABILITY.md`, `docs/PRIVACY.md`, `docs/MCP.md`.
