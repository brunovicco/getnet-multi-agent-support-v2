"""Gemini-embedding-backed semantic retriever (P1.1).

Real embeddings, replacing the local hashed vectorizer (`semantic_retriever.py`,
T02/REQ-16) when a `GOOGLE_API_KEY` is configured and `RETRIEVER=semantic_embeddings`.
Corpus embeddings are computed once, at construction, with
`task_type="RETRIEVAL_DOCUMENT"`; each query is embedded per call with
`task_type="RETRIEVAL_QUERY"` — asymmetric embeddings improve retrieval quality over
using the same task type for both sides.

Construction failure (bad key, network down) propagates to the composition root, which
falls back to the local `SemanticRetriever` (see `entrypoints/http.py`) rather than
crashing the process. A query-time failure degrades to "no candidates" instead of raising
into the request path, matching the rest of the Knowledge pipeline's honesty rule (REQ-13):
no accepted evidence just means the Knowledge Agent falls through to the web-search step.
"""

import random
import time
from math import sqrt

from google import genai
from google.genai import errors, types

from getnet_support.application.errors import LLMUnavailableError
from getnet_support.domain.chat import Market
from getnet_support.domain.knowledge import CorpusChunk, RetrievedChunk, coverage_lexical

_MODEL = "text-embedding-004"
_MAX_ATTEMPTS = 2
_BACKOFF_BASE_SECONDS = 0.2


def _cosine(vector_a: list[float], vector_b: list[float]) -> float:
    """Cosine similarity between two equal-length dense vectors."""
    dot = sum(a * b for a, b in zip(vector_a, vector_b, strict=True))
    norm_a = sqrt(sum(a * a for a in vector_a))
    norm_b = sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class GeminiSemanticRetriever:
    """`RetrieverPort` implementation backed by real Gemini embeddings."""

    def __init__(
        self, api_key: str, corpus: tuple[CorpusChunk, ...], *, timeout_seconds: float
    ) -> None:
        """Embed the whole corpus once; raises `LLMUnavailableError` on failure."""
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000))
        )
        self._corpus = corpus
        texts = [f"{chunk.title}\n{chunk.text}" for chunk in corpus]
        embeddings = self._embed(texts, task_type="RETRIEVAL_DOCUMENT")
        self._vectors = dict(zip((chunk.id for chunk in corpus), embeddings, strict=True))

    def search(
        self, query: str, *, market: Market | None = None, top_k: int = 20
    ) -> list[RetrievedChunk]:
        """Rank corpus chunks by cosine similarity of real embeddings."""
        candidates = [chunk for chunk in self._corpus if market is None or chunk.market == market]
        try:
            query_vector = self._embed([query], task_type="RETRIEVAL_QUERY")[0]
        except LLMUnavailableError:
            return []
        scored = [
            RetrievedChunk(
                chunk=chunk,
                score_retrieval=_cosine(query_vector, self._vectors[chunk.id]),
                coverage=coverage_lexical(query, f"{chunk.title} {chunk.text}"),
            )
            for chunk in candidates
        ]
        scored.sort(key=lambda item: item.score_retrieval, reverse=True)
        return scored[:top_k]

    def _embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        """Embed `texts` in one batch call, retrying only transient server errors."""
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._client.models.embed_content(
                    model=_MODEL,
                    # mypy [arg-type]: google-genai's overloaded `contents` union
                    # does not resolve cleanly for `list[str]`, though it is an
                    # explicitly supported input at runtime. No ticket: this is
                    # an SDK stub limitation, not project debt to schedule.
                    contents=texts,  # type: ignore[arg-type]
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                return [list(embedding.values or []) for embedding in response.embeddings or []]
            except errors.ServerError as exc:
                last_error = exc
                if attempt + 1 < _MAX_ATTEMPTS:
                    # S311/B311: retry jitter, not a cryptographic use of random;
                    # permanent, not tech debt, no ticket needed.
                    jitter = random.uniform(0, 0.1)  # noqa: S311  # nosec B311
                    time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt) + jitter)
            except errors.ClientError as exc:
                raise LLMUnavailableError("Gemini embedding request rejected") from exc
        raise LLMUnavailableError("Gemini embedding did not respond after retries") from last_error
