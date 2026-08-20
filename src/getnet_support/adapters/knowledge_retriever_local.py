"""Local RAG retriever: IDF-weighted term-overlap scoring over the persisted Getnet corpus.

No vector DB or embeddings here — a small, explainable pure-Python retrieval strategy. Always the
safe-degradation fallback when no embedding provider is configured (see
`knowledge_retriever_semantic.py` for the P1 upgrade). Retrieval always filters by market before
scoring, so BR and GLOBAL chunks never mix in one result set. Terms are weighted by inverse
document frequency so a brand word that appears in nearly every chunk (e.g. "Getnet") does not
outweigh a rare, discriminative term (e.g. "antecipação") when a query happens to share both.
"""

import math
import re
from pathlib import Path

from getnet_support.adapters.corpus_loader import load_corpus_chunks
from getnet_support.application.ports import KnowledgeRetrieverPort
from getnet_support.domain.models import KnowledgeChunk, Market, RetrievedChunk

_TOKEN_PATTERN = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)

# Common PT-BR/EN function words. Cross-language queries (e.g. an English question over the
# PT-BR corpus) otherwise pick up accidental stopword collisions ("via", "do", "a") as if they
# were real signal, drowning out the one or two domain terms that actually matter.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "do",
        "does",
        "to",
        "of",
        "in",
        "on",
        "at",
        "my",
        "i",
        "you",
        "your",
        "for",
        "and",
        "or",
        "via",
        "with",
        "can",
        "how",
        "what",
        "when",
        "will",
        "it",
        "this",
        "that",
        "be",
        "not",
        "no",
        "yes",
        "if",
        "as",
        "by",
        "from",
        "me",
        "we",
        "o",
        "os",
        "de",
        "da",
        "dos",
        "das",
        "em",
        "um",
        "uma",
        "para",
        "com",
        "se",
        "que",
        "é",
        "ao",
        "aos",
        "às",
        "na",
        "nos",
        "nas",
        "e",
        "ou",
        "por",
        "como",
        "quando",
        "não",
        "sim",
        "sua",
        "seu",
        "suas",
        "seus",
        "isso",
        "esse",
        "essa",
        "há",
        "hoje",
        "amanhã",
        "amanha",
        "ontem",
        "today",
        "tomorrow",
        "yesterday",
        "agora",
        "now",
        "tempo",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, stopword-filtered word tokenizer for PT-BR/EN corpus text."""
    return [
        token
        for token in (match.lower() for match in _TOKEN_PATTERN.findall(text))
        if token not in _STOPWORDS and len(token) > 1
    ]


def _build_term_index(
    chunks: tuple[KnowledgeChunk, ...],
) -> tuple[dict[str, frozenset[str]], dict[str, float]]:
    """Return each chunk's term set and each term's inverse document frequency weight."""
    term_sets = {chunk.id: frozenset(_tokenize(f"{chunk.title} {chunk.text}")) for chunk in chunks}
    document_frequency: dict[str, int] = {}
    for terms in term_sets.values():
        for term in terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    total_documents = max(len(chunks), 1)
    idf = {
        term: math.log((total_documents + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }
    return term_sets, idf


def _score(
    query_terms: frozenset[str], chunk_terms: frozenset[str], idf: dict[str, float]
) -> float:
    overlap = query_terms & chunk_terms
    if not overlap:
        return 0.0
    weight = sum(idf.get(term, 1.0) for term in overlap)
    return weight / math.sqrt(len(query_terms))


class LocalKnowledgeRetriever(KnowledgeRetrieverPort):
    """IDF-weighted term-overlap retriever over the persisted, market-scoped Getnet corpus."""

    def __init__(self, corpus_dir: Path | None = None) -> None:
        """Load and index the persisted corpus once, at construction time."""
        self._chunks = load_corpus_chunks(corpus_dir)
        self._term_sets, self._idf = _build_term_index(self._chunks)

    async def retrieve(
        self, query: str, *, market: Market, top_k: int = 3
    ) -> tuple[RetrievedChunk, ...]:
        """Return the top matching chunks scoped to market, best score first."""
        query_terms = frozenset(_tokenize(query))
        if not query_terms:
            return ()
        scored = [
            RetrievedChunk(
                chunk=chunk,
                score=_score(query_terms, self._term_sets[chunk.id], self._idf),
            )
            for chunk in self._chunks
            if chunk.market is market
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return tuple(scored[:top_k])
