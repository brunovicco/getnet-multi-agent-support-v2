"""Corpus and retrieval Value Objects and the deterministic evidence gate.

REQ-09: evidence acceptance is decided here, before any LLM call, by two
independent deterministic signals — a retrieval score and a lexical coverage
fraction. Neither signal is a topic/keyword classifier (REQ-06): coverage
only measures whether the query's own content words show up in one already
-retrieved chunk, it never decides whether a *topic* needs the web.
"""

import re
import unicodedata
from dataclasses import dataclass

from getnet_support.domain.chat import Language, Market, Volatility


@dataclass(frozen=True, slots=True)
class CorpusChunk:
    """One curated, committed knowledge-base passage (REQ-14)."""

    id: str
    text: str
    title: str
    source: str
    market: Market
    language: Language
    topic: str
    retrieved_at: str
    volatility: Volatility


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A corpus chunk scored against one query."""

    chunk: CorpusChunk
    score_retrieval: float
    coverage: float


_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_STEM_PREFIX_LEN = 5

_STOPWORDS = frozenset(
    {
        # Portuguese
        "a",
        "o",
        "os",
        "as",
        "de",
        "da",
        "do",
        "das",
        "dos",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "um",
        "uma",
        "uns",
        "umas",
        "e",
        "ou",
        "que",
        "qual",
        "quais",
        "quando",
        "como",
        "onde",
        "quem",
        "porque",
        "por",
        "para",
        "pra",
        "com",
        "sem",
        "sobre",
        "entre",
        "ate",
        "se",
        "sua",
        "seu",
        "suas",
        "seus",
        "eu",
        "voce",
        "ele",
        "ela",
        "esta",
        "estou",
        "sao",
        "ser",
        "tem",
        "tenho",
        "preciso",
        "precisa",
        "posso",
        "pode",
        "vou",
        "vai",
        "ir",
        "minha",
        "meu",
        "minhas",
        "meus",
        "dá",
        "gente",
        "eh",
        "é",
        "the",
        # English
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "is",
        "are",
        "does",
        "i",
        "my",
        "your",
        "you",
        "with",
        "what",
        "when",
        "how",
        "can",
        "will",
        "using",
        "use",
        "through",
        "into",
        "from",
        "this",
        "that",
        "these",
        "those",
        "need",
        "needed",
        "about",
        "should",
        "did",
        "have",
        "has",
    }
)

# Small hand-curated bilingual glossary of domain vocabulary, so lexical
# coverage can match an English query against the Portuguese-only corpus
# (REQ-14) without falling back to a topic classifier (REQ-06 forbids that
# for the *web-necessity* decision only — this table never decides whether
# to search the web, it only normalizes retrieval matching, same purpose as
# a search engine's synonym expansion).
_BILINGUAL_ALIASES = {
    "receivables": "recebiveis",
    "advance": "antecipacao",
    "installments": "parcela",
    "installment": "parcela",
    "split": "parcela",
    "credit": "credito",
    "bank": "bancaria",
    "account": "conta",
    "receive": "receber",
    "deposit": "recebimento",
    "connect": "conecta",
    "connection": "conexao",
    "machine": "maquininha",
    "terminal": "maquininha",
    "card": "cartao",
    "decline": "recusada",
    "declined": "recusada",
    "error": "erro",
    "transaction": "transacao",
    "payment": "pagamento",
    "sell": "vender",
    "sale": "venda",
    "sales": "venda",
    "fee": "taxa",
    "fees": "taxa",
    "price": "preco",
    "pricing": "preco",
    "plan": "plano",
}


def _strip_accents(token: str) -> str:
    """Remove diacritics so `"antecipação"` and `"antecipacao"` compare equal."""
    normalized = unicodedata.normalize("NFKD", token)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _canonicalize(token: str) -> str:
    """Normalize one token to a language-neutral form for lexical matching."""
    base = _strip_accents(token.lower())
    return _BILINGUAL_ALIASES.get(base, base)


def _stem(token: str) -> str:
    """Crude prefix stem tolerating PT/EN inflection (`parcelar`/`parcelas`)."""
    return token[:_STEM_PREFIX_LEN]


def content_terms(text: str) -> list[str]:
    """Extract canonicalized, non-stopword terms from free text."""
    return [
        canonical
        for raw in _TOKEN_RE.findall(text)
        for canonical in (_canonicalize(raw),)
        if canonical not in _STOPWORDS and len(canonical) > 2
    ]


def coverage_lexical(query: str, chunk_text: str) -> float:
    """REQ-09: fraction of non-stopword query terms present in one chunk."""
    query_terms = content_terms(query)
    if not query_terms:
        return 0.0
    chunk_stems = {_stem(term) for term in content_terms(chunk_text)}
    covered = sum(1 for term in query_terms if _stem(term) in chunk_stems)
    return covered / len(query_terms)
