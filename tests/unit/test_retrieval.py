"""Unit tests for the deterministic evidence gate (REQ-09/REQ-16)."""

import pytest

from getnet_support.adapters.retrieval.corpus_loader import load_corpus
from getnet_support.adapters.retrieval.lexical_retriever import LexicalRetriever
from getnet_support.adapters.retrieval.semantic_retriever import SemanticRetriever
from getnet_support.application.evidence_gate import accepts, best_accepted
from getnet_support.domain.chat import Language, Market, Volatility
from getnet_support.domain.knowledge import CorpusChunk, RetrievedChunk, coverage_lexical

SCORE_MIN = 0.1
COVERAGE_MIN = 0.55


@pytest.fixture(params=[LexicalRetriever, SemanticRetriever])
def retriever(request: pytest.FixtureRequest) -> LexicalRetriever | SemanticRetriever:
    """Both retrievers must satisfy the same evidence-gate contract (REQ-16)."""
    retriever_class: type[LexicalRetriever] | type[SemanticRetriever] = request.param
    return retriever_class(load_corpus())


def test_in_corpus_question_clears_the_gate(
    retriever: LexicalRetriever | SemanticRetriever,
) -> None:
    """A question covered by the corpus is accepted, for both retriever modes."""
    candidates = retriever.search("What's the difference between the Get Clássica and Get Smart?")
    assert best_accepted(candidates, score_min=SCORE_MIN, coverage_min=COVERAGE_MIN) is not None


def test_out_of_corpus_question_never_clears_the_gate(
    retriever: LexicalRetriever | SemanticRetriever,
) -> None:
    """REQ-10: a question with no corpus evidence is rejected before any LLM call."""
    candidates = retriever.search("Quem foi Maradona?")
    assert best_accepted(candidates, score_min=SCORE_MIN, coverage_min=COVERAGE_MIN) is None


def test_topic_words_present_but_scattered_across_chunks_still_reject(
    retriever: LexicalRetriever | SemanticRetriever,
) -> None:
    """REQ-06/09: 'tempo' and 'hoje' each appear in the corpus, in different
    chunks, but no single chunk covers the query — the gate must still reject.
    """
    candidates = retriever.search("Como está o tempo hoje?")
    assert best_accepted(candidates, score_min=SCORE_MIN, coverage_min=COVERAGE_MIN) is None


@pytest.mark.parametrize(
    "query",
    [
        "Qual a taxa de débito?",
        "Qual a taxa no débito?",
        "Quanto é a taxa de débito da Getnet?",
    ],
)
def test_singular_query_matches_a_chunk_that_only_has_the_plural(
    retriever: LexicalRetriever | SemanticRetriever, query: str
) -> None:
    """Regression: found via manual testing, not the eval dataset.

    `br-ofertas-pricing` only says "taxas" (plural); "taxa" (singular, 4
    chars) must still match it — both in coverage and in the retrieval
    score, since the gate requires both (REQ-09). Before the fix, this
    query fell through to a live web search and returned unrelated
    (non-Getnet) content instead of the corpus' own pricing chunk.
    """
    candidates = retriever.search(query)
    best = best_accepted(candidates, score_min=SCORE_MIN, coverage_min=COVERAGE_MIN)
    assert best is not None
    assert best.chunk.id == "br-ofertas-pricing"


def test_coverage_lexical_matches_across_languages_via_domain_glossary() -> None:
    """A Portuguese-glossed English query matches its Portuguese chunk."""
    chunk_text = "A antecipação de recebíveis permite receber o valor antes do prazo."
    coverage = coverage_lexical(
        "How does receivables advance (antecipação) work with Getnet?", chunk_text
    )
    assert coverage >= COVERAGE_MIN


def test_gate_requires_both_score_and_coverage() -> None:
    """REQ-09: score alone, or coverage alone, is not enough to accept."""
    chunk = CorpusChunk(
        id="x",
        text="irrelevant",
        title="irrelevant",
        source="https://example.com",
        market=Market.BR,
        language=Language.PT_BR,
        topic="x",
        retrieved_at="2026-01-01",
        volatility=Volatility.LOW,
    )
    high_score_low_coverage = RetrievedChunk(chunk=chunk, score_retrieval=0.9, coverage=0.1)
    low_score_high_coverage = RetrievedChunk(chunk=chunk, score_retrieval=0.01, coverage=0.9)
    assert not accepts(high_score_low_coverage, score_min=SCORE_MIN, coverage_min=COVERAGE_MIN)
    assert not accepts(low_score_high_coverage, score_min=SCORE_MIN, coverage_min=COVERAGE_MIN)
