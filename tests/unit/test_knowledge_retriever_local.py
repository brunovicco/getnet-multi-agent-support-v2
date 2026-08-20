"""Behavior tests for the local RAG retriever, including market isolation."""

from getnet_support.adapters.knowledge_retriever_local import LocalKnowledgeRetriever
from getnet_support.domain.models import Market


def test_br_query_never_returns_global_chunks() -> None:
    retriever = LocalKnowledgeRetriever()
    results = retriever.retrieve("Getnet products and payments", market=Market.BR, top_k=10)
    assert all(item.chunk.market is Market.BR for item in results)


def test_global_query_never_returns_br_chunks() -> None:
    retriever = LocalKnowledgeRetriever()
    results = retriever.retrieve("Getnet produtos e taxas", market=Market.GLOBAL, top_k=10)
    assert all(item.chunk.market is Market.GLOBAL for item in results)


def test_retrieves_relevant_chunk_for_product_comparison_question() -> None:
    retriever = LocalKnowledgeRetriever()
    results = retriever.retrieve(
        "Qual a diferença entre Get Clássica e Get Smart?", market=Market.BR, top_k=3
    )
    assert results
    assert results[0].score > 0
    titles = {item.chunk.title for item in results}
    assert titles & {"Get Clássica", "Get Smart", "Ofertas Getnet — mensalidades e taxas"}


def test_unrelated_query_scores_zero() -> None:
    retriever = LocalKnowledgeRetriever()
    results = retriever.retrieve("xyzzy unrelated gibberish nonsense", market=Market.BR, top_k=3)
    assert all(item.score == 0.0 for item in results)


def test_ofertas_chunk_is_marked_high_volatility() -> None:
    retriever = LocalKnowledgeRetriever()
    results = retriever.retrieve("ofertas taxas mensalidade", market=Market.BR, top_k=10)
    ofertas = next(item for item in results if item.chunk.id == "br-ofertas-pricing")
    assert ofertas.chunk.volatility == "high"
    assert ofertas.chunk.retrieved_at
