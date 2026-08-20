"""Behavior tests for deterministic guardrails."""

from getnet_support.application.guardrails import (
    has_sufficient_knowledge_evidence,
    is_unsupported_financial_operation,
)
from getnet_support.domain.models import KnowledgeChunk, Locale, Market, RetrievedChunk


def _chunk(chunk_id: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=chunk_id,
        text="text",
        title="title",
        source="https://example.test",
        market=Market.BR,
        language=Locale.PT_BR,
        topic="topic",
        retrieved_at="2026-08-19",
        volatility="low",
    )


def test_detects_unsupported_financial_operation_in_portuguese() -> None:
    assert is_unsupported_financial_operation("Quero cancelar a conta e estornar tudo.")


def test_detects_unsupported_financial_operation_in_english() -> None:
    assert is_unsupported_financial_operation("I want to dispute a charge and get a refund.")


def test_does_not_flag_ordinary_product_question() -> None:
    assert not is_unsupported_financial_operation(
        "Qual a diferença entre Get Clássica e Get Smart?"
    )


def test_evidence_insufficient_when_all_scores_below_threshold() -> None:
    chunks = (RetrievedChunk(chunk=_chunk("a"), score=0.0),)
    assert not has_sufficient_knowledge_evidence(chunks)


def test_evidence_sufficient_when_one_score_clears_threshold() -> None:
    chunks = (
        RetrievedChunk(chunk=_chunk("a"), score=0.0),
        RetrievedChunk(chunk=_chunk("b"), score=0.5),
    )
    assert has_sufficient_knowledge_evidence(chunks)
