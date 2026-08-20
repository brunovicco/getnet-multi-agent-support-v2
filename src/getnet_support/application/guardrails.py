"""Deterministic guardrails.

These checks are never delegated to an LLM: authorization, unsupported financial operations, and
evidence sufficiency are decided by code, not by model output.
"""

import re

from getnet_support.domain.models import RetrievedChunk

MIN_RETRIEVAL_SCORE = 0.05

_FINANCIAL_OPERATION_PATTERN = re.compile(
    r"\b("
    r"estorn\w*|cancelar\s+(a\s+)?conta|encerrar\s+(a\s+)?conta|contestar\s+(a\s+)?compra|"
    r"chargeback|refund|dispute\s+(a\s+)?charge|cancel\s+my\s+account|close\s+my\s+account|"
    r"reverse\s+the\s+payment"
    r")\b",
    re.IGNORECASE,
)


def is_unsupported_financial_operation(message: str) -> bool:
    """Return True when the message asks for a financial operation we do not support."""
    return bool(_FINANCIAL_OPERATION_PATTERN.search(message))


def has_sufficient_knowledge_evidence(chunks: tuple[RetrievedChunk, ...]) -> bool:
    """Return True when at least one retrieved chunk clears the minimum relevance score."""
    return any(item.score >= MIN_RETRIEVAL_SCORE for item in chunks)
