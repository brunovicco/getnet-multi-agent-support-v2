"""Deterministic evidence gate (REQ-09): decides KB acceptance before any LLM call.

`accepts` is the only place that turns a retrieval result into a "grounded"
decision. It never looks at topic or subject matter (REQ-06) — only at the
two scores a :class:`RetrieverPort` and `coverage_lexical` already computed.
"""

from getnet_support.domain.knowledge import RetrievedChunk


def accepts(candidate: RetrievedChunk, *, score_min: float, coverage_min: float) -> bool:
    """REQ-09: accept iff score_retrieval >= score_min AND coverage >= coverage_min."""
    return candidate.score_retrieval >= score_min and candidate.coverage >= coverage_min


def best_accepted(
    candidates: list[RetrievedChunk], *, score_min: float, coverage_min: float
) -> RetrievedChunk | None:
    """Return the highest-scoring candidate that clears the evidence gate."""
    accepted = [c for c in candidates if accepts(c, score_min=score_min, coverage_min=coverage_min)]
    if not accepted:
        return None
    return max(accepted, key=lambda item: item.score_retrieval)
