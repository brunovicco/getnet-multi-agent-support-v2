"""Application-level result of handling one chat turn."""

from dataclasses import dataclass

from getnet_support.domain.chat import DecisionSource, GroundingOrigin, Language, Route, Source


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Everything the API and UI need to render one chat turn (REQ-02)."""

    trace_id: str
    answer: str
    language: Language
    route: Route
    agents: list[str]
    tools: list[str]
    sources: list[Source]
    handoff_required: bool
    grounding: GroundingOrigin
    web_search_attempted: bool
    latency_ms: int
    decision_source: DecisionSource
    classifier_latency_ms: int
