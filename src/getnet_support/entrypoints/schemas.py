"""Pydantic request/response contracts for the HTTP API boundary.

Converts transport payloads into domain/application types; internal exceptions and infrastructure
details never cross this boundary.
"""

from pydantic import BaseModel, ConfigDict, Field

from getnet_support.domain.models import ChatResult, Locale, Market


class ChatRequestBody(BaseModel):
    """Inbound `POST /chat` payload. Preserves the challenge's original contract."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(min_length=1, max_length=100)
    market: Market = Market.BR
    locale: Locale | None = None


class SourceView(BaseModel):
    """One cited source in the API response."""

    title: str
    url: str
    market: Market
    retrieved_at: str
    volatility: str


class ChatResponseBody(BaseModel):
    """`POST /chat` response, including execution metadata for observability."""

    answer: str
    sources: list[SourceView]
    route: str
    agents: list[str]
    tools: list[str]
    handoff_required: bool
    trace_id: str
    latency_ms: int

    @classmethod
    def from_result(cls, result: ChatResult) -> "ChatResponseBody":
        """Build the API response from the application-layer result."""
        return cls(
            answer=result.answer,
            sources=[
                SourceView(
                    title=source.title,
                    url=source.url,
                    market=source.market,
                    retrieved_at=source.retrieved_at,
                    volatility=source.volatility,
                )
                for source in result.sources
            ],
            route=result.route.value,
            agents=[agent.value for agent in result.agents],
            tools=list(result.tools),
            handoff_required=result.handoff_required,
            trace_id=result.trace_id,
            latency_ms=result.latency_ms,
        )
