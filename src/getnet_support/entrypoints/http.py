"""FastAPI entrypoint and composition root.

Builds the process' single application service, wires it into both the HTTP
API and the Gradio UI (mounted at ``/``, REQ-04), and exposes the endpoints
required by REQ-01/02/03.
"""

from typing import Literal

import gradio as gr
import structlog
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from getnet_support.adapters.customer.in_memory_customer_repository import (
    InMemoryCustomerRepository,
)
from getnet_support.adapters.llm.gemini_adapter import GeminiAdapter
from getnet_support.adapters.retrieval.corpus_loader import load_corpus
from getnet_support.adapters.retrieval.gemini_semantic_retriever import GeminiSemanticRetriever
from getnet_support.adapters.retrieval.lexical_retriever import LexicalRetriever
from getnet_support.adapters.retrieval.semantic_retriever import SemanticRetriever
from getnet_support.adapters.search.tavily_web_search_adapter import TavilyWebSearchAdapter
from getnet_support.application.agents.customer_support_agent import CustomerSupportAgent
from getnet_support.application.agents.escalation_agent import EscalationAgent
from getnet_support.application.agents.knowledge_agent import KnowledgeAgent
from getnet_support.application.agents.router_agent import RouterAgent
from getnet_support.application.chat_service import ChatApplicationService
from getnet_support.application.dto import ChatResult
from getnet_support.application.errors import LLMUnavailableError
from getnet_support.application.ports.customer_data_port import CustomerDataPort
from getnet_support.application.ports.llm_port import LLMPort
from getnet_support.application.ports.retriever_port import RetrieverPort
from getnet_support.application.ports.web_search_port import WebSearchPort
from getnet_support.domain.knowledge import CorpusChunk
from getnet_support.entrypoints.logging import configure_logging
from getnet_support.entrypoints.settings import Settings
from getnet_support.entrypoints.ui import CUSTOM_CSS, build_theme, build_ui

_logger = structlog.get_logger(__name__)


def _build_retriever(
    settings: Settings, corpus: tuple[CorpusChunk, ...]
) -> tuple[RetrieverPort, str]:
    """Build the configured retriever, falling back safely if it can't start.

    Returns the retriever together with the mode actually running, since a
    fallback means it can differ from `settings.retriever` — `/health`
    reports the real one (REQ-03: diagnose actual capability, not intent).

    `semantic_embeddings` calls Gemini at construction time to embed the
    whole corpus (P1.1); if that fails or no key is configured, this falls
    back to the local, offline `SemanticRetriever` (T02/REQ-16) instead of
    crashing the process (REQ-24: degrade explicitly, never fabricate).
    """
    if settings.retriever == "semantic_embeddings":
        if not settings.llm_configured:
            _logger.warning(
                "retriever_fallback",
                configured="semantic_embeddings",
                used="semantic",
                reason="google_api_key_missing",
            )
            return SemanticRetriever(corpus), "semantic"
        try:
            retriever = GeminiSemanticRetriever(
                settings.google_api_key, corpus, timeout_seconds=settings.llm_timeout_seconds
            )
            return retriever, "semantic_embeddings"
        except LLMUnavailableError:
            _logger.warning(
                "retriever_fallback",
                configured="semantic_embeddings",
                used="semantic",
                reason="embedding_call_failed",
            )
            return SemanticRetriever(corpus), "semantic"
    if settings.retriever == "semantic":
        return SemanticRetriever(corpus), "semantic"
    return LexicalRetriever(corpus), "lexical"


class ChatRequestModel(BaseModel):
    """`POST /chat` request contract (REQ-01)."""

    model_config = ConfigDict(extra="ignore")

    message: str
    user_id: str
    market: Literal["BR", "GLOBAL"] | None = None
    locale: Literal["pt-BR", "en"] | None = None


class SourceModel(BaseModel):
    """One cited source in a chat response (REQ-02)."""

    title: str
    url: str
    origin: str
    retrieved_at: str
    volatility: str
    market: str | None = None


class ChatResponseModel(BaseModel):
    """`POST /chat` response contract (REQ-02)."""

    trace_id: str
    answer: str
    language: str
    route: str
    agents: list[str]
    tools: list[str]
    sources: list[SourceModel]
    handoff_required: bool
    grounding: str
    web_search_attempted: bool
    latency_ms: int


def _to_response(result: ChatResult) -> ChatResponseModel:
    """Translate the internal :class:`ChatResult` into the HTTP contract."""
    return ChatResponseModel(
        trace_id=result.trace_id,
        answer=result.answer,
        language=result.language.value,
        route=result.route.value,
        agents=result.agents,
        tools=result.tools,
        sources=[
            SourceModel(
                title=source.title,
                url=source.url,
                origin=source.origin.value,
                retrieved_at=source.retrieved_at,
                volatility=source.volatility.value,
                market=source.market.value if source.market else None,
            )
            for source in result.sources
        ],
        handoff_required=result.handoff_required,
        grounding=result.grounding.value,
        web_search_attempted=result.web_search_attempted,
        latency_ms=result.latency_ms,
    )


class HealthResponseModel(BaseModel):
    """`GET /health` capability report contract (REQ-03)."""

    status: Literal["ok"] = "ok"
    llm: Literal["configured", "missing"]
    web_search: Literal["configured", "missing"]
    retriever: Literal["lexical", "semantic", "semantic_embeddings"]
    corpus_chunks: int = Field(ge=0)


def build_app(settings: Settings | None = None) -> FastAPI:
    """Compose the process: configuration, application service, API, and UI."""
    settings = settings or Settings()
    configure_logging(service=settings.service_name, environment=settings.app_env, version="0.1.0")

    corpus = load_corpus()
    retriever, retriever_mode = _build_retriever(settings, corpus)
    llm: LLMPort | None = (
        GeminiAdapter(settings.google_api_key) if settings.llm_configured else None
    )
    web_search: WebSearchPort | None = (
        TavilyWebSearchAdapter(settings.tavily_api_key) if settings.web_search_configured else None
    )
    knowledge_agent = KnowledgeAgent(
        retriever=retriever,
        llm=llm,
        web_search=web_search,
        score_min=settings.score_min,
        coverage_min=settings.coverage_min,
        llm_timeout_seconds=settings.llm_timeout_seconds,
    )
    customer_data: CustomerDataPort = InMemoryCustomerRepository()
    customer_support_agent = CustomerSupportAgent(customer_data)
    escalation_agent = EscalationAgent()
    router = RouterAgent(confidence_min=settings.router_confidence_min)
    chat_service = ChatApplicationService(
        router=router,
        knowledge_agent=knowledge_agent,
        customer_support_agent=customer_support_agent,
        escalation_agent=escalation_agent,
    )
    corpus_chunk_count = len(corpus)

    app = FastAPI(title="Getnet Multi-Agent Support")

    @app.post("/chat", response_model=ChatResponseModel)
    def chat(payload: ChatRequestModel) -> ChatResponseModel:
        """Answer one chat message (REQ-01/02)."""
        result = chat_service.handle(
            message=payload.message,
            user_id=payload.user_id,
            market=payload.market,
            locale=payload.locale,
        )
        return _to_response(result)

    @app.get("/health", response_model=HealthResponseModel)
    def health() -> HealthResponseModel:
        """Report capability readiness, not just liveness (REQ-03)."""
        return HealthResponseModel(
            llm="configured" if settings.llm_configured else "missing",
            web_search="configured" if settings.web_search_configured else "missing",
            retriever=retriever_mode,
            corpus_chunks=corpus_chunk_count,
        )

    ui = build_ui(chat_service)
    gr.mount_gradio_app(app, ui, path="/", theme=build_theme(), css=CUSTOM_CSS)

    return app
