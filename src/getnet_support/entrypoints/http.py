"""FastAPI entrypoint: composition root for the Getnet multi-agent support service.

Wires adapters into application ports, builds the shared `ChatApplicationService`, exposes the
HTTP API, and mounts the Gradio UI on the same process — the UI and the API call the same service
instance, with no internal HTTP hop between them.
"""

import os
from typing import cast

import gradio as gr
from fastapi import FastAPI

from getnet_support.adapters.customer_data_memory import InMemoryCustomerDataAdapter
from getnet_support.adapters.knowledge_retriever_local import LocalKnowledgeRetriever
from getnet_support.adapters.llm_gemini import DEFAULT_MODEL, GeminiLLMAdapter
from getnet_support.adapters.web_search_tavily import TavilyWebSearchAdapter
from getnet_support.application.chat_service import ChatApplicationService
from getnet_support.application.customer_support_agent import CustomerSupportAgent
from getnet_support.application.escalation_agent import EscalationAgent
from getnet_support.application.knowledge_agent import KnowledgeAgent
from getnet_support.application.ports import LLMGenerationError, LLMPort
from getnet_support.application.router_agent import RouterAgent
from getnet_support.domain.models import Locale
from getnet_support.entrypoints.locale_detection import detect_locale
from getnet_support.entrypoints.logging import configure_logging
from getnet_support.entrypoints.schemas import ChatRequestBody, ChatResponseBody
from getnet_support.entrypoints.ui_gradio import build_blocks

_SERVICE_NAME = os.environ.get("SERVICE_NAME", "getnet-multi-agent-support")

configure_logging(
    service=_SERVICE_NAME,
    environment=os.environ.get("APP_ENV", "development"),
    version="0.1.0",
)


class _NullLLMAdapter(LLMPort):
    """Used when no LLM provider is configured; always defers callers to the extractive fallback."""

    async def generate(self, *, system_prompt: str, user_prompt: str, locale: Locale) -> str:
        """Always raise, forcing the caller onto the deterministic extractive fallback."""
        raise LLMGenerationError("no LLM provider configured")


def _build_llm() -> LLMPort:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return _NullLLMAdapter()
    return GeminiLLMAdapter(api_key=api_key, model=os.environ.get("GOOGLE_MODEL", DEFAULT_MODEL))


def build_chat_service() -> ChatApplicationService:
    """Wire every adapter into the shared application service."""
    return ChatApplicationService(
        router=RouterAgent(),
        knowledge_agent=KnowledgeAgent(
            retriever=LocalKnowledgeRetriever(),
            web_search=TavilyWebSearchAdapter(api_key=os.environ.get("TAVILY_API_KEY")),
            llm=_build_llm(),
        ),
        customer_support_agent=CustomerSupportAgent(InMemoryCustomerDataAdapter()),
        escalation_agent=EscalationAgent(),
    )


def create_app(chat_service: ChatApplicationService | None = None) -> FastAPI:
    """Build the FastAPI application: API routes first, then the mounted Gradio UI at `/`."""
    service = chat_service or build_chat_service()
    fastapi_app = FastAPI(title="Getnet Multi-Agent Support", version="0.1.0")

    @fastapi_app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness/readiness probe."""
        return {"status": "ok"}

    @fastapi_app.post("/chat", response_model=ChatResponseBody)
    async def chat(body: ChatRequestBody) -> ChatResponseBody:
        """Route and answer one chat message through the shared application service."""
        locale = body.locale or detect_locale(body.message)
        result = await service.handle(
            message=body.message, user_id=body.user_id, market=body.market, locale=locale
        )
        return ChatResponseBody.from_result(result)

    return cast(FastAPI, gr.mount_gradio_app(fastapi_app, build_blocks(service), path="/"))


app = create_app()
