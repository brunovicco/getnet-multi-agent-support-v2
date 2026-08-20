"""Knowledge Agent: evidence-gated RAG with an honest web fallback.

The evidence gate (REQ-09) runs before any LLM call. Only a chunk that
clears it is ever handed to the LLM for grounding; anything else falls
through to a web-search attempt, never to a fabricated answer (REQ-12/13).
"""

from dataclasses import dataclass

from getnet_support.application.errors import LLMUnavailableError, WebSearchUnavailableError
from getnet_support.application.evidence_gate import best_accepted
from getnet_support.application.ports.llm_port import LLMPort
from getnet_support.application.ports.retriever_port import RetrieverPort
from getnet_support.application.ports.web_search_port import WebSearchPort
from getnet_support.domain.chat import (
    GroundingOrigin,
    Language,
    Market,
    Source,
    SourceOrigin,
    Volatility,
)
from getnet_support.domain.knowledge import RetrievedChunk

SENTINEL = "NO_EVIDENCE_IN_CONTEXT"

_NO_EVIDENCE_MESSAGE = {
    Language.PT_BR: (
        "Não encontrei essa informação na base de conhecimento da Getnet nem em uma "
        "busca externa confiável agora. Um atendente humano vai continuar essa conversa."
    ),
    Language.EN: (
        "I could not find this in Getnet's knowledge base or in a reliable external "
        "search right now. A human agent will follow up on this conversation."
    ),
}

_VOLATILE_SUFFIX = {
    Language.PT_BR: " Valores e condições podem mudar; confirme no site oficial da Getnet.",
    Language.EN: " Prices and terms may change; please confirm on Getnet's official site.",
}


@dataclass(frozen=True, slots=True)
class KnowledgeResult:
    """What the Knowledge Agent produced for one query."""

    answer: str
    sources: list[Source]
    grounding: GroundingOrigin
    web_search_attempted: bool
    handoff_required: bool


class KnowledgeAgent:
    """RAG-first, evidence-gated, web-fallback knowledge answering."""

    def __init__(
        self,
        *,
        retriever: RetrieverPort,
        llm: LLMPort | None,
        web_search: WebSearchPort | None,
        score_min: float,
        coverage_min: float,
        llm_timeout_seconds: float,
    ) -> None:
        """Wire the retriever and the optional, key-gated external services."""
        self._retriever = retriever
        self._llm = llm
        self._web_search = web_search
        self._score_min = score_min
        self._coverage_min = coverage_min
        self._llm_timeout_seconds = llm_timeout_seconds

    def answer(self, query: str, *, market: Market | None, language: Language) -> KnowledgeResult:
        """Answer `query`, grounded in the corpus first, the web second."""
        candidates = self._retriever.search(query, market=market)
        accepted = best_accepted(
            candidates, score_min=self._score_min, coverage_min=self._coverage_min
        )
        if accepted is not None:
            return self._answer_from_kb(query, accepted, language=language)
        return self._answer_from_web(query, language=language)

    def _answer_from_kb(
        self, query: str, candidate: RetrievedChunk, *, language: Language
    ) -> KnowledgeResult:
        """REQ-10/11/15: ground the answer in the accepted corpus chunk."""
        chunk = candidate.chunk
        source = Source(
            title=chunk.title,
            url=chunk.source,
            origin=SourceOrigin.GETNET_KB,
            retrieved_at=chunk.retrieved_at,
            volatility=chunk.volatility,
            market=chunk.market,
        )
        answer = self._generate_or_extract(query, chunk.text, language=language)
        if chunk.volatility is Volatility.HIGH:
            answer += _VOLATILE_SUFFIX[language]
        return KnowledgeResult(
            answer=answer,
            sources=[source],
            grounding=GroundingOrigin.GETNET_KB,
            web_search_attempted=False,
            handoff_required=False,
        )

    def _answer_from_web(self, query: str, *, language: Language) -> KnowledgeResult:
        """REQ-10/13: no corpus evidence — attempt the web, then degrade honestly."""
        results = []
        if self._web_search is not None:
            try:
                results = self._web_search.search(query, timeout_seconds=self._llm_timeout_seconds)
            except WebSearchUnavailableError:
                results = []

        if not results:
            return KnowledgeResult(
                answer=_NO_EVIDENCE_MESSAGE[language],
                sources=[],
                grounding=GroundingOrigin.NONE,
                web_search_attempted=True,
                handoff_required=True,
            )

        sources = [
            Source(
                title=result.title,
                url=result.url,
                origin=SourceOrigin.WEB,
                retrieved_at=result.retrieved_at,
                volatility=Volatility.HIGH,
            )
            for result in results
        ]
        context = "\n\n".join(result.snippet for result in results)
        answer = self._generate_or_extract(query, context, language=language)
        return KnowledgeResult(
            answer=answer,
            sources=sources,
            grounding=GroundingOrigin.WEB,
            web_search_attempted=True,
            handoff_required=False,
        )

    def _generate_or_extract(self, query: str, context: str, *, language: Language) -> str:
        """Ground `query` in `context` via the LLM, or fall back to extraction.

        REQ-12: the internal sentinel never reaches the user — if the model
        emits it, or fails, or is not configured, the raw context stands in.
        """
        if self._llm is None:
            return context
        prompt = self._build_prompt(query, context, language=language)
        try:
            generated = self._llm.generate(prompt=prompt, timeout_seconds=self._llm_timeout_seconds)
        except LLMUnavailableError:
            return context
        if not generated or SENTINEL in generated:
            return context
        return generated

    def _build_prompt(self, query: str, context: str, *, language: Language) -> str:
        """Build a strictly grounded generation prompt."""
        language_name = "Brazilian Portuguese" if language is Language.PT_BR else "English"
        return (
            f"Answer the user's question in {language_name}, grounded ONLY in the context "
            f"below. If the context does not answer the question, output exactly the token "
            f"{SENTINEL} and nothing else. Never invent facts.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )
