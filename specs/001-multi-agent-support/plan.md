# PLAN 001 — decisões

## Arquitetura

```
Gradio Blocks ─┐
               ├─> ChatApplicationService ─> Orchestrator ─> RouterAgent
FastAPI /chat ─┘                                              ├─ KnowledgeAgent
                                                              ├─ CustomerSupportAgent
                                                              └─ EscalationAgent
```

Um único processo. UI e API compartilham o mesmo application service — nenhuma chamada HTTP
interna. Orquestração explícita por chamadas de função; sem LangGraph.

## Decisões

| # | Decisão | Razão |
|---|---|---|
| D1 | Orquestração direta, sem framework | 4 agentes não justificam abstração; mais fácil de depurar e de explicar em vídeo |
| D2 | Gradio montado no FastAPI | Um container, um comando, zero build Node |
| D3 | Retrieval léxico como default; semântico opt-in | Semântico só ativa com `GOOGLE_API_KEY`; paridade obrigatória (REQ-16) |
| D4 | Gate de evidência determinístico antes do LLM | Score denso não discrimina em corpus pequeno; economiza chamada de LLM |
| D5 | Sentinela do LLM é segunda rede, não a primeira | Groundedness é propriedade de segurança; LLM não é autoridade (REQ-08) |
| D6 | Tavily real via `WebSearchPort`; mock só em teste | Exigência do desafio; nunca fabricar informação atual |
| D7 | Gemini via `LLMPort`, Groq como fallback de implementação | Um provider por rodada; domínio não importa SDK |
| D8 | Corpus curado à mão e commitado | Boilerplate de navegação domina índice lexical; startup sem crawling |
| D9 | Tools de cliente in-memory, customer-scoped | Dados privados nunca vêm de LLM nem de RAG |
| D10 | `Settings` único ponto de composição, sem espelhar em `os.environ` | Evita mutação global; `configure_logging` recebe config |

## Portas e adapters

```
application/ports/     LLMPort, WebSearchPort, RetrieverPort, EmbeddingPort, CustomerDataPort
adapters/llm/          GeminiAdapter, GroqAdapter
adapters/search/       TavilyWebSearchAdapter
adapters/retrieval/    LexicalRetriever, SemanticRetriever
adapters/customer/     InMemoryCustomerRepository
entrypoints/           http.py (FastAPI), ui.py (Gradio), settings.py
```

## Degradação

| Falta | Comportamento |
|---|---|
| `GOOGLE_API_KEY` | Retriever léxico; respostas extrativas com fonte; `/health: llm=missing` |
| `TAVILY_API_KEY` | Perguntas fora do corpus escalam com motivo explícito; nunca inventam |
| Ambas | Serviço sobe, rotas e sources continuam corretas, `answer` degradado |

## Adiado (P1, documentado no README)

Mercado GLOBAL completo · chaining Support→Knowledge · reranking · MCP server ·
eval de qualidade de resposta (LLM-as-judge) · branding Getnet além do básico.
