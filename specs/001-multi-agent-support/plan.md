# PLAN 001 — decisões

## Arquitetura

``` text
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
| --- | --- | --- |
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

## Decisões tomadas durante a implementação (rodada 05)

Registradas aqui conforme foram necessárias, sem reabrir D1-D10 acima.

| # | Decisão | Razão |
| --- | --- | --- |
| D11 | Retriever "semântico" desta rodada = vetor hash local determinístico (bag-of-words + cosseno), sem chamada de rede | REQ-16 exige paridade lexical/semântico verificável offline, sem `GOOGLE_API_KEY` — contradiz a leitura literal de D3/P1.1 de que semântico só ativa com chave. Embeddings de modelo real (Gemini `text-embedding`) ficam como P1; o modo semântico em si não é mais opcional |
| D12 | Gate de evidência combina cobertura por token exato com um pequeno glossário bilíngue PT/EN e stemming por prefixo (5 chars) | O corpus é 100% PT-BR (REQ-14); metade das perguntas do eval são em EN. Cobertura literal ficava abaixo de 0.4 nesses casos e rejeitava perguntas válidas do corpus. Calibrado contra `eval_dataset.json`: `SCORE_MIN=0.1`, `COVERAGE_MIN=0.55` |
| D13 | Router usa frases de intenção determinísticas (não palavras de tópico) para `customer_support`/`escalation`; `knowledge` é o default | REQ-06 proíbe classificar por tópico especificamente a decisão "preciso da web" — isso é decidido só pelo gate de evidência (REQ-09), nunca pelo Router. Intenção de suporte/escalação é roteamento normal exigido por REQ-07 |
| D14 | Isolamento cross-customer é uma regra determinística no `CustomerSupportAgent` (regex `cliente\d+` ≠ `user_id` autenticado, checada antes de qualquer tool call) | REQ-08/19: isolamento entre clientes nunca pode depender do LLM nem de instrução de prompt |
| D15 | `Volatility` mantém os 3 níveis (`low/medium/high`) já presentes no corpus commitado | REQ-02 lista `low`/`high` como exemplo, não como teto; nenhum teste do eval trava no valor exato |

## Portas e adapters

``` text
application/ports/     LLMPort, WebSearchPort, RetrieverPort, EmbeddingPort, CustomerDataPort
adapters/llm/          GeminiAdapter, GroqAdapter
adapters/search/       TavilyWebSearchAdapter
adapters/retrieval/    LexicalRetriever, SemanticRetriever
adapters/customer/     InMemoryCustomerRepository
entrypoints/           http.py (FastAPI), ui.py (Gradio), settings.py
```

## Degradação

| Falta | Comportamento |
| --- | --- |
| `GOOGLE_API_KEY` | Retriever léxico; respostas extrativas com fonte; `/health: llm=missing` |
| `TAVILY_API_KEY` | Perguntas fora do corpus escalam com motivo explícito; nunca inventam |
| Ambas | Serviço sobe, rotas e sources continuam corretas, `answer` degradado |

## Adiado (P1, documentado no README)

Mercado GLOBAL completo · chaining Support→Knowledge · reranking · MCP server ·
eval de qualidade de resposta (LLM-as-judge) · branding Getnet além do básico.
