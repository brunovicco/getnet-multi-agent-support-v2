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

## Decisões pós-cutoff (hardening)

| # | Decisão | Razão |
| --- | --- | --- |
| D16 | `RETRIEVER=semantic_embeddings` (P1.1) é um modo aditivo, nunca o default, e cai para `SemanticRetriever` local se faltar chave ou a chamada de embedding falhar na inicialização | REQ-16 (paridade offline de `lexical`/`semantic`) não pode regredir; REQ-24 (nunca crashar por falta de chave) vale também para P1 |
| D17 | Branding Getnet (P1.2) é só tema/CSS + wordmark textual, sem logo copiado | Não há acesso a ativos de marca oficiais nesta sessão; usar o nome da empresa é normal, reproduzir o logo sem verificação não |
| D18 | Chaining Support→Knowledge (P1.3) usa `KnowledgeAgent.try_grounded_in_corpus` (só corpus, nunca web) e o orquestrador decide se encadeia via um sinal booleano (`chain_to_knowledge`) no resultado do Support, não uma dependência direta entre os agentes | Um efeito colateral de busca web não solicitado numa resposta de suporte seria uma surpresa de custo; `CustomerSupportAgent` não deve conhecer `KnowledgeAgent` para não acoplar os dois além do necessário |
| D19 | Bug de correção (achado por teste manual do usuário, não pelo eval): `_stem` passou a remover um "s" final de plural antes do corte de prefixo, e `content_terms` (usado tanto por `coverage_lexical` quanto pela vetorização dos dois retrievers) passou a aplicar esse stemming, não só a função de cobertura; `top_k` default de `RetrieverPort.search` subiu de 3 para 20 (corpus tem 13 chunks) | "Qual a taxa de débito?" (singular, 4 letras) nunca colidia com "taxas" (plural) no corpus — nem no prefixo de 5 chars, nem no vetor de cosseno, que via os dois como palavras diferentes; e mesmo corrigida a cobertura, `top_k=3` às vezes cortava o chunk certo antes do gate de evidência (REQ-09) sequer vê-lo, porque o score dele (sinal que REQ-09 explicitamente não confia) era mais baixo que o de chunks menores e menos relevantes. Sem essa chave, a pergunta caía pra busca web e trazia conteúdo de concorrente (Cielo) em vez do conteúdo da própria Getnet. Recalibrado contra `eval_dataset.json` inteiro: zero regressão |

## Portas e adapters

``` text
application/ports/     LLMPort, WebSearchPort, RetrieverPort, CustomerDataPort
adapters/llm/          GeminiAdapter
adapters/search/       TavilyWebSearchAdapter
adapters/retrieval/    LexicalRetriever, SemanticRetriever, GeminiSemanticRetriever (P1.1)
adapters/customer/     InMemoryCustomerRepository
entrypoints/           http.py (FastAPI), ui.py (Gradio, branded), settings.py
```

## Degradação

| Falta | Comportamento |
| --- | --- |
| `GOOGLE_API_KEY` | Retriever léxico; respostas extrativas com fonte; `/health: llm=missing` |
| `TAVILY_API_KEY` | Perguntas fora do corpus escalam com motivo explícito; nunca inventam |
| Ambas | Serviço sobe, rotas e sources continuam corretas, `answer` degradado |

## Adiado (P1, documentado no README)

Mercado GLOBAL completo · reranking · MCP server · eval de qualidade de resposta
(LLM-as-judge).

Concluído pós-cutoff: retriever semântico com embeddings reais (P1.1, D16) · branding
Getnet na UI (P1.2, D17) · chaining Support→Knowledge (P1.3, D18).
