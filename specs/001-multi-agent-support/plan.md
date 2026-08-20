# Plan: Getnet Multi-Agent Support (001)

## Decisões de arquitetura

- **Orquestração explícita**: `ChatApplicationService` chama `RouterAgent` (regras determinísticas)
  que retorna uma `Route`; o serviço então invoca o(s) agente(s) correspondente(s) em sequência.
  Sem framework de orquestração (sem LangGraph).
- **UI = adapter**: Gradio Blocks é montado no mesmo processo FastAPI (`gr.mount_gradio_app`) e
  chama a mesma instância de `ChatApplicationService` que o endpoint HTTP — sem chamada HTTP
  interna entre UI e backend.
- **RAG local simples**: corpus JSON persistido em `adapters/corpus/`, retrieval por overlap de
  termos (TF puro em Python), sem vector DB. Evolução futura: embeddings via `EmbeddingPort`.
- **Tavily real**: `TavilyWebSearchAdapter` chama a REST API via `httpx` com timeout; usado só
  para perguntas correntes/externas identificadas pelo Router.
- **LLM via port**: `LLMPort` (Protocol em `application/ports.py`); `GeminiLLMAdapter` chama a
  REST API do Gemini via `httpx` (sem SDK, para manter a dependência mínima). Sem chave, um
  responder extrativo determinístico monta a resposta a partir dos chunks recuperados.
- **Customer tools in-memory**: `InMemoryCustomerDataAdapter` simula CRM + settlement + terminal
  management com dados fixos para `cliente1988` e `cliente2001`; qualquer outro `user_id` levanta
  `CustomerNotFoundError`, tratado pelo Escalation Agent.
- **Guardrails determinísticos** (`application/guardrails.py`): escopo de cliente pelo `user_id`
  da requisição, filtro de mercado antes do retrieval, bloqueio de operações financeiras
  state-changing (estorno, cancelamento, chargeback), e escalonamento quando não há evidência
  (nenhum chunk relevante e Tavily indisponível/não aplicável).
- **Falhas de provider**: qualquer exceção de rede/HTTP em Gemini ou Tavily é capturada no
  adapter, mapeada para um erro de aplicação, e tratada como fallback seguro — nunca propaga
  stack trace ao usuário nem derruba o processo.

## Sequência de tarefas

Ver `tasks.md`.

## Riscos residuais aceitos

- Corpus RAG é escrito à mão a partir do conteúdo público conhecido das páginas oficiais
  listadas (não há crawling ao vivo no timebox); cada chunk cita a URL real como fonte.
- Novas dependências de runtime (`fastapi`, `uvicorn`, `gradio`, `httpx`) passam pelo scan
  automatizado do `pip-audit` do quality gate, sem revisão manual aprofundada de licença.
- Detecção de idioma é heurística (stopwords/diacríticos), não uma biblioteca de NLP.
- Sem streaming, sem cache de resposta, sem persistência de conversa entre requisições.
