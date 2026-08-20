# Spec: Getnet Multi-Agent Support (001)

## Contexto

Sistema multi-agente que responde perguntas sobre produtos Getnet, dados de clientes e informação
externa atual, via API HTTP e UI Gradio, em PT-BR e EN. Desafio técnico "AI Hardcore Engineer",
timebox de 40 minutos, escopo P0 apenas.

## Requisitos

- **REQ-001** `POST /chat` aceita `{"message": str, "user_id": str}` (contrato original preservado)
  e aceita opcionalmente `market` (`BR`|`GLOBAL`, default `BR`) e `locale` (`pt-BR`|`en`, default
  detectado). Retorna resposta, `sources`, e metadados de execução (`route`, `agents`, `tools`,
  `handoff_required`, `latency_ms`, `trace_id`).
- **REQ-002** `GET /health` retorna status 200 com corpo indicando disponibilidade.
- **REQ-003** `GET /` serve a UI Gradio montada no mesmo processo FastAPI.
- **REQ-004** `GET /docs` disponível (OpenAPI automático do FastAPI).
- **REQ-005** Router Agent classifica cada mensagem deterministicamente entre Knowledge, Customer
  Support e Escalation; LLM pode auxiliar apenas em casos ambíguos e nunca decide autorização,
  guardrails, dados de cliente, isolamento de mercado ou operação financeira.
- **REQ-006** Knowledge Agent responde perguntas de produto via RAG local sobre um corpus oficial
  persistido no repositório, citando `sources` reais.
- **REQ-007** Knowledge Agent responde perguntas atuais/externas (clima, câmbio, etc.) via Tavily
  real quando `TAVILY_API_KEY` está configurada; sem chave, informa indisponibilidade e nunca
  inventa o dado.
- **REQ-008** Customer Support Agent expõe ao menos duas tools determinísticas e sempre escopadas
  por `user_id` da requisição (`get_customer_profile`, `get_recent_transactions`,
  `get_terminal_status`); nunca aceita `user_id` vindo do texto da mensagem.
- **REQ-009** Escalation Agent retorna `handoff_required=true` para usuário desconhecido, evidência
  insuficiente, operação financeira não suportada (ex.: estorno, cancelamento) ou pedido sensível.
- **REQ-010** Fluxo combinado: problema de terminal ("maquininha sem internet") aciona Customer
  Support (perfil + status do terminal) seguido de Knowledge (troubleshooting via RAG), com
  resposta única combinando os dois.
- **REQ-011** UI Gradio permite enviar mensagem, escolher `user_id` fake, idioma, mercado, nova
  conversa, e visualizar resposta, sources, agentes/tools usados, handoff e latência.
- **REQ-012** Respostas respeitam `locale` (PT-BR ou EN) explícito ou detectado da pergunta;
  idioma e mercado são dimensões independentes (EN + `market=BR` deve usar conteúdo brasileiro).
- **REQ-013** Corpus RAG inclui obrigatoriamente `https://site.getnet.com.br/ofertas/` marcado
  `volatility=high` com `retrieved_at`; conteúdo volátil nunca é apresentado como garantido.
- **REQ-014** Retrieval nunca mistura chunks de mercados diferentes: `market=BR` não recupera
  chunks `GLOBAL` e vice-versa.
- **REQ-015** `LLMPort` desacopla o núcleo do provider; `application`/`domain` não importam SDK do
  provider. Sem `GOOGLE_API_KEY`, a aplicação continua funcional com fallback extrativo
  determinístico — nunca fabrica resposta.
- **REQ-016** Dockerfile builda e executa a aplicação com `docker run` expondo a porta 8000,
  aceitando as env vars de provider via `-e`.
- **REQ-017** Testes automatizados cobrem os fluxos centrais (API, RAG, Tavily indisponível,
  tools de cliente, usuário desconhecido, PT, EN) sem rede real.

## Fora de escopo (P0)

RAG semântico com embeddings, vector DB, branding visual Getnet, streaming, multi-provider
routing, reranking, dataset de avaliação, dashboards externos — ver `plan.md`.
