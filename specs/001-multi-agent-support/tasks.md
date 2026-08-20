# Tasks: Getnet Multi-Agent Support (001)

- **T01** Domain models + application ports (`domain/models.py`, `application/ports.py`)
- **T02** Router agent + Chat application service (orquestração)
- **T03** Customer support tools (in-memory adapter) + Customer Support Agent + Escalation Agent
- **T04** RAG local: corpus persistido (BR + GLOBAL) + retriever + Knowledge Agent (sem LLM ainda)
- **T05** LLM adapter (Gemini via REST) + fallback extrativo, plugado no Knowledge Agent
- **T06** Web search adapter (Tavily via REST) + fallback de indisponibilidade
- **T07** FastAPI entrypoint (`/`, `/health`, `POST /chat`, `/docs`) — composition root
- **T08** Gradio Blocks UI montada no FastAPI, mesmo `ChatApplicationService`
- **T09** Bilíngue (detecção de locale + templates PT/EN) e dimensão de mercado (isolamento BR/GLOBAL)
- **T10** Testes de aceitação, Dockerfile (CMD real), `.env.example`, `uv lock`, quality gate, README

## P1 (aplicado após T01–T10 completo e verde)

- **T11** RAG semântico: `EmbeddingPort` + `GeminiEmbeddingAdapter` (REST) + `SemanticKnowledgeRetriever`
  (cosine similarity, top_k=3, score mínimo 0.3); usado automaticamente com `GOOGLE_API_KEY`,
  com fallback seguro para o retriever lexical se o embedding falhar no startup
- **T12** Identidade visual Getnet no Gradio: tema `primary_hue="red"`, wordmark, painéis, labels
  bilíngues, badge de handoff, estado de erro
- **T13** Escalation Agent mais completo: mensagens por `EscalationReason` + canal de contato BR real
- **T14** Guardrail adicional: conteúdo recuperado (RAG/web) tratado como dado não confiável no
  prompt do LLM, nunca como instrução
- **T15** Dataset de avaliação mínimo (`evaluation/scenarios.json`, os 10 cenários oficiais do
  desafio) + teste de regressão parametrizado — encontrou e motivou a correção de um bug real de
  roteamento (frase em inglês sobre conectividade da maquininha)

Não aplicado (menor prioridade, adiado): reranking.
