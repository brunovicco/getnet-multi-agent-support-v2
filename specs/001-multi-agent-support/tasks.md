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
