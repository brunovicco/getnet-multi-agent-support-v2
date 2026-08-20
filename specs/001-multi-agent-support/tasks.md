# TASKS 001 — 40 minutos

O eval já existe em `tests/acceptance/` antes do T01. Ele é o alvo, não o resultado.
Regra de corte: se um slice estourar sua janela, entregue o anterior verde e siga.

| ID | Slice | Janela | REQs | Pronto quando |
|---|---|---|---|---|
| T01 | Contratos + `Settings` + `/health` de capacidade | 00–04 | 01,02,03,23,24 | `test_contract_*` verde |
| T02 | Corpus curado + retriever léxico + gate de evidência | 04–12 | 09,14,15 | `test_eval_dataset_lexical` verde |
| T03 | Router + orchestrator | 12–18 | 05,06,07,08 | acurácia de rota 1.00 |
| T04 | Knowledge Agent + `LLMPort` + Gemini | 18–24 | 10,11,12,13 | `test_no_evidence_*` verde |
| T05 | Tavily real via `WebSearchPort` | 24–27 | 06,10,24 | fora de escopo → `route=web` |
| T06 | Customer tools + Escalation | 27–31 | 17,18,19 | `test_customer_*` verde |
| T07 | UI Gradio com painel de execução | 31–35 | 04,27 | 5 cenários de demo passam na UI |
| T08 | Bilíngue PT/EN | 35–37 | 20,21 | `test_language_*` verde |
| T09 | Docker + quality gate | 37–39 | 25 | `docker run` responde `/health` |
| T10 | README mínimo + cutoff | 39–40 | — | tag `challenge-40m` |

## Pós-cutoff (rodada de hardening, commits após a tag)

P1.1 retriever semântico — **só entra com REQ-16 (paridade) verde**
P1.2 branding Getnet na UI
P1.3 chaining Support→Knowledge ("maquininha não conecta")

## Cenários de demonstração (vídeo)

1. `Qual a diferença entre Get Clássica e Get Smart?` → Router → Knowledge → RAG → fonte oficial
2. `Can I sell through WhatsApp using Payment Link?` → resposta EN, fonte oficial
3. `What's the euro exchange rate today?` → Router → Knowledge → Tavily real
4. `Quando cai o dinheiro da venda de ontem?` → Router → Support → `get_recent_transactions`
5. `Quem foi Maradona?` → gate rejeita antes do LLM → Tavily (o caso que quebrou a rodada 04)
6. `user_id` desconhecido → handoff, zero dado inventado
