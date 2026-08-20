# SPEC 001 — Multi-Agent Support System (Getnet)

Fonte oficial dos requisitos: `docs/AI Hardcore Engineer - Multi-Agent Support System.md`.
Esta spec é o contrato. O quality gate reprova quando um `REQ-*` marcado `[gate]` falha.

**Nota de honestidade:** esta spec foi escrita ANTES do cronômetro. O timebox de 40 minutos
cobre implementação, testes, Docker e README — não a especificação. Isso está declarado no
README e no vídeo.

---

## Escopo

Sistema multi-agente que responde dúvidas de clientes Getnet, com RAG sobre conteúdo oficial,
busca web real para informação atual, tools determinísticas para dados de cliente, e escalação
quando não há evidência. UI própria para avaliação pela perspectiva do usuário. PT-BR e EN.

Fora de escopo nesta rodada: LangGraph, vector DB gerenciado, Redis/Postgres, MCP server,
streaming, multi-provider routing, autenticação, memória persistente.

---

## REQ — API

**REQ-01 [gate]** `POST /chat` aceita o payload original do desafio sem quebrar:
`{"message": str, "user_id": str}`. Campos opcionais `market` (`BR`|`GLOBAL`) e
`locale` (`pt-BR`|`en`) são aceitos e ignoráveis com segurança.

**REQ-02 [gate]** A resposta de `/chat` contém, sempre:
```
trace_id, answer, language, route, agents[], tools[], sources[], handoff_required, latency_ms
```
Cada item de `sources[]` tem: `title`, `url`, `origin` (`getnet_kb`|`web`), `retrieved_at`,
`volatility` (`low`|`high`).

**REQ-03 [gate]** `GET /health` reporta prontidão de capacidade, não apenas `ok`:
```json
{"status":"ok","llm":"configured|missing","web_search":"configured|missing","retriever":"lexical|semantic","corpus_chunks":<int>}
```
> Motivação: na rodada anterior o `.env` não era carregado e isso custou uma rodada inteira de
> debug. Com este endpoint, o diagnóstico é uma requisição.

**REQ-04 [gate]** `GET /` serve a UI no mesmo processo. A UI chama o application service
diretamente. É proibida qualquer chamada HTTP da UI para a própria API.

---

## REQ — Roteamento

**REQ-05 [gate]** Acurácia de rota = 1.00 sobre `tests/acceptance/eval_dataset.json`
(cenários oficiais + paráfrases PT/EN + fora de escopo).

**REQ-06 [gate]** É proibido decidir "isto é pergunta atual/externa?" por regex ou lista de
palavras-chave de tópico (clima, câmbio, cotação, tempo...). A decisão de consultar a web é
consequência de evidência insuficiente, nunca de classificação de assunto.
Verificação: nenhum padrão de tópico no código de roteamento/knowledge.

**REQ-07** Regras determinísticas decidem os casos de alta confiança. LLM só é consultado
como desempate abaixo do limiar, com timeout ≤ 2s. O evento de decisão registra
`decision_source` (`rule`|`llm`|`fallback`) e `classifier_latency_ms`.

**REQ-08 [gate]** LLM nunca é autoridade para: autorização, escopo de cliente, isolamento de
mercado, ou operação financeira que altera estado.

---

## REQ — Knowledge / RAG

**REQ-09 [gate]** Gate de evidência **determinístico antes** de qualquer chamada de LLM:
```
aceita  ⟺  score_retrieval ≥ SCORE_MIN  E  cobertura_lexical(query, chunk) ≥ COVERAGE_MIN
```
`COVERAGE_MIN` = fração de termos não-stopword da query presentes no chunk recuperado.
> Motivação: na rodada anterior, "quem foi Maradona?" passou do limiar de cosseno (0.3) contra
> chunks de WhatsApp/Crediário. Score denso sozinho não tem piso discriminativo num corpus de
> ~13 chunks. Cobertura lexical rejeita isso antes de gastar uma chamada de Gemini.

**REQ-10 [gate]** Pergunta fora do corpus → `route == "web"` e `sources[]` sem nenhum item
com `origin == "getnet_kb"`. Casos obrigatórios: Maradona, capital da França, previsão do
tempo, cotação do euro.

**REQ-11 [gate]** Pergunta coberta pelo corpus → **zero** chamadas ao web search.
> Motivação: custo. Sem isto, toda pergunta paga RAG + LLM + Tavily.

**REQ-12 [gate]** A sentinela de contexto insuficiente (`NO_EVIDENCE_IN_CONTEXT`) nunca aparece
no `answer` entregue ao usuário, em nenhum idioma. Se o modelo não emitir a sentinela e o gate
determinístico já tiver reprovado, o gate vence.

**REQ-13 [gate]** Nem RAG nem web resolveram → `handoff_required == true` e mensagem única e
honesta, cobrindo as duas falhas.

**REQ-14 [gate]** Corpus persistido no repositório. Startup não faz crawling.
Cada chunk carrega: `id, text, title, source, market, language, topic, retrieved_at, volatility`.

**REQ-15** Conteúdo de `https://site.getnet.com.br/ofertas/` é marcado `volatility=high`.
Resposta que o utiliza qualifica a informação, cita a fonte e recomenda confirmação.
Valores nunca são inventados nem apresentados como condição atual garantida.

**REQ-16 [gate]** **Paridade de retriever.** `eval_dataset.json` produz o mesmo resultado com o
retriever léxico e com o semântico. Promoção para semântico só com paridade verde.
> Motivação: a evolução P1.1 (embeddings) introduziu regressão P0 silenciosa na rodada anterior,
> porque o retriever semântico só ativa quando há `GOOGLE_API_KEY`.

---

## REQ — Customer Support

**REQ-17 [gate]** Mínimo 3 tools determinísticas, todas customer-scoped:
`get_customer_profile`, `get_recent_transactions`, `get_terminal_status`.
Fixtures: `cliente1988`, `cliente2001`.

**REQ-18 [gate]** `user_id` desconhecido → `handoff_required == true`, `tools == []`, e nenhum
dado de cliente no `answer`.

**REQ-19 [gate]** Isolamento cross-customer: resposta para `cliente1988` nunca contém dado de
`cliente2001`, mesmo se a mensagem pedir explicitamente.

---

## REQ — Bilíngue e mercado

**REQ-20 [gate]** Idioma da resposta = `locale` quando explícito; senão, idioma detectado da
mensagem. `language` na resposta reflete o idioma efetivamente usado.

**REQ-21 [gate]** Idioma ≠ mercado. Pergunta em EN com `market=BR` usa conteúdo BR.
Nunca assumir `pt-BR ⇒ BR` nem `en ⇒ GLOBAL`.

**REQ-22** Sem evidência do mercado solicitado, escala em vez de responder com o outro mercado.
(Se BR/GLOBAL ameaçar o cutoff: BR funcional é P0, GLOBAL vira P1 documentado.)

---

## REQ — Configuração, Docker, degradação

**REQ-23 [gate]** `.env` carregado via `pydantic-settings` num único ponto de composição.
Variável de ambiente real vence `.env`.

**REQ-24 [gate]** Sem nenhuma chave configurada: o serviço sobe, `/health` reporta `missing`,
`/chat` responde com degradação explícita, e **nenhuma** resposta é fabricada.

**REQ-25** `docker build` + `docker run -p 8000:8000 -e ...` funcionam sem passos extras.

---

## REQ — Observabilidade

**REQ-26** Log estruturado por requisição com `trace_id, route, decision_source, agents, tools,
sources_count, latency_ms, handoff_required`.

**REQ-27** A UI exibe, por mensagem: rota, agentes, tools, sources clicáveis, handoff e latência.

---

## Rastreabilidade

| Grupo | Testes |
|---|---|
| REQ-01..04 | `tests/acceptance/test_acceptance_eval.py::test_contract_*` |
| REQ-05, 10, 11, 16 | `test_acceptance_eval.py::test_eval_dataset_*` |
| REQ-12, 13 | `test_acceptance_eval.py::test_no_evidence_*` |
| REQ-17..19 | `test_acceptance_eval.py::test_customer_*` |
| REQ-20, 21 | `test_acceptance_eval.py::test_language_*` |
| REQ-23, 24 | `tests/unit/test_settings.py` |

## Definição de pronto (cutoff)

`uv run python scripts/quality_gate.py` verde, com todos os `[gate]` cobertos, e tag
`challenge-40m` apontando exatamente para o estado do cutoff.
