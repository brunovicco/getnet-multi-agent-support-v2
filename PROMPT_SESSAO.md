# Prompt de sessão — rodada 05

Atue como Senior/Staff AI Engineer.

Leia, nesta ordem, e trate como fonte da verdade:

1. `specs/001-multi-agent-support/spec.md` — o contrato. Os `REQ-*` marcados `[gate]` são
   critérios de aceitação, não sugestões.
2. `specs/001-multi-agent-support/plan.md` — decisões já tomadas. Não as reabra.
3. `specs/001-multi-agent-support/tasks.md` — a sequência e o timebox.
4. `tests/acceptance/test_acceptance_eval.py` e `tests/acceptance/eval_dataset.json` — **o
   eval já existe e está vermelho de propósito. Ele é o alvo, não o resultado.**
5. `CLAUDE.md`, `AGENTS.md`, `.claude/rules/` e
   `docs/AI Hardcore Engineer - Multi-Agent Support System.md`.

## Regras da sessão

- Português como idioma da conversa.
- 40 minutos de implementação, contados a partir da minha aprovação do plano. O tempo que eu
  levo revisando os checkpoints não conta — não se apresse por causa deles.
- Suposições não críticas: assuma, registre em `plan.md`, siga. Não pergunte.
- Não altere `spec.md`, `eval_dataset.json` nem as asserções de `test_acceptance_eval.py`
  para fazer o gate passar. Se um `REQ-*` estiver errado ou impossível, **pare e me diga** —
  não o contorne, não relaxe a asserção, não marque `xfail`, não pule com `skip`.
- Se um slice ameaçar o cutoff, corte a feature, anote no README como P1 e siga. Escopo é
  decisão, não fracasso.

## Ciclo por task — aprovação em lote

Você trabalha uma task inteira do `tasks.md` sem parar. Ao fim de cada uma:

1. `uv run pytest tests/acceptance -q --no-cov` (sempre com `--no-cov`; o coverage global
   reprova subconjuntos e não significa nada aqui)
2. `git status --short` e `git diff --stat`
3. Um resumo de até 5 linhas: o que a task entregou, quais `REQ-*` fechou, o que ficou
   pendente.
4. **Pare e aguarde meu OK.** Eu reviso e commito. Você não avança para a próxima task
   antes disso.

Não avance com o eval da task anterior vermelho, salvo se eu autorizar explicitamente.

## Git — você não escreve na história

- Você **não** commita, não faz push, não cria nem move tags. Eu faço.
- **Nunca** rode `git reset`, `git checkout --`, `git restore`, `git clean`, `git stash` nem
  nada que descarte trabalho não commitado. `.env`, `data/` e o corpus não estão versionados
  — um "clean" bem-intencionado destrói a preparação inteira.
- Se precisar desfazer algo, descreva o que quer desfazer e espere.
- Se uma permissão for negada, **pergunte**. Não contorne com outro comando, outra flag,
  `subprocess`, script auxiliar ou qualquer caminho alternativo.

## Agora — Plan Mode

Não edite nada. Inspecione o repo e devolva um plano revisável em 2 minutos:

- arquivos que vão nascer, por task;
- a ordem, com as janelas do `tasks.md`;
- o que fica adiado e por quê;
- qualquer conflito que você encontrar entre a spec, o eval e o harness — este é o momento
  de levantar, não durante a implementação.

Termine com `READY TO IMPLEMENT` e aguarde.
