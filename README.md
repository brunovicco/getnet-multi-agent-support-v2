# getnet-multi-agent-support-v2

Python 3.13 project using uv and the team Claude Code engineering harness with the
`service` technical profile and `none` governance profile.

## Development

```bash
uv sync --frozen
uv run pytest
```

## Quality gate

```bash
uv run python scripts/quality_gate.py
```

List or select checks with `--list` and `--check NAME`.

When governance is enabled, maintain the project-owned records under `governance/`. The gate writes
metadata-only evidence to `build/governance-evidence/governance-report.json`.

## Container

```bash
docker build -t getnet-multi-agent-support-v2 .
docker run --rm getnet-multi-agent-support-v2
```

`Dockerfile` ships a placeholder `CMD`; replace it with the project's real entrypoint (an ASGI
server, a worker loop, etc.) once one exists.

See `AGENTS.md` for the engineering contract and `docs/ARCHITECTURE.md` for dependency rules.
