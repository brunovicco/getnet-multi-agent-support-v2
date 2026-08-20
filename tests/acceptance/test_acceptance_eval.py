"""Acceptance evaluation for SPEC 001.

Contrato deste arquivo (escrito antes da implementacao):

- roda OFFLINE, sem nenhuma API key, dentro do quality gate;
- asserta sobre metadados deterministicos da resposta (route, sources, tools,
  handoff), nunca sobre o texto gerado pelo LLM;
- os testes marcados `live` sao pulados por padrao e so rodam com chaves reais.

A implementacao deve expor `build_app(settings)` em
`getnet_support.entrypoints.http` e honrar o contrato de REQ-01/REQ-02.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

DATASET = json.loads((Path(__file__).parent / "eval_dataset.json").read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = DATASET["cases"]

OFFLINE_ENV = {
    "APP_ENV": "test",
    "GOOGLE_API_KEY": "",
    "GROQ_API_KEY": "",
    "TAVILY_API_KEY": "",
}


def _ids(prefix: str) -> list[str]:
    return [case["id"] for case in CASES if case["id"].startswith(prefix)]


@pytest.fixture(params=["lexical", "semantic"])
def client(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """REQ-16: o mesmo dataset roda contra os dois retrievers, com o mesmo resultado.

    O retriever semantico usa embeddings pre-computados do corpus commitado, para
    que a paridade seja verificavel sem chave de API.
    """
    from getnet_support.entrypoints.http import build_app

    for key, value in OFFLINE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("RETRIEVER", request.param)
    with TestClient(build_app()) as test_client:
        yield test_client


def _ask(client: TestClient, case: dict[str, Any]) -> dict[str, Any]:
    payload = {"message": case["message"], "user_id": case["user_id"]}
    for optional in ("market", "locale"):
        if optional in case:
            payload[optional] = case[optional]
    response = client.post("/chat", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------- contratos


def test_contract_original_payload_still_works(client: TestClient) -> None:
    """REQ-01: o payload do enunciado nao pode quebrar."""
    response = client.post("/chat", json={"message": "Olá", "user_id": "cliente1988"})
    assert response.status_code == 200


def test_contract_response_shape(client: TestClient) -> None:
    """REQ-02: campos obrigatorios presentes em toda resposta."""
    body = _ask(client, CASES[0])
    for field in (
        "trace_id",
        "answer",
        "language",
        "route",
        "agents",
        "tools",
        "sources",
        "handoff_required",
        "grounding",
        "web_search_attempted",
        "latency_ms",
    ):
        assert field in body, f"campo ausente: {field}"


def test_contract_health_reports_capabilities(client: TestClient) -> None:
    """REQ-03: /health diagnostica configuracao, nao apenas liveness."""
    body = client.get("/health").json()
    assert body["llm"] == "missing"
    assert body["web_search"] == "missing"
    assert body["retriever"] in {"lexical", "semantic"}
    assert body["corpus_chunks"] > 0


def test_contract_ui_is_served(client: TestClient) -> None:
    """REQ-04: a UI responde na raiz, no mesmo processo."""
    assert client.get("/").status_code == 200


# ----------------------------------------------------------------------------- dataset


@pytest.mark.parametrize("case_id", _ids("OF-") + _ids("PT-"))
def test_eval_dataset_routing(client: TestClient, case_id: str) -> None:
    """REQ-05: acuracia de rota 1.00 nos cenarios oficiais e paráfrases."""
    case = next(item for item in CASES if item["id"] == case_id)
    body = _ask(client, case)
    assert body["route"] == case["expect"]["route"], f"{case_id}: {case['message']}"


@pytest.mark.parametrize("case_id", _ids("OF-") + _ids("PT-") + _ids("OOS-"))
def test_eval_dataset_source_provenance(client: TestClient, case_id: str) -> None:
    """REQ-10/REQ-11: origem das fontes coerente com a rota."""
    case = next(item for item in CASES if item["id"] == case_id)
    expect = case["expect"]
    origins = {source["origin"] for source in _ask(client, case)["sources"]}
    if "source_origin" in expect:
        assert expect["source_origin"] in origins or not origins
    if "forbid_source_origin" in expect:
        assert expect["forbid_source_origin"] not in origins, (
            f"{case_id}: fonte Getnet citada para pergunta fora do corpus"
        )
    if "grounding_offline" in expect:
        assert _ask(client, case)["grounding"] == expect["grounding_offline"]


@pytest.mark.parametrize("case_id", _ids("OOS-") + ["OF-02", "OF-07"])
def test_no_evidence_reaches_web_step(client: TestClient, case_id: str) -> None:
    """REQ-09/REQ-10: o gate rejeita a KB e a cadeia chega ao passo de web.

    Este e o teste que a rodada 04 nao tinha: "quem foi Maradona?" era marcado
    como resolvido pela KB e nunca tentava a web. Vale offline, porque a
    tentativa e observavel mesmo sem TAVILY_API_KEY.
    """
    case = next(item for item in CASES if item["id"] == case_id)
    body = _ask(client, case)
    assert body["web_search_attempted"] is True, "a cadeia de evidencia parou antes da web"
    assert body["grounding"] == "none", "sem chave, nao ha grounding — e nao se inventa"
    assert body["handoff_required"] is True, "degrada com honestidade"
    assert not [s for s in body["sources"] if s["origin"] == "getnet_kb"]


@pytest.mark.parametrize("case_id", _ids("OF-") + _ids("PT-"))
def test_in_corpus_never_calls_web(client: TestClient, case_id: str) -> None:
    """REQ-11: pergunta coberta pela KB nao paga chamada de web."""
    case = next(item for item in CASES if item["id"] == case_id)
    expect = case["expect"]
    if expect.get("web_search_attempted") is not False:
        pytest.skip("caso nao e de corpus")
    assert _ask(client, case)["web_search_attempted"] is False


@pytest.mark.parametrize("case_id", [case["id"] for case in CASES])
def test_no_evidence_sentinel_never_leaks(client: TestClient, case_id: str) -> None:
    """REQ-12: a sentinela interna nunca chega ao usuario."""
    case = next(item for item in CASES if item["id"] == case_id)
    assert "NO_EVIDENCE_IN_CONTEXT" not in _ask(client, case)["answer"]


# ---------------------------------------------------------------------------- cliente


def test_customer_unknown_user_never_invents_data(client: TestClient) -> None:
    """REQ-18: usuario desconhecido escala sem inventar dado."""
    case = next(item for item in CASES if item["id"] == "SEC-01")
    body = _ask(client, case)
    assert body["handoff_required"] is True
    assert body["tools"] == []


def test_customer_cross_access_is_blocked(client: TestClient) -> None:
    """REQ-19: isolamento entre clientes e deterministico."""
    case = next(item for item in CASES if item["id"] == "SEC-02")
    body = _ask(client, case)
    assert "cliente2001" not in body["answer"]
    assert body["handoff_required"] is True


def test_customer_state_changing_operation_escalates(client: TestClient) -> None:
    """REQ-08: operacao financeira nao suportada nunca e executada."""
    case = next(item for item in CASES if item["id"] == "SEC-03")
    body = _ask(client, case)
    assert body["route"] == "escalation"
    assert body["handoff_required"] is True


# ----------------------------------------------------------------------------- idioma


@pytest.mark.parametrize("case_id", ["MKT-01", "MKT-02", "OF-01", "PT-01"])
def test_language_contract(client: TestClient, case_id: str) -> None:
    """REQ-20/REQ-21: locale vence deteccao; idioma nao define mercado."""
    case = next(item for item in CASES if item["id"] == case_id)
    expect = case["expect"]
    body = _ask(client, case)
    if "language" in expect:
        assert body["language"] == expect["language"]
    if "source_market" in expect:
        markets = {source.get("market") for source in body["sources"]}
        assert markets <= {expect["source_market"], None}


# ------------------------------------------------------------------------------- live


@pytest.mark.live
def test_live_web_search_returns_real_result() -> None:
    """Smoke com chaves reais. Rode com: uv run pytest -m live."""
    import os

    if not os.environ.get("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY ausente")
    from getnet_support.entrypoints.http import build_app

    with TestClient(build_app()) as live:
        body = live.post(
            "/chat",
            json={"message": "What's the euro exchange rate today?", "user_id": "cliente1988"},
        ).json()
        assert body["route"] == "knowledge"
        assert body["grounding"] == "web"
        assert body["sources"], "Tavily real deve devolver fontes"
        assert body["handoff_required"] is False
