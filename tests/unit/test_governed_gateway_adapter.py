from types import TracebackType
from typing import ClassVar

import pytest
from governed_llm_gateway_contracts import DataClassification, RiskLevel

from getnet_support.adapters.llm import governed_gateway_adapter as gateway_module
from getnet_support.adapters.llm.governed_gateway_adapter import (
    GovernedGatewayAdapter,
    GovernedGatewayConfig,
)
from getnet_support.application.errors import LLMUnavailableError
from getnet_support.entrypoints.settings import Settings


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeGatewayClient:
    observed_kwargs: ClassVar[dict[str, object]] = {}
    response_content: ClassVar[str | None] = "governed answer"

    def __init__(self, config: object) -> None:
        self.config = config

    async def __aenter__(self) -> "_FakeGatewayClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    async def generate(self, **kwargs: object) -> _FakeResponse:
        type(self).observed_kwargs = kwargs
        return _FakeResponse(type(self).response_content)


def _adapter() -> GovernedGatewayAdapter:
    return GovernedGatewayAdapter(
        GovernedGatewayConfig(
            base_url="https://gateway.example.com",
            api_key="test-key",
            workload="support.getnet.answer",
            risk_level=RiskLevel.LOW,
            data_classification=DataClassification.PUBLIC,
        )
    )


def test_gateway_adapter_maps_provider_neutral_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_module, "GatewayClient", _FakeGatewayClient)
    _FakeGatewayClient.response_content = " governed answer "

    answer = _adapter().generate(prompt="approved evidence", timeout_seconds=2.0)

    assert answer == "governed answer"
    assert _FakeGatewayClient.observed_kwargs["workload"] == "support.getnet.answer"
    assert _FakeGatewayClient.observed_kwargs["risk_level"] is RiskLevel.LOW
    assert _FakeGatewayClient.observed_kwargs["data_classification"] is DataClassification.PUBLIC
    assert _FakeGatewayClient.observed_kwargs["provider_timeout_seconds"] == 2.0


def test_gateway_adapter_fails_closed_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_module, "GatewayClient", _FakeGatewayClient)
    _FakeGatewayClient.response_content = None

    with pytest.raises(LLMUnavailableError, match="no text content"):
        _adapter().generate(prompt="approved evidence", timeout_seconds=2.0)


def test_gateway_generation_configuration_is_independent_from_gemini_key() -> None:
    settings = Settings(
        llm_provider="gateway",
        gateway_url="https://gateway.example.com",
        gateway_api_key="gateway-key",
        google_api_key="",
    )

    assert settings.llm_configured is True
    assert settings.gateway_configured is True
    assert settings.google_llm_configured is False


def test_gateway_generation_fails_closed_when_gateway_credential_is_incomplete() -> None:
    settings = Settings(
        llm_provider="gateway",
        gateway_url="https://gateway.example.com",
        gateway_api_key="",
        google_api_key="google-key-must-not-be-used-as-fallback",
    )

    assert settings.gateway_configured is False
    assert settings.llm_configured is False


def test_semantic_embeddings_still_require_google_credential() -> None:
    settings = Settings(
        llm_provider="gateway",
        gateway_url="https://gateway.example.com",
        gateway_api_key="gateway-key",
        retriever="semantic_embeddings",
        google_api_key="",
    )

    assert settings.llm_configured is True
    assert settings.google_llm_configured is False
