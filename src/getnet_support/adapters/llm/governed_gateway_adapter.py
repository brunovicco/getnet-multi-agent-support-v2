"""Governed LLM Gateway-backed `LLMPort` implementation."""

import asyncio
from dataclasses import dataclass

from governed_llm_gateway_client import (
    GatewayClient,
    GatewayClientConfig,
    GatewayClientError,
    GatewayHTTPError,
)
from governed_llm_gateway_contracts import (
    DataClassification,
    ExecutionStatus,
    Message,
    MessageRole,
    RiskLevel,
)

from getnet_support.application.errors import LLMUnavailableError


@dataclass(frozen=True, slots=True)
class GovernedGatewayConfig:
    """Explicit provider-neutral gateway request configuration."""

    base_url: str
    api_key: str
    workload: str
    risk_level: RiskLevel
    data_classification: DataClassification


class GovernedGatewayAdapter:
    """Generate already-authorized support prose through the governed gateway."""

    def __init__(self, config: GovernedGatewayConfig) -> None:
        """Store explicit gateway configuration without provider credentials."""
        self._config = config

    def generate(self, *, prompt: str, timeout_seconds: float) -> str:
        """Generate text through one gateway request with no consumer-side retry."""
        return asyncio.run(self._generate(prompt=prompt, timeout_seconds=timeout_seconds))

    async def _generate(self, *, prompt: str, timeout_seconds: float) -> str:
        client_config = GatewayClientConfig(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            request_timeout_seconds=timeout_seconds,
        )
        try:
            async with GatewayClient(client_config) as client:
                response = await client.generate(
                    workload=self._config.workload,
                    messages=(Message(role=MessageRole.USER, content=prompt),),
                    risk_level=self._config.risk_level,
                    data_classification=self._config.data_classification,
                    max_output_tokens=1200,
                    provider_timeout_seconds=timeout_seconds,
                )
        except GatewayHTTPError as exc:
            raise LLMUnavailableError(
                f"Governed gateway rejected the request: {exc.code}"
            ) from None
        except GatewayClientError as exc:
            raise LLMUnavailableError(
                f"Governed gateway request failed: {type(exc).__name__}"
            ) from None

        if response.status is not ExecutionStatus.SUCCEEDED:
            raise LLMUnavailableError("Governed gateway execution did not succeed")
        text = (response.content or "").strip()
        if not text:
            raise LLMUnavailableError("Governed gateway returned no text content")
        return text
