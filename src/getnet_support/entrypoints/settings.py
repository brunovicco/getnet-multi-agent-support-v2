"""Process configuration: the single composition point for environment values.

REQ-23: `.env` is loaded via pydantic-settings; a real environment variable
always wins over `.env` (pydantic-settings' default precedence). Read
:class:`Settings` once, in :func:`getnet_support.entrypoints.http.build_app`,
and pass it down explicitly — never re-read `os.environ` deeper in the call
stack.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    service_name: str = "getnet-multi-agent-support-v2"
    log_level: str = "INFO"
    log_format: str = "json"

    llm_provider: Literal["gemini", "gateway"] = "gemini"
    google_api_key: str = ""
    gateway_url: str = ""
    gateway_api_key: str = ""
    gateway_workload: str = "support.getnet.answer"
    gateway_risk_level: Literal["low", "medium", "high", "critical"] = "low"
    gateway_data_classification: Literal[
        "public", "internal", "confidential", "restricted"
    ] = "public"

    tavily_api_key: str = ""
    groq_api_key: str = ""

    retriever: Literal["lexical", "semantic", "semantic_embeddings"] = "lexical"
    score_min: float = 0.1
    coverage_min: float = 0.55
    router_confidence_min: float = 0.6
    llm_timeout_seconds: float = 2.0

    @property
    def google_llm_configured(self) -> bool:
        """Whether the direct Gemini generation/embedding credential is present."""
        return bool(self.google_api_key.strip())

    @property
    def gateway_configured(self) -> bool:
        """Whether the governed gateway endpoint and credential are both present."""
        return bool(self.gateway_url.strip()) and bool(self.gateway_api_key.strip())

    @property
    def llm_configured(self) -> bool:
        """Whether the selected generation path is fully configured."""
        if self.llm_provider == "gateway":
            return self.gateway_configured
        return self.google_llm_configured

    @property
    def web_search_configured(self) -> bool:
        """Whether a web search provider key is present."""
        return bool(self.tavily_api_key.strip())
