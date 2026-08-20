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

    google_api_key: str = ""
    tavily_api_key: str = ""
    groq_api_key: str = ""

    retriever: Literal["lexical", "semantic", "semantic_embeddings"] = "lexical"
    score_min: float = 0.1
    coverage_min: float = 0.55
    router_confidence_min: float = 0.6
    llm_timeout_seconds: float = 2.0

    @property
    def llm_configured(self) -> bool:
        """Whether an LLM provider key is present."""
        return bool(self.google_api_key.strip())

    @property
    def web_search_configured(self) -> bool:
        """Whether a web search provider key is present."""
        return bool(self.tavily_api_key.strip())
