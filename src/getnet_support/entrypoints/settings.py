"""Process configuration, loaded from environment variables and an optional local `.env` file.

Uses `pydantic-settings` (already a project dependency) so `.env` values are picked up the same
way in local dev as in Docker/CI, without every module needing its own dotenv-loading logic. Real
process environment variables always take precedence over `.env` file values (pydantic-settings'
default precedence — standard 12-factor behavior), so `docker run -e VAR=...` still wins.
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process configuration for the Getnet multi-agent support service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    log_format: str = "json"
    service_name: str = "getnet-multi-agent-support"

    google_api_key: str | None = None
    google_model: str | None = None
    google_embedding_model: str | None = None
    tavily_api_key: str | None = None


def load_settings() -> Settings:
    """Load settings from `.env`/the environment and mirror them into `os.environ`.

    Several modules (the structlog bootstrap, the adapter factories) read configuration directly
    from `os.environ` rather than taking a `Settings` object, so this also exports every loaded
    value back into the process environment — call it once, before any of those run.
    """
    settings = Settings()
    for field_name, value in settings.model_dump().items():
        if value is not None:
            os.environ.setdefault(field_name.upper(), str(value))
    return settings
