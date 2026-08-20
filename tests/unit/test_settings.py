"""Unit tests for the single configuration composition point (REQ-23/24)."""

from pathlib import Path

import pytest

from getnet_support.entrypoints.settings import Settings


def test_real_env_var_wins_over_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-23: a real environment variable overrides a value loaded from `.env`."""
    dotenv = tmp_path / "does-not-need-to-exist.env"
    dotenv.write_text("GOOGLE_API_KEY=dotenv-value\n", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_API_KEY", "real-env-value")
    settings = Settings(_env_file=dotenv)
    assert settings.google_api_key == "real-env-value"


def test_missing_keys_report_as_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-24: without any key configured, capability flags are false."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    settings = Settings(_env_file=None)
    assert settings.llm_configured is False
    assert settings.web_search_configured is False


def test_configured_keys_report_as_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-empty key marks the corresponding capability as configured."""
    monkeypatch.setenv("GOOGLE_API_KEY", "some-key")
    monkeypatch.setenv("TAVILY_API_KEY", "some-key")
    settings = Settings(_env_file=None)
    assert settings.llm_configured is True
    assert settings.web_search_configured is True


def test_retriever_defaults_to_lexical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lexical retrieval is the safe default; semantic is opt-in via `RETRIEVER`."""
    monkeypatch.delenv("RETRIEVER", raising=False)
    settings = Settings(_env_file=None)
    assert settings.retriever == "lexical"
