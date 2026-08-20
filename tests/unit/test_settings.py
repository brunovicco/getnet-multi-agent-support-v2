"""Behavior tests for the `.env`/environment settings loader.

Regression coverage for a real deployment gap: nothing in the app loaded `.env` before this,
so provider keys placed only in a local `.env` file were silently invisible to the running
process (`os.environ.get("TAVILY_API_KEY")` returned None even with a real key on disk).
"""

import os
from pathlib import Path

import pytest

from getnet_support.entrypoints.settings import Settings, load_settings


def test_loads_value_from_a_dotenv_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_API_KEY=test-key-from-dotenv\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.google_api_key == "test-key-from-dotenv"


def test_real_environment_variable_takes_precedence_over_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "from-real-env")
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_API_KEY=from-dotenv\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.google_api_key == "from-real-env"


def test_defaults_when_nothing_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    settings = Settings(_env_file=tmp_path / "does-not-exist.env")

    assert settings.google_api_key is None
    assert settings.tavily_api_key is None
    assert settings.app_env == "development"


def test_load_settings_mirrors_dotenv_values_into_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # isolate from any real .env in the repo root
    (tmp_path / ".env").write_text("TAVILY_API_KEY=test-tavily-key\n", encoding="utf-8")

    load_settings()

    assert os.environ["TAVILY_API_KEY"] == "test-tavily-key"
