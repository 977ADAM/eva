import os
from pathlib import Path

import pytest

from eva.config import Config


def test_load_uses_defaults_when_env_unset(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    cfg = Config.load()
    assert cfg.deepseek_api_key == "test-key"
    assert cfg.sample_rate == 16000
    assert cfg.whisper_model_name == "small"
    assert cfg.history_window == 10
    assert "ева" in cfg.wake_words
    assert cfg.eva_dir == Path.home() / "eva"
    assert cfg.piper_bin == Path.home() / "eva" / "piper" / "piper"
    assert cfg.conversation_timeout_sec == 60.0


def test_load_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        Config.load()


def test_config_is_frozen():
    cfg = Config(
        eva_dir=Path("/tmp"),
        piper_bin=Path("/tmp/p"),
        voice_model=Path("/tmp/v"),
        deepseek_api_key="x",
    )
    with pytest.raises(Exception):
        cfg.sample_rate = 99  # type: ignore[misc]
