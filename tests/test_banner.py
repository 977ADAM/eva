from pathlib import Path
from unittest.mock import MagicMock

from rich.console import Console

from eva.banner import _plural, build_panel
from eva.config import Config


def make_config():
    return Config(
        eva_dir=Path("/tmp"),
        piper_bin=Path("/tmp"),
        voice_model=Path("/tmp"),
        memory_path=Path("/tmp"),
        deepseek_api_key="x",
    )


def fake_memory(facts):
    m = MagicMock()
    m.all.return_value = list(facts)
    return m


def render(panel) -> str:
    console = Console(record=True, width=80)
    console.print(panel)
    return console.export_text()


def test_plural_one():
    assert _plural(1, "факт", "факта", "фактов") == "факт"


def test_plural_few():
    assert _plural(2, "факт", "факта", "фактов") == "факта"
    assert _plural(4, "факт", "факта", "фактов") == "факта"


def test_plural_many():
    assert _plural(5, "факт", "факта", "фактов") == "фактов"
    assert _plural(11, "факт", "факта", "фактов") == "фактов"
    assert _plural(14, "факт", "факта", "фактов") == "фактов"
    assert _plural(21, "факт", "факта", "фактов") == "факт"


def test_panel_contains_version():
    panel = build_panel(make_config(), fake_memory([]), "1.2.3", mic_name="тест")
    assert "1.2.3" in render(panel)


def test_panel_contains_model_and_wake_word():
    text = render(build_panel(make_config(), fake_memory([]), "0.1.0", mic_name="тест"))
    assert "Whisper small" in text
    assert "ева" in text


def test_panel_shows_fact_count():
    panel = build_panel(make_config(), fake_memory(["a", "b", "c", "d"]), "0.1.0", mic_name="тест")
    assert "4 факта" in render(panel)


def test_panel_empty_memory_says_empty():
    text = render(build_panel(make_config(), fake_memory([]), "0.1.0", mic_name="тест"))
    assert "пусто" in text
    assert "0 фактов" not in text


def test_mic_name_injected():
    panel = build_panel(make_config(), fake_memory([]), "0.1.0", mic_name="USB Microphone")
    assert "USB Microphone" in render(panel)
