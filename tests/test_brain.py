from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eva.brain import Brain, ResponseDelta
from eva.config import Config


def make_config():
    return Config(
        eva_dir=Path("/tmp"),
        piper_bin=Path("/tmp"),
        voice_model=Path("/tmp"),
        deepseek_api_key="test-key",
    )


def make_chunk(content: str):
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    chunk.choices = [MagicMock(delta=delta)]
    return chunk


def install_fake_stream(brain, chunks):
    """Подменяет brain._client.chat.completions.create фейком,
    возвращающим переданные chunks (итерируется как стрим)."""
    brain._client.chat.completions.create = MagicMock(return_value=iter(chunks))


def test_yields_sentences_from_streaming_talk_response():
    brain = Brain(make_config())
    install_fake_stream(brain, [
        make_chunk('{"say":"Привет."'),
        make_chunk(',"action":"talk"}'),
    ])
    deltas = list(brain.ask_stream("привет"))
    sentences = [d.sentence for d in deltas if d.sentence]
    assert "Привет." in sentences
    assert deltas[-1].done is True
    assert deltas[-1].command is None


def test_emits_command_for_shell_action():
    brain = Brain(make_config())
    install_fake_stream(brain, [
        make_chunk('{"say":"Открываю.","action":"shell",'),
        make_chunk('"command":"firefox &"}'),
    ])
    deltas = list(brain.ask_stream("открой firefox"))
    assert deltas[-1].done is True
    assert deltas[-1].command == "firefox &"


def test_api_error_yields_single_error_delta():
    brain = Brain(make_config())
    def boom(**_):
        raise RuntimeError("network down")
    brain._client.chat.completions.create = boom
    deltas = list(brain.ask_stream("привет"))
    assert len(deltas) == 1
    assert deltas[0].error is not None
    assert deltas[0].done is True


def test_invalid_json_yields_error_delta():
    brain = Brain(make_config())
    install_fake_stream(brain, [make_chunk("это не json")])
    deltas = list(brain.ask_stream("привет"))
    # стриминг ничего не нашёл, парс упал
    assert any(d.error for d in deltas)
    assert deltas[-1].done is True


def test_markdown_json_fence_is_stripped():
    brain = Brain(make_config())
    install_fake_stream(brain, [
        make_chunk('```json\n{"say":"Готово.","action":"talk"}\n```'),
    ])
    deltas = list(brain.ask_stream("ну"))
    sentences = [d.sentence for d in deltas if d.sentence]
    assert sentences == ["Готово."]
    assert deltas[-1].done is True


def test_falls_back_to_full_parse_when_command_first():
    """LLM поставил command раньше say — стриминг ничего не отдал,
    но финальный парс должен вытащить say как одно предложение."""
    brain = Brain(make_config())
    install_fake_stream(brain, [
        make_chunk('{"action":"shell","command":"ls",'),
        make_chunk('"say":"Список."}'),
    ])
    deltas = list(brain.ask_stream("покажи"))
    sentences = [d.sentence for d in deltas if d.sentence]
    assert sentences == ["Список."]
    assert deltas[-1].command == "ls"


def test_history_truncates_to_window():
    cfg = make_config()
    brain = Brain(cfg)
    # Заполняем историю > history_window сообщениями
    for i in range(cfg.history_window + 5):
        install_fake_stream(brain, [
            make_chunk(f'{{"say":"ответ {i}.","action":"talk"}}'),
        ])
        list(brain.ask_stream(f"вопрос {i}"))

    # Следующий вызов должен передать <= history_window последних сообщений
    install_fake_stream(brain, [make_chunk('{"say":"ок.","action":"talk"}')])
    list(brain.ask_stream("ещё"))

    _, kwargs = brain._client.chat.completions.create.call_args
    sent_messages = kwargs["messages"]
    # 1 system + не больше history_window истории
    assert len(sent_messages) <= 1 + cfg.history_window
