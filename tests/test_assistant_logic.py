from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eva.assistant import Assistant
from eva.brain import ResponseDelta
from eva.config import Config


class FakeClock:
    """Контролируемый источник времени для тестов сессии. Подменяет
    time.monotonic в Assistant — никаких time.sleep в тестах."""
    def __init__(self, t: float = 0.0):
        self.now = t
    def __call__(self) -> float:
        return self.now
    def advance(self, dt: float) -> None:
        self.now += dt


def make_config(**overrides):
    base = dict(
        eva_dir=Path("/tmp"),
        piper_bin=Path("/tmp"),
        voice_model=Path("/tmp"),
        deepseek_api_key="x",
    )
    base.update(overrides)
    return Config(**base)


def make_assistant(brain_response=None):
    cfg = make_config()
    brain = MagicMock()
    if brain_response is not None:
        brain.ask_stream.return_value = iter(brain_response)
    return Assistant(
        config=cfg,
        capture=MagicMock(),
        segmenter=MagicMock(),
        transcriber=MagicMock(),
        synthesizer=MagicMock(),
        brain=brain,
        executor=MagicMock(),
    ), brain


def make_assistant_with_clock(brain_response=None, *, timeout=60.0):
    """Как make_assistant, но с контролируемыми часами и заданным таймаутом."""
    cfg = make_config(conversation_timeout_sec=timeout)
    brain = MagicMock()
    if brain_response is not None:
        brain.ask_stream.return_value = iter(brain_response)
    clock = FakeClock()
    asst = Assistant(
        config=cfg,
        capture=MagicMock(),
        segmenter=MagicMock(),
        transcriber=MagicMock(),
        synthesizer=MagicMock(),
        brain=brain,
        executor=MagicMock(),
        time_source=clock,
    )
    return asst, brain, clock


def test_exit_word_stops_running():
    asst, _ = make_assistant()
    asst.handle_text("ева выключись")
    assert asst.running is False


def test_sleep_word_sets_sleeping():
    asst, _ = make_assistant()
    asst.handle_text("ева замолчи")
    assert asst.sleeping is True


def test_wake_again_word_clears_sleeping():
    asst, _ = make_assistant()
    asst.sleeping = True
    asst.handle_text("ева проснись")
    assert asst.sleeping is False


def test_sleeping_ignores_normal_command():
    asst, brain = make_assistant()
    asst.sleeping = True
    asst.handle_text("ева открой firefox")
    brain.ask_stream.assert_not_called()


def test_wake_word_stripped_before_brain():
    asst, brain = make_assistant(brain_response=[
        ResponseDelta(sentence="Открываю."),
        ResponseDelta(command=None, done=True),
    ])
    asst.handle_text("ева открой firefox")
    brain.ask_stream.assert_called_once_with("открой firefox")


def test_command_is_executed_when_present():
    asst, _ = make_assistant(brain_response=[
        ResponseDelta(sentence="Открываю."),
        ResponseDelta(command="firefox &", done=True),
    ])
    asst.handle_text("ева открой firefox")
    asst.executor.run.assert_called_once_with("firefox &")


def test_no_command_means_no_execution():
    asst, _ = make_assistant(brain_response=[
        ResponseDelta(sentence="Всё хорошо."),
        ResponseDelta(command=None, done=True),
    ])
    asst.handle_text("ева как дела")
    asst.executor.run.assert_not_called()


def test_sentences_are_spoken_in_order():
    asst, _ = make_assistant(brain_response=[
        ResponseDelta(sentence="Один."),
        ResponseDelta(sentence="Два."),
        ResponseDelta(command=None, done=True),
    ])
    asst.handle_text("ева скажи два")
    spoken = [call.args[0] for call in asst.synthesizer.say.call_args_list]
    assert spoken == ["Один.", "Два."]


def test_error_delta_speaks_friendly_message():
    asst, _ = make_assistant(brain_response=[
        ResponseDelta(error="api down", done=True),
    ])
    asst.handle_text("ева привет")
    # Должно быть произнесено что-то с упоминанием связи или ошибки
    asst.synthesizer.say.assert_called()
    msg = asst.synthesizer.say.call_args.args[0].lower()
    assert "связь" in msg or "не поняла" in msg or "ошибка" in msg


def test_text_without_wake_word_ignored_when_require_wake():
    asst, brain = make_assistant()
    asst.handle_text("открой firefox")  # нет "ева"
    brain.ask_stream.assert_not_called()


def test_text_without_wake_word_processed_when_require_wake_false():
    cfg = make_config(require_wake=False)
    brain = MagicMock()
    brain.ask_stream.return_value = iter([
        ResponseDelta(sentence="Открываю."),
        ResponseDelta(command=None, done=True),
    ])
    asst = Assistant(
        config=cfg, capture=MagicMock(), segmenter=MagicMock(),
        transcriber=MagicMock(), synthesizer=MagicMock(),
        brain=brain, executor=MagicMock(),
    )
    asst.handle_text("открой firefox")
    brain.ask_stream.assert_called_once_with("открой firefox")


def test_in_session_no_wake_word_processed():
    # Первый обмен с wake-word открывает сессию
    asst, brain, _ = make_assistant_with_clock(brain_response=[
        ResponseDelta(sentence="Привет."),
        ResponseDelta(command=None, done=True),
    ])
    asst.handle_text("ева привет")
    # Следующая фраза без wake-word должна тоже уйти в Brain
    brain.ask_stream.return_value = iter([
        ResponseDelta(sentence="Хорошо."),
        ResponseDelta(command=None, done=True),
    ])
    asst.handle_text("как дела")
    # Второй вызов был сделан с очищенным текстом (без wake-word)
    assert brain.ask_stream.call_args_list[-1].args == ("как дела",)


def test_each_exchange_extends_session():
    asst, brain, clock = make_assistant_with_clock(timeout=60.0, brain_response=[
        ResponseDelta(sentence="Ок."), ResponseDelta(command=None, done=True),
    ])
    asst.handle_text("ева раз")   # сессия до t=60
    clock.advance(30)              # t=30
    brain.ask_stream.return_value = iter([
        ResponseDelta(sentence="Ок."), ResponseDelta(command=None, done=True),
    ])
    asst.handle_text("два")        # сессия должна продлиться до t=90
    clock.advance(50)              # t=80 — внутри новой сессии
    brain.ask_stream.return_value = iter([
        ResponseDelta(sentence="Ок."), ResponseDelta(command=None, done=True),
    ])
    asst.handle_text("три")
    # Brain вызван трижды — сессия не закрылась
    assert brain.ask_stream.call_count == 3


def test_session_expires_after_timeout():
    asst, brain, clock = make_assistant_with_clock(timeout=60.0, brain_response=[
        ResponseDelta(sentence="Ок."), ResponseDelta(command=None, done=True),
    ])
    asst.handle_text("ева раз")    # сессия до t=60
    clock.advance(61)               # t=61, сессия истекла
    asst.handle_text("два")         # без wake-word — должно игнорироваться
    # Brain вызван только один раз (для первой реплики)
    assert brain.ask_stream.call_count == 1


def test_sleeping_blocks_session_use():
    asst, brain, _ = make_assistant_with_clock(brain_response=[
        ResponseDelta(sentence="Ок."), ResponseDelta(command=None, done=True),
    ])
    asst.handle_text("ева раз")    # открыли сессию
    asst.sleeping = True            # ушли в sleep
    asst.handle_text("два")         # без wake-word
    # Brain вызван только один раз — sleep заблокировал второе обращение
    assert brain.ask_stream.call_count == 1


def test_brain_error_still_extends_session():
    # Brain отдаёт error — но сессия всё равно продлевается (extension до Brain)
    asst, brain, clock = make_assistant_with_clock(timeout=60.0, brain_response=[
        ResponseDelta(error="api down", done=True),
    ])
    asst.handle_text("ева привет")   # error — но сессия открылась
    # Сессия должна быть открыта до t=60 (продление произошло до вызова Brain)
    assert asst.conversation_until is not None
    assert asst.conversation_until == 60.0
