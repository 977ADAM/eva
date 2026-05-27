# Conversation Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the first wake-word exchange, Eva stays "open" for 60 seconds and responds to follow-ups without requiring "Ева" again. The session extends on every exchange and auto-closes after the timeout.

**Architecture:** Add `conversation_until` timestamp + `_time_source` (injectable for tests) + `_conversation_timeout` to `Assistant`. Modify `handle_text` to bypass the wake-word check when the session is active. Single extension per exchange (right before `Brain` call). Sleep mode still takes priority.

**Tech Stack:** Python 3.11, existing `eva/` package, `time.monotonic` for clock (injectable), pytest with `pytest-mock`.

**Spec:** [docs/superpowers/specs/2026-05-27-conversation-mode-design.md](../specs/2026-05-27-conversation-mode-design.md)
**Branch:** `feat/conversation-mode` (already created, off `main`)

---

## File Structure

```
eva/
├── config.py          # +1 field: conversation_timeout_sec
└── assistant.py       # +time_source param, +conversation_until, modified handle_text

tests/
├── test_config.py             # +1 assertion in existing default test
└── test_assistant_logic.py    # +FakeClock helper, +5 new tests
```

No new files. Two modules touched. All existing tests stay green.

---

## Task 1: Config field

**Files:**
- Modify: `/home/adam/eva/eva/config.py`
- Modify: `/home/adam/eva/tests/test_config.py`

- [ ] **Step 1.1: Add the failing test assertion**

Open `/home/adam/eva/tests/test_config.py`. Find `test_load_uses_defaults_when_env_unset` and add one line at the end (after the last existing `assert`):

```python
    assert cfg.conversation_timeout_sec == 60.0
```

The full test now looks like:

```python
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
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
cd /home/adam/eva && uv run pytest tests/test_config.py::test_load_uses_defaults_when_env_unset -v
```

Expected: FAIL with `AttributeError: 'Config' object has no attribute 'conversation_timeout_sec'`.

- [ ] **Step 1.3: Add the field to Config**

In `/home/adam/eva/eva/config.py`, find the `# Поведение` section (near the bottom of the dataclass, with `debug: bool = False` and `require_wake: bool = True`). Add one line right after `require_wake`:

```python
    # Поведение
    debug: bool = False
    require_wake: bool = True

    # Разговорный режим — сколько секунд после последнего обмена сессия живёт
    conversation_timeout_sec: float = 60.0
```

- [ ] **Step 1.4: Run test to verify it passes**

```bash
cd /home/adam/eva && uv run pytest tests/test_config.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
cd /home/adam/eva && git add eva/config.py tests/test_config.py && git commit -m "feat(config): add conversation_timeout_sec field"
```

---

## Task 2: Assistant conversation logic

**Files:**
- Modify: `/home/adam/eva/eva/assistant.py`
- Modify: `/home/adam/eva/tests/test_assistant_logic.py`

This is the meat of the feature. We add a `time_source` parameter (defaults to `time.monotonic` so production code doesn't change), a `conversation_until` attribute, and modify `handle_text` to honor the session.

- [ ] **Step 2.1: Add FakeClock helper and the 5 failing tests**

Open `/home/adam/eva/tests/test_assistant_logic.py`. ADD (do not replace) the following to the file — keep all existing tests and helpers intact:

At the top of the file, after the existing imports, add:

```python
class FakeClock:
    """Контролируемый источник времени для тестов сессии. Подменяет
    time.monotonic в Assistant — никаких time.sleep в тестах."""
    def __init__(self, t: float = 0.0):
        self.now = t
    def __call__(self) -> float:
        return self.now
    def advance(self, dt: float) -> None:
        self.now += dt
```

Then add a new helper right next to `make_assistant`:

```python
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
```

Then at the END of the file, append the 5 new tests:

```python
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
    asst.handle_text("ева раз")  # сессия до t=60
    clock.advance(30)             # t=30
    brain.ask_stream.return_value = iter([
        ResponseDelta(sentence="Ок."), ResponseDelta(command=None, done=True),
    ])
    asst.handle_text("два")       # сессия должна продлиться до t=90
    clock.advance(50)             # t=80 — внутри новой сессии
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
    asst.handle_text("ева раз")   # сессия до t=60
    clock.advance(61)              # t=61, сессия истекла
    asst.handle_text("два")        # без wake-word — должно игнорироваться
    # Brain вызван только один раз (для первой реплики)
    assert brain.ask_stream.call_count == 1


def test_sleeping_blocks_session_use():
    asst, brain, _ = make_assistant_with_clock(brain_response=[
        ResponseDelta(sentence="Ок."), ResponseDelta(command=None, done=True),
    ])
    asst.handle_text("ева раз")   # открыли сессию
    asst.sleeping = True           # ушли в sleep
    asst.handle_text("два")        # без wake-word
    # Brain вызван только один раз — sleep заблокировал второе обращение
    assert brain.ask_stream.call_count == 1


def test_brain_error_still_extends_session():
    # Brain отдаёт error — но сессия всё равно продлевается (extension до Brain)
    asst, brain, clock = make_assistant_with_clock(timeout=60.0, brain_response=[
        ResponseDelta(error="api down", done=True),
    ])
    asst.handle_text("ева привет")  # error — но сессия открылась
    # Сессия должна быть открыта до t=60 (продление произошло до вызова Brain)
    assert asst.conversation_until is not None
    assert asst.conversation_until == 60.0
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
cd /home/adam/eva && uv run pytest tests/test_assistant_logic.py -v
```

Expected: existing 11 tests still pass (Assistant constructor doesn't accept `time_source` yet, but new helper isn't called by old tests). The 5 NEW tests FAIL with `TypeError: Assistant.__init__() got an unexpected keyword argument 'time_source'`.

If existing tests also fail, stop — that means the helper/import broke them. Check the file state before continuing.

- [ ] **Step 2.3: Modify Assistant — add time_source, conversation_until, session logic**

Open `/home/adam/eva/eva/assistant.py`. Replace its FULL contents with:

```python
import logging
import time
from typing import Callable

from eva.audio import AudioCapture, SpeechSegmenter
from eva.brain import Brain
from eva.config import Config
from eva.executor import ShellExecutor
from eva.stt import Transcriber
from eva.tts import Synthesizer

log = logging.getLogger(__name__)


class Assistant:
    """Главный оркестратор. Связывает все компоненты, держит флаги сессии
    (`sleeping`, `running`, `conversation_until`) и реализует логику
    wake/sleep/exit слов плюс разговорный режим.

    `run()` крутит главный цикл: получает utterance из segmenter,
    транскрибирует, прогоняет через `handle_text()`.

    `handle_text()` — отдельный метод для unit-тестирования логики
    без аудио-потока. Источник времени `time_source` инжектится для
    детерминистских тестов сессии (в проде — `time.monotonic`)."""

    def __init__(self, *, config: Config, capture: AudioCapture,
                 segmenter: SpeechSegmenter, transcriber: Transcriber,
                 synthesizer: Synthesizer, brain: Brain,
                 executor: ShellExecutor,
                 time_source: Callable[[], float] = time.monotonic):
        self._cfg = config
        self.capture = capture
        self.segmenter = segmenter
        self.transcriber = transcriber
        self.synthesizer = synthesizer
        self.brain = brain
        self.executor = executor
        self.sleeping = False
        self.running = True
        # Разговорный режим: timestamp монотонного времени, до которого
        # сессия активна. None — сессии нет.
        self.conversation_until: float | None = None
        self._time_source = time_source
        self._conversation_timeout = config.conversation_timeout_sec

    def run(self) -> None:
        self.synthesizer.say("Ева запущена. Скажи Ева чтобы обратиться.")
        print("\n=== Постоянно слушаю микрофон. Скажи 'Ева, ...' ===\n")
        self.capture.start()
        try:
            for utterance in self.segmenter.segments():
                if not self.running:
                    break
                text = self.transcriber.transcribe(utterance)
                if not text:
                    continue
                print(f"🎤 Услышала: {text}")
                self.handle_text(text)
                # Дренируем чтобы не услышать собственный голос
                self.capture.drain()
        finally:
            self.segmenter.stop()
            self.capture.stop()
            print("Ева остановлена.")

    def stop(self) -> None:
        self.running = False
        self.segmenter.stop()

    def handle_text(self, text: str) -> None:
        text = text.strip().lower()
        if not text:
            return

        if any(w in text for w in self._cfg.exit_words):
            self.synthesizer.say("Выключаюсь. Пока!")
            self.running = False
            return

        if not self.sleeping and any(w in text for w in self._cfg.sleep_words):
            self.synthesizer.say(
                "Молчу. Скажи 'Ева, проснись' когда понадоблюсь."
            )
            self.sleeping = True
            return

        if self.sleeping and any(w in text for w in self._cfg.wake_again_words):
            self.synthesizer.say("Я снова с тобой.")
            self.sleeping = False
            return

        if self.sleeping:
            return

        has_wake = any(w in text for w in self._cfg.wake_words)
        in_session = self._conversation_active()

        if self._cfg.require_wake and not (has_wake or in_session):
            return

        clean = self._strip_wake_word(text)
        if not clean:
            self.synthesizer.say("Слушаю.")
            return

        # Продлеваем сессию ОДИН раз за обмен — до вызова Brain.
        # Так юзер может пробовать снова без wake-word даже если Brain
        # отдал ошибку.
        self._extend_conversation()
        self._ask_brain_and_respond(clean)

    def _conversation_active(self) -> bool:
        if self.conversation_until is None:
            return False
        return self._time_source() < self.conversation_until

    def _extend_conversation(self) -> None:
        self.conversation_until = self._time_source() + self._conversation_timeout

    def _strip_wake_word(self, text: str) -> str:
        for w in self._cfg.wake_words:
            if text.startswith(w):
                return text[len(w):].strip(" ,.;:!?")
        return text

    def _ask_brain_and_respond(self, prompt: str) -> None:
        command_to_run: str | None = None
        for delta in self.brain.ask_stream(prompt):
            if delta.error:
                self.synthesizer.say("Связь пропала, попробуй позже.")
                return
            if delta.sentence:
                self.synthesizer.say(delta.sentence)
            if delta.done and delta.command:
                command_to_run = delta.command
        if command_to_run:
            self.executor.run(command_to_run)
```

Key behavior differences from the old version:
- New `time_source` kwarg in `__init__` (default `time.monotonic` — CLI/production untouched).
- New attribute `conversation_until` plus helpers `_conversation_active` / `_extend_conversation`.
- `handle_text` collapses the old `require_wake` branches into a single check: if `require_wake=True`, you need either an in-progress session or a wake-word — otherwise ignore. Session is extended right before `_ask_brain_and_respond`.

- [ ] **Step 2.4: Run the new tests to verify they pass**

```bash
cd /home/adam/eva && uv run pytest tests/test_assistant_logic.py -v
```

Expected: all 16 tests PASS (11 old + 5 new).

If `test_text_without_wake_word_processed_when_require_wake_false` fails, double-check the new `handle_text`: with `require_wake=False`, the `not (has_wake or in_session)` short-circuit should NOT return. Trace: `require_wake=False` → the `if self._cfg.require_wake and ...` is False → we don't return → fall through to strip + ask_brain. Good.

- [ ] **Step 2.5: Run the full suite to confirm nothing regressed**

```bash
cd /home/adam/eva && uv run pytest -v
```

Expected: 36 tests PASS (3 config + 3 segmenter + 7 parser + 7 brain + 16 assistant).

If any other test file breaks, stop and report — that means the Assistant change leaked through cli.py or another module.

- [ ] **Step 2.6: Commit**

```bash
cd /home/adam/eva && git add eva/assistant.py tests/test_assistant_logic.py && git commit -m "feat(assistant): add conversation mode after wake-word"
```

---

## Task 3: Smoke test + merge to main

This is a user-driven step. The user runs Eva live, verifies the new behavior, and only then we merge.

**Files:** none (verification + git operations)

- [ ] **Step 3.1: Run Eva and verify conversation mode**

```bash
cd /home/adam/eva && DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY .venv/bin/eva --debug
```

Manual checks (user, on real mic):
1. Say «Ева, привет» → Ева отвечает.
2. **Без** слова «Ева» сказать «как дела» в течение 60 секунд → Ева отвечает (раньше бы проигнорировала).
3. Помолчать > 60 секунд → сказать «как дела» без «Ева» → Ева НЕ отвечает (сессия закрылась).
4. «Ева, замолчи» → молчит. «Ева, проснись» → возобновляется.
5. «Ева, выключись» → корректно завершается.

Если что-то не работает — НЕ мерджить. Запустить `--debug`, посмотреть в логах появление `conversation_until` updates, диагностировать.

- [ ] **Step 3.2: Merge feat/conversation-mode → main**

После того как smoke test зелёный:

```bash
cd /home/adam/eva && git checkout main && git merge --no-ff feat/conversation-mode -m "Merge branch 'feat/conversation-mode'"
```

Expected: успешный merge commit. `git log --oneline -5` показывает merge с ветки.

- [ ] **Step 3.3: Push to origin**

```bash
cd /home/adam/eva && git push origin main
```

Если push сломается из-за auth (как уже случалось в этой среде) — push делает пользователь сам.

- [ ] **Step 3.4: Delete the merged branch**

```bash
cd /home/adam/eva && git branch -d feat/conversation-mode
```

Если git ругается что ветка не fully merged — это означает что Step 3.2 не сработал; разбираться, а не форсить `-D`.

---

## Definition of Done

- All 3 tasks complete.
- 36 tests PASS (`uv run pytest`).
- Live conversation mode works on the user's machine (Step 3.1 verified).
- `feat/conversation-mode` merged into `main` via `--no-ff` and deleted.
- `main` pushed to `origin` (or user pushed manually).
