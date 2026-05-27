# Command Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перестать выполнять любую сгенерированную DeepSeek shell-команду без вопросов. Команды из allowlist (xdg-open, firefox, playerctl и т.п.) бегут как раньше; всё остальное требует голосового подтверждения «да/нет».

**Architecture:** Новый `SafetyGate` класс (`eva/safety.py`) — pure function-style: `is_safe(command) -> bool` по allowlist первого токена + проверка shell-метасимволов. `Assistant` получает `pending_command: str | None` и новую ветку в `handle_text` для обработки ответа на подтверждение. Pending привязан к существующей conversation-сессии (нет нового таймера). Никаких изменений в Brain, Synthesizer, Transcriber, Executor, AudioCapture, SpeechSegmenter, CLI.

**Tech Stack:** Python 3.11, существующий `eva/` пакет, pytest. Никаких новых зависимостей.

**Spec:** [docs/superpowers/specs/2026-05-27-command-safety-design.md](../specs/2026-05-27-command-safety-design.md)
**Branch:** `feat/command-safety` (уже создана, off `main`, spec уже закоммичен)

---

## File Structure

```
eva/
├── safety.py          # NEW: SafetyGate + _SHELL_METACHARS
├── config.py          # MOD: +safe_command_prefixes, +confirm_yes_words, +confirm_no_words
└── assistant.py       # MOD: +pending_command, +_gate, new handle_text branch,
                       #      modified _ask_brain_and_respond, sleep clears pending

tests/
├── test_safety.py             # NEW: 7 tests for SafetyGate
├── test_config.py             # MOD: assertions for 3 new Config fields
└── test_assistant_logic.py    # MOD: +5 tests for pending/confirmation flow
```

Все 36 текущих тестов должны остаться зелёными. Подсчёт тестов после всех задач: 36 текущих + 7 новых в test_safety + 6 новых в test_assistant_logic = **49 тестов**. (Изменения в test_config — это добавленные assert'ы в существующем тесте, общее число тестов в файле не меняется.)

---

## Task 1: SafetyGate

**Files:**
- Create: `/home/adam/eva/eva/safety.py`
- Create: `/home/adam/eva/tests/test_safety.py`

TDD: пишем тесты, видим что падают, реализуем, видим что зелёные, коммитим.

- [ ] **Step 1.1: Write the failing tests**

Create `/home/adam/eva/tests/test_safety.py`:

```python
import pytest

from eva.safety import SafetyGate


@pytest.fixture
def gate():
    return SafetyGate(safe_prefixes=("firefox", "xdg-open", "playerctl"))


def test_empty_command_is_safe(gate):
    assert gate.is_safe("") is True
    assert gate.is_safe("   ") is True


def test_allowlist_prefix_is_safe(gate):
    assert gate.is_safe("firefox") is True


def test_firefox_with_args_is_safe(gate):
    assert gate.is_safe("firefox https://ya.ru") is True


def test_firefox_with_trailing_amp_is_safe(gate):
    # Запуск в фоне — типичный паттерн, не должен требовать подтверждения
    assert gate.is_safe("firefox &") is True


def test_unknown_command_needs_confirm(gate):
    assert gate.is_safe("shutdown now") is False


def test_shell_chain_needs_confirm(gate):
    # Даже если первое слово allowlisted, цепочка через && делает команду опасной
    assert gate.is_safe("firefox && rm -rf /") is False


def test_pipe_needs_confirm(gate):
    assert gate.is_safe("firefox | grep x") is False
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
cd /home/adam/eva && uv run pytest tests/test_safety.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'eva.safety'`.

- [ ] **Step 1.3: Implement SafetyGate**

Create `/home/adam/eva/eva/safety.py`:

```python
"""Решает, нужно ли голосовое подтверждение перед запуском shell-команды."""

_SHELL_METACHARS = (";", "&", "|", ">", "<", "$(", "`", "\n")


class SafetyGate:
    """Команда считается безопасной если её первая лексема (имя бинарника)
    есть в allowlist И в команде нет shell-метасимволов. Единственное
    исключение: одиночный трейлинг `&` (бэкграунд-запуск) — разрешён."""

    def __init__(self, safe_prefixes: tuple[str, ...]):
        self._safe = set(safe_prefixes)

    def is_safe(self, command: str) -> bool:
        stripped = command.strip()
        if not stripped:
            return True  # пустая команда — нечего выполнять, нечего бояться

        # Отрезаем разрешённый трейлинг `&`, остаток проверяем на метасимволы.
        # `firefox &` → head = "firefox" → метасимволов нет → ok.
        # `firefox && rm` → head = "firefox && rm" → содержит & → не safe.
        head = stripped[:-1].rstrip() if stripped.endswith("&") else stripped
        if any(meta in head for meta in _SHELL_METACHARS):
            return False

        first_token = head.split()[0]
        return first_token in self._safe
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
cd /home/adam/eva && uv run pytest tests/test_safety.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
cd /home/adam/eva && git add eva/safety.py tests/test_safety.py && git commit -m "feat(safety): add SafetyGate with allowlist + shell-metachar check"
```

---

## Task 2: Config fields

**Files:**
- Modify: `/home/adam/eva/eva/config.py`
- Modify: `/home/adam/eva/tests/test_config.py`

- [ ] **Step 2.1: Add failing assertions to the existing test**

Open `/home/adam/eva/tests/test_config.py`. Find `test_load_uses_defaults_when_env_unset` and add three new assertions at the end (after the existing `assert cfg.conversation_timeout_sec == 60.0` line):

```python
    assert "firefox" in cfg.safe_command_prefixes
    assert "да" in cfg.confirm_yes_words
    assert "нет" in cfg.confirm_no_words
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
    assert "firefox" in cfg.safe_command_prefixes
    assert "да" in cfg.confirm_yes_words
    assert "нет" in cfg.confirm_no_words
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
cd /home/adam/eva && uv run pytest tests/test_config.py::test_load_uses_defaults_when_env_unset -v
```

Expected: FAIL with `AttributeError: 'Config' object has no attribute 'safe_command_prefixes'`.

- [ ] **Step 2.3: Add the fields to Config**

In `/home/adam/eva/eva/config.py`, find the `# Разговорный режим` block near the bottom (with `conversation_timeout_sec: float = 60.0`). ADD a new section right after it, before `@classmethod`:

```python
    # Разговорный режим — сколько секунд после последнего обмена сессия живёт
    conversation_timeout_sec: float = 60.0

    # Безопасность shell-команд: allowlist префиксов и слова для подтверждения
    safe_command_prefixes: tuple[str, ...] = (
        # Открывалки файлов/папок/URL
        "xdg-open", "gio",
        # GUI-приложения
        "firefox", "chromium", "google-chrome", "code", "nautilus",
        "gnome-terminal", "vlc", "telegram-desktop", "spotify",
        # Медиа-контроль и громкость
        "playerctl", "amixer", "pactl",
    )
    confirm_yes_words: tuple[str, ...] = (
        "да", "ага", "конечно", "подтверждаю", "давай", "хорошо",
    )
    confirm_no_words: tuple[str, ...] = (
        "нет", "отмена", "не надо", "не нужно", "стоп",
    )

    @classmethod
    def load(cls, ...
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
cd /home/adam/eva && uv run pytest tests/test_config.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
cd /home/adam/eva && git add eva/config.py tests/test_config.py && git commit -m "feat(config): add safe_command_prefixes + confirm yes/no words"
```

---

## Task 3: Assistant — pending command + confirmation flow

**Files:**
- Modify: `/home/adam/eva/eva/assistant.py` (full rewrite — content below)
- Modify: `/home/adam/eva/tests/test_assistant_logic.py` (append 5 new tests)

Самая большая задача в этом плане — но это всё ещё одна логическая единица: обработка нового состояния pending_command.

- [ ] **Step 3.1: Write 5 failing tests**

Open `/home/adam/eva/tests/test_assistant_logic.py`. Keep ALL existing imports, helpers, and 16 existing tests intact.

**At the END of the file**, append:

```python
def test_safe_command_runs_immediately():
    # firefox в allowlist — должна выполниться без подтверждения
    asst, _, _ = make_assistant_with_clock(brain_response=[
        ResponseDelta(sentence="Открываю."),
        ResponseDelta(command="firefox", done=True),
    ])
    asst.handle_text("ева открой firefox")
    asst.executor.run.assert_called_once_with("firefox")
    assert asst.pending_command is None


def test_unsafe_command_sets_pending_not_runs():
    # shutdown не в allowlist — не должна выполниться, ставится pending
    asst, _, _ = make_assistant_with_clock(brain_response=[
        ResponseDelta(sentence="Сейчас выключу."),
        ResponseDelta(command="shutdown now", done=True),
    ])
    asst.handle_text("ева выключи компьютер")
    asst.executor.run.assert_not_called()
    assert asst.pending_command == "shutdown now"
    # Последняя say-реплика должна быть "Подтверди."
    last_said = asst.synthesizer.say.call_args_list[-1].args[0]
    assert last_said == "Подтверди."


def test_yes_word_runs_pending():
    asst, _, _ = make_assistant_with_clock(brain_response=[
        ResponseDelta(sentence="Удаляю."),
        ResponseDelta(command="rm -rf /tmp/foo", done=True),
    ])
    asst.handle_text("ева удали /tmp/foo")
    assert asst.pending_command == "rm -rf /tmp/foo"  # ждём подтверждения
    asst.handle_text("да")
    asst.executor.run.assert_called_once_with("rm -rf /tmp/foo")
    assert asst.pending_command is None
    # После "да" Ева сказала "Хорошо."
    last_said = asst.synthesizer.say.call_args_list[-1].args[0]
    assert last_said == "Хорошо."


def test_no_word_cancels_pending():
    asst, _, _ = make_assistant_with_clock(brain_response=[
        ResponseDelta(sentence="Удаляю."),
        ResponseDelta(command="rm -rf /tmp/foo", done=True),
    ])
    asst.handle_text("ева удали /tmp/foo")
    asst.handle_text("нет")
    asst.executor.run.assert_not_called()
    assert asst.pending_command is None
    last_said = asst.synthesizer.say.call_args_list[-1].args[0]
    assert last_said == "Отменено."


def test_no_wins_when_both_yes_and_no_present():
    # «да, отмени» — содержит и yes-word и no-word; no должен победить
    asst, _, _ = make_assistant_with_clock(brain_response=[
        ResponseDelta(sentence="Удаляю."),
        ResponseDelta(command="rm -rf /tmp/foo", done=True),
    ])
    asst.handle_text("ева удали /tmp/foo")
    asst.handle_text("да, отмени")
    asst.executor.run.assert_not_called()
    assert asst.pending_command is None
    last_said = asst.synthesizer.say.call_args_list[-1].args[0]
    assert last_said == "Отменено."


def test_pending_cleared_when_session_expires():
    # pending живёт только пока сессия активна; после истечения — забывается
    asst, _, clock = make_assistant_with_clock(
        timeout=60.0,
        brain_response=[
            ResponseDelta(sentence="Удаляю."),
            ResponseDelta(command="rm -rf /tmp/foo", done=True),
        ],
    )
    asst.handle_text("ева удали /tmp/foo")
    assert asst.pending_command == "rm -rf /tmp/foo"
    clock.advance(61)  # сессия истекла
    asst.handle_text("да")
    # Команда НЕ выполнена — pending был очищен по истечению сессии
    asst.executor.run.assert_not_called()
    assert asst.pending_command is None
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
cd /home/adam/eva && uv run pytest tests/test_assistant_logic.py -v
```

Expected: the 16 OLD tests still PASS. The 5 NEW tests FAIL with `AttributeError: 'Assistant' object has no attribute 'pending_command'`.

- [ ] **Step 3.3: Rewrite Assistant with pending + confirmation logic**

Open `/home/adam/eva/eva/assistant.py`. REPLACE the entire file contents with:

```python
import logging
import time
from typing import Callable

from eva.audio import AudioCapture, SpeechSegmenter
from eva.brain import Brain
from eva.config import Config
from eva.executor import ShellExecutor
from eva.safety import SafetyGate
from eva.stt import Transcriber
from eva.tts import Synthesizer

log = logging.getLogger(__name__)


class Assistant:
    """Главный оркестратор. Связывает все компоненты, держит флаги сессии
    (`sleeping`, `running`, `conversation_until`, `pending_command`) и
    реализует логику wake/sleep/exit слов, разговорный режим и подтверждение
    опасных команд.

    `run()` крутит главный цикл: получает utterance из segmenter,
    транскрибирует, прогоняет через `handle_text()`.

    `handle_text()` — отдельный метод для unit-тестирования логики
    без аудио-потока. Источник времени `time_source` инжектится для
    детерминистских тестов (в проде — `time.monotonic`)."""

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
        self._gate = SafetyGate(config.safe_command_prefixes)
        self.sleeping = False
        self.running = True
        # Разговорный режим: timestamp монотонного времени, до которого
        # сессия активна. None — сессии нет.
        self.conversation_until: float | None = None
        # Команда, ожидающая голосового подтверждения. None — нет pending.
        self.pending_command: str | None = None
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

        # 1. Exit — высший приоритет, любое состояние
        if any(w in text for w in self._cfg.exit_words):
            self.synthesizer.say("Выключаюсь. Пока!")
            # stop() и флаг и сегментер останавливает — иначе main loop
            # висит в segments() пока не придёт следующий чанк аудио.
            self.stop()
            return

        # 2. Sleep — отменяет pending перед тем как уйти в молчание
        if not self.sleeping and any(w in text for w in self._cfg.sleep_words):
            self.synthesizer.say(
                "Молчу. Скажи 'Ева, проснись' когда понадоблюсь."
            )
            self.pending_command = None
            self.sleeping = True
            return

        # 3. Wake-again — выходим из sleep
        if self.sleeping and any(w in text for w in self._cfg.wake_again_words):
            self.synthesizer.say("Я снова с тобой.")
            self.sleeping = False
            return

        # 4. В sleep — всё ниже игнорируется
        if self.sleeping:
            return

        # 5. Если сессия истекла — pending тоже мёртв
        if (self.pending_command is not None
                and not self._conversation_active()):
            self.pending_command = None

        # 6. Если есть pending — текущая реплика ТОЛЬКО ответ на подтверждение.
        #    Не отправляется в Brain. Безопасный дефолт: если непонятно — нет.
        if self.pending_command is not None:
            has_no = any(w in text for w in self._cfg.confirm_no_words)
            has_yes = any(w in text for w in self._cfg.confirm_yes_words)
            cmd = self.pending_command
            self.pending_command = None
            if has_yes and not has_no:
                self.synthesizer.say("Хорошо.")
                self.executor.run(cmd)
            else:
                # Явный no, или непонятно, или «да, отмени» — отменяем
                self.synthesizer.say("Отменено.")
            return

        # 7. Обычный путь: wake-word или активная сессия
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
            if self._gate.is_safe(command_to_run):
                self.executor.run(command_to_run)
            else:
                # Опасная команда — запоминаем, ждём голосового «да/нет»
                self.pending_command = command_to_run
                self.synthesizer.say("Подтверди.")
```

Key behavior changes from the previous version:
- Imports `SafetyGate` and instantiates `self._gate` in `__init__`.
- New attribute `pending_command: str | None`.
- `handle_text` has new numbered structure with two new branches:
  - Block 5: clear pending if session expired
  - Block 6: handle pending as yes/no answer, don't fall through to Brain
- Sleep branch (block 2) now also clears pending.
- `_ask_brain_and_respond` now gates on `SafetyGate.is_safe`: safe → run, unsafe → set pending + say "Подтверди.".

- [ ] **Step 3.4: Run the new tests**

```bash
cd /home/adam/eva && uv run pytest tests/test_assistant_logic.py -v
```

Expected: all 22 tests PASS (16 old + 6 new).

- [ ] **Step 3.5: Run the full suite**

```bash
cd /home/adam/eva && uv run pytest -v
```

Expected: 49 tests PASS (3 config + 3 segmenter + 7 parser + 7 brain + 22 assistant + 7 safety).

If `test_command_is_executed_when_present` fails — check that its `command="firefox &"` is properly recognized as safe by the new gate. (It should be: `firefox` is in the default allowlist, `&` is a trailing background marker.) If the gate logic is wrong, debug there, NOT by editing the test.

- [ ] **Step 3.6: Commit**

```bash
cd /home/adam/eva && git add eva/assistant.py tests/test_assistant_logic.py && git commit -m "feat(assistant): require voice confirmation for unsafe shell commands"
```

---

## Task 4: Smoke test + merge to main

User-driven step. The whole point of this feature is to prevent another PC-shutdown incident — so the smoke test should explicitly verify that case.

**Files:** none (verification + git operations)

- [ ] **Step 4.1: Run Eva and verify command safety**

```bash
cd /home/adam/eva && DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY .venv/bin/eva --debug
```

Manual checks (user, on real mic):
1. «Ева, открой firefox» → Firefox открывается БЕЗ подтверждения.
2. «Ева, выключи компьютер» → Ева говорит описание + «Подтверди.» → **компьютер НЕ выключается**. На «нет» → «Отменено.», ничего не происходит.
3. Повтори (2), но скажи «да» — компьютер должен выключиться. (Готов перезагружаться, конечно.)
4. «Ева, удали файл /tmp/test» → «Подтверди.» (rm не в allowlist).
5. Существующие команды (открывалки, плеер) работают как раньше.
6. Голосовые меты «замолчи/проснись/выключись» работают.

Если на (2) компьютер всё-таки выключился — НЕ мерджить. В `--debug` логах смотреть вызовы `_gate.is_safe()`.

- [ ] **Step 4.2: Merge feat/command-safety → main**

After smoke test green:

```bash
cd /home/adam/eva && git checkout main && git merge --no-ff feat/command-safety -m "Merge branch 'feat/command-safety'"
```

- [ ] **Step 4.3: Push to origin**

```bash
cd /home/adam/eva && git push origin main
```

Если push не пройдёт из-за auth (как в предыдущих ветках) — push делает пользователь.

- [ ] **Step 4.4: Delete the merged branch**

```bash
cd /home/adam/eva && git branch -d feat/command-safety
```

---

## Definition of Done

- 4 задачи выполнены.
- 49 тестов PASS (`uv run pytest`).
- Smoke test подтвердил: «выключи компьютер» НЕ выключает компьютер без «да».
- `feat/command-safety` смерджена в `main` через `--no-ff` и удалена.
- `main` запушен в `origin`.
