# Terminal Startup Banner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** При запуске `eva` показывать скруглённую рамку (rich) с логотипом, статусом (модель, микрофон, wake-word, память) и подсказкой; во время загрузки моделей — спиннер.

**Architecture:** Новый модуль `eva/banner.py` отвечает только за отрисовку и ничего не грузит. `cli.main` оборачивает блок загрузки компонентов в `console.status(...)` (спиннер) и после него печатает панель из `build_panel(config, memory, version)`.

**Tech Stack:** Python 3.11, `rich` (новая зависимость), `sounddevice` (уже есть), `pytest`, `uv`.

Спек: `docs/superpowers/specs/2026-05-29-startup-banner-design.md`. Ветка: `feat/startup-banner` (уже создана).

---

### Task 1: Добавить зависимость rich

**Files:**
- Modify: `pyproject.toml` (через `uv add`, правит автоматически)

- [ ] **Step 1: Установить rich через uv**

Run: `uv add rich`
Expected: `rich` появляется в `[project].dependencies` в `pyproject.toml`, обновляется окружение.

- [ ] **Step 2: Проверить импорт**

Run: `uv run python -c "from rich.console import Console; from rich.panel import Panel; from rich.text import Text; print('rich ok')"`
Expected: выводит `rich ok` без ошибок.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add rich dependency for startup banner"
```

(Если `uv` тронул `uv.lock` — он в `.gitignore`, не коммитим.)

---

### Task 2: Помощник `_plural` (русское множественное число)

**Files:**
- Create: `eva/banner.py`
- Test: `tests/test_banner.py`

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_banner.py`:

```python
from eva.banner import _plural


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
```

- [ ] **Step 2: Запустить — убедиться что падает**

Run: `uv run pytest tests/test_banner.py -q`
Expected: ошибка коллекции — `ModuleNotFoundError: No module named 'eva.banner'`.

- [ ] **Step 3: Создать `eva/banner.py` с `_plural`**

```python
import logging

log = logging.getLogger(__name__)


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Русская форма множественного числа: 1 факт, 2 факта, 5 фактов."""
    if 11 <= n % 100 <= 14:
        return many
    d = n % 10
    if d == 1:
        return one
    if 2 <= d <= 4:
        return few
    return many
```

- [ ] **Step 4: Запустить — убедиться что зелёные**

Run: `uv run pytest tests/test_banner.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add eva/banner.py tests/test_banner.py
git commit -m "feat(banner): add Russian plural helper"
```

---

### Task 3: `build_panel` — сборка панели статуса

**Files:**
- Modify: `eva/banner.py`
- Test: `tests/test_banner.py`

- [ ] **Step 1: Дописать падающие тесты**

Добавить в начало `tests/test_banner.py` (после строки `from eva.banner import _plural`):

```python
from pathlib import Path
from unittest.mock import MagicMock

from rich.console import Console

from eva.banner import build_panel
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
```

Добавить в конец `tests/test_banner.py`:

```python
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
```

- [ ] **Step 2: Запустить — убедиться что падает**

Run: `uv run pytest tests/test_banner.py -q`
Expected: ошибка коллекции модуля — `ImportError: cannot import name 'build_panel' from 'eva.banner'` (импорт в начале файла валит сбор всех тестов, включая `_plural`).

- [ ] **Step 3: Реализовать `build_panel` и `_input_device_name`**

Дописать в `eva/banner.py` (вверху — импорты, ниже — функции):

```python
import sounddevice as sd
from rich.panel import Panel
from rich.text import Text

ACCENT = "bright_cyan"  # акцентный цвет рамки и имени; легко поменять


def _input_device_name() -> str:
    """Имя дефолтного входного устройства; best-effort, с фолбэком."""
    try:
        info = sd.query_devices(kind="input")
        name = info.get("name") if isinstance(info, dict) else None
        return name or "по умолчанию"
    except Exception as exc:
        log.debug("Не смогла определить микрофон: %s", exc)
        return "по умолчанию"


def build_panel(config, memory, version: str, *, mic_name: str | None = None) -> Panel:
    if mic_name is None:
        mic_name = _input_device_name()

    n = len(memory.all())
    facts = f"{n} {_plural(n, 'факт', 'факта', 'фактов')}" if n else "пусто"
    wake = ", ".join(f"«{w}»" for w in config.wake_words[:2])
    model = f"Whisper {config.whisper_model_name} ({config.whisper_compute_type})"

    body = Text()
    body.append("✦ Ева", style=f"bold {ACCENT}")
    body.append(f" · голосовой ассистент    v{version}\n\n", style="dim")
    body.append(f"  Модель    {model}\n")
    body.append(f"  Микрофон  {mic_name}\n")
    body.append(f"  Будит     {wake}\n")
    body.append(f"  Память    {facts}\n\n")
    body.append("  «ева, открой firefox» · «ева, запомни …»\n", style="dim")
    body.append("  «спи» — пауза · «выключись» — выход", style="dim")

    return Panel(body, border_style=ACCENT, expand=False, padding=(0, 1))
```

Итоговая шапка импортов в `eva/banner.py`:

```python
import logging

import sounddevice as sd
from rich.panel import Panel
from rich.text import Text
```

- [ ] **Step 4: Запустить — убедиться что зелёные**

Run: `uv run pytest tests/test_banner.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add eva/banner.py tests/test_banner.py
git commit -m "feat(banner): build rich status panel"
```

---

### Task 4: Проводка в `cli.main` — спиннер + панель

**Files:**
- Modify: `eva/cli.py`

(Юнит-тестов нет: `cli.main` — оркестрация/IO, как и сейчас без тестов. Проверка — полный прогон сьюта + ручной запуск.)

- [ ] **Step 1: Добавить импорты rich и banner**

В `eva/cli.py`, внутри `main()`, в блоке отложенных импортов найти:

```python
    from eva.brain import Brain
    from eva.config import Config
    from eva.executor import ShellExecutor
    from eva.memory import Memory
    from eva.stt import Transcriber
    from eva.tts import Synthesizer
```

Заменить на:

```python
    from rich.console import Console

    from eva.banner import build_panel
    from eva.brain import Brain
    from eva.config import Config
    from eva.executor import ShellExecutor
    from eva.memory import Memory
    from eva.stt import Transcriber
    from eva.tts import Synthesizer
```

- [ ] **Step 2: Обернуть загрузку в спиннер и напечатать панель**

В `eva/cli.py` найти блок (от `log.info("Загружаю компоненты...")` до создания `assistant`):

```python
    log.info("Загружаю компоненты...")
    capture = AudioCapture(
        sample_rate=config.sample_rate,
        chunk_samples=config.chunk_samples,
    )
    vad_iterator = make_silero_iterator(
        threshold=config.vad_threshold,
        min_silence_ms=config.vad_min_silence_ms,
        speech_pad_ms=config.vad_speech_pad_ms,
        sample_rate=config.sample_rate,
    )
    segmenter = SpeechSegmenter(
        capture, vad_iterator,
        min_speech_ms=config.vad_min_speech_ms,
        sample_rate=config.sample_rate,
    )
    transcriber = Transcriber(
        model_name=config.whisper_model_name,
        compute_type=config.whisper_compute_type,
        initial_prompt=config.whisper_initial_prompt,
    )
    synthesizer = Synthesizer(
        piper_bin=config.piper_bin,
        voice_model=config.voice_model,
    )
    memory = Memory(config.memory_path)
    brain = Brain(config, memory)
    executor = ShellExecutor()

    assistant = Assistant(
        config=config, capture=capture, segmenter=segmenter,
        transcriber=transcriber, synthesizer=synthesizer,
        brain=brain, executor=executor,
    )
```

Заменить на:

```python
    console = Console()
    with console.status("Загружаю Еву…", spinner="dots"):
        capture = AudioCapture(
            sample_rate=config.sample_rate,
            chunk_samples=config.chunk_samples,
        )
        vad_iterator = make_silero_iterator(
            threshold=config.vad_threshold,
            min_silence_ms=config.vad_min_silence_ms,
            speech_pad_ms=config.vad_speech_pad_ms,
            sample_rate=config.sample_rate,
        )
        segmenter = SpeechSegmenter(
            capture, vad_iterator,
            min_speech_ms=config.vad_min_speech_ms,
            sample_rate=config.sample_rate,
        )
        transcriber = Transcriber(
            model_name=config.whisper_model_name,
            compute_type=config.whisper_compute_type,
            initial_prompt=config.whisper_initial_prompt,
        )
        synthesizer = Synthesizer(
            piper_bin=config.piper_bin,
            voice_model=config.voice_model,
        )
        memory = Memory(config.memory_path)
        brain = Brain(config, memory)
        executor = ShellExecutor()

    console.print(build_panel(config, memory, __version__))

    assistant = Assistant(
        config=config, capture=capture, segmenter=segmenter,
        transcriber=transcriber, synthesizer=synthesizer,
        brain=brain, executor=executor,
    )
```

(`__version__` уже импортируется вверху `eva/cli.py`: `from eva import __version__`.)

- [ ] **Step 3: Проверить, что cli импортируется и весь сьют зелёный**

Run: `uv run python -c "import eva.cli; print('cli import ok')" && uv run pytest -q`
Expected: `cli import ok`, затем `69 passed`.

- [ ] **Step 4: Проверить, что --help/--version не падают (быстрый путь без загрузки)**

Run: `uv run eva --version && uv run eva --help`
Expected: печатает `eva 0.1.0` и текст помощи, без трейсбеков.

- [ ] **Step 5: Ручная проверка (выполняет пользователь)**

Запустить `eva` с заданным `DEEPSEEK_API_KEY` и микрофоном: должен показаться спиннер «Загружаю Еву…», затем скруглённая рамка со статусом и подсказкой, после чего Ева начинает слушать. Дополнительно `eva | cat` (не-TTY) не должен падать и должен выдать читаемый текст без мусора анимации.

- [ ] **Step 6: Commit**

```bash
git add eva/cli.py
git commit -m "feat(cli): show startup banner with loading spinner"
```

---

## Критерии готовности

- `eva/banner.py` создан: `_plural`, `_input_device_name`, `build_panel`.
- `tests/test_banner.py`: 8 тестов зелёные.
- Полный сьют: 69 passed (61 прежних + 8 новых).
- `rich` в `pyproject.toml`.
- `cli.main` показывает спиннер при загрузке и панель после неё; `--version`/`--help` работают.

## Хендофф

После реализации — ветка `feat/startup-banner` готова к ручной проверке и затем к PR (пуш и мердж делает пользователь, см. рабочий процесс проекта).
