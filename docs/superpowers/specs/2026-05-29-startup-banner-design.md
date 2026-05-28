# Eva — красивый стартовый экран в терминале

**Дата:** 2026-05-29
**Статус:** draft (ожидает реализации)
**Ветка:** `feat/startup-banner`

## Цель

При запуске `eva` в терминале показывать аккуратный стартовый экран — скруглённую
рамку с именем ассистента, версией, статусом (модель, микрофон, wake-word, память)
и подсказкой по голосовым командам. В духе приветственного окна Claude Code. Сейчас
при запуске видны только сухие логи (`log.info("Загружаю компоненты...")`).

## Контекст

`cli.main` синхронно грузит тяжёлые компоненты (Whisper, Silero, Piper) несколько
секунд, потом запускает `Assistant.run()`. Часть статуса (имя микрофона, число
фактов в памяти) известна только после инициализации, поэтому панель рисуется
**после** загрузки. Во время загрузки крутится спиннер.

Выбор из брейнсторма: наполнение — **логотип + статус + подсказка**; стиль —
**скруглённая рамка** с акцентным цветом; во время загрузки — **спиннер, затем
панель**.

## Решение

Новая зависимость `rich` (чистый Python, без нативных зависимостей, быстрый импорт)
— она даёт скруглённые рамки, цвет и корректный расчёт ширины. Новый модуль
`eva/banner.py` отвечает **только за отрисовку** и ничего не грузит. `cli.main`
оборачивает блок загрузки в `console.status(...)` (спиннер) и после него печатает
панель. Остальные компоненты не меняются.

## Архитектура

### Зависимость

`uv add rich` (управление зависимостями — через uv). Импортируется внутри `cli.main()` (не на
верхнем уровне модуля), чтобы `--version`/`--help` оставались мгновенными.

### Новый модуль `eva/banner.py`

```python
import logging

import sounddevice as sd
from rich.panel import Panel
from rich.text import Text

log = logging.getLogger(__name__)

ACCENT = "bright_cyan"  # акцентный цвет рамки и имени; легко поменять


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
    body.append(f"✦ Ева", style=f"bold {ACCENT}")
    body.append(f" · голосовой ассистент    v{version}\n\n", style="dim")
    body.append(f"  Модель    {model}\n")
    body.append(f"  Микрофон  {mic_name}\n")
    body.append(f"  Будит     {wake}\n")
    body.append(f"  Память    {facts}\n\n")
    body.append("  «ева, открой firefox» · «ева, запомни …»\n", style="dim")
    body.append("  «спи» — пауза · «выключись» — выход", style="dim")

    return Panel(body, border_style=ACCENT, expand=False, padding=(0, 1))
```

(Точные отступы/формулировки — на усмотрение реализации; важна структура: заголовок,
блок статуса, строка-подсказка.)

### Изменения в `cli.main`

```python
from rich.console import Console
from eva.banner import build_panel
from eva.memory import Memory

console = Console()
with console.status("Загружаю Еву…", spinner="dots"):
    capture = AudioCapture(...)
    vad_iterator = make_silero_iterator(...)
    segmenter = SpeechSegmenter(...)
    transcriber = Transcriber(...)
    synthesizer = Synthesizer(...)
    memory = Memory(config.memory_path)
    brain = Brain(config, memory)
    executor = ShellExecutor()

console.print(build_panel(config, memory, __version__))

assistant = Assistant(...)
...
assistant.run()
```

Строку `log.info("Загружаю компоненты...")` убираем — её заменяет спиннер.

## Поток данных

```
$ eva
  → cli.main: console.status("Загружаю Еву…") крутит спиннер
    → грузятся Whisper / Silero / Piper / Memory (несколько секунд)
  → блок with завершился → спиннер исчез
  → build_panel(config, memory, __version__) собирает Panel
    → запрос имени микрофона (best-effort)
    → число фактов из memory.all()
  → console.print(panel) → скруглённая рамка со статусом
  → assistant.run() начинает слушать
```

## Обработка ошибок

| Сценарий | Поведение |
|---|---|
| Вывод не в TTY (пайп/файл) | rich сам деградирует: без анимации спиннера, печатает рамку/текст как обычный текст |
| `sd.query_devices` упал/нет устройства | `_input_device_name` логирует debug и возвращает «по умолчанию» |
| Память пустая (0 фактов) | показываем «пусто» вместо «0 фактов» |
| `--debug` | логи идут в stderr через logging; панель — в stdout; не конфликтуют |

## Тестирование

### `tests/test_banner.py`

Рендер Panel в текст через `Console(record=True)` → `console.print(panel)` →
`console.export_text()`, затем проверка подстрок. `mic_name` передаём явно, чтобы
тесты не зависели от железа. `memory` — `MagicMock` с заданным `all()` или реальный
`Memory(tmp_path)`.

1. `test_panel_contains_version` — в тексте есть `v0.1.0` (или переданная версия).
2. `test_panel_contains_model_and_wake_word` — есть «Whisper small» и «ева».
3. `test_panel_shows_fact_count` — память с 4 фактами → в тексте «4 факта».
4. `test_panel_empty_memory_says_empty` — память пустая → «пусто», нет «0 фактов».
5. `test_plural_one` — `_plural(1, ...)` → «факт».
6. `test_plural_few` — `_plural(2, ...)` → «факта»; `_plural(4, ...)` → «факта».
7. `test_plural_many` — `_plural(5, ...)` → «фактов»; `_plural(11, ...)` → «фактов».
8. `test_mic_name_injected` — переданное `mic_name="тест"` попадает в текст.

Спиннер и проводку `cli.main` не юнит-тестим (анимация/IO; cli и сейчас без тестов).

## Критерии приёмки

1. `eva` при запуске показывает спиннер «Загружаю Еву…», затем скруглённую рамку со
   статусом и подсказкой.
2. В рамке корректные значения: версия, модель Whisper, имя микрофона, wake-words,
   число фактов (с правильной русской формой).
3. Запуск с выводом в пайп (`eva | cat`) не падает и выдаёт читаемый текст без
   мусорной анимации.
4. Тесты зелёные: текущие 61 + новые из `test_banner.py` (8).
5. `rich` добавлен в `pyproject.toml` через `uv add`.

## Что НЕ входит

- Живой статус-бар во время работы (постоянная панель, пока Ева слушает)
- Флаг `--no-banner` / отключение экрана
- Настройка цвета/темы через конфиг
- ASCII-арт логотип (выбран вариант со скруглённой рамкой, не вордмарк)
- Прогресс-бар с этапами загрузки (просто спиннер)
