# Eva — оптимизация производительности и переход на модульный ООП-пакет

**Дата:** 2026-05-27
**Статус:** draft (ожидает реализации)

## Цель

Сделать голосовой ассистент Eva заметно отзывчивее и точнее на ноутбуке пользователя (Intel i5-1335U, 8 ГБ RAM, Iris Xe, без CUDA), не теряя текущую функциональность. Параллельно превратить однофайловый скрипт в установленный Python-пакет с консольной командой `eva`.

## Контекст

Текущая реализация — единственный файл [eva.py](../../../eva.py), ~280 строк. Стек: openai-whisper (`small`) → DeepSeek API → Piper TTS. Wake-word детектится через подстроку в выводе Whisper, активность голоса — через порог амплитуды, ответ DeepSeek читается целиком перед началом озвучки.

Пользователь жалуется на три симптома:
1. Длинная пауза между фразой и ответом
2. Whisper неверно слышит слова (особенно технический жаргон)
3. Иногда реальная речь игнорируется (отсекается по громкости или режется посередине)

Дополнительные требования: разбить на модули с ООП, добавить команду `eva` в консоли.

## Обзор решения

Заменяем три ключевых блока движков:

- **STT:** `openai-whisper` → `faster-whisper` с моделью `small` в `int8`. На AVX-VNNI это даёт 3-5× ускорение при том же качестве. Передаём numpy-массив напрямую, без записи tmp .wav. Используем `initial_prompt` с релевантным жаргоном.
- **VAD:** амплитудный порог → **Silero VAD**. Точное определение начала/конца речи, перестаёт игнорировать тихую речь и резать фразы посередине.
- **LLM/TTS pipeline:** ответ DeepSeek приходит стримом, как только готово первое предложение в поле `say` — оно сразу уходит в Piper. Первый звук слышен через ~1 сек вместо 3-5.

Параллельно: код реорганизуем в пакет `eva/` с разделением ответственностей по классам. Управление зависимостями через uv.

## Архитектура

### Поток данных

```
mic
 ↓ (PCM-блоки по 32 мс)
AudioCapture (queue)
 ↓
SpeechSegmenter (Silero VAD: 32ms окна → start/end of speech)
 ↓ (numpy-массив целой фразы)
Transcriber (faster-whisper, int8, initial_prompt)
 ↓ (текст)
Assistant.handle_command
 ↓ (wake-word / sleep / exit checks)
Brain.ask_stream (DeepSeek streaming → накопление JSON → парсинг)
 ↓ (delta событий: предложение say-текста, готовая shell-команда)
Synthesizer.say_stream  +  ShellExecutor.run (параллельно)
 ↓
Piper → aplay (по предложениям, pipeline)
```

### Структура пакета

```
/home/adam/eva/
├── pyproject.toml         # метаданные, зависимости, entry point eva = "eva.cli:main"
├── README.md              # обновлённый (uv install, eva command)
├── uv.lock                # генерируется uv
├── eva/                   # сам пакет
│   ├── __init__.py
│   ├── __main__.py        # позволяет `python -m eva`
│   ├── cli.py             # main(): argparse, логирование, signal handlers, запуск Assistant
│   ├── config.py          # @dataclass(frozen=True) Config + Config.load()
│   ├── audio.py           # AudioCapture, SpeechSegmenter
│   ├── stt.py             # Transcriber
│   ├── tts.py             # Synthesizer
│   ├── brain.py           # Brain, ResponseDelta (dataclass)
│   ├── executor.py        # ShellExecutor
│   └── assistant.py       # Assistant (оркестратор + sleeping/running)
├── tests/
│   ├── test_brain.py
│   ├── test_assistant_logic.py
│   └── test_segmenter.py
├── piper/                 # без изменений (бинарь Piper + espeak-ng)
├── voices/                # без изменений (русский голос)
└── docs/superpowers/specs/2026-05-27-eva-performance-and-modularization-design.md
```

Старый `venv/` удаляется, новый создаётся через `uv venv` в `.venv/`.

### Классы и ответственности

| Класс | Файл | Ответственность | Публичный интерфейс |
|---|---|---|---|
| `Config` | config.py | Все настройки одним immutable объектом. Загрузка из env + дефолтов | `Config.load() -> Config` |
| `AudioCapture` | audio.py | Открывает `sd.InputStream`, кладёт PCM-блоки в `queue.Queue` | `start()`, `stop()`, `queue: Queue[np.ndarray]` |
| `SpeechSegmenter` | audio.py | Читает блоки из AudioCapture, прогоняет через Silero VAD, отдаёт готовые utterance | `segments() -> Iterator[np.ndarray]` |
| `Transcriber` | stt.py | Обёртка над faster-whisper. Хранит модель в памяти, принимает numpy | `transcribe(audio: np.ndarray) -> str` |
| `Synthesizer` | tts.py | Обёртка над Piper. Стриминговая озвучка по предложениям | `say(text: str)`, `say_stream(sentences: Iterator[str])` |
| `Brain` | brain.py | DeepSeek-клиент. Держит conversation history. Streaming + парсинг JSON | `ask_stream(user_text: str) -> Iterator[ResponseDelta]` |
| `ShellExecutor` | executor.py | Запуск shell-команд от LLM | `run(command: str)` |
| `Assistant` | assistant.py | Орекстратор: wire-up всех частей, состояние sleeping/running, wake/sleep/exit логика | `run()`, `stop()` |

### Streaming-протокол `Brain` → `Assistant` → `Synthesizer`

`Brain.ask_stream` накапливает chunks от OpenAI, инкрементально парсит ответ и отдаёт события через простой dataclass:

```python
@dataclass
class ResponseDelta:
    sentence: str | None = None      # очередное готовое предложение для озвучки
    command: str | None = None       # shell-команда (приходит один раз в конце)
    done: bool = False               # последний дельта в стриме
    error: str | None = None         # сообщение об ошибке (если есть)
```

Подход к инкрементальному парсингу: LLM по-прежнему возвращает JSON, но мы меняем системный промпт чтобы поле `say` шло **первым**, команда **второй**. На каждом новом чанке пробуем найти готовый префикс `say` (regex по началу строки `{"say":"..."`). Как только видим целое предложение внутри `say` — отдаём его как `ResponseDelta(sentence=...)`, продолжаем накапливать. Команду берём после полного парса всего JSON и отдаём как `ResponseDelta(command=..., done=True)`.

Если LLM проигнорировал инструкцию по порядку полей (`command` пришёл раньше `say`) — стриминг просто не сработает, накапливаем весь ответ и парсим целиком в конце. Это деградация по скорости, не по корректности.

Если JSON не парсится вообще — отдаём `ResponseDelta(error=..., done=True)`, и Assistant произносит «не поняла, повтори».

## Зависимости

`pyproject.toml`:

```toml
[project]
name = "eva-assistant"
version = "0.1.0"
description = "Голосовой ассистент Eva на Whisper + DeepSeek + Piper"
requires-python = ">=3.10"
dependencies = [
    "faster-whisper>=1.0",
    "silero-vad>=5.0",
    "sounddevice>=0.4",
    "scipy>=1.10",
    "numpy>=1.24",
    "openai>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-mock"]

[project.scripts]
eva = "eva.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

Удаляются: `openai-whisper`, torch (если он стоял только ради whisper — silero-vad тоже требует torch, поэтому может остаться как транзитивная зависимость).

## CLI и установка

Команды установки (все через uv):

```bash
cd /home/adam/eva
rm -rf venv                              # старый pyenv-venv больше не нужен
uv venv                                  # создаёт .venv/ с Python из uv (или системным)
uv pip install -e ".[dev]"               # editable install + тестовые зависимости
ln -sf /home/adam/eva/.venv/bin/eva ~/.local/bin/eva
```

После этого в любой консоли:

```bash
eva                       # обычный запуск (использует Python из .venv через shebang)
eva --debug               # подробные логи через logging
eva --model medium        # переопределение модели Whisper
eva --no-wake             # отвечать без wake-word (отладка)
python -m eva             # альтернативный запуск
```

`DEEPSEEK_API_KEY` читается из окружения как и сейчас. Если ключа нет — `cli.main()` падает с понятной ошибкой до старта Assistant.

## Конфигурация (Config dataclass)

```python
@dataclass(frozen=True)
class Config:
    # пути
    eva_dir: Path
    piper_bin: Path
    voice_model: Path
    whisper_model_name: str = "small"

    # аудио
    sample_rate: int = 16000
    block_ms: int = 32

    # VAD
    vad_threshold: float = 0.5
    silence_end_ms: int = 700           # сколько тишины = конец фразы
    min_speech_ms: int = 300

    # слова
    wake_words: tuple[str, ...] = ("ева", "эва", "eva", "ява")
    sleep_words: tuple[str, ...] = ("замолчи", "спи", "тихо")
    wake_again_words: tuple[str, ...] = ("проснись", "слушай")
    exit_words: tuple[str, ...] = ("выключись", "выйди", "стоп", "остановись", "завершись")

    # LLM
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    history_window: int = 10

    # отладка
    debug: bool = False
    require_wake: bool = True
```

## Обработка ошибок

| Сценарий | Поведение |
|---|---|
| Нет `DEEPSEEK_API_KEY` | `cli.main` выходит с кодом 1 и понятным сообщением до старта Assistant |
| Нет интернета / ошибка DeepSeek API | `Brain.ask_stream` отдаёт `ResponseDelta(error="...", done=True)`; Assistant говорит «связь пропала, попробуй позже»; история не загрязняется |
| LLM вернул невалидный JSON | то же — Assistant говорит «не поняла, повтори»; в `--debug` пишется сырой ответ |
| Whisper выдал пустую строку или один шум | Assistant молча игнорит, цикл продолжается |
| Piper / aplay упали (subprocess.CalledProcessError) | warning в stderr, цикл продолжается — Ева не валится из-за одной озвучки |
| Микрофон отвалился (sounddevice exception) | `AudioCapture.start()` пробует переоткрыть один раз; если снова падает — exit 2 |
| Ctrl+C / SIGTERM | `cli` ставит флаг, `Assistant.stop()` корректно закрывает стрим, ждёт worker-потоки и выходит |

В `--debug` логирование на DEBUG (через стандартный `logging`), без него — WARNING+.

## Тестирование

Добавляем минимальный набор unit-тестов **только на нашу логику склейки**. Стороннюю STT/LLM/TTS не тестируем — сетевые и хрупкие.

- **`tests/test_brain.py`** — `Brain` с замоканным OpenAI-клиентом:
  - валидный JSON со streaming-чанками → правильно нарезанные `sentence`-дельты
  - JSON в markdown-обёртке (```json ... ```) → парсится корректно
  - сломанный JSON → один `ResponseDelta(error=..., done=True)`
  - история обрезается до `history_window` последних сообщений
- **`tests/test_assistant_logic.py`** — `Assistant` с заглушками всех зависимостей:
  - sleep/wake transitions по голосовым командам
  - exit-слова устанавливают `running = False`
  - wake-word отрезается из начала фразы перед отправкой в Brain
  - в sleeping-режиме всё кроме wake_again_words игнорится
- **`tests/test_segmenter.py`** — `SpeechSegmenter` с искусственным потоком numpy-блоков и фейковым VAD-callable (`lambda x: True/False`):
  - правильно склеивает соседние speech-блоки в одну utterance
  - режет фразу когда тишина превышает `silence_end_ms`
  - игнорирует utterance короче `min_speech_ms`

Запуск: `uv run pytest`. Целевой результат — `pytest` зелёный после рефакторинга, на старом-новом тестовом наборе.

## Что НЕ входит в этот спек

- Dedicated wake-word engine (openWakeWord). Откладываем — закроем в отдельной итерации если ложные срабатывания станут проблемой после A.
- Долгосрочная память между сессиями. Отдельный проект.
- Function calling вместо JSON. Отдельный проект.
- Безопасность shell-команд (allowlist, dry-run). Отдельный проект.
- Тесты на сам faster-whisper / Piper / DeepSeek.

## Критерии приёмки

1. `eva` запускается из любой консоли, отвечает на «Ева, открой firefox» и реально открывает Firefox.
2. На той же фразе латентность от конца реплики до первого звука Евы ≤ 1.5 сек (раньше ~3-5 сек).
3. Whisper-ошибки на технических словах («Firefox», «папка загрузки», «терминал») снижены за счёт `initial_prompt`.
4. Реальная тихая речь больше не игнорируется (Silero VAD против амплитудного порога).
5. `uv run pytest` зелёный.
6. `eva --help` показывает все CLI-опции.
7. Существующие голосовые команды («замолчи»/«проснись»/«выключись») работают как раньше.
