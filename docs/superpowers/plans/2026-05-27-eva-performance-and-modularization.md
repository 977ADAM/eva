# Eva Performance and Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-file `eva.py` with a modular OOP Python package installable as the `eva` console command. Swap STT to `faster-whisper` + Silero VAD, add streaming TTS pipeline so first audio appears in ~1 sec.

**Architecture:** Eight focused classes (Config, AudioCapture, SpeechSegmenter, Transcriber, Synthesizer, Brain, ShellExecutor, Assistant) split across `eva/` package modules. `cli.main()` wires them together. Streaming flow: mic → Silero VAD → faster-whisper → DeepSeek (stream) → JSON-parsed sentence deltas → Piper TTS per sentence.

**Tech Stack:** Python 3.11+, uv (package manager), faster-whisper (CTranslate2 int8), silero-vad (ONNX), openai SDK (DeepSeek base_url, streaming), Piper TTS (existing binary), pytest.

**Spec:** [docs/superpowers/specs/2026-05-27-eva-performance-and-modularization-design.md](../specs/2026-05-27-eva-performance-and-modularization-design.md)

---

## File Structure

```
/home/adam/eva/
├── pyproject.toml                     [Task 1]
├── uv.lock                            [Task 1, generated]
├── .venv/                             [Task 1, generated]
├── eva/
│   ├── __init__.py                    [Task 1]
│   ├── __main__.py                    [Task 1]
│   ├── cli.py                         [Task 1 stub → Task 11]
│   ├── config.py                      [Task 2]
│   ├── executor.py                    [Task 3]
│   ├── tts.py                         [Task 4]
│   ├── stt.py                         [Task 5]
│   ├── audio.py                       [Tasks 6, 7]
│   ├── brain.py                       [Tasks 8, 9]
│   └── assistant.py                   [Task 10]
├── tests/
│   ├── __init__.py                    [Task 1]
│   ├── test_config.py                 [Task 2]
│   ├── test_segmenter.py              [Task 7]
│   ├── test_streaming_say_parser.py   [Task 8]
│   ├── test_brain.py                  [Task 9]
│   └── test_assistant_logic.py        [Task 10]
├── piper/                             [no changes]
├── voices/                            [no changes]
├── eva.py                             [delete at Task 12]
├── venv/                              [delete at Task 1]
└── README.md                          [update at Task 12]
```

**Per-file responsibility:**
- `config.py` — single `Config` frozen dataclass + `Config.load()` (env + defaults). No logic.
- `executor.py` — `ShellExecutor.run(cmd)` — fire-and-forget subprocess.
- `tts.py` — `Synthesizer.say(text)` + `say_stream(iter)` — Piper subprocess wrapper.
- `stt.py` — `Transcriber.transcribe(audio_np) -> str` — faster-whisper wrapper.
- `audio.py` — `AudioCapture` (sounddevice stream → queue), `SpeechSegmenter` (queue + VAD → utterances), `make_silero_iterator()` factory.
- `brain.py` — `Brain.ask_stream(text) -> Iterator[ResponseDelta]` + `StreamingSayParser` (incremental JSON `say` extraction) + `SYSTEM_PROMPT` constant.
- `assistant.py` — `Assistant` orchestrator with `run()` / `stop()` / `handle_text()` (the last is the testable unit).
- `cli.py` — `main()` — argparse, logging setup, signal handlers, Config.load(), Assistant lifecycle.

---

## Task 1: Project scaffolding with uv

**Files:**
- Create: `/home/adam/eva/pyproject.toml`
- Create: `/home/adam/eva/eva/__init__.py`
- Create: `/home/adam/eva/eva/__main__.py`
- Create: `/home/adam/eva/eva/cli.py` (stub)
- Create: `/home/adam/eva/tests/__init__.py`
- Create: `/home/adam/eva/.gitignore`
- Delete: `/home/adam/eva/venv/`

- [ ] **Step 1: Remove the old pip-created venv**

```bash
cd /home/adam/eva && rm -rf venv
```

- [ ] **Step 2: Create pyproject.toml**

Write `/home/adam/eva/pyproject.toml`:

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
    "torch>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-mock"]

[project.scripts]
eva = "eva.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["eva*"]
exclude = ["tests*", "piper*", "voices*", "venv*", ".venv*", "docs*"]
```

- [ ] **Step 3: Create the package skeleton**

Write `/home/adam/eva/eva/__init__.py`:

```python
__version__ = "0.1.0"
```

Write `/home/adam/eva/eva/__main__.py`:

```python
from eva.cli import main

if __name__ == "__main__":
    main()
```

Write `/home/adam/eva/eva/cli.py` (stub — full version in Task 11):

```python
import argparse

from eva import __version__


def main() -> int:
    parser = argparse.ArgumentParser(prog="eva", description="Голосовой ассистент Eva")
    parser.add_argument("--version", action="version", version=f"eva {__version__}")
    parser.add_argument("--debug", action="store_true", help="Подробные логи")
    parser.add_argument("--model", default="small", help="Модель Whisper (small|medium)")
    parser.add_argument("--no-wake", action="store_true", help="Отвечать без wake-word")
    parser.parse_args()
    print("eva: stub — реализация в Task 11")
    return 0
```

Write `/home/adam/eva/tests/__init__.py`:

```python
```

Write `/home/adam/eva/.gitignore`:

```
.venv/
venv/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
/tmp/
uv.lock
```

- [ ] **Step 4: Create venv and install with uv**

```bash
cd /home/adam/eva && uv venv --python 3.11 && uv sync --extra dev
```

Expected: creates `.venv/` and installs all deps + the `eva` editable package. Output ends with "Installed N packages".

- [ ] **Step 5: Verify console command works**

```bash
cd /home/adam/eva && .venv/bin/eva --help
```

Expected output:
```
usage: eva [-h] [--version] [--debug] [--model MODEL] [--no-wake]
Голосовой ассистент Eva
...
```

```bash
cd /home/adam/eva && .venv/bin/eva --version
```

Expected: `eva 0.1.0`

- [ ] **Step 6: Initialize git and make first commit**

```bash
cd /home/adam/eva && git init -b main && git add pyproject.toml .gitignore eva/ tests/ piper/ voices/ docs/ README.md eva.py && git commit -m "chore: scaffold eva package with uv, keep old eva.py temporarily"
```

Expected: commit succeeds. Note: keeping old `eva.py` for the moment so we can reference it; deleted in Task 12.

---

## Task 2: Config dataclass

**Files:**
- Create: `/home/adam/eva/eva/config.py`
- Create: `/home/adam/eva/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Write `/home/adam/eva/tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/adam/eva && uv run pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'eva.config'`.

- [ ] **Step 3: Implement Config**

Write `/home/adam/eva/eva/config.py`:

```python
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Пути
    eva_dir: Path
    piper_bin: Path
    voice_model: Path

    # LLM
    deepseek_api_key: str
    deepseek_model: str = "deepseek-chat"
    history_window: int = 10

    # STT
    whisper_model_name: str = "small"
    whisper_compute_type: str = "int8"
    whisper_initial_prompt: str = (
        "Ева. Linux, Ubuntu, Firefox, терминал, файл, папка, "
        "открой, закрой, запусти, выключи."
    )

    # Аудио
    sample_rate: int = 16000
    chunk_samples: int = 512  # 32 ms at 16 kHz — Silero requirement

    # VAD
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 700
    vad_min_speech_ms: int = 300
    vad_speech_pad_ms: int = 100

    # Голосовые команды
    wake_words: tuple[str, ...] = ("ева", "эва", "eva", "ява")
    sleep_words: tuple[str, ...] = ("замолчи", "спи", "тихо")
    wake_again_words: tuple[str, ...] = ("проснись", "слушай")
    exit_words: tuple[str, ...] = (
        "выключись", "выйди", "стоп", "остановись", "завершись",
    )

    # Поведение
    debug: bool = False
    require_wake: bool = True

    @classmethod
    def load(cls, *, debug: bool = False, require_wake: bool = True,
             whisper_model_name: str = "small") -> "Config":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "Переменная окружения DEEPSEEK_API_KEY не задана"
            )
        home = Path.home()
        eva_dir = home / "eva"
        return cls(
            eva_dir=eva_dir,
            piper_bin=eva_dir / "piper" / "piper",
            voice_model=eva_dir / "voices" / "ru_RU-irina-medium.onnx",
            deepseek_api_key=api_key,
            whisper_model_name=whisper_model_name,
            debug=debug,
            require_wake=require_wake,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/adam/eva && uv run pytest tests/test_config.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/adam/eva && git add eva/config.py tests/test_config.py && git commit -m "feat(config): add Config frozen dataclass with env loading"
```

---

## Task 3: ShellExecutor

**Files:**
- Create: `/home/adam/eva/eva/executor.py`

No tests — it's a 5-line subprocess wrapper, mocking subprocess just re-implements the wrapper.

- [ ] **Step 1: Implement ShellExecutor**

Write `/home/adam/eva/eva/executor.py`:

```python
import logging
import subprocess

log = logging.getLogger(__name__)


class ShellExecutor:
    """Запускает shell-команды от LLM в фоне (fire-and-forget)."""

    def run(self, command: str) -> None:
        if not command:
            return
        log.info("Выполняю: %s", command)
        try:
            subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            log.warning("Ошибка выполнения команды %r: %s", command, exc)
```

- [ ] **Step 2: Sanity check the import**

```bash
cd /home/adam/eva && uv run python -c "from eva.executor import ShellExecutor; ShellExecutor().run('true')"
```

Expected: no output, no error, exit code 0.

- [ ] **Step 3: Commit**

```bash
cd /home/adam/eva && git add eva/executor.py && git commit -m "feat(executor): add ShellExecutor for LLM shell commands"
```

---

## Task 4: Synthesizer (Piper wrapper)

**Files:**
- Create: `/home/adam/eva/eva/tts.py`

No unit tests — wraps Piper binary. Verified by smoke test in Task 12.

- [ ] **Step 1: Implement Synthesizer**

Write `/home/adam/eva/eva/tts.py`:

```python
import logging
import subprocess
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


class Synthesizer:
    """Озвучивает текст через Piper TTS. По одному предложению за вызов
    `say()`; `say_stream()` принимает итератор готовых предложений
    и проигрывает их по мере поступления."""

    def __init__(self, piper_bin: Path, voice_model: Path,
                 tmp_wav: Path = Path("/tmp/eva_out.wav")):
        self._piper_bin = piper_bin
        self._voice_model = voice_model
        self._tmp_wav = tmp_wav

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        print(f"🔊 Ева: {text}")
        try:
            subprocess.run(
                [str(self._piper_bin),
                 "--model", str(self._voice_model),
                 "--output_file", str(self._tmp_wav)],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
                check=True,
            )
        except Exception as exc:
            log.warning("Piper упал: %s", exc)
            return
        try:
            subprocess.run(
                ["aplay", "-q", str(self._tmp_wav)],
                check=False,
            )
        except Exception as exc:
            log.warning("aplay упал: %s", exc)

    def say_stream(self, sentences: Iterable[str]) -> None:
        for sentence in sentences:
            self.say(sentence)
```

- [ ] **Step 2: Sanity check the import**

```bash
cd /home/adam/eva && uv run python -c "from eva.tts import Synthesizer; from pathlib import Path; Synthesizer(Path('/tmp'), Path('/tmp'))"
```

Expected: no output, no error.

- [ ] **Step 3: Commit**

```bash
cd /home/adam/eva && git add eva/tts.py && git commit -m "feat(tts): add Synthesizer wrapping Piper with say_stream"
```

---

## Task 5: Transcriber (faster-whisper wrapper)

**Files:**
- Create: `/home/adam/eva/eva/stt.py`

No unit tests — wraps faster-whisper, model loading is expensive. Verified end-to-end in smoke test.

- [ ] **Step 1: Implement Transcriber**

Write `/home/adam/eva/eva/stt.py`:

```python
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

log = logging.getLogger(__name__)


class Transcriber:
    """Обёртка над faster-whisper. Принимает numpy float32 mono 16kHz,
    возвращает текст в нижнем регистре."""

    def __init__(self, model_name: str, compute_type: str,
                 initial_prompt: str, language: str = "ru"):
        from faster_whisper import WhisperModel
        log.info("Загружаю faster-whisper модель %s (%s)...",
                 model_name, compute_type)
        self._model: "WhisperModel" = WhisperModel(
            model_name,
            device="cpu",
            compute_type=compute_type,
        )
        self._initial_prompt = initial_prompt
        self._language = language

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=1,
            initial_prompt=self._initial_prompt,
            vad_filter=False,
        )
        text = "".join(seg.text for seg in segments).strip().lower()
        return text
```

- [ ] **Step 2: Sanity check the import (no model loading)**

```bash
cd /home/adam/eva && uv run python -c "from eva.stt import Transcriber; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
cd /home/adam/eva && git add eva/stt.py && git commit -m "feat(stt): add Transcriber wrapping faster-whisper int8"
```

---

## Task 6: AudioCapture

**Files:**
- Create: `/home/adam/eva/eva/audio.py`

No unit tests for `AudioCapture` itself — it's a thin sounddevice wrapper. `SpeechSegmenter` in Task 7 is the testable unit.

- [ ] **Step 1: Implement AudioCapture (only)**

Write `/home/adam/eva/eva/audio.py` (SpeechSegmenter and Silero factory added in Task 7):

```python
import logging
import queue

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class AudioCapture:
    """Открывает микрофон через sounddevice, кладёт PCM-блоки фиксированного
    размера в `self.queue`. Размер блока соответствует требованиям Silero VAD
    (512 семплов = 32 мс при 16 kHz)."""

    def __init__(self, sample_rate: int, chunk_samples: int):
        self._sample_rate = sample_rate
        self._chunk_samples = chunk_samples
        self.queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        try:
            self._open_stream()
        except Exception as exc:
            log.warning("Не удалось открыть микрофон: %s. Пробую ещё раз...",
                        exc)
            # Вторая попытка; если снова упадёт — exception улетит в cli.main()
            # и Eva завершится с exit code 2 (это поведение покрыто в Task 11).
            self._open_stream()

    def _open_stream(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._chunk_samples,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def drain(self) -> None:
        """Очистить буфер — используется после того как Ева сама что-то
        сказала, чтобы не пытаться распознать собственный голос."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.debug("sounddevice status: %s", status)
        self.queue.put(indata.copy().flatten())
```

- [ ] **Step 2: Sanity check the import**

```bash
cd /home/adam/eva && uv run python -c "from eva.audio import AudioCapture; print(AudioCapture(16000, 512))"
```

Expected: prints `<eva.audio.AudioCapture object at 0x...>`.

- [ ] **Step 3: Commit**

```bash
cd /home/adam/eva && git add eva/audio.py && git commit -m "feat(audio): add AudioCapture sounddevice wrapper"
```

---

## Task 7: SpeechSegmenter with Silero VAD

**Files:**
- Modify: `/home/adam/eva/eva/audio.py`
- Create: `/home/adam/eva/tests/test_segmenter.py`

- [ ] **Step 1: Write the failing tests**

Write `/home/adam/eva/tests/test_segmenter.py`:

```python
import queue
import threading

import numpy as np
import pytest

from eva.audio import AudioCapture, SpeechSegmenter


def make_capture_with_chunks(chunks):
    """Создаёт AudioCapture-подобный объект с заранее заполненной очередью."""
    capture = AudioCapture(sample_rate=16000, chunk_samples=512)
    for ch in chunks:
        capture.queue.put(ch)
    return capture


def fake_vad_callable(events):
    """Возвращает callable, который для каждого вызова отдаёт следующий
    событийный dict из списка (или None)."""
    events_iter = iter(events)
    def call(_chunk):
        try:
            return next(events_iter)
        except StopIteration:
            return None
    return call


def test_emits_utterance_between_start_and_end():
    chunks = [np.ones(512, dtype=np.float32) for _ in range(5)]
    vad = fake_vad_callable([
        {"start": 0},
        None,
        None,
        None,
        {"end": 0},
    ])
    capture = make_capture_with_chunks(chunks)
    seg = SpeechSegmenter(capture, vad, min_speech_ms=10, sample_rate=16000)

    # Запускаем segments() в потоке, забираем первую utterance, останавливаем
    results = []
    def consume():
        for utt in seg.segments():
            results.append(utt)
            seg.stop()
            return
    t = threading.Thread(target=consume)
    t.start()
    t.join(timeout=2.0)
    assert len(results) == 1
    assert len(results[0]) == 5 * 512


def test_drops_utterance_shorter_than_min_speech():
    chunks = [np.ones(512, dtype=np.float32) for _ in range(2)]
    vad = fake_vad_callable([
        {"start": 0},
        {"end": 0},
    ])
    capture = make_capture_with_chunks(chunks)
    # min_speech_ms требует 5 chunks (5 * 32 ms = 160 ms), у нас 2 (64 ms)
    seg = SpeechSegmenter(capture, vad, min_speech_ms=160, sample_rate=16000)

    results = []
    done = threading.Event()
    def consume():
        for utt in seg.segments():
            results.append(utt)
        done.set()
    t = threading.Thread(target=consume)
    t.start()
    # Дать сегментеру время обработать оба chunk и не отдать ничего
    import time; time.sleep(0.5)
    seg.stop()
    t.join(timeout=2.0)
    assert results == []


def test_ignores_silence_before_speech():
    chunks = [np.zeros(512, dtype=np.float32) for _ in range(3)] + \
             [np.ones(512, dtype=np.float32) for _ in range(3)]
    vad = fake_vad_callable([
        None, None, None,                # тишина
        {"start": 0}, None, {"end": 0},  # речь
    ])
    capture = make_capture_with_chunks(chunks)
    seg = SpeechSegmenter(capture, vad, min_speech_ms=10, sample_rate=16000)

    results = []
    def consume():
        for utt in seg.segments():
            results.append(utt)
            seg.stop()
            return
    t = threading.Thread(target=consume)
    t.start()
    t.join(timeout=2.0)
    assert len(results) == 1
    # 3 chunks речи * 512 семплов
    assert len(results[0]) == 3 * 512
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/adam/eva && uv run pytest tests/test_segmenter.py -v
```

Expected: FAIL with `ImportError: cannot import name 'SpeechSegmenter'`.

- [ ] **Step 3: Append SpeechSegmenter + Silero factory to audio.py**

Append to `/home/adam/eva/eva/audio.py`:

```python
from typing import Callable, Iterator


class SpeechSegmenter:
    """Читает PCM-блоки из AudioCapture.queue, прогоняет каждый через
    VAD-callable (возвращает {'start': t} / {'end': t} / None), накапливает
    речь между start и end, отдаёт каждую готовую фразу как numpy-массив.

    VAD-callable инжектится: в тестах — фейк, в проде — обёртка над
    silero_vad.VADIterator (см. make_silero_iterator)."""

    def __init__(self, capture: AudioCapture,
                 vad_iterator: Callable[[np.ndarray], dict | None],
                 *, min_speech_ms: int = 300, sample_rate: int = 16000):
        self._capture = capture
        self._iterator = vad_iterator
        self._min_speech_samples = int(min_speech_ms / 1000 * sample_rate)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def segments(self) -> Iterator[np.ndarray]:
        buffer: list[np.ndarray] = []
        speaking = False

        while not self._stop:
            try:
                chunk = self._capture.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            event = self._iterator(chunk)

            if event and "start" in event:
                speaking = True
                buffer = [chunk]
            elif event and "end" in event:
                if speaking:
                    buffer.append(chunk)
                    utterance = np.concatenate(buffer)
                    if len(utterance) >= self._min_speech_samples:
                        yield utterance
                buffer = []
                speaking = False
            elif speaking:
                buffer.append(chunk)


def make_silero_iterator(*, threshold: float, min_silence_ms: int,
                         speech_pad_ms: int, sample_rate: int = 16000
                         ) -> Callable[[np.ndarray], dict | None]:
    """Фабрика: загружает Silero и возвращает callable, который SpeechSegmenter
    может вызывать на каждом chunk."""
    import torch
    from silero_vad import VADIterator, load_silero_vad

    model = load_silero_vad()
    raw = VADIterator(
        model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )

    def wrapped(chunk: np.ndarray) -> dict | None:
        return raw(torch.from_numpy(chunk.copy()), return_seconds=False)

    return wrapped
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/adam/eva && uv run pytest tests/test_segmenter.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/adam/eva && git add eva/audio.py tests/test_segmenter.py && git commit -m "feat(audio): add SpeechSegmenter with injectable VAD + Silero factory"
```

---

## Task 8: StreamingSayParser

**Files:**
- Create: `/home/adam/eva/eva/brain.py` (parser only — full Brain class in Task 9)
- Create: `/home/adam/eva/tests/test_streaming_say_parser.py`

- [ ] **Step 1: Write the failing tests**

Write `/home/adam/eva/tests/test_streaming_say_parser.py`:

```python
import pytest

from eva.brain import StreamingSayParser


def feed_all(parser, chunks):
    out = []
    for c in chunks:
        out.extend(parser.feed(c))
    return out


def test_extracts_single_sentence_from_one_chunk():
    parser = StreamingSayParser()
    result = feed_all(parser, ['{"say":"Привет, мир.","action":"talk"}'])
    assert result == ["Привет, мир."]


def test_waits_for_sentence_terminator():
    parser = StreamingSayParser()
    assert feed_all(parser, ['{"say":"Прив']) == []
    assert feed_all(parser, ['ет, мир.']) == ["Привет, мир."]


def test_emits_multiple_sentences_in_order():
    parser = StreamingSayParser()
    result = feed_all(parser, [
        '{"say":"Один. Два! Три?","action":"talk"}',
    ])
    assert result == ["Один.", "Два!", "Три?"]


def test_emits_remainder_on_close_quote():
    parser = StreamingSayParser()
    # Текст без точки в конце; закрывающая кавычка `"` должна вытолкнуть остаток
    result = feed_all(parser, ['{"say":"Готово","action":"talk"}'])
    assert result == ["Готово"]


def test_no_say_field_yields_nothing():
    parser = StreamingSayParser()
    # command идёт первым — стриминг не сработает
    result = feed_all(parser, ['{"command":"firefox","action":"shell"}'])
    assert result == []
    assert not parser.emitted_anything()


def test_chunk_boundary_inside_say_prefix():
    parser = StreamingSayParser()
    # Префикс `"say":"` бьётся пополам
    assert feed_all(parser, ['{"sa']) == []
    assert feed_all(parser, ['y":"Готово.","action":"talk"}']) == ["Готово."]


def test_emitted_anything_flag():
    parser = StreamingSayParser()
    assert not parser.emitted_anything()
    feed_all(parser, ['{"say":"Привет.","action":"talk"}'])
    assert parser.emitted_anything()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/adam/eva && uv run pytest tests/test_streaming_say_parser.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'eva.brain'`.

- [ ] **Step 3: Implement StreamingSayParser**

Write `/home/adam/eva/eva/brain.py` (full Brain class added in Task 9):

```python
import re
from typing import Iterator


_SAY_PREFIX_RE = re.compile(r'"say"\s*:\s*"')
_SENTENCE_END_RE = re.compile(r'[.!?\n]+\s*')


class StreamingSayParser:
    """Инкрементально извлекает готовые предложения из поля `say` JSON-ответа
    LLM, пока ответ ещё стримится по кусочкам.

    Работает в предположении, что `say` идёт ПЕРВЫМ полем в JSON (это
    зашито в SYSTEM_PROMPT). Если LLM нарушил порядок и `say` пришёл после
    `command`/`action` — парсер просто не вернёт ничего, и Brain отрендерит
    весь `say` целиком из полного JSON.

    Также делает упрощающее предположение: внутри `say` нет литеральных
    `"` (LLM их экранирует или вообще не использует в русском тексте).
    Если кавычка всё-таки появится — мы посчитаем её закрытием поля; ничего
    страшного не произойдёт, парс полного JSON в Brain всё равно отработает."""

    def __init__(self):
        self._pre_buffer = ""        # текст до того как нашли `"say":"`
        self._extracting = False     # true когда уже внутри значения `say`
        self._say_text = ""          # накопленное содержимое `say`
        self._emitted_to = 0         # индекс в _say_text до которого уже отдано
        self._closed = False         # увидели закрывающую кавычку поля
        self._emit_count = 0

    def feed(self, chunk: str) -> Iterator[str]:
        if not self._extracting:
            self._pre_buffer += chunk
            match = _SAY_PREFIX_RE.search(self._pre_buffer)
            if not match:
                return
            self._extracting = True
            chunk = self._pre_buffer[match.end():]
            self._pre_buffer = ""

        if self._closed:
            return

        quote_idx = chunk.find('"')
        if quote_idx == -1:
            self._say_text += chunk
        else:
            self._say_text += chunk[:quote_idx]
            self._closed = True

        yield from self._extract_sentences()

        if self._closed:
            remainder = self._say_text[self._emitted_to:].strip()
            if remainder:
                self._emit_count += 1
                self._emitted_to = len(self._say_text)
                yield remainder

    def _extract_sentences(self) -> Iterator[str]:
        text = self._say_text[self._emitted_to:]
        last_end = 0
        for m in _SENTENCE_END_RE.finditer(text):
            end = m.end()
            sentence = text[last_end:end].strip()
            if sentence:
                self._emit_count += 1
                yield sentence
            last_end = end
        self._emitted_to += last_end

    def emitted_anything(self) -> bool:
        return self._emit_count > 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/adam/eva && uv run pytest tests/test_streaming_say_parser.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/adam/eva && git add eva/brain.py tests/test_streaming_say_parser.py && git commit -m "feat(brain): add StreamingSayParser for incremental JSON 'say' extraction"
```

---

## Task 9: Brain (DeepSeek streaming)

**Files:**
- Modify: `/home/adam/eva/eva/brain.py`
- Create: `/home/adam/eva/tests/test_brain.py`

- [ ] **Step 1: Write the failing tests**

Write `/home/adam/eva/tests/test_brain.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/adam/eva && uv run pytest tests/test_brain.py -v
```

Expected: FAIL with `ImportError: cannot import name 'Brain'`.

- [ ] **Step 3: Implement Brain**

Append to `/home/adam/eva/eva/brain.py`:

```python
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from openai import OpenAI

if TYPE_CHECKING:
    from eva.config import Config

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """Тебя зовут Ева. Ты голосовой ассистент в Ubuntu Linux.
Отвечай ОЧЕНЬ коротко (1-2 предложения), потому что твой ответ озвучивается.

Возвращай ТОЛЬКО валидный JSON, без markdown, без ```.

ВАЖНО: поле "say" должно идти ПЕРВЫМ в JSON — это нужно для того чтобы
ассистент начал озвучивать ответ пока ты ещё его дописываешь.

Форматы:
- Действие: {"say":"что сказать вслух","action":"shell","command":"shell-команда"}
- Разговор: {"say":"ответ","action":"talk"}

Примеры:
- "открой firefox" → {"say":"Открываю Firefox","action":"shell","command":"firefox &"}
- "как дела" → {"say":"Всё хорошо, чем помочь?","action":"talk"}
- "открой загрузки" → {"say":"Открыла загрузки","action":"shell","command":"xdg-open ~/Downloads"}"""


@dataclass
class ResponseDelta:
    sentence: str | None = None
    command: str | None = None
    done: bool = False
    error: str | None = None


class Brain:
    """DeepSeek-клиент со стриминговым извлечением предложений из поля `say`."""

    def __init__(self, config: "Config"):
        self._config = config
        self._client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url="https://api.deepseek.com",
        )
        self._history: list[dict] = []

    def ask_stream(self, user_text: str) -> Iterator[ResponseDelta]:
        self._history.append({"role": "user", "content": user_text})
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + self._history[-self._config.history_window:]
        )

        try:
            stream = self._client.chat.completions.create(
                model=self._config.deepseek_model,
                messages=messages,
                temperature=0.7,
                stream=True,
            )
        except Exception as exc:
            log.warning("DeepSeek API error: %s", exc)
            yield ResponseDelta(error=str(exc), done=True)
            return

        parser = StreamingSayParser()
        full_parts: list[str] = []

        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                full_parts.append(delta)
                for sentence in parser.feed(delta):
                    yield ResponseDelta(sentence=sentence)
        except Exception as exc:
            log.warning("DeepSeek stream error: %s", exc)
            yield ResponseDelta(error=str(exc), done=True)
            return

        full_text = "".join(full_parts).strip()
        cleaned = self._strip_markdown_fence(full_text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            log.warning("Невалидный JSON от LLM: %r", cleaned)
            yield ResponseDelta(error=f"invalid json: {exc}", done=True)
            return

        self._history.append({"role": "assistant", "content": cleaned})

        # Fallback: если стриминг не выдал предложений (LLM поставил command
        # раньше say или весь say поместился в одном фрагменте без
        # терминатора), выдаём весь say одним delta.
        if not parser.emitted_anything():
            say = (data.get("say") or "").strip()
            if say:
                yield ResponseDelta(sentence=say)

        command = None
        if data.get("action") == "shell":
            command = data.get("command")

        yield ResponseDelta(command=command, done=True)

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        if not text.startswith("```"):
            return text
        parts = text.split("```")
        if len(parts) < 2:
            return text
        body = parts[1]
        if body.startswith("json"):
            body = body[4:]
        return body.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/adam/eva && uv run pytest tests/test_brain.py tests/test_streaming_say_parser.py -v
```

Expected: all tests PASS (7 parser + 7 brain = 14).

- [ ] **Step 5: Commit**

```bash
cd /home/adam/eva && git add eva/brain.py tests/test_brain.py && git commit -m "feat(brain): add Brain with DeepSeek streaming + JSON sentence deltas"
```

---

## Task 10: Assistant orchestrator

**Files:**
- Create: `/home/adam/eva/eva/assistant.py`
- Create: `/home/adam/eva/tests/test_assistant_logic.py`

- [ ] **Step 1: Write the failing tests**

Write `/home/adam/eva/tests/test_assistant_logic.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eva.assistant import Assistant
from eva.brain import ResponseDelta
from eva.config import Config


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/adam/eva && uv run pytest tests/test_assistant_logic.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'eva.assistant'`.

- [ ] **Step 3: Implement Assistant**

Write `/home/adam/eva/eva/assistant.py`:

```python
import logging

from eva.audio import AudioCapture, SpeechSegmenter
from eva.brain import Brain
from eva.config import Config
from eva.executor import ShellExecutor
from eva.stt import Transcriber
from eva.tts import Synthesizer

log = logging.getLogger(__name__)


class Assistant:
    """Главный оркестратор. Связывает все компоненты, держит флаги sessии
    (`sleeping`, `running`) и реализует логику wake/sleep/exit слов.

    `run()` крутит главный цикл: получает utterance из segmenter,
    транскрибирует, прогоняет через `handle_text()`.

    `handle_text()` — отдельный метод для возможности unit-тестирования
    логики без аудио-потока."""

    def __init__(self, *, config: Config, capture: AudioCapture,
                 segmenter: SpeechSegmenter, transcriber: Transcriber,
                 synthesizer: Synthesizer, brain: Brain,
                 executor: ShellExecutor):
        self._cfg = config
        self.capture = capture
        self.segmenter = segmenter
        self.transcriber = transcriber
        self.synthesizer = synthesizer
        self.brain = brain
        self.executor = executor
        self.sleeping = False
        self.running = True

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

        if self._cfg.require_wake:
            has_wake = any(w in text for w in self._cfg.wake_words)
            if not has_wake:
                return
            clean = self._strip_wake_word(text)
        else:
            clean = self._strip_wake_word(text)

        if not clean:
            self.synthesizer.say("Слушаю.")
            return

        self._ask_brain_and_respond(clean)

    def _strip_wake_word(self, text: str) -> str:
        for w in self._cfg.wake_words:
            if text.startswith(w):
                return text[len(w):].strip(" ,.;:!?")
        return text

    def _ask_brain_and_respond(self, prompt: str) -> None:
        command_to_run: str | None = None
        had_error = False
        for delta in self.brain.ask_stream(prompt):
            if delta.error:
                had_error = True
                self.synthesizer.say("Связь пропала, попробуй позже.")
                return
            if delta.sentence:
                self.synthesizer.say(delta.sentence)
            if delta.done and delta.command:
                command_to_run = delta.command
        if command_to_run and not had_error:
            self.executor.run(command_to_run)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/adam/eva && uv run pytest tests/test_assistant_logic.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Run the full suite to make sure nothing regressed**

```bash
cd /home/adam/eva && uv run pytest -v
```

Expected: all tests across all files PASS (config 3 + segmenter 3 + parser 7 + brain 7 + assistant 11 = 31).

- [ ] **Step 6: Commit**

```bash
cd /home/adam/eva && git add eva/assistant.py tests/test_assistant_logic.py && git commit -m "feat(assistant): add Assistant orchestrator with handle_text logic"
```

---

## Task 11: CLI wire-up + signal handling

**Files:**
- Modify: `/home/adam/eva/eva/cli.py`

- [ ] **Step 1: Replace cli.py stub with the full implementation**

Overwrite `/home/adam/eva/eva/cli.py`:

```python
import argparse
import logging
import os
import signal
import sys

from eva import __version__


def main() -> int:
    # Снимаем proxy-переменные перед импортом сетевых клиентов —
    # они мешали запросам к DeepSeek (вынесено из старого eva.py)
    for var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                "HTTPS_PROXY", "https_proxy", "SOCKS_PROXY", "socks_proxy"):
        os.environ.pop(var, None)

    parser = argparse.ArgumentParser(
        prog="eva",
        description="Голосовой ассистент Eva",
    )
    parser.add_argument("--version", action="version", version=f"eva {__version__}")
    parser.add_argument("--debug", action="store_true", help="Подробные логи")
    parser.add_argument("--model", default="small",
                        help="Модель Whisper (small|medium)")
    parser.add_argument("--no-wake", action="store_true",
                        help="Отвечать без wake-word (отладка)")
    args = parser.parse_args()

    _setup_logging(args.debug)
    log = logging.getLogger("eva")

    # Импорты тяжёлых модулей внутри main() чтобы --help/--version были быстрыми
    from eva.assistant import Assistant
    from eva.audio import AudioCapture, SpeechSegmenter, make_silero_iterator
    from eva.brain import Brain
    from eva.config import Config
    from eva.executor import ShellExecutor
    from eva.stt import Transcriber
    from eva.tts import Synthesizer

    try:
        config = Config.load(
            debug=args.debug,
            require_wake=not args.no_wake,
            whisper_model_name=args.model,
        )
    except RuntimeError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1

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
    brain = Brain(config)
    executor = ShellExecutor()

    assistant = Assistant(
        config=config, capture=capture, segmenter=segmenter,
        transcriber=transcriber, synthesizer=synthesizer,
        brain=brain, executor=executor,
    )

    def _signal_handler(signum, _frame):
        print(f"\n⚙️  Получен сигнал {signum}, останавливаю Еву...")
        assistant.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        assistant.run()
    except Exception as exc:
        log.exception("Фатальная ошибка: %s", exc)
        return 2
    return 0


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
```

- [ ] **Step 2: Verify --help still works without loading heavy deps**

```bash
cd /home/adam/eva && .venv/bin/eva --help
```

Expected: help text appears in < 1 sec (heavy imports are deferred).

- [ ] **Step 3: Verify --version works**

```bash
cd /home/adam/eva && .venv/bin/eva --version
```

Expected: `eva 0.1.0`.

- [ ] **Step 4: Verify missing API key gives a clean error**

```bash
cd /home/adam/eva && env -u DEEPSEEK_API_KEY .venv/bin/eva --debug
```

Expected: `ОШИБКА: Переменная окружения DEEPSEEK_API_KEY не задана` on stderr, exit code 1.

- [ ] **Step 5: Run full test suite to confirm nothing broke**

```bash
cd /home/adam/eva && uv run pytest -v
```

Expected: 31 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/adam/eva && git add eva/cli.py && git commit -m "feat(cli): wire all components, add signal handling and CLI flags"
```

---

## Task 12: Cleanup, install, smoke test

**Files:**
- Delete: `/home/adam/eva/eva.py`
- Modify: `/home/adam/eva/README.md`
- Create symlink: `~/.local/bin/eva`

- [ ] **Step 1: Delete the old single-file eva.py**

```bash
cd /home/adam/eva && git rm eva.py
```

Expected: file removed and staged.

- [ ] **Step 2: Update README.md**

Overwrite `/home/adam/eva/README.md`:

```markdown
# Голосовой ассистент Ева

Голосовой ассистент для Ubuntu: слушает микрофон, реагирует на wake-word «Ева»,
обращается к DeepSeek API, выполняет shell-команды и поддерживает разговор.

## Стек

- **STT:** faster-whisper (int8) + Silero VAD
- **LLM:** DeepSeek API через OpenAI SDK
- **TTS:** Piper (русский голос `ru_RU-irina-medium`)
- **Менеджер пакетов:** uv

## Установка

```bash
cd ~/eva
uv venv --python 3.11
uv sync --extra dev
ln -sf "$HOME/eva/.venv/bin/eva" "$HOME/.local/bin/eva"
export DEEPSEEK_API_KEY=sk-...   # лучше в ~/.bashrc
```

После этого в любой консоли:

```bash
eva                    # запуск
eva --debug            # подробные логи
eva --model medium     # более точная (но медленная) модель Whisper
eva --no-wake          # отвечать без wake-word (отладка)
python -m eva          # альтернативный запуск
```

## Голосовое управление

- «Ева, ...» — задать команду или вопрос
- «Ева, замолчи» / «Ева, проснись» — тихий режим
- «Ева, выключись» / Ctrl+C — выйти

## Тесты

```bash
uv run pytest
```

## Структура

```
eva/
├── cli.py          # точка входа, wire-up
├── config.py       # настройки
├── audio.py        # AudioCapture + SpeechSegmenter (Silero VAD)
├── stt.py          # Transcriber (faster-whisper)
├── tts.py          # Synthesizer (Piper)
├── brain.py        # Brain (DeepSeek streaming)
├── executor.py     # ShellExecutor
└── assistant.py    # Assistant (оркестратор)
```
```

- [ ] **Step 3: Create the symlink for global access**

```bash
mkdir -p "$HOME/.local/bin" && ln -sf "$HOME/eva/.venv/bin/eva" "$HOME/.local/bin/eva"
```

- [ ] **Step 4: Verify symlink works from any directory**

```bash
cd /tmp && eva --version
```

Expected: `eva 0.1.0`.

If command not found: `~/.local/bin` is not in PATH. Add to `~/.bashrc`:
```bash
export PATH="$HOME/.local/bin:$PATH"
```
Then `source ~/.bashrc` and retry.

- [ ] **Step 5: Smoke test — actually run Eva**

```bash
cd /tmp && eva --debug
```

Expected behaviors (manual verification):
1. Logs show «Загружаю faster-whisper модель small (int8)...»
2. Ева говорит «Ева запущена. Скажи Ева чтобы обратиться.»
3. Скажи «Ева, привет» → Ева должна услышать (видно в логах) и ответить через 1-2 сек.
4. Скажи «Ева, открой firefox» → Firefox открывается.
5. Скажи «Ева, замолчи» → перестаёт реагировать на обычные фразы.
6. Скажи «Ева, проснись» → снова реагирует.
7. Ctrl+C → корректно завершается с надписью «Ева остановлена.»

Если любой из шагов 3-7 не работает — НЕ комитьте, диагностируйте через `--debug` логи.

- [ ] **Step 6: Final commit**

```bash
cd /home/adam/eva && git add README.md && git commit -m "chore: remove old eva.py, update README with uv install and CLI usage"
```

- [ ] **Step 7: Run the full test suite one last time**

```bash
cd /home/adam/eva && uv run pytest -v
```

Expected: 31 tests PASS.

---

## Definition of Done

- All 12 tasks complete with passing tests at each checkpoint.
- `eva` command works from any console (verified in Task 12 step 4).
- Live voice interaction verified (Task 12 step 5).
- Latency from end-of-speech to first audio ≤ 1.5 sec on the user's laptop (subjective verification during smoke test; if it feels closer to 3-5 sec, flag and investigate before declaring done).
- Old `eva.py` deleted, new modular package + tests committed.
- README reflects new install instructions and CLI flags.
