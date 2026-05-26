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
