#!/usr/bin/env python3
"""
Ева — голосовой ассистент на DeepSeek + Whisper + Piper
Активация: скажи "Ева, ..." — например "Ева, открой firefox"
Отключение: "Ева, выключись" или Ctrl+C или `pkill -f eva.py`
Тихий режим: "Ева, замолчи" / "Ева, проснись"
"""

import os
for var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
            "HTTPS_PROXY", "https_proxy", "SOCKS_PROXY", "socks_proxy"):
    os.environ.pop(var, None)
import sys
import subprocess
import tempfile
import json
import signal
import queue
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
from openai import OpenAI

# ---------- НАСТРОЙКИ ----------
HOME = Path.home()
EVA_DIR = HOME / "eva"
PIPER_BIN = EVA_DIR / "piper" / "piper"
VOICE_MODEL = EVA_DIR / "voices" / "ru_RU-irina-medium.onnx"

WAKE_WORDS = ["ева", "эва", "eva", "ява"]   # на что просыпается
SLEEP_WORDS = ["замолчи", "спи", "тихо"]
WAKE_AGAIN_WORDS = ["проснись", "слушай", "ева проснись"]
EXIT_WORDS = ["выключись", "выйди", "стоп", "остановись", "завершись"]

SAMPLE_RATE = 16000
CHUNK_SECONDS = 4        # сколько секунд писать звук перед распознаванием
SILENCE_THRESHOLD = 0.005 # ниже = тишина, не отправляем в whisper

# ---------- ИНИЦИАЛИЗАЦИЯ ----------
print("Загружаю Whisper (первый запуск — скачается модель ~150 МБ)...")
stt = whisper.load_model("small")  # можно "base" если медленно

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    print("ОШИБКА: переменная DEEPSEEK_API_KEY не задана")
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

SYSTEM_PROMPT = """Тебя зовут Ева. Ты голосовой ассистент в Ubuntu Linux.
Отвечай ОЧЕНЬ коротко (1-2 предложения), потому что твой ответ озвучивается.
Если пользователь просит выполнить действие в системе (открыть приложение, файл, сайт),
верни JSON в формате: {"action": "shell", "command": "команда", "say": "что сказать вслух"}
Если просто разговор — верни JSON: {"action": "talk", "say": "ответ"}

Примеры:
- "открой firefox" → {"action": "shell", "command": "firefox &", "say": "Открываю Firefox"}
- "открой папку загрузки" → {"action": "shell", "command": "xdg-open ~/Downloads", "say": "Открыла загрузки"}
- "сколько времени" → {"action": "talk", "say": "Сейчас..."} (укажи примерно или попроси посмотреть)
- "как дела" → {"action": "talk", "say": "Всё хорошо, чем помочь?"}

ВАЖНО: возвращай ТОЛЬКО валидный JSON, без markdown, без ```."""

conversation = []
sleeping = False
running = True

# ---------- ФУНКЦИИ ----------
def speak(text):
    """Озвучить текст через piper"""
    if not text:
        return
    print(f"🔊 Ева: {text}")
    try:
        proc = subprocess.run(
            [str(PIPER_BIN), "--model", str(VOICE_MODEL), "--output_file", "/tmp/eva_out.wav"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        subprocess.run(["aplay", "-q", "/tmp/eva_out.wav"], check=False)
    except Exception as e:
        print(f"Ошибка озвучки: {e}")

# Очередь аудио из микрофона
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Колбэк sounddevice — пишет звук в очередь нон-стоп"""
    audio_queue.put(indata.copy().flatten())

def listen_for_speech():
    """Слушает поток непрерывно, возвращает фразу когда поймает речь+паузу"""
    block_duration = 0.1  # 100 мс блоки
    silence_limit = 1.0   # сек тишины = конец фразы
    min_speech = 0.3      # мин длина речи

    silent_blocks_limit = int(silence_limit / block_duration)
    min_speech_blocks = int(min_speech / block_duration)

    frames = []
    silent_count = 0
    speaking = False

    while running:
        try:
            block = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        level = np.abs(block).mean()

        if level > SILENCE_THRESHOLD:
            frames.append(block)
            silent_count = 0
            speaking = True
        elif speaking:
            frames.append(block)
            silent_count += 1
            if silent_count >= silent_blocks_limit:
                if len(frames) >= min_speech_blocks:
                    audio = np.concatenate(frames)
                    frames = []
                    silent_count = 0
                    speaking = False
                    return audio
                else:
                    # слишком коротко — игнор
                    frames = []
                    silent_count = 0
                    speaking = False

    return None

def is_silent(audio):
    """Проверить, тихо ли"""
    return np.abs(audio).mean() < SILENCE_THRESHOLD

def transcribe(audio):
    """Whisper → текст"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        result = stt.transcribe(f.name, language="ru", fp16=False)
        os.unlink(f.name)
    return result["text"].strip().lower()

def ask_deepseek(user_text):
    """Отправить в DeepSeek, получить JSON-ответ"""
    conversation.append({"role": "user", "content": user_text})
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation[-10:],
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
        # убираем возможные ```json
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        conversation.append({"role": "assistant", "content": text})
        return data
    except json.JSONDecodeError:
        return {"action": "talk", "say": "Я не поняла, можешь повторить?"}
    except Exception as e:
        print(f"Ошибка API: {e}")
        return {"action": "talk", "say": "Что-то пошло не так с подключением."}

def execute(command):
    """Выполнить shell-команду"""
    print(f"💻 Выполняю: {command}")
    try:
        subprocess.Popen(command, shell=True,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Ошибка выполнения: {e}")

def handle_command(text):
    """Главная логика: что делать с распознанной фразой"""
    global sleeping, running

    # Выключение
    if any(w in text for w in EXIT_WORDS):
        speak("Выключаюсь. Пока!")
        running = False
        return

    # Тихий режим — спать
    if not sleeping and any(w in text for w in SLEEP_WORDS):
        speak("Молчу. Скажи 'Ева, проснись' когда понадоблюсь.")
        sleeping = True
        return

    # Тихий режим — проснуться
    if sleeping and any(w in text for w in WAKE_AGAIN_WORDS):
        speak("Я снова с тобой.")
        sleeping = False
        return

    if sleeping:
        return

    # Убираем "ева" из начала
    clean = text
    for w in WAKE_WORDS:
        if clean.startswith(w):
            clean = clean[len(w):].strip(" ,.")
            break

    if not clean:
        speak("Слушаю.")
        return

    # Отправляем в DeepSeek
    response = ask_deepseek(clean)
    say_text = response.get("say", "")

    if response.get("action") == "shell" and response.get("command"):
        speak(say_text)
        execute(response["command"])
    else:
        speak(say_text)

def signal_handler(sig, frame):
    global running
    print("\n⚙️  Останавливаю Еву...")
    running = False

# ---------- MAIN LOOP ----------
def main():
    global running
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    speak("Ева запущена. Скажи Ева чтобы обратиться.")
    print("\n=== Постоянно слушаю микрофон. Скажи 'Ева, ...' ===\n")

    block_size = int(SAMPLE_RATE * 0.1)
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=block_size,
                            callback=audio_callback)
    stream.start()

    try:
        while running:
            audio = listen_for_speech()
            if audio is None or len(audio) == 0:
                continue

            text = transcribe(audio)
            if not text:
                continue

            print(f"🎤 Услышала: {text}")

            # очищаем очередь пока думаем, чтобы не накапливалось
            has_wake = any(w in text for w in WAKE_WORDS)
            if has_wake or (sleeping and any(w in text for w in WAKE_AGAIN_WORDS)):
                # пока обрабатываем — выкидываем буфер чтобы не слышать саму себя
                while not audio_queue.empty():
                    try: audio_queue.get_nowait()
                    except queue.Empty: break
                handle_command(text)
                # ещё раз чистим — после ответа Евы
                while not audio_queue.empty():
                    try: audio_queue.get_nowait()
                    except queue.Empty: break
    finally:
        stream.stop()
        stream.close()
        print("Ева остановлена.")

if __name__ == "__main__":
    main()
