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
