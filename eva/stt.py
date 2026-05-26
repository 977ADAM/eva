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
