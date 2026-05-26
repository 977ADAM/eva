import logging
import subprocess
from pathlib import Path

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
