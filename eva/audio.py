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
