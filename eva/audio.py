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
