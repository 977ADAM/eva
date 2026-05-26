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
