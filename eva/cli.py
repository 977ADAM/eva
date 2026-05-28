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
    from eva.memory import Memory
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
    memory = Memory(config.memory_path)
    brain = Brain(config, memory)
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
