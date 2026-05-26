import logging

from eva.audio import AudioCapture, SpeechSegmenter
from eva.brain import Brain
from eva.config import Config
from eva.executor import ShellExecutor
from eva.stt import Transcriber
from eva.tts import Synthesizer

log = logging.getLogger(__name__)


class Assistant:
    """Главный оркестратор. Связывает все компоненты, держит флаги сессии
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
