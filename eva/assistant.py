import logging
import time
from typing import Callable

from eva.audio import AudioCapture, SpeechSegmenter
from eva.brain import Brain
from eva.config import Config
from eva.executor import ShellExecutor
from eva.safety import SafetyGate
from eva.stt import Transcriber
from eva.tts import Synthesizer

log = logging.getLogger(__name__)


class Assistant:
    """Главный оркестратор. Связывает все компоненты, держит флаги сессии
    (`sleeping`, `running`, `conversation_until`, `pending_command`) и
    реализует логику wake/sleep/exit слов, разговорный режим и подтверждение
    опасных команд.

    `run()` крутит главный цикл: получает utterance из segmenter,
    транскрибирует, прогоняет через `handle_text()`.

    `handle_text()` — отдельный метод для unit-тестирования логики
    без аудио-потока. Источник времени `time_source` инжектится для
    детерминистских тестов (в проде — `time.monotonic`)."""

    def __init__(self, *, config: Config, capture: AudioCapture,
                 segmenter: SpeechSegmenter, transcriber: Transcriber,
                 synthesizer: Synthesizer, brain: Brain,
                 executor: ShellExecutor,
                 time_source: Callable[[], float] = time.monotonic):
        self._cfg = config
        self.capture = capture
        self.segmenter = segmenter
        self.transcriber = transcriber
        self.synthesizer = synthesizer
        self.brain = brain
        self.executor = executor
        self._gate = SafetyGate(config.safe_command_prefixes)
        self.sleeping = False
        self.running = True
        # Разговорный режим: timestamp монотонного времени, до которого
        # сессия активна. None — сессии нет.
        self.conversation_until: float | None = None
        # Команда, ожидающая голосового подтверждения. None — нет pending.
        self.pending_command: str | None = None
        self._time_source = time_source
        self._conversation_timeout = config.conversation_timeout_sec

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

        # 1. Exit — высший приоритет, любое состояние
        if any(w in text for w in self._cfg.exit_words):
            self.synthesizer.say("Выключаюсь. Пока!")
            # stop() и флаг и сегментер останавливает — иначе main loop
            # висит в segments() пока не придёт следующий чанк аудио.
            self.stop()
            return

        # 2. Sleep — отменяет pending перед тем как уйти в молчание
        if not self.sleeping and any(w in text for w in self._cfg.sleep_words):
            self.synthesizer.say(
                "Молчу. Скажи 'Ева, проснись' когда понадоблюсь."
            )
            self.pending_command = None
            self.sleeping = True
            return

        # 3. Wake-again — выходим из sleep
        if self.sleeping and any(w in text for w in self._cfg.wake_again_words):
            self.synthesizer.say("Я снова с тобой.")
            self.sleeping = False
            return

        # 4. В sleep — всё ниже игнорируется
        if self.sleeping:
            return

        # 5. Если сессия истекла — pending тоже мёртв
        if (self.pending_command is not None
                and not self._conversation_active()):
            self.pending_command = None

        # 6. Если есть pending — текущая реплика ТОЛЬКО ответ на подтверждение.
        #    Не отправляется в Brain. Безопасный дефолт: если непонятно — нет.
        if self.pending_command is not None:
            has_no = any(w in text for w in self._cfg.confirm_no_words)
            has_yes = any(w in text for w in self._cfg.confirm_yes_words)
            cmd = self.pending_command
            self.pending_command = None
            if has_yes and not has_no:
                self.synthesizer.say("Хорошо.")
                self.executor.run(cmd)
            else:
                # Явный no, или непонятно, или «да, отмени» — отменяем
                self.synthesizer.say("Отменено.")
            return

        # 7. Обычный путь: wake-word или активная сессия
        has_wake = any(w in text for w in self._cfg.wake_words)
        in_session = self._conversation_active()

        if self._cfg.require_wake and not (has_wake or in_session):
            return

        clean = self._strip_wake_word(text)
        if not clean:
            self.synthesizer.say("Слушаю.")
            return

        # Продлеваем сессию ОДИН раз за обмен — до вызова Brain.
        # Так юзер может пробовать снова без wake-word даже если Brain
        # отдал ошибку.
        self._extend_conversation()
        self._ask_brain_and_respond(clean)

    def _conversation_active(self) -> bool:
        if self.conversation_until is None:
            return False
        return self._time_source() < self.conversation_until

    def _extend_conversation(self) -> None:
        self.conversation_until = self._time_source() + self._conversation_timeout

    def _strip_wake_word(self, text: str) -> str:
        for w in self._cfg.wake_words:
            if text.startswith(w):
                return text[len(w):].strip(" ,.;:!?")
        return text

    def _ask_brain_and_respond(self, prompt: str) -> None:
        command_to_run: str | None = None
        for delta in self.brain.ask_stream(prompt):
            if delta.error:
                self.synthesizer.say("Связь пропала, попробуй позже.")
                return
            if delta.sentence:
                self.synthesizer.say(delta.sentence)
            if delta.done and delta.command:
                command_to_run = delta.command
        if command_to_run:
            if self._gate.is_safe(command_to_run):
                self.executor.run(command_to_run)
            else:
                # Опасная команда — запоминаем, ждём голосового «да/нет»
                self.pending_command = command_to_run
                self.synthesizer.say("Подтверди.")
