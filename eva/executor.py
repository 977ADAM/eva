import logging
import subprocess

log = logging.getLogger(__name__)


class ShellExecutor:
    """Запускает shell-команды от LLM в фоне (fire-and-forget)."""

    def run(self, command: str) -> None:
        if not command:
            return
        log.info("Выполняю: %s", command)
        try:
            subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            log.warning("Ошибка выполнения команды %r: %s", command, exc)
