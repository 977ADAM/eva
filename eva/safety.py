"""Решает, нужно ли голосовое подтверждение перед запуском shell-команды."""

_SHELL_METACHARS = (";", "&", "|", ">", "<", "$(", "`", "\n")


class SafetyGate:
    """Команда считается безопасной если её первая лексема (имя бинарника)
    есть в allowlist И в команде нет shell-метасимволов. Единственное
    исключение: одиночный трейлинг `&` (бэкграунд-запуск) — разрешён."""

    def __init__(self, safe_prefixes: tuple[str, ...]):
        self._safe = set(safe_prefixes)

    def is_safe(self, command: str) -> bool:
        stripped = command.strip()
        if not stripped:
            return True  # пустая команда — нечего выполнять, нечего бояться

        # Отрезаем разрешённый трейлинг `&`, остаток проверяем на метасимволы.
        # `firefox &` → head = "firefox" → метасимволов нет → ok.
        # `firefox && rm` → head = "firefox && rm" → содержит & → не safe.
        head = stripped[:-1].rstrip() if stripped.endswith("&") else stripped
        if any(meta in head for meta in _SHELL_METACHARS):
            return False

        first_token = head.split()[0]
        return first_token in self._safe
