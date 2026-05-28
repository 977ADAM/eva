import logging

import sounddevice as sd
from rich.panel import Panel
from rich.text import Text

log = logging.getLogger(__name__)

ACCENT = "bright_cyan"  # акцентный цвет рамки и имени; легко поменять


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Русская форма множественного числа: 1 факт, 2 факта, 5 фактов."""
    if 11 <= n % 100 <= 14:
        return many
    d = n % 10
    if d == 1:
        return one
    if 2 <= d <= 4:
        return few
    return many


def _input_device_name() -> str:
    """Имя дефолтного входного устройства; best-effort, с фолбэком."""
    try:
        info = sd.query_devices(kind="input")
        name = info.get("name") if isinstance(info, dict) else None
        return name or "по умолчанию"
    except Exception as exc:
        log.debug("Не смогла определить микрофон: %s", exc)
        return "по умолчанию"


def build_panel(config, memory, version: str, *, mic_name: str | None = None) -> Panel:
    if mic_name is None:
        mic_name = _input_device_name()

    n = len(memory.all())
    facts = f"{n} {_plural(n, 'факт', 'факта', 'фактов')}" if n else "пусто"
    wake = ", ".join(f"«{w}»" for w in config.wake_words[:2])
    model = f"Whisper {config.whisper_model_name} ({config.whisper_compute_type})"

    body = Text()
    body.append("✦ Ева", style=f"bold {ACCENT}")
    body.append(f" · голосовой ассистент    v{version}\n\n", style="dim")
    body.append(f"  Модель    {model}\n")
    body.append(f"  Микрофон  {mic_name}\n")
    body.append(f"  Будит     {wake}\n")
    body.append(f"  Память    {facts}\n\n")
    body.append("  «ева, открой firefox» · «ева, запомни …»\n", style="dim")
    body.append("  «спи» — пауза · «выключись» — выход", style="dim")

    return Panel(body, border_style=ACCENT, expand=False, padding=(0, 1))
