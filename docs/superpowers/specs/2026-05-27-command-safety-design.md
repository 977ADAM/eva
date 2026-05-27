# Eva — безопасность shell-команд

**Дата:** 2026-05-27
**Статус:** draft (ожидает реализации)
**Ветка:** `feat/command-safety`

## Цель

Перестать выполнять любую сгенерированную DeepSeek shell-команду без вопросов. Команды из allowlist бегут как раньше; всё остальное требует голосового подтверждения «да/нет» от пользователя.

## Контекст

В версии 0.1.0 `ShellExecutor.run()` выполняет `subprocess.Popen(command, shell=True)` для любой строки от LLM. Это уже привело к реальному инциденту 2026-05-27: пользователь сказал «Ева, выключи компьютер», DeepSeek сгенерил `shutdown` команду, Ева её выполнила, ноут отключился. Спек первого рефакторинга явно отложил command safety как отдельный проект — этот спек закрывает долг.

## Решение

Двухуровневая защита:

1. **Allowlist префиксов** — команды, чья первая лексема (имя бинарника) есть в списке, выполняются молча. Список покрывает обыденные действия: открыть файл/папку/сайт, запустить GUI-приложение, управлять громкостью/плеером.

2. **Голосовое подтверждение для остального** — Ева запоминает команду в `pending_command`, спрашивает «Подтверди.» (контекст уже дан в `say`-поле LLM-ответа). Следующая реплика пользователя интерпретируется ТОЛЬКО как «да/нет» — не отправляется в Brain. По умолчанию (если ни yes, ни no — отменяем).

Pending привязан к conversation-сессии: если юзер молчит N секунд (60 по дефолту) — сессия истекает, pending автоматически забывается. Sleep тоже сбрасывает pending.

## Архитектура

### Новый модуль `eva/safety.py`

Один маленький класс:

```python
_SHELL_METACHARS = (";", "&", "|", ">", "<", "$(", "`", "\n")


class SafetyGate:
    """Решает, нужно ли голосовое подтверждение перед запуском команды.
    Команда safe если её первая лексема — в allowlist И в ней нет
    shell-метасимволов (за единственным исключением: одиночный `&` в самом
    конце разрешён для запуска в фоне)."""

    def __init__(self, safe_prefixes: tuple[str, ...]):
        self._safe = set(safe_prefixes)

    def is_safe(self, command: str) -> bool:
        stripped = command.strip()
        if not stripped:
            return True  # пустая команда — нечего бояться

        # Отрезаем разрешённый трейлинг `&` (бэкграунд), остаток проверяем
        # на метасимволы. `firefox &` → head = "firefox" → ok.
        # `firefox && rm` → head = "firefox && rm" → содержит & → не safe.
        head = stripped[:-1].rstrip() if stripped.endswith("&") else stripped
        if any(meta in head for meta in _SHELL_METACHARS):
            return False

        first_token = head.split()[0]
        return first_token in self._safe
```

Это сознательное упрощение, НЕ полноценный shell-парс: всё что выглядит как пайп/чейн/редирект/подставка/бэктики идёт через подтверждение. Так чтобы провернуть «firefox && shutdown» через safe-путь — нужна явная санкция юзера.

### Config: новые поля

```python
# Безопасность shell-команд
safe_command_prefixes: tuple[str, ...] = (
    "xdg-open", "gio",                                          # открывалки
    "firefox", "chromium", "google-chrome", "code", "nautilus",
    "gnome-terminal", "vlc", "telegram-desktop", "spotify",     # GUI-приложения
    "playerctl", "amixer", "pactl",                             # медиа/громкость
)
confirm_yes_words: tuple[str, ...] = (
    "да", "ага", "конечно", "подтверждаю", "давай", "хорошо",
)
confirm_no_words: tuple[str, ...] = (
    "нет", "отмена", "не надо", "не нужно", "стоп",
)
```

### Assistant: новое состояние и логика

**Новые атрибуты:**
- `pending_command: str | None = None` — команда, ждущая подтверждения
- `_gate: SafetyGate` — инстанцируется в `__init__` из конфига

**Новая ветка в `handle_text` ВВЕРХУ** (после exit/sleep, ДО wake-word check):

```python
# Если есть pending подтверждение — текущая реплика трактуется ТОЛЬКО как ответ.
# Сначала проверяем NO (если есть и yes и no — побеждает no, безопасный дефолт).
if self.pending_command is not None:
    has_no  = any(w in text for w in self._cfg.confirm_no_words)
    has_yes = any(w in text for w in self._cfg.confirm_yes_words)
    cmd = self.pending_command
    self.pending_command = None
    if has_yes and not has_no:
        self.synthesizer.say("Хорошо.")
        self.executor.run(cmd)
    else:
        # явный no, или непонятно, или «да, отмени» — отменяем
        self.synthesizer.say("Отменено.")
    return
```

**Изменение в `_ask_brain_and_respond`:**

```python
if command_to_run:
    if self._gate.is_safe(command_to_run):
        self.executor.run(command_to_run)
    else:
        self.pending_command = command_to_run
        self.synthesizer.say("Подтверди.")
```

**Sleep сбрасывает pending:** в ветке sleep_words добавить `self.pending_command = None` перед `self.sleeping = True`.

**Истечение сессии сбрасывает pending:** в начале `handle_text`, ПЕРЕД проверкой pending — если conversation_until истёк и pending был — очистить:

```python
# Сессия истекла — pending тоже мёртв
if (self.pending_command is not None
        and not self._conversation_active()):
    self.pending_command = None
```

**Порядок проверок в `handle_text` (после изменений):**

1. `text.strip().lower()` + empty check
2. `exit_words` → `self.stop()` + return
3. `sleep_words` (если не sleeping) → clear pending → sleeping=True + return
4. `wake_again_words` (если sleeping) → sleeping=False + return
5. `if self.sleeping: return`
6. **clear-pending-if-session-expired** (новое)
7. **pending_command check** (новое) → return если pending был
8. `has_wake` + `in_session` + `require_wake` gate (как сейчас)
9. strip wake-word + extend_conversation + ask_brain

### Что НЕ меняется
- `Brain`, `Synthesizer`, `Transcriber`, `Executor`, `AudioCapture`, `SpeechSegmenter` — без изменений
- Формат JSON от LLM — без изменений
- CLI флаги — без изменений
- Existing tests — все должны остаться зелёными

## Поток данных (с подтверждением)

```
text "ева выключи компьютер"
  → has_wake=True → extend_conversation → ask_brain
    → DeepSeek: {"say":"Сейчас выключу.","action":"shell","command":"shutdown now"}
    → ResponseDelta(sentence="Сейчас выключу.")  → synthesizer.say
    → ResponseDelta(command="shutdown now", done=True)
  → command_to_run="shutdown now"
  → gate.is_safe? "shutdown" не в allowlist → False
  → pending_command="shutdown now", synthesizer.say("Подтверди.")

[пользователь говорит "да"]
text "да"
  → pending_command есть → "да" в yes_words → executor.run("shutdown now")
  → pending=None, synthesizer.say("Хорошо.")

[или пользователь говорит "нет" / молчит 60s / уходит в sleep]
  → pending очищается, executor НЕ вызывается
```

## Обработка ошибок и крайние случаи

| Сценарий | Поведение |
|---|---|
| Команда содержит `&&` или `;` | Не safe — нужно подтверждение |
| Команда `firefox &` (бэкграунд) | Safe — `&` в самом конце разрешён |
| Pending висит, юзер сказал что-то непонятное (не yes и не no) | Отмена (безопасный дефолт). Eva говорит «Отменено.», pending очищен |
| Pending + «да открой что-нибудь» (yes без no) | yes-word найден → выполняется pending команда (не новая просьба). Это сознательное упрощение — иначе нужен LLM-разбор намерения |
| Pending + «да, отмени» (yes И no) | no побеждает → отменяем. Безопасный дефолт |
| Pending + sleep word | Sleep отменяет pending, Eva говорит «Молчу...», pending=None, sleeping=True |
| Pending + exit word | Exit срабатывает (выходим), pending не важен |
| Pending висит, сессия истекла | На следующем utterance pending очистится молча; ничего не выполнится |
| Один большой блок команд `firefox; echo ok` | Не safe (есть `;`) → подтверждение |

## Тестирование

### `tests/test_safety.py` — 7 тестов

1. `test_empty_command_is_safe` — `""` → True
2. `test_allowlist_prefix_is_safe` — `"firefox"` → True (если firefox в списке)
3. `test_firefox_with_args_is_safe` — `"firefox https://ya.ru"` → True
4. `test_firefox_with_trailing_amp_is_safe` — `"firefox &"` → True
5. `test_unknown_command_needs_confirm` — `"shutdown now"` → False
6. `test_shell_chain_needs_confirm` — `"firefox && rm -rf /"` → False (даже firefox впереди)
7. `test_pipe_needs_confirm` — `"firefox | grep x"` → False

### `tests/test_assistant_logic.py` — 5 новых тестов

1. `test_safe_command_runs_immediately` — Brain отдал `command="firefox"` → executor.run сразу, pending_command остался None
2. `test_unsafe_command_sets_pending_not_runs` — Brain отдал `command="shutdown now"` → pending_command="shutdown now", executor.run НЕ вызвался, последняя say-реплика = "Подтверди."
3. `test_yes_word_runs_pending` — pending="rm -rf /tmp/x" → handle_text("да") → executor.run("rm -rf /tmp/x"), pending=None, say="Хорошо."
4. `test_no_word_cancels_pending` — pending="X" → handle_text("нет") → executor НЕ вызван, pending=None, say="Отменено."
5. `test_pending_cleared_when_session_expires` — pending="X" + advance clock past timeout → handle_text("да") → executor НЕ вызван (pending был очищен по истечению сессии до того как «да» обработался)

Существующие тесты `test_command_is_executed_when_present` и `test_no_command_means_no_execution` уже используют команду `"firefox &"` (safe) — должны остаться зелёными.

## Критерии приёмки

1. Голосом: «Ева, выключи компьютер» → Ева говорит описание + «Подтверди.» → компьютер НЕ выключается.
2. На «да» после неё → компьютер выключается.
3. На «нет» → Ева говорит «Отменено.», ничего не происходит.
4. «Ева, открой firefox» — Firefox открывается БЕЗ подтверждения (он в allowlist).
5. Все unit-тесты зелёные: 36 старых + 7 в test_safety + 5 в test_assistant_logic = 48.
6. `eva --help` работает без регрессий.

## Что НЕ входит

- LLM-разбор намерения для двусмысленных ответов («ну вроде да, но...»)
- Persistent log выполненных команд
- sudo-detection как отдельная категория («sudo» как первое слово → автоматически не safe, потому что first_token будет `"sudo"` которого нет в allowlist; то есть всё что под sudo проходит через подтверждение)
- Hash-based command identity (запомнить «эту команду я разрешил раньше»)
- Per-command категоризация уровня опасности (warning vs danger)
- Изменение allowlist голосом во время сессии
- Время жизни pending отдельное от сессии — переиспользуем conversation_until
