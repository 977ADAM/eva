# Eva — долговременная память фактов о пользователе

**Дата:** 2026-05-27
**Статус:** draft (ожидает реализации)
**Ветка:** `feat/memory`

## Цель

Дать Еве память между сессиями: пользователь говорит «Ева, запомни что...» — факт сохраняется на диск и автоматически учитывается во всех будущих разговорах, пока пользователь не скажет «забудь». Модель — как persistent memory у ChatGPT.

## Контекст

Сейчас `Brain` держит только `self._history` (последние 10 сообщений в RAM), всё теряется при перезапуске `eva`. Пользователь хочет, чтобы Ева помнила факты о нём (имя, привычки, предпочтения) надолго.

Выбор из брейнсторма: **факты о пользователе** (не вся история, не RAG — слишком тяжело для 8 ГБ без GPU), запись **явно по команде** (не авто-извлечение — предсказуемо и без лишних LLM-вызовов), операции: **запомнить / вспомнить(авто) / список / забыть**.

## Решение

Новый модуль `eva/memory.py` с классом `Memory` — JSON-хранилище фактов. `Brain` получает `Memory`, инжектит факты в системный промпт на каждом запросе (это и есть авто-recall + ответ на «что ты обо мне знаешь»), и обрабатывает два новых LLM-action: `remember` и `forget`. Остальные компоненты (`Assistant`, `Synthesizer`, `SpeechSegmenter`, `Executor`, `SafetyGate`) не меняются — вся память в Brain + Memory + Config.

## Архитектура

### Новый модуль `eva/memory.py`

```python
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class Memory:
    """Долговременная память фактов о пользователе. Хранится в JSON-файле
    вида {"facts": ["факт1", "факт2"]}. Любая мутация сразу пишется на диск."""

    def __init__(self, path: Path):
        self._path = path
        self._facts: list[str] = self._load()

    def _load(self) -> list[str]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            facts = data.get("facts", [])
            return [str(f) for f in facts if str(f).strip()]
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Не смогла прочитать память %s: %s", self._path, exc)
            return []

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"facts": self._facts}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("Не смогла сохранить память %s: %s", self._path, exc)

    def add(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        self._facts.append(fact)
        self._save()

    def remove(self, query: str) -> int:
        """Удаляет все факты, содержащие query (подстрока, без регистра).
        Возвращает число удалённых."""
        query = query.strip().lower()
        if not query:
            return 0
        before = len(self._facts)
        self._facts = [f for f in self._facts if query not in f.lower()]
        removed = before - len(self._facts)
        if removed:
            self._save()
        return removed

    def all(self) -> list[str]:
        return list(self._facts)

    def as_prompt(self) -> str:
        if not self._facts:
            return ""
        lines = "\n".join(f"- {f}" for f in self._facts)
        return f"Что ты знаешь о пользователе:\n{lines}"
```

### Config: новое поле

```python
# Память: путь к JSON-файлу с фактами о пользователе
memory_path: Path = ...  # eva_dir / "memory.json", задаётся в Config.load()
```

Так как `eva_dir` вычисляется в `Config.load()`, `memory_path` тоже проставляется там:

```python
return cls(
    ...
    memory_path=eva_dir / "memory.json",
    ...
)
```

Поле объявляется в dataclass с заглушкой-дефолтом (нужно потому что у dataclass без дефолта обязателен порядок); проще объявить `memory_path: Path` среди обязательных полей рядом с `eva_dir`/`piper_bin`/`voice_model` и заполнять в `load()`. См. план реализации для точного места.

`.gitignore` дополняется строкой `memory.json` (и `/memory.json` чтобы не заигнорить случайные вложенные).

### Расширение LLM-протокола

Два новых `action` в JSON-ответе:

```json
{"say":"Запомнила.","action":"remember","fact":"пьёт кофе без сахара"}
{"say":"Забыла.","action":"forget","fact":"кофе"}
```

LLM сам вычленяет канонический факт из реплики пользователя.

### Изменения в Brain

`Brain.__init__` получает `memory: Memory`:

```python
def __init__(self, config: "Config", memory: "Memory"):
    ...
    self._memory = memory
```

**System prompt с фактами.** В `ask_stream` системное сообщение собирается из базового `SYSTEM_PROMPT` + блок фактов:

```python
facts_block = self._memory.as_prompt()
system_content = SYSTEM_PROMPT
if facts_block:
    system_content = f"{SYSTEM_PROMPT}\n\n{facts_block}"
messages = [{"role": "system", "content": system_content}] + self._history[-N:]
```

**Обработка remember/forget после парсинга полного JSON.** В конце `ask_stream`, там где сейчас определяется `command` для action=="shell", добавляются ветки:

```python
action = data.get("action")
if action == "remember":
    fact = (data.get("fact") or "").strip()
    if fact:
        self._memory.add(fact)
elif action == "forget":
    fact = (data.get("fact") or "").strip()
    if fact:
        self._memory.remove(fact)

command = data.get("command") if action == "shell" else None
yield ResponseDelta(command=command, done=True)
```

Произнесение `say` остаётся прежним (стриминг предложений + fallback). remember/forget НЕ дают shell-команды → `command=None` → `Assistant` ничего не выполняет, только озвучивает. Assistant не нужно менять.

### SYSTEM_PROMPT — дополнение

К существующему промпту добавляются описания новых action и примеры:

```
- Запомнить факт о пользователе: {"say":"Запомнила.","action":"remember","fact":"чистый факт"}
- Забыть факт: {"say":"Забыла.","action":"forget","fact":"ключевое слово"}

Примеры:
- "запомни что я пью кофе без сахара" → {"say":"Запомнила.","action":"remember","fact":"пьёт кофе без сахара"}
- "забудь про кофе" → {"say":"Забыла.","action":"forget","fact":"кофе"}
- "что ты обо мне знаешь" → {"say":"Ты пьёшь кофе без сахара...","action":"talk"} (используй факты из блока ниже)
```

### CLI — проводка

В `cli.main`, рядом с созданием Brain:

```python
from eva.memory import Memory
memory = Memory(config.memory_path)
brain = Brain(config, memory)
```

## Поток данных

```
"ева запомни что я пью кофе без сахара"
  → Assistant.handle_text → ask_brain
    → system prompt = SYSTEM_PROMPT + (текущие факты, если есть)
    → DeepSeek: {"say":"Запомнила.","action":"remember","fact":"пьёт кофе без сахара"}
    → стриминг say → "Запомнила." озвучено
    → action=remember → memory.add("пьёт кофе без сахара") → запись в ~/eva/memory.json
    → ResponseDelta(command=None, done=True)
  → Assistant: command None → ничего не выполняет

[следующий запуск eva, новый разговор]
"ева что приготовить на завтрак"
  → system prompt включает "Что ты знаешь о пользователе:\n- пьёт кофе без сахара"
  → DeepSeek учитывает это в ответе
```

## Обработка ошибок

| Сценарий | Поведение |
|---|---|
| memory.json не существует (первый запуск) | `_load` возвращает `[]`, память пустая, не падает |
| memory.json повреждён (невалидный JSON) | `_load` логирует warning, возвращает `[]` (не теряем работоспособность; старый файл перезапишется при следующем add) |
| Нет прав на запись файла | `_save` логирует warning, факт остаётся в RAM на эту сессию, на диск не попадает |
| LLM вернул remember без поля fact | пустой fact игнорируется (`add` на "" — no-op) |
| forget по слову которого нет в фактах | `remove` возвращает 0, ничего не удалено; Ева всё равно скажет «Забыла» (LLM не знает результат) — приемлемо для v1 |
| Очень много фактов | системный промпт растёт; для десятков фактов норм, лимита нет (YAGNI) |

## Тестирование

### `tests/test_memory.py` — 8 тестов

1. `test_empty_when_no_file` — `Memory(несуществующий путь).all()` → `[]`
2. `test_add_appears_in_all` — `add("x")` → `"x" in all()`
3. `test_add_persists_to_disk` — `add` в одном инстансе, новый `Memory(тот же путь).all()` содержит факт
4. `test_remove_by_substring` — добавили «пьёт кофе без сахара», `remove("кофе")` → вернул 1, `all()` пуст
5. `test_remove_nonexistent_returns_zero` — `remove("чай")` когда такого нет → 0
6. `test_remove_multiple_matches` — два факта с «кофе», `remove("кофе")` → 2
7. `test_as_prompt_empty` — пустая память → `as_prompt() == ""`
8. `test_as_prompt_formats_facts` — два факта → строка содержит «Что ты знаешь о пользователе» и оба факта с «- »
9. `test_corrupted_file_returns_empty` — записать мусор в файл, `Memory(path).all()` → `[]` (не падает)

(9 тестов — пункт «8 тестов» в заголовке неточен, итог 9.)

### `tests/test_brain.py` — 3 новых теста (с моком Memory)

1. `test_remember_action_calls_memory_add` — LLM вернул remember → `memory.add("факт")` вызван, последняя реплика — say
2. `test_forget_action_calls_memory_remove` — LLM вернул forget → `memory.remove("слово")` вызван
3. `test_facts_injected_into_system_prompt` — `memory.as_prompt()` возвращает непустое → текст попал в `messages[0]["content"]` запроса

Существующие тесты `test_brain.py` обновляются минимально: фабрика `make_config`/Brain-конструктор теперь требует `Memory` — добавляем `MagicMock()` или реальный `Memory(tmp_path)` в хелпер создания Brain. Все прежние ассерты остаются.

## Критерии приёмки

1. Голосом: «Ева, запомни что меня зовут Адам» → «Запомнила». Файл `~/eva/memory.json` содержит факт.
2. Перезапуск `eva`, «Ева, как меня зовут?» → Ева отвечает «Адам» (факт из памяти).
3. «Ева, что ты обо мне знаешь?» → перечисляет факты.
4. «Ева, забудь что меня зовут Адам» → «Забыла», факт исчез из файла.
5. Все тесты зелёные: 49 текущих + 9 memory + 3 brain = 61.
6. `memory.json` в `.gitignore`, не коммитится.

## Что НЕ входит

- Авто-извлечение фактов из разговора (только явная команда)
- Дедупликация / нормализация фактов
- Редактирование существующего факта (только add/remove)
- Лимит на число фактов
- Векторный поиск / RAG
- Вся история разговоров на диск
- Шифрование файла памяти
- Синхронизация между устройствами
