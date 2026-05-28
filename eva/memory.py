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
        except (json.JSONDecodeError, OSError, AttributeError) as exc:
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
