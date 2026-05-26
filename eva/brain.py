import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from openai import OpenAI

if TYPE_CHECKING:
    from eva.config import Config


_SAY_PREFIX_RE = re.compile(r'"say"\s*:\s*"')
_SENTENCE_END_RE = re.compile(r'[.!?\n]+\s*')

log = logging.getLogger(__name__)


class StreamingSayParser:
    """Инкрементально извлекает готовые предложения из поля `say` JSON-ответа
    LLM, пока ответ ещё стримится по кусочкам.

    Работает в предположении, что `say` идёт ПЕРВЫМ полем в JSON (это
    зашито в SYSTEM_PROMPT). Если LLM нарушил порядок и `say` пришёл после
    `command`/`action` — парсер просто не вернёт ничего, и Brain отрендерит
    весь `say` целиком из полного JSON.

    Также делает упрощающее предположение: внутри `say` нет литеральных
    `"` (LLM их экранирует или вообще не использует в русском тексте).
    Если кавычка всё-таки появится — мы посчитаем её закрытием поля; ничего
    страшного не произойдёт, парс полного JSON в Brain всё равно отработает."""

    def __init__(self):
        self._pre_buffer = ""        # текст до того как нашли `"say":"`
        self._extracting = False     # true когда уже внутри значения `say`
        self._say_text = ""          # накопленное содержимое `say`
        self._emitted_to = 0         # индекс в _say_text до которого уже отдано
        self._closed = False         # увидели закрывающую кавычку поля
        self._emit_count = 0

    def feed(self, chunk: str) -> Iterator[str]:
        if not self._extracting:
            self._pre_buffer += chunk
            match = _SAY_PREFIX_RE.search(self._pre_buffer)
            if not match:
                return
            self._extracting = True
            chunk = self._pre_buffer[match.end():]
            self._pre_buffer = ""

        if self._closed:
            return

        quote_idx = chunk.find('"')
        if quote_idx == -1:
            self._say_text += chunk
        else:
            self._say_text += chunk[:quote_idx]
            self._closed = True

        yield from self._extract_sentences()

        if self._closed:
            remainder = self._say_text[self._emitted_to:].strip()
            if remainder:
                self._emit_count += 1
                self._emitted_to = len(self._say_text)
                yield remainder

    def _extract_sentences(self) -> Iterator[str]:
        text = self._say_text[self._emitted_to:]
        last_end = 0
        for m in _SENTENCE_END_RE.finditer(text):
            end = m.end()
            sentence = text[last_end:end].strip()
            if sentence:
                self._emit_count += 1
                yield sentence
            last_end = end
        self._emitted_to += last_end

    def emitted_anything(self) -> bool:
        return self._emit_count > 0


SYSTEM_PROMPT = """Тебя зовут Ева. Ты голосовой ассистент в Ubuntu Linux.
Отвечай ОЧЕНЬ коротко (1-2 предложения), потому что твой ответ озвучивается.

Возвращай ТОЛЬКО валидный JSON, без markdown, без ```.

ВАЖНО: поле "say" должно идти ПЕРВЫМ в JSON — это нужно для того чтобы
ассистент начал озвучивать ответ пока ты ещё его дописываешь.

Форматы:
- Действие: {"say":"что сказать вслух","action":"shell","command":"shell-команда"}
- Разговор: {"say":"ответ","action":"talk"}

Примеры:
- "открой firefox" → {"say":"Открываю Firefox","action":"shell","command":"firefox &"}
- "как дела" → {"say":"Всё хорошо, чем помочь?","action":"talk"}
- "открой загрузки" → {"say":"Открыла загрузки","action":"shell","command":"xdg-open ~/Downloads"}"""


@dataclass
class ResponseDelta:
    sentence: str | None = None
    command: str | None = None
    done: bool = False
    error: str | None = None


class Brain:
    """DeepSeek-клиент со стриминговым извлечением предложений из поля `say`."""

    def __init__(self, config: "Config"):
        self._config = config
        self._client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url="https://api.deepseek.com",
        )
        self._history: list[dict] = []

    def ask_stream(self, user_text: str) -> Iterator[ResponseDelta]:
        self._history.append({"role": "user", "content": user_text})
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + self._history[-self._config.history_window:]
        )

        try:
            stream = self._client.chat.completions.create(
                model=self._config.deepseek_model,
                messages=messages,
                temperature=0.7,
                stream=True,
            )
        except Exception as exc:
            log.warning("DeepSeek API error: %s", exc)
            yield ResponseDelta(error=str(exc), done=True)
            return

        parser = StreamingSayParser()
        full_parts: list[str] = []

        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                full_parts.append(delta)
                for sentence in parser.feed(delta):
                    yield ResponseDelta(sentence=sentence)
        except Exception as exc:
            log.warning("DeepSeek stream error: %s", exc)
            yield ResponseDelta(error=str(exc), done=True)
            return

        full_text = "".join(full_parts).strip()
        cleaned = self._strip_markdown_fence(full_text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            log.warning("Невалидный JSON от LLM: %r", cleaned)
            yield ResponseDelta(error=f"invalid json: {exc}", done=True)
            return

        self._history.append({"role": "assistant", "content": cleaned})

        # Fallback: если стриминг не выдал предложений (LLM поставил command
        # раньше say или весь say поместился в одном фрагменте без
        # терминатора), выдаём весь say одним delta.
        if not parser.emitted_anything():
            say = (data.get("say") or "").strip()
            if say:
                yield ResponseDelta(sentence=say)

        command = None
        if data.get("action") == "shell":
            command = data.get("command")

        yield ResponseDelta(command=command, done=True)

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        if not text.startswith("```"):
            return text
        parts = text.split("```")
        if len(parts) < 2:
            return text
        body = parts[1]
        if body.startswith("json"):
            body = body[4:]
        return body.strip()
