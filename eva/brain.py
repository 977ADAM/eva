import re
from typing import Iterator


_SAY_PREFIX_RE = re.compile(r'"say"\s*:\s*"')
_SENTENCE_END_RE = re.compile(r'[.!?\n]+\s*')


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
