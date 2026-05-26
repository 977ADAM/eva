import pytest

from eva.brain import StreamingSayParser


def feed_all(parser, chunks):
    out = []
    for c in chunks:
        out.extend(parser.feed(c))
    return out


def test_extracts_single_sentence_from_one_chunk():
    parser = StreamingSayParser()
    result = feed_all(parser, ['{"say":"Привет, мир.","action":"talk"}'])
    assert result == ["Привет, мир."]


def test_waits_for_sentence_terminator():
    parser = StreamingSayParser()
    assert feed_all(parser, ['{"say":"Прив']) == []
    assert feed_all(parser, ['ет, мир.']) == ["Привет, мир."]


def test_emits_multiple_sentences_in_order():
    parser = StreamingSayParser()
    result = feed_all(parser, [
        '{"say":"Один. Два! Три?","action":"talk"}',
    ])
    assert result == ["Один.", "Два!", "Три?"]


def test_emits_remainder_on_close_quote():
    parser = StreamingSayParser()
    # Текст без точки в конце; закрывающая кавычка `"` должна вытолкнуть остаток
    result = feed_all(parser, ['{"say":"Готово","action":"talk"}'])
    assert result == ["Готово"]


def test_no_say_field_yields_nothing():
    parser = StreamingSayParser()
    # command идёт первым — стриминг не сработает
    result = feed_all(parser, ['{"command":"firefox","action":"shell"}'])
    assert result == []
    assert not parser.emitted_anything()


def test_chunk_boundary_inside_say_prefix():
    parser = StreamingSayParser()
    # Префикс `"say":"` бьётся пополам
    assert feed_all(parser, ['{"sa']) == []
    assert feed_all(parser, ['y":"Готово.","action":"talk"}']) == ["Готово."]


def test_emitted_anything_flag():
    parser = StreamingSayParser()
    assert not parser.emitted_anything()
    feed_all(parser, ['{"say":"Привет.","action":"talk"}'])
    assert parser.emitted_anything()
