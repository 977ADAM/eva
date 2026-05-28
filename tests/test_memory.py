from eva.memory import Memory


def test_empty_when_no_file(tmp_path):
    mem = Memory(tmp_path / "memory.json")
    assert mem.all() == []


def test_add_appears_in_all(tmp_path):
    mem = Memory(tmp_path / "memory.json")
    mem.add("любит чай")
    assert "любит чай" in mem.all()


def test_add_persists_to_disk(tmp_path):
    path = tmp_path / "memory.json"
    Memory(path).add("пьёт кофе без сахара")
    assert "пьёт кофе без сахара" in Memory(path).all()


def test_remove_by_substring(tmp_path):
    mem = Memory(tmp_path / "memory.json")
    mem.add("пьёт кофе без сахара")
    assert mem.remove("кофе") == 1
    assert mem.all() == []


def test_remove_nonexistent_returns_zero(tmp_path):
    mem = Memory(tmp_path / "memory.json")
    mem.add("пьёт кофе без сахара")
    assert mem.remove("чай") == 0
    assert mem.all() == ["пьёт кофе без сахара"]


def test_remove_multiple_matches(tmp_path):
    mem = Memory(tmp_path / "memory.json")
    mem.add("любит кофе по утрам")
    mem.add("пьёт кофе без сахара")
    assert mem.remove("кофе") == 2
    assert mem.all() == []


def test_as_prompt_empty(tmp_path):
    mem = Memory(tmp_path / "memory.json")
    assert mem.as_prompt() == ""


def test_as_prompt_formats_facts(tmp_path):
    mem = Memory(tmp_path / "memory.json")
    mem.add("зовут Адам")
    mem.add("пьёт кофе без сахара")
    prompt = mem.as_prompt()
    assert "Что ты знаешь о пользователе" in prompt
    assert "- зовут Адам" in prompt
    assert "- пьёт кофе без сахара" in prompt


def test_corrupted_file_returns_empty(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("это не json {{{", encoding="utf-8")
    assert Memory(path).all() == []
