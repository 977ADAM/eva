from eva.banner import _plural


def test_plural_one():
    assert _plural(1, "факт", "факта", "фактов") == "факт"


def test_plural_few():
    assert _plural(2, "факт", "факта", "фактов") == "факта"
    assert _plural(4, "факт", "факта", "фактов") == "факта"


def test_plural_many():
    assert _plural(5, "факт", "факта", "фактов") == "фактов"
    assert _plural(11, "факт", "факта", "фактов") == "фактов"
    assert _plural(14, "факт", "факта", "фактов") == "фактов"
    assert _plural(21, "факт", "факта", "фактов") == "факт"
