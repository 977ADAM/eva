import logging

log = logging.getLogger(__name__)


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
