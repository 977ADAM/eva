import pytest

from eva.safety import SafetyGate


@pytest.fixture
def gate():
    return SafetyGate(safe_prefixes=("firefox", "xdg-open", "playerctl"))


def test_empty_command_is_safe(gate):
    assert gate.is_safe("") is True
    assert gate.is_safe("   ") is True


def test_allowlist_prefix_is_safe(gate):
    assert gate.is_safe("firefox") is True


def test_firefox_with_args_is_safe(gate):
    assert gate.is_safe("firefox https://ya.ru") is True


def test_firefox_with_trailing_amp_is_safe(gate):
    # Запуск в фоне — типичный паттерн, не должен требовать подтверждения
    assert gate.is_safe("firefox &") is True


def test_unknown_command_needs_confirm(gate):
    assert gate.is_safe("shutdown now") is False


def test_shell_chain_needs_confirm(gate):
    # Даже если первое слово allowlisted, цепочка через && делает команду опасной
    assert gate.is_safe("firefox && rm -rf /") is False


def test_pipe_needs_confirm(gate):
    assert gate.is_safe("firefox | grep x") is False
