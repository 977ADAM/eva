import argparse

from eva import __version__


def main() -> int:
    parser = argparse.ArgumentParser(prog="eva", description="Голосовой ассистент Eva")
    parser.add_argument("--version", action="version", version=f"eva {__version__}")
    parser.add_argument("--debug", action="store_true", help="Подробные логи")
    parser.add_argument("--model", default="small", help="Модель Whisper (small|medium)")
    parser.add_argument("--no-wake", action="store_true", help="Отвечать без wake-word")
    parser.parse_args()
    print("eva: stub — реализация в Task 11")
    return 0
