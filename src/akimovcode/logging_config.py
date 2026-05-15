import logging
from pathlib import Path


def setup_logging() -> None:
    """Настройка логирования приложения"""

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    # Логирование в файл
    file_handler = logging.FileHandler(
        "logs/app.log",
        encoding="utf-8"
    )

    file_handler.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(filename)s:%(lineno)d | "
        "%(message)s"
    )

    file_handler.setFormatter(file_formatter)

    # Логирование в консоль
    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    console_formatter = logging.Formatter(
        "%(levelname)s - %(message)s"
    )

    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
