import argparse
import asyncio
import json
import logging
from pathlib import Path

from akimovcode.images.async_pipeline import ImageProcessorPaths
from akimovcode.images.image_processor import ImageProcessor
from akimovcode.analysis import main as run_analysis

logger = logging.getLogger(__name__)


async def prepare_logic(csv_path, output_json):
    """Логика формирования JSON списка для скачивания"""

    logger.info(f"Подготовка метаданных из {csv_path}")
    paths = ImageProcessorPaths(download_dir="images/originals/", output_dir="images/processed/")
    proc = ImageProcessor(paths)
    ids = proc.get_random_painting_ids(count=100)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(ids, f)
    logger.info(f"Список ID сохранен в {output_json}")


async def process_logic(input_json, output_dir, num):
    """Запуск сравнения последовательной и асинхронной обработки"""
    logger.info(f"Запуск сравнения на {num} изображениях")

    paths = ImageProcessorPaths(
        download_dir=str(Path(output_dir) / "originals"),
        output_dir=str(Path(output_dir) / "processed")
    )
    proc = ImageProcessor(paths)
    await proc.compare_versions(num)


def main():
    parser = argparse.ArgumentParser(prog="akimov6304", description="Получение картин из музея")
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # Команда prepare
    prep = subparsers.add_parser("prepare", help="Подготовка JSON с метаданными")
    prep.add_argument("--csv", default="data/MetObjects.csv", help="Путь к исходному CSV")
    prep.add_argument("--output", default="data/to_download.json", help="Путь к JSON")

    # Команда process
    proc = subparsers.add_parser("process", help="Скачивание и обработка")
    proc.add_argument("--input", default="data/to_download.json", help="Входной JSON")
    proc.add_argument("--output", default="images", help="Папка для сохранения")
    proc.add_argument("--num", type=int, required=True, help="Количество изображений")

    # Команда analyze
    subparsers.add_parser("analyze", help="Анализ датасета (Лаба №3)")

    args = parser.parse_args()

    if args.command == "prepare":
        asyncio.run(prepare_logic(args.csv, args.output))
    elif args.command == "process":
        asyncio.run(process_logic(args.input, args.output, args.num))
    elif args.command == "analyze":
        run_analysis()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
