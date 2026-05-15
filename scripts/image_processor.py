import csv
import logging
import random
from time import perf_counter
from .decorators import timer_decorator
from .async_pipeline import AsyncImagePipeline, ImageProcessorPaths


class ImageProcessor:
    """Главный управляющий класc"""

    def __init__(self, paths: ImageProcessorPaths) -> None:
        self.pipeline = AsyncImagePipeline(paths)

    def get_random_painting_ids(self, count: int) -> list[str]:
        """Получение случайных ID картин из CSV файла"""
        painting_ids = []

        with open("data/MetObjects.csv", mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("Classification") == "Paintings" and row.get("Is Public Domain") == "True":
                    object_id = row.get("Object ID")
                    if object_id:
                        painting_ids.append(object_id)

        return random.sample(painting_ids, count)

    def build_numbered_ids(self, count: int) -> list[tuple[int, str]]:
        """Подготовка пронумерованного списка ID"""
        ids = self.get_random_painting_ids(count)
        numbered_ids = list(enumerate(ids, start=1))

        logging.info("Список изображений подготовлен")
        for number, object_id in numbered_ids:
            logging.info(f"Изображение {number}: ID {object_id}")

        return numbered_ids

    @timer_decorator
    async def process_images_async(self, numbered_ids: list[tuple[int, str]]) -> None:
        """Асинхронная обработка изображений"""
        logging.info("Асинхронная версия")
        await self.pipeline.run(numbered_ids)
        logging.info("Асинхронное выполнение завершено")

    @timer_decorator
    async def process_images_sequential(self, numbered_ids: list[tuple[int, str]]) -> None:
        """Последовательная обработка изображений"""
        logging.info("Последовательная версия")

        for item in numbered_ids:
            await self.pipeline.run([item])

        logging.info("Последовательное выполнение завершено")

    async def compare_versions(self, count: int) -> None:
        """Сравнение последовательной и асинхронной версий"""
        logging.info("Начало сравнения")
        numbered_ids = self.build_numbered_ids(count)

        logging.info("Запуск последовательной версии")
        sequential_start = perf_counter()
        await self.process_images_sequential(numbered_ids)
        sequential_time = perf_counter() - sequential_start

        logging.info(
            f"Последовательная версия завершена за "
            f"{sequential_time:.2f} сек"
        )

        logging.info("Запуск асинхронной версии")
        async_start = perf_counter()
        await self.process_images_async(numbered_ids)
        async_time = perf_counter() - async_start

        logging.info(
            f"Асинхронная версия завершена за "
            f"{async_time:.2f} сек"
        )

        logging.info("Результаты сравнения")
        logging.info(f"Последовательно: {sequential_time:.2f} сек")

        logging.info(f"Асинхронно: {async_time:.2f} сек")

        if async_time > 0:
            speedup = sequential_time / async_time

            logging.info(
                f"Ускорение асинхронной версии: "
                f"{speedup:.2f}x"
            )
